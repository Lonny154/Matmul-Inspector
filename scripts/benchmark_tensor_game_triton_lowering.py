#!/usr/bin/env python3
"""Correctness and paired GPU benchmark for Tensor Game Triton lowerings."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import shutil
import statistics
from pathlib import Path


def distribution(values):
    ordered = sorted(values)
    return {
        "median_ms": statistics.median(ordered),
        "mean_ms": statistics.mean(ordered),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "stdev_ms": statistics.stdev(ordered) if len(ordered) > 1 else 0.0,
    }


def error_metrics(torch, actual, expected):
    difference = (actual - expected).abs()
    denominator = expected.abs().clamp_min(1e-12)
    return {
        "max_absolute_error": difference.max().item(),
        "mean_absolute_error": difference.mean().item(),
        "max_relative_error": (difference / denominator).max().item(),
    }


def execution_estimate(algorithm, lowering, size, strategy):
    """Return static launch/storage/read estimates, not measured traffic."""
    half = size // 2
    block_elements = half * half
    block_bytes = block_elements * 4
    source_terms = sum(
        sum(value != 0 for value in action.u + action.v)
        for action in lowering.actions
    )
    preparation_launches = (
        2 * lowering.metadata.rank + lowering.metadata.execution_preparation_updates
        if strategy == "unfused" else 0
    )
    reconstruction_launches = lowering.metadata.execution_reconstruction_updates + 4
    if strategy == "unfused":
        estimated_operand_reads = (source_terms + 2 * lowering.metadata.rank) * block_bytes
    else:
        tiles = (half + 31) // 32
        estimated_operand_reads = source_terms * tiles * block_bytes
    return {
        "size": size,
        "algorithm": algorithm,
        "strategy": strategy,
        "estimate_kind": "static source-level model; not hardware-measured traffic",
        "preparation_kernel_launches": preparation_launches,
        "gemm_kernel_launches": lowering.metadata.rank,
        "reconstruction_kernel_launches": reconstruction_launches,
        "output_assembly_copy_launches": 4,
        "total_device_kernel_launches": preparation_launches + lowering.metadata.rank + reconstruction_launches + 4,
        "full_size_input_combination_buffers": 2 * lowering.metadata.rank if strategy == "unfused" else 0,
        "input_combination_bytes_materialized": 2 * lowering.metadata.rank * block_bytes if strategy == "unfused" else 0,
        "rank_product_buffers": lowering.metadata.rank,
        "rank_product_storage_bytes": lowering.metadata.rank * block_bytes,
        "estimated_operand_read_bytes": estimated_operand_reads,
        "estimated_operand_read_model": "unfused source-combination reads plus GEMM reads; fused source terms repeated across the orthogonal 32-element output-tile dimension",
        "temporary_lifetime": "input combinations are call-scoped temporaries; rank products remain live through reconstruction",
    }


def copy_compiler_artifacts(cache: Path, destination: Path):
    extensions = {".ptx", ".ttir", ".ttgir", ".sass", ".cubin", ".json"}
    rows = []
    for source in sorted(path for path in cache.rglob("*") if path.is_file() and path.suffix in extensions):
        relative = source.relative_to(cache)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        precision = None
        siblings = list(source.parent.glob("*.ttir")) + list(source.parent.glob("*.ttgir"))
        for sibling in siblings:
            text = sibling.read_text(encoding="utf-8")
            if "tf32x3" in text:
                precision = "tf32x3"
                break
        if precision is None and siblings:
            # Triton 3.6 omits the explicit IEEE attribute after lowering while
            # retaining tf32x3. This experiment supports only those two modes.
            precision = "ieee"
        kernel_name = source.stem
        strategy = "fused_input" if "fused_input" in kernel_name else "unfused"
        rows.append({
            "path": str(relative),
            "bytes": source.stat().st_size,
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "kind": source.suffix.lstrip("."),
            "dot_input_precision": precision,
            "execution_strategy": strategy,
            "kernel_name": kernel_name,
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=(128, 256, 512, 1024))
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--precisions", nargs="+", choices=("ieee", "tf32x3"), default=("ieee",))
    parser.add_argument("--strategies", nargs="+", choices=("unfused", "fused_input"), default=("unfused",))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must not already exist")
    if any(size <= 0 or size % 2 for size in args.sizes):
        parser.error("all --sizes must be positive and even")
    if min(args.warmups, args.iterations, args.runs) <= 0:
        parser.error("warmups, iterations, and runs must be positive")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cache = args.output.parent / "triton-cache"
    os.environ["TRITON_CACHE_DIR"] = str(cache.resolve())
    try:
        import torch
        import triton
    except ImportError as exc:
        parser.error(f"PyTorch and Triton are required: {exc}")
    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    from matmul_inspector.tensor_game_triton_lowering import lowering_manifest, reference_lowering

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    randomizer = random.Random(args.seed)
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    lowerings = {name: reference_lowering(name) for name in ("conventional", "strassen")}
    correctness = []
    raw_timings = []
    try:
        for size in args.sizes:
            cases = {
                "zero": (torch.zeros((size, size), device="cuda"), torch.zeros((size, size), device="cuda")),
                "identity": (torch.eye(size, device="cuda"), torch.eye(size, device="cuda")),
                "seeded_random": (torch.randn((size, size), device="cuda"), torch.randn((size, size), device="cuda")),
                "negative": (-torch.rand((size, size), device="cuda"), torch.rand((size, size), device="cuda") - 0.5),
            }
            for case, (left, right) in cases.items():
                expected = left @ right
                for name, lowering in lowerings.items():
                    for precision in args.precisions:
                        outputs = {
                            strategy: lowering.execute(left, right, precision, strategy)
                            for strategy in args.strategies
                        }
                        unfused = outputs.get("unfused")
                        for strategy, actual in outputs.items():
                            metrics = error_metrics(torch, actual, expected)
                            comparison = error_metrics(torch, actual, unfused) if unfused is not None else None
                            passed = torch.allclose(actual, expected, rtol=5e-3, atol=5e-3)
                            correctness.append({"size": size, "case": case, "algorithm": name, "precision": precision, "strategy": strategy, **metrics, "versus_unfused": comparison, "relative_error_denominator": "max(abs(reference), 1e-12) elementwise", "rtol": 5e-3, "atol": 5e-3, "passed": bool(passed)})
                            if not passed:
                                raise RuntimeError(f"correctness failed for {name}, {precision}, {strategy}, {case}, size {size}: {metrics}")

            left, right = cases["seeded_random"]
            implementations = {"pytorch_cublas": lambda: left @ right}
            for name, lowering in lowerings.items():
                for precision in args.precisions:
                    for strategy in args.strategies:
                        implementations[f"{name}_{strategy}_{precision}"] = lambda lowering=lowering, precision=precision, strategy=strategy: lowering.execute(left, right, precision, strategy)
            for implementation in implementations.values():
                for _ in range(args.warmups):
                    implementation()
            torch.cuda.synchronize()
            for run in range(args.runs):
                order = list(implementations)
                randomizer.shuffle(order)
                for _ in range(args.iterations):
                    for name in order:
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record(); implementations[name](); end.record(); end.synchronize()
                        raw_timings.append({"size": size, "run": run, "implementation": name, "elapsed_ms": start.elapsed_time(end)})
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32

    summaries = []
    for size in args.sizes:
        names = ("pytorch_cublas", *(f"{algorithm}_{strategy}_{precision}" for algorithm in lowerings for strategy in args.strategies for precision in args.precisions))
        for name in names:
            values = [row["elapsed_ms"] for row in raw_timings if row["size"] == size and row["implementation"] == name]
            run_medians = [statistics.median(row["elapsed_ms"] for row in raw_timings if row["size"] == size and row["implementation"] == name and row["run"] == run) for run in range(args.runs)]
            row = {"size": size, "implementation": name, "sample_count": len(values), "run_medians_ms": run_medians, **distribution(run_medians)}
            reference = [statistics.median(x["elapsed_ms"] for x in raw_timings if x["size"] == size and x["implementation"] == "pytorch_cublas" and x["run"] == run) for run in range(args.runs)]
            row["pytorch_median_speedup"] = statistics.median(reference) / row["median_ms"]
            summaries.append(row)
    compiler_dir = args.output.parent / "compiler-artifacts"
    compiler = copy_compiler_artifacts(cache, compiler_dir)
    payload = {
        "schema": "tensor_game_triton_benchmark.v1",
        "configuration": {"output": str(args.output), "sizes": args.sizes, "warmups": args.warmups, "iterations": args.iterations, "runs": args.runs, "seed": args.seed, "precisions": args.precisions, "strategies": args.strategies, "timing": "CUDA events; complete device operation; host allocations excluded", "measurement_order": "randomized per independent run", "pytorch_allow_tf32": False},
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__, "triton": triton.__version__, "cuda_runtime": torch.version.cuda, "device": torch.cuda.get_device_name(0), "capability": torch.cuda.get_device_capability(0)},
        "lowerings": {f"{name}_{strategy}_{precision}": lowering_manifest(lowering, precision, strategy) for name, lowering in lowerings.items() for strategy in args.strategies for precision in args.precisions},
        "static_execution_estimates": [
            execution_estimate(name, lowering, size, strategy)
            for size in args.sizes for name, lowering in lowerings.items() for strategy in args.strategies
        ],
        "correctness": correctness,
        "all_correctness_passed": all(row["passed"] for row in correctness),
        "summaries": summaries,
        "raw_timings": raw_timings,
        "compiler_artifacts": compiler,
        "compiler_notes": "Cache artifacts are compiler intermediates. PTX is not final SASS. Resource usage requires separate compiler or disassembler evidence.",
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "all_correctness_passed": payload["all_correctness_passed"], "summaries": summaries}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
