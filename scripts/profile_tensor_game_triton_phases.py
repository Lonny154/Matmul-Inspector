#!/usr/bin/env python3
"""Measure preparation, GEMM, and reconstruction phases separately."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path


def distribution(values):
    return {
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
        "mean_ms": statistics.mean(values),
        "stdev_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=(128, 256, 512, 1024))
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--precisions", nargs="+", choices=("ieee", "tf32x3"), default=("ieee",))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must not already exist")

    import torch

    from matmul_inspector.tensor_game_triton_lowering import reference_lowering
    from matmul_inspector.tensor_game_triton_runtime import _combination, fp32_gemm

    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    randomizer = random.Random(args.seed)
    torch.manual_seed(args.seed)
    raw = []

    for size in args.sizes:
        half = size // 2
        left = torch.randn((size, size), device="cuda", dtype=torch.float32)
        right = torch.randn((size, size), device="cuda", dtype=torch.float32)
        a_blocks = tuple(left[i * half:(i + 1) * half, j * half:(j + 1) * half] for i in range(2) for j in range(2))
        b_blocks = tuple(right[i * half:(i + 1) * half, j * half:(j + 1) * half] for i in range(2) for j in range(2))
        for algorithm in ("conventional", "strassen"):
            for precision in args.precisions:
                lowering = reference_lowering(algorithm)
                combinations = [(_combination(a_blocks, action.u), _combination(b_blocks, action.v)) for action in lowering.actions]
                products = [fp32_gemm(a, b, precision) for a, b in combinations]

                def preparation():
                    return [(_combination(a_blocks, action.u), _combination(b_blocks, action.v)) for action in lowering.actions]

                def gemms():
                    return [fp32_gemm(a, b, precision) for a, b in combinations]

                def reconstruction():
                    output_blocks = []
                    for output_index in range(4):
                        block = torch.zeros_like(products[0])
                        for action, product in zip(lowering.actions, products):
                            coefficient = action.w[output_index]
                            if coefficient:
                                block.add_(product, alpha=float(coefficient))
                        output_blocks.append(block)
                    output = torch.empty_like(left)
                    for index, block in enumerate(output_blocks):
                        i, j = divmod(index, 2)
                        output[i * half:(i + 1) * half, j * half:(j + 1) * half].copy_(block)
                    return output

                phases = {"preparation": preparation, "gemm_only": gemms, "reconstruction": reconstruction}
                for function in phases.values():
                    for _ in range(args.warmups):
                        function()
                torch.cuda.synchronize()
                for run in range(args.runs):
                    order = list(phases)
                    randomizer.shuffle(order)
                    for _ in range(args.iterations):
                        for phase in order:
                            start = torch.cuda.Event(enable_timing=True)
                            end = torch.cuda.Event(enable_timing=True)
                            start.record(); phases[phase](); end.record(); end.synchronize()
                            raw.append({"size": size, "algorithm": algorithm, "precision": precision, "phase": phase, "run": run, "elapsed_ms": start.elapsed_time(end)})

    summaries = []
    for size in args.sizes:
        for algorithm in ("conventional", "strassen"):
            for precision in args.precisions:
                for phase in ("preparation", "gemm_only", "reconstruction"):
                    run_medians = [statistics.median(row["elapsed_ms"] for row in raw if row["size"] == size and row["algorithm"] == algorithm and row["precision"] == precision and row["phase"] == phase and row["run"] == run) for run in range(args.runs)]
                    summaries.append({"size": size, "algorithm": algorithm, "precision": precision, "phase": phase, "run_medians_ms": run_medians, **distribution(run_medians)})
    payload = {
        "schema": "tensor_game_triton_phase_profile.v1",
        "configuration": {"sizes": args.sizes, "warmups": args.warmups, "iterations": args.iterations, "runs": args.runs, "seed": args.seed, "precisions": args.precisions, "timing": "CUDA events; phases measured separately and must not be summed as a substitute for complete-operation latency"},
        "summaries": summaries,
        "raw_timings": raw,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
