#!/usr/bin/env python3
"""Paired CUDA-event timing of baseline and fused output reconstruction."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path


def distribution(values):
    return {
        "median_ms": statistics.median(values), "min_ms": min(values),
        "max_ms": max(values), "mean_ms": statistics.mean(values),
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
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must not already exist")

    import torch
    from matmul_inspector.tensor_game_triton_lowering import reference_lowering
    from matmul_inspector.tensor_game_triton_runtime import (
        baseline_reconstruction, fused_input_gemm, fused_reconstruction,
    )
    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    torch.manual_seed(args.seed)
    randomizer = random.Random(args.seed)
    raw = []
    for size in args.sizes:
        left = torch.randn((size, size), device="cuda")
        right = torch.randn((size, size), device="cuda")
        for algorithm in ("conventional", "strassen"):
            lowering = reference_lowering(algorithm)
            products = [fused_input_gemm(left, right, a.u, a.v, "ieee") for a in lowering.actions]
            implementations = {
                "baseline": lambda: baseline_reconstruction(products, lowering.actions, size),
                "fused": lambda: fused_reconstruction(products, lowering.actions, size),
            }
            for function in implementations.values():
                for _ in range(args.warmups):
                    function()
            torch.cuda.synchronize()
            for run in range(args.runs):
                order = list(implementations)
                randomizer.shuffle(order)
                for _ in range(args.iterations):
                    for reconstruction in order:
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record(); implementations[reconstruction](); end.record(); end.synchronize()
                        raw.append({"size": size, "algorithm": algorithm, "reconstruction": reconstruction, "run": run, "elapsed_ms": start.elapsed_time(end)})
    summaries = []
    for size in args.sizes:
        for algorithm in ("conventional", "strassen"):
            for reconstruction in ("baseline", "fused"):
                run_medians = [statistics.median(x["elapsed_ms"] for x in raw if x["size"] == size and x["algorithm"] == algorithm and x["reconstruction"] == reconstruction and x["run"] == run) for run in range(args.runs)]
                summaries.append({"size": size, "algorithm": algorithm, "reconstruction": reconstruction, "run_medians_ms": run_medians, **distribution(run_medians)})
    payload = {"schema": "tensor_game_reconstruction_profile.v1", "configuration": vars(args) | {"output": str(args.output), "timing": "CUDA events; reconstruction only; precomputed rank products; do not add to complete-operation medians"}, "summaries": summaries, "raw_timings": raw}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
