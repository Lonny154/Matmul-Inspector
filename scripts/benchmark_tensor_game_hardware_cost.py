#!/usr/bin/env python3
"""Compare bounded_128 candidate generation with optional static cost analysis."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

from matmul_inspector.tensor_game import matrix_multiplication_tensor, strassen_2x2_decomposition, subtract_action
from matmul_inspector.tensor_game_bounded_residual import bounded_residual_candidates_memory_efficient, squared_norm


def summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "min_ms": ordered[0],
        "median_ms": statistics.median(ordered),
        "mean_ms": statistics.mean(ordered),
        "max_ms": ordered[-1],
        "stdev_ms": statistics.stdev(ordered) if len(ordered) > 1 else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repetitions < 2:
        parser.error("--repetitions must be at least 2")
    if args.output.exists():
        parser.error("--output must not already exist")

    residual = matrix_multiplication_tensor(2, 2, 2)
    targets = [("initial", residual)]
    for step, action in enumerate(strassen_2x2_decomposition()[:-1], 1):
        residual = subtract_action(residual, action)
        targets.append((f"strassen_after_{step}", residual))
    modes = (("baseline", False), ("cost_analysis", True))
    timings = {name: [] for name, _ in modes}
    observations = {}
    raw = []
    # Alternate order by repetition. Both modes use the identical target,
    # pair budget, return limit, and deterministic candidate implementation.
    for repetition in range(args.repetitions):
        ordered_modes = modes if repetition % 2 == 0 else tuple(reversed(modes))
        for name, enabled in ordered_modes:
            started = time.perf_counter_ns(); mode_observations = []
            for target_id, target in targets:
                rows, audit = bounded_residual_candidates_memory_efficient(
                    target, 128, 12, analyze_costs=enabled
                )
                observation = {
                    "target_id": target_id,
                    "residual_squared_norm": squared_norm(target),
                    "enumerated_uv_pairs": audit["enumerated_uv_pairs"],
                    "fully_evaluated_pairs": audit["fully_evaluated_pairs"],
                    "unique_legal_candidates": audit["unique_legal_candidates"],
                    "returned_candidates": len(rows),
                    "returned_action_ids": [row["action_id"] for row in rows],
                    "returned_scores": [row["score"] for row in rows],
                }
                if enabled:
                    analysis = audit["cost_analysis"]
                    observation.update({
                        "analyzed_candidates": analysis["candidate_count"],
                        "pareto_frontier_size": analysis["pareto_frontier_size"],
                        "pareto_candidates": [
                            row for row in analysis["candidates"] if row["pareto_optimal"]
                        ],
                    })
                mode_observations.append(observation)
            elapsed_ms = (time.perf_counter_ns() - started) / 1e6
            timings[name].append(elapsed_ms)
            raw.append({"repetition": repetition, "mode": name, "elapsed_ms": elapsed_ms})
            observations[name] = mode_observations

    baseline = summary(timings["baseline"])
    analyzed = summary(timings["cost_analysis"])
    for baseline_row, analyzed_row in zip(observations["baseline"], observations["cost_analysis"], strict=True):
        if baseline_row["target_id"] != analyzed_row["target_id"]:
            raise RuntimeError("mode target order changed")
        for key in ("fully_evaluated_pairs", "unique_legal_candidates", "returned_candidates", "returned_action_ids", "returned_scores"):
            if baseline_row[key] != analyzed_row[key]:
                raise RuntimeError(f"cost analysis changed {key} for {baseline_row['target_id']}")
    payload = {
        "schema": "tensor_game_hardware_cost_benchmark.v1",
        "configuration": {
            "target_set": [target_id for target_id, _ in targets],
            "target_definition": "initial 2x2 multiplication tensor and six exact Strassen-prefix residuals",
            "seed": None,
            "seed_note": "candidate generation is deterministic and consumes no RNG",
            "pair_budget": 128,
            "return_limit": 12,
            "repetitions": args.repetitions,
            "order": "alternated by repetition",
            "timer": "time.perf_counter_ns",
        },
        "host": {"python": platform.python_version(), "platform": platform.platform(), "processor": platform.processor()},
        "summaries": {"baseline": baseline, "cost_analysis": analyzed},
        "overhead": {
            "mean_ms": analyzed["mean_ms"] - baseline["mean_ms"],
            "mean_ratio": analyzed["mean_ms"] / baseline["mean_ms"],
        },
        "search_outcomes": "not run; instrumentation does not change candidate selection",
        "observations": observations,
        "raw_timings": raw,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "summaries": payload["summaries"], "overhead": payload["overhead"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
