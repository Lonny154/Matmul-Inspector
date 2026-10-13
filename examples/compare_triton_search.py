#!/usr/bin/env python3
"""Plan or explicitly execute paired uniform-versus-adaptive Triton search."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from matmul_inspector.hardware_search import (
    HardwareSearchConfig, HardwareSearchExperiment, default_fc1_search_space,
    render_hardware_search_report, write_hardware_search,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true",
                      help="validate and print the plan without GPU execution (default)")
    mode.add_argument("--execute", action="store_true",
                      help="explicitly enable real CUDA/Triton evaluation")
    parser.add_argument("--output", type=Path,
                        help="new output directory; required with --execute")
    parser.add_argument("--trial-seed", action="append", type=int, dest="trial_seeds",
                        help="repeat for multiple trials (defaults: 101, 202, 303)")
    parser.add_argument("--budget", type=int, default=16)
    parser.add_argument("--initial-size", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epsilon", type=float, default=0.25)
    parser.add_argument("--prior-strength", type=float, default=2.0)
    parser.add_argument("--latency-threshold-us", type=float, default=62.0)
    parser.add_argument("--warmups", type=int, default=100)
    parser.add_argument("--iterations-per-sample", type=int, default=500)
    parser.add_argument("--samples", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = HardwareSearchConfig(
        trial_seeds=tuple(args.trial_seeds or (101, 202, 303)),
        per_arm_budget=args.budget, shared_initial_size=args.initial_size,
        adaptive_batch_size=args.batch_size, epsilon=args.epsilon,
        prior_strength=args.prior_strength,
        latency_threshold_us=args.latency_threshold_us,
    )
    experiment = HardwareSearchExperiment(default_fc1_search_space(), config)
    if not args.execute:
        manifest = experiment.dry_run()
        print(render_hardware_search_report(manifest))
        print("\nSeed plan and shared initial candidates:")
        for plan in manifest["trial_plans"]:
            print(json.dumps({
                "trial_id": plan["trial_id"], "seeds": plan["seeds"],
                "shared_initial_candidates": plan["shared_initial_candidates"],
                "adaptive_rebuild_after_logical_counts": (
                    plan["adaptive_rebuild_after_logical_counts"]),
                "future_adaptive_decisions": plan["future_adaptive_decisions"],
            }, sort_keys=True))
        print("\nOptional dependency check:")
        print(json.dumps(manifest["dependency_check"], indent=2, sort_keys=True))
        print("Expected artifacts after --execute: experiment.json, evaluations.jsonl, "
              "progress.json, report.md")
        if args.output is not None:
            write_hardware_search(args.output, {"manifest": manifest, "evaluations": []})
            print(f"Dry-run plan written to {args.output}")
        print("NO BENCHMARK WAS RUN. CUDA was not queried.")
        return 0

    if args.output is None:
        raise SystemExit("--execute requires --output with a new directory")
    from matmul_inspector.triton_evaluator import TritonEvaluatorConfig, TritonGemmEvaluator

    evaluator_config = TritonEvaluatorConfig(
        warmups=args.warmups, iterations_per_sample=args.iterations_per_sample,
        samples=args.samples,
    )

    def evaluator_factory(trial_seed: int, trial_id: str) -> TritonGemmEvaluator:
        del trial_id
        return TritonGemmEvaluator(TritonEvaluatorConfig(
            seed=trial_seed, atol=evaluator_config.atol, rtol=evaluator_config.rtol,
            warmups=evaluator_config.warmups,
            iterations_per_sample=evaluator_config.iterations_per_sample,
            samples=evaluator_config.samples,
        ))

    result = experiment.execute(evaluator_factory)
    destination = write_hardware_search(args.output, result)
    print(render_hardware_search_report(result["manifest"]))
    print(f"Hardware search artifacts written to {destination}")
    print("Confirmation was not run. Invoke it separately after reviewing search evidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
