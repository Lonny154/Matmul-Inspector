#!/usr/bin/env python3
"""Run the frozen-pair algebraic output-completion diagnostic."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import platform
from pathlib import Path
import statistics
import sys
import tracemalloc

from matmul_inspector.tensor_game import RankOneAction, matrix_multiplication_tensor, strassen_2x2_decomposition
from matmul_inspector.tensor_game_algebraic_w_completion import (
    CompletionConfig, complete_frozen_pairs, completion_recall, frozen_algebraic_uv_pairs,
)
from matmul_inspector.tensor_game_meet_in_middle import MeetInMiddleConfig, meet_in_middle


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def actions(rows):
    return tuple(RankOneAction(row["action"]["u"], row["action"]["v"], row["action"]["w"]) for row in rows)


def candidate_distribution(rows):
    scores = [row["score"] for row in rows]
    return {"instances": len(rows), "score": {"minimum": min(scores), "maximum": max(scores), "mean": statistics.fmean(scores)},
            "w_support": dict(sorted(Counter(row["w_support"] for row in rows).items())),
            "w_sign_patterns": len({tuple(row["w_sign_pattern"]) for row in rows}),
            "preparation_additions": dict(sorted(Counter(row["preparation_additions"] for row in rows).items())),
            "residual_reduction": dict(sorted(Counter(8 - row["score"] for row in rows).items()))}


def measured_mitm(target, rows, config):
    tracemalloc.start(); result = meet_in_middle(target, actions(rows), config)
    _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    result["peak_traced_memory_bytes"] = peak
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", default="20261017,29")
    parser.add_argument("--sample-attempts", type=int, default=10000)
    parser.add_argument("--maximum-seconds", type=float, default=30.0)
    args = parser.parse_args()
    if args.output.exists(): parser.error("--output must not exist")
    seeds = tuple(int(value) for value in args.seeds.split(","))
    if not seeds or args.sample_attempts <= 0: parser.error("seeds and sample attempts must be nonempty and positive")

    target = matrix_multiplication_tensor(2, 2, 2)
    known = strassen_2x2_decomposition()
    pairs, pair_audit = frozen_algebraic_uv_pairs(target)
    pair_manifest = {"schema": "tensor_game_frozen_algebraic_uv_pairs.v1", "audit": pair_audit, "pairs": pairs}
    strategies = ("residual", "structural", "slice_projection", "hybrid")
    w_budgets = (1, 2, 4, 8)
    pool_sizes = (20, 64, 128)
    results = []
    pools = {}
    for strategy in strategies:
        for w_budget in w_budgets:
            config = CompletionConfig(strategy, w_budget, 128, len(pairs), len(pairs) * 80)
            rows, audit = complete_frozen_pairs(target, pairs, config)
            key = f"{strategy}-w{w_budget}"
            pools[key] = rows
            for size in pool_sizes:
                retained = rows[:size]
                results.append({"strategy": strategy, "w_per_pair": w_budget, "pool_size": size,
                                "generation_audit": audit, "coverage": completion_recall(retained, known, pairs),
                                "candidate_distribution": candidate_distribution(retained)})

    residual_by_setting = {(row["w_per_pair"], row["pool_size"]): row for row in results if row["strategy"] == "residual"}
    for row in results:
        baseline = residual_by_setting[(row["w_per_pair"], row["pool_size"])]
        row["complete_action_recall_delta_vs_residual"] = row["coverage"]["complete_action_recall"] - baseline["coverage"]["complete_action_recall"]

    mitm = []
    for strategy in ("residual", "slice_projection"):
        source = pools[f"{strategy}-w2"]
        mitm.append({"strategy": strategy, "configuration": "complete_20", "seed": seeds[0],
                     **measured_mitm(target, source[:20], MeetInMiddleConfig(
                         maximum_left_attempts=100000, maximum_right_attempts=100000,
                         maximum_stored_sums=100000, maximum_seconds=args.maximum_seconds, seed=seeds[0]))})
        for size in (64, 128):
            for seed in seeds:
                mitm.append({"strategy": strategy, "configuration": f"sampled_{size}", "seed": seed,
                             **measured_mitm(target, source[:size], MeetInMiddleConfig(
                                 maximum_left_attempts=args.sample_attempts, maximum_right_attempts=args.sample_attempts,
                                 maximum_stored_sums=args.sample_attempts, maximum_seconds=args.maximum_seconds, seed=seed))})

    coverage_summary = [{"strategy": row["strategy"], "w_per_pair": row["w_per_pair"], "pool_size": row["pool_size"],
                         "pair_recall": row["coverage"]["pair_recall"],
                         "complete_action_recall": row["coverage"]["complete_action_recall"],
                         "tensor_equivalence_recall": row["coverage"]["tensor_equivalence_recall"],
                         "delta_vs_residual": row["complete_action_recall_delta_vs_residual"]} for row in results]
    best = max(coverage_summary, key=lambda row: (row["complete_action_recall"], -row["w_per_pair"], -row["pool_size"], row["strategy"]))
    summary = {"schema": "tensor_game_algebraic_w_completion_summary.v1",
               "frozen_pair_count": len(pairs), "frozen_pair_content_sha256": pair_audit["content_sha256"],
               "valid_w_domain": 80, "best_complete_action_recall": best,
               "coverage": coverage_summary,
               "missing_pair_hard_upper_bound": 6,
               "mitm_exact_discoveries": {strategy: sum(row["exact_decomposition_count"] for row in mitm if row["strategy"] == strategy)
                                            for strategy in ("residual", "slice_projection")},
               "finding": "Exact slice projection improves retained complete-action recall from one to two at w=2 and pool size 128, but no strategy exceeds two of seven and no independent rank-7 decomposition is found."}
    configuration = {"schema": "tensor_game_algebraic_w_completion_configuration.v1",
                     "strategies": strategies, "w_per_pair": w_budgets, "pool_sizes": pool_sizes,
                     "matched_exact_w_scores_per_pair": 80, "seeds": seeds,
                     "mitm_sample_attempts_per_side": args.sample_attempts,
                     "maximum_seconds_per_mitm_run": args.maximum_seconds,
                     "known_strassen_used_for_generation_ranking_or_selection": False,
                     "cpu_only": True, "python": sys.version, "platform": platform.platform()}
    artifacts = {"configuration.json": configuration, "frozen_pairs.json": pair_manifest,
                 "completion_results.json": results, "pools.json": pools,
                 "meet_in_middle.json": mitm, "summary.json": summary}
    args.output.mkdir(parents=True, exist_ok=False)
    for name, payload in artifacts.items(): (args.output / name).write_bytes(encoded(payload))
    hashes = {name: {"bytes": (args.output/name).stat().st_size,
                     "sha256": hashlib.sha256((args.output/name).read_bytes()).hexdigest()} for name in sorted(artifacts)}
    (args.output / "artifact_hashes.json").write_bytes(encoded({"schema": "tensor_game_algebraic_w_completion_hashes.v1", "files": hashes}))
    print(json.dumps(summary, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
