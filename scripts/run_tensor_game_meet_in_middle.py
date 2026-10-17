#!/usr/bin/env python3
"""Run the bounded CPU-only Tensor Game meet-in-the-middle study."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
from pathlib import Path
import sys
import tracemalloc

from matmul_inspector.tensor_game import RankOneAction, matrix_multiplication_tensor, strassen_2x2_decomposition, subtract_action
from matmul_inspector.tensor_game_bounded_residual import VECTORS, squared_norm
from matmul_inspector.tensor_game_experience import action_identity, action_payload, tensor_digest
from matmul_inspector.tensor_game_meet_in_middle import (
    MeetInMiddleConfig, action_domain_complexity, canonical_action_pool, meet_in_middle,
)
from matmul_inspector.tensor_game_structural_pairs import structural_pair_candidates, structural_signature
from matmul_inspector.tensor_game_search import canonicalize_action


def encoded(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def residual_pool(target, size: int):
    unique = {
        canonicalize_action(RankOneAction(u, v, w))
        for u, v, w in itertools.product(VECTORS, repeat=3)
    }
    ranked = sorted(unique, key=lambda action: (
        squared_norm(subtract_action(target, action)), action.u, action.v, action.w,
    ))
    return tuple(ranked[:size]), {
        "method": "exact full-domain immediate residual ranking",
        "raw_actions": len(VECTORS) ** 3, "canonical_actions": len(unique),
    }


def structural_pool(target, size: int):
    rows, audit = structural_pair_candidates(
        target, protected_residual_pairs=64, structural_pairs=64,
        w_mode="top_k", w_candidates_per_pair=4, limit=512,
    )
    selected = []
    signatures = set()
    for row in rows:
        signature = structural_signature(row["features"])
        if signature not in signatures:
            selected.append(row["action"]); signatures.add(signature)
        if len(selected) == size:
            break
    for row in rows:
        if len(selected) == size:
            break
        if row["action"] not in selected:
            selected.append(row["action"])
    return canonical_action_pool(selected), {**audit, "selection": "one action per prior structural signature, then score-ranked fill"}


def mixed_pool(residual, structural, size: int):
    ordered = []
    for index in range(max(len(residual), len(structural))):
        for pool in (residual, structural):
            if index < len(pool) and pool[index] not in ordered:
                ordered.append(pool[index])
        if len(ordered) >= size:
            break
    return tuple(ordered[:size])


def measured_run(target, pool, config):
    tracemalloc.start()
    result = meet_in_middle(target, pool, config)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    result["peak_traced_memory_bytes"] = peak
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--pool-size", type=int, default=64)
    parser.add_argument("--sample-attempts", type=int, default=25_000)
    parser.add_argument("--maximum-seconds", type=float, default=30.0)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must not exist")
    if args.pool_size < 8 or args.sample_attempts <= 0:
        parser.error("pool size must be at least 8 and sample attempts must be positive")

    target = matrix_multiplication_tensor(2, 2, 2)
    residual, residual_audit = residual_pool(target, args.pool_size)
    structural, structural_audit = structural_pool(target, args.pool_size)
    mixed = mixed_pool(residual, structural, args.pool_size)
    if min(map(len, (residual, structural, mixed))) < args.pool_size:
        raise RuntimeError("an ordinary pool did not reach the requested matched size")
    ordinary = {"residual_ranked": residual, "structural_diverse": structural, "deterministic_mixed": mixed}
    oracle = canonical_action_pool(strassen_2x2_decomposition())
    oracle_ids = {action_identity(action) for action in oracle}

    experiments = []
    complete_size = min(20, args.pool_size)
    for name, pool in ordinary.items():
        experiments.append({"pool": name, "configuration": "complete_small",
                            **measured_run(target, pool[:complete_size], MeetInMiddleConfig(
                                maximum_left_attempts=100_000, maximum_right_attempts=100_000,
                                maximum_stored_sums=100_000, maximum_seconds=args.maximum_seconds,
                                seed=args.seed))})
        experiments.append({"pool": name, "configuration": "sampled_large",
                            **measured_run(target, pool, MeetInMiddleConfig(
                                maximum_left_attempts=args.sample_attempts,
                                maximum_right_attempts=args.sample_attempts,
                                maximum_stored_sums=args.sample_attempts,
                                maximum_seconds=args.maximum_seconds, seed=args.seed))})
    experiments.append({"pool": "strassen_oracle", "configuration": "oracle_validation",
                        **measured_run(target, oracle, MeetInMiddleConfig(
                            maximum_left_attempts=100, maximum_right_attempts=100,
                            maximum_stored_sums=100, maximum_seconds=args.maximum_seconds,
                            seed=args.seed))})

    pool_rows = {}
    for name, pool in {**ordinary, "strassen_oracle": oracle}.items():
        ids = [action_identity(action) for action in pool]
        pool_rows[name] = {
            "size": len(pool), "actions": [action_payload(action) for action in pool],
            "action_ids": ids, "oracle_overlap_count": len(set(ids) & oracle_ids),
            "oracle_used_for_construction": name == "strassen_oracle",
        }
    exact = [row for row in experiments if row["exact_decomposition_count"]]
    summary = {
        "schema": "tensor_game_meet_in_middle_summary.v1",
        "target_digest": tensor_digest(target),
        "ordinary_exact_discoveries": sum(row["exact_decomposition_count"] for row in experiments if row["pool"] != "strassen_oracle"),
        "oracle_exact_recoveries": sum(row["exact_decomposition_count"] for row in experiments if row["pool"] == "strassen_oracle"),
        "partial_sum_collision_runs": sum(bool(row["left"]["duplicate_sums"]) for row in experiments),
        "runs_with_exact_decomposition": [{"pool": row["pool"], "configuration": row["configuration"]} for row in exact],
        "interpretation": "Oracle recovery validates mechanics. Duplicate partial sums are collisions, not solutions. Only ordinary-pool reconstruction counts as independent discovery.",
    }
    configuration = {
        "schema": "tensor_game_meet_in_middle_configuration.v1", "seed": args.seed,
        "pool_size": args.pool_size, "sample_attempts_per_side": args.sample_attempts,
        "maximum_seconds_per_run": args.maximum_seconds, "python": sys.version,
        "platform": platform.platform(), "cpu_only": True,
        "ordinary_pool_construction_consulted_oracle": False,
        "residual_pool_audit": residual_audit, "structural_pool_audit": structural_audit,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    files = {
        "configuration.json": configuration, "domain_complexity.json": action_domain_complexity(),
        "pools.json": pool_rows, "experiments.json": experiments, "summary.json": summary,
    }
    for name, value in files.items():
        (args.output / name).write_bytes(encoded(value))
    hashes = {name: hashlib.sha256((args.output / name).read_bytes()).hexdigest() for name in sorted(files)}
    (args.output / "artifact_hashes.json").write_bytes(encoded({"schema": "tensor_game_meet_in_middle_hashes.v1", "sha256": hashes}))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
