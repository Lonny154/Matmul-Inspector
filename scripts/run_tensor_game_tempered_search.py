#!/usr/bin/env python3
"""Run the bounded CPU-only TG-Global-1 experiment."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

from matmul_inspector.tensor_game import RankOneAction, matrix_multiplication_tensor, strassen_2x2_decomposition
from matmul_inspector.tensor_game_algebraic_w_completion import CompletionConfig, complete_frozen_pairs, frozen_algebraic_uv_pairs
from matmul_inspector.tensor_game_experience import action_payload, tensor_digest
from matmul_inspector.tensor_game_meet_in_middle import canonical_action_pool
from matmul_inspector.tensor_game_tempered_search import (
    TemperedSearchConfig, parallel_tempering_search, pool_identity, validate_pool_mode,
)


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def git_value(*args):
    try:
        return subprocess.run(("git", *args), check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_pool(name, target):
    if name == "strassen_oracle":
        return canonical_action_pool(strassen_2x2_decomposition()), {
            "method": "known Strassen actions", "oracle_only": True,
            "known_strassen_used_for_construction": True,
        }
    if name == "algebraic_slice_projection_128":
        pairs, pair_audit = frozen_algebraic_uv_pairs(target)
        rows, audit = complete_frozen_pairs(target, pairs, CompletionConfig(
            strategy="slice_projection", w_per_pair=2, final_pool_size=128,
            maximum_pairs=len(pairs), maximum_complete_scores=len(pairs) * 80,
        ))
        actions = tuple(RankOneAction(row["action"]["u"], row["action"]["v"], row["action"]["w"]) for row in rows)
        return actions, {"method": "frozen algebraic pairs with slice-projection top-2",
                         "oracle_only": False, "known_strassen_used_for_construction": False,
                         "frozen_pair_hash": pair_audit["content_sha256"], "generation_audit": audit}
    raise ValueError(f"unknown pool: {name}")


def main():
    parser = argparse.ArgumentParser(description="TG-Global-1 bounded parallel tempering")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pool", choices=("strassen_oracle", "algebraic_slice_projection_128"),
                        default="algebraic_slice_projection_128")
    parser.add_argument("--oracle-positive-control", action="store_true")
    parser.add_argument("--rank", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--temperatures", default="0.5,1,2,4")
    parser.add_argument("--replicas", type=int)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--local-steps", type=int, default=4)
    parser.add_argument("--exchange-interval", type=int, default=1)
    parser.add_argument("--energy", choices=("l1", "nnz_l1"), default="nnz_l1")
    parser.add_argument("--mismatch-weight", type=int, default=4)
    parser.add_argument("--repair", action="store_true")
    parser.add_argument("--repair-interval", type=int, default=10)
    parser.add_argument("--repair-max-pool-size", type=int, default=256)
    parser.add_argument("--repair-max-pair-records", type=int, default=32896)
    parser.add_argument("--repair-max-seconds", type=float, default=5.0)
    parser.add_argument("--wall-clock-seconds", type=float)
    parser.add_argument("--history-interval", type=int, default=1)
    parser.add_argument("--allow-repeated-actions", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must not exist")
    validate_pool_mode(args.pool, args.oracle_positive_control)
    if args.oracle_positive_control and args.pool != "strassen_oracle":
        parser.error("--oracle-positive-control requires --pool strassen_oracle")
    temperatures = tuple(float(value) for value in args.temperatures.split(","))
    target = matrix_multiplication_tensor(2, 2, 2)
    pool, pool_provenance = build_pool(args.pool, target)
    config = TemperedSearchConfig(
        rank=args.rank, temperatures=temperatures, replicas=args.replicas or len(temperatures), rounds=args.rounds,
        local_steps_per_round=args.local_steps, exchange_interval=args.exchange_interval,
        energy_mode=args.energy, mismatch_weight=args.mismatch_weight, seed=args.seed,
        allow_repeated_actions=args.allow_repeated_actions, repair_enabled=args.repair,
        repair_interval=args.repair_interval, repair_maximum_pool_size=args.repair_max_pool_size,
        repair_maximum_pair_records=args.repair_max_pair_records,
        repair_maximum_seconds=args.repair_max_seconds,
        wall_clock_seconds=args.wall_clock_seconds, history_interval=args.history_interval,
    )
    result = parallel_tempering_search(target, pool, config)
    oracle = args.pool == "strassen_oracle"
    config_payload = {"schema": "tensor_game_tempered_search_config.v1", **asdict(config),
                      "pool": args.pool, "pool_hash": pool_identity(pool),
                      "pool_size": len(pool), "cpu_only": True,
                      "oracle_data_permitted": oracle}
    summary = {key: result[key] for key in (
        "status", "stop_reason", "rounds_completed", "best_metrics", "exact_reconstruction",
        "repair_modified_best", "counters", "unique_canonical_states_visited", "pool_hash")}
    summary.update({"schema": "tensor_game_tempered_search_summary.v1",
                    "oracle_assisted": oracle,
                    "interpretation": ("Oracle-assisted mechanics validation; not a discovery."
                                       if oracle else "Discovery-mode bounded experiment; exact validation is mandatory.")})
    best = {"schema": "tensor_game_tempered_search_candidate.v1",
            "action_ids": result["best_action_ids"],
            "actions": [action_payload(pool[index]) for index in result["best_state"]],
            "metrics": result["best_metrics"], "exact_reconstruction": result["exact_reconstruction"],
            "repair_modified_candidate": result["repair_modified_best"]}
    provenance = {"schema": "tensor_game_tempered_search_provenance.v1",
                  "source_branch": git_value("branch", "--show-current"),
                  "source_commit": git_value("rev-parse", "HEAD"),
                  "target_digest": tensor_digest(target), "pool_name": args.pool,
                  "pool_hash": pool_identity(pool), "pool_provenance": pool_provenance,
                  "master_seed": args.seed, "oracle_data_permitted": oracle,
                  "oracle_assisted": oracle, "python": sys.version, "platform": platform.platform()}
    args.output.mkdir(parents=True, exist_ok=False)
    files = {"config.json": config_payload, "summary.json": summary,
             "best_candidate.json": best, "provenance.json": provenance}
    for name, payload in files.items():
        (args.output / name).write_bytes(encoded(payload))
    history_path = args.output / "search_history.jsonl"
    history_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in result["history"]), encoding="utf-8")
    names = (*files, "search_history.jsonl")
    hashes = {name: hashlib.sha256((args.output / name).read_bytes()).hexdigest() for name in names}
    (args.output / "artifact_hashes.json").write_bytes(encoded({"schema": "tensor_game_tempered_search_hashes.v1", "sha256": hashes}))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
