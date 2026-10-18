#!/usr/bin/env python3
"""Run the four matched TG-Global-2 CPU search variants."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import shlex
import sys
import tracemalloc

from matmul_inspector.tensor_game import matrix_multiplication_tensor
from matmul_inspector.tensor_game_experience import tensor_digest
from matmul_inspector.tensor_game_tempered_search import TemperedSearchConfig, pool_identity, validate_pool_mode
from matmul_inspector.tensor_game_tempered_study import aggregate_matched_runs, run_matched_study
from run_tensor_game_tempered_search import build_pool, encoded, git_value


def main() -> int:
    parser = argparse.ArgumentParser(description="TG-Global-2 matched CPU study")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pool", choices=("strassen_oracle", "algebraic_slice_projection_128"),
                        default="algebraic_slice_projection_128")
    parser.add_argument("--oracle-positive-control", action="store_true")
    parser.add_argument("--profile", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--seeds", help="comma-separated paired master seeds")
    parser.add_argument("--seed-count", type=int)
    parser.add_argument("--starting-seed", type=int, default=20261018)
    parser.add_argument("--replicas", type=int, default=2)
    parser.add_argument("--temperatures", default="1,2")
    parser.add_argument("--rank", type=int, default=7)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--local-steps", type=int, default=2)
    parser.add_argument("--exchange-interval", type=int, default=1)
    parser.add_argument("--energy", choices=("l1", "nnz_l1"), default="nnz_l1")
    parser.add_argument("--mismatch-weight", type=int, default=4)
    parser.add_argument("--repair-interval", type=int, default=2)
    parser.add_argument("--repair-max-pool-size", type=int, default=256)
    parser.add_argument("--repair-max-pair-records", type=int, default=32896)
    parser.add_argument("--repair-max-seconds", type=float, default=10.0)
    parser.add_argument("--wall-clock-seconds", type=float, default=30.0)
    parser.add_argument("--history-interval", type=int, default=1)
    parser.add_argument("--allow-repeated-actions", action="store_true")
    parser.add_argument("--no-audit", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must not exist")
    if args.oracle_positive_control != (args.pool == "strassen_oracle"):
        parser.error("the oracle pool requires --oracle-positive-control; discovery mode requires a non-oracle pool")
    validate_pool_mode(args.pool, args.oracle_positive_control)
    if args.seeds and args.seed_count:
        parser.error("choose --seeds or --seed-count")
    seeds = (tuple(int(value) for value in args.seeds.split(",")) if args.seeds else
             tuple(range(args.starting_seed, args.starting_seed +
                         (args.seed_count if args.seed_count is not None else (12 if args.profile == "pilot" else 30)))))
    if not seeds:
        parser.error("at least one seed is required")
    temperatures = tuple(float(value) for value in args.temperatures.split(","))
    target = matrix_multiplication_tensor(2, 2, 2)
    pool, pool_provenance = build_pool(args.pool, target)
    config = TemperedSearchConfig(
        rank=args.rank, temperatures=temperatures, replicas=args.replicas,
        rounds=args.rounds if args.rounds is not None else (4 if args.profile == "pilot" else 100),
        local_steps_per_round=args.local_steps, exchange_interval=args.exchange_interval,
        energy_mode=args.energy, mismatch_weight=args.mismatch_weight,
        allow_repeated_actions=args.allow_repeated_actions,
        repair_interval=args.repair_interval,
        repair_maximum_pool_size=args.repair_max_pool_size,
        repair_maximum_pair_records=args.repair_max_pair_records,
        repair_maximum_seconds=args.repair_max_seconds,
        wall_clock_seconds=args.wall_clock_seconds, history_interval=args.history_interval,
    )
    config.validate(len(pool))
    tracemalloc.start()
    initial, runs, audits = run_matched_study(target, pool, config, seeds,
        audit_enabled=not args.no_audit, oracle_assisted=args.oracle_positive_control)
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    aggregate = aggregate_matched_runs(runs)
    aggregate["peak_traced_memory_bytes_study"] = peak_memory
    aggregate["oracle_assisted"] = args.oracle_positive_control
    configuration = {"schema": "tensor_game_tempered_study_config.v1", **asdict(config),
                     "profile": args.profile, "seeds": seeds, "pool": args.pool,
                     "pool_hash": pool_identity(pool), "pool_size": len(pool),
                     "audit_enabled": not args.no_audit, "oracle_data_permitted": args.oracle_positive_control}
    provenance = {"schema": "tensor_game_tempered_study_provenance.v1",
                  "source_branch": git_value("branch", "--show-current"),
                  "source_commit": git_value("rev-parse", "HEAD"),
                  "command": shlex.join(sys.argv), "python": sys.version,
                  "platform": platform.platform(), "target_digest": tensor_digest(target),
                  "pool_hash": pool_identity(pool), "pool_provenance": pool_provenance,
                  "oracle_assisted": args.oracle_positive_control}
    notes = ("# TG-Global-2 evidence\n\n"
             f"Pool: `{args.pool}`. Oracle-assisted: `{args.oracle_positive_control}`.\n\n"
             "Four variants share each seed's initial states, replica temperatures, and local proposal budget. "
             "The pair table is constructed once and reused. Wall times cover search runs, while the study peak memory "
             "covers table construction and all runs. A near-zero residual is not an exact decomposition. "
             "Oracle recovery validates mechanics only. This pilot supports descriptive comparisons only.\n")
    args.output.mkdir(parents=True, exist_ok=False)
    files = {"config.json": encoded(configuration), "provenance.json": encoded(provenance),
             "initial_states.json": encoded(initial), "aggregate_summary.json": encoded(aggregate),
             "per_run_summary.json": encoded([{key: row[key] for key in (
                 "seed", "variant", "status", "exact_reconstruction", "best_metrics", "counters",
                 "initial_state_hashes", "best_action_ids", "best_candidate_hash", "time_to_best_seconds",
                 "best_round", "best_proposal_index", "wall_clock_seconds",
                 "unique_canonical_states_visited", "repairability_audit") if key in row} for row in runs]),
             "paired_runs.jsonl": b"".join((json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode() for row in runs),
             "repairability_audit.jsonl": b"".join((json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode() for row in audits),
             "study_notes.md": notes.encode()}
    for name, content in files.items():
        (args.output / name).write_bytes(content)
    hashes = {name: hashlib.sha256(content).hexdigest() for name, content in files.items()}
    (args.output / "artifact_hashes.json").write_bytes(encoded({
        "schema": "tensor_game_tempered_study_hashes.v1", "sha256": hashes}))
    print(json.dumps({"output": str(args.output), "variants": aggregate["variants"],
                      "paired_comparisons": {key: {k: v for k, v in value.items() if k != "per_seed"}
                                             for key, value in aggregate["paired_comparisons"].items()}},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
