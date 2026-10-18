#!/usr/bin/env python3
"""Run bounded exact completion-depth and algebraic recombination diagnostics."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import platform
import shlex
import sys
import tracemalloc

from matmul_inspector.tensor_game import matrix_multiplication_tensor, standard_2x2_decomposition, strassen_2x2_decomposition
from matmul_inspector.tensor_game_experience import action_identity
from matmul_inspector.tensor_game_meet_in_middle import canonical_action_pool
from matmul_inspector.tensor_game_tempered_search import (
    TemperedSearchConfig, parallel_tempering_search, pool_identity, residual_metrics,
    state_identity, validate_pool_mode,
)
from matmul_inspector.tensor_game_completion_depth import (
    CompletionConfig, audit_completion_depth, build_completion_lookup, recombine_parents,
)
from run_tensor_game_tempered_search import build_pool, encoded, git_value


def _jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def load_candidates(source: Path, pool, *, oracle_assisted: bool) -> list[dict]:
    """Read TG-Global-2 final best states with pool and oracle provenance checks."""
    provenance = json.loads((source / "provenance.json").read_text())
    configuration = json.loads((source / "config.json").read_text())
    active_hash = pool_identity(pool)
    if provenance["pool_hash"] != active_hash or configuration["pool_hash"] != active_hash:
        raise ValueError("source candidate pool hash does not match the active pool")
    if bool(provenance["oracle_assisted"]) != oracle_assisted or bool(configuration["oracle_data_permitted"]) != oracle_assisted:
        raise ValueError("source candidate oracle status does not match the requested mode")
    records = []
    for row in _jsonl(source / "paired_runs.jsonl"):
        if row["pool_hash"] != active_hash:
            raise ValueError("candidate pool hash does not match")
        state = tuple(row["best_state"])
        if row["best_action_ids"] != [action_identity(pool[index]) for index in state]:
            raise ValueError("candidate action identifiers do not match the pool")
        digest = state_identity(state, pool)
        if digest != row["best_candidate_hash"]:
            raise ValueError("candidate hash does not match canonical action identities")
        records.append({"candidate_hash": digest, "state": state,
                        "metrics": row["best_metrics"], "source": {"kind": "tg_global_2_final_best",
                        "seed": row["seed"], "variant": row["variant"]}})
    return records


def retained_improvements(pool, target, seeds: tuple[int, ...], *, limit: int) -> list[dict]:
    """Replay bounded TG-Global-2 settings to retain actual strict incumbents."""
    records = []
    for seed in seeds:
        config = TemperedSearchConfig(rank=7, temperatures=(1.0, 2.0), replicas=2,
            rounds=4, local_steps_per_round=2, seed=seed, stop_on_exact=False,
            retain_improvements_limit=limit)
        result = parallel_tempering_search(target, pool, config)
        for row in result["retained_improvements"]:
            state = tuple(row["state"])
            records.append({"candidate_hash": state_identity(state, pool), "state": state,
                            "metrics": row["metrics"], "source": {"kind": ("new_replay_initial_best" if row["round"] == 0
                                                                            else "new_strict_incumbent_replay"),
                            "seed": seed, "round": row["round"], "proposal_index": row["proposal_index"]}})
    return records


def aggregate(audits: list[dict], recombinations: list[dict], lookup, *, oracle_assisted: bool) -> dict:
    counts = {}
    for depth in (2, 3, 4):
        rows = [audit["depths"][str(depth)] for audit in audits if str(depth) in audit["depths"]]
        count = sum(row["cores_with_verified_matches"] > 0 for row in rows)
        counts[str(depth)] = {"candidates_audited": len(rows), "candidates_completable": count,
                              "fraction_completable": count / len(rows) if rows else None,
                              "budget_limited_cores": sum(row["budget_limited"] for row in rows),
                              "lookup_count": sum(row["lookup_count"] for row in rows),
                              "raw_matches": sum(row["raw_matches"] for row in rows),
                              "verified_matches": sum(row["verified_matches"] for row in rows)}
    children = [child for row in recombinations for child in row["partial_children"]]
    exact = {tuple(state) for child in children for state in child["exact_states"]}
    energy_buckets = Counter((audit["metrics"]["energy"] // 10) * 10 for audit in audits)
    bucket_rows = {}
    for bucket in sorted(energy_buckets):
        group = [audit for audit in audits if (audit["metrics"]["energy"] // 10) * 10 == bucket]
        bucket_rows[str(bucket)] = {"candidates": len(group),
                                   "completable_by_depth": {str(depth): sum(
                                       audit["depths"].get(str(depth), {}).get("cores_with_verified_matches", 0) > 0
                                       for audit in group) for depth in (2, 3, 4)}}
    return {"schema": "tensor_game_completion_depth_aggregate.v1",
            "oracle_assisted": oracle_assisted, "candidates_audited": len(audits),
            "completion_depths": counts,
            "candidate_energy_buckets_tens": dict(sorted(energy_buckets.items())),
            "energy_bucket_completion": bucket_rows,
            "work": {key: sum(audit["depths"][str(depth)]["work"][key]
                              for audit in audits for depth in (2, 3, 4)
                              if str(depth) in audit["depths"])
                     for key in ("tensor_additions", "tensor_subtractions", "pair_sums_hashed",
                                 "exact_tensor_equalities", "candidate_reconstructions",
                                 "probe_seconds", "validation_seconds")},
            "pair_table": {"status": lookup.status, "pair_records": lookup.pair_records,
                           "unique_pair_sums": lookup.unique_pair_sums,
                           "pair_sums_hashed": lookup.pair_sums_hashed,
                           "build_seconds": lookup.build_seconds},
            "recombination": {"parent_pairs_tested": len(recombinations),
                              "wall_clock_seconds": sum(row["wall_clock_seconds"] for row in recombinations),
                              "partial_children_generated": len(children),
                              "completion_attempts": sum(child["completion_attempts"] for child in children),
                              "complement_probes": sum(child["completion_probes"] for child in children),
                              "exact_completions": sum(child["verified_matches"] for child in children),
                              "unique_verified_rank7_candidates": len(exact),
                              "best_pre_completion_residual": (min((child["pre_completion_metrics"] for child in children),
                                    key=lambda row: (row["energy"], row["nonzero_entries"])) if children else None),
                              "completion_depth_distribution": {"3": len(children)}},
            "interpretation": "Oracle mechanics control." if oracle_assisted else
                "Bounded finite-pool diagnostic; budget-limited depths remain unknown."}


def main() -> int:
    parser = argparse.ArgumentParser(description="TG-Global-3 exact completion and recombination")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-artifacts", type=Path,
                        default=Path("artifacts/tensor-game/tempered-study-pilot-v1"))
    parser.add_argument("--pool", choices=("strassen_oracle", "algebraic_slice_projection_128"),
                        default="algebraic_slice_projection_128")
    parser.add_argument("--oracle-positive-control", action="store_true")
    parser.add_argument("--depths", default="2,3,4")
    parser.add_argument("--max-pool-size", type=int, default=128)
    parser.add_argument("--max-pair-records", type=int, default=8128)
    parser.add_argument("--max-unique-pair-sums", type=int, default=8128)
    parser.add_argument("--max-probes", type=int, default=10000)
    parser.add_argument("--max-verified-per-core", type=int, default=1)
    parser.add_argument("--max-seconds", type=float, default=10.0)
    parser.add_argument("--allow-repeated-actions", action="store_true")
    parser.add_argument("--recombine", action="store_true")
    parser.add_argument("--top-k-parents", type=int, default=8)
    parser.add_argument("--max-parent-pairs", type=int, default=4)
    parser.add_argument("--max-partial-children", type=int, default=4)
    parser.add_argument("--recombination-seed", type=int, default=17)
    parser.add_argument("--retain-improvements", type=int, default=8)
    parser.add_argument("--replay-seeds", default="20261018,20261019")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must not exist")
    if args.oracle_positive_control != (args.pool == "strassen_oracle"):
        parser.error("oracle-only pool requires explicit positive-control mode")
    validate_pool_mode(args.pool, args.oracle_positive_control)
    target = matrix_multiplication_tensor(2, 2, 2)
    pool, pool_provenance = build_pool(args.pool, target)
    config = CompletionConfig(depths=tuple(int(value) for value in args.depths.split(",")),
        maximum_pool_size=args.max_pool_size, maximum_pair_records=args.max_pair_records,
        maximum_unique_pair_sums=args.max_unique_pair_sums,
        maximum_complement_probes=args.max_probes,
        maximum_verified_per_core=args.max_verified_per_core,
        maximum_seconds=args.max_seconds,
        allow_repeated_actions=args.allow_repeated_actions)
    config.validate()
    tracemalloc.start()
    lookup = build_completion_lookup(pool, config, oracle_assisted=args.oracle_positive_control)
    if args.oracle_positive_control:
        strassen = canonical_action_pool(strassen_2x2_decomposition())
        indices = {action: index for index, action in enumerate(pool)}
        exact_state = tuple(sorted(indices[action] for action in strassen))
        candidates = [{"candidate_hash": state_identity(exact_state, pool), "state": exact_state,
                       "metrics": asdict(residual_metrics(target, tuple(pool[i] for i in exact_state),
                                   config.energy_mode, config.mismatch_weight)),
                       "source": {"kind": "known_strassen_oracle"}}]
        if args.recombine:
            # Two artificial seven-term parents share terms but each lacks parts of Strassen.
            extras = canonical_action_pool(standard_2x2_decomposition())
            control_pool = canonical_action_pool((*pool, *extras))
            lookup = build_completion_lookup(control_pool, config, oracle_assisted=True)
            pool = control_pool
            pool_provenance = {**pool_provenance, "method": "known Strassen actions plus conventional control fillers",
                               "conventional_fillers": len(extras)}
            indices = {action: index for index, action in enumerate(pool)}
            exact_state = tuple(sorted(indices[action] for action in strassen))
            candidates = [{"candidate_hash": state_identity(exact_state, pool), "state": exact_state,
                           "metrics": asdict(residual_metrics(target, tuple(pool[i] for i in exact_state),
                                       config.energy_mode, config.mismatch_weight)),
                           "source": {"kind": "known_strassen_oracle"}}]
            left = tuple(sorted(indices[action] for action in (*strassen[:4], *extras[:3])))
            right = tuple(sorted(indices[action] for action in (*strassen[4:], *extras[3:7])))
            parents = [{"candidate_hash": state_identity(state, pool), "state": state,
                        "metrics": asdict(residual_metrics(target, tuple(pool[i] for i in state),
                                    config.energy_mode, config.mismatch_weight))}
                       for state in (left, right)]
        else:
            parents = []
    else:
        candidates = load_candidates(args.input_artifacts, pool, oracle_assisted=False)
        replay_seeds = tuple(int(value) for value in args.replay_seeds.split(",") if value)
        if args.retain_improvements and replay_seeds:
            candidates.extend(retained_improvements(pool, target, replay_seeds,
                              limit=args.retain_improvements))
        parents = []
    unique = {}
    for row in candidates:
        existing = unique.get(row["candidate_hash"])
        if existing is None:
            unique[row["candidate_hash"]] = {**row, "sources": [row["source"]]}
        else:
            existing["sources"].append(row["source"])
    candidates = sorted(unique.values(), key=lambda row: (row["metrics"]["energy"], row["candidate_hash"]))
    audits = [audit_completion_depth(target, row["state"], pool, lookup, config,
                oracle_assisted=args.oracle_positive_control, source={"records": row["sources"]}) for row in candidates]
    if args.recombine:
        if not args.oracle_positive_control:
            parents = candidates
        recombinations = recombine_parents(target, parents, pool, lookup, config,
            oracle_assisted=args.oracle_positive_control, top_k=args.top_k_parents,
            maximum_parent_pairs=args.max_parent_pairs,
            maximum_partial_children_per_pair=args.max_partial_children,
            seed=args.recombination_seed)
    else:
        recombinations = []
    aggregate_result = aggregate(audits, recombinations, lookup,
                                 oracle_assisted=args.oracle_positive_control)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    aggregate_result["peak_traced_memory_bytes"] = peak
    exact = {}
    for audit in audits:
        for row in audit["exact_candidates"]:
            exact.setdefault(row["candidate_hash"], row)
    for pair in recombinations:
        for child in pair["partial_children"]:
            for state in child["exact_states"]:
                digest = state_identity(state, pool)
                exact.setdefault(digest, {"candidate_hash": digest, "state": state,
                               "exact_reconstruction": True,
                               "oracle_assisted": args.oracle_positive_control})
    known_ids = sorted(action_identity(action) for action in canonical_action_pool(strassen_2x2_decomposition()))
    for row in exact.values():
        row["action_ids"] = [action_identity(pool[i]) for i in row["state"]]
        row["matches_known_strassen_action_multiset"] = sorted(row["action_ids"]) == known_ids
        row["novelty_assessed"] = False
    configuration = {"schema": "tensor_game_completion_depth_config.v1", **asdict(config),
                     "pool": args.pool, "pool_hash": pool_identity(pool),
                     "input_artifacts": str(args.input_artifacts) if not args.oracle_positive_control else None,
                     "recombine": args.recombine, "top_k_parents": args.top_k_parents,
                     "maximum_parent_pairs": args.max_parent_pairs,
                     "maximum_partial_children_per_pair": args.max_partial_children,
                     "recombination_seed": args.recombination_seed,
                     "retained_improvements": args.retain_improvements,
                     "replay_seeds": args.replay_seeds,
                     "oracle_data_permitted": args.oracle_positive_control}
    provenance = {"schema": "tensor_game_completion_depth_provenance.v1",
                  "source_branch": git_value("branch", "--show-current"),
                  "source_commit": git_value("rev-parse", "HEAD"),
                  "command": shlex.join(sys.argv), "python": sys.version,
                  "platform": platform.platform(), "pool_hash": pool_identity(pool),
                  "pool_provenance": pool_provenance,
                  "source_tg_global_2_artifacts": str(args.input_artifacts) if not args.oracle_positive_control else None,
                  "source_candidate_hashes": [row["candidate_hash"] for row in candidates],
                  "oracle_assisted": args.oracle_positive_control,
                  "known_strassen_used_for_pool_or_search": args.oracle_positive_control,
                  "known_strassen_consulted_only_after_search_for_reference_comparison": not args.oracle_positive_control}
    notes = ("# TG-Global-3 evidence\n\n"
             f"Pool: `{args.pool}`. Oracle-assisted: `{args.oracle_positive_control}`.\n\n"
             "Exact tensor sums, hash-confirmed complement lookup, bounded probes, and independent reconstruction. "
             "A bounded absence of a match is not a proof of rank-7 impossibility.\n")
    args.output.mkdir(parents=True, exist_ok=False)
    files = {"config.json": encoded(configuration), "provenance.json": encoded(provenance),
             "candidate_manifest.json": encoded(candidates),
             "completion_depth_results.jsonl": b"".join((json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode() for row in audits),
             "recombination_results.jsonl": b"".join((json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode() for row in recombinations),
             "aggregate_summary.json": encoded(aggregate_result),
             "exact_candidates.json": encoded(list(exact.values())), "study_notes.md": notes.encode()}
    for name, content in files.items():
        (args.output / name).write_bytes(content)
    (args.output / "artifact_hashes.json").write_bytes(encoded({
        "schema": "tensor_game_completion_depth_hashes.v1",
        "sha256": {name: hashlib.sha256(content).hexdigest() for name, content in files.items()}}))
    print(json.dumps({"output": str(args.output), "summary": aggregate_result}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
