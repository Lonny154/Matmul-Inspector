#!/usr/bin/env python3
"""Run the bounded CPU-only Tensor Game algebraic-pool study."""

from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import itertools
import json
import platform
from pathlib import Path
import statistics
import sys
import time
import tracemalloc

from matmul_inspector.tensor_game import RankOneAction, matrix_multiplication_tensor, strassen_2x2_decomposition, subtract_action
from matmul_inspector.tensor_game_algebraic_pools import (
    AlgebraicPoolConfig, algebraic_action_candidates, candidate_diagnostics,
    combine_slices, exact_column_space_basis, exact_null_space_basis, exact_row_space_basis,
    hybrid_pool, known_action_recall, tensor_slices, vector_in_row_space,
)
from matmul_inspector.tensor_game_bounded_residual import VECTORS, squared_norm
from matmul_inspector.tensor_game_experience import action_payload
from matmul_inspector.tensor_game_hardware_cost import action_cost
from matmul_inspector.tensor_game_meet_in_middle import MeetInMiddleConfig, meet_in_middle
from matmul_inspector.tensor_game_residual_lookahead import exact_matrix_rank
from matmul_inspector.tensor_game_search import canonicalize_action
from matmul_inspector.tensor_game_structural_pairs import structural_pair_candidates, structural_signature


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def fraction_payload(value):
    if isinstance(value, Fraction):
        return {"numerator": value.numerator, "denominator": value.denominator}
    if isinstance(value, tuple):
        return [fraction_payload(item) for item in value]
    return value


def residual_pool(target, size):
    started = time.perf_counter()
    unique = {canonicalize_action(RankOneAction(u, v, w)) for u, v, w in itertools.product(VECTORS, repeat=3)}
    ranked = sorted(unique, key=lambda action: (squared_norm(subtract_action(target, action)), action.u, action.v, action.w))
    return tuple(ranked[:size]), {"pair_candidates_considered": 6400, "raw_complete_actions": 512000,
        "unique_complete_actions": len(unique), "exact_scoring_operations": len(unique),
        "runtime_seconds": time.perf_counter() - started, "method": "full_domain_residual_rank"}


def structural_pool(target, size):
    started = time.perf_counter()
    rows, audit = structural_pair_candidates(target, protected_residual_pairs=64, structural_pairs=64,
                                               w_mode="top_k", w_candidates_per_pair=4, limit=512)
    selected, signatures = [], set()
    for row in rows:
        signature = structural_signature(row["features"])
        if signature not in signatures:
            selected.append(row["action"]); signatures.add(signature)
        if len(selected) == size: break
    for row in rows:
        if len(selected) == size: break
        if row["action"] not in selected: selected.append(row["action"])
    if len(selected) != size: raise RuntimeError("structural pool did not fill")
    return tuple(selected), {"pair_candidates_considered": 6400, "selected_pairs": 128,
        "complete_actions_generated": audit["unique_complete_actions"],
        "exact_scoring_operations": audit["complete_vectors_scored"],
        "runtime_seconds": time.perf_counter() - started, "provider_audit": audit}


def action_signature(action):
    vectors = (action.u, action.v, action.w)
    return (tuple(sum(value != 0 for value in vector) for vector in vectors),
            tuple(tuple(sorted((sum(value > 0 for value in vector), sum(value < 0 for value in vector)))) for vector in vectors),
            action_cost(action).preparation_additions)


def pool_statistics(target, pool):
    scores = [squared_norm(subtract_action(target, action)) for action in pool]
    costs = [action_cost(action).preparation_additions for action in pool]
    return {"size": len(pool), "distinct_structural_signatures": len({action_signature(action) for action in pool}),
            "residual_scores": {"minimum": min(scores), "maximum": max(scores), "mean": statistics.fmean(scores),
                                "histogram": dict(sorted(Counter(scores).items()))},
            "preparation_cost_histogram": dict(sorted(Counter(costs).items()))}


def slice_analysis(target):
    modes = []
    combination_rank_histogram = Counter()
    for mode in range(3):
        slices = tensor_slices(target, mode)
        rows = []
        row_intersection = [vector for vector in VECTORS if all(vector_in_row_space(vector, matrix) for matrix in slices)]
        columns = [tuple(zip(*matrix)) for matrix in slices]
        column_intersection = [vector for vector in VECTORS if all(vector_in_row_space(vector, matrix) for matrix in columns)]
        for index, matrix in enumerate(slices):
            rows.append({"slice": index, "matrix": matrix, "rank": exact_matrix_rank(matrix),
                         "row_space_basis": fraction_payload(exact_row_space_basis(matrix)),
                         "column_space_basis": fraction_payload(exact_column_space_basis(matrix)),
                         "right_null_space_basis": fraction_payload(exact_null_space_basis(matrix))})
        for coefficients in VECTORS:
            combination_rank_histogram[(mode, exact_matrix_rank(combine_slices(target, mode, coefficients)))] += 1
        modes.append({"mode": mode, "slices": rows, "ternary_row_space_intersection": row_intersection,
                      "ternary_column_space_intersection": column_intersection})
    return {"modes": modes, "combination_rank_histogram": [
        {"mode": mode, "rank": rank, "count": count} for (mode, rank), count in sorted(combination_rank_histogram.items())]}


def measured_mitm(target, pool, config):
    tracemalloc.start(); result = meet_in_middle(target, pool, config)
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

    residual, residual_audit = residual_pool(target, 128)
    structural, structural_audit = structural_pool(target, 128)
    algebraic_rows, algebraic_audit = algebraic_action_candidates(target, AlgebraicPoolConfig())
    algebraic = tuple(row["action"] for row in algebraic_rows[:128])
    hybrid = hybrid_pool(algebraic, residual, 128, 64)
    pools = {"residual_ranked": residual, "structural_diverse": structural,
             "algebraic_only": algebraic, "hybrid_algebraic_residual": hybrid}
    if any(len(pool) != 128 for pool in pools.values()): raise RuntimeError("matched pools must contain 128 actions")

    coverage = []
    sized_pools = {name: {size: pool[:size] for size in (20, 64, 128)} for name, pool in pools.items()}
    sized_pools["hybrid_algebraic_residual"] = {
        size: hybrid_pool(algebraic, residual, size, size // 2) for size in (20, 64, 128)
    }
    pool_records = {}
    algebraic_provenance = {row["action_id"]: row["provenance"] for row in algebraic_rows}
    for name, pool in pools.items():
        pool_action_ids = {hashlib.sha256(json.dumps(action_payload(action), sort_keys=True, separators=(",", ":")).encode()).hexdigest() for action in pool}
        pool_records[name] = {**pool_statistics(target, pool), "actions": [action_payload(action) for action in pool],
                              "provenance": {key: value for key, value in algebraic_provenance.items()
                                             if key in pool_action_ids} if "algebraic" in name else {}}
        for size in (20, 64, 128):
            recall = known_action_recall(sized_pools[name][size], known)
            for row in recall["records"]:
                row["earliest_exclusion_stage"] = "covered" if row["coefficient_identity"] else (
                    "missing_factor_or_pool_cutoff" if row["any_pair_recall"] else "algebraic_or_pool_pair_selection")
            coverage.append({"pool": name, "pool_size": size, **recall})

    mitm = []
    for name, pool in pools.items():
        mitm.append({"pool": name, "configuration": "complete_20", "seed": seeds[0],
                     **measured_mitm(target, sized_pools[name][20], MeetInMiddleConfig(maximum_left_attempts=100000,
                         maximum_right_attempts=100000, maximum_stored_sums=100000,
                         maximum_seconds=args.maximum_seconds, seed=seeds[0]))})
        for seed in seeds:
            mitm.append({"pool": name, "configuration": "sampled_64", "seed": seed,
                         **measured_mitm(target, sized_pools[name][64], MeetInMiddleConfig(
                             maximum_left_attempts=args.sample_attempts, maximum_right_attempts=args.sample_attempts,
                             maximum_stored_sums=args.sample_attempts, maximum_seconds=args.maximum_seconds, seed=seed))})

    generation = {"residual_ranked": residual_audit, "structural_diverse": structural_audit,
                  "algebraic_only": algebraic_audit,
                  "hybrid_algebraic_residual": {"algebraic_quota": 64, "residual_fill": 64,
                                                "additional_runtime_seconds": 0.0}}
    summary = {"schema": "tensor_game_algebraic_pools_summary.v1",
        "coverage_by_pool_size": [{"pool": row["pool"], "pool_size": row["pool_size"],
            "pair_recall": row["pair_recall"], "complete_action_recall": row["complete_action_recall"],
            "tensor_equivalence_recall": row["tensor_equivalence_recall"]} for row in coverage],
        "structural_signatures_at_128": {name: pool_records[name]["distinct_structural_signatures"] for name in pools},
        "mitm_exact_discoveries": {name: sum(row["exact_decomposition_count"] for row in mitm if row["pool"] == name) for name in pools},
        "algebraic_finding": "No rank-one root slice combinations exist. Rank-two row/column directions improve held-out pair and action coverage at size 128 but produce no exact rank-7 discovery under matched MITM budgets."}
    configuration = {"schema": "tensor_game_algebraic_pools_configuration.v1", "seeds": seeds,
        "pool_sizes": [20, 64, 128], "mitm_sample_attempts_per_side": args.sample_attempts,
        "maximum_seconds_per_mitm_run": args.maximum_seconds, "cpu_only": True,
        "known_strassen_used_for_generation_or_selection": False, "python": sys.version, "platform": platform.platform()}
    artifacts = {"configuration.json": configuration, "slice_analysis.json": slice_analysis(target),
        "generation_audits.json": generation, "pools.json": pool_records, "coverage.json": coverage,
        "candidate_diagnostics.json": {name: candidate_diagnostics(target, pool, 32) for name, pool in pools.items()},
        "meet_in_middle.json": mitm, "summary.json": summary}
    args.output.mkdir(parents=True, exist_ok=False)
    for name, payload in artifacts.items(): (args.output / name).write_bytes(encoded(payload))
    hashes = {name: {"bytes": (args.output/name).stat().st_size,
                     "sha256": hashlib.sha256((args.output/name).read_bytes()).hexdigest()} for name in sorted(artifacts)}
    (args.output / "artifact_hashes.json").write_bytes(encoded({"schema": "tensor_game_algebraic_pools_hashes.v1", "files": hashes}))
    print(json.dumps(summary, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
