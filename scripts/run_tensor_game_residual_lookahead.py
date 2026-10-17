#!/usr/bin/env python3
"""Run the bounded, CPU-only Tensor Game residual lookahead study."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

from matmul_inspector.tensor_game import (
    RankOneAction,
    matrix_multiplication_tensor,
    standard_2x2_decomposition,
    strassen_2x2_decomposition,
    subtract_action,
)
from matmul_inspector.tensor_game_bounded_residual import (
    action_from_payload,
    action_identity,
    bounded_residual_candidates_memory_efficient,
    squared_norm,
)
from matmul_inspector.tensor_game_residual_lookahead import (
    bounded_two_step_scores,
    exact_two_step_score,
    residual_subspace_features,
)


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def unique(actions):
    by_id = {action_identity(action): action for action in actions}
    return tuple(by_id[key] for key in sorted(by_id))


def provider_actions(residual, limit):
    rows, audit = bounded_residual_candidates_memory_efficient(residual, 128, limit)
    return tuple(action_from_payload(row["action"]) for row in rows), audit


def subtract_prefix(target, actions, depth):
    residual = target
    for action in actions[:depth]:
        residual = subtract_action(residual, action)
    return residual


def state_specs():
    target = matrix_multiplication_tensor(2, 2, 2)
    conventional = standard_2x2_decomposition()
    strassen = strassen_2x2_decomposition()
    return (
        ("target/depth-0", target, "shared", ()),
        ("conventional/forward/depth-1", subtract_prefix(target, conventional, 1), "conventional", conventional[:1]),
        ("conventional/forward/depth-4", subtract_prefix(target, conventional, 4), "conventional", conventional[:4]),
        ("strassen/forward/depth-1", subtract_prefix(target, strassen, 1), "strassen", strassen[:1]),
        ("strassen/forward/depth-3", subtract_prefix(target, strassen, 3), "strassen", strassen[:3]),
        (
            "strassen/reverse/depth-3",
            subtract_prefix(target, tuple(reversed(strassen)), 3),
            "strassen",
            tuple(reversed(strassen))[:3],
        ),
        ("strassen/forward/depth-5", subtract_prefix(target, strassen, 5), "strassen", strassen[:5]),
    )


def structural_key(residual, action):
    features = residual_subspace_features(subtract_action(residual, action))
    ranks = features["flattening_ranks"]
    return (sum(ranks), max(ranks), features["nonzero_entries"], features["coefficient_l1"],
            features["squared_norm"], action_identity(action)), features


def ranked_actions(residual, actions):
    immediate = sorted(actions, key=lambda action: (
        squared_norm(subtract_action(residual, action)), action_identity(action)))
    structural_rows = [(action, *structural_key(residual, action)) for action in actions]
    structural_rows.sort(key=lambda row: row[1])
    immediate_rank = {action_identity(action): rank for rank, action in enumerate(immediate, 1)}
    structural_rank = {action_identity(row[0]): rank for rank, row in enumerate(structural_rows, 1)}
    return immediate, structural_rows, immediate_rank, structural_rank


def continuation_map(residual, first_actions, limit, probes):
    result = {}
    audits = {}
    for first in first_actions:
        after = subtract_action(residual, first)
        actions, audit = provider_actions(after, limit)
        candidates = unique(actions + probes)
        if probes:
            candidates = tuple(sorted(candidates, key=lambda action: (
                squared_norm(subtract_action(after, action)), action_identity(action))))
        result[action_identity(first)] = candidates
        audits[action_identity(first)] = audit
    return result, audits


def summarize_distribution(values):
    if not values:
        return {"count": 0, "minimum": None, "maximum": None, "mean": None}
    return {"count": len(values), "minimum": min(values), "maximum": max(values),
            "mean": sum(values) / len(values), "histogram": dict(sorted(Counter(values).items()))}


def trajectory_pairs(name, actions):
    target = matrix_multiplication_tensor(2, 2, 2)
    ordered = []
    endpoints = {}
    for first_index, second_index in itertools.permutations(range(len(actions)), 2):
        score = exact_two_step_score(target, actions[first_index], actions[second_index])
        endpoint_key = tuple(sorted((first_index, second_index)))
        endpoints.setdefault(endpoint_key, score["second_squared_norm"])
        ordered.append({
            "family": name, "first_term": first_index, "second_term": second_index,
            "first_action_id": score["first_action_id"], "second_action_id": score["second_action_id"],
            "immediate_reduction": score["immediate_reduction"],
            "second_step_improvement": score["second_step_improvement"],
            "two_step_reduction": score["two_step_reduction"],
            "first_features": score["first_features"], "second_features": score["second_features"],
        })
    best_by_first = []
    for first_index in range(len(actions)):
        candidates = [row for row in ordered if row["first_term"] == first_index]
        best = max(candidates, key=lambda row: (
            row["second_step_improvement"], row["two_step_reduction"], -row["second_term"]))
        best_by_first.append({
            "first_term": first_index, "first_action_id": best["first_action_id"],
            "first_action_reduction": best["immediate_reduction"],
            "best_second_term": best["second_term"],
            "best_second_action_id": best["second_action_id"],
            "best_second_step_improvement": best["second_step_improvement"],
            "best_two_step_reduction": best["two_step_reduction"],
        })
    return {
        "family": name, "ordered_pair_count": len(ordered), "unique_endpoint_count": len(endpoints),
        "negative_first_positive_second": sum(
            row["immediate_reduction"] < 0 and row["second_step_improvement"] > 0 for row in ordered),
        "negative_two_step_total": sum(row["two_step_reduction"] < 0 for row in ordered),
        "positive_two_step_total": sum(row["two_step_reduction"] > 0 for row in ordered),
        "best_continuation_by_first_term": best_by_first,
        "ordered_pairs": ordered,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--first-actions", type=int, default=12)
    parser.add_argument("--continuations", type=int, default=12)
    parser.add_argument("--total-evaluations", type=int, default=144)
    parser.add_argument("--oracle-first-actions", type=int, default=32)
    parser.add_argument("--oracle-total-evaluations", type=int, default=384)
    parser.add_argument("--cpu-seconds", type=float, default=120.0)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must not exist")
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    target = matrix_multiplication_tensor(2, 2, 2)
    conventional = standard_2x2_decomposition()
    strassen = strassen_2x2_decomposition()
    probes = unique(conventional + strassen)
    probe_family = ({action_identity(action): "conventional" for action in conventional}
                    | {action_identity(action): "strassen" for action in strassen})
    states = []
    sensitivity = ((4, 3, 12), (12, 4, 48), (12, 12, 144))
    for label, residual, family, prefix in state_specs():
        if time.perf_counter() - started > args.cpu_seconds:
            raise TimeoutError("explicit CPU runtime limit exceeded")
        state_started = time.perf_counter()
        reachable, provider_audit = provider_actions(residual, args.first_actions)
        oracle = unique(reachable + probes)
        immediate, structural, immediate_rank, structural_rank = ranked_actions(residual, oracle)
        provider_continuations, continuation_audits = continuation_map(
            residual, reachable, args.continuations, ())
        provider_ranked, provider_budget = bounded_two_step_scores(
            residual, reachable, provider_continuations,
            first_action_limit=args.first_actions, continuations_per_action=args.continuations,
            total_evaluation_budget=args.total_evaluations)
        oracle_continuations, oracle_continuation_audits = continuation_map(
            residual, oracle[:args.oracle_first_actions], args.continuations, probes)
        oracle_ranked, oracle_budget = bounded_two_step_scores(
            residual, immediate, oracle_continuations,
            first_action_limit=args.oracle_first_actions, continuations_per_action=args.continuations,
            total_evaluation_budget=args.oracle_total_evaluations)
        two_step_rank = {row["first_action_id"]: row["two_step_rank"] for row in oracle_ranked}
        oracle_ids = {action_identity(action) for action in oracle}
        reachable_ids = {action_identity(action) for action in reachable}
        probe_rows = []
        for action in probes:
            identifier = action_identity(action)
            next_features = residual_subspace_features(subtract_action(residual, action))
            probe_rows.append({
                "action_id": identifier, "family": probe_family[identifier],
                "provider_reachable": identifier in reachable_ids,
                "oracle_present": identifier in oracle_ids,
                "immediate_rank": immediate_rank[identifier],
                "structural_rank": structural_rank[identifier],
                "two_step_rank": two_step_rank.get(identifier),
                "immediate_reduction": squared_norm(residual) - next_features["squared_norm"],
                "residual_features": next_features,
            })
        sensitivity_rows = []
        for first_limit, continuation_limit, budget in sensitivity:
            rows, audit = bounded_two_step_scores(
                residual, immediate, oracle_continuations, first_action_limit=first_limit,
                continuations_per_action=continuation_limit, total_evaluation_budget=budget)
            top_ids = {row["first_action_id"] for row in rows[:first_limit]}
            sensitivity_rows.append({
                "first_action_limit": first_limit, "continuations_per_action": continuation_limit,
                "total_evaluation_budget": budget, "audit": audit,
                "strassen_probe_recall": sum(action_identity(action) in top_ids for action in strassen),
                "negative_immediate_retained": sum(
                    row["immediate_squared_norm"] > squared_norm(residual) for row in rows
                ),
            })
        states.append({
            "state_label": label, "trajectory_family": family,
            "prefix_action_ids": [action_identity(action) for action in prefix],
            "initial_squared_norm": squared_norm(residual),
            "initial_features": residual_subspace_features(residual),
            "provider_pool": {"label": "deployable bounded_128 output", "size": len(reachable),
                              "action_ids": sorted(reachable_ids), "audit": provider_audit},
            "oracle_pool": {"label": "diagnostic only: provider union validated conventional/Strassen probes",
                            "size": len(oracle), "known_coefficients_not_used_by_provider": True},
            "immediate_score_distribution": summarize_distribution(
                [squared_norm(subtract_action(residual, action)) for action in oracle]),
            "two_step_score_distribution": summarize_distribution(
                [row["best_two_step_squared_norm"] for row in oracle_ranked
                 if row["best_two_step_squared_norm"] is not None]),
            "flattening_rank_sum_distribution": summarize_distribution(
                [sum(row[2]["flattening_ranks"]) for row in structural]),
            "provider_lookahead": {"ranked": provider_ranked, "budget": provider_budget,
                                   "continuation_provider_calls": len(continuation_audits)},
            "oracle_lookahead": {"ranked": oracle_ranked, "budget": oracle_budget,
                                 "continuation_provider_calls": len(oracle_continuation_audits)},
            "known_action_rankings": probe_rows, "sensitivity": sensitivity_rows,
            "runtime_seconds": time.perf_counter() - state_started,
        })

    trajectories = [trajectory_pairs("conventional", conventional), trajectory_pairs("strassen", strassen)]
    initial = states[0]
    strassen_rows = [row for row in initial["known_action_rankings"] if row["family"] == "strassen"]
    summary = {
        "schema": "tensor_game_residual_lookahead_summary.v1",
        "state_count": len(states),
        "ordered_trajectory_pairs": {row["family"]: row["ordered_pair_count"] for row in trajectories},
        "unique_trajectory_endpoints": {row["family"]: row["unique_endpoint_count"] for row in trajectories},
        "negative_first_positive_second": {
            row["family"]: row["negative_first_positive_second"] for row in trajectories
        },
        "positive_two_step_total": {row["family"]: row["positive_two_step_total"] for row in trajectories},
        "initial_strassen_provider_recall_at_12": sum(row["provider_reachable"] for row in strassen_rows),
        "initial_strassen_immediate_ranks": [row["immediate_rank"] for row in strassen_rows],
        "initial_strassen_two_step_ranks": [row["two_step_rank"] for row in strassen_rows],
        "initial_strassen_structural_ranks": [row["structural_rank"] for row in strassen_rows],
        "baseline_complete_action_evaluations": len(initial["known_action_rankings"]),
        "provider_lookahead_complete_pair_evaluations": initial["provider_lookahead"]["budget"]
        ["continuation_evaluations"],
        "oracle_lookahead_is_not_deployable": True,
        "total_runtime_seconds": time.perf_counter() - started,
        "failure_states": (
            "No serialized prior failure residuals were available; representative known-trajectory states were used."
        ),
    }
    configuration = {
        "schema": "tensor_game_residual_lookahead_configuration.v1", "command": " ".join(sys.argv),
        "configuration": vars(args) | {"output": str(args.output), "pair_budget": 128,
                                         "coefficient_domain": [-1, 0, 1], "sensitivity": sensitivity},
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
    }
    payloads = {
        "configuration.json": encoded(configuration),
        "states.json": encoded({"schema": "tensor_game_residual_lookahead_states.v1", "states": states}),
        "trajectory_pairs.json": encoded({
            "schema": "tensor_game_residual_lookahead_pairs.v1", "trajectories": trajectories
        }),
        "summary.json": encoded(summary),
    }
    for name, data in payloads.items():
        (args.output / name).write_bytes(data)
    hashes = {name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
              for name, data in payloads.items()}
    (args.output / "artifact_hashes.json").write_bytes(encoded({
        "schema": "tensor_game_residual_lookahead_hashes.v1", "files": hashes}))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
