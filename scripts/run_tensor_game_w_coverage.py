#!/usr/bin/env python3
"""Run exhaustive offline Strassen proposal-coverage diagnostics."""

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
    matrix_multiplication_tensor,
    standard_2x2_decomposition,
    strassen_2x2_decomposition,
    subtract_action,
    zero_tensor,
)
from matmul_inspector.tensor_game_bounded_residual import action_identity, bounded_residual_candidates_memory_efficient
from matmul_inspector.tensor_game_candidate_dataset import validate_decomposition
from matmul_inspector.tensor_game_quota_w import quota_multi_w_candidates
from matmul_inspector.tensor_game_w_coverage import (
    analyze_action_at_state,
    exact_complete_action_ranks,
    generated_complete_action_rank,
)


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def residual_for_subset(actions, subset):
    residual = matrix_multiplication_tensor(2, 2, 2)
    for index in sorted(subset):
        residual = subtract_action(residual, actions[index])
    return residual


def provider_ids(residual):
    baseline, _ = bounded_residual_candidates_memory_efficient(residual, 128, 12)
    result = {"baseline": {row["action_id"] for row in baseline}}
    for strategy in ("residual", "diversity"):
        for protected, alternative in ((11, 1), (10, 2), (8, 4)):
            rows, _ = quota_multi_w_candidates(
                residual, protected_baseline=protected, alternative_slots=alternative,
                w_candidates_per_pair=2, strategy=strategy,
            )
            result[f"quota_{strategy}_{protected}_{alternative}"] = {row["action_id"] for row in rows}
    return result


def first_missing(order, edges, provider):
    subset = frozenset()
    for depth, term in enumerate(order):
        edge = edges[(subset, term)]
        if not edge["provider_returned"][provider]:
            return {"depth": depth + 1, "term": term + 1, "stage": edge["provider_exclusion"][provider]}
        subset = subset | {term}
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must not exist")
    args.output.mkdir(parents=True)
    started = time.perf_counter()

    target = matrix_multiplication_tensor(2, 2, 2)
    strassen = strassen_2x2_decomposition()
    conventional = standard_2x2_decomposition()
    validate_decomposition(strassen)
    full_ranks = exact_complete_action_ranks(target, strassen + conventional)

    edges = {}
    provider_cache = {}
    for size in range(7):
        for subset_tuple in itertools.combinations(range(7), size):
            subset = frozenset(subset_tuple)
            residual = residual_for_subset(strassen, subset)
            providers = provider_cache.setdefault(subset, provider_ids(residual))
            for term in range(7):
                if term in subset:
                    continue
                row = analyze_action_at_state(residual, strassen[term])
                action_id = row["action_id"]
                row["subset"] = [index + 1 for index in sorted(subset)]
                row["next_term"] = term + 1
                row["provider_returned"] = {name: action_id in ids for name, ids in providers.items()}
                row["provider_exclusion"] = {}
                for name, returned in row["provider_returned"].items():
                    if returned:
                        stage = None
                    elif row["best_pair_rank"] > 128:
                        stage = "pair_proposal_selection"
                    elif (row["best_retained_required_w_rank"] or 81) > (1 if name == "baseline" else 2):
                        stage = "alternative_w_generation"
                    else:
                        stage = "final_candidate_budget"
                    row["provider_exclusion"][name] = stage
                edges[(subset, term)] = row

    initial = []
    conventional_ranks = [full_ranks["ranks"][action_identity(action)] for action in conventional]
    for term in range(7):
        row = dict(edges[(frozenset(), term)])
        action_id = row["action_id"]
        row["full_complete_action_rank"] = full_ranks["ranks"][action_id]
        row["full_score_tie_count"] = full_ranks["score_tie_counts"][action_id]
        row["conventional_full_rank_range"] = [min(conventional_ranks), max(conventional_ranks)]
        row["rank_relative_to_best_conventional"] = row["full_complete_action_rank"] - min(conventional_ranks)
        initial.append(row)

    provider_names = tuple(next(iter(provider_cache.values())).keys())
    order_rows = []
    zero = zero_tensor((4, 4, 4))
    for order in itertools.permutations(range(7)):
        subset = frozenset()
        max_pair = max_w = 0
        reductions = []
        edge_rows_for_order = []
        for term in order:
            edge = edges[(subset, term)]
            edge_rows_for_order.append(edge)
            max_pair = max(max_pair, edge["best_pair_rank"])
            max_w = max(max_w, edge["best_required_w_rank"])
            reductions.append(edge["immediate_residual_reduction"])
            subset = subset | {term}
        budget_points = []
        for w_budget in range(1, 81):
            required_widths = []
            for edge in edge_rows_for_order:
                eligible = [item["pair_rank"] for item in edge["representations"] if item["required_w_rank"] <= w_budget]
                required_widths.append(min(eligible) if eligible else 6401)
            pair_width = max(required_widths)
            if pair_width <= 6400:
                budget_points.append({"pair_width": pair_width, "w_budget": w_budget})
        joint_frontier = [point for point in budget_points if not any(
            other["pair_width"] <= point["pair_width"] and other["w_budget"] <= point["w_budget"]
            and other != point for other in budget_points
        )]
        final_residual = residual_for_subset(strassen, subset)
        order_rows.append({
            "order": [term + 1 for term in order],
            "maximum_required_pair_rank": max_pair,
            "maximum_required_w_rank": max_w,
            "joint_budget_frontier": joint_frontier,
            "negative_reduction_steps": sum(value < 0 for value in reductions),
            "residual_reductions": reductions,
            "exact_zero_residual": final_residual == zero,
            "first_missing": {name: first_missing(order, edges, name) for name in provider_names},
        })

    all_joint = [(point["pair_width"], point["w_budget"], row) for row in order_rows for point in row["joint_budget_frontier"]]
    global_points = sorted({(width, budget) for width, budget, _ in all_joint})
    global_frontier = [point for point in global_points if not any(
        other[0] <= point[0] and other[1] <= point[1] and other != point for other in global_points
    )]
    counterfactuals = []
    for selected_width, selected_w in global_frontier:
        supporting = [row for width, budget, row in all_joint if (width, budget) == (selected_width, selected_w)]
        representative = min(supporting, key=lambda row: row["order"])
        counterfactual_steps = []
        subset = frozenset()
        for term_number in representative["order"]:
            term = term_number - 1
            residual = residual_for_subset(strassen, subset)
            ranking = generated_complete_action_rank(
                residual, strassen[term], pair_width=selected_width, w_budget=selected_w
            )
            counterfactual_steps.append({"depth": len(subset) + 1, "term": term_number, **ranking})
            subset = subset | {term}
        counterfactuals.append({
            "ordering": representative["order"],
            "pair_width": selected_width,
            "w_budget": selected_w,
            "steps": counterfactual_steps,
            "minimum_complete_action_budget_for_this_order": max(row["complete_action_rank"] for row in counterfactual_steps),
            "full_trajectory_generated": all(row["generated"] for row in counterfactual_steps),
        })
    provider_summary = {}
    for name in provider_names:
        reachable = [row for row in order_rows if row["first_missing"][name] is None]
        losses = Counter(
            row["first_missing"][name]["stage"] if row["first_missing"][name] else "reachable"
            for row in order_rows
        )
        provider_summary[name] = {
            "fully_reachable_orderings": len(reachable),
            "first_exclusion_stage_counts": dict(sorted(losses.items())),
        }
    summary = {
        "schema": "tensor_game_w_coverage_summary.v1",
        "coefficient_domain": [-1, 0, 1],
        "strassen_terms": 7,
        "subset_states": len(provider_cache),
        "subset_term_edges": len(edges),
        "orderings": len(order_rows),
        "all_orderings_exact_zero": all(row["exact_zero_residual"] for row in order_rows),
        "full_canonical_action_count": full_ranks["canonical_action_count"],
        "provider_reachability": provider_summary,
        "minimum_independent_maximum_pair_rank": min(row["maximum_required_pair_rank"] for row in order_rows),
        "minimum_independent_maximum_w_rank": min(row["maximum_required_w_rank"] for row in order_rows),
        "global_joint_budget_frontier": [{"pair_width": width, "w_budget": budget} for width, budget in global_frontier],
        "joint_frontier_supporting_orderings": sum(
            any((point["pair_width"], point["w_budget"]) in global_frontier for point in row["joint_budget_frontier"])
            for row in order_rows
        ),
        "representative_counterfactuals": counterfactuals,
        "wall_seconds": time.perf_counter() - started,
    }
    edge_rows = [edges[key] for key in sorted(edges, key=lambda key: (len(key[0]), tuple(sorted(key[0])), key[1]))]
    payloads = {
        "initial_actions.json": encoded({"schema": "tensor_game_w_coverage_initial.v1", "actions": initial}),
        "trajectory_edges.json": encoded({"schema": "tensor_game_w_coverage_edges.v1", "edges": edge_rows}),
        "orderings.json": encoded({"schema": "tensor_game_w_coverage_orderings.v1", "orderings": order_rows}),
        "summary.json": encoded(summary),
        "configuration.json": encoded({
            "command": " ".join(sys.argv), "pair_width": 128, "w_budget": 2,
            "environment": {"python": platform.python_version(), "platform": platform.platform()},
            "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        }),
    }
    for name, payload in payloads.items():
        (args.output / name).write_bytes(payload)
    hashes = {name: {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
              for name, payload in payloads.items()}
    (args.output / "artifact_hashes.json").write_bytes(encoded(hashes))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
