"""Exact offline tensor-flattening rank feasibility diagnostics."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import time
from typing import Any, Sequence

from .tensor_game import RankOneAction, Tensor, subtract_action, tensor_shape, zero_tensor
from .tensor_game_bounded_residual import (
    action_from_payload,
    action_identity,
    bounded_residual_candidates_memory_efficient,
    squared_norm,
)
from .tensor_game_residual_lookahead import exact_matrix_rank, tensor_flattening


def tensor_payload(tensor: Tensor) -> list[list[list[int]]]:
    """Return the stable JSON representation used by offline state artifacts."""
    tensor_shape(tensor)
    return [[list(row) for row in plane] for plane in tensor]


def tensor_digest(tensor: Tensor) -> str:
    """Hash a residual tensor using its compact deterministic JSON encoding."""
    payload = json.dumps(tensor_payload(tensor), separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def flattening_rank_lower_bound(tensor: Tensor) -> dict[str, Any]:
    """Return all exact flattening ranks and their tensor-rank lower bound."""
    shape = tensor_shape(tensor)
    flattening_shapes = []
    ranks = []
    for mode in range(3):
        matrix = tensor_flattening(tensor, mode)
        flattening_shapes.append([len(matrix), len(matrix[0])])
        ranks.append(exact_matrix_rank(matrix))
    return {
        "tensor_shape": list(shape),
        "flattening_shapes": flattening_shapes,
        "flattening_ranks": ranks,
        "lower_bound": max(ranks),
        "field": "rational_numbers",
    }


def feasibility_record(
    residual: Tensor,
    remaining_budget: int,
    *,
    known_completion: bool | None = None,
) -> dict[str, Any]:
    """Test the rigorous field-rank necessary condition for a completion."""
    if not isinstance(remaining_budget, int) or isinstance(remaining_budget, bool):
        raise TypeError("remaining_budget must be an integer")
    if remaining_budget < 0:
        raise ValueError("remaining_budget must be nonnegative")
    bound = flattening_rank_lower_bound(residual)
    nonzeros = sum(value != 0 for plane in residual for row in plane for value in row)
    return {
        "residual_digest": tensor_digest(residual),
        "remaining_budget": remaining_budget,
        "squared_norm": squared_norm(residual),
        "nonzero_entries": nonzeros,
        **bound,
        "lower_bound_margin": bound["lower_bound"] - remaining_budget,
        "provably_infeasible": bound["lower_bound"] > remaining_budget,
        "known_valid_completion": known_completion,
        "passing_bound_semantics": "unknown_feasibility",
    }


def trajectory_feasibility(
    target: Tensor,
    actions: Sequence[RankOneAction],
    order: Sequence[int],
    *,
    family: str,
) -> list[dict[str, Any]]:
    """Evaluate every residual along one validated known-completion order."""
    if sorted(order) != list(range(len(actions))):
        raise ValueError("order must be a permutation of every action index")
    residual = target
    records = []
    for depth in range(len(actions) + 1):
        row = feasibility_record(
            residual, len(actions) - depth, known_completion=True
        )
        row.update({"family": family, "depth": depth, "order": list(order)})
        records.append(row)
        if depth < len(actions):
            residual = subtract_action(residual, actions[order[depth]])
    if residual != zero_tensor(tensor_shape(target)):
        raise ValueError("actions do not reconstruct the target in the requested order")
    return records


def provider_child_feasibility(
    residual: Tensor,
    remaining_budget: int,
    *,
    max_actions: int = 12,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Analyze unchanged bounded-provider children without filtering them."""
    if remaining_budget <= 0:
        raise ValueError("remaining_budget must be positive before taking an action")
    rows, provider_audit = bounded_residual_candidates_memory_efficient(
        residual, 128, max_actions
    )
    children = []
    rank_check_ns = 0
    for row in rows:
        action = action_from_payload(row["action"])
        child = subtract_action(residual, action)
        rank_started = time.perf_counter_ns()
        record = feasibility_record(child, remaining_budget - 1)
        elapsed_ns = time.perf_counter_ns() - rank_started
        rank_check_ns += elapsed_ns
        record.update({
            "action_id": action_identity(action),
            "provider_rank": row["rank"],
            "provider_score": row["score"],
            "parent_squared_norm": squared_norm(residual),
            "immediate_reduction": squared_norm(residual) - row["score"],
            "rank_check_ns": elapsed_ns,
        })
        children.append(record)
    provider_audit = dict(provider_audit)
    provider_audit["rank_check_ns"] = rank_check_ns
    provider_audit["rank_check_mean_ns"] = rank_check_ns / len(children) if children else 0.0
    return children, provider_audit


def sample_frozen_rank7_failures(
    target: Tensor,
    *,
    node_budget: int = 256,
    beam_width: int = 128,
    max_actions: int = 12,
    samples_per_depth: int = 3,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Replay frozen baseline beam ordering and retain representative failed states."""
    if min(node_budget, beam_width, max_actions, samples_per_depth) <= 0:
        raise ValueError("search and sampling budgets must be positive")
    rank = 7
    zero = zero_tensor(tensor_shape(target))
    frontier: list[tuple[Tensor, tuple[str, ...]]] = [(target, ())]
    samples = []
    depth_counts = []
    nodes_expanded = 0
    generated = 0
    exact_solutions = 0
    for depth in range(rank):
        by_residual: dict[Tensor, tuple[str, ...]] = {}
        for residual, path in frontier:
            if nodes_expanded >= node_budget:
                break
            nodes_expanded += 1
            candidate_rows, _ = bounded_residual_candidates_memory_efficient(
                residual, 128, max_actions
            )
            for candidate_row in candidate_rows:
                generated += 1
                action = action_from_payload(candidate_row["action"])
                child = subtract_action(residual, action)
                child_path = path + (action_identity(action),)
                if child == zero:
                    exact_solutions += depth + 1 == rank
                    continue
                by_residual.setdefault(child, child_path)
        ranked = sorted(
            by_residual.items(),
            key=lambda item: (squared_norm(item[0]), item[0], item[1]),
        )
        remaining = rank - depth - 1
        if ranked:
            indices = sorted({
                round(index * (len(ranked) - 1) / max(1, samples_per_depth - 1))
                for index in range(samples_per_depth)
            })
            for sample_index, ranked_index in enumerate(indices):
                residual, path = ranked[ranked_index]
                record = feasibility_record(residual, remaining, known_completion=None)
                record.update({
                    "source": "new_frozen_baseline_rank7_sample",
                    "depth": depth + 1,
                    "path_action_ids": list(path),
                    "ranked_state_index": ranked_index,
                    "ranked_state_count": len(ranked),
                    "selection_quantile_index": sample_index,
                    "residual": tensor_payload(residual),
                })
                samples.append(record)
        depth_counts.append({
            "depth": depth + 1,
            "remaining_budget": remaining,
            "generated_unique_nonzero_states": len(ranked),
            "sampled_states": len(indices) if ranked else 0,
            "norm_histogram": dict(sorted(Counter(squared_norm(item[0]) for item in ranked).items())),
        })
        if depth + 1 == rank or nodes_expanded >= node_budget:
            break
        frontier = [(residual, path) for residual, path in ranked[:beam_width]]
        if not frontier:
            break
    return samples, {
        "rank": rank,
        "node_budget": node_budget,
        "beam_width": beam_width,
        "max_actions": max_actions,
        "samples_per_depth": samples_per_depth,
        "nodes_expanded": nodes_expanded,
        "candidate_edges_generated": generated,
        "exact_solutions": exact_solutions,
        "depths": depth_counts,
    }
