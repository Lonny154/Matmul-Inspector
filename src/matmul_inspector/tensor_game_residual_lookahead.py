"""Exact offline residual-subspace and bounded two-step Tensor Game diagnostics."""

from __future__ import annotations

from dataclasses import asdict
from fractions import Fraction
from typing import Any, Sequence

from .tensor_game import RankOneAction, Tensor, subtract_action
from .tensor_game_bounded_residual import action_identity, squared_norm
from .tensor_game_hardware_cost import action_cost


def exact_matrix_rank(matrix: Sequence[Sequence[int]]) -> int:
    """Compute matrix rank over the rationals with no floating-point arithmetic."""
    if not matrix:
        return 0
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise ValueError("matrix must be rectangular")
    rows = [[Fraction(value) for value in row] for row in matrix]
    rank = column = 0
    while rank < len(rows) and column < width:
        pivot = next((index for index in range(rank, len(rows)) if rows[index][column]), None)
        if pivot is None:
            column += 1
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        pivot_value = rows[rank][column]
        rows[rank] = [value / pivot_value for value in rows[rank]]
        for index in range(len(rows)):
            if index == rank or not rows[index][column]:
                continue
            factor = rows[index][column]
            rows[index] = [value - factor * normalized for value, normalized in zip(rows[index], rows[rank])]
        rank += 1
        column += 1
    return rank


def tensor_flattening(tensor: Tensor, mode: int) -> tuple[tuple[int, ...], ...]:
    """Return the exact mode-n matrix flattening of a third-order tensor."""
    if mode not in (0, 1, 2):
        raise ValueError("mode must be 0, 1, or 2")
    dimensions = (len(tensor), len(tensor[0]), len(tensor[0][0]))
    rows = []
    for primary in range(dimensions[mode]):
        row = []
        for first in range(dimensions[(mode + 1) % 3]):
            for second in range(dimensions[(mode + 2) % 3]):
                indices = [0, 0, 0]
                indices[mode] = primary
                indices[(mode + 1) % 3] = first
                indices[(mode + 2) % 3] = second
                row.append(tensor[indices[0]][indices[1]][indices[2]])
        rows.append(tuple(row))
    return tuple(rows)


def residual_subspace_features(tensor: Tensor) -> dict[str, Any]:
    """Describe exact residual sparsity and matrix flattening/slice subspaces."""
    values = [value for plane in tensor for row in plane for value in row]
    flattenings = [tensor_flattening(tensor, mode) for mode in range(3)]
    slice_ranks = []
    for mode in range(3):
        mode_ranks = []
        dimensions = (len(tensor), len(tensor[0]), len(tensor[0][0]))
        for primary in range(dimensions[mode]):
            matrix = []
            for first in range(dimensions[(mode + 1) % 3]):
                row = []
                for second in range(dimensions[(mode + 2) % 3]):
                    indices = [0, 0, 0]
                    indices[mode] = primary
                    indices[(mode + 1) % 3] = first
                    indices[(mode + 2) % 3] = second
                    row.append(tensor[indices[0]][indices[1]][indices[2]])
                matrix.append(row)
            mode_ranks.append(exact_matrix_rank(matrix))
        slice_ranks.append(mode_ranks)
    return {
        "squared_norm": sum(value * value for value in values),
        "nonzero_entries": sum(value != 0 for value in values),
        "coefficient_l1": sum(abs(value) for value in values),
        "max_abs_coefficient": max((abs(value) for value in values), default=0),
        "negative_entries": sum(value < 0 for value in values),
        "flattening_ranks": [exact_matrix_rank(matrix) for matrix in flattenings],
        "flattening_nonzero_rows": [sum(any(row) for row in matrix) for matrix in flattenings],
        "flattening_nonzero_columns": [
            sum(any(matrix[row][column] for row in range(len(matrix))) for column in range(len(matrix[0])))
            for matrix in flattenings
        ],
        "slice_ranks": slice_ranks,
    }


def action_descriptors(action: RankOneAction) -> dict[str, Any]:
    """Return coefficient and support descriptors without changing the action."""
    vectors = (action.u, action.v, action.w)
    coefficients = tuple(value for vector in vectors for value in vector)
    return {
        "vectors": {name: list(vector) for name, vector in zip(("u", "v", "w"), vectors)},
        "support_sizes": [sum(value != 0 for value in vector) for vector in vectors],
        "coefficient_l1": sum(abs(value) for value in coefficients),
        "max_abs_coefficient": max(abs(value) for value in coefficients),
        "negative_coefficients": sum(value < 0 for value in coefficients),
        "outer_product_support": sum(u != 0 for u in action.u)
        * sum(v != 0 for v in action.v)
        * sum(w != 0 for w in action.w),
    }


def exact_two_step_score(residual: Tensor, first: RankOneAction, second: RankOneAction) -> dict[str, Any]:
    """Evaluate an ordered pair with exact integer residual arithmetic."""
    first_residual = subtract_action(residual, first)
    second_residual = subtract_action(first_residual, second)
    initial_norm = squared_norm(residual)
    first_norm = squared_norm(first_residual)
    second_norm = squared_norm(second_residual)
    return {
        "first_action_id": action_identity(first),
        "second_action_id": action_identity(second),
        "first_action": action_descriptors(first),
        "second_action": action_descriptors(second),
        "initial_squared_norm": initial_norm,
        "first_squared_norm": first_norm,
        "second_squared_norm": second_norm,
        "immediate_reduction": initial_norm - first_norm,
        "two_step_reduction": initial_norm - second_norm,
        "second_step_improvement": first_norm - second_norm,
        "first_action_cost": asdict(action_cost(first)),
        "second_action_cost": asdict(action_cost(second)),
        "initial_features": residual_subspace_features(residual),
        "first_features": residual_subspace_features(first_residual),
        "second_features": residual_subspace_features(second_residual),
    }


def bounded_two_step_scores(
    residual: Tensor,
    first_actions: Sequence[RankOneAction],
    continuations: dict[str, Sequence[RankOneAction]],
    *,
    first_action_limit: int,
    continuations_per_action: int,
    total_evaluation_budget: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Rank first actions by their best bounded continuation under an exact budget."""
    if first_action_limit <= 0 or continuations_per_action <= 0 or total_evaluation_budget <= 0:
        raise ValueError("lookahead budgets must be positive")
    evaluated = []
    count = 0
    for first_rank, first in enumerate(first_actions[:first_action_limit], 1):
        best = None
        evaluated_for_first = 0
        candidates = continuations.get(action_identity(first), ())[:continuations_per_action]
        for second_rank, second in enumerate(candidates, 1):
            if count >= total_evaluation_budget:
                break
            score = exact_two_step_score(residual, first, second)
            count += 1
            evaluated_for_first += 1
            key = (score["second_squared_norm"], action_identity(second))
            if best is None or key < best[0]:
                best = (key, second_rank, score)
        immediate_norm = squared_norm(subtract_action(residual, first))
        evaluated.append({
            "first_action_id": action_identity(first), "first_action_rank": first_rank,
            "immediate_squared_norm": immediate_norm,
            "best_second_rank": best[1] if best else None,
            "best_two_step_squared_norm": best[2]["second_squared_norm"] if best else None,
            "best_two_step_reduction": best[2]["two_step_reduction"] if best else None,
            "continuations_evaluated": evaluated_for_first,
        })
        if count >= total_evaluation_budget:
            break
    ranked = sorted(evaluated, key=lambda row: (
        row["best_two_step_squared_norm"] if row["best_two_step_squared_norm"] is not None else 10**18,
        row["immediate_squared_norm"], row["first_action_id"],
    ))
    for rank, row in enumerate(ranked, 1):
        row["two_step_rank"] = rank
    return ranked, {
        "first_actions_considered": len(evaluated),
        "continuation_evaluations": count,
        "total_evaluation_budget": total_evaluation_budget,
    }
