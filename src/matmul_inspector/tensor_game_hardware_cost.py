"""Static algebraic cost descriptors for Tensor Game actions.

These counts describe a simple scalar evaluation model. They are not register,
memory-transaction, instruction, latency, or device-performance measurements.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Sequence

from .tensor_game import RankOneAction, Tensor, sum_actions, tensor_shape


@dataclass(frozen=True)
class LinearCombinationCost:
    length: int
    nonzeros: int
    zeros: int
    additions: int
    negative_coefficients: int
    unit_coefficients: int
    nonunit_coefficients: int
    coefficient_l1: int
    max_abs_coefficient: int
    magnitude_histogram: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class ActionCost:
    u: LinearCombinationCost
    v: LinearCombinationCost
    w: LinearCombinationCost
    preparation_additions: int
    output_accumulation_additions: int
    coefficient_scalings: int
    temporary_scalars: int


@dataclass(frozen=True)
class DecompositionCost:
    rank_multiplications: int
    pre_additions: int
    post_additions: int
    coefficient_scalings: int
    peak_temporary_scalars: int
    shared_input_combinations: int


@dataclass(frozen=True)
class CandidateObjectives:
    candidate_index: int
    residual_reduction: int
    preparation_additions: int


def linear_combination_cost(coefficients: Sequence[int]) -> LinearCombinationCost:
    """Describe one unshared linear combination exactly from its coefficients."""
    nonzero = [int(value) for value in coefficients if value]
    magnitudes = Counter(abs(value) for value in nonzero)
    return LinearCombinationCost(
        length=len(coefficients),
        nonzeros=len(nonzero),
        zeros=len(coefficients) - len(nonzero),
        additions=max(len(nonzero) - 1, 0),
        negative_coefficients=sum(value < 0 for value in nonzero),
        unit_coefficients=sum(abs(value) == 1 for value in nonzero),
        nonunit_coefficients=sum(abs(value) > 1 for value in nonzero),
        coefficient_l1=sum(abs(value) for value in nonzero),
        max_abs_coefficient=max((abs(value) for value in nonzero), default=0),
        magnitude_histogram=tuple(sorted(magnitudes.items())),
    )


def action_cost(action: RankOneAction) -> ActionCost:
    """Describe one rank-one action under an unshared scalar schedule.

    The schedule forms u and v, multiplies them, then adds the scaled product to
    each nonzero w output. Two input-combination scalars and one product scalar
    are live at the multiplication point, hence three temporary scalars.
    """
    u = linear_combination_cost(action.u)
    v = linear_combination_cost(action.v)
    w = linear_combination_cost(action.w)
    return ActionCost(
        u=u,
        v=v,
        w=w,
        preparation_additions=u.additions + v.additions,
        output_accumulation_additions=w.nonzeros,
        coefficient_scalings=u.nonunit_coefficients + v.nonunit_coefficients + w.nonunit_coefficients,
        temporary_scalars=3,
    )


def candidate_objectives(actions_and_scores: Sequence[tuple[RankOneAction, int]], residual_norm: int) -> tuple[CandidateObjectives, ...]:
    """Build deterministic maximize-reduction/minimize-preparation objectives."""
    return tuple(
        CandidateObjectives(index, residual_norm - score, action_cost(action).preparation_additions)
        for index, (action, score) in enumerate(actions_and_scores)
    )


def pareto_frontier(candidates: Sequence[CandidateObjectives]) -> tuple[CandidateObjectives, ...]:
    """Return nondominated candidates in input order; equal points all survive."""
    return tuple(
        candidate
        for candidate in candidates
        if not any(
            other.residual_reduction >= candidate.residual_reduction
            and other.preparation_additions <= candidate.preparation_additions
            and (
                other.residual_reduction > candidate.residual_reduction
                or other.preparation_additions < candidate.preparation_additions
            )
            for other in candidates
        )
    )


def decomposition_cost(actions: Sequence[RankOneAction], target: Tensor | None = None) -> DecompositionCost:
    """Aggregate a complete decomposition using a documented naive schedule.

    Each distinct u and v combination is formed once and may be reused. Products
    are evaluated sequentially. Each output coordinate accumulates all of its
    contributing product terms, so its post-add count is max(terms - 1, 0).
    When ``target`` is supplied, exact integer reconstruction is required.
    """
    if not actions:
        raise ValueError("decomposition must contain at least one action")
    shape = tuple(map(len, (actions[0].u, actions[0].v, actions[0].w)))
    if any(tuple(map(len, (a.u, a.v, a.w))) != shape for a in actions):
        raise ValueError("all actions must have the same shape")
    if target is not None:
        if tensor_shape(target) != shape or sum_actions(actions, shape) != target:
            raise ValueError("actions do not exactly reconstruct the target")
    unique_u = tuple(dict.fromkeys(action.u for action in actions))
    unique_v = tuple(dict.fromkeys(action.v for action in actions))
    output_terms = [sum(action.w[index] != 0 for action in actions) for index in range(shape[2])]
    all_costs = tuple(action_cost(action) for action in actions)
    return DecompositionCost(
        rank_multiplications=len(actions),
        pre_additions=sum(linear_combination_cost(vector).additions for vector in unique_u + unique_v),
        post_additions=sum(max(terms - 1, 0) for terms in output_terms),
        coefficient_scalings=sum(cost.coefficient_scalings for cost in all_costs),
        peak_temporary_scalars=3 + shape[2],
        shared_input_combinations=(2 * len(actions)) - len(unique_u) - len(unique_v),
    )


def as_json(value: object) -> dict[str, object]:
    """Convert a descriptor dataclass to a JSON-ready mapping."""
    return asdict(value)
