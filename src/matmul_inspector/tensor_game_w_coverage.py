"""Offline Strassen action-coverage diagnostics for the 2x2 Tensor Game."""

from __future__ import annotations

from dataclasses import asdict
import itertools
from typing import Any, Sequence

from .tensor_game import RankOneAction, Tensor, outer_product, subtract_action
from .tensor_game_bounded_residual import (
    COEFFICIENTS,
    VECTORS,
    action_identity,
    coarse_pair_scores,
    exact_score_from_contraction,
    squared_norm,
)
from .tensor_game_hardware_cost import action_cost
from .tensor_game_quota_w import exhaustive_w_vectors, top_k_w_vectors
from .tensor_game_search import canonicalize_action, quantize_ratio


def sign_equivalent_representations(action: RankOneAction) -> tuple[RankOneAction, ...]:
    """Return the four coefficient-domain factorizations of one rank-one tensor."""
    factors = ((1, 1, 1), (-1, -1, 1), (-1, 1, -1), (1, -1, -1))
    return tuple(RankOneAction(
        (su * value for value in action.u),
        (sv * value for value in action.v),
        (sw * value for value in action.w),
    ) for su, sv, sw in factors)


def _payload(action: RankOneAction) -> dict[str, list[int]]:
    return {"u": list(action.u), "v": list(action.v), "w": list(action.w)}


def analyze_action_at_state(
    residual: Tensor,
    action: RankOneAction,
    *,
    pair_width: int = 128,
    w_budget: int = 2,
) -> dict[str, Any]:
    """Measure pair and exact-w coverage for an action without changing a provider."""
    pairs = coarse_pair_scores(residual)
    pair_by_factors = {(pair.u, pair.v): (rank, pair) for rank, pair in enumerate(pairs, 1)}
    representations = []
    for represented in sign_equivalent_representations(action):
        pair_rank, pair = pair_by_factors[(represented.u, represented.v)]
        exhaustive = exhaustive_w_vectors(pair.contraction, pair.q)
        top_k, _ = top_k_w_vectors(pair.contraction, pair.q, len(exhaustive))
        if top_k != exhaustive:
            raise AssertionError("top-k and exhaustive w rankings disagree")
        required_rank = next(index for index, row in enumerate(exhaustive, 1) if row["w"] == represented.w)
        required_score = exhaustive[required_rank - 1]["score_delta"]
        original = tuple(quantize_ratio(value, pair.q, COEFFICIENTS) for value in pair.contraction)
        representations.append({
            "action": _payload(represented),
            "pair_rank": pair_rank,
            "pair_retained": pair_rank <= pair_width,
            "heuristic_score_numerator": pair.bound_numerator,
            "q": pair.q,
            "contraction": list(pair.contraction),
            "required_w_rank": required_rank,
            "alternatives_ahead": required_rank - 1,
            "required_w_score_delta": required_score,
            "required_w_tie_count": sum(row["score_delta"] == required_score for row in exhaustive),
            "minimum_top_k": required_rank,
            "required_w_support": sum(value != 0 for value in represented.w),
            "deterministic_w": list(original),
            "deterministic_w_is_required": original == represented.w,
            "deterministic_w_rank": next((index for index, row in enumerate(exhaustive, 1) if row["w"] == original), None),
            "required_w_generated": pair_rank <= pair_width and required_rank <= w_budget,
        })
    retained = [row for row in representations if row["pair_retained"]]
    best_retained = min(retained, key=lambda row: (row["required_w_rank"], row["pair_rank"])) if retained else None
    residual_norm = squared_norm(residual)
    next_residual = subtract_action(residual, action)
    cost = action_cost(canonicalize_action(action))
    if not any(row["pair_rank"] <= pair_width for row in representations):
        exclusion = "pair_proposal_selection"
    elif not any(row["required_w_generated"] for row in representations):
        exclusion = "alternative_w_generation"
    else:
        exclusion = None
    return {
        "action_id": action_identity(action),
        "action": _payload(action),
        "coefficient_domain_representable": all(value in COEFFICIENTS for value in action.u + action.v + action.w),
        "rank_one_reconstruction": outer_product(action) == tuple(
            tuple(tuple(before - after for before, after in zip(before_row, after_row))
                  for before_row, after_row in zip(before_plane, after_plane))
            for before_plane, after_plane in zip(residual, next_residual)
        ),
        "uv_pair_in_domain": action.u in VECTORS and action.v in VECTORS,
        "best_pair_rank": min(row["pair_rank"] for row in representations),
        "best_required_w_rank": min(row["required_w_rank"] for row in representations),
        "best_retained_required_w_rank": best_retained["required_w_rank"] if best_retained else None,
        "residual_norm_before": residual_norm,
        "residual_norm_after": squared_norm(next_residual),
        "immediate_residual_reduction": residual_norm - squared_norm(next_residual),
        "cost": asdict(cost),
        "representations": sorted(representations, key=lambda row: row["pair_rank"]),
        "earliest_exclusion_stage": exclusion,
    }


def exact_complete_action_ranks(residual: Tensor, actions: Sequence[RankOneAction]) -> dict[str, Any]:
    """Rank requested actions in the full canonical 80^3 ternary action domain."""
    residual_norm = squared_norm(residual)
    unique = set()
    for u, v, w in itertools.product(VECTORS, repeat=3):
        unique.add(canonicalize_action(RankOneAction(u, v, w)))
    scored = [(squared_norm(subtract_action(residual, candidate)), candidate) for candidate in unique]
    ranked = sorted(scored, key=lambda item: (item[0], item[1].u, item[1].v, item[1].w))
    rank_by_id = {action_identity(action): rank for rank, (_, action) in enumerate(ranked, 1)}
    score_ties = {}
    for action in actions:
        score = squared_norm(subtract_action(residual, action))
        score_ties[action_identity(action)] = sum(
            candidate_score == score for candidate_score, _ in ranked
        )
    return {
        "canonical_action_count": len(ranked),
        "ranks": {action_identity(action): rank_by_id[action_identity(action)] for action in actions},
        "score_tie_counts": score_ties,
        "residual_norm": residual_norm,
    }


def generated_complete_action_rank(
    residual: Tensor,
    action: RankOneAction,
    *,
    pair_width: int,
    w_budget: int,
) -> dict[str, int | bool | None]:
    """Rank an action after bounded pair and exact top-k-w generation."""
    residual_norm = squared_norm(residual)
    candidates: dict[RankOneAction, int] = {}
    for pair in coarse_pair_scores(residual)[:pair_width]:
        rows, _ = top_k_w_vectors(pair.contraction, pair.q, w_budget)
        for row in rows:
            candidate = canonicalize_action(RankOneAction(pair.u, pair.v, row["w"]))
            score = exact_score_from_contraction(residual_norm, pair.contraction, pair.q, row["w"])
            candidates[candidate] = min(score, candidates.get(candidate, score))
    ranked = sorted(candidates.items(), key=lambda item: (item[1], item[0].u, item[0].v, item[0].w))
    wanted = action_identity(action)
    rank = next((index for index, (candidate, _) in enumerate(ranked, 1) if action_identity(candidate) == wanted), None)
    return {
        "generated": rank is not None,
        "complete_action_rank": rank,
        "unique_generated_actions": len(ranked),
    }
