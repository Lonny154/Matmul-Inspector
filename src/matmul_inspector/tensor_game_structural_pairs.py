"""Optional trajectory-agnostic structural diversity for Tensor Game pair proposals."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import time
from typing import Any, Literal, Sequence

from .tensor_game import RankOneAction, Tensor
from .tensor_game_bounded_residual import (
    COEFFICIENTS,
    PairScore,
    action_identity,
    coarse_pair_scores,
    exact_score_from_contraction,
    squared_norm,
)
from .tensor_game_hardware_cost import action_cost
from .tensor_game_quota_w import top_k_w_vectors
from .tensor_game_search import canonicalize_action, quantize_ratio

WMode = Literal["deterministic", "top_k"]


def _normalized_sign_pattern(vector: Sequence[int]) -> tuple[int, ...]:
    first = next(value for value in vector if value)
    sign = 1 if first > 0 else -1
    return tuple(sign * value for value in vector)


def pair_structural_features(pair: PairScore) -> dict[str, Any]:
    """Return deterministic features invariant to independent global u/v signs."""
    u_positive = sum(value > 0 for value in pair.u)
    u_negative = sum(value < 0 for value in pair.u)
    v_positive = sum(value > 0 for value in pair.v)
    v_negative = sum(value < 0 for value in pair.v)
    u_support = u_positive + u_negative
    v_support = v_positive + v_negative
    contraction_abs = tuple(sorted((abs(value) for value in pair.contraction), reverse=True))
    return {
        "u_support": u_support,
        "v_support": v_support,
        "combined_support": u_support + v_support,
        "u_sign_counts_unordered": tuple(sorted((u_positive, u_negative))),
        "v_sign_counts_unordered": tuple(sorted((v_positive, v_negative))),
        "u_sign_pattern_normalized": _normalized_sign_pattern(pair.u),
        "v_sign_pattern_normalized": _normalized_sign_pattern(pair.v),
        "preparation_additions": max(u_support - 1, 0) + max(v_support - 1, 0),
        "contraction_support": sum(value != 0 for value in pair.contraction),
        "contraction_abs_profile": contraction_abs,
        "contraction_l1": sum(contraction_abs),
        "contraction_max_abs": max(contraction_abs, default=0),
        "heuristic_score_numerator": pair.bound_numerator,
    }


def structural_signature(features: dict[str, Any]) -> tuple[Any, ...]:
    return (
        features["u_support"], features["v_support"],
        features["u_sign_counts_unordered"], features["v_sign_counts_unordered"],
        features["u_sign_pattern_normalized"], features["v_sign_pattern_normalized"],
        features["preparation_additions"], features["contraction_support"],
        features["contraction_abs_profile"],
    )


def _distance(left: tuple[Any, ...], right: tuple[Any, ...]) -> int:
    return sum(a != b for a, b in zip(left, right))


def select_structural_pairs(
    residual: Tensor,
    *,
    protected_residual_pairs: int,
    structural_pairs: int,
    total_pairs: int = 128,
) -> tuple[tuple[PairScore, ...], dict[str, Any]]:
    """Protect a residual prefix and greedily cover distant structural signatures."""
    if protected_residual_pairs < 0 or structural_pairs < 0:
        raise ValueError("pair quotas must be nonnegative")
    if protected_residual_pairs + structural_pairs != total_pairs or total_pairs <= 0 or total_pairs > 6400:
        raise ValueError("pair quotas must sum to total_pairs in [1, 6400]")
    started = time.perf_counter_ns()
    ranked = coarse_pair_scores(residual)
    ranking_ns = time.perf_counter_ns() - started
    if structural_pairs == 0:
        selected = ranked[:total_pairs]
        return tuple(selected), {
            "protected_residual_pairs": total_pairs, "structural_pairs": 0,
            "fallback_pairs": 0, "distinct_structural_signatures": len({
                structural_signature(pair_structural_features(pair)) for pair in selected
            }),
            "selected_original_ranks": list(range(1, total_pairs + 1)),
            "ranking_ms": ranking_ns / 1e6, "selection_ms": 0.0,
            "total_ms": (time.perf_counter_ns() - started) / 1e6,
        }

    phase = time.perf_counter_ns()
    rank_by_pair = {(pair.u, pair.v): rank for rank, pair in enumerate(ranked, 1)}
    protected = list(ranked[:protected_residual_pairs])
    selected = list(protected)
    selected_keys = {(pair.u, pair.v) for pair in selected}
    selected_signatures = [structural_signature(pair_structural_features(pair)) for pair in selected]
    representatives = {}
    for pair in ranked[protected_residual_pairs:]:
        features = pair_structural_features(pair)
        signature = structural_signature(features)
        representatives.setdefault(signature, (pair, features))
    distances = {
        signature: min((_distance(signature, existing) for existing in selected_signatures), default=len(signature))
        for signature in representatives
    }
    while representatives and len(selected) < total_pairs:
        signature, (pair, _) = max(
            representatives.items(),
            key=lambda item: (distances[item[0]], -rank_by_pair[(item[1][0].u, item[1][0].v)]),
        )
        del representatives[signature]
        del distances[signature]
        if (pair.u, pair.v) in selected_keys:
            continue
        selected.append(pair)
        selected_keys.add((pair.u, pair.v))
        selected_signatures.append(signature)
        for remaining in distances:
            distances[remaining] = min(distances[remaining], _distance(remaining, signature))
    fallback = 0
    for pair in ranked:
        if len(selected) >= total_pairs:
            break
        if (pair.u, pair.v) not in selected_keys:
            selected.append(pair); selected_keys.add((pair.u, pair.v)); fallback += 1
    selection_ns = time.perf_counter_ns() - phase
    return tuple(selected), {
        "protected_residual_pairs": protected_residual_pairs,
        "structural_pairs": len(selected) - protected_residual_pairs - fallback,
        "fallback_pairs": fallback,
        "distinct_structural_signatures": len(set(selected_signatures)),
        "selected_original_ranks": [rank_by_pair[(pair.u, pair.v)] for pair in selected],
        "ranking_ms": ranking_ns / 1e6,
        "selection_ms": selection_ns / 1e6,
        "total_ms": (time.perf_counter_ns() - started) / 1e6,
    }


def structural_pair_candidates(
    residual: Tensor,
    *,
    protected_residual_pairs: int,
    structural_pairs: int,
    w_mode: WMode = "top_k",
    w_candidates_per_pair: int = 2,
    limit: int = 12,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate exactly scored actions from structurally selected pairs."""
    if w_mode not in ("deterministic", "top_k") or w_candidates_per_pair <= 0 or limit <= 0:
        raise ValueError("invalid w mode or candidate budget")
    pairs, pair_audit = select_structural_pairs(
        residual, protected_residual_pairs=protected_residual_pairs,
        structural_pairs=structural_pairs,
    )
    residual_norm = squared_norm(residual)
    candidates = {}
    complete_scored = 0
    for selected_rank, pair in enumerate(pairs, 1):
        if w_mode == "deterministic":
            vectors = [(tuple(quantize_ratio(value, pair.q, COEFFICIENTS) for value in pair.contraction), 1)]
        else:
            rows, audit = top_k_w_vectors(pair.contraction, pair.q, w_candidates_per_pair)
            complete_scored += audit["complete_vectors_scored"]
            vectors = [(row["w"], row["rank"]) for row in rows]
        for w, w_rank in vectors:
            if not any(w):
                continue
            action = canonicalize_action(RankOneAction(pair.u, pair.v, w))
            score = exact_score_from_contraction(residual_norm, pair.contraction, pair.q, w)
            features = pair_structural_features(pair)
            row = {
                "action": action, "action_id": action_identity(action), "score": score,
                "residual_reduction": residual_norm - score,
                "selected_pair_rank": selected_rank,
                "original_pair_rank": pair_audit["selected_original_ranks"][selected_rank - 1],
                "pair_source": "protected_residual" if selected_rank <= protected_residual_pairs else "structural",
                "w_rank_within_pair": w_rank,
                "features": features,
                "preparation_additions": action_cost(action).preparation_additions,
            }
            previous = candidates.get(action)
            if previous is None or (score, row["original_pair_rank"], w_rank) < (
                previous["score"], previous["original_pair_rank"], previous["w_rank_within_pair"]
            ):
                candidates[action] = row
    ordered = sorted(candidates.values(), key=lambda row: (
        row["score"], row["action"].u, row["action"].v, row["action"].w
    ))
    ties = Counter(row["score"] for row in ordered)
    returned = []
    for rank, row in enumerate(ordered[:limit], 1):
        returned.append({**row, "rank": rank, "score_tie_count": ties[row["score"]]})
    return returned, {
        "provider": "structural_pairs", "pair_selection": pair_audit,
        "w_mode": w_mode, "w_candidates_per_pair": w_candidates_per_pair,
        "complete_vectors_scored": complete_scored if w_mode == "top_k" else len(pairs),
        "unique_complete_actions": len(ordered), "returned_candidates": len(returned),
    }


@dataclass
class StructuralPairProvider:
    protected_residual_pairs: int = 112
    structural_pairs: int = 16
    w_mode: WMode = "deterministic"
    w_candidates_per_pair: int = 2
    trace_limit: int = 2
    name: str = field(default="structural_pairs", init=False)
    calls: int = field(default=0, init=False)
    complete_actions_scored: int = field(default=0, init=False)
    returned_actions: int = field(default=0, init=False)
    generation_seconds: float = field(default=0.0, init=False)
    trace: list[dict[str, Any]] = field(default_factory=list, init=False)
    negative_reduction_actions: int = field(default=0, init=False)
    preparation_histogram: Counter = field(default_factory=Counter, init=False)
    returned_signature_ids: set[str] = field(default_factory=set, init=False)

    def __call__(self, residual: Tensor, limit: int) -> tuple[RankOneAction, ...]:
        started = time.perf_counter()
        rows, audit = structural_pair_candidates(
            residual, protected_residual_pairs=self.protected_residual_pairs,
            structural_pairs=self.structural_pairs, w_mode=self.w_mode,
            w_candidates_per_pair=self.w_candidates_per_pair, limit=limit,
        )
        self.calls += 1; self.complete_actions_scored += audit["complete_vectors_scored"]
        self.returned_actions += len(rows); self.generation_seconds += time.perf_counter() - started
        self.negative_reduction_actions += sum(row["residual_reduction"] < 0 for row in rows)
        self.preparation_histogram.update(row["preparation_additions"] for row in rows)
        self.returned_signature_ids.update(repr(structural_signature(row["features"])) for row in rows)
        if len(self.trace) < self.trace_limit:
            self.trace.append(audit)
        return tuple(row["action"] for row in rows)

    def diagnostic_summary(self) -> dict[str, Any]:
        return {
            "provider_calls": self.calls, "complete_actions_scored": self.complete_actions_scored,
            "returned_actions": self.returned_actions, "generation_seconds": self.generation_seconds,
            "negative_reduction_actions": self.negative_reduction_actions,
            "preparation_additions": dict(sorted(self.preparation_histogram.items())),
            "distinct_returned_pair_signatures": len(self.returned_signature_ids),
        }
