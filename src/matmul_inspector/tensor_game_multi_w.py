"""Optional bounded multi-output-factor proposals for the 2x2 Tensor Game."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import itertools
import time
from typing import Any, Literal, Sequence

from .tensor_game import RankOneAction, Tensor
from .tensor_game_bounded_residual import (
    COEFFICIENTS,
    action_identity,
    coarse_pair_scores_streaming,
    exact_score_from_contraction,
    squared_norm,
)
from .tensor_game_hardware_cost import action_cost
from .tensor_game_search import canonicalize_action, quantize_ratio

Strategy = Literal["residual", "diversity"]
W_VECTORS = tuple(vector for vector in itertools.product(COEFFICIENTS, repeat=4) if any(vector))


def _sign_pattern(w: Sequence[int]) -> tuple[int, ...]:
    return tuple(0 if value == 0 else (1 if value > 0 else -1) for value in w)


def _hamming(left: Sequence[int], right: Sequence[int]) -> int:
    return sum(a != b for a, b in zip(left, right))


def alternative_w_vectors(
    contraction: Sequence[int],
    q: int,
    count: int,
    strategy: Strategy,
) -> tuple[dict[str, Any], ...]:
    """Return the original quantized w plus bounded deterministic alternatives."""
    if len(contraction) != 4 or q <= 0:
        raise ValueError("multi-w currently requires a four-coordinate contraction and positive q")
    if count <= 0 or count > len(W_VECTORS):
        raise ValueError("w candidate count must be in [1, 80]")
    if strategy not in ("residual", "diversity"):
        raise ValueError("strategy must be residual or diversity")
    original = tuple(quantize_ratio(value, q, COEFFICIENTS) for value in contraction)
    scored = [
        {
            "w": w,
            "score_delta": exact_score_from_contraction(0, contraction, q, w),
            "support": sum(value != 0 for value in w),
            "sign_pattern": _sign_pattern(w),
        }
        for w in W_VECTORS
    ]
    scored.sort(key=lambda row: (row["score_delta"], row["w"]))
    selected: list[dict[str, Any]] = []
    if any(original):
        selected.append(next(row for row in scored if row["w"] == original) | {"source": "original"})
    if strategy == "residual":
        remaining = (row for row in scored if row["w"] != original)
        selected.extend((row | {"source": "alternative_residual"}) for row in itertools.islice(remaining, count - len(selected)))
    else:
        remaining = [row for row in scored if row["w"] != original]
        residual_slots = (count - len(selected) + 1) // 2
        for row in remaining[:residual_slots]:
            selected.append(row | {"source": "alternative_diversity_residual_quota"})
        remaining = [row for row in remaining if row["w"] not in {item["w"] for item in selected}]
        while len(selected) < count and remaining:
            used_supports = {row["support"] for row in selected}
            used_signs = {row["sign_pattern"] for row in selected}
            choice = min(
                remaining,
                key=lambda row: (
                    row["support"] in used_supports,
                    row["sign_pattern"] in used_signs,
                    -min((_hamming(row["w"], item["w"]) for item in selected), default=4),
                    row["score_delta"],
                    row["w"],
                ),
            )
            remaining.remove(choice)
            selected.append(choice | {"source": "alternative_diversity_quota"})
    return tuple(selected)


def multi_w_candidates(
    residual: Tensor,
    *,
    pair_width: int = 128,
    w_candidates_per_pair: int = 4,
    strategy: Strategy = "residual",
    complete_action_budget: int | None = None,
    limit: int = 12,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate and exactly rank bounded multi-w complete actions."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    if complete_action_budget is not None and complete_action_budget <= 0:
        raise ValueError("complete action budget must be positive")
    started = time.perf_counter_ns()
    pairs = coarse_pair_scores_streaming(residual, pair_width)
    pair_ns = time.perf_counter_ns() - started
    residual_norm = squared_norm(residual)
    alternatives_ns = exact_ns = 0
    candidates: dict[RankOneAction, dict[str, Any]] = {}
    scored_w_vectors = 0
    for pair_rank, pair in enumerate(pairs, 1):
        phase = time.perf_counter_ns()
        alternatives = alternative_w_vectors(pair.contraction, pair.q, w_candidates_per_pair, strategy)
        alternatives_ns += time.perf_counter_ns() - phase
        scored_w_vectors += len(W_VECTORS)
        for w_rank, item in enumerate(alternatives, 1):
            phase = time.perf_counter_ns()
            action = canonicalize_action(RankOneAction(pair.u, pair.v, item["w"]))
            score = exact_score_from_contraction(residual_norm, pair.contraction, pair.q, item["w"])
            exact_ns += time.perf_counter_ns() - phase
            row = {
                "action": action,
                "action_id": action_identity(action),
                "score": score,
                "residual_reduction": residual_norm - score,
                "pair_rank": pair_rank,
                "w_rank_within_pair": w_rank,
                "source": item["source"],
                "w_support": item["support"],
                "w_sign_pattern": list(item["sign_pattern"]),
                "preparation_additions": action_cost(action).preparation_additions,
            }
            previous = candidates.get(action)
            if previous is None or (score, pair_rank, w_rank) < (
                previous["score"], previous["pair_rank"], previous["w_rank_within_pair"]
            ):
                candidates[action] = row
    ordered = sorted(
        candidates.values(),
        key=lambda row: (
            row["score"],
            row["action"].u,
            row["action"].v,
            row["action"].w,
            row["source"],
        ),
    )
    before_budget = len(ordered)
    if complete_action_budget is not None:
        ordered = ordered[:complete_action_budget]
    returned = []
    ties = Counter(row["score"] for row in ordered)
    for rank, row in enumerate(ordered[:limit], 1):
        returned.append({
            "action": {"u": list(row["action"].u), "v": list(row["action"].v), "w": list(row["action"].w)},
            "action_id": row["action_id"],
            "score": row["score"],
            "rank": rank,
            "score_tie_count": ties[row["score"]],
            "residual_reduction": row["residual_reduction"],
            "preparation_additions": row["preparation_additions"],
            "w_support": row["w_support"],
            "w_sign_pattern": row["w_sign_pattern"],
            "source": row["source"],
            "pair_rank": row["pair_rank"],
            "w_rank_within_pair": row["w_rank_within_pair"],
        })
    return returned, {
        "provider": "bounded_multi_w",
        "pair_width": pair_width,
        "w_candidates_per_pair": w_candidates_per_pair,
        "strategy": strategy,
        "complete_action_budget": complete_action_budget,
        "w_domain_size": len(W_VECTORS),
        "complete_actions_scored_for_w_ranking": scored_w_vectors,
        "selected_pair_w_instances": pair_width * w_candidates_per_pair,
        "unique_complete_actions_before_budget": before_budget,
        "unique_complete_actions_after_budget": len(ordered),
        "returned_candidates": len(returned),
        "pair_ranking_ms": pair_ns / 1e6,
        "alternative_generation_ms": alternatives_ns / 1e6,
        "selected_exact_scoring_ms": exact_ns / 1e6,
        "total_ms": (time.perf_counter_ns() - started) / 1e6,
    }


@dataclass
class MultiWProvider:
    name: str = field(default="bounded_multi_w", init=False)
    pair_width: int = 128
    w_candidates_per_pair: int = 4
    strategy: Strategy = "residual"
    complete_action_budget: int | None = 512
    candidate_pool: int = 128
    seed: int = 0
    trace_limit: int = 8
    trace: list[dict[str, Any]] = field(default_factory=list, init=False)
    calls: int = field(default=0, init=False)
    complete_actions_scored: int = field(default=0, init=False)
    retained_pair_w_instances: int = field(default=0, init=False)
    returned_actions: int = field(default=0, init=False)
    generation_seconds: float = field(default=0.0, init=False)
    depth: int | None = field(default=None, init=False)
    remaining_moves: int | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.candidate_pool <= 0 or self.trace_limit < 0:
            raise ValueError("candidate pool must be positive and trace limit nonnegative")

    def set_context(self, depth: int, remaining_moves: int) -> None:
        self.depth = depth
        self.remaining_moves = remaining_moves

    def __call__(self, residual: Tensor, limit: int) -> tuple[RankOneAction, ...]:
        started = time.perf_counter()
        rows, audit = multi_w_candidates(
            residual,
            pair_width=self.pair_width,
            w_candidates_per_pair=self.w_candidates_per_pair,
            strategy=self.strategy,
            complete_action_budget=self.complete_action_budget,
            limit=max(limit, self.candidate_pool),
        )
        selected = rows[:limit]
        self.calls += 1
        self.complete_actions_scored += audit["complete_actions_scored_for_w_ranking"]
        self.retained_pair_w_instances += audit["selected_pair_w_instances"]
        self.returned_actions += len(selected)
        self.generation_seconds += time.perf_counter() - started
        if len(self.trace) < self.trace_limit:
            self.trace.append({
                "depth": self.depth,
                "remaining_moves": self.remaining_moves,
                "audit": audit,
                "selected": selected,
            })
        return tuple(RankOneAction(row["action"]["u"], row["action"]["v"], row["action"]["w"]) for row in selected)

    def diagnostic_summary(self) -> dict[str, Any]:
        return {
            "provider_calls": self.calls,
            "complete_actions_scored": self.complete_actions_scored,
            "retained_pair_w_instances": self.retained_pair_w_instances,
            "returned_actions": self.returned_actions,
            "mean_branching_factor": self.returned_actions / self.calls if self.calls else 0.0,
            "generation_seconds": self.generation_seconds,
        }


def create_multi_w_candidate_provider(**settings: Any) -> MultiWProvider:
    """Create the explicitly selected experimental provider."""
    return MultiWProvider(**settings)
