"""Quota-preserving multi-w proposals with exact separable top-k generation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import heapq
import time
from typing import Any, Literal, Sequence

from .tensor_game import RankOneAction, Tensor
from .tensor_game_bounded_residual import (
    COEFFICIENTS,
    action_from_payload,
    action_identity,
    bounded_residual_candidates_memory_efficient,
    coarse_pair_scores_streaming,
    exact_score_from_contraction,
    squared_norm,
)
from .tensor_game_hardware_cost import action_cost
from .tensor_game_search import canonicalize_action, quantize_ratio

Strategy = Literal["residual", "diversity"]


def top_k_w_vectors(
    contraction: Sequence[int], q: int, k: int, *, complete_score_budget: int | None = None
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    """Return exact k-best nonzero ternary w vectors without full enumeration."""
    if not contraction or q <= 0:
        raise ValueError("contraction must be nonempty and q must be positive")
    valid_count = len(COEFFICIENTS) ** len(contraction) - 1
    if k <= 0:
        raise ValueError("k must be positive")
    if complete_score_budget is not None and complete_score_budget < 0:
        raise ValueError("complete score budget must be nonnegative")
    requested = min(k, valid_count)
    started = time.perf_counter_ns()
    choices = []
    coordinate_evaluations = 0
    for value in contraction:
        options = []
        for coefficient in COEFFICIENTS:
            coordinate_evaluations += 1
            contribution = q * coefficient * coefficient - 2 * value * coefficient
            options.append((contribution, coefficient))
        choices.append(tuple(sorted(options)))

    origin = tuple(0 for _ in contraction)

    def make_entry(indices):
        w = tuple(choices[index][choice_index][1] for index, choice_index in enumerate(indices))
        score_delta = sum(choices[index][choice_index][0] for index, choice_index in enumerate(indices))
        return score_delta, w, indices

    heap = [make_entry(origin)]
    seen = {origin}
    rows = []
    heap_pops = heap_pushes = 0
    while heap and len(rows) < requested and (complete_score_budget is None or heap_pops < complete_score_budget):
        score_delta, w, indices = heapq.heappop(heap)
        heap_pops += 1
        if any(w):
            rows.append({"w": w, "score_delta": score_delta, "rank": len(rows) + 1})
        for coordinate in range(len(indices)):
            if indices[coordinate] + 1 >= len(choices[coordinate]):
                continue
            neighbor = list(indices)
            neighbor[coordinate] += 1
            neighbor = tuple(neighbor)
            if neighbor not in seen:
                seen.add(neighbor)
                heapq.heappush(heap, make_entry(neighbor))
                heap_pushes += 1
    return tuple(rows), {
        "coefficient_domain": list(COEFFICIENTS),
        "dimension": len(contraction),
        "valid_vector_count": valid_count,
        "requested_k": k,
        "returned_k": len(rows),
        "coordinate_choice_evaluations": coordinate_evaluations,
        "complete_vectors_scored": heap_pops,
        "complete_score_budget": complete_score_budget,
        "heap_pushes": heap_pushes,
        "generation_ms": (time.perf_counter_ns() - started) / 1e6,
    }


def exhaustive_w_vectors(contraction: Sequence[int], q: int) -> tuple[dict[str, Any], ...]:
    """Exhaustive correctness oracle. Normal providers must not call this."""
    import itertools

    rows = []
    for w in itertools.product(COEFFICIENTS, repeat=len(contraction)):
        if any(w):
            rows.append({"w": w, "score_delta": exact_score_from_contraction(0, contraction, q, w)})
    rows.sort(key=lambda row: (row["score_delta"], row["w"]))
    return tuple(row | {"rank": rank} for rank, row in enumerate(rows, 1))


def _diversity_order(rows: list[dict[str, Any]], slots: int) -> list[dict[str, Any]]:
    if not rows or slots <= 0:
        return []
    best_reduction = max(row["residual_reduction"] for row in rows)
    eligible = [row for row in rows if row["residual_reduction"] >= best_reduction - 1]
    selected = []
    while eligible and len(selected) < slots:
        used_supports = {row["w_support"] for row in selected}
        used_signs = {row["w_sign_pattern"] for row in selected}
        used_reductions = {row["residual_reduction"] for row in selected}
        choice = min(
            eligible,
            key=lambda row: (
                row["w_support"] in used_supports,
                row["w_sign_pattern"] in used_signs,
                row["residual_reduction"] in used_reductions,
                -row["residual_reduction"],
                row["preparation_additions"],
                row["action"].u,
                row["action"].v,
                row["action"].w,
            ),
        )
        eligible.remove(choice)
        selected.append(choice)
    return selected


def _baseline_rows_from_pairs(pairs, residual_norm: int, limit: int) -> list[dict[str, Any]]:
    """Reproduce bounded baseline ranking from an already retained pair prefix."""
    candidates = {}
    for pair in pairs:
        w = tuple(quantize_ratio(value, pair.q, COEFFICIENTS) for value in pair.contraction)
        if not any(w):
            continue
        action = canonicalize_action(RankOneAction(pair.u, pair.v, w))
        score = exact_score_from_contraction(residual_norm, pair.contraction, pair.q, w)
        candidates[action] = min(score, candidates.get(action, score))
    ordered = sorted(candidates.items(), key=lambda item: (item[1], item[0].u, item[0].v, item[0].w))
    ties = Counter(score for _, score in ordered)
    return [{
        "action": {"u": list(action.u), "v": list(action.v), "w": list(action.w)},
        "action_id": action_identity(action),
        "score": score,
        "rank": rank,
        "score_tie_count": ties[score],
    } for rank, (action, score) in enumerate(ordered[:limit], 1)]


def quota_multi_w_candidates(
    residual: Tensor,
    *,
    protected_baseline: int,
    alternative_slots: int,
    pair_width: int = 128,
    w_candidates_per_pair: int = 2,
    strategy: Strategy = "residual",
    complete_action_score_budget: int | None = None,
    total_returned: int = 12,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Protect baseline actions and fill reserved slots with bounded alternatives."""
    if protected_baseline < 0 or alternative_slots < 0 or protected_baseline + alternative_slots != total_returned:
        raise ValueError("protected baseline plus alternative slots must equal total returned")
    if strategy not in ("residual", "diversity"):
        raise ValueError("strategy must be residual or diversity")
    if w_candidates_per_pair <= 0:
        raise ValueError("w candidates per pair must be positive")
    if complete_action_score_budget is not None and complete_action_score_budget < pair_width:
        raise ValueError("complete-action score budget must cover at least one result per pair")
    total_started = time.perf_counter_ns()
    phase = time.perf_counter_ns()
    pairs = None
    residual_norm = squared_norm(residual)
    if alternative_slots:
        pairs = coarse_pair_scores_streaming(residual, pair_width)
        baseline_rows = _baseline_rows_from_pairs(pairs, residual_norm, total_returned)
        baseline_audit = {
            "enumerated_uv_pairs": 6400,
            "fully_evaluated_pairs": pair_width,
            "returned_candidates": len(baseline_rows),
            "selection": "shared_pair_prefix_equivalent_to_bounded_128",
        }
    else:
        baseline_rows, baseline_audit = bounded_residual_candidates_memory_efficient(residual, pair_width, total_returned)
    baseline_ns = time.perf_counter_ns() - phase
    protected = baseline_rows[:protected_baseline]
    baseline_ids = {row["action_id"] for row in baseline_rows}

    alternatives = []
    top_k_ns = exact_ns = dedup_ns = 0
    top_k_complete_scores = 0
    coordinate_evaluations = 0
    if alternative_slots:
        unique = {}
        exhausted_budget = False
        for pair_rank, pair in enumerate(pairs, 1):
            remaining_score_budget = (
                None if complete_action_score_budget is None
                else complete_action_score_budget - top_k_complete_scores
            )
            if remaining_score_budget is not None and remaining_score_budget <= 0:
                exhausted_budget = True
                break
            phase = time.perf_counter_ns()
            w_rows, top_k_audit = top_k_w_vectors(
                pair.contraction,
                pair.q,
                w_candidates_per_pair,
                complete_score_budget=remaining_score_budget,
            )
            top_k_ns += time.perf_counter_ns() - phase
            top_k_complete_scores += top_k_audit["complete_vectors_scored"]
            coordinate_evaluations += top_k_audit["coordinate_choice_evaluations"]
            for w_row in w_rows:
                phase = time.perf_counter_ns()
                action = canonicalize_action(RankOneAction(pair.u, pair.v, w_row["w"]))
                score = exact_score_from_contraction(residual_norm, pair.contraction, pair.q, w_row["w"])
                exact_ns += time.perf_counter_ns() - phase
                action_id = action_identity(action)
                if action_id in baseline_ids:
                    continue
                cost = action_cost(action)
                row = {
                    "action": action,
                    "action_id": action_id,
                    "score": score,
                    "residual_reduction": residual_norm - score,
                    "pair_rank": pair_rank,
                    "w_rank_within_pair": w_row["rank"],
                    "w_support": cost.w.nonzeros,
                    "w_sign_pattern": tuple(0 if value == 0 else (1 if value > 0 else -1) for value in action.w),
                    "preparation_additions": cost.preparation_additions,
                }
                phase = time.perf_counter_ns()
                previous = unique.get(action_id)
                if previous is None or (row["score"], row["pair_rank"], row["w_rank_within_pair"]) < (
                    previous["score"], previous["pair_rank"], previous["w_rank_within_pair"]
                ):
                    unique[action_id] = row
                dedup_ns += time.perf_counter_ns() - phase
            if exhausted_budget:
                break
        alternatives = sorted(
            unique.values(),
            key=lambda row: (row["score"], row["action"].u, row["action"].v, row["action"].w),
        )
    if strategy == "residual":
        selected_alternatives = alternatives[:alternative_slots]
    else:
        selected_alternatives = _diversity_order(alternatives, alternative_slots)

    selected_ids = {row["action_id"] for row in protected} | {row["action_id"] for row in selected_alternatives}
    fallback = [row for row in baseline_rows[protected_baseline:] if row["action_id"] not in selected_ids]
    merged = [(row, "protected_baseline") for row in protected]
    merged += [(row, f"quota_{strategy}_alternative") for row in selected_alternatives]
    merged += [(row, "baseline_fallback") for row in fallback[: total_returned - len(merged)]]
    returned = []
    for rank, (row, source) in enumerate(merged[:total_returned], 1):
        action = row["action"] if isinstance(row["action"], dict) else {
            "u": list(row["action"].u), "v": list(row["action"].v), "w": list(row["action"].w)
        }
        returned.append({
            "action": action,
            "action_id": row["action_id"],
            "score": row["score"],
            "rank": rank,
            "source": source,
            "residual_reduction": row.get("residual_reduction", squared_norm(residual) - row["score"]),
            "preparation_additions": row.get("preparation_additions", action_cost(action_from_payload(action)).preparation_additions),
            "w_support": row.get("w_support", sum(value != 0 for value in action["w"])),
        })
    return returned, {
        "provider": "bounded_quota_multi_w",
        "protected_baseline": protected_baseline,
        "alternative_slots": alternative_slots,
        "pair_width": pair_width,
        "w_candidates_per_pair": w_candidates_per_pair,
        "strategy": strategy,
        "complete_action_score_budget": complete_action_score_budget,
        "total_returned_budget": total_returned,
        "baseline_work": baseline_audit,
        "baseline_generation_ms": baseline_ns / 1e6,
        "top_k_generation_ms": top_k_ns / 1e6,
        "top_k_complete_vectors_scored": top_k_complete_scores,
        "coordinate_choice_evaluations": coordinate_evaluations,
        "selected_exact_scoring_ms": exact_ns / 1e6,
        "deduplication_ms": dedup_ns / 1e6,
        "unique_alternatives": len(alternatives),
        "generated_alternative_action_ids": [row["action_id"] for row in alternatives],
        "selected_alternatives": len(selected_alternatives),
        "excluded_by_alternative_slot_budget": max(0, len(alternatives) - len(selected_alternatives)),
        "fallback_baseline_actions": max(0, len(returned) - len(protected) - len(selected_alternatives)),
        "returned_candidates": len(returned),
        "total_ms": (time.perf_counter_ns() - total_started) / 1e6,
    }


@dataclass
class QuotaMultiWProvider:
    protected_baseline: int = 10
    alternative_slots: int = 2
    pair_width: int = 128
    w_candidates_per_pair: int = 2
    strategy: Strategy = "residual"
    complete_action_score_budget: int | None = None
    total_returned: int = 12
    trace_limit: int = 2
    name: str = field(default="bounded_quota_multi_w", init=False)
    calls: int = field(default=0, init=False)
    complete_actions_scored: int = field(default=0, init=False)
    returned_actions: int = field(default=0, init=False)
    generation_seconds: float = field(default=0.0, init=False)
    trace: list[dict[str, Any]] = field(default_factory=list, init=False)
    depth: int | None = field(default=None, init=False)
    remaining_moves: int | None = field(default=None, init=False)

    def set_context(self, depth: int, remaining_moves: int) -> None:
        self.depth = depth
        self.remaining_moves = remaining_moves

    def __call__(self, residual: Tensor, limit: int) -> tuple[RankOneAction, ...]:
        if limit != self.total_returned:
            raise ValueError(f"quota provider requires limit={self.total_returned}")
        started = time.perf_counter()
        rows, audit = quota_multi_w_candidates(
            residual,
            protected_baseline=self.protected_baseline,
            alternative_slots=self.alternative_slots,
            pair_width=self.pair_width,
            w_candidates_per_pair=self.w_candidates_per_pair,
            strategy=self.strategy,
            complete_action_score_budget=self.complete_action_score_budget,
            total_returned=self.total_returned,
        )
        self.calls += 1
        self.complete_actions_scored += audit["baseline_work"]["fully_evaluated_pairs"] + audit["top_k_complete_vectors_scored"]
        self.returned_actions += len(rows)
        self.generation_seconds += time.perf_counter() - started
        if len(self.trace) < self.trace_limit:
            self.trace.append({"depth": self.depth, "remaining_moves": self.remaining_moves, "audit": audit, "rows": rows})
        return tuple(action_from_payload(row["action"]) for row in rows)

    def diagnostic_summary(self) -> dict[str, Any]:
        return {
            "provider_calls": self.calls,
            "complete_actions_scored": self.complete_actions_scored,
            "returned_actions": self.returned_actions,
            "mean_branching_factor": self.returned_actions / self.calls if self.calls else 0.0,
            "generation_seconds": self.generation_seconds,
        }
