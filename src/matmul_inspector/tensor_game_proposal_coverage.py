"""CPU-only diagnostics for bounded Tensor Game proposal coverage."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import time
from typing import Any, Iterable, Sequence

from .tensor_game import RankOneAction, Tensor, subtract_action
from .tensor_game_bounded_residual import (
    COEFFICIENTS,
    action_identity,
    coarse_pair_scores,
    exact_score_from_contraction,
    squared_norm,
)
from .tensor_game_hardware_cost import action_cost
from .tensor_game_search import canonicalize_action, quantize_ratio

DEFAULT_WIDTHS = (32, 64, 128, 256, 512, 1024, 6400)


def tensor_identity(tensor: Tensor) -> str:
    payload = json.dumps(tensor, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _payload(action: RankOneAction) -> dict[str, list[int]]:
    return {"u": list(action.u), "v": list(action.v), "w": list(action.w)}


def _pareto_ids(rows: Sequence[dict[str, Any]]) -> set[str]:
    return {
        row["action_id"]
        for row in rows
        if not any(
            other["residual_reduction"] >= row["residual_reduction"]
            and other["preparation_additions"] <= row["preparation_additions"]
            and (
                other["residual_reduction"] > row["residual_reduction"]
                or other["preparation_additions"] < row["preparation_additions"]
            )
            for other in rows
        )
    }


def _distribution(rows: Sequence[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[key]) for row in rows).items(), key=lambda item: int(item[0])))


def _summary(rows: Sequence[dict[str, Any]], full_pareto: set[str]) -> dict[str, Any]:
    ids = {row["action_id"] for row in rows}
    pareto = _pareto_ids(rows)
    nonzero = sum(row["preparation_additions"] > 0 for row in rows)
    return {
        "candidate_count": len(rows),
        "nonzero_preparation_count": nonzero,
        "nonzero_preparation_fraction": nonzero / len(rows) if rows else 0.0,
        "distinct_residual_reductions": len({row["residual_reduction"] for row in rows}),
        "distinct_preparation_costs": len({row["preparation_additions"] for row in rows}),
        "pareto_frontier_size": len(pareto),
        "full_pareto_recalled": len(ids & full_pareto),
        "full_pareto_recall": len(ids & full_pareto) / len(full_pareto) if full_pareto else 1.0,
        "residual_reduction_histogram": _distribution(rows, "residual_reduction"),
        "preparation_additions_histogram": _distribution(rows, "preparation_additions"),
        "input_support_histogram": _distribution(rows, "input_support"),
        "output_support_histogram": _distribution(rows, "output_support"),
        "coefficient_magnitude_histogram": _distribution(rows, "max_abs_coefficient"),
    }


def evaluate_state(
    residual: Tensor,
    *,
    widths: Iterable[int] = DEFAULT_WIDTHS,
    expected_action: RankOneAction | None = None,
) -> dict[str, Any]:
    """Evaluate every provider-reachable action and bounded prefixes for one state."""
    started = time.perf_counter_ns()
    residual_norm = squared_norm(residual)
    pairs = coarse_pair_scores(residual)
    pair_records: list[dict[str, Any]] = []
    unique: dict[RankOneAction, dict[str, Any]] = {}
    exact_started = time.perf_counter_ns()
    for pair_rank, pair in enumerate(pairs, 1):
        w = tuple(quantize_ratio(value, pair.q, COEFFICIENTS) for value in pair.contraction)
        record: dict[str, Any] = {
            "pair_rank": pair_rank,
            "u": list(pair.u),
            "v": list(pair.v),
            "q": pair.q,
            "contraction": list(pair.contraction),
            "heuristic_score_numerator": pair.bound_numerator,
            "retained_by_bounded_128": pair_rank <= 128,
        }
        if not any(w):
            record.update({"w": list(w), "status": "zero_w_rejected", "action_id": None})
            pair_records.append(record)
            continue
        action = canonicalize_action(RankOneAction(pair.u, pair.v, w))
        score = exact_score_from_contraction(residual_norm, pair.contraction, pair.q, w)
        cost = action_cost(action)
        action_id = action_identity(action)
        existing = unique.get(action)
        record.update({
            "w": list(w),
            "status": "duplicate_equivalent_action" if existing is not None else "valid",
            "action_id": action_id,
            "exact_score": score,
        })
        pair_records.append(record)
        if existing is None or pair_rank < existing["first_pair_rank"]:
            unique[action] = {
                "action_id": action_id,
                "action": _payload(action),
                "exact_score": score,
                "residual_reduction": residual_norm - score,
                "preparation_additions": cost.preparation_additions,
                "u_support": cost.u.nonzeros,
                "v_support": cost.v.nonzeros,
                "input_support": cost.u.nonzeros + cost.v.nonzeros,
                "output_support": cost.w.nonzeros,
                "total_support": cost.u.nonzeros + cost.v.nonzeros + cost.w.nonzeros,
                "max_abs_coefficient": max(cost.u.max_abs_coefficient, cost.v.max_abs_coefficient, cost.w.max_abs_coefficient),
                "first_pair_rank": pair_rank,
                "heuristic_score_numerator": pair.bound_numerator,
            }
    exact_ns = time.perf_counter_ns() - exact_started
    actions = sorted(unique.values(), key=lambda row: (row["exact_score"], row["action"]["u"], row["action"]["v"], row["action"]["w"]))
    for exact_rank, row in enumerate(actions, 1):
        row["exact_rank"] = exact_rank
        row["retained_by_bounded_128"] = row["first_pair_rank"] <= 128
        row["exclusion_reason"] = None if row["retained_by_bounded_128"] else "heuristic_rank_below_cutoff"
    full_pareto = _pareto_ids(actions)
    width_rows = []
    expected_id = action_identity(expected_action) if expected_action is not None else None
    for width in widths:
        if width <= 0 or width > len(pairs):
            raise ValueError("proposal widths must be in [1, 6400]")
        retained = [row for row in actions if row["first_pair_rank"] <= width]
        summary = _summary(retained, full_pareto)
        summary.update({
            "width": width,
            "exact_scoring_pair_workload": width,
            "known_action_recalled": expected_id in {row["action_id"] for row in retained} if expected_id else None,
        })
        width_rows.append(summary)
    known = None
    if expected_action is not None:
        canonical = canonicalize_action(expected_action)
        matching = unique.get(canonical)
        uv_ranks = [index for index, pair in enumerate(pairs, 1) if pair.u == expected_action.u and pair.v == expected_action.v]
        pair_exists = bool(uv_ranks)
        expected_score = squared_norm(subtract_action(residual, expected_action))
        known = {
            "action_id": expected_id,
            "action": _payload(canonical),
            "complete_action_in_coefficient_domain": all(value in COEFFICIENTS for value in canonical.u + canonical.v + canonical.w),
            "uv_pair_in_domain": pair_exists,
            "complete_action_reachable": matching is not None,
            "retained_by_bounded_128": bool(matching and matching["first_pair_rank"] <= 128),
            "heuristic_rank": min(uv_ranks) if uv_ranks else None,
            "exact_rank": matching["exact_rank"] if matching else None,
            "exact_residual_reduction": residual_norm - expected_score,
            "preparation_additions": action_cost(canonical).preparation_additions,
            "next_residual_identity": tensor_identity(subtract_action(residual, expected_action)),
            "classification": (
                "retained" if matching and matching["first_pair_rank"] <= 128
                else "excluded_by_heuristic" if matching
                else "uv_present_but_incompatible_with_w_rule" if pair_exists
                else "absent_from_uv_domain"
            ),
        }
    bounded = [row for row in actions if row["retained_by_bounded_128"]]
    return {
        "state_identity": tensor_identity(residual),
        "residual_squared_norm": residual_norm,
        "enumerated_uv_pairs": len(pairs),
        "evaluated_complete_actions": len(actions),
        "zero_w_pairs": sum(row["status"] == "zero_w_rejected" for row in pair_records),
        "full_space": _summary(actions, full_pareto),
        "bounded_128": _summary(bounded, full_pareto),
        "counterfactual_widths": width_rows,
        "known_action": known,
        "timing": {
            "exact_and_descriptor_ms": exact_ns / 1e6,
            "total_ms": (time.perf_counter_ns() - started) / 1e6,
        },
        "actions": actions,
        "pairs": pair_records,
    }


def stable_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, separators=(",", ": ")) + "\n"
