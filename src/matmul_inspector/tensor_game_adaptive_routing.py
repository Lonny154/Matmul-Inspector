"""Validation-only candidate routing utilities for TG-4P."""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import mean, median
from typing import Any, Iterable, Sequence

from .tensor_game import RankOneAction
from .tensor_game_experience import action_identity
from .tensor_game_search import canonicalize_action


@dataclass(frozen=True)
class RoutedCandidate:
    action: RankOneAction
    sources: tuple[str, ...]
    frozen_score: float | None = None
    residual_score: float | None = None


def residual_features(residual: Any) -> dict[str, float | int]:
    """Return cheap, construction-metadata-free routing features."""
    values = [int(value) for plane in residual for row in plane for value in row]
    nonzero = sum(value != 0 for value in values)
    return {
        "residual_nonzero": nonzero,
        "residual_sparsity": 1.0 - nonzero / len(values),
        "residual_l1": sum(abs(value) for value in values),
        "residual_l2_squared": sum(value * value for value in values),
        "residual_max_abs": max(map(abs, values), default=0),
    }


def adaptive_decision(features: dict[str, float | int], spec: dict[str, Any]) -> str:
    """Apply a sealed one-feature threshold rule without labels or outcomes."""
    value = float(features[spec["feature"]])
    matched = value >= float(spec["threshold"]) if spec["operator"] == ">=" else value <= float(spec["threshold"])
    return spec["provider_when_true"] if matched else spec["provider_when_false"]


def merge_candidates(
    frozen: Sequence[tuple[RankOneAction, float]],
    residual: Sequence[dict[str, Any]],
    frozen_limit: int,
    residual_limit: int,
    budget: int = 12,
) -> list[RoutedCandidate]:
    """Merge quota prefixes, then alternately fill from unused provider ranks."""
    if min(frozen_limit, residual_limit, budget) < 0:
        raise ValueError("candidate limits must be nonnegative")
    from .tensor_game_experience import action_from_payload

    streams = {
        "frozen": [(canonicalize_action(a), float(s)) for a, s in frozen],
        "residual": [(canonicalize_action(action_from_payload(r["action"])), float(r["score"])) for r in residual],
    }
    selected: list[tuple[str, RankOneAction, float]] = []
    for source, limit in (("frozen", frozen_limit), ("residual", residual_limit)):
        selected.extend((source, action, score) for action, score in streams[source][:limit])
    offsets = {"frozen": frozen_limit, "residual": residual_limit}
    turn = 0
    while len(selected) < budget and any(offsets[s] < len(streams[s]) for s in streams):
        source = ("frozen", "residual")[turn % 2]
        turn += 1
        if offsets[source] < len(streams[source]):
            action, score = streams[source][offsets[source]]
            offsets[source] += 1
            selected.append((source, action, score))
    merged: dict[str, RoutedCandidate] = {}
    order: list[str] = []
    for source, action, score in selected:
        identifier = action_identity(action)
        if identifier not in merged:
            order.append(identifier)
            merged[identifier] = RoutedCandidate(action, (source,), score if source == "frozen" else None, score if source == "residual" else None)
        else:
            old = merged[identifier]
            merged[identifier] = RoutedCandidate(
                old.action, tuple(sorted(set(old.sources + (source,)))),
                score if source == "frozen" else old.frozen_score,
                score if source == "residual" else old.residual_score,
            )
    # Continue the same deterministic alternating fill after deduplication.
    while len(order) < budget and any(offsets[s] < len(streams[s]) for s in streams):
        source = ("frozen", "residual")[turn % 2]
        turn += 1
        if offsets[source] >= len(streams[source]):
            continue
        action, score = streams[source][offsets[source]]
        offsets[source] += 1
        identifier = action_identity(action)
        if identifier in merged:
            old = merged[identifier]
            merged[identifier] = RoutedCandidate(old.action, tuple(sorted(set(old.sources + (source,)))), score if source == "frozen" else old.frozen_score, score if source == "residual" else old.residual_score)
        else:
            order.append(identifier)
            merged[identifier] = RoutedCandidate(action, (source,), score if source == "frozen" else None, score if source == "residual" else None)
    return [merged[key] for key in order[:budget]]


def select_threshold(rows: Sequence[dict[str, Any]], feature: str) -> dict[str, Any]:
    """Select a deterministic train-only residual/frozen threshold rule."""
    values = sorted({float(row[feature]) for row in rows})
    candidates = []
    for operator in (">=", "<="):
        for threshold in values:
            decisions = [(float(row[feature]) >= threshold) if operator == ">=" else (float(row[feature]) <= threshold) for row in rows]
            covered = sum(row["residual_covered"] if use_residual else row["frozen_covered"] for row, use_residual in zip(rows, decisions))
            calls = sum(decisions)
            candidates.append((covered, -calls, operator, threshold))
    covered, negative_calls, operator, threshold = max(candidates, key=lambda item: (item[0], item[1], item[2], -item[3]))
    return {"feature": feature, "operator": operator, "threshold": threshold, "provider_when_true": "residual", "provider_when_false": "frozen", "training_states_covered": covered, "training_residual_calls": -negative_calls}


def percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * probability
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] if low == high else ordered[low] * (high - position) + ordered[high] * (position - low)


def aggregate_latency(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("latency values must not be empty")
    return {"mean_ms": mean(values), "median_ms": median(values), "p95_ms": percentile(values, .95), "min_ms": min(values), "max_ms": max(values)}


def frontier(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mark strict mean-cost/coverage dominance and retain variability context."""
    output = []
    for row in rows:
        dominators = [other["strategy"] for other in rows if other is not row and other["covered_states"] >= row["covered_states"] and other["mean_total_ms"] <= row["mean_total_ms"] and (other["covered_states"] > row["covered_states"] or other["mean_total_ms"] < row["mean_total_ms"])]
        output.append({**row, "pareto_dominated_by_mean": bool(dominators), "dominated_by": ";".join(dominators)})
    return output
