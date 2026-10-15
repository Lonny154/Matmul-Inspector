"""Offline residual-teacher supervision analysis for TG-4H."""

from __future__ import annotations

from collections import Counter
import hashlib
import itertools
import json
import math
import random
import time
from typing import Any, Iterable, Sequence

from .tensor_game import RankOneAction, subtract_action, validate_action, zero_tensor
from .tensor_game_experience import action_from_payload, action_identity, tensor_digest
from .tensor_game_search import canonicalize_action, quantize_ratio

COEFFICIENTS = (-1, 0, 1)
SHAPE = (4, 4, 4)
TEACHER_VERSION = "tg4h.exhaustive_residual_teacher.v1"


def squared_norm(tensor: Any) -> int:
    return sum(value * value for plane in tensor for row in plane for value in row)


def sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def action_round_trip(action: RankOneAction) -> bool:
    """Verify the policy's 12 coefficient representation without model inference."""
    payload = {"u": list(action.u), "v": list(action.v), "w": list(action.w)}
    return canonicalize_action(action_from_payload(payload)) == canonicalize_action(action)


def enumerate_residual_teacher(
    residual: Any, limit: int = 12
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply the exact TG-4F residual scorer and retain scores and audit counts."""
    if limit <= 0:
        raise ValueError("teacher limit must be positive")
    started = time.perf_counter()
    vectors = [vector for vector in itertools.product(COEFFICIENTS, repeat=4) if any(vector)]
    candidates: dict[RankOneAction, int] = {}
    score_ties: Counter[int] = Counter()
    for u in vectors:
        uu = sum(value * value for value in u)
        for v in vectors:
            denominator = uu * sum(value * value for value in v)
            contraction = tuple(
                sum(
                    residual[i][j][k] * u[i] * v[j]
                    for i in range(4)
                    for j in range(4)
                )
                for k in range(4)
            )
            w = tuple(quantize_ratio(value, denominator, COEFFICIENTS) for value in contraction)
            if not any(w):
                continue
            action = canonicalize_action(RankOneAction(u, v, w))
            score = squared_norm(subtract_action(residual, action))
            candidates[action] = min(score, candidates.get(action, math.inf))
    ordered = sorted(candidates.items(), key=lambda item: (item[1], item[0].u, item[0].v, item[0].w))
    score_ties.update(score for _, score in ordered)
    rows = [
        {
            "action": {"u": list(action.u), "v": list(action.v), "w": list(action.w)},
            "action_id": action_identity(action),
            "score": score,
            "rank": rank,
            "score_tie_count": score_ties[score],
        }
        for rank, (action, score) in enumerate(ordered[:limit], 1)
    ]
    return rows, {
        "enumerated_uv_pairs": len(vectors) ** 2,
        "nonzero_vectors_per_axis": len(vectors),
        "unique_legal_candidates": len(candidates),
        "returned_candidates": len(rows),
        "elapsed_seconds": time.perf_counter() - started,
        "minimum_score": ordered[0][1] if ordered else None,
        "minimum_score_ties": score_ties[ordered[0][1]] if ordered else 0,
    }


def residual_score_distribution(rows: Sequence[dict[str, Any]], temperature: float) -> list[float]:
    """Turn lower-is-better residual scores into a stable probability target."""
    if temperature <= 0 or not rows:
        raise ValueError("temperature and teacher rows must be positive")
    minimum = min(row["score"] for row in rows)
    weights = [math.exp(-(row["score"] - minimum) / temperature) for row in rows]
    total = sum(weights)
    return [weight / total for weight in weights]


def multiple_positive_target(
    action_ids: Sequence[str], positive_ids: Iterable[str]
) -> list[float]:
    """Return a uniform target over every labeled positive in a candidate support."""
    positives = set(positive_ids)
    selected = [identifier in positives for identifier in action_ids]
    count = sum(selected)
    if count == 0:
        raise ValueError("no positive action is present in the candidate support")
    return [int(value) / count for value in selected]


def support_overlap(teacher_ids: Iterable[str], mcts_actions: Sequence[dict[str, Any]]) -> dict[str, Any]:
    teacher = set(teacher_ids)
    support = {row["action_id"] for row in mcts_actions}
    overlap = teacher & support
    return {
        "teacher_count": len(teacher),
        "mcts_count": len(support),
        "overlap_count": len(overlap),
        "teacher_absent_from_mcts": len(teacher - support),
        "overlap_action_ids": sorted(overlap),
        "mcts_probability_on_teacher": sum(
            row["search_probability"] for row in mcts_actions if row["action_id"] in teacher
        ),
    }


def select_states(
    training_manifest: dict[str, Any],
    calibration: dict[str, Any],
    decisions: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Select roots by identifier before teacher scoring: two train and two calibration per depth."""
    decision_by_state: dict[str, list[dict[str, Any]]] = {}
    for decision in decisions:
        decision_by_state.setdefault(decision["state_digest"], []).append(decision)
    selected = []
    sources = (
        ("training", [row for row in training_manifest["metadata"]["target_records"] if row["split"] == "train"]),
        ("calibration", calibration["targets"]),
    )
    for source, records in sources:
        for depth in range(1, 5):
            candidates = sorted(
                (row for row in records if row["construction_length"] == depth),
                key=lambda row: row["target_id"],
            )[:2]
            if len(candidates) != 2:
                raise ValueError(f"insufficient {source} targets at depth {depth}")
            for record in candidates:
                residual = tuple(tuple(tuple(line) for line in plane) for plane in record["target"])
                state_id = tensor_digest(residual)
                matches = sorted(
                    decision_by_state.get(state_id, []),
                    key=lambda row: (row["random_seed"], row["experience_id"]),
                )
                selected.append(
                    {
                        "state_id": state_id,
                        "target_id": record["target_id"],
                        "target_digest": record["target_digest"],
                        "source": source,
                        "constructed_depth": depth,
                        "residual": residual,
                        "reference_actions": record["canonical_factors"],
                        "mcts_decision": matches[0] if source == "training" and matches else None,
                    }
                )
    return selected


def assert_heldout_excluded(states: Sequence[dict[str, Any]], heldout: dict[str, Any]) -> None:
    heldout_targets = {row["target_digest"] for row in heldout["targets"]}
    heldout_states = set()
    for record in heldout["targets"]:
        residual = tuple(tuple(tuple(line) for line in plane) for plane in record["target"])
        for payload in record["canonical_factors"]:
            heldout_states.add(tensor_digest(residual))
            residual = subtract_action(residual, action_from_payload(payload))
    collisions = [
        row["state_id"]
        for row in states
        if row["target_digest"] in heldout_targets or row["state_id"] in heldout_states
    ]
    if collisions:
        raise ValueError(f"TG-4G held-out collision: {collisions}")


def greedy_teacher_continuation(residual: Any, moves: int) -> dict[str, Any]:
    actions = []
    scores = []
    for _ in range(moves):
        ranked, _ = enumerate_residual_teacher(residual, 1)
        if not ranked:
            break
        action = action_from_payload(ranked[0]["action"])
        actions.append(ranked[0]["action_id"])
        scores.append(ranked[0]["score"])
        residual = subtract_action(residual, action)
        if residual == zero_tensor(SHAPE):
            break
    return {
        "success": residual == zero_tensor(SHAPE),
        "steps": len(actions),
        "action_ids": actions,
        "residual_scores": scores,
        "final_residual_norm_squared": squared_norm(residual),
    }


def canonical_json_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_prototype_record(record: dict[str, Any], heldout_digests: set[str]) -> None:
    required = {
        "state_id",
        "target_digest",
        "source",
        "residual",
        "teacher_actions",
        "provenance",
    }
    if not required <= record.keys():
        raise ValueError("prototype label is missing provenance fields")
    if record["target_digest"] in heldout_digests:
        raise ValueError("prototype label includes a held-out target")
    if tensor_digest(tuple(tuple(tuple(line) for line in plane) for plane in record["residual"])) != record["state_id"]:
        raise ValueError("prototype state digest mismatch")
    if record["provenance"].get("teacher_version") != TEACHER_VERSION:
        raise ValueError("prototype teacher provenance mismatch")
    for row in record["teacher_actions"]:
        action = action_from_payload(row["action"])
        validate_action(action, SHAPE, frozenset(COEFFICIENTS))
        if not action_round_trip(action) or action_identity(action) != row["action_id"]:
            raise ValueError("prototype action encoding mismatch")
