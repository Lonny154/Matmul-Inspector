"""Coverage-aware TensorGame policy training utilities for TG-4I."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

from .tensor_game import subtract_action, validate_action, zero_tensor
from .tensor_game_experience import action_from_payload, action_identity, tensor_digest
from .tensor_game_offline_learning import ExperienceDataset, collate_experience
from .tensor_game_residual_supervision import (
    COEFFICIENTS,
    SHAPE,
    action_round_trip,
    enumerate_residual_teacher,
    greedy_teacher_continuation,
)

VERSION = 1


def canonical_action_log_probabilities(logits, actions, shape=SHAPE):
    """Differentiable globally normalized probabilities of canonical sign orbits."""
    import torch

    if logits.ndim != 3 or actions.ndim != 3 or actions.shape[0] != logits.shape[0]:
        raise ValueError("invalid logits or positive-action tensor")
    classes = {-1: 0, 0: 1, 1: 2}
    log_coefficients = torch.log_softmax(logits, dim=-1)
    output = []
    for batch_index in range(actions.shape[0]):
        batch_scores = []
        for action in actions[batch_index]:
            values = tuple(int(value) for value in action.tolist())
            first, second = shape[0], shape[0] + shape[1]
            u, v, w = values[:first], values[first:second], values[second:]
            forms = (
                (u, v, w),
                (tuple(-x for x in u), tuple(-x for x in v), w),
                (tuple(-x for x in u), v, tuple(-x for x in w)),
                (u, tuple(-x for x in v), tuple(-x for x in w)),
            )
            form_scores = []
            for form in forms:
                flat = form[0] + form[1] + form[2]
                indices = torch.tensor(
                    [classes[value] for value in flat], device=logits.device
                )
                form_scores.append(
                    log_coefficients[batch_index].gather(1, indices[:, None]).sum()
                )
            batch_scores.append(torch.logsumexp(torch.stack(form_scores), dim=0))
        output.append(torch.stack(batch_scores))
    return output


def coverage_loss(logits, positive_actions, positive_mask, shape=SHAPE):
    """Mean negative log probability mass assigned to each nonempty positive set."""
    import torch

    if positive_mask.ndim != 2 or positive_actions.shape[:2] != positive_mask.shape:
        raise ValueError("invalid positive mask")
    if torch.any(positive_mask.sum(1) == 0):
        raise ValueError("coverage loss requires a nonempty positive set")
    scores = canonical_action_log_probabilities(logits, positive_actions, shape)
    losses = []
    for row_scores, row_mask in zip(scores, positive_mask):
        losses.append(-torch.logsumexp(row_scores[row_mask], dim=0))
    return torch.stack(losses).mean()


def per_positive_coverage_loss(logits, positive_actions, positive_mask, shape=SHAPE):
    """Mean canonical NLL over every verified positive, then over states."""
    import torch

    if torch.any(positive_mask.sum(1) == 0):
        raise ValueError("per-positive loss requires a nonempty positive set")
    scores = canonical_action_log_probabilities(logits, positive_actions, shape)
    return torch.stack(
        [-row_scores[row_mask].mean() for row_scores, row_mask in zip(scores, positive_mask)]
    ).mean()


def selective_retention_kl(student_logits, frozen_logits, state_mask):
    """Token-level KL(frozen || student) on preregistered useful-frozen states."""
    import torch

    if frozen_logits.requires_grad:
        raise ValueError("frozen retention targets must be detached")
    if state_mask.ndim != 1 or state_mask.shape[0] != student_logits.shape[0]:
        raise ValueError("invalid retention state mask")
    if not torch.any(state_mask):
        return student_logits.sum() * 0
    student = torch.log_softmax(student_logits[state_mask], dim=-1)
    teacher_log = torch.log_softmax(frozen_logits[state_mask], dim=-1)
    teacher = teacher_log.exp()
    return (teacher * (teacher_log - student)).sum(-1).mean()


def first_decoder_pruning_position(logits, action, beam_width: int = 256) -> int | None:
    """Return the first 1-based token position where every sign form leaves the beam."""
    import torch

    values = action.u + action.v + action.w
    forms = (
        values,
        tuple(-x for x in action.u) + tuple(-x for x in action.v) + action.w,
        tuple(-x for x in action.u) + action.v + tuple(-x for x in action.w),
        action.u + tuple(-x for x in action.v) + tuple(-x for x in action.w),
    )
    coefficients = (-1, 0, 1)
    logp = torch.log_softmax(logits.detach().cpu(), dim=-1)
    beam = [((), 0.0)]
    for position in range(logp.shape[0]):
        candidates = [
            (prefix + (coefficient,), score + float(logp[position, class_index]))
            for prefix, score in beam
            for class_index, coefficient in enumerate(coefficients)
        ]
        candidates.sort(key=lambda item: (-item[1], item[0]))
        beam = candidates[:beam_width]
        prefixes = {prefix for prefix, _ in beam}
        if not any(form[: position + 1] in prefixes for form in forms):
            return position + 1
    return None


def verified_labels(target_records: Sequence[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    """Create root labels from exact constructions and verified teacher continuations."""
    rows = []
    for target in sorted(
        (row for row in target_records if row["split"] == split),
        key=lambda row: row["target_id"],
    ):
        residual = tuple(tuple(tuple(line) for line in plane) for plane in target["target"])
        positives = {}
        construction = action_from_payload(target["canonical_factors"][0])
        positives[action_identity(construction)] = {
            "action": target["canonical_factors"][0],
            "action_id": action_identity(construction),
            "provenance": ["validated_construction_continuation"],
            "continuation_verified": True,
        }
        teacher, _ = enumerate_residual_teacher(residual, 3)
        for candidate in teacher:
            action = action_from_payload(candidate["action"])
            next_residual = subtract_action(residual, action)
            if next_residual == zero_tensor(SHAPE):
                success = True
            else:
                success = greedy_teacher_continuation(
                    next_residual, target["construction_length"] - 1
                )["success"]
            if success:
                entry = positives.setdefault(
                    candidate["action_id"],
                    {
                        "action": candidate["action"],
                        "action_id": candidate["action_id"],
                        "provenance": [],
                        "continuation_verified": True,
                    },
                )
                entry["provenance"].append("residual_teacher_verified_greedy_continuation")
        for entry in positives.values():
            action = action_from_payload(entry["action"])
            validate_action(action, SHAPE, frozenset(COEFFICIENTS))
            if not action_round_trip(action):
                raise ValueError("positive action failed encoding round trip")
            entry["provenance"] = sorted(set(entry["provenance"]))
        rows.append(
            {
                "state_id": tensor_digest(residual),
                "target_id": target["target_id"],
                "target_digest": target["target_digest"],
                "constructed_depth": target["construction_length"],
                "split": split,
                "residual": residual,
                "positive_actions": sorted(positives.values(), key=lambda row: row["action_id"]),
                "validation": {
                    "legal": True,
                    "encoding_round_trip": True,
                    "nonempty": True,
                },
            }
        )
    return rows


class CoverageDataset:
    def __init__(self, experience: Path, split: str, labels: Sequence[dict[str, Any]]):
        self.base = ExperienceDataset(experience, split)
        self.labels = {row["state_id"]: row for row in labels if row["split"] == split}

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        item = dict(self.base[index])
        state_id = tensor_digest(self.base.rows[index]["residual"])
        label = self.labels.get(state_id)
        item["state_id"] = state_id
        item["positive_actions"] = (
            [action_from_payload(row["action"]) for row in label["positive_actions"]]
            if label
            else []
        )
        return item


def collate_coverage(rows):
    import torch

    result = collate_experience(rows)
    labeled = [index for index, row in enumerate(rows) if row["positive_actions"]]
    maximum = max((len(rows[index]["positive_actions"]) for index in labeled), default=0)
    actions = torch.zeros((len(labeled), maximum, sum(SHAPE)), dtype=torch.long)
    mask = torch.zeros((len(labeled), maximum), dtype=torch.bool)
    for output_index, input_index in enumerate(labeled):
        values = [action.u + action.v + action.w for action in rows[input_index]["positive_actions"]]
        actions[output_index, : len(values)] = torch.tensor(values)
        mask[output_index, : len(values)] = True
    result["coverage_indices"] = torch.tensor(labeled, dtype=torch.long)
    result["positive_actions"] = actions
    result["positive_mask"] = mask
    result["state_id"] = [row["state_id"] for row in rows]
    return result


def labels_digest(labels: Sequence[dict[str, Any]]) -> str:
    return hashlib.sha256(
        json.dumps(labels, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_separation(
    labels: Sequence[dict[str, Any]], reserved_manifests: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    label_targets = {row["target_digest"] for row in labels}
    reserved = {
        target["target_digest"]
        for manifest in reserved_manifests
        for target in manifest["targets"]
    }
    overlap = label_targets & reserved
    if overlap:
        raise ValueError(f"coverage-label target leakage: {sorted(overlap)}")
    if len({row["state_id"] for row in labels}) != len(labels):
        raise ValueError("duplicate labeled state")
    return {
        "valid": True,
        "labeled_states": len(labels),
        "labeled_targets": len(label_targets),
        "reserved_targets": len(reserved),
        "target_overlap": 0,
        "split_counts": dict(Counter(row["split"] for row in labels)),
        "positive_actions": sum(len(row["positive_actions"]) for row in labels),
    }


def checkpoint_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_checkpoint_provenance(manifest: dict[str, Any], source_sha256: str) -> None:
    if manifest.get("source_checkpoint_sha256") != source_sha256:
        raise ValueError("checkpoint source provenance mismatch")
    if not manifest.get("checkpoint_sha256") or manifest.get("fixed_epochs_completed") != 30:
        raise ValueError("checkpoint manifest is incomplete")


def aggregate_discovery(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "trials": len(rows),
        "trial_successes": sum(bool(row["success"]) for row in rows),
        "unique_targets": len({row["target_id"] for row in rows}),
        "targets_solved": len({row["target_id"] for row in rows if row["success"]}),
    }


def verify_artifact_hash(path: Path, expected: str) -> None:
    if checkpoint_sha256(path) != expected:
        raise ValueError(f"artifact hash mismatch: {path}")
