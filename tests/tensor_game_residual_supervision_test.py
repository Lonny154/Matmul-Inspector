import json

import pytest

from matmul_inspector.tensor_game import RankOneAction, outer_product
from matmul_inspector.tensor_game_experience import action_identity, tensor_digest
from matmul_inspector.tensor_game_residual_supervision import (
    TEACHER_VERSION,
    action_round_trip,
    assert_heldout_excluded,
    enumerate_residual_teacher,
    multiple_positive_target,
    support_overlap,
    validate_prototype_record,
)


ACTION = RankOneAction((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0))


def test_action_encoding_round_trip():
    assert action_round_trip(ACTION)


def test_teacher_score_and_order_are_deterministic():
    residual = outer_product(ACTION)
    first, first_audit = enumerate_residual_teacher(residual, 12)
    second, second_audit = enumerate_residual_teacher(residual, 12)
    assert first == second
    assert first[0]["action_id"] == action_identity(ACTION)
    assert first[0]["score"] == 0
    assert [
        (row["score"], row["action"]["u"], row["action"]["v"], row["action"]["w"])
        for row in first
    ] == sorted(
        (row["score"], row["action"]["u"], row["action"]["v"], row["action"]["w"])
        for row in first
    )
    for key in ("enumerated_uv_pairs", "unique_legal_candidates", "minimum_score"):
        assert first_audit[key] == second_audit[key]


def test_support_overlap_distinguishes_absence_from_probability():
    teacher = ["a", "b"]
    mcts = [
        {"action_id": "b", "search_probability": 0.25},
        {"action_id": "c", "search_probability": 0.75},
    ]
    result = support_overlap(teacher, mcts)
    assert result["overlap_count"] == 1
    assert result["teacher_absent_from_mcts"] == 1
    assert result["mcts_probability_on_teacher"] == 0.25


def test_multiple_positive_target_is_uniform_over_present_positives():
    assert multiple_positive_target(["a", "b", "c"], {"a", "c"}) == [0.5, 0.0, 0.5]
    with pytest.raises(ValueError, match="no positive"):
        multiple_positive_target(["a"], {"missing"})


def test_prototype_provenance_and_integrity():
    residual = outer_product(ACTION)
    state_id = tensor_digest(residual)
    record = {
        "state_id": state_id,
        "target_digest": "training-target",
        "source": "training",
        "residual": residual,
        "teacher_actions": [
            {
                "action": {"u": list(ACTION.u), "v": list(ACTION.v), "w": list(ACTION.w)},
                "action_id": action_identity(ACTION),
            }
        ],
        "provenance": {"teacher_version": TEACHER_VERSION},
    }
    validate_prototype_record(record, {"heldout-target"})
    bad = json.loads(json.dumps(record))
    bad["provenance"] = {}
    with pytest.raises(ValueError, match="provenance"):
        validate_prototype_record(bad, set())


def test_heldout_targets_are_exclusion_only():
    residual = outer_product(ACTION)
    target_digest = tensor_digest(residual)
    states = [{"state_id": target_digest, "target_digest": target_digest}]
    heldout = {
        "targets": [
            {
                "target": residual,
                "target_digest": target_digest,
                "canonical_factors": [
                    {"u": list(ACTION.u), "v": list(ACTION.v), "w": list(ACTION.w)}
                ],
            }
        ]
    }
    with pytest.raises(ValueError, match="held-out collision"):
        assert_heldout_excluded(states, heldout)
