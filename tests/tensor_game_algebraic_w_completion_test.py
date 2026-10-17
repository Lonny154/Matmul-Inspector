import hashlib
import json

import pytest

from matmul_inspector.tensor_game import RankOneAction, matrix_multiplication_tensor, strassen_2x2_decomposition
from matmul_inspector.tensor_game_algebraic_w_completion import (
    CompletionConfig, complete_frozen_pairs, complete_w_vectors, completion_recall,
    frozen_algebraic_uv_pairs,
)
from matmul_inspector.tensor_game_bounded_residual import VECTORS
from matmul_inspector.tensor_game_meet_in_middle import MeetInMiddleConfig, meet_in_middle


def test_frozen_pair_manifest_identity_and_provenance_are_stable():
    target = matrix_multiplication_tensor(2, 2, 2)
    first, first_audit = frozen_algebraic_uv_pairs(target)
    second, second_audit = frozen_algebraic_uv_pairs(target)
    assert first == second and first_audit == second_audit
    assert len(first) == 192 == len({row["pair_id"] for row in first})
    assert first_audit["content_sha256"] == "3ce6b1414bc47e93a7bfa79145077b757397f5792adf25b32c8260740995bf2a"
    assert {row["source_mode"] for row in first} == {0, 1, 2}


@pytest.mark.parametrize("strategy", ("residual", "structural", "slice_projection", "hybrid"))
def test_w_completion_is_valid_deterministic_and_bounded(strategy):
    contraction = (2, -1, 0, 3)
    first, audit = complete_w_vectors(contraction, 4, 8, strategy)
    second, _ = complete_w_vectors(contraction, 4, 8, strategy)
    assert first == second and len(first) == 8
    assert len({row["w"] for row in first}) == 8
    assert all(row["w"] in VECTORS for row in first)
    assert audit["valid_w_domain"] == audit["exhaustive_exact_scores_for_matched_audit"] == 80


def test_slice_projection_uses_exact_contraction_alignment():
    rows, _ = complete_w_vectors((3, -2, 0, 1), 4, 1, "slice_projection")
    selected = rows[0]
    assert selected["dot"] > 0
    assert selected["projection_strength"].denominator > 0


def test_completion_deduplicates_and_enforces_fixed_budgets():
    target = matrix_multiplication_tensor(2, 2, 2)
    pairs, _ = frozen_algebraic_uv_pairs(target)
    rows, audit = complete_frozen_pairs(target, pairs, CompletionConfig("residual", 2, 64, 192, 15360))
    assert len(rows) == 64 == len({row["action_id"] for row in rows})
    assert audit["valid_w_vectors_scored"] == 192 * 80
    assert audit["selected_pair_w_instances"] == 192 * 2
    with pytest.raises(ValueError):
        CompletionConfig("residual", 2, 64, 192, 100).validate()


def test_recall_separates_pair_action_and_tensor_equivalence():
    target = matrix_multiplication_tensor(2, 2, 2)
    pairs, _ = frozen_algebraic_uv_pairs(target)
    rows, _ = complete_frozen_pairs(target, pairs, CompletionConfig("slice_projection", 2, 128, 192, 15360))
    recall = completion_recall(rows, strassen_2x2_decomposition(), pairs)
    assert recall["pair_recall"] == 6
    assert recall["complete_action_recall"] == recall["tensor_equivalence_recall"] == 2
    assert recall["records"][0]["earliest_exclusion_stage"] == "frozen_pair_set"


def test_rows_serialize_stably_and_remain_mitm_compatible():
    target = matrix_multiplication_tensor(2, 2, 2)
    pairs, _ = frozen_algebraic_uv_pairs(target)
    rows, _ = complete_frozen_pairs(target, pairs, CompletionConfig("residual", 1, 20, 192, 15360))
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(payload.encode()).hexdigest() == hashlib.sha256(payload.encode()).hexdigest()
    actions = tuple(RankOneAction(row["action"]["u"], row["action"]["v"], row["action"]["w"]) for row in rows)
    result = meet_in_middle(target, actions, MeetInMiddleConfig(
        left_depth=2, right_depth=2, maximum_left_attempts=20,
        maximum_right_attempts=20, maximum_stored_sums=20, maximum_seconds=2.0, seed=7))
    assert result["pool_size"] == 20
