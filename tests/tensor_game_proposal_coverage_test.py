import json

from matmul_inspector.tensor_game import (
    matrix_multiplication_tensor,
    standard_2x2_decomposition,
    strassen_2x2_decomposition,
    subtract_action,
)
from matmul_inspector.tensor_game_bounded_residual import (
    action_from_payload,
    action_identity,
    bounded_residual_candidates_memory_efficient,
    exact_score_from_contraction,
)
from matmul_inspector.tensor_game_proposal_coverage import evaluate_state, stable_json


def test_full_space_scores_ids_membership_and_order_are_exact():
    residual = matrix_multiplication_tensor(2, 2, 2)
    result = evaluate_state(residual, widths=(128,))
    actions = result["actions"]
    assert [row["exact_rank"] for row in actions] == list(range(1, len(actions) + 1))
    assert len({row["action_id"] for row in actions}) == len(actions)
    for row in actions:
        assert row["action_id"] == action_identity(action_from_payload(row["action"]))
        assert row["retained_by_bounded_128"] == (row["first_pair_rank"] <= 128)
    bounded, _ = bounded_residual_candidates_memory_efficient(residual, 128, 6400)
    assert {row["action_id"] for row in bounded} == {row["action_id"] for row in actions if row["retained_by_bounded_128"]}


def test_pair_w_and_exact_score_match_existing_convention():
    result = evaluate_state(matrix_multiplication_tensor(2, 2, 2), widths=(128,))
    by_id = {row["action_id"]: row for row in result["actions"]}
    for pair in result["pairs"]:
        if pair["status"] in {"valid", "duplicate_equivalent_action"}:
            row = by_id[pair["action_id"]]
            assert exact_score_from_contraction(result["residual_squared_norm"], pair["contraction"], pair["q"], pair["w"]) == row["exact_score"]


def test_known_conventional_and_strassen_recall_classification():
    target = matrix_multiplication_tensor(2, 2, 2)
    conventional = evaluate_state(target, widths=(128, 6400), expected_action=standard_2x2_decomposition()[0])
    assert conventional["known_action"]["complete_action_reachable"]
    assert conventional["known_action"]["retained_by_bounded_128"]
    strassen = evaluate_state(target, widths=(128, 6400), expected_action=strassen_2x2_decomposition()[0])
    assert strassen["known_action"]["complete_action_in_coefficient_domain"]
    assert strassen["known_action"]["uv_pair_in_domain"]
    assert not strassen["known_action"]["complete_action_reachable"]
    assert strassen["known_action"]["classification"] == "uv_present_but_incompatible_with_w_rule"


def test_counterfactual_pareto_ties_and_serialization_are_stable():
    residual = subtract_action(matrix_multiplication_tensor(2, 2, 2), strassen_2x2_decomposition()[0])
    first = evaluate_state(residual, widths=(32, 128, 6400))
    second = evaluate_state(residual, widths=(32, 128, 6400))
    assert first["actions"] == second["actions"]
    counts = [row["candidate_count"] for row in first["counterfactual_widths"]]
    assert counts == sorted(counts)
    assert 0 <= first["bounded_128"]["full_pareto_recall"] <= 1
    assert json.loads(stable_json(first))["state_identity"] == first["state_identity"]


def test_default_bounded_128_behavior_is_unchanged():
    residual = matrix_multiplication_tensor(2, 2, 2)
    before, audit = bounded_residual_candidates_memory_efficient(residual, 128, 12)
    evaluate_state(residual, widths=(128,))
    after, audit_after = bounded_residual_candidates_memory_efficient(residual, 128, 12)
    assert before == after
    assert audit["fully_evaluated_pairs"] == audit_after["fully_evaluated_pairs"] == 128
