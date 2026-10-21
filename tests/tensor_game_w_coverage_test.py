import json

from matmul_inspector.tensor_game import (
    matrix_multiplication_tensor,
    standard_2x2_decomposition,
    strassen_2x2_decomposition,
    subtract_action,
    zero_tensor,
)
from matmul_inspector.tensor_game_bounded_residual import action_identity, bounded_residual_candidates_memory_efficient
from matmul_inspector.tensor_game_candidate_dataset import validate_decomposition
from matmul_inspector.tensor_game_quota_w import exhaustive_w_vectors, top_k_w_vectors
from matmul_inspector.tensor_game_w_coverage import (
    analyze_action_at_state,
    exact_complete_action_ranks,
    sign_equivalent_representations,
)


def test_strassen_terms_are_exact_and_finish_at_zero():
    actions = strassen_2x2_decomposition()
    assert len(actions) == 7
    validate_decomposition(actions)
    residual = matrix_multiplication_tensor(2, 2, 2)
    for action in actions:
        residual = subtract_action(residual, action)
    assert residual == zero_tensor((4, 4, 4))


def test_all_first_actions_have_exact_expected_residual_scores_and_pair_ranks():
    target = matrix_multiplication_tensor(2, 2, 2)
    rows = [analyze_action_at_state(target, action) for action in strassen_2x2_decomposition()]
    assert [row["immediate_residual_reduction"] for row in rows] == [-4, -2, -2, -2, -2, -2, -2]
    assert [row["best_pair_rank"] for row in rows] == [2189, 3077, 2271, 3176, 1952, 41, 89]
    assert [row["best_required_w_rank"] for row in rows] == [3, 7, 5, 2, 3, 2, 1]
    assert all(row["rank_one_reconstruction"] and row["uv_pair_in_domain"] for row in rows)
    assert [row["earliest_exclusion_stage"] for row in rows[:5]] == ["pair_proposal_selection"] * 5


def test_sign_representations_and_top_k_match_exhaustive():
    target = matrix_multiplication_tensor(2, 2, 2)
    action = strassen_2x2_decomposition()[5]
    assert len(sign_equivalent_representations(action)) == 4
    row = analyze_action_at_state(target, action)
    best = min(row["representations"], key=lambda item: item["pair_rank"])
    exhaustive = exhaustive_w_vectors(best["contraction"], best["q"])
    top_k, _ = top_k_w_vectors(best["contraction"], best["q"], 80)
    assert top_k == exhaustive
    assert best["required_w_rank"] == 2


def test_full_action_rank_is_deterministic_and_conventional_scores_better():
    target = matrix_multiplication_tensor(2, 2, 2)
    strassen = strassen_2x2_decomposition()
    conventional = standard_2x2_decomposition()
    first = exact_complete_action_ranks(target, strassen + conventional)
    second = exact_complete_action_ranks(target, strassen + conventional)
    assert first == second
    assert first["canonical_action_count"] == 128000
    conventional_ranks = [first["ranks"][action_identity(action)] for action in conventional]
    strassen_ranks = [first["ranks"][action_identity(action)] for action in strassen]
    assert max(conventional_ranks) < min(strassen_ranks)


def test_diagnostic_is_stably_serializable_and_preserves_baseline_provider():
    target = matrix_multiplication_tensor(2, 2, 2)
    before = bounded_residual_candidates_memory_efficient(target, 128, 12)[0]
    first = analyze_action_at_state(target, strassen_2x2_decomposition()[0])
    second = analyze_action_at_state(target, strassen_2x2_decomposition()[0])
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert bounded_residual_candidates_memory_efficient(target, 128, 12)[0] == before


def test_subset_residual_is_independent_of_term_order():
    actions = strassen_2x2_decomposition()
    forward = reverse = matrix_multiplication_tensor(2, 2, 2)
    for index in (0, 2, 5):
        forward = subtract_action(forward, actions[index])
    for index in (5, 2, 0):
        reverse = subtract_action(reverse, actions[index])
    assert forward == reverse
