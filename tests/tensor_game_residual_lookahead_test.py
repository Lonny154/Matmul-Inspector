import json

import pytest

from matmul_inspector.tensor_game import (
    RankOneAction,
    matrix_multiplication_tensor,
    standard_2x2_decomposition,
    strassen_2x2_decomposition,
    subtract_action,
)
from matmul_inspector.tensor_game_bounded_residual import (
    action_identity,
    bounded_residual_candidates_memory_efficient,
    squared_norm,
)
from matmul_inspector.tensor_game_residual_lookahead import (
    bounded_two_step_scores,
    exact_matrix_rank,
    exact_two_step_score,
    residual_subspace_features,
    tensor_flattening,
)


def test_exact_matrix_and_multiplication_tensor_flattening_ranks():
    assert exact_matrix_rank(()) == 0
    assert exact_matrix_rank(((0, 0), (0, 0))) == 0
    assert exact_matrix_rank(((1, 2), (2, 4), (-1, -2))) == 1
    assert exact_matrix_rank(((1, 2), (3, 4))) == 2
    with pytest.raises(ValueError, match="rectangular"):
        exact_matrix_rank(((1,), (1, 2)))

    target = matrix_multiplication_tensor(2, 2, 2)
    assert [exact_matrix_rank(tensor_flattening(target, mode)) for mode in range(3)] == [4, 4, 4]
    with pytest.raises(ValueError, match="mode"):
        tensor_flattening(target, 3)


def test_two_step_score_matches_direct_exact_residual_and_has_no_floats():
    target = matrix_multiplication_tensor(2, 2, 2)
    first, second = strassen_2x2_decomposition()[:2]
    score = exact_two_step_score(target, first, second)
    first_residual = subtract_action(target, first)
    endpoint = subtract_action(first_residual, second)

    assert score["initial_squared_norm"] == squared_norm(target)
    assert score["first_squared_norm"] == squared_norm(first_residual)
    assert score["second_squared_norm"] == squared_norm(endpoint)
    assert score["immediate_reduction"] == squared_norm(target) - squared_norm(first_residual)
    assert score["two_step_reduction"] == squared_norm(target) - squared_norm(endpoint)
    assert score["second_step_improvement"] == squared_norm(first_residual) - squared_norm(endpoint)

    def values(value):
        if isinstance(value, dict):
            for nested in value.values():
                yield from values(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                yield from values(nested)
        else:
            yield value

    assert not any(isinstance(value, float) for value in values(score))


def test_two_action_endpoint_is_order_independent_but_intermediate_can_differ():
    target = matrix_multiplication_tensor(2, 2, 2)
    conventional = standard_2x2_decomposition()[0]
    strassen = strassen_2x2_decomposition()[0]
    forward = exact_two_step_score(target, conventional, strassen)
    reverse = exact_two_step_score(target, strassen, conventional)

    assert forward["second_squared_norm"] == reverse["second_squared_norm"]
    assert forward["second_features"] == reverse["second_features"]
    assert forward["first_squared_norm"] != reverse["first_squared_norm"]


def test_features_handle_zero_and_negative_coefficients_exactly():
    zero = tuple(tuple(tuple(0 for _ in range(2)) for _ in range(2)) for _ in range(2))
    negative = (((-2, 0), (0, 0)), ((0, 0), (0, 1)))
    assert residual_subspace_features(zero)["flattening_ranks"] == [0, 0, 0]
    features = residual_subspace_features(negative)
    assert features["squared_norm"] == 5
    assert features["coefficient_l1"] == 3
    assert features["max_abs_coefficient"] == 2
    assert features["negative_entries"] == 1
    assert features["flattening_ranks"] == [2, 2, 2]


def test_bounded_lookahead_enforces_budget_and_serializes_stably():
    target = matrix_multiplication_tensor(2, 2, 2)
    actions = standard_2x2_decomposition()[:3]
    continuations = {
        action_identity(action): tuple(other for other in actions if other != action)
        for action in actions
    }
    first, audit = bounded_two_step_scores(
        target,
        actions,
        continuations,
        first_action_limit=3,
        continuations_per_action=2,
        total_evaluation_budget=3,
    )
    second, repeated_audit = bounded_two_step_scores(
        target,
        actions,
        continuations,
        first_action_limit=3,
        continuations_per_action=2,
        total_evaluation_budget=3,
    )
    assert audit == repeated_audit
    assert audit["continuation_evaluations"] == 3
    assert sum(row["continuations_evaluated"] for row in first) == 3
    assert json.dumps(first, sort_keys=True, separators=(",", ":")) == json.dumps(
        second, sort_keys=True, separators=(",", ":")
    )
    with pytest.raises(ValueError, match="positive"):
        bounded_two_step_scores(target, actions, continuations, first_action_limit=0,
                                continuations_per_action=2, total_evaluation_budget=3)


def test_offline_diagnostics_do_not_change_default_provider_results():
    target = matrix_multiplication_tensor(2, 2, 2)
    before = bounded_residual_candidates_memory_efficient(target, 128, 12)
    exact_two_step_score(target, *strassen_2x2_decomposition()[:2])
    residual_subspace_features(target)
    after = bounded_residual_candidates_memory_efficient(target, 128, 12)
    assert before[0] == after[0]
    for audit in (before[1], after[1]):
        audit.pop("total_ms")
        audit.pop("coarse_scoring_ms")
        audit.pop("exact_rescoring_ms")
        audit.pop("ranking_dedup_ms")
    assert before[1] == after[1]
