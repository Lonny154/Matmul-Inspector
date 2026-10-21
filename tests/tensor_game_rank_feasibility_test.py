import json

import pytest

from matmul_inspector.tensor_game import (
    matrix_multiplication_tensor,
    standard_2x2_decomposition,
    strassen_2x2_decomposition,
    zero_tensor,
)
from matmul_inspector.tensor_game_bounded_residual import (
    bounded_residual_candidates_memory_efficient,
)
from matmul_inspector.tensor_game_rank_feasibility import (
    exact_matrix_rank,
    feasibility_record,
    flattening_rank_lower_bound,
    sample_frozen_rank7_failures,
    tensor_flattening,
    trajectory_feasibility,
)


def test_all_three_flattening_modes_follow_tensor_axis_conventions():
    tensor = (((1, 2), (3, 4)), ((5, 6), (7, 8)))
    assert tensor_flattening(tensor, 0) == ((1, 2, 3, 4), (5, 6, 7, 8))
    assert tensor_flattening(tensor, 1) == ((1, 5, 2, 6), (3, 7, 4, 8))
    assert tensor_flattening(tensor, 2) == ((1, 3, 5, 7), (2, 4, 6, 8))


def test_exact_rank_handles_identity_singular_zero_and_negative_matrices():
    assert exact_matrix_rank(((1, 0, 0), (0, 1, 0), (0, 0, 1))) == 3
    assert exact_matrix_rank(((0, 0), (0, 0))) == 0
    assert exact_matrix_rank(((-1, 2), (2, -4))) == 1
    assert exact_matrix_rank(((-1, 2), (3, -5))) == 2
    # Independent 2x2 determinant cross-check. A nonzero determinant means rank 2.
    matrix = ((-1, 2), (3, -5))
    assert matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0] != 0
    assert all(not isinstance(value, float) for row in matrix for value in row)


def test_lower_bound_for_multiplication_and_zero_tensors():
    target = matrix_multiplication_tensor(2, 2, 2)
    zero = zero_tensor((4, 4, 4))
    assert flattening_rank_lower_bound(target) == {
        "tensor_shape": [4, 4, 4],
        "flattening_shapes": [[4, 16], [4, 16], [4, 16]],
        "flattening_ranks": [4, 4, 4],
        "lower_bound": 4,
        "field": "rational_numbers",
    }
    assert flattening_rank_lower_bound(zero)["lower_bound"] == 0


def test_remaining_budget_accounting_and_infeasibility_semantics():
    target = matrix_multiplication_tensor(2, 2, 2)
    assert feasibility_record(target, 3)["provably_infeasible"]
    assert not feasibility_record(target, 4)["provably_infeasible"]
    assert feasibility_record(target, 4)["passing_bound_semantics"] == "unknown_feasibility"
    with pytest.raises(ValueError, match="nonnegative"):
        feasibility_record(target, -1)
    with pytest.raises(TypeError, match="integer"):
        feasibility_record(target, 1.5)


@pytest.mark.parametrize(
    ("name", "actions"),
    (("conventional", standard_2x2_decomposition()),
     ("strassen", strassen_2x2_decomposition())),
)
def test_known_completion_paths_are_never_incorrectly_pruned(name, actions):
    target = matrix_multiplication_tensor(2, 2, 2)
    orders = (tuple(range(len(actions))), tuple(reversed(range(len(actions)))))
    for order in orders:
        rows = trajectory_feasibility(target, actions, order, family=name)
        assert len(rows) == len(actions) + 1
        assert [row["remaining_budget"] for row in rows] == list(reversed(range(len(actions) + 1)))
        assert all(row["known_valid_completion"] for row in rows)
        assert not any(row["provably_infeasible"] for row in rows)
        assert rows[-1]["lower_bound"] == rows[-1]["squared_norm"] == 0


def test_failure_sampling_and_serialization_are_deterministic():
    target = matrix_multiplication_tensor(2, 2, 2)
    first = sample_frozen_rank7_failures(
        target, node_budget=10, beam_width=4, max_actions=3, samples_per_depth=2
    )
    second = sample_frozen_rank7_failures(
        target, node_budget=10, beam_width=4, max_actions=3, samples_per_depth=2
    )
    assert first == second
    assert json.dumps(first, sort_keys=True, separators=(",", ":")) == json.dumps(
        second, sort_keys=True, separators=(",", ":")
    )


def test_offline_rank_checks_preserve_default_provider_output():
    target = matrix_multiplication_tensor(2, 2, 2)
    before, _ = bounded_residual_candidates_memory_efficient(target, 128, 12)
    flattening_rank_lower_bound(target)
    sample_frozen_rank7_failures(
        target, node_budget=2, beam_width=2, max_actions=2, samples_per_depth=1
    )
    after, _ = bounded_residual_candidates_memory_efficient(target, 128, 12)
    assert before == after
