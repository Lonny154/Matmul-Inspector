from dataclasses import asdict

import pytest

from matmul_inspector.tensor_game import (
    RankOneAction,
    matrix_multiplication_tensor,
    standard_2x2_decomposition,
    strassen_2x2_decomposition,
)
from matmul_inspector.tensor_game_bounded_residual import bounded_residual_candidates_memory_efficient
from matmul_inspector.tensor_game_hardware_cost import (
    CandidateObjectives,
    action_cost,
    decomposition_cost,
    linear_combination_cost,
    pareto_frontier,
)


def residual():
    return tuple(tuple(tuple(1 if i == j == k else 0 for k in range(4)) for j in range(4)) for i in range(4))


def test_linear_combination_zero_singleton_sign_and_magnitude():
    zero = linear_combination_cost((0, 0, 0))
    assert (zero.nonzeros, zero.additions, zero.max_abs_coefficient) == (0, 0, 0)
    singleton = linear_combination_cost((0, -3, 0))
    assert (singleton.additions, singleton.negative_coefficients, singleton.nonunit_coefficients) == (0, 1, 1)
    assert singleton.magnitude_histogram == ((3, 1),)
    mixed = linear_combination_cost((-2, 1, 0, 2))
    assert (mixed.nonzeros, mixed.additions, mixed.coefficient_l1) == (3, 2, 5)


def test_action_schedule_counts_input_and_output_separately():
    cost = action_cost(RankOneAction((1, -1), (2, 0), (1, 0, -2)))
    assert cost.preparation_additions == 1
    assert cost.output_accumulation_additions == 2
    assert cost.coefficient_scalings == 2
    assert cost.temporary_scalars == 3


def test_pareto_dominance_ties_zero_cost_and_stable_order():
    candidates = (
        CandidateObjectives(0, 4, 2), CandidateObjectives(1, 4, 2),
        CandidateObjectives(2, 3, 3), CandidateObjectives(3, 2, 0),
        CandidateObjectives(4, 1, 0), CandidateObjectives(5, 5, 4),
    )
    assert [item.candidate_index for item in pareto_frontier(candidates)] == [0, 1, 3, 5]


def test_reference_decomposition_counts_and_exact_validation():
    target = matrix_multiplication_tensor(2, 2, 2)
    standard = decomposition_cost(standard_2x2_decomposition(), target)
    strassen = decomposition_cost(strassen_2x2_decomposition(), target)
    assert asdict(standard) == {
        "rank_multiplications": 8, "pre_additions": 0, "post_additions": 4,
        "coefficient_scalings": 0, "peak_temporary_scalars": 7,
        "shared_input_combinations": 8,
    }
    assert (strassen.rank_multiplications, strassen.pre_additions, strassen.post_additions) == (7, 10, 8)
    with pytest.raises(ValueError, match="exactly reconstruct"):
        decomposition_cost(strassen_2x2_decomposition()[:-1], target)


def test_optional_analysis_preserves_default_rows_and_is_deterministic():
    baseline_rows, baseline_audit = bounded_residual_candidates_memory_efficient(residual(), 128, 12)
    first_rows, first_audit = bounded_residual_candidates_memory_efficient(residual(), 128, 12, analyze_costs=True)
    second_rows, second_audit = bounded_residual_candidates_memory_efficient(residual(), 128, 12, analyze_costs=True)
    assert first_rows == baseline_rows == second_rows
    assert "cost_analysis" not in baseline_audit
    first = first_audit["cost_analysis"]
    second = second_audit["cost_analysis"]
    assert first["candidates"] == second["candidates"]
    assert first["candidate_count"] <= 128
    assert first["pareto_frontier_size"] > 0
