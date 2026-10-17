from fractions import Fraction

import pytest

from matmul_inspector.tensor_game import RankOneAction, matrix_multiplication_tensor, standard_2x2_decomposition, strassen_2x2_decomposition
from matmul_inspector.tensor_game_algebraic_pools import (
    AlgebraicPoolConfig, algebraic_action_candidates, combine_slices, exact_column_space_basis,
    exact_null_space_basis, exact_row_space_basis, factor_rank_one_matrix, hybrid_pool,
    known_action_recall, low_rank_matrix_factor_proposals, tensor_slices, vector_in_row_space,
)
from matmul_inspector.tensor_game_meet_in_middle import MeetInMiddleConfig, meet_in_middle
from matmul_inspector.tensor_game_residual_lookahead import exact_matrix_rank


def test_exact_slice_extraction_and_combination():
    target = matrix_multiplication_tensor(2, 2, 2)
    slices = tensor_slices(target, 2)
    assert len(slices) == 4 and all(len(matrix) == 4 for matrix in slices)
    assert combine_slices(target, 2, (1, 0, 0, 0)) == slices[0]
    assert exact_matrix_rank(slices[0]) == 2
    with pytest.raises(ValueError):
        combine_slices(target, 2, (1, 0))


def test_exact_spaces_and_null_space():
    matrix = ((1, 1, 0), (0, 0, 1), (1, 1, 1))
    assert exact_row_space_basis(matrix) == ((Fraction(1), Fraction(1), Fraction(0)), (Fraction(0), Fraction(0), Fraction(1)))
    assert exact_column_space_basis(matrix) == ((Fraction(1), Fraction(0), Fraction(1)), (Fraction(0), Fraction(1), Fraction(1)))
    null = exact_null_space_basis(matrix)
    assert null == ((Fraction(-1), Fraction(1), Fraction(0)),)
    assert vector_in_row_space((1, 1, 2), matrix)
    assert not vector_in_row_space((1, 0, 0), matrix)


def test_rank_one_factorization_and_rank_two_factor_proposals():
    matrix = tuple(tuple(left * right for right in (1, 0, -1, 0)) for left in (1, 1, 0, 0))
    pair = factor_rank_one_matrix(matrix)
    assert pair is not None
    assert tuple(pair[0][i] * pair[1][j] for i in range(4) for j in range(4)) in (
        tuple(value for row in matrix for value in row), tuple(-value for row in matrix for value in row))
    rank_two = ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0))
    assert low_rank_matrix_factor_proposals(rank_two)


def test_algebraic_generation_is_deterministic_valid_and_budgeted():
    target = matrix_multiplication_tensor(2, 2, 2)
    config = AlgebraicPoolConfig(combination_vectors=12, missing_factors_per_pair=2,
                                 maximum_pair_proposals=10, maximum_exact_scores=800)
    first, audit = algebraic_action_candidates(target, config)
    second, second_audit = algebraic_action_candidates(target, config)
    assert [(row["action_id"], row["provenance"]) for row in first] == [(row["action_id"], row["provenance"]) for row in second]
    assert audit["exact_scoring_operations"] <= 800
    assert audit["pair_proposals"] <= 10
    assert audit["config"] == second_audit["config"]
    assert len({row["action_id"] for row in first}) == len(first)


def test_hybrid_dedup_recall_and_mitm_compatibility():
    conventional = standard_2x2_decomposition()
    oracle = strassen_2x2_decomposition()
    mixed = hybrid_pool(conventional[:4], conventional, 8, 4)
    assert len(mixed) == len(set(mixed)) == 8
    recall = known_action_recall(oracle, oracle)
    assert recall["pair_recall"] == recall["complete_action_recall"] == recall["tensor_equivalence_recall"] == 7
    result = meet_in_middle(matrix_multiplication_tensor(2, 2, 2), oracle, MeetInMiddleConfig(
        maximum_left_attempts=100, maximum_right_attempts=100, maximum_stored_sums=100,
        maximum_seconds=2.0, seed=3))
    assert result["exact_decomposition_count"] == 1


def test_invalid_algebraic_budgets_fail_closed():
    with pytest.raises(ValueError):
        AlgebraicPoolConfig(maximum_exact_scores=0).validate()
