import pytest

from matmul_inspector.tensor_game import (
    RankOneAction, matrix_multiplication_tensor, outer_product,
    standard_2x2_decomposition, strassen_2x2_decomposition, sum_actions,
)
from matmul_inspector.tensor_game_meet_in_middle import (
    MeetInMiddleConfig, action_domain_complexity, add_tensors, canonical_action_pool,
    meet_in_middle, partial_sum_identity, subtract_tensors,
)


def config(**changes):
    values = dict(left_depth=3, right_depth=4, maximum_left_attempts=100,
                  maximum_right_attempts=100, maximum_stored_sums=100,
                  maximum_seconds=5.0, seed=17)
    values.update(changes)
    return MeetInMiddleConfig(**values)


def test_exact_tensor_addition_subtraction_and_stable_identity():
    first, second = standard_2x2_decomposition()[:2]
    left, right = outer_product(first), outer_product(second)
    combined = add_tensors(left, right)
    assert combined == sum_actions((first, second), (4, 4, 4))
    assert subtract_tensors(combined, right) == left
    assert partial_sum_identity(combined) == partial_sum_identity(add_tensors(right, left))


def test_canonical_pool_removes_sign_equivalent_factorizations():
    action = RankOneAction((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0))
    equivalent = RankOneAction((-1, 0, 0, 0), (0, -1, 0, 0), (0, 0, 1, 0))
    assert len(canonical_action_pool((action, equivalent))) == 1


def test_oracle_complement_matching_deduplicates_partitions_and_reconstructs():
    result = meet_in_middle(matrix_multiplication_tensor(2, 2, 2), strassen_2x2_decomposition(), config())
    assert result["left"]["mode"] == result["right"]["mode"] == "complete"
    assert result["complement_matches"] == 35
    assert result["exact_decomposition_count"] == 1
    assert result["decompositions"][0]["exact_reconstruction"]


def test_duplicate_partial_sums_are_counted_as_collisions():
    def signed(index, sign=1):
        vector = [0] * 4; vector[index] = sign
        return RankOneAction(vector, (1, 0, 0, 0), (1, 0, 0, 0))
    pool = (signed(0), signed(0, -1), signed(1), signed(1, -1))
    target = sum_actions(pool, (4, 4, 4))
    result = meet_in_middle(target, pool, config(left_depth=2, right_depth=2))
    assert result["left"]["duplicate_sums"] >= 1
    assert result["exact_decomposition_count"] == 1


def test_budgets_and_sampling_are_deterministic():
    pool = standard_2x2_decomposition()
    limited = config(left_depth=2, right_depth=2, maximum_left_attempts=3,
                     maximum_right_attempts=3, maximum_stored_sums=3)
    one = meet_in_middle(matrix_multiplication_tensor(2, 2, 2), pool, limited)
    two = meet_in_middle(matrix_multiplication_tensor(2, 2, 2), pool, limited)
    for result in (one, two):
        assert result["left"]["mode"] == result["right"]["mode"] == "sampled"
        assert result["left"]["attempts"] <= 3 and result["right"]["attempts"] <= 3
        assert result["left"]["unique_sums_stored"] <= 3
    comparable = lambda row: {key: value for key, value in row.items() if key != "runtime_seconds"}
    assert comparable(one) == comparable(two)


def test_invalid_budgets_fail_closed_and_domain_counts_are_exact():
    with pytest.raises(ValueError):
        config(maximum_left_attempts=0).validate()
    audit = action_domain_complexity()
    assert audit["raw_factor_triples"] == 512_000
    assert audit["unique_rank_one_tensors"] == 128_000
    assert audit["ordered_two_term_sums"] == 16_384_000_000
