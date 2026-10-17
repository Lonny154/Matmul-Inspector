import json
import random

import pytest

from matmul_inspector.tensor_game import matrix_multiplication_tensor, strassen_2x2_decomposition, subtract_action
from matmul_inspector.tensor_game_bounded_residual import action_identity, bounded_residual_candidates_memory_efficient
from matmul_inspector.tensor_game_diversity_search import DiscoveryConfig, discover
from matmul_inspector.tensor_game_multi_w import multi_w_candidates
from matmul_inspector.tensor_game_quota_w import (
    QuotaMultiWProvider,
    exhaustive_w_vectors,
    quota_multi_w_candidates,
    top_k_w_vectors,
)


@pytest.mark.parametrize("k", (1, 2, 4, 8, 80, 100))
def test_top_k_exactly_matches_exhaustive_oracle(k):
    rng = random.Random(20261017)
    for _ in range(12):
        contraction = tuple(rng.randint(-8, 8) for _ in range(4))
        q = rng.randint(1, 16)
        actual, audit = top_k_w_vectors(contraction, q, k)
        expected = exhaustive_w_vectors(contraction, q)[: min(k, 80)]
        assert actual == expected
        assert audit["coordinate_choice_evaluations"] == 12
        assert audit["complete_vectors_scored"] <= 81


def test_top_k_handles_zero_negative_and_tied_choices_deterministically():
    first, _ = top_k_w_vectors((0, -1, 1, 0), 2, 8)
    second, _ = top_k_w_vectors((0, -1, 1, 0), 2, 8)
    assert first == second
    assert all(any(row["w"]) for row in first)
    assert [(row["score_delta"], row["w"]) for row in first] == sorted(
        (row["score_delta"], row["w"]) for row in first
    )
    limited, audit = top_k_w_vectors((0, 0, 0, 0), 2, 8, complete_score_budget=3)
    assert audit["complete_vectors_scored"] == 3
    assert len(limited) <= 3


def test_twelve_zero_reproduces_baseline_top_twelve_exactly():
    residual = subtract_action(matrix_multiplication_tensor(2, 2, 2), strassen_2x2_decomposition()[0])
    expected, _ = bounded_residual_candidates_memory_efficient(residual, 128, 12)
    actual, audit = quota_multi_w_candidates(
        residual, protected_baseline=12, alternative_slots=0, strategy="residual"
    )
    assert [row["action_id"] for row in actual] == [row["action_id"] for row in expected]
    assert [row["source"] for row in actual] == ["protected_baseline"] * len(expected)
    assert audit["top_k_complete_vectors_scored"] == 0


@pytest.mark.parametrize("protected,alternatives", ((11, 1), (10, 2), (8, 4)))
def test_quota_size_protection_order_deduplication_and_fallback(protected, alternatives):
    residual = subtract_action(matrix_multiplication_tensor(2, 2, 2), strassen_2x2_decomposition()[0])
    baseline, _ = bounded_residual_candidates_memory_efficient(residual, 128, 12)
    rows, _ = quota_multi_w_candidates(
        residual,
        protected_baseline=protected,
        alternative_slots=alternatives,
        w_candidates_per_pair=2,
        strategy="residual",
    )
    protected_count = min(protected, len(baseline))
    assert len(rows) == min(12, len(baseline) + alternatives)
    assert len({row["action_id"] for row in rows}) == len(rows)
    assert [row["action_id"] for row in rows[:protected_count]] == [row["action_id"] for row in baseline[:protected_count]]


def test_fallback_uses_remaining_baseline_when_no_alternative_is_available(monkeypatch):
    import matmul_inspector.tensor_game_quota_w as module

    monkeypatch.setattr(module, "top_k_w_vectors", lambda contraction, q, k, **kwargs: ((), {
        "complete_vectors_scored": 0, "coordinate_choice_evaluations": 0
    }))
    residual = matrix_multiplication_tensor(2, 2, 2)
    for action in strassen_2x2_decomposition()[:3]:
        residual = subtract_action(residual, action)
    baseline, _ = bounded_residual_candidates_memory_efficient(residual, 128, 12)
    rows, audit = quota_multi_w_candidates(
        residual, protected_baseline=10, alternative_slots=2, strategy="residual"
    )
    assert [row["action_id"] for row in rows] == [row["action_id"] for row in baseline]
    assert audit["fallback_baseline_actions"] == 2


def test_quota_improves_exact_strassen_return_recall():
    actions = strassen_2x2_decomposition()
    residual = matrix_multiplication_tensor(2, 2, 2)
    baseline_hits = quota_hits = 0
    for action in actions:
        baseline, _ = bounded_residual_candidates_memory_efficient(residual, 128, 12)
        quota, _ = quota_multi_w_candidates(
            residual, protected_baseline=11, alternative_slots=1, w_candidates_per_pair=2
        )
        expected = action_identity(action)
        baseline_hits += expected in {row["action_id"] for row in baseline}
        quota_hits += expected in {row["action_id"] for row in quota}
        residual = subtract_action(residual, action)
    assert quota_hits > baseline_hits


def test_budget_search_serialization_and_existing_providers_are_unchanged():
    residual = matrix_multiplication_tensor(2, 2, 2)
    baseline_before, _ = bounded_residual_candidates_memory_efficient(residual, 128, 12)
    exhaustive_before, _ = multi_w_candidates(residual, w_candidates_per_pair=2, limit=12)
    provider = QuotaMultiWProvider(10, 2, trace_limit=1)
    result = discover(DiscoveryConfig(7, 3, node_budget=2, beam_width=4, max_actions=12), provider)
    assert result["nodes_expanded"] == 2 and result["budget_exhausted"]
    assert json.dumps(provider.trace, sort_keys=True) == json.dumps(provider.trace, sort_keys=True)
    assert bounded_residual_candidates_memory_efficient(residual, 128, 12)[0] == baseline_before
    assert multi_w_candidates(residual, w_candidates_per_pair=2, limit=12)[0] == exhaustive_before
