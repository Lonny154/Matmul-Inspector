import json

import pytest

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
    squared_norm,
)
from matmul_inspector.tensor_game_candidate_dataset import validate_decomposition
from matmul_inspector.tensor_game_diversity_search import DiscoveryConfig, discover
from matmul_inspector.tensor_game_multi_w import (
    MultiWProvider,
    W_VECTORS,
    alternative_w_vectors,
    multi_w_candidates,
)


def test_original_w_is_first_and_alternatives_handle_signs_and_zeroes():
    rows = alternative_w_vectors((2, -2, 0, 1), 2, 4, "residual")
    assert rows[0]["source"] == "original"
    assert rows[0]["w"] == (1, -1, 0, 0)
    assert len(rows) == len({row["w"] for row in rows}) == 4
    assert all(row["w"] in W_VECTORS for row in rows)


def test_multi_w_deduplicates_scores_orders_and_enforces_budget():
    residual = matrix_multiplication_tensor(2, 2, 2)
    first, audit = multi_w_candidates(
        residual, w_candidates_per_pair=4, strategy="residual", complete_action_budget=17, limit=30
    )
    second, _ = multi_w_candidates(
        residual, w_candidates_per_pair=4, strategy="residual", complete_action_budget=17, limit=30
    )
    assert first == second
    assert len(first) == len({row["action_id"] for row in first}) == 17
    assert [row["score"] for row in first] == sorted(row["score"] for row in first)
    assert audit["unique_complete_actions_after_budget"] == 17
    assert audit["complete_actions_scored_for_w_ranking"] == 128 * 80
    for row in first:
        action = action_from_payload(row["action"])
        assert row["score"] == squared_norm(subtract_action(residual, action))


def test_known_strassen_recall_improves_at_width_128():
    actions = strassen_2x2_decomposition()
    residual = matrix_multiplication_tensor(2, 2, 2)
    baseline_hits = multi_hits = 0
    for action in actions:
        baseline, _ = bounded_residual_candidates_memory_efficient(residual, 128, 1024)
        expanded, _ = multi_w_candidates(
            residual, w_candidates_per_pair=2, strategy="residual", complete_action_budget=256, limit=256
        )
        expected = action_identity(action)
        baseline_hits += expected in {row["action_id"] for row in baseline}
        multi_hits += expected in {row["action_id"] for row in expanded}
        residual = subtract_action(residual, action)
    assert multi_hits > baseline_hits


def test_diversity_strategy_is_stable_and_keeps_residual_quota():
    first = alternative_w_vectors((3, -1, 2, 0), 4, 8, "diversity")
    second = alternative_w_vectors((3, -1, 2, 0), 4, 8, "diversity")
    assert first == second
    assert any(row["source"] == "alternative_diversity_quota" for row in first)
    assert any(row["source"] == "alternative_diversity_residual_quota" for row in first)


def test_search_node_budget_serialization_and_exact_completion():
    provider = MultiWProvider(w_candidates_per_pair=2, strategy="residual", trace_limit=1)
    result = discover(DiscoveryConfig(7, 3, node_budget=2, beam_width=4, max_actions=2), provider)
    assert result["nodes_expanded"] == 2
    assert result["budget_exhausted"]
    assert json.dumps(provider.trace, sort_keys=True) == json.dumps(provider.trace, sort_keys=True)
    validate_decomposition(standard_2x2_decomposition())


def test_default_bounded_provider_is_unchanged():
    residual = matrix_multiplication_tensor(2, 2, 2)
    before, _ = bounded_residual_candidates_memory_efficient(residual, 128, 12)
    MultiWProvider()(residual, 12)
    after, _ = bounded_residual_candidates_memory_efficient(residual, 128, 12)
    assert before == after


@pytest.mark.parametrize("count", (0, 81))
def test_invalid_w_count_is_rejected(count):
    with pytest.raises(ValueError):
        alternative_w_vectors((1, 0, 0, 0), 1, count, "residual")
