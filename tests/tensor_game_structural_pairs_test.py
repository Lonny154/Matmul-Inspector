import json

from matmul_inspector.tensor_game import matrix_multiplication_tensor, strassen_2x2_decomposition, subtract_action
from matmul_inspector.tensor_game_bounded_residual import (
    coarse_pair_scores,
    exact_score_from_contraction,
    squared_norm,
)
from matmul_inspector.tensor_game_diversity_search import DiscoveryConfig, discover
from matmul_inspector.tensor_game_structural_pairs import (
    StructuralPairProvider,
    pair_structural_features,
    select_structural_pairs,
    structural_pair_candidates,
    structural_signature,
)


def target():
    return matrix_multiplication_tensor(2, 2, 2)


def test_structural_features_are_deterministic_and_sign_invariant():
    pair = coarse_pair_scores(target())[2000]
    inverse = type(pair)(tuple(-x for x in pair.u), pair.v, pair.q,
                         tuple(-x for x in pair.contraction), pair.bound_numerator)
    first = pair_structural_features(pair)
    second = pair_structural_features(pair)
    assert first == second
    assert structural_signature(first) == structural_signature(pair_structural_features(inverse))


def test_128_zero_exactly_reproduces_original_pair_prefix():
    expected = coarse_pair_scores(target())[:128]
    actual, audit = select_structural_pairs(
        target(), protected_residual_pairs=128, structural_pairs=0
    )
    assert list(actual) == expected
    assert audit["selected_original_ranks"] == list(range(1, 129))


def test_protected_order_quota_uniqueness_count_and_determinism():
    first, audit = select_structural_pairs(target(), protected_residual_pairs=112, structural_pairs=16)
    second, second_audit = select_structural_pairs(target(), protected_residual_pairs=112, structural_pairs=16)
    original = coarse_pair_scores(target())
    assert first == second and audit["selected_original_ranks"] == second_audit["selected_original_ranks"]
    assert list(first[:112]) == original[:112]
    assert len(first) == len({(pair.u, pair.v) for pair in first}) == 128
    assert audit["structural_pairs"] == 16 and audit["fallback_pairs"] == 0


def test_structural_signature_fallback_fills_budget(monkeypatch):
    import matmul_inspector.tensor_game_structural_pairs as module

    monkeypatch.setattr(module, "structural_signature", lambda features: (0,))
    selected, audit = module.select_structural_pairs(
        target(), protected_residual_pairs=120, structural_pairs=8
    )
    assert len(selected) == 128
    assert audit["structural_pairs"] == 1 and audit["fallback_pairs"] == 7


def test_top_k_compatibility_and_exact_scores_are_unchanged():
    rows, audit = structural_pair_candidates(
        target(), protected_residual_pairs=112, structural_pairs=16,
        w_mode="top_k", w_candidates_per_pair=2, limit=256,
    )
    assert audit["pair_selection"]["selected_original_ranks"][:112] == list(range(1, 113))
    residual_norm = squared_norm(target())
    for row in rows:
        action = row["action"]
        assert row["score"] == squared_norm(subtract_action(target(), action))
        assert row["residual_reduction"] == residual_norm - row["score"]


def test_provider_serialization_and_default_search_behavior_remain_stable():
    provider = StructuralPairProvider(120, 8, trace_limit=0)
    first = provider(target(), 12)
    second = StructuralPairProvider(120, 8, trace_limit=0)(target(), 12)
    assert first == second
    assert json.dumps(provider.diagnostic_summary(), sort_keys=True)
    baseline = discover(DiscoveryConfig(8, 3, node_budget=2, beam_width=4, max_actions=2))
    assert baseline["config"]["admission_policy"] == "global"


def test_known_pair_coverage_report_is_exact_not_structural_similarity():
    selected, _ = select_structural_pairs(target(), protected_residual_pairs=128, structural_pairs=0)
    keys = {(pair.u, pair.v) for pair in selected}
    action = strassen_2x2_decomposition()[5]
    exact = (action.u, action.v) in keys
    assert exact == ((action.u, action.v) in keys)
