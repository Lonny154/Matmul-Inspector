import pytest

torch = pytest.importorskip("torch")

from matmul_inspector.tensor_game import RankOneAction
from matmul_inspector.tensor_game_policy_interference import (
    architecture_semantics,
    component_log_scores,
    component_ranks,
    counterfactual_segment_logits,
    flattened_gradients,
    gradient_cosine,
    state_dict_digest,
)


ACTION = RankOneAction((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0))


def test_state_digest_detects_and_restores_exact_parameters():
    model = torch.nn.Linear(2, 2); saved = {k: v.clone() for k, v in model.state_dict().items()}; before = state_dict_digest(model)
    with torch.no_grad(): model.weight.add_(1)
    assert state_dict_digest(model) != before
    model.load_state_dict(saved)
    assert state_dict_digest(model) == before


def test_gradient_helpers_are_independent_and_directional():
    value = torch.tensor([1.0, 2.0], requires_grad=True)
    first = flattened_gradients([value], (value**2).sum())
    second = flattened_gradients([value], -(value**2).sum())
    assert torch.equal(first, -second)
    assert gradient_cosine(first, second) == pytest.approx(-1)
    assert value.grad is None


def test_component_scores_and_ranks_follow_independent_logits():
    logits = torch.zeros((12, 3)); values = ACTION.u + ACTION.v + ACTION.w
    for i, token in enumerate(values): logits[i, {-1: 0, 0: 1, 1: 2}[token]] = 5
    scores = component_log_scores(logits, ACTION); ranks = component_ranks(logits, ACTION)
    assert all(score < 0 for score in scores)
    assert ranks == (1, 1, 1)


def test_counterfactual_replacement_is_deterministic_and_local():
    trained = torch.ones((12, 3)); frozen = torch.zeros((12, 3))
    result = counterfactual_segment_logits(trained, frozen, "v")
    assert torch.equal(result[:4], trained[:4]) and torch.equal(result[4:8], frozen[4:8]) and torch.equal(result[8:], trained[8:])
    assert torch.equal(result, counterfactual_segment_logits(trained, frozen, "v"))
    with pytest.raises(ValueError): counterfactual_segment_logits(trained, frozen, "bad")


def test_architecture_audit_records_non_autoregressive_semantics():
    audit = architecture_semantics()
    assert audit["factorization"].startswith("conditionally independent")
    assert audit["autoregressive"] is False and audit["policy_output_shape"] == [12, 3]
