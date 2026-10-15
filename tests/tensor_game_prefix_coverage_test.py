import pytest

from matmul_inspector.tensor_game import RankOneAction
from matmul_inspector.tensor_game_learning import rank_complete_actions
from matmul_inspector.tensor_game_prefix_coverage import (
    compile_positive_sign_forms,
    prefix_survival_loss,
    sign_forms,
    trace_positive_prefixes,
)


A = RankOneAction((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0))
B = RankOneAction((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 1))


def batch(actions):
    torch = pytest.importorskip("torch")
    values = torch.tensor([[a.u + a.v + a.w for a in actions]])
    return torch, values, torch.ones((1, len(actions)), dtype=torch.bool)


def test_trace_matches_production_decoder_membership():
    torch, _, _ = batch([A])
    logits = torch.randn((12, 3), generator=torch.Generator().manual_seed(7))
    traces = trace_positive_prefixes(logits, [A], requested_actions=12)
    decoded = {action for action, _ in rank_complete_actions(logits, (4, 4, 4), 12)}
    assert traces[-1]["survives"] == any(
        tuple(action.u + action.v + action.w) in sign_forms(A) for action in decoded
    )
    assert all(row["beam_width"] == 384 for row in traces)


def test_prefix_loss_is_finite_and_gradients_flow():
    torch, actions, mask = batch([A, B])
    logits = torch.zeros((1, 12, 3), requires_grad=True)
    loss = prefix_survival_loss(logits, actions, mask)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None and torch.any(logits.grad != 0)


def test_shared_prefixes_are_deduplicated():
    torch, actions, mask = batch([A, A])
    logits = torch.randn((1, 12, 3), generator=torch.Generator().manual_seed(2))
    duplicate = prefix_survival_loss(logits, actions, mask)
    single = prefix_survival_loss(logits, actions[:, :1], mask[:, :1])
    assert torch.allclose(duplicate, single)


def test_empty_positive_set_and_negative_margin_are_rejected():
    torch, actions, mask = batch([A])
    with pytest.raises(ValueError, match="nonempty"):
        prefix_survival_loss(torch.zeros((1, 12, 3)), actions, torch.zeros_like(mask))
    with pytest.raises(ValueError, match="invalid"):
        prefix_survival_loss(torch.zeros((1, 12, 3)), actions, mask, margin=-1)


def test_safe_positive_has_reduced_penalty():
    torch, actions, mask = batch([A])
    neutral = torch.zeros((1, 12, 3))
    favored = neutral.clone()
    classes = {-1: 0, 0: 1, 1: 2}
    for position, token in enumerate(A.u + A.v + A.w):
        favored[0, position, classes[token]] = 5
    assert prefix_survival_loss(favored, actions, mask) < prefix_survival_loss(neutral, actions, mask)


def test_compiled_targets_preserve_loss_and_gradients():
    torch, actions, mask = batch([A, B])
    compiled = compile_positive_sign_forms(actions, mask)
    first = torch.randn((1, 12, 3), generator=torch.Generator().manual_seed(11), requires_grad=True)
    second = first.detach().clone().requires_grad_(True)
    original = prefix_survival_loss(first, actions, mask)
    cached = prefix_survival_loss(second, actions, mask, compiled_forms=compiled)
    original.backward(); cached.backward()
    assert torch.equal(original, cached)
    assert torch.equal(first.grad, second.grad)
