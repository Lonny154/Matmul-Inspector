import pytest

from matmul_inspector.tensor_game import RankOneAction

torch = pytest.importorskip("torch")

from matmul_inspector.tensor_game_final_ranking import (  # noqa: E402
    complete_action_score,
    final_ranking_loss,
    select_hard_negative,
)


def action(values):
    return RankOneAction(values[:4], values[4:8], values[8:])


POSITIVE = action((1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0))


def tensors():
    logits = torch.zeros((1, 12, 3), requires_grad=True)
    actions = torch.tensor([[POSITIVE.u + POSITIVE.v + POSITIVE.w]])
    mask = torch.tensor([[True]])
    return logits, actions, mask


def test_complete_score_matches_manual_best_sign_form():
    logits, _, _ = tensors()
    logits.data[:, :, 2] = 1
    indices = torch.tensor([2 if value == 1 else 1 for value in POSITIVE.u + POSITIVE.v + POSITIVE.w])
    expected = torch.log_softmax(logits[0], -1).gather(1, indices[:, None]).sum()
    assert torch.allclose(complete_action_score(logits[0], POSITIVE), expected)


def test_hard_negative_selection_is_deterministic_and_valid():
    logits, _, _ = tensors()
    first = select_hard_negative(logits[0])
    assert first == select_hard_negative(logits[0])
    assert any(first.u) and any(first.v) and any(first.w)


def test_loss_is_finite_and_backpropagates():
    logits, actions, mask = tensors()
    loss = final_ranking_loss(logits, actions, mask)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


def test_update_improves_selected_positive_margin():
    logits, actions, mask = tensors()
    negative = select_hard_negative(logits[0])
    before = complete_action_score(logits[0], POSITIVE) - complete_action_score(logits[0], negative)
    final_ranking_loss(logits, actions, mask).backward()
    with torch.no_grad():
        logits -= 0.1 * logits.grad
    after = complete_action_score(logits[0], POSITIVE) - complete_action_score(logits[0], negative)
    assert after > before


def test_margin_direction_zero_and_input_validation(monkeypatch):
    logits, actions, mask = tensors()
    monkeypatch.setattr(
        "matmul_inspector.tensor_game_final_ranking.select_hard_negative",
        lambda *_args, **_kwargs: POSITIVE,
    )
    assert final_ranking_loss(logits, actions, mask, margin=0).item() == 0
    with pytest.raises(ValueError):
        final_ranking_loss(logits, actions, torch.tensor([[False]]))
    invalid = actions.clone(); invalid[0, 0, :4] = 0
    with pytest.raises(ValueError):
        final_ranking_loss(logits, invalid, mask)
