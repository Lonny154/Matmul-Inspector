import pytest

from matmul_inspector.tensor_game import RankOneAction
from matmul_inspector.tensor_game_coverage_training import (
    first_decoder_pruning_position,
    per_positive_coverage_loss,
    selective_retention_kl,
)


A = RankOneAction((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0))
B = RankOneAction((0, 1, 0, 0), (1, 0, 0, 0), (0, 0, 0, 1))


def test_per_positive_loss_protects_each_positive_and_gradients_flow():
    torch = pytest.importorskip("torch")
    actions = torch.tensor([[A.u + A.v + A.w, B.u + B.v + B.w]])
    mask = torch.ones((1, 2), dtype=torch.bool)
    logits = torch.zeros((1, 12, 3), requires_grad=True)
    loss = per_positive_coverage_loss(logits, actions, mask)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.any(logits.grad != 0)


def test_per_positive_loss_rejects_empty_set():
    torch = pytest.importorskip("torch")
    actions = torch.tensor([[A.u + A.v + A.w]])
    with pytest.raises(ValueError, match="nonempty"):
        per_positive_coverage_loss(torch.zeros((1, 12, 3)), actions, torch.zeros((1, 1), dtype=torch.bool))


def test_retention_has_no_teacher_gradient_and_zeroes_unselected_states():
    torch = pytest.importorskip("torch")
    student = torch.randn((2, 12, 3), requires_grad=True)
    teacher = torch.randn((2, 12, 3))
    loss = selective_retention_kl(student, teacher, torch.tensor([True, False]))
    loss.backward()
    assert student.grad[0].abs().sum() > 0
    assert student.grad[1].abs().sum() == 0
    assert teacher.grad is None


def test_retention_rejects_attached_teacher():
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="detached"):
        selective_retention_kl(
            torch.zeros((1, 12, 3)),
            torch.zeros((1, 12, 3), requires_grad=True),
            torch.tensor([True]),
        )


def test_decoder_pruning_reports_bounded_position():
    torch = pytest.importorskip("torch")
    logits = torch.zeros((12, 3))
    position = first_decoder_pruning_position(logits, A, beam_width=1)
    assert position is not None and 1 <= position <= 12
