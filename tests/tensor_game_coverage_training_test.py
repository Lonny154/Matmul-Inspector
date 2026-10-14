import json

import pytest

from matmul_inspector.tensor_game import RankOneAction, outer_product
from matmul_inspector.tensor_game_coverage_training import (
    canonical_action_log_probabilities,
    aggregate_discovery,
    checkpoint_sha256,
    coverage_loss,
    labels_digest,
    validate_checkpoint_provenance,
    validate_separation,
    verify_artifact_hash,
)
from matmul_inspector.tensor_game_offline_learning import sampled_support_log_probabilities


ACTION = RankOneAction((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0))


def tensors():
    torch = pytest.importorskip("torch")
    values = torch.tensor([[ACTION.u + ACTION.v + ACTION.w]], dtype=torch.long)
    mask = torch.ones((1, 1), dtype=torch.bool)
    return torch, values, mask


def test_coverage_loss_is_finite_and_gradients_flow():
    torch, actions, mask = tensors()
    logits = torch.zeros((1, 12, 3), requires_grad=True)
    loss = coverage_loss(logits, actions, mask)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None and torch.any(logits.grad != 0)


def test_increasing_positive_logits_reduces_coverage_loss():
    torch, actions, mask = tensors()
    baseline = torch.zeros((1, 12, 3))
    improved = baseline.clone()
    classes = {-1: 0, 0: 1, 1: 2}
    for position, value in enumerate(ACTION.u + ACTION.v + ACTION.w):
        improved[0, position, classes[value]] += 2
    assert coverage_loss(improved, actions, mask) < coverage_loss(baseline, actions, mask)


def test_multiple_positives_accumulate_probability_mass():
    torch, actions, mask = tensors()
    second = RankOneAction((0, 1, 0, 0), (1, 0, 0, 0), (0, 0, 0, 1))
    multiple = torch.tensor([[ACTION.u + ACTION.v + ACTION.w, second.u + second.v + second.w]])
    multi_mask = torch.ones((1, 2), dtype=torch.bool)
    logits = torch.zeros((1, 12, 3))
    assert coverage_loss(logits, multiple, multi_mask) < coverage_loss(logits, actions, mask)


def test_empty_positive_set_is_rejected():
    torch, actions, mask = tensors()
    with pytest.raises(ValueError, match="nonempty"):
        coverage_loss(torch.zeros((1, 12, 3)), actions, torch.zeros_like(mask))


def test_mcts_support_loss_path_is_unchanged():
    torch, actions, mask = tensors()
    logits = torch.randn((1, 12, 3), generator=torch.Generator().manual_seed(3))
    before = sampled_support_log_probabilities(logits, actions, mask, (4, 4, 4))
    canonical_action_log_probabilities(logits, actions)
    after = sampled_support_log_probabilities(logits, actions, mask, (4, 4, 4))
    assert torch.equal(before, after)


def test_tiny_positive_set_can_be_overfit():
    torch, actions, mask = tensors()
    logits = torch.nn.Parameter(torch.zeros((1, 12, 3)))
    optimizer = torch.optim.SGD([logits], lr=0.5)
    initial = float(coverage_loss(logits, actions, mask).detach())
    for _ in range(20):
        optimizer.zero_grad(); loss = coverage_loss(logits, actions, mask); loss.backward(); optimizer.step()
    assert float(coverage_loss(logits, actions, mask).detach()) < initial * 0.2


def test_dataset_separation_and_label_digest_are_deterministic():
    label = {"state_id": "s", "target_digest": "train", "split": "train", "positive_actions": [{"action_id": "a"}]}
    reserved = {"targets": [{"target_digest": "heldout"}]}
    assert validate_separation([label], [reserved])["target_overlap"] == 0
    assert labels_digest([label]) == labels_digest(json.loads(json.dumps([label])))
    with pytest.raises(ValueError, match="leakage"):
        validate_separation([{**label, "target_digest": "heldout"}], [reserved])


def test_evaluation_aggregation_keeps_targets_and_seeds_separate():
    rows = [
        {"target_id": "a", "seed": 3, "success": True},
        {"target_id": "a", "seed": 19, "success": True},
        {"target_id": "b", "seed": 3, "success": False},
    ]
    assert aggregate_discovery(rows) == {
        "trials": 3,
        "trial_successes": 2,
        "unique_targets": 2,
        "targets_solved": 1,
    }


def test_checkpoint_provenance_and_artifact_integrity(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.write_bytes(b"weights")
    digest = checkpoint_sha256(checkpoint)
    validate_checkpoint_provenance(
        {
            "source_checkpoint_sha256": digest,
            "checkpoint_sha256": "candidate",
            "fixed_epochs_completed": 30,
        },
        digest,
    )
    verify_artifact_hash(checkpoint, digest)
    with pytest.raises(ValueError, match="provenance"):
        validate_checkpoint_provenance({}, digest)
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_artifact_hash(checkpoint, "bad")
