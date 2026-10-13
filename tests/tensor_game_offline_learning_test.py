"""CPU tests for TG-4B.1 offline learning from MCTS experience."""

import copy
import tempfile
from pathlib import Path
import unittest

from matmul_inspector.tensor_game_learning import PolicyValueConfig, create_policy_value_model
from matmul_inspector.tensor_game_offline_learning import (
    ExperienceDataset,
    OfflineTrainingConfig,
    collate_experience,
    configure_trainable,
    grouped_splits,
    load_offline_checkpoint,
    offline_loss,
    sampled_support_log_probabilities,
    train_offline,
)


ROOT = Path(__file__).resolve().parents[1]
EXPERIENCE = ROOT / "artifacts/tensor-game/tg4a-experience-v1"
SOURCE = ROOT / "artifacts/tensor-game/tg3b-mlp-256-seed42"


class OfflineLearningTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("PyTorch is optional")
        cls.torch = torch

    def test_validated_adapter_and_grouped_split_isolation(self):
        datasets = {name: ExperienceDataset(EXPERIENCE, name) for name in ("train", "validation", "test")}
        self.assertEqual(sum(map(len, datasets.values())), 42)
        episodes = [{row["episode_id"] for row in data.rows} for data in datasets.values()]
        targets = [{data[index]["target_id"] for index in range(len(data))} for data in datasets.values()]
        residuals = [{row["state_digest"] for row in data.rows} for data in datasets.values()]
        for collections in (episodes, targets, residuals):
            self.assertFalse(collections[0] & collections[1])
            self.assertFalse(collections[0] & collections[2])
            self.assertFalse(collections[1] & collections[2])
        self.assertIn("small replay buffer", datasets["train"].split_report["warning"])
        item = datasets["train"][0]
        self.assertEqual(item["actions"].shape[0], item["search_policy"].shape[0])
        self.assertIn("checkpoint_sha256", item["provenance"])

    def test_grouped_split_rejects_repeated_residual_across_targets(self):
        dataset = ExperienceDataset(EXPERIENCE, "train")
        buffer = copy.deepcopy(dataset.buffer)
        target_by_episode = {episode["episode_id"]: episode["initial_target_digest"] for episode in buffer.episodes}
        first = next(iter(buffer.decisions.values()))
        other = next(row for row in buffer.decisions.values() if target_by_episode[row["episode_id"]] != target_by_episode[first["episode_id"]])
        other["state_digest"] = first["state_digest"]
        with self.assertRaisesRegex(ValueError, "residual tensors cross"):
            grouped_splits(buffer)

    def test_variable_support_normalization_loss_masks_and_gradients(self):
        torch = self.torch
        shape = (1, 2, 1)
        logits = torch.zeros((2, sum(shape), 3), requires_grad=True)
        actions = torch.tensor([
            [[1, 1, 0, 1], [1, 0, 1, 1], [0, 0, 0, 0]],
            [[1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]],
        ])
        mask = torch.tensor([[True, True, False], [True, False, False]])
        logp = sampled_support_log_probabilities(logits, actions, mask, shape)
        self.assertTrue(torch.allclose(torch.exp(logp[0, :2]).sum(), torch.tensor(1.0)))
        self.assertEqual(float(torch.exp(logp[1, 0]).detach()), 1.0)
        batch = {
            "actions": actions, "action_mask": mask,
            "search_policy": torch.tensor([[0.25, 0.75, 0.0], [1.0, 0.0, 0.0]]),
            "value_target": torch.tensor([-2.0, 99.0]),
            "value_valid": torch.tensor([True, False]),
        }
        values = torch.tensor([0.0, 1000.0], requires_grad=True)
        loss, policy, value = offline_loss(logits, values, batch, shape)
        self.assertAlmostEqual(float(value.detach()), 4.0)
        self.assertGreater(float(policy.detach()), 0.0)
        loss.backward()
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)
        self.assertEqual(float(values.grad[1]), 0.0)

    def test_duplicate_sign_equivalent_actions_and_empty_support_rejected(self):
        torch = self.torch
        logits = torch.zeros((1, 3, 3))
        equivalent = torch.tensor([[[1, 1, 1], [-1, -1, 1]]])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            sampled_support_log_probabilities(logits, equivalent, torch.ones((1, 2), dtype=torch.bool), (1, 1, 1))
        with self.assertRaises(ValueError):
            sampled_support_log_probabilities(logits, equivalent, torch.zeros((1, 2), dtype=torch.bool), (1, 1, 1))

    def test_collation_and_training_modes(self):
        data = ExperienceDataset(EXPERIENCE, "train")
        batch = collate_experience([data[0], data[1]])
        self.assertEqual(batch["actions"].shape[:2], batch["action_mask"].shape)
        self.assertEqual(batch["visits"].shape, batch["action_mask"].shape)
        self.assertEqual(len(batch["provenance"]), 2)
        expected = {
            "policy_only": (True, False),
            "value_only": (False, True),
            "joint": (True, True),
        }
        for mode, (policy, value) in expected.items():
            model = create_policy_value_model(PolicyValueConfig())
            names = configure_trainable(model, mode, False)
            self.assertEqual(any(name.startswith("policy_head") for name in names), policy)
            self.assertEqual(any(name.startswith("value_head") for name in names), value)
            self.assertTrue(any(name.startswith("features") for name in names))
        model = create_policy_value_model(PolicyValueConfig())
        names = configure_trainable(model, "joint", True)
        self.assertFalse(any(name.startswith("features") for name in names))

    def test_deterministic_checkpoint_save_reload_and_provenance(self):
        manifests = []
        states = []
        with tempfile.TemporaryDirectory() as directory:
            for run in range(2):
                output = Path(directory) / f"run-{run}"
                manifest = train_offline(
                    EXPERIENCE, SOURCE, output,
                    OfflineTrainingConfig(epochs=1, batch_size=16, seed=17, mode="joint"),
                )
                model, config, loaded = load_offline_checkpoint(output)
                manifests.append(manifest)
                states.append({name: value.clone() for name, value in model.state_dict().items()})
                self.assertEqual(config.shape, (4, 4, 4))
                self.assertEqual(loaded["value_head_semantics"], "direct_observed_mcts_return_not_completion_length")
                self.assertEqual(loaded["effective_samples"]["train"], len(ExperienceDataset(EXPERIENCE, "train")))
                self.assertEqual(loaded["training_config"]["seed"], 17)
            self.assertEqual(manifests[0]["history"], manifests[1]["history"])
            for name in states[0]:
                self.assertTrue(self.torch.equal(states[0][name], states[1][name]))

    def test_policy_only_and_value_only_training(self):
        with tempfile.TemporaryDirectory() as directory:
            for mode, expected in (("policy_only", (1.0, 0.0)), ("value_only", (0.0, 1.0))):
                manifest = train_offline(
                    EXPERIENCE, SOURCE, Path(directory) / mode,
                    OfflineTrainingConfig(epochs=1, batch_size=32, seed=23, mode=mode),
                )
                coefficients = manifest["effective_loss_coefficients"]
                self.assertEqual((coefficients["policy"], coefficients["value"]), expected)
                frozen_head = "value_head" if mode == "policy_only" else "policy_head"
                self.assertFalse(any(name.startswith(frozen_head) for name in manifest["trainable_parameters"]))
                self.assertTrue(manifest["shared_torso_caveat"])


if __name__ == "__main__":
    unittest.main()
