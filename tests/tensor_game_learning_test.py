"""Optional CPU tests for TG-3B neural policy/value learning."""

import builtins
import importlib
import inspect
import math
from pathlib import Path
import tempfile
import unittest

from matmul_inspector.tensor_game_dataset import SyntheticDatasetConfig, generate_dataset
from matmul_inspector import tensor_game_learning as learning

try:
    import torch
except ImportError:
    torch = None


class ImportIsolationTest(unittest.TestCase):
    def test_module_import_does_not_require_torch(self):
        original = builtins.__import__

        def reject_torch(name, *args, **kwargs):
            if name == "torch":
                raise AssertionError("module import attempted to load torch")
            return original(name, *args, **kwargs)

        builtins.__import__ = reject_torch
        try:
            importlib.reload(learning)
        finally:
            builtins.__import__ = original


@unittest.skipUnless(torch is not None, "optional PyTorch is not installed")
class LearningTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.config = learning.PolicyValueConfig(shape=(2, 2, 2), hidden_dim=16, input_scale=2.0)

    def test_model_shapes_mapping_and_fixed_seed(self):
        torch.manual_seed(4)
        first = learning.create_policy_value_model(self.config)
        torch.manual_seed(4)
        second = learning.create_policy_value_model(self.config)
        residual = torch.zeros((3, 2, 2, 2))
        logits, value = first(residual)
        self.assertEqual(tuple(logits.shape), (3, 6, 3))
        self.assertEqual(tuple(value.shape), (3,))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(first.parameters(), second.parameters())))
        encoded = learning.encode_action_coefficients(torch.tensor([[-1, 0, 1]]))
        self.assertTrue(torch.equal(encoded, torch.tensor([[0, 1, 2]])))

    def test_joint_probability_multi_action_mask_and_deduplication(self):
        logits = torch.zeros((1, 6, 3))
        first = torch.tensor([-1, 0, 1, -1, 0, 1])
        second = torch.tensor([1, 0, -1, 1, 0, -1])
        targets = torch.stack((first, second, first)).unsqueeze(0)
        mask = torch.tensor([[True, True, True]])
        log_mass = learning.valid_action_log_probabilities(logits, targets, mask)
        self.assertAlmostEqual(float(log_mass), math.log(2) - 6 * math.log(3), places=5)
        padded = targets.clone()
        padded[0, 2] = 99
        masked = torch.tensor([[True, True, False]])
        self.assertAlmostEqual(
            float(learning.valid_action_log_probabilities(logits, padded, masked)),
            math.log(2) - 6 * math.log(3),
            places=5,
        )
        with self.assertRaisesRegex(ValueError, "at least one"):
            learning.valid_action_log_probabilities(logits, targets, torch.zeros_like(mask))

    def test_nonzero_canonical_inference_and_training_gradients(self):
        logits = torch.zeros((6, 3))
        ranked = learning.rank_complete_actions(logits, (2, 2, 2), 5)
        self.assertEqual(len(ranked), 5)
        for action, _ in ranked:
            self.assertTrue(all(any(vector) for vector in (action.u, action.v, action.w)))
            self.assertEqual(action, learning.canonicalize_action(action))
        model = learning.create_policy_value_model(self.config)
        residual = torch.randn((2, 2, 2, 2))
        policy_logits, values = model(residual)
        targets = torch.tensor(
            [[[-1, 0, -1, 0, -1, 0]], [[0, 1, 0, 1, 0, 1]]], dtype=torch.int64
        )
        mask = torch.ones((2, 1), dtype=torch.bool)
        completion = torch.tensor([1, 2])
        loss, policy, value = learning.policy_value_loss(
            policy_logits, values, targets, mask, completion, 0.5
        )
        loss.backward()
        self.assertTrue(math.isfinite(float(loss.detach())))
        self.assertGreater(float(policy.detach()), 0)
        self.assertGreaterEqual(float(value.detach()), 0)
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_checkpoint_reload_and_heldout_evaluation(self):
        dataset = generate_dataset(
            SyntheticDatasetConfig(
                shape=(2, 2, 2), construction_lengths=(1, 2), decompositions_per_length=5, seed=13
            )
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "experiment"
            manifest = learning.train_model(
                dataset,
                output,
                self.config,
                learning.TrainingConfig(
                    epochs=2, batch_size=4, learning_rate=0.001, patience=2, seed=5, device="cpu"
                ),
            )
            model, config, loaded = learning.load_checkpoint(output)
            self.assertEqual(config, self.config)
            self.assertEqual(loaded["best_epoch"], manifest["best_epoch"])
            metrics = learning.evaluate_model(model, dataset, "test", config, 4, torch.device("cpu"))
            self.assertEqual(metrics["examples"], sum(row["split"] == "test" for row in dataset.samples))
            comparison = learning.compare_proposals(
                model, dataset, "test", config, proposal_count=3, seeds=(2,), device=torch.device("cpu")
            )
            self.assertEqual(set(comparison), {"learned", "uniform_sampled", "residual_informed", "hybrid"})

    def test_policy_inference_has_no_reference_access(self):
        source = inspect.getsource(learning.rank_complete_actions) + inspect.getsource(learning.compare_proposals)
        self.assertNotIn("strassen", source.lower())
        self.assertNotIn("standard_2x2", source.lower())
        self.assertNotIn("ReferenceGuidedPolicy", source)


if __name__ == "__main__":
    unittest.main()
