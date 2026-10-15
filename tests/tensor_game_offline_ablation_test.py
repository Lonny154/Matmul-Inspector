"""CPU tests for TG-4B.2B controlled offline-learning ablations."""

import copy
import json
import tempfile
from pathlib import Path
import unittest

from matmul_inspector.tensor_game_learning import PolicyValueConfig, create_policy_value_model
from matmul_inspector.tensor_game_learning import load_checkpoint
from matmul_inspector.tensor_game_offline_ablation import (
    CANDIDATES,
    FROZEN_VALUE_SEMANTICS,
    VALUE_SEMANTICS,
    create_evaluation_plan,
    integrity_audit,
    load_ablation_checkpoint,
    outcome_balance_weights,
    source_evidence_digests,
    train_candidate,
    validate_evaluation_plan,
)
from matmul_inspector.tensor_game_offline_learning import OfflineTrainingConfig, configure_trainable


ROOT = Path(__file__).resolve().parents[1]
EXPERIENCE = ROOT / "artifacts/tensor-game/tg4b2a-diverse-full-v1"
SOURCE = ROOT / "artifacts/tensor-game/tg3b-mlp-256-seed42"


class FakeDataset:
    def __init__(self, outcomes):
        self.rows = [{"episode_id": f"e{index}"} for index in range(len(outcomes))]
        self.episode_by_id = {
            f"e{index}": {"outcome": outcome} for index, outcome in enumerate(outcomes)
        }

    def __len__(self):
        return len(self.rows)


class OfflineAblationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("PyTorch is optional")
        cls.torch = torch

    def test_integrity_audit_preserves_groups_and_never_scores_test(self):
        audit = integrity_audit(EXPERIENCE)
        self.assertTrue(audit["valid"])
        self.assertEqual(audit["persisted_target_counts"], {"train": 140, "validation": 30, "test": 30})
        self.assertEqual((audit["splits"]["train"]["eligible_policy_decisions"], audit["splits"]["validation"]["eligible_policy_decisions"]), (1400, 292))
        self.assertEqual((audit["splits"]["train"]["unique_residual_states"], audit["splits"]["validation"]["unique_residual_states"]), (687, 137))
        self.assertFalse(audit["held_out_metrics_accessed"])
        self.assertNotIn("test", audit["splits"])

    def test_balancing_is_deterministic_capped_and_preserves_records(self):
        dataset = FakeDataset(["success"] + ["truncated"] * 99)
        first, report = outcome_balance_weights(dataset, cap_ratio=10.0)
        second, _ = outcome_balance_weights(dataset, cap_ratio=10.0)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 100)
        self.assertLessEqual(max(first) / min(first), 10.0 + 1e-12)
        self.assertAlmostEqual(sum(first) / len(first), 1.0)
        self.assertTrue(report["repeated_states_preserved"])
        self.assertEqual(report["validation_weighting"], "none")

    def test_strict_freezing_matches_ablation_definitions(self):
        for name, definition in CANDIDATES.items():
            model = create_policy_value_model(PolicyValueConfig())
            names = configure_trainable(model, definition["mode"], definition["strict_torso_freeze"])
            if definition["mode"] in {"policy_only", "value_only"}:
                self.assertFalse(any(value.startswith("features") for value in names), name)
            if definition["mode"] == "policy_only":
                self.assertTrue(all(value.startswith("policy_head") for value in names))
            if definition["mode"] == "value_only":
                self.assertTrue(all(value.startswith("value_head") for value in names))

    def test_source_initialization_is_identical(self):
        first, _, _ = load_checkpoint(SOURCE, "cpu")
        second, _, _ = load_checkpoint(SOURCE, "cpu")
        for name, value in first.state_dict().items():
            self.assertTrue(self.torch.equal(value, second.state_dict()[name]))

    def test_checkpoint_selection_provenance_reload_and_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "candidate"
            manifest = train_candidate(
                "value_only_balanced", EXPERIENCE, SOURCE, output,
                OfflineTrainingConfig(
                    mode="value_only", epochs=2, batch_size=256, learning_rate=0.001,
                    patience=1, seed=42, strict_torso_freeze=True,
                ),
                balanced=True,
            )
            self.assertEqual(manifest["splits_accessed"], ["train", "validation"])
            self.assertFalse(manifest["held_out_test_accessed"])
            self.assertIn(manifest["best_epoch"], (1, 2))
            self.assertEqual(manifest["validation_objective"], "uncensored_return_mean_squared_error")
            self.assertIn("value_bias", manifest["validation_metrics"]["overall"])
            self.assertIn("gradient_norm", manifest["history"][0]["train"])
            self.assertTrue(manifest["parameter_update_verification"]["frozen_parameters_unchanged"])
            _, _, loaded = load_ablation_checkpoint(output, VALUE_SEMANTICS)
            self.assertEqual(loaded["checkpoint_sha256"], manifest["checkpoint_sha256"])
            self.assertEqual(manifest["source_checkpoint_sha256"], source_evidence_digests(EXPERIENCE, SOURCE)["source_checkpoint_sha256"])
            with self.assertRaisesRegex(ValueError, "semantics"):
                load_ablation_checkpoint(output, FROZEN_VALUE_SEMANTICS)

    def test_evaluation_plan_is_serializable_frozen_and_rejects_bad_adapter(self):
        source = source_evidence_digests(EXPERIENCE, SOURCE)
        manifests = {
            name: {
                "checkpoint_sha256": str(index),
                "value_head_semantics": FROZEN_VALUE_SEMANTICS if definition["mode"] == "policy_only" else VALUE_SEMANTICS,
            }
            for index, (name, definition) in enumerate(CANDIDATES.items())
        }
        plan = create_evaluation_plan(manifests, source)
        validate_evaluation_plan(plan)
        self.assertIsInstance(json.loads(json.dumps(plan)), dict)
        self.assertFalse(plan["held_out_results_present"])
        self.assertEqual(plan["target_selection"]["training_or_selection_access"], "prohibited")
        corrupted = copy.deepcopy(plan)
        corrupted["checkpoint_comparisons"]["joint_natural"]["value_adapter"] = "completion_length_to_leaf_value_with_terminal_overrides"
        with self.assertRaisesRegex(ValueError, "incompatible"):
            validate_evaluation_plan(corrupted)


if __name__ == "__main__":
    unittest.main()
