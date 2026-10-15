"""CPU tests for paired TG-3B evaluation auditing."""

from pathlib import Path
import tempfile
import unittest

from matmul_inspector.tensor_game_dataset import SyntheticDatasetConfig, generate_dataset
from matmul_inspector.tensor_game_evaluation_audit import (
    AUDIT_SCHEMA_VERSION,
    decomposition_cluster_bootstrap,
    load_audit,
    paired_disagreements,
    reference_exclusion_audit,
    run_evaluation_audit,
    save_audit,
)
from matmul_inspector.tensor_game_learning import PolicyValueConfig, create_policy_value_model

try:
    import torch
except ImportError:
    torch = None


class PairingTest(unittest.TestCase):
    def test_paired_counts_per_seed_and_denominators(self):
        learned = {"a": True, "b": False}
        records = [
            {"sample_id": "a", "decomposition_id": "g1", "seed": 3, "hit": True},
            {"sample_id": "b", "decomposition_id": "g2", "seed": 3, "hit": True},
            {"sample_id": "a", "decomposition_id": "g1", "seed": 11, "hit": False},
            {"sample_id": "b", "decomposition_id": "g2", "seed": 11, "hit": False},
        ]
        paired = paired_disagreements(learned, records)
        self.assertEqual(
            {key: paired[key] for key in ("both_hit", "learned_only", "heuristic_only", "neither")},
            {"both_hit": 1, "learned_only": 1, "heuristic_only": 1, "neither": 1},
        )
        self.assertEqual(paired["state_seed_evaluations"], 4)
        self.assertEqual(paired["learned_minus_heuristic"], 0.0)
        self.assertEqual(paired["per_seed"]["3"]["state_evaluations"], 2)

    def test_group_bootstrap_is_reproducible_and_group_aware(self):
        rows = [
            {"decomposition_id": "g1", "learned": 1, "heuristic": 0},
            {"decomposition_id": "g1", "learned": 1, "heuristic": 0},
            {"decomposition_id": "g2", "learned": 0, "heuristic": 1},
        ]
        first = decomposition_cluster_bootstrap(rows, seed=7, replicates=100)
        second = decomposition_cluster_bootstrap(rows, seed=7, replicates=100)
        self.assertEqual(first, second)
        self.assertEqual(first["groups"], 2)
        self.assertLessEqual(first["lower"], first["upper"])

    def test_versioned_persistence_does_not_overwrite(self):
        value = {"audit_schema_version": AUDIT_SCHEMA_VERSION, "records": {}}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "audit.json"
            save_audit(path, value)
            self.assertEqual(load_audit(path), value)
            with self.assertRaises(FileExistsError):
                save_audit(path, value)


@unittest.skipUnless(torch is not None, "optional PyTorch is not installed")
class FullAuditTest(unittest.TestCase):
    def test_matched_states_budgets_untrained_and_seeded_accounting(self):
        dataset = generate_dataset(
            SyntheticDatasetConfig(
                shape=(2, 2, 2), construction_lengths=(1, 2), decompositions_per_length=5, seed=19
            )
        )
        config = PolicyValueConfig(shape=(2, 2, 2), hidden_dim=16, input_scale=2.0)
        torch.manual_seed(1)
        trained = create_policy_value_model(config)
        torch.manual_seed(2)
        untrained = create_policy_value_model(config)
        audit = run_evaluation_audit(
            trained,
            untrained,
            dataset,
            config,
            device=torch.device("cpu"),
            proposal_count=3,
            policy_seeds=(3, 11),
            untrained_initialization_seed=2,
        )
        states = audit["test_residual_rows"]
        self.assertEqual(audit["summaries"]["trained"]["aggregate"]["state_seed_evaluations"], states)
        self.assertEqual(
            audit["summaries"]["residual_informed"]["aggregate"]["state_seed_evaluations"],
            states * 2,
        )
        self.assertEqual(audit["summaries"]["trained"]["aggregate"]["requested_action_slots"], states * 3)
        self.assertEqual(set(audit["records"]), {"trained", "untrained", "uniform_sampled", "residual_informed", "hybrid"})
        trained_ids = {row["sample_id"] for row in audit["records"]["trained"]}
        for name in ("uniform_sampled", "residual_informed", "hybrid"):
            self.assertEqual({row["sample_id"] for row in audit["records"][name]}, trained_ids)
            self.assertEqual(set(audit["summaries"][name]["per_seed"]), {"3", "11"})
        self.assertTrue(audit["summaries"]["trained"]["deterministic"])
        self.assertTrue(audit["summaries"]["untrained"]["deterministic"])
        self.assertGreater(audit["summaries"]["trained"]["aggregate"]["internal_candidates_considered"], 0)
        self.assertEqual(
            audit["paired_trained_vs_residual_informed"]["state_seed_evaluations"], states * 2
        )
        self.assertTrue(reference_exclusion_audit(dataset)["passed"])


if __name__ == "__main__":
    unittest.main()
