"""CPU tests for TG-4B.2A diverse experience collection."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from matmul_inspector.tensor_game_diverse_experience import (
    DIVERSE_COLLECTION_VERSION,
    SEARCH_CONFIGURATIONS,
    DiverseCollectionConfig,
    DiverseTargetConfig,
    collect_diverse_experience,
    generate_diverse_targets,
    planned_episode_schedule,
    validate_diverse_experiment,
    validate_target_curriculum,
)
from matmul_inspector.tensor_game_experience import load_buffer
from matmul_inspector.tensor_game_offline_learning import ExperienceDataset


class DiverseExperienceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = DiverseTargetConfig(target_count=20, seed=71)
        cls.curriculum = generate_diverse_targets(cls.config)

    def test_generation_is_deterministic_balanced_and_exact(self):
        repeated = generate_diverse_targets(self.config)
        self.assertEqual(self.curriculum, repeated)
        rows = self.curriculum["targets"]
        self.assertEqual(len(rows), 20)
        self.assertEqual(len({row["target_digest"] for row in rows}), 20)
        self.assertEqual({length: sum(row["construction_length"] == length for row in rows) for length in range(1, 5)}, {1: 5, 2: 5, 3: 5, 4: 5})
        self.assertEqual({split: sum(row["split"] == split for row in rows) for split in ("train", "validation", "test")}, {"train": 14, "validation": 3, "test": 3})

    def test_duplicate_target_and_cross_target_residual_are_rejected(self):
        duplicate = copy.deepcopy(self.curriculum)
        duplicate["targets"][1]["target_digest"] = duplicate["targets"][0]["target_digest"]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_target_curriculum(duplicate)
        collision = copy.deepcopy(self.curriculum)
        collision["targets"][1]["known_trajectory_residual_digests"][0] = collision["targets"][0]["known_trajectory_residual_digests"][0]
        with self.assertRaisesRegex(ValueError, "residual collision"):
            validate_target_curriculum(collision)

    def test_fixed_schedule_counts_and_configuration_isolation(self):
        collection = DiverseCollectionConfig(seeds=(3, 11), simulations=2, wall_time_limit=0.05)
        smoke = planned_episode_schedule(self.curriculum, collection, "smoke")
        pilot = planned_episode_schedule(self.curriculum, collection, "pilot")
        full = planned_episode_schedule(self.curriculum, collection, "full")
        self.assertEqual((len(smoke), len(pilot), len(full)), (8, 40, 80))
        self.assertEqual(smoke, planned_episode_schedule(self.curriculum, collection, "smoke"))
        grouped = {}
        for row in full:
            grouped.setdefault((row["target_id"], row["seed"]), set()).add(row["configuration_id"])
        self.assertTrue(all(values == set(SEARCH_CONFIGURATIONS) for values in grouped.values()))

    def test_bounded_collection_round_trip_audit_and_offline_compatibility(self):
        collection = DiverseCollectionConfig(seeds=(3, 11), simulations=2, proposals=4, node_limit=20, wall_time_limit=0.05)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "experience"
            audit = collect_diverse_experience(self.curriculum, collection, "smoke", output)
            self.assertTrue(audit["replay_validation"]["valid"])
            self.assertEqual(audit["state_diversity"]["cross_split_residual_overlap"], 0)
            self.assertEqual(audit["training_readiness"]["eligible_policy_decisions"], audit["state_diversity"]["played_decisions"])
            self.assertGreater(audit["action_diversity"]["unique_canonical_actions"], 0)
            self.assertEqual(validate_diverse_experiment(output)["episodes"], 8)
            buffer = load_buffer(output)
            self.assertTrue(buffer.validate()["valid"])
            for episode in buffer.episodes:
                if episode["outcome"] == "success":
                    self.assertTrue(episode["exact_reconstruction"])
            datasets = [ExperienceDataset(output, split) for split in ("train", "validation", "test")]
            self.assertEqual(sum(len(dataset) for dataset in datasets), len(buffer.decisions))
            self.assertTrue(all(dataset.split_report["assignment_source"] == "persisted_precollection_target_partitions" for dataset in datasets))
            required = {"manifest.json", "episodes.json", "decisions.json", "replay_validation.json", "dataset_audit.json", "report.md"}
            self.assertTrue(required.issubset(path.name for path in output.iterdir()))
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["metadata"]["diverse_collection_version"], DIVERSE_COLLECTION_VERSION)
            with self.assertRaises(FileExistsError):
                collect_diverse_experience(self.curriculum, collection, "smoke", output)

    def test_partial_collection_is_recorded_not_relabelled(self):
        collection = DiverseCollectionConfig(seeds=(3, 11), simulations=1, proposals=2, node_limit=10, wall_time_limit=0.02)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "partial"
            collect_diverse_experience(self.curriculum, collection, "pilot", output, episode_limit=3)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertTrue(manifest["metadata"]["interrupted_collection"])
            self.assertEqual((manifest["metadata"]["executed_episode_count"], manifest["metadata"]["planned_episode_count"]), (3, 40))
            buffer = load_buffer(output)
            for decision in buffer.decisions.values():
                expected_valid = not decision["outcome_censored"] and decision["observed_return_target"] is not None
                self.assertEqual(expected_valid, not decision["outcome_censored"])


if __name__ == "__main__":
    unittest.main()
