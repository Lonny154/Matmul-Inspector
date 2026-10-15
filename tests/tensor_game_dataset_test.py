"""CPU-only tests for TG-3A synthetic demonstration datasets."""

import builtins
import copy
import importlib
import json
from pathlib import Path
import random
import tempfile
import unittest

from matmul_inspector.tensor_game import RankOneAction, outer_product, sum_actions
from matmul_inspector.tensor_game_dataset import (
    GENERATOR_VERSION,
    SCHEMA_VERSION,
    SyntheticDataset,
    SyntheticDatasetConfig,
    TorchSyntheticDataset,
    generate_dataset,
    load_dataset,
    sample_action,
    save_dataset,
    validate_dataset,
)
from matmul_inspector.tensor_game_search import canonicalize_action


class GenerationTest(unittest.TestCase):
    def test_deterministic_generation_shapes_lengths_and_reconstruction(self):
        config = SyntheticDatasetConfig(
            shape=(2, 3, 2), construction_lengths=(1, 2, 3), decompositions_per_length=2, seed=17
        )
        first = generate_dataset(config)
        second = generate_dataset(config)
        self.assertEqual(first, second)
        self.assertEqual({record["construction_length"] for record in first.decompositions}, {1, 2, 3})
        self.assertEqual(first.manifest["decomposition_count"], 6)
        self.assertEqual(first.manifest["sample_count"], 12)
        for record in first.decompositions:
            target = tuple(tuple(tuple(row) for row in plane) for plane in record["target"])
            factors = tuple(RankOneAction(item["u"], item["v"], item["w"]) for item in record["factors"])
            self.assertEqual(sum_actions(factors, config.shape), target)

    def test_factor_sparsity_zero_rejection_and_canonicalization(self):
        sparse = SyntheticDatasetConfig(shape=(8, 8, 8), construction_lengths=(1,), zero_probability=0.9)
        action = sample_action(sparse, random.Random(4))
        self.assertTrue(all(any(vector) for vector in (action.u, action.v, action.w)))
        self.assertGreater(sum(value == 0 for value in action.u + action.v + action.w), 12)
        self.assertEqual(action, canonicalize_action(action))

    def test_intermediate_residuals_multi_action_labels_and_multiplicity(self):
        action = canonicalize_action(RankOneAction((1, 0), (1, 0), (1, 0)))
        target = sum_actions((action, action), (2, 2, 2))
        config = SyntheticDatasetConfig(shape=(2, 2, 2), construction_lengths=(2,), decompositions_per_length=1)
        from matmul_inspector.tensor_game_dataset import _trajectory

        rows = _trajectory("repeat", (action, action), target, config)
        self.assertEqual(len(rows[0]["remaining_factors"]), 2)
        self.assertEqual(len(rows[0]["next_actions"]), 1)
        self.assertEqual(rows[0]["known_completion_length"], 2)
        self.assertEqual(rows[1]["residual"], [[[1, 0], [0, 0]], [[0, 0], [0, 0]]])

    def test_group_splits_and_no_target_or_residual_leakage(self):
        dataset = generate_dataset(
            SyntheticDatasetConfig(shape=(3, 3, 3), decompositions_per_length=3, seed=31)
        )
        decomposition_splits = {record["decomposition_id"]: record["split"] for record in dataset.decompositions}
        self.assertTrue(all(decomposition_splits[row["decomposition_id"]] == row["split"] for row in dataset.samples))
        target_digests = [record["target_digest"] for record in dataset.decompositions]
        self.assertEqual(len(target_digests), len(set(target_digests)))
        residual_splits = {}
        for row in dataset.samples:
            residual_splits.setdefault(row["residual_digest"], set()).add(row["split"])
        self.assertTrue(all(len(splits) == 1 for splits in residual_splits.values()))

    def test_deliberate_duplicate_target_and_residual_leakage_is_rejected(self):
        dataset = generate_dataset(
            SyntheticDatasetConfig(shape=(2, 2, 2), construction_lengths=(1,), decompositions_per_length=3, seed=8)
        )
        duplicated = copy.deepcopy(dataset.decompositions[0])
        corrupt_manifest = copy.deepcopy(dataset.manifest)
        corrupt_manifest["decomposition_count"] += 1
        corrupt = SyntheticDataset(corrupt_manifest, dataset.decompositions + (duplicated,), dataset.samples)
        with self.assertRaises(ValueError):
            validate_dataset(corrupt)
        leaked_samples = copy.deepcopy(list(dataset.samples))
        shared = next(row for row in leaked_samples if row["split"] == "train")
        other = next(row for row in leaked_samples if row["split"] != "train")
        for key in (
            "residual", "residual_digest", "remaining_factors", "next_actions",
            "known_completion_length", "step",
        ):
            other[key] = copy.deepcopy(shared[key])
        from matmul_inspector.tensor_game_dataset import _digest

        other["sample_id"] = _digest(
            [other["decomposition_id"], other["step"], other["residual_digest"]]
        )
        content = dict(other)
        content.pop("content_digest")
        other["content_digest"] = _digest(content)
        with self.assertRaisesRegex(ValueError, "across dataset splits"):
            validate_dataset(SyntheticDataset(dataset.manifest, dataset.decompositions, tuple(leaked_samples)))


class PersistenceAndAdapterTest(unittest.TestCase):
    def setUp(self):
        self.dataset = generate_dataset(
            SyntheticDatasetConfig(shape=(2, 2, 2), construction_lengths=(1, 2), decompositions_per_length=3, seed=9)
        )

    def test_versioned_round_trip_is_exact_and_non_overwriting(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "dataset"
            save_dataset(self.dataset, output)
            loaded = load_dataset(output)
            self.assertEqual(loaded, self.dataset)
            self.assertEqual(loaded.manifest["schema_version"], SCHEMA_VERSION)
            self.assertEqual(loaded.manifest["generator_version"], GENERATOR_VERSION)
            with self.assertRaises(FileExistsError):
                save_dataset(self.dataset, output)

    def test_unsupported_version_and_corrupt_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "dataset"
            save_dataset(self.dataset, output)
            manifest_path = output / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["schema_version"] = 999
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "unsupported"):
                load_dataset(output)
        corrupt = copy.deepcopy(self.dataset.samples[0])
        corrupt["known_completion_length"] += 1
        with self.assertRaises(ValueError):
            validate_dataset(SyntheticDataset(self.dataset.manifest, self.dataset.decompositions, (corrupt,) + self.dataset.samples[1:]))

    def test_module_import_does_not_import_torch(self):
        module = importlib.import_module("matmul_inspector.tensor_game_dataset")
        original = builtins.__import__

        def reject_torch(name, *args, **kwargs):
            if name == "torch":
                raise AssertionError("module import attempted to load torch")
            return original(name, *args, **kwargs)

        builtins.__import__ = reject_torch
        try:
            importlib.reload(module)
        finally:
            builtins.__import__ = original

    def test_optional_torch_adapter(self):
        try:
            import torch
        except ImportError:
            self.skipTest("optional PyTorch is not installed")
        adapter = TorchSyntheticDataset(self.dataset, split="train")
        item = adapter[0]
        self.assertEqual(tuple(item["residual"].shape), (2, 2, 2))
        self.assertEqual(tuple(item["next_actions"].shape), (2, 6))
        self.assertEqual(tuple(item["action_mask"].shape), (2,))
        self.assertEqual(item["residual"].dtype, torch.float32)
        self.assertEqual(item["next_actions"].dtype, torch.int64)


if __name__ == "__main__":
    unittest.main()
