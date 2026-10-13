"""CPU tests for TG-3C neural-guided sampled MCTS."""

import json
import math
from pathlib import Path
import random
import tempfile
import unittest

try:
    import torch
except ImportError:
    torch = None

from matmul_inspector.tensor_game import RankOneAction, outer_product
from matmul_inspector.tensor_game_dataset import SyntheticDatasetConfig, generate_dataset
from matmul_inspector.tensor_game_experiments import CurriculumProblem
from matmul_inspector.tensor_game_learning import PolicyValueConfig
from matmul_inspector.tensor_game_neural_experiments import (
    CONFIGURATIONS,
    NeuralMCTSExperimentConfig,
    load_experiment,
    run_trials,
    save_experiment,
)
from matmul_inspector.tensor_game_neural_mcts import (
    NeuralActionPolicy,
    NeuralResidualHybridPolicy,
    NeuralValueEvaluator,
    canonical_action_log_probability,
)
from matmul_inspector.tensor_game_search import ResidualInformedPolicy, SampledMCTS, SearchConfig, SearchNode


@unittest.skipIf(torch is None, "optional PyTorch is unavailable")
class NeuralAdapterTest(unittest.TestCase):
    class FakeModel(torch.nn.Module if torch is not None else object):
        def __init__(self, action_width, value=1.0):
            super().__init__()
            self.action_width = action_width
            self.value = value
            self.inputs = []

        def forward(self, residual):
            self.inputs.append(residual.detach().cpu().clone())
            logits = torch.zeros((residual.shape[0], self.action_width, 3), device=residual.device)
            logits[..., 2] = 2.0
            value = torch.full((residual.shape[0],), self.value, device=residual.device)
            return logits, value

    def test_complete_probability_aggregates_sign_equivalent_forms(self):
        logits = torch.zeros((3, 3))
        action = RankOneAction((1,), (1,), (1,))
        observed = math.exp(canonical_action_log_probability(logits, action, (1, 1, 1)))
        self.assertAlmostEqual(observed, 4 / 27, places=6)

    def test_policy_is_legal_normalized_deterministic_and_residual_only(self):
        config = PolicyValueConfig(shape=(1, 1, 1), hidden_dim=4)
        model = self.FakeModel(config.action_width)
        policy = NeuralActionPolicy(model, config)
        target = outer_product(RankOneAction((1,), (1,), (1,)))
        first = policy.propose_actions(target, 1, random.Random(3), 3)
        second = policy.propose_actions(target, 1, random.Random(99), 3)
        self.assertEqual(first, second)
        self.assertAlmostEqual(sum(item.prior for item in first), 1.0)
        self.assertEqual(len(first), len({item.action for item in first}))
        self.assertTrue(all(set(item.action.u + item.action.v + item.action.w) <= {-1, 0, 1} for item in first))
        self.assertEqual(tuple(model.inputs[0].shape), (1, 1, 1, 1))

    def test_neural_value_conversion_terminal_budget_and_clamping(self):
        config = PolicyValueConfig(shape=(1, 1, 1), hidden_dim=4)
        target = outer_product(RankOneAction((1,), (1,), (1,)))
        evaluator = NeuralValueEvaluator(self.FakeModel(config.action_width, value=2.0), config)
        self.assertEqual(evaluator.evaluate(target, 3, 5, 1), 3.0)
        evaluator.model.value = 99.0
        self.assertEqual(evaluator.evaluate(target, 3, 5, 1), 2.0)
        self.assertEqual(evaluator.metrics.clipped, 1)
        zero = (((0,),),)
        self.assertEqual(evaluator.evaluate(zero, 2, 5, 1), 5.0)
        self.assertEqual(evaluator.evaluate(target, 0, 5, 1), -6.0)

    def test_hybrid_mixture_and_independent_search_components(self):
        config = PolicyValueConfig(shape=(1, 1, 1), hidden_dim=4)
        model = self.FakeModel(config.action_width, value=1.0)
        neural = NeuralActionPolicy(model, config)
        hybrid = NeuralResidualHybridPolicy(neural, ResidualInformedPolicy(config.shape), 0.5, 0.25)
        target = outer_product(RankOneAction((1,), (1,), (1,)))
        proposals = hybrid.propose_actions(target, 1, random.Random(4), 4)
        self.assertLessEqual(len(proposals), 4)
        self.assertAlmostEqual(sum(item.prior for item in proposals), 1.0)
        self.assertTrue(hybrid.last_provenance)
        result = SampledMCTS(
            target, neural, SearchConfig(simulations=4, max_actions=3, max_moves=1),
            NeuralValueEvaluator(model, config),
        ).run()
        self.assertTrue(result.exact_reconstruction)
        self.assertEqual(result.value_evaluator_type, "neural_completion_length")

    def test_five_configurations_persist_and_reload_without_overwrite(self):
        model_config = PolicyValueConfig(shape=(1, 1, 1), hidden_dim=4)
        model = self.FakeModel(model_config.action_width)
        target = outer_product(RankOneAction((1,), (1,), (1,)))
        problem = CurriculumProblem("heldout", "heldout_synthetic_length_1", target, 1, "test fixture")
        config = NeuralMCTSExperimentConfig(seeds=(3,), simulations=3, max_actions=3, node_limit=10)
        records = run_trials(model, model_config, config, (problem,))
        self.assertEqual({row["configuration"] for row in records}, set(CONFIGURATIONS))
        self.assertTrue(all(row["exact_reconstruction"] for row in records))
        dataset = generate_dataset(SyntheticDatasetConfig(shape=(2, 1, 1), decompositions_per_length=3, construction_lengths=(1,)))
        digest = __import__("matmul_inspector.tensor_game_neural_experiments", fromlist=["dataset_content_digest"]).dataset_content_digest(dataset)
        manifest = {"dataset_id": dataset.manifest["dataset_id"], "dataset_content_digest": digest}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "evidence"
            save_experiment(output, config, records, dataset=dataset, checkpoint_manifest=manifest, checkpoint_sha256="abc")
            loaded = load_experiment(output)
            self.assertEqual(len(loaded["trials"]), 5)
            with self.assertRaises(FileExistsError):
                save_experiment(output, config, records, dataset=dataset, checkpoint_manifest=manifest, checkpoint_sha256="abc")


if __name__ == "__main__":
    unittest.main()
