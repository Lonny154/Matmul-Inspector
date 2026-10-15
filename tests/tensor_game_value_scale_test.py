"""CPU tests for TG-3E value transforms and trajectory accounting."""

import random
import unittest

try:
    import torch
except ImportError:
    torch = None

from matmul_inspector.tensor_game import RankOneAction, outer_product
from matmul_inspector.tensor_game_learning import PolicyValueConfig
from matmul_inspector.tensor_game_search import (
    ActionStatistics, ReferenceGuidedPolicy, SampledMCTS, SearchConfig, SearchNode, select_puct,
)
from matmul_inspector.tensor_game_value_scale import AffineNeuralValueEvaluator, _digest_solution


@unittest.skipIf(torch is None, "optional PyTorch is unavailable")
class ValueScaleTest(unittest.TestCase):
    class FakeModel(torch.nn.Module if torch is not None else object):
        def forward(self, residual):
            return torch.zeros((len(residual), 3, 3)), torch.ones(len(residual))

    def test_positive_affine_transform_preserves_order(self):
        config = PolicyValueConfig(shape=(1, 1, 1), hidden_dim=4)
        evaluator = AffineNeuralValueEvaluator(self.FakeModel(), config, "cpu", 2.0, 1.0, "expanded")
        residual = outer_product(RankOneAction((1,), (1,), (1,)))
        self.assertEqual(evaluator.evaluate(residual, 3, 4, 1), 7.0)
        values = [-1.0, 0.5, 3.0]
        transformed = [evaluator.scale * x + evaluator.offset for x in values]
        self.assertEqual(sorted(range(3), key=values.__getitem__), sorted(range(3), key=transformed.__getitem__))

    def test_root_puct_components_and_bounded_trace(self):
        target = outer_product(RankOneAction((1,), (1,), (1,)))
        action = RankOneAction((1,), (1,), (1,))
        result = SampledMCTS(
            target, ReferenceGuidedPolicy(target, (action,), randomized_order=False),
            SearchConfig(simulations=3, max_actions=1, max_moves=1, trace_simulations=2),
        ).run()
        self.assertEqual(len(result.simulation_trace), 2)
        self.assertTrue(result.simulation_trace[1]["terminal_success"])
        root = result.simulation_trace[1]["root_actions"][0]
        self.assertAlmostEqual(root["puct_score"], root["q_value"] + root["puct_exploration_bonus"])

    def test_factor_sequence_digest_is_deterministic_and_order_sensitive(self):
        first = RankOneAction((1,), (1,), (1,))
        second = RankOneAction((-1,), (1,), (1,))
        self.assertEqual(_digest_solution((first, second)), _digest_solution((first, second)))
        self.assertNotEqual(_digest_solution((first, second)), _digest_solution((second, first)))

    def test_state_identity_remains_depth_aware(self):
        target = outer_product(RankOneAction((1,), (1,), (1,)))
        search = SampledMCTS(target, ReferenceGuidedPolicy(target, (RankOneAction((1,), (1,), (1,)),)), SearchConfig(simulations=1))
        self.assertNotEqual(search._node(target, 1).key, search._node(target, 2).key)


if __name__ == "__main__":
    unittest.main()
