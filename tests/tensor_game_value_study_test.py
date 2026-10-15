"""CPU tests for TG-3D value guidance controls and diagnostics."""

import tempfile
from pathlib import Path
import unittest

try:
    import torch
except ImportError:
    torch = None

from matmul_inspector.tensor_game import RankOneAction, outer_product
from matmul_inspector.tensor_game_learning import PolicyValueConfig
from matmul_inspector.tensor_game_neural_mcts import ConstantValueEvaluator, NeuralValueEvaluator
from matmul_inspector.tensor_game_search import SampledMCTS, SearchConfig, UniformSampledPolicy
from matmul_inspector.tensor_game_value_study import (
    ValueStudyConfig,
    create_untrained_model,
    load_value_study,
    paired_summary,
    save_value_study,
)


@unittest.skipIf(torch is None, "optional PyTorch is unavailable")
class ValueStudyTest(unittest.TestCase):
    class FakeModel(torch.nn.Module if torch is not None else object):
        def __init__(self, width, value):
            super().__init__()
            self.width = width
            self.value = value

        def forward(self, residual):
            return (
                torch.zeros((residual.shape[0], self.width, 3), device=residual.device),
                torch.full((residual.shape[0],), self.value, device=residual.device),
            )

    def test_value_query_instrumentation_and_depth_clipping(self):
        config = PolicyValueConfig(shape=(1, 1, 1), hidden_dim=4)
        residual = outer_product(RankOneAction((1,), (1,), (1,)))
        evaluator = NeuralValueEvaluator(self.FakeModel(config.action_width, 9.0), config, trace_limit=2)
        self.assertEqual(evaluator.evaluate(residual, 2, 4, 1), 2.0)
        self.assertEqual((evaluator.metrics.calls, evaluator.metrics.clipped), (1, 1))
        self.assertEqual(evaluator.clipping_by_depth, {2: 1})
        self.assertEqual(evaluator.clip_reasons, {"above_remaining_budget": 1})
        self.assertEqual(evaluator.trace[0]["remaining_moves"], 2)
        self.assertEqual(len(evaluator.unique_state_digests), 1)

    def test_constant_value_and_terminal_precedence(self):
        residual = outer_product(RankOneAction((1,), (1,), (1,)))
        evaluator = ConstantValueEvaluator(-0.25)
        search = SampledMCTS(
            residual, UniformSampledPolicy((1, 1, 1)),
            SearchConfig(simulations=1, max_moves=3), evaluator,
        )
        self.assertEqual(search._evaluate(search._node(residual, 2)), -0.25)
        self.assertEqual(search._evaluate(search._node((((0,),),), 2)), 3.0)
        self.assertEqual(search._evaluate(search._node(residual, 0)), -4.0)

    def test_untrained_initialization_is_reproducible(self):
        config = PolicyValueConfig(shape=(1, 1, 1), hidden_dim=4)
        first = create_untrained_model(config, 17).state_dict()
        second = create_untrained_model(config, 17).state_dict()
        self.assertTrue(all(torch.equal(first[key], second[key]) for key in first))

    def test_paired_aggregation_and_evidence_reload(self):
        rows = []
        for moves in (8, 7):
            for seed in (3, 11):
                for name in ("heuristic", "trained_neural", "constant_zero", "untrained_neural"):
                    rows.append({
                        "max_moves": moves, "seed": seed, "value_configuration": name,
                        "success": name == "trained_neural" and moves == 8,
                        "search_wall_seconds": 0.1, "termination_reason": "simulation_budget_exhausted",
                    })
        summary = paired_summary(rows)
        self.assertEqual(summary["trained_neural"]["successes"], 2)
        self.assertEqual(summary["heuristic"]["trained_only"], 2)
        config = ValueStudyConfig()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "study"
            save_value_study(output, config, rows, [], [], {"checkpoint_sha256": "abc"})
            experiment, results, diagnostics = load_value_study(output)
            self.assertEqual(experiment["experiment"], "tg3d_neural_value_study")
            self.assertEqual(len(results["trials"]), 16)
            self.assertEqual(diagnostics["trials"], [])
            with self.assertRaises(FileExistsError):
                save_value_study(output, config, rows, [], [], {})


if __name__ == "__main__":
    unittest.main()
