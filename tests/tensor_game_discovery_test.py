"""CPU-only tests for TG-2B proposals and persisted experiments."""

import json
import inspect
from pathlib import Path
import random
import tempfile
import unittest

from matmul_inspector.tensor_game import RankOneAction, matrix_multiplication_tensor, outer_product, sum_actions
from matmul_inspector.tensor_game_experiments import (
    CurriculumProblem,
    ExperimentConfig,
    curriculum,
    load_config,
    mode_zero_flattening_rank,
    run_trials,
    save_experiment,
)
from matmul_inspector.tensor_game_search import (
    HybridSampledPolicy,
    ResidualInformedPolicy,
    SampledMCTS,
    SearchConfig,
    contract_residual,
    quantize_ratio,
)


class ResidualPolicyTest(unittest.TestCase):
    def test_independent_policies_contain_no_reference_fixture_access(self):
        source = inspect.getsource(ResidualInformedPolicy) + inspect.getsource(HybridSampledPolicy)
        self.assertNotIn("strassen", source.lower())
        self.assertNotIn("standard_2x2", source.lower())
        self.assertNotIn("ReferenceGuidedPolicy", source)

    def test_exact_contraction_and_zero_factors(self):
        residual = matrix_multiplication_tensor(1, 2, 1)
        self.assertEqual(contract_residual(residual, (1, -1), (1, 1)), (0,))
        self.assertEqual(contract_residual(residual, (0, 0), (1, 1)), (0,))
        policy = ResidualInformedPolicy((2, 2, 1))
        proposals = policy.propose_actions(residual, 2, random.Random(4), 8)
        self.assertTrue(proposals)
        self.assertTrue(all(all(any(vector) for vector in (p.action.u, p.action.v, p.action.w)) for p in proposals))

    def test_quantization_uses_exact_nearest_allowed_value(self):
        self.assertEqual(quantize_ratio(7, 4, (-1, 0, 1)), 1)
        self.assertEqual(quantize_ratio(-3, 2, (-1, 0, 1)), -1)
        self.assertEqual(quantize_ratio(1, 3, (-1, 0, 1)), 0)
        with self.assertRaises(ValueError):
            quantize_ratio(1, 0, (-1, 0, 1))

    def test_residual_and_hybrid_proposals_are_reproducible_and_normalized(self):
        residual = matrix_multiplication_tensor(1, 2, 1)
        for factory in (
            lambda: ResidualInformedPolicy((2, 2, 1)),
            lambda: HybridSampledPolicy((2, 2, 1), residual_probability=0.6),
        ):
            first = factory().propose_actions(residual, 2, random.Random(9), 10)
            second = factory().propose_actions(residual, 2, random.Random(9), 10)
            self.assertEqual(first, second)
            self.assertEqual(len(first), len({item.action for item in first}))
            self.assertAlmostEqual(sum(item.prior for item in first), 1.0)

    def test_rank_one_discovery_and_timing_metrics(self):
        target = outer_product(RankOneAction((1, -1), (1, 1), (1, -1)))
        result = SampledMCTS(
            target,
            ResidualInformedPolicy((2, 2, 2)),
            SearchConfig(simulations=100, max_actions=16, max_moves=1, seed=3, node_limit=100),
        ).run()
        self.assertTrue(result.exact_reconstruction)
        self.assertEqual(result.scalar_multiplications, 1)
        self.assertGreater(result.proposal_attempts, 0)
        self.assertGreaterEqual(result.proposal_generation_seconds, 0)

    def test_wall_time_limit_is_explicit(self):
        target = matrix_multiplication_tensor(2, 2, 2)
        result = SampledMCTS(
            target,
            ResidualInformedPolicy((4, 4, 4)),
            SearchConfig(
                simulations=1000,
                max_actions=8,
                max_moves=7,
                seed=1,
                node_limit=1000,
                max_wall_time_seconds=1e-12,
            ),
        ).run()
        self.assertEqual(result.termination_reason, "wall_time_limit_reached")
        self.assertLess(result.simulations_run, 1000)
        self.assertFalse(result.exact_reconstruction)


class CurriculumExperimentTest(unittest.TestCase):
    def test_rank_two_fixture_has_certificate_and_rectangular_level(self):
        problems = curriculum()
        rank_two = next(problem for problem in problems if problem.category == "synthetic_rank_two")
        self.assertEqual(mode_zero_flattening_rank(rank_two.target), 2)
        rectangular = next(problem for problem in problems if problem.identifier == "level3_matmul_1x2x1")
        self.assertEqual(len(rectangular.target), 2)
        self.assertEqual(rectangular.max_moves, 2)

    def test_nonzero_reduction_is_not_accepted_as_completion(self):
        target = sum_actions(
            (
                RankOneAction((1, 0), (1, 0), (1, 0)),
                RankOneAction((0, 1), (0, 1), (0, 1)),
            ),
            (2, 2, 2),
        )
        problem = CurriculumProblem("two_terms_one_move", "synthetic", target, 1, "fixture")
        records = run_trials(
            ExperimentConfig(seeds=(1,), simulations=20, max_actions=4, node_limit=20),
            (problem,),
        )
        self.assertTrue(all(not record["success"] for record in records))
        self.assertTrue(all(record["solution"] is None for record in records))

    def test_matched_budgets_and_configuration_replay(self):
        problem = curriculum()[0]
        config = ExperimentConfig(seeds=(2,), simulations=12, max_actions=4, node_limit=12)
        records = run_trials(config, (problem,))
        self.assertEqual({record["simulations_permitted"] for record in records}, {12})
        self.assertEqual({record["simulations_completed"] for record in records}, {12})
        self.assertEqual({record["strategy"] for record in records}, {"uniform_sampled", "residual_informed", "hybrid"})
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "evidence"
            save_experiment(output, config, records)
            self.assertEqual(load_config(output / "results.json"), config)
            payload = json.loads((output / "results.json").read_text())
            self.assertEqual(payload["format_version"], 1)
            self.assertTrue((output / "report.md").is_file())
            with self.assertRaises(FileExistsError):
                save_experiment(output, config, records)


if __name__ == "__main__":
    unittest.main()
