"""Synthetic-only tests for TG-4B.2C execution wiring."""

import json
from pathlib import Path
import tempfile
import unittest

try:
    import torch
except ImportError:
    torch = None

from matmul_inspector.tensor_game import RankOneAction, outer_product
from matmul_inspector.tensor_game_heldout_execution import (
    ConfigurationFactory,
    TrialRecord,
    TrialStore,
    build_trial_schedule,
    validate_ablation_isolation,
)
from matmul_inspector.tensor_game_neural_mcts import DirectReturnValueEvaluator, NeuralValueEvaluator
from matmul_inspector.tensor_game_search import HeuristicValueEvaluator, ResidualInformedPolicy


@unittest.skipIf(torch is None, "optional PyTorch is unavailable")
class HeldoutExecutionWiringTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan_path = Path("artifacts/tensor-game/tg4b2b-offline-ablation-v1/evaluation_plan.json")
        cls.addendum_path = Path("artifacts/tensor-game/tg4b2b-offline-ablation-v1/evaluation_protocol_addendum_v1.json")
        cls.plan = json.loads(cls.plan_path.read_text())
        cls.addendum = json.loads(cls.addendum_path.read_text())
        cls.factory = ConfigurationFactory(cls.plan, cls.addendum, "cpu")

    def test_all_ten_mappings_and_model_cache(self):
        expected = {
            "residual_heuristic_control": ("residual_informed", "nonzero_heuristic", None, None),
            "residual_constant_max_control": ("residual_informed", "constant_max_moves", None, None),
            "frozen_policy_heuristic_value": ("neural", "nonzero_heuristic", "frozen_tg3b", None),
            "policy_only_natural_policy_heuristic_value": ("neural", "nonzero_heuristic", "policy_only_natural", None),
            "residual_frozen_checkpoint_value": ("residual_informed", "completion_length", None, "frozen_tg3b"),
            "residual_value_only_natural_checkpoint_value": ("residual_informed", "direct_return", None, "value_only_natural"),
            "residual_value_only_balanced_checkpoint_value": ("residual_informed", "direct_return", None, "value_only_balanced"),
            "frozen_policy_frozen_value": ("neural", "completion_length", "frozen_tg3b", "frozen_tg3b"),
            "joint_natural_policy_value": ("neural", "direct_return", "joint_natural", "joint_natural"),
            "joint_balanced_policy_value": ("neural", "direct_return", "joint_balanced", "joint_balanced"),
        }
        self.assertEqual(set(self.factory.configurations), set(expected))
        for name, values in expected.items():
            config = self.factory.configurations[name]
            self.assertEqual((config.policy_kind, config.value_kind, config.policy_checkpoint, config.value_checkpoint), values)
            self.assertEqual(config.coefficients, (-1, 0, 1))
            self.assertEqual((config.simulations, config.proposals, config.c_puct, config.node_limit), (200, 12, 1.5, 200))
        first = self.factory._load_model("frozen_tg3b")[0]
        self.assertIs(first, self.factory._load_model("frozen_tg3b")[0])

    def test_factory_types_and_value_dispatch(self):
        residual = self.factory.configurations["residual_heuristic_control"]
        self.assertIsInstance(self.factory.policy_factory(residual)(), ResidualInformedPolicy)
        self.assertIsInstance(self.factory.evaluator_factory(residual, 1)(), HeuristicValueEvaluator)
        completion = self.factory.configurations["residual_frozen_checkpoint_value"]
        direct = self.factory.configurations["residual_value_only_natural_checkpoint_value"]
        self.assertIsInstance(self.factory.evaluator_factory(completion, 1)(), NeuralValueEvaluator)
        self.assertIsInstance(self.factory.evaluator_factory(direct, 1)(), DirectReturnValueEvaluator)

    def test_ablation_isolation(self):
        self.assertTrue(all(validate_ablation_isolation(self.factory.configurations).values()))

    def test_all_ten_execute_through_episode_runner(self):
        target = outer_product(RankOneAction((1, 0, 0, 0), (1, 0, 0, 0), (1, 0, 0, 0)))
        for identifier in self.factory.configurations:
            episode, decisions = self.factory.run_fixture_episode(identifier, target)
            self.assertIn(episode["outcome"], {"success", "truncated", "interrupted"})
            self.assertEqual(episode["configuration_id"], identifier)
            self.assertEqual(episode["model_provenance"]["configuration_id"], identifier)
            self.assertLessEqual(len(decisions), 1)

    def test_schedule_is_deterministic_complete_and_unique(self):
        first = build_trial_schedule(self.factory.configurations, self.plan)
        second = build_trial_schedule(self.factory.configurations, self.plan)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 1800)
        self.assertEqual(len({row.trial_id for row in first}), 1800)
        self.assertEqual(len({row.target_group_id for row in first}), 30)
        self.assertEqual({row.seed for row in first}, set(self.plan["trial_seeds"]))

    def test_invalid_semantics_and_mapping_are_rejected(self):
        changed = json.loads(json.dumps(self.addendum))
        changed["checkpoint_path_resolutions"]["joint_natural"]["value_head_semantics"] = "unknown"
        next(row for row in changed["model_search_matrix"] if row["configuration_id"] == "joint_natural_policy_value")["value_head_semantics"] = "unknown"
        with self.assertRaisesRegex(ValueError, "unsupported checkpoint semantics"):
            ConfigurationFactory(self.plan, changed)
        changed = json.loads(json.dumps(self.addendum))
        changed["model_search_matrix"][0]["value_source"] = "checkpoint:missing"
        with self.assertRaisesRegex(ValueError, "missing checkpoint reference"):
            ConfigurationFactory(self.plan, changed)

    def test_trial_records_and_resume_round_trip(self):
        schedule = build_trial_schedule(self.factory.configurations, self.plan)
        trial = schedule[0]
        with tempfile.TemporaryDirectory() as directory:
            store = TrialStore(Path(directory), self.addendum["content_digest"], {})
            store.initialize()
            record = TrialRecord(
                trial.trial_id, "bounded_search_failure", "simulation_budget_exhausted",
                trial.configuration_id, trial.target_group_id, trial.seed, True, False,
                trial.checkpoint_provenance,
            )
            self.assertEqual(store.write(record), "recorded")
            self.assertEqual(store.write(record), "already_recorded")
            self.assertIn(trial.trial_id, store.completed_trial_ids())
            self.assertEqual(store.progress(schedule), {"scheduled": 1800, "recorded": 1, "never_started": 1799})
            conflicting = TrialRecord(
                trial.trial_id, "episode_timeout", "episode_wall_time_limit",
                trial.configuration_id, trial.target_group_id, trial.seed, True, False,
                trial.checkpoint_provenance,
            )
            with self.assertRaisesRegex(ValueError, "conflicting duplicate"):
                store.write(conflicting)
            with self.assertRaisesRegex(ValueError, "resume protocol"):
                TrialStore(Path(directory), "different", {}).initialize()

    def test_crash_between_result_write_and_progress_update_recovers_without_reexecution(self):
        schedule = build_trial_schedule(self.factory.configurations, self.plan)
        trial = schedule[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = TrialStore(root, self.addendum["content_digest"], {"frozen_tg3b": "digest"})
            store.initialize()
            store.update_progress_manifest(schedule)
            record = TrialRecord(
                trial.trial_id, "interrupted", "fault_injection_after_atomic_result",
                trial.configuration_id, trial.target_group_id, trial.seed, True, False,
                trial.checkpoint_provenance,
            )
            store.write(record)  # Simulated crash: progress.json is intentionally stale.
            self.assertEqual(json.loads((root / "progress.json").read_text())["recorded"], 0)
            resumed = TrialStore(root, self.addendum["content_digest"], {"frozen_tg3b": "digest"})
            resumed.initialize()
            recovered = resumed.recover_progress(schedule)
            self.assertEqual(recovered, {"scheduled": 1800, "recorded": 1, "never_started": 1799})
            self.assertIn(trial.trial_id, resumed.completed_trial_ids())
            self.assertEqual(resumed.write(record), "already_recorded")

    def test_failure_and_interruption_records(self):
        base = dict(
            trial_id="x", configuration_id="c", target_group_id="t", seed=3,
            attempted=True, exact_reconstruction=False, checkpoint_provenance={},
        )
        failure = TrialRecord(status="infrastructure_failure", termination_reason="exception", exception_type="RuntimeError", diagnostic="safe", **base)
        interrupted = TrialRecord(status="interrupted", termination_reason="external_stop", **base)
        self.assertEqual(failure.exception_type, "RuntimeError")
        self.assertTrue(interrupted.attempted)
        with self.assertRaises(ValueError):
            TrialRecord(status="bounded_search_failure", termination_reason="budget", exact_reconstruction=True, **{k: v for k, v in base.items() if k != "exact_reconstruction"})


if __name__ == "__main__":
    unittest.main()
