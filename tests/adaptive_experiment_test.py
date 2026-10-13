#!/usr/bin/env python3
"""CPU-only tests for controlled adaptive-search experiments."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from matmul_inspector.adaptive_experiment import (
    ARM_NAMES, PairedTrialRunner, SimulatedEvaluator, SimulatedLandscape, TrialConfig,
    derive_seed, read_trial_result, validate_trial_result, write_trial_result,
)
from matmul_inspector.evaluation import FailureCategory, EvaluationStatus
from matmul_inspector.sampling import (
    SamplingRecord, SearchContext, TritonGemmSearchSpace, candidate_identifier,
)


def space():
    return TritonGemmSearchSpace(
        {"block_m": [32, 64], "block_n": [64, 128], "block_k": [16, 32],
         "num_warps": [4, 8], "num_stages": [2, 3]},
        SearchContext(512, 3072, 768, "fp16", hardware_target="simulated-sm89"),
        identifier="paired-simulation-test",
    )


def run(*, kind="structured", budget=12, batch=3, replicates=2,
        failure_modulus=None, failure_category=FailureCategory.COMPILATION):
    return PairedTrialRunner(
        space(), SimulatedLandscape(kind=kind, seed=91, noise_us=0.25,
                                    failure_modulus=failure_modulus,
                                    failure_category=failure_category),
        TrialConfig(budget, batch, replicates, 1234),
    ).run()


class AdaptiveExperimentTests(unittest.TestCase):
    def test_exact_replay_and_seed_derivation(self):
        first, second = run(), run()
        self.assertEqual(first, second)
        self.assertEqual(derive_seed(4, "a"), derive_seed(4, "a"))
        self.assertNotEqual(derive_seed(4, "a"), derive_seed(4, "b"))

    def test_equal_submitted_budgets_and_isolated_arm_sequences(self):
        result = run(budget=15)
        for replicate in result["replicates"]:
            self.assertTrue(replicate["paired_budget_equal"])
            arms = replicate["arms"]
            for arm in ARM_NAMES:
                budget = arms[arm]["budget"]
                self.assertEqual(budget["submitted_evaluations"], 15)
                self.assertEqual(budget["proposal_attempts"], 15)
                self.assertEqual(budget["duplicate_proposals"], 0)
                self.assertEqual(budget["invalid_proposals"], 0)
            self.assertEqual(arms["uniform"]["candidate_ids"][:3],
                             arms["adaptive"]["candidate_ids"][:3])
        changed_master_seed = PairedTrialRunner(
            space(), SimulatedLandscape(seed=91, noise_us=0.25),
            TrialConfig(15, 3, 2, 1235),
        ).run()
        self.assertNotEqual(result["replicates"], changed_master_seed["replicates"])

    def test_failures_consume_budget_without_replacement(self):
        result = run(failure_modulus=2)
        for replicate in result["replicates"]:
            for arm in ARM_NAMES:
                budget = replicate["arms"][arm]["budget"]
                self.assertEqual(budget["submitted_evaluations"], 12)
                self.assertEqual(budget["successful_evaluations"]
                                 + budget["failed_evaluations"], 12)
                self.assertGreater(budget["failed_evaluations"], 0)
                self.assertEqual(budget["unique_candidates_evaluated"], 12)

    def test_infrastructure_failure_stops_both_arms_at_equal_count(self):
        result = run(budget=20, failure_modulus=2,
                     failure_category=FailureCategory.INFRASTRUCTURE)
        for replicate in result["replicates"]:
            counts = [replicate["arms"][arm]["budget"]["submitted_evaluations"]
                      for arm in ARM_NAMES]
            self.assertEqual(counts[0], counts[1])
            self.assertLess(counts[0], 20)
            self.assertTrue(all(replicate["arms"][arm]["termination"]
                                == "paired_infrastructure_stop" for arm in ARM_NAMES))

    def test_cumulative_rebuilds_are_auditable_and_grow(self):
        arm = run(replicates=1)["replicates"][0]["arms"]["adaptive"]
        self.assertEqual([item["feedback_count"] for item in arm["rebuilds"]], [3, 6, 9])
        for rebuild in arm["rebuilds"]:
            self.assertEqual(len(rebuild["source_evaluation_ids"]), rebuild["round"] * 3)
            self.assertEqual(len(set(rebuild["source_evaluation_ids"])),
                             len(rebuild["source_evaluation_ids"]))
            self.assertEqual(len(rebuild["feedback_set_digest"]), 64)
            self.assertIn("proposal_rng_state_before", rebuild)

    def test_feedback_deduplication_keeps_first_evaluation_identity(self):
        trial = run(replicates=1)
        raw = trial["replicates"][0]["arms"]["adaptive"]["evaluations"][0]
        from matmul_inspector.evaluation import EvaluationRecord
        record = EvaluationRecord.from_dict(raw)
        deduplicated = PairedTrialRunner._deduplicate([record, record])
        self.assertEqual(deduplicated, [record])

    def test_rebuilding_same_history_replays_proposals(self):
        first = run(replicates=1)
        second = run(replicates=1)
        self.assertEqual(first["replicates"][0]["arms"]["adaptive"]["rebuilds"],
                         second["replicates"][0]["arms"]["adaptive"]["rebuilds"])
        self.assertEqual(first["replicates"][0]["arms"]["adaptive"]["candidate_ids"],
                         second["replicates"][0]["arms"]["adaptive"]["candidate_ids"])

    def test_objective_is_independent_of_evaluation_order_and_arm(self):
        search = space()
        landscape = SimulatedLandscape(kind="interaction", seed=5, noise_us=2.0)
        configs = search.valid_candidates()[:5]
        observed = {}
        for arm, sequence in (("uniform", configs), ("adaptive", reversed(configs))):
            evaluator = SimulatedEvaluator(landscape, trial_id="same", arm_id=arm)
            for order, config in enumerate(sequence):
                record = SamplingRecord(
                    candidate_identifier(config, search.context), search.identifier, arm, 1,
                    config, search.context, order, "2000-01-01T00:00:00Z", {"simulated": True},
                )
                outcome = evaluator.evaluate(record)
                observed.setdefault(record.candidate_id, []).append(outcome.benchmark.latency)
        self.assertTrue(all(len(set(values)) == 1 for values in observed.values()))

    def test_all_landscape_shapes_have_known_rankings(self):
        for kind in ("structured", "interaction", "flat"):
            with self.subTest(kind=kind):
                landscape = SimulatedLandscape(kind=kind)
                ranked = landscape.ranked_candidates(space())
                self.assertEqual(len(ranked), len(space().valid_candidates()))
                self.assertLessEqual(landscape.latency(ranked[0]), landscape.latency(ranked[-1]))
        interaction = run(kind="interaction", replicates=1)
        flat = run(kind="flat", replicates=1)
        self.assertEqual(interaction["landscape"]["kind"], "interaction")
        self.assertEqual(flat["landscape"]["kind"], "flat")

    def test_persistence_round_trip_and_strict_top_level_schema(self):
        value = run(replicates=1)
        value["extensions"]["future"] = {"accepted": True}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trial.json"
            write_trial_result(path, value)
            self.assertEqual(read_trial_result(path), value)
        malformed = deepcopy(value)
        malformed["unknown"] = 1
        with self.assertRaisesRegex(ValueError, "unknown version-1"):
            validate_trial_result(malformed)
        malformed = deepcopy(value)
        malformed["schema_version"] = 2
        with self.assertRaisesRegex(ValueError, "unsupported trial result schema"):
            validate_trial_result(malformed)
        malformed = deepcopy(value)
        malformed["replicates"][0]["arms"]["uniform"]["evaluations"][0]["metadata"] = {}
        with self.assertRaisesRegex(ValueError, "evaluation metadata"):
            validate_trial_result(malformed)

    def test_aggregate_preserves_raw_paired_replicate_metrics(self):
        result = run(replicates=3)
        aggregate = result["aggregate"]
        self.assertEqual(aggregate["replicate_count"], 3)
        for arm in ARM_NAMES:
            self.assertEqual(aggregate["arms"][arm]["completed_replicates"], 3)
            self.assertEqual(len(aggregate["arms"][arm]["raw_final_metrics"]), 3)
        self.assertEqual(len(aggregate["paired"]["raw_deltas"]), 3)
        compared = (aggregate["paired"]["adaptive_lower_count"]
                    + aggregate["paired"]["uniform_lower_count"]
                    + aggregate["paired"]["tie_count"])
        self.assertEqual(compared, 3)


if __name__ == "__main__":
    unittest.main()
