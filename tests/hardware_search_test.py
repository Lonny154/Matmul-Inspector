#!/usr/bin/env python3
"""CPU-only tests for dry-run and injected hardware-search orchestration."""

from __future__ import annotations

from datetime import datetime, timezone
from io import StringIO
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from matmul_inspector.evaluation import (
    BenchmarkResult, CandidateIdentity, CompilationResult, CorrectnessResult,
    EvaluationIdentity, EvaluationMetadata, EvaluationRecord, EvaluationStatus,
    FailureCategory, FailureInfo, read_evaluation_jsonl,
)
from matmul_inspector.hardware_search import (
    ARM_NAMES, HardwareSearchConfig, HardwareSearchExperiment,
    default_fc1_search_space, read_hardware_search_manifest,
    render_hardware_search_report, validate_hardware_search_manifest,
    search_winner_records, search_winner_selection, write_hardware_search,
)


FIXED_TIME = datetime(2026, 10, 13, 12, 0, tzinfo=timezone.utc)


class FakeHardwareEvaluator:
    def __init__(self, *, infrastructure_at=None, fail_candidate_ids=()):
        self.calls = []
        self.infrastructure_at = infrastructure_at
        self.fail_candidate_ids = set(fail_candidate_ids)

    def evaluate(self, sampled):
        call = len(self.calls)
        self.calls.append(sampled)
        metadata = EvaluationMetadata(
            "sm_89", "triton", (512, 3072, 768), "fp16",
            "2026-10-13T12:00:00Z", "torch=fake; triton=fake; cuda_runtime=fake",
        )
        common = dict(
            evaluation_id=EvaluationIdentity(f"evaluation-{id(self)}-{call}"),
            candidate_id=CandidateIdentity(sampled.candidate_id), metadata=metadata,
            extensions={
                "candidate_parameters": sampled.parameters.to_dict(),
                "measurement_provenance": "real_hardware",
                "environment": {"device": {"name": "Fake RTX 4060 Ti",
                                            "compute_capability": "8.9"},
                                "versions": {"torch": "fake", "triton": "fake"}},
                "benchmark_distribution": {"cuda_event_samples_us": [60.0, 61.0, 62.0]},
            },
        )
        if call == self.infrastructure_at:
            return EvaluationRecord(
                status=EvaluationStatus.FAILED,
                lifecycle=(EvaluationStatus.PENDING, EvaluationStatus.FAILED),
                failure=FailureInfo(FailureCategory.INFRASTRUCTURE, "fake outage"), **common,
            )
        if sampled.candidate_id in self.fail_candidate_ids:
            return EvaluationRecord(
                status=EvaluationStatus.FAILED,
                lifecycle=(EvaluationStatus.PENDING, EvaluationStatus.COMPILATION,
                           EvaluationStatus.CORRECTNESS_VALIDATION, EvaluationStatus.FAILED),
                compilation=CompilationResult(True), correctness=CorrectnessResult(False),
                failure=FailureInfo(FailureCategory.CORRECTNESS, "fake mismatch"), **common,
            )
        latency = (150.0 - sampled.parameters.block_m * 0.2
                   - sampled.parameters.block_n * 0.1
                   + sampled.parameters.num_warps)
        return EvaluationRecord(
            status=EvaluationStatus.COMPLETED,
            lifecycle=(EvaluationStatus.PENDING, EvaluationStatus.COMPILATION,
                       EvaluationStatus.CORRECTNESS_VALIDATION,
                       EvaluationStatus.BENCHMARKING, EvaluationStatus.COMPLETED),
            compilation=CompilationResult(True, "fake"), correctness=CorrectnessResult(True),
            benchmark=BenchmarkResult(latency, "median CUDA-event kernel latency",
                                      "us", 3, False), **common,
        )


def experiment(*, seeds=(11,), budget=9, initial=3, batch=2, timer=None):
    return HardwareSearchExperiment(
        default_fc1_search_space(),
        HardwareSearchConfig(seeds, budget, initial, batch, latency_threshold_us=120.0),
        clock=lambda: FIXED_TIME, timer=timer,
    )


class HardwareSearchTests(unittest.TestCase):
    def test_dry_run_never_constructs_or_calls_evaluator(self):
        result = experiment().dry_run()
        self.assertEqual(result["mode"], "dry_run")
        self.assertEqual(result["trials"], [])
        self.assertFalse(result["dependency_check"]["cuda_queried"])
        self.assertEqual(result["accounting"]["completed_physical_evaluations"], 0)
        self.assertIn("no benchmark was run", render_hardware_search_report(result).lower())

    def test_default_plan_matches_proposed_experiment_and_pending_decisions(self):
        result = HardwareSearchExperiment(default_fc1_search_space()).dry_run()
        self.assertEqual(result["config"]["trial_seeds"], [101, 202, 303])
        self.assertEqual(result["config"]["per_arm_budget"], 16)
        self.assertEqual(result["config"]["shared_initial_size"], 5)
        self.assertEqual(result["config"]["adaptive_batch_size"], 2)
        self.assertEqual(result["search_space"]["valid_candidate_count"], 48)
        self.assertEqual(result["accounting"]["planned_logical_evaluations"], 96)
        self.assertEqual(result["accounting"]["maximum_physical_evaluations"], 81)
        for plan in result["trial_plans"]:
            self.assertEqual(len(plan["shared_initial_candidates"]), 5)
            self.assertEqual(plan["adaptive_rebuild_after_logical_counts"],
                             [5, 7, 9, 11, 13, 15])
            self.assertTrue(all(item["status"] == "pending_hardware_feedback"
                                for item in plan["future_adaptive_decisions"]))

    def test_seed_plan_and_initial_candidates_are_deterministic(self):
        first = experiment(seeds=(11, 12)).dry_run()["trial_plans"]
        second = experiment(seeds=(11, 12)).dry_run()["trial_plans"]
        self.assertEqual(first, second)
        self.assertNotEqual(first[0]["seeds"], first[1]["seeds"])
        self.assertNotEqual(first[0]["shared_initial_candidates"],
                            first[1]["shared_initial_candidates"])

    def test_execute_requires_explicit_method_and_uses_shared_physical_records(self):
        evaluators = []

        def factory(seed, trial_id):
            del seed, trial_id
            value = FakeHardwareEvaluator()
            evaluators.append(value)
            return value

        result = experiment().execute(factory)
        trial = result["manifest"]["trials"][0]
        self.assertEqual(len(evaluators[0].calls), 15)
        self.assertEqual(trial["physical_evaluation_count"], 15)
        for arm in ARM_NAMES:
            self.assertEqual(trial["arms"][arm]["submitted_evaluations"], 9)
            self.assertEqual(trial["arms"][arm]["unique_candidate_count"], 9)
        shared = trial["shared_evaluation_ids"]
        self.assertEqual(len(shared), 3)
        for evaluation_id in shared:
            self.assertIn(evaluation_id, {item["evaluation_id"]
                                         for item in trial["arms"]["uniform"]["history"]})
            self.assertIn(evaluation_id, {item["evaluation_id"]
                                         for item in trial["arms"]["adaptive"]["history"]})

    def test_feedback_isolation_and_cumulative_rebuilding(self):
        result = experiment().execute(lambda seed, trial: FakeHardwareEvaluator())
        trial = result["manifest"]["trials"][0]
        adaptive_ids = {item["evaluation_id"] for item in trial["arms"]["adaptive"]["history"]}
        uniform_only = ({item["evaluation_id"] for item in trial["arms"]["uniform"]["history"]}
                        - set(trial["shared_evaluation_ids"]))
        self.assertEqual([item["eligible_feedback_count"]
                          for item in trial["adaptive_rebuilds"]], [3, 5, 7])
        for rebuild in trial["adaptive_rebuilds"]:
            self.assertTrue(set(rebuild["source_evaluation_ids"]) <= adaptive_ids)
            self.assertFalse(set(rebuild["source_evaluation_ids"]) & uniform_only)

    def test_correctness_failure_consumes_budget_without_latency(self):
        plan = experiment().dry_run()["trial_plans"][0]
        failed_id = plan["shared_initial_candidates"][0]["candidate_id"]
        result = experiment().execute(
            lambda seed, trial: FakeHardwareEvaluator(fail_candidate_ids={failed_id}))
        hardware_trial = result["manifest"]["trials"][0]
        for arm in ARM_NAMES:
            summary = hardware_trial["arms"][arm]
            self.assertEqual(summary["submitted_evaluations"], 9)
            self.assertEqual(summary["failures_by_category"]["correctness_failure"], 1)
            failed = next(item for item in summary["history"]
                          if item["candidate_id"] == failed_id)
            self.assertIsNone(failed["latency_us"])

    def test_infrastructure_failure_terminates_trial_and_is_recorded(self):
        result = experiment().execute(
            lambda seed, trial: FakeHardwareEvaluator(infrastructure_at=4))
        trial = result["manifest"]["trials"][0]
        self.assertIn("infrastructure_failure", trial["status"])
        self.assertLess(trial["physical_evaluation_count"], 15)
        categories = [item["failure_category"] for arm in ARM_NAMES
                      for item in trial["arms"][arm]["history"]]
        self.assertIn("evaluation_infrastructure_error", categories)

    def test_physical_schedule_is_reproducible_and_counterbalanced(self):
        first = experiment().execute(lambda seed, trial: FakeHardwareEvaluator())
        second = experiment().execute(lambda seed, trial: FakeHardwareEvaluator())
        first_order = [item["logical_arms"]
                       for item in first["manifest"]["trials"][0]["physical_execution_order"]]
        second_order = [item["logical_arms"]
                        for item in second["manifest"]["trials"][0]["physical_execution_order"]]
        self.assertEqual(first_order, second_order)
        later = first_order[3:]
        self.assertIn(["uniform"], later)
        self.assertIn(["adaptive"], later)
        first_arm_per_step = [later[index][0] for index in range(0, len(later), 2)]
        self.assertTrue(all(current != following
                            for current, following in zip(first_arm_per_step,
                                                          first_arm_per_step[1:])))

    def test_search_timing_and_progress_are_separate_from_kernel_latency(self):
        ticks = iter((10.0, 11.0, 14.0, 15.0))
        result = experiment(timer=lambda: next(ticks)).execute(
            lambda seed, trial: FakeHardwareEvaluator())
        trial = result["manifest"]["trials"][0]
        self.assertEqual(trial["search_wall_clock_seconds"], 3.0)
        self.assertEqual(result["manifest"]["extensions"]["search_wall_clock_seconds"], 5.0)
        self.assertEqual(len(trial["arms"]["adaptive"]["progress"]), 9)
        self.assertEqual(len(trial["paired_progress"]), 9)
        self.assertEqual(trial["paired_progress"][-1]["equal_logical_budget"], 9)

    def test_confirmation_selection_deduplicates_arm_winners(self):
        result = experiment().execute(lambda seed, trial: FakeHardwareEvaluator())
        winners = search_winner_records(result["manifest"])
        self.assertEqual(len({item.candidate_id for item in winners}), len(winners))
        self.assertGreaterEqual(len(winners), 1)
        self.assertLessEqual(len(winners), 2)
        self.assertTrue(all(item.context.hardware_target == "sm_89" for item in winners))

    def test_confirmation_selection_combines_manifests_and_preserves_discoveries(self):
        result = experiment().execute(lambda seed, trial: FakeHardwareEvaluator())
        single_winners, single_selection = search_winner_selection((result["manifest"],))
        winners, selection = search_winner_selection(
            (result["manifest"], result["manifest"]))
        self.assertEqual([item.candidate_id for item in winners],
                         [item.candidate_id for item in single_winners])
        self.assertEqual(
            [len(item["discoveries"]) for item in selection],
            [2 * len(item["discoveries"]) for item in single_selection],
        )

    def test_persistence_replay_report_and_strict_schema(self):
        result = experiment().execute(lambda seed, trial: FakeHardwareEvaluator())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "search"
            write_hardware_search(output, result)
            loaded = read_hardware_search_manifest(output / "experiment.json")
            evaluations = read_evaluation_jsonl(output / "evaluations.jsonl")
            report = (output / "report.md").read_text(encoding="utf-8")
            progress = json.loads((output / "progress.json").read_text(encoding="utf-8"))
        self.assertEqual(loaded, result["manifest"])
        self.assertEqual(len(evaluations), 15)
        self.assertIn("exploratory hardware search", report)
        self.assertEqual(len(progress), 1)
        with self.assertRaisesRegex(ValueError, "unknown hardware search"):
            validate_hardware_search_manifest({**loaded, "unknown": True})
        with self.assertRaisesRegex(ValueError, "unsupported hardware search"):
            validate_hardware_search_manifest({**loaded, "schema_version": 2})

    def test_dry_run_cli_is_default_and_does_not_import_triton_evaluator(self):
        import examples.compare_triton_search as command

        with patch.object(sys, "argv", ["compare_triton_search.py", "--trial-seed", "7"]), \
                patch("sys.stdout", new_callable=StringIO) as output:
            self.assertEqual(command.main(), 0)
        self.assertIn("NO BENCHMARK WAS RUN", output.getvalue())


@unittest.skipUnless(os.environ.get("MATMUL_INSPECTOR_RUN_CUDA_TESTS") == "1",
                     "set MATMUL_INSPECTOR_RUN_CUDA_TESTS=1 for RTX 4060 Ti validation")
class HardwareSearchCudaTests(unittest.TestCase):
    def test_one_shared_candidate_smoke(self):
        from matmul_inspector.triton_evaluator import TritonEvaluatorConfig, TritonGemmEvaluator

        search = HardwareSearchExperiment(
            default_fc1_search_space(), HardwareSearchConfig((7,), 1, 1, 1),
        )
        result = search.execute(lambda seed, trial: TritonGemmEvaluator(
            TritonEvaluatorConfig(seed=seed, warmups=5, iterations_per_sample=10, samples=3)))
        self.assertEqual(result["manifest"]["accounting"]["completed_physical_evaluations"], 1)


if __name__ == "__main__":
    unittest.main()
