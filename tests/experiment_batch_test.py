#!/usr/bin/env python3
"""CPU-only tests for sequential experiment batch orchestration."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))

from matmul_inspector import experiment, experiment_batch


def batch_config():
    shared = copy.deepcopy(experiment.DEFAULT_CONFIG)
    shared.pop("candidate")
    return {
        "shared": shared,
        "candidates": [
            {"id": "a", "implementation": "triton_fc1",
             "parameters": {**experiment.DEFAULT_CONFIG["candidate"]["parameters"], "block_m": 128}},
            {"id": "b", "implementation": "triton_fc1",
             "parameters": {**experiment.DEFAULT_CONFIG["candidate"]["parameters"], "block_n": 64}},
            {"id": "c", "implementation": "triton_fc1",
             "parameters": copy.deepcopy(experiment.DEFAULT_CONFIG["candidate"]["parameters"])},
        ],
        "scheduling_seed": 17,
        "repetitions": 2,
    }


class FakeExecutor:
    def __init__(self, outcomes=None, interrupt_at=None):
        self.calls = []
        self.outcomes = outcomes or {}
        self.interrupt_at = interrupt_at

    def __call__(self, config, *, output_root):
        self.calls.append(copy.deepcopy(config))
        if self.interrupt_at == len(self.calls):
            raise KeyboardInterrupt("synthetic interruption")
        identifier = next(
            key for key, marker in (("a", (128, 128)), ("b", (64, 64)), ("c", (64, 128)))
            if (config["candidate"]["parameters"]["block_m"],
                config["candidate"]["parameters"]["block_n"]) == marker
        )
        status = self.outcomes.get((identifier, len(self.calls)), "success")
        run_directory = Path(output_root) / f"fake-{len(self.calls)}"
        run_directory.mkdir(parents=True)
        (run_directory / "result.json").write_text("{}\n", encoding="utf-8")
        base = {"run_id": f"run-{len(self.calls)}", "run_directory": str(run_directory)}
        if status == "raise":
            raise RuntimeError("synthetic executor failure")
        if status == "failure":
            return {**base, "status": "failed", "correctness": {"passed": False},
                    "failure": {"stage": "correctness", "exception_type": "CorrectnessFailure",
                                "message": "synthetic mismatch", "traceback": None}}
        if status == "missing":
            return {**base, "status": "success", "correctness": {"passed": True}}
        speedup = {"a": 1.2, "b": 0.8, "c": 1.0}[identifier]
        return {
            **base, "status": "success", "correctness": {"passed": True},
            "benchmark": {
                "baseline_summary": {"median_ms": 2.0},
                "candidate_summary": {"median_ms": 2.0 / speedup},
                "speedup_baseline_over_candidate": speedup,
            },
        }


class ExperimentBatchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="matmul-batch-test-")
        self.output = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def run_batch(self, config=None, executor=None):
        return experiment_batch.run_batch(
            config or batch_config(), output_root=self.output,
            experiment_executor=executor or FakeExecutor(),
        )

    def test_schedule_is_deterministic_and_covers_every_candidate(self):
        first = self.run_batch()
        second = self.run_batch()
        first_schedule = [(row["repetition"], row["candidate_id"])
                          for row in first["planned_schedule"]]
        second_schedule = [(row["repetition"], row["candidate_id"])
                           for row in second["planned_schedule"]]
        self.assertEqual(first_schedule, second_schedule)
        for repetition in (1, 2):
            self.assertCountEqual(
                [identifier for current, identifier in first_schedule if current == repetition],
                ["a", "b", "c"],
            )

    def test_shared_settings_and_candidate_parameters_reach_executor(self):
        config = batch_config()
        executor = FakeExecutor()
        self.run_batch(config, executor)
        self.assertEqual(len(executor.calls), 6)
        for call in executor.calls:
            self.assertEqual(call["workload"], config["shared"]["workload"])
            self.assertEqual(call["seed"], config["shared"]["seed"])
            self.assertEqual(call["benchmark"], config["shared"]["benchmark"])
        observed = {json.dumps(call["candidate"]["parameters"], sort_keys=True)
                    for call in executor.calls}
        expected = {json.dumps(candidate["parameters"], sort_keys=True)
                    for candidate in config["candidates"]}
        self.assertEqual(observed, expected)

    def test_failure_and_unexpected_exception_do_not_stop_later_runs(self):
        executor = FakeExecutor({("a", 1): "failure", ("b", 2): "raise"})
        result = self.run_batch(executor=executor)
        self.assertEqual(len(executor.calls), 6)
        self.assertEqual(result["status"], "completed_with_failures")
        stages = [entry["failure_stage"] for entry in result["entries"]]
        self.assertIn("correctness", stages)
        self.assertIn("orchestration", stages)

    def test_invalid_candidate_is_recorded_but_valid_candidate_runs(self):
        config = batch_config()
        config["candidates"][1]["parameters"].pop("block_k")
        executor = FakeExecutor()
        result = self.run_batch(config, executor)
        self.assertEqual(len(executor.calls), 4)
        invalid = [entry for entry in result["entries"] if entry["repetition"] is None]
        self.assertEqual([entry["candidate_id"] for entry in invalid], ["b"])
        self.assertEqual(invalid[0]["failure_stage"], "configuration")
        self.assertEqual(result["summary"]["counts"]["invalid_candidates"], 1)

    def test_duplicate_ids_reject_whole_batch_and_persist_failure(self):
        config = batch_config()
        config["candidates"][1]["id"] = "a"
        result = self.run_batch(config)
        self.assertEqual(result["status"], "configuration_failed")
        self.assertIn("duplicate candidate IDs", result["failure"]["message"])
        root = Path(result["batch_directory"])
        self.assertTrue((root / "batch_results.json").is_file())
        self.assertTrue((root / "summary.json").is_file())

    def test_aggregation_ranking_and_missing_measurements(self):
        config = batch_config()
        config["repetitions"] = 1
        # The deterministic schedule's first call is discovered without relying on its order.
        probe = self.run_batch(config, FakeExecutor())
        missing_id = probe["planned_schedule"][1]["candidate_id"]
        executor = FakeExecutor({(missing_id, 2): "missing"})
        result = self.run_batch(config, executor)
        aggregates = {row["candidate_id"]: row for row in result["summary"]["candidate_aggregates"]}
        self.assertIsNone(aggregates[missing_id]["median_run_speedup"])
        self.assertFalse(aggregates[missing_id]["complete"])
        ranking_ids = [row["candidate_id"] for row in result["summary"]["complete_candidate_ranking"]]
        self.assertNotIn(missing_id, ranking_ids)
        expected = sorted((identifier for identifier in ("a", "b", "c") if identifier != missing_id),
                          key=lambda identifier: -{"a": 1.2, "b": 0.8, "c": 1.0}[identifier])
        self.assertEqual(ranking_ids, expected)

    def test_interruption_persists_partial_summary_and_stops_scheduling(self):
        executor = FakeExecutor(interrupt_at=2)
        result = self.run_batch(executor=executor)
        self.assertEqual(result["status"], "interrupted")
        self.assertEqual(len(executor.calls), 2)
        self.assertEqual(result["summary"]["counts"]["completed_experiments"], 2)
        self.assertEqual(result["summary"]["counts"]["remaining_experiments"], 4)
        root = Path(result["batch_directory"])
        saved = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["status"], "interrupted")
        self.assertEqual(saved["schema_version"], experiment_batch.BATCH_SCHEMA_VERSION)
        self.assertIn("parameters", saved["experiments"][0])
        self.assertTrue((root / "summary.csv").is_file())


if __name__ == "__main__":
    unittest.main()
