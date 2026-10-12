#!/usr/bin/env python3
"""CPU-only lifecycle tests for the experiment runner."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from matmul_inspector import experiment


class FakeBackend:
    mode = "success"

    def run(self, config, log):
        log("fake setup")
        if self.mode == "execution_failure":
            raise experiment.StageFailure("execution", "synthetic launch failure")
        correctness = {
            "passed": self.mode != "correctness_failure", "shape_match": True,
            "reference_shape": [2, 2], "candidate_shape": [2, 2],
            "max_absolute_error": 0.0 if self.mode == "success" else 1.0,
            "mean_absolute_error": 0.0 if self.mode == "success" else 0.25,
            "reference_nonfinite_count": 0, "candidate_nonfinite_count": 0,
            "atol": config["correctness"]["atol"], "rtol": config["correctness"]["rtol"],
            "reference": "fake_reference", "candidate_accumulation": "fake",
            "reference_accumulation": "fake",
        }
        base = {
            "environment": {"device": {"type": "fake"}, "versions": {"fake": "1"}},
            "setup": {"measurement": "fake setup", "unit": "ms",
                      "duration_ms": {"baseline": 2.0, "candidate": 3.0}},
            "correctness": correctness,
        }
        if not correctness["passed"]:
            return {**base, "status": "failed", "failure": {
                "stage": "correctness", "exception_type": "CorrectnessFailure",
                "message": "synthetic mismatch", "traceback": None,
            }}
        rounds = [
            {"round": 1, "order": ["baseline", "candidate"],
             "baseline_latency_ms": 2.0, "candidate_latency_ms": 1.0},
            {"round": 2, "order": ["candidate", "baseline"],
             "baseline_latency_ms": 2.2, "candidate_latency_ms": 1.1},
        ]
        return {**base, "status": "success", "benchmark": {
            "measurement": "fake steady-state time", "unit": "ms", "rounds": rounds,
            "warmups_per_implementation": 1, "iterations_per_measurement": 1,
            "baseline_summary": experiment._summary([2.0, 2.2]),
            "candidate_summary": experiment._summary([1.0, 1.1]),
            "speedup_baseline_over_candidate": 2.0,
        }}


class CorrectnessFailureBackend(FakeBackend):
    mode = "correctness_failure"


class ExecutionFailureBackend(FakeBackend):
    mode = "execution_failure"


class HangingBackend(FakeBackend):
    def run(self, config, log):
        time.sleep(10)
        return super().run(config, log)


class ExperimentRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="matmul-experiment-test-")
        self.output = Path(self.temporary.name)
        self.config = copy.deepcopy(experiment.DEFAULT_CONFIG)
        self.config["benchmark"] = {"warmups": 1, "iterations": 1, "rounds": 2}

    def tearDown(self):
        self.temporary.cleanup()

    def run_with(self, factory):
        return experiment.run_experiment(self.config, output_root=self.output,
                                         backend_factory=factory)

    def assert_record(self, result):
        root = Path(result["run_directory"])
        self.assertTrue((root / "submitted_config.json").is_file())
        self.assertTrue((root / "resolved_config.json").is_file())
        saved = json.loads((root / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["schema_version"], experiment.SCHEMA_VERSION)
        self.assertEqual(saved["run_id"], result["run_id"])
        return root, saved

    def test_success_serializes_results_and_raw_timings(self):
        result = self.run_with(FakeBackend)
        self.assertEqual(result["status"], "success")
        root, saved = self.assert_record(result)
        self.assertEqual(saved["benchmark"]["speedup_baseline_over_candidate"], 2.0)
        self.assertEqual(saved["benchmark"]["rounds"][0]["order"], ["baseline", "candidate"])
        self.assertEqual(saved["benchmark"]["rounds"][1]["order"], ["candidate", "baseline"])
        self.assertTrue((root / "timings.json").is_file())
        self.assertTrue((root / "timings.csv").is_file())
        self.assertIn("fake setup", (root / "stage.log").read_text(encoding="utf-8"))

    def test_correctness_failure_is_not_benchmarked(self):
        result = self.run_with(CorrectnessFailureBackend)
        self.assertEqual(result["failure"]["stage"], "correctness")
        root, saved = self.assert_record(result)
        self.assertNotIn("benchmark", saved)
        self.assertFalse((root / "timings.json").exists())

    def test_execution_failure_has_structured_traceback(self):
        result = self.run_with(ExecutionFailureBackend)
        self.assertEqual(result["failure"]["stage"], "execution")
        self.assertEqual(result["failure"]["exception_type"], "StageFailure")
        self.assertIn("synthetic launch failure", result["failure"]["message"])
        self.assertIn("Traceback", result["failure"]["traceback"])
        self.assert_record(result)

    def test_timeout_terminates_worker_and_persists_failure(self):
        self.config["timeout_seconds"] = 0.05
        started = time.monotonic()
        result = self.run_with(HangingBackend)
        self.assertLess(time.monotonic() - started, 3.0)
        self.assertEqual(result["failure"]["stage"], "timeout")
        root, _ = self.assert_record(result)
        self.assertIn("worker terminated", (root / "stage.log").read_text(encoding="utf-8"))

    def test_invalid_configuration_is_persisted_without_worker(self):
        self.config["candidate"]["parameters"]["block_k"] = 31
        result = self.run_with(FakeBackend)
        self.assertEqual(result["failure"]["stage"], "configuration")
        root = Path(result["run_directory"])
        self.assertFalse((root / "resolved_config.json").exists())
        self.assertFalse((root / "timings.json").exists())
        self.assertTrue((root / "result.json").is_file())


if __name__ == "__main__":
    unittest.main()
