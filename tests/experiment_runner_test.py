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
from unittest import mock

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
             "baseline_latency_ms": 2.0, "candidate_latency_ms": 1.0,
             "baseline_cuda_event_elapsed_ms_per_operation": 2.0,
             "candidate_cuda_event_elapsed_ms_per_operation": 1.0,
             "baseline_cuda_event_elapsed_ms_per_block": 2.0,
             "candidate_cuda_event_elapsed_ms_per_block": 1.0,
             "baseline_host_submission_ms_per_block": 0.2,
             "candidate_host_submission_ms_per_block": 0.1,
             "baseline_host_wall_ms_per_block": 2.2,
             "candidate_host_wall_ms_per_block": 1.1},
            {"round": 2, "order": ["candidate", "baseline"],
             "baseline_latency_ms": 2.2, "candidate_latency_ms": 1.1,
             "baseline_cuda_event_elapsed_ms_per_operation": 2.2,
             "candidate_cuda_event_elapsed_ms_per_operation": 1.1,
             "baseline_cuda_event_elapsed_ms_per_block": 2.2,
             "candidate_cuda_event_elapsed_ms_per_block": 1.1,
             "baseline_host_submission_ms_per_block": 0.2,
             "candidate_host_submission_ms_per_block": 0.1,
             "baseline_host_wall_ms_per_block": 2.4,
             "candidate_host_wall_ms_per_block": 1.2},
        ]
        return {**base, "status": "success", "benchmark": {
            "mode": config["benchmark"]["mode"],
            "measurement": "fake steady-state time", "unit": "ms",
            "units": {"cuda_event_elapsed_ms_per_operation": "ms/operation",
                      "host_submission_ms_per_block": "ms/block",
                      "host_wall_ms_per_block": "ms/block"}, "rounds": rounds,
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
        header = (root / "timings.csv").read_text(encoding="utf-8").splitlines()[0]
        self.assertIn("benchmark_mode", header)
        self.assertIn("baseline_host_submission_ms_per_block", header)
        self.assertEqual(saved["config"]["benchmark"]["mode"], "allocating")
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

    def test_omitted_mode_defaults_to_allocating_and_invalid_mode_is_rejected(self):
        config = copy.deepcopy(experiment.DEFAULT_CONFIG)
        config["benchmark"].pop("mode")
        self.assertEqual(experiment.validate_config(config)["benchmark"]["mode"], "allocating")
        config["benchmark"]["mode"] = "cuda_graph"
        with self.assertRaisesRegex(ValueError, "benchmark.mode"):
            experiment.validate_config(config)

    def test_preallocated_mode_propagates_and_correctness_still_gates_benchmark(self):
        self.config["benchmark"]["mode"] = "preallocated"
        result = self.run_with(FakeBackend)
        self.assertEqual(result["config"]["benchmark"]["mode"], "preallocated")
        self.assertEqual(result["benchmark"]["mode"], "preallocated")
        failed = self.run_with(CorrectnessFailureBackend)
        self.assertEqual(failed["config"]["benchmark"]["mode"], "preallocated")
        self.assertNotIn("benchmark", failed)

    def test_measurement_host_wall_brackets_event_completion_and_records_delta(self):
        order = []

        class Event:
            def record(self):
                order.append("record")

            def synchronize(self):
                order.append("synchronize")

            def elapsed_time(self, other):
                order.append("elapsed")
                return 4.0

        class Cuda:
            Event = staticmethod(lambda **kwargs: Event())

        class Torch:
            cuda = Cuda()

        with mock.patch.object(experiment.time, "perf_counter_ns",
                               side_effect=[1_000_000, 2_000_000, 4_000_000]):
            measured = experiment.CudaTritonBackend._measure(
                Torch(), lambda: order.append("submit"), 1)
        self.assertEqual(order, ["record", "submit", "record", "synchronize", "elapsed"])
        self.assertEqual(measured["host_wall_ms_per_block"], 3.0)
        self.assertEqual(measured["cuda_event_minus_host_wall_ms_per_block"], 1.0)


if __name__ == "__main__":
    unittest.main()
