#!/usr/bin/env python3
"""CPU-only contract tests for the real Triton evaluator adapter."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from matmul_inspector.evaluation import (
    EvaluationStatus, FailureCategory, read_evaluation_jsonl, write_evaluation_jsonl,
)
from matmul_inspector.sampling import (
    SearchContext, SamplingRecord, TritonGemmConfig, candidate_identifier,
)
from matmul_inspector.triton_evaluator import (
    CudaTritonRunner, TritonEvaluatorConfig, TritonGemmEvaluator,
    TritonRunEvidence, TritonRunnerFailure,
)


FIXED_TIME = datetime(2026, 10, 13, 12, 0, tzinfo=timezone.utc)


def candidate(*, backend="triton", precision="fp16", hardware_target="sm_89",
              shape=(512, 3072, 768), block_k=32):
    context = SearchContext(
        m=shape[0], n=shape[1], k=shape[2], precision=precision,
        backend=backend, hardware_target=hardware_target,
    )
    parameters = TritonGemmConfig(64, 128, block_k, 4, 3)
    return SamplingRecord(
        candidate_id=candidate_identifier(parameters, context), search_space_id="test",
        sampler="uniform", seed=7, parameters=parameters, context=context,
        sampling_order=0, timestamp="2026-10-13T11:00:00Z",
    )


def success_evidence():
    return TritonRunEvidence(
        device={"name": "Synthetic RTX 4060 Ti", "compute_capability": "8.9"},
        versions={"torch": "test", "triton": "test", "cuda_runtime": "test"},
        compilation_setup_host_us=1000.0,
        correctness={
            "passed": True, "max_absolute_error": 0.001, "mean_absolute_error": 0.0001,
            "reference": "torch.matmul", "atol": 0.01, "rtol": 0.01,
        },
        cuda_event_samples_us=(60.0, 58.0, 62.0),
        host_submission_samples_us=(45.0, 44.0, 46.0),
        host_wall_samples_us=(65.0, 63.0, 67.0),
    )


class FakeRunner:
    def __init__(self, result=None, failure=None):
        self.result = result or success_evidence()
        self.failure = failure
        self.calls = []

    def run(self, sampled, config):
        self.calls.append((sampled, config))
        if self.failure:
            raise self.failure
        return self.result


def evaluator(runner):
    return TritonGemmEvaluator(
        TritonEvaluatorConfig(warmups=2, iterations_per_sample=4, samples=3),
        runner=runner, id_factory=lambda: "evaluation-test", clock=lambda: FIXED_TIME,
    )


class TritonEvaluatorTests(unittest.TestCase):
    def test_success_maps_hardware_evidence_without_mutating_candidate(self):
        sampled = candidate()
        before = sampled.to_dict()
        runner = FakeRunner()
        result = evaluator(runner).evaluate(sampled)
        self.assertEqual(sampled.to_dict(), before)
        self.assertEqual(runner.calls[0][0], sampled)
        self.assertEqual(result.status, EvaluationStatus.COMPLETED)
        self.assertEqual(result.candidate_id.value, sampled.candidate_id)
        self.assertEqual(result.benchmark.latency, 60.0)
        self.assertEqual(result.benchmark.unit, "us")
        self.assertEqual(result.benchmark.measurement_type, "median CUDA-event kernel latency")
        self.assertEqual(result.benchmark.repetitions, 12)
        self.assertFalse(result.benchmark.synthetic)
        evidence = result.extensions["correctness_evidence"]
        self.assertEqual(evidence["max_absolute_error"], 0.001)
        self.assertEqual(evidence["mean_absolute_error"], 0.0001)
        distribution = result.extensions["benchmark_distribution"]
        self.assertEqual(distribution["cuda_event_samples_us"], [60.0, 58.0, 62.0])
        self.assertIn("host_wall_summary", distribution)
        self.assertEqual(result.extensions["candidate_parameters"], sampled.parameters.to_dict())

    def test_hardware_evidence_round_trips_through_existing_jsonl_schema(self):
        result = evaluator(FakeRunner()).evaluate(candidate())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evaluation.jsonl"
            write_evaluation_jsonl(path, [result])
            loaded = read_evaluation_jsonl(path)
        self.assertEqual(loaded, [result])

    def test_invalid_candidates_fail_before_hardware_runner(self):
        cases = (
            candidate(backend="nki"), candidate(precision="bf16"),
            candidate(hardware_target="sm_80"), candidate(shape=(256, 3072, 768)),
            candidate(block_k=50),
        )
        for sampled in cases:
            with self.subTest(context=sampled.context, parameters=sampled.parameters):
                runner = FakeRunner()
                result = evaluator(runner).evaluate(sampled)
                self.assertEqual(result.status, EvaluationStatus.FAILED)
                self.assertEqual(result.failure.category, FailureCategory.INVALID_CANDIDATE)
                self.assertIsNone(result.benchmark)
                self.assertEqual(runner.calls, [])

    def test_tampered_candidate_identity_is_rejected(self):
        sampled = candidate()
        tampered = SamplingRecord(
            candidate_id="candidate-tampered", search_space_id=sampled.search_space_id,
            sampler=sampled.sampler, seed=sampled.seed, parameters=sampled.parameters,
            context=sampled.context, sampling_order=sampled.sampling_order,
            timestamp=sampled.timestamp,
        )
        runner = FakeRunner()
        result = evaluator(runner).evaluate(tampered)
        self.assertEqual(result.failure.category, FailureCategory.INVALID_CANDIDATE)
        self.assertIn("identity", result.failure.description)
        self.assertEqual(runner.calls, [])

    def test_stage_failures_map_to_contract_and_never_have_latency(self):
        cases = (
            (FailureCategory.COMPILATION, EvaluationStatus.COMPILATION, False, None),
            (FailureCategory.CORRECTNESS, EvaluationStatus.CORRECTNESS_VALIDATION, True,
             {"passed": False, "max_absolute_error": 2.0}),
            (FailureCategory.BENCHMARK, EvaluationStatus.BENCHMARKING, True,
             {"passed": True}),
            (FailureCategory.INFRASTRUCTURE, EvaluationStatus.PENDING, None, None),
            (FailureCategory.INVALID_CANDIDATE, EvaluationStatus.PENDING, None, None),
        )
        for category, previous, compiled, correctness in cases:
            failure = TritonRunnerFailure(
                category, f"controlled {category.value}", versions={"triton": "test"},
                compilation_succeeded=compiled, correctness=correctness,
            )
            result = evaluator(FakeRunner(failure=failure)).evaluate(candidate())
            with self.subTest(category=category):
                self.assertEqual(result.status, EvaluationStatus.FAILED)
                self.assertEqual(result.lifecycle[-2], previous)
                self.assertEqual(result.failure.category, category)
                self.assertIsNone(result.benchmark)
                if correctness is not None:
                    self.assertEqual(result.extensions["correctness_evidence"], correctness)

    def test_unexpected_runner_error_is_infrastructure_failure(self):
        class BrokenRunner:
            def run(self, sampled, config):
                raise RuntimeError("unexpected test failure")

        result = evaluator(BrokenRunner()).evaluate(candidate())
        self.assertEqual(result.failure.category, FailureCategory.INFRASTRUCTURE)
        self.assertIn("unexpected test failure", result.failure.description)
        self.assertIn("Traceback", result.extensions["failure_traceback"])

    def test_configuration_validation(self):
        for values in ({"warmups": 0}, {"iterations_per_sample": 0}, {"samples": 0},
                       {"atol": -1}, {"rtol": float("nan")}, {"seed": -1}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                TritonEvaluatorConfig(**values)


@unittest.skipUnless(os.environ.get("MATMUL_INSPECTOR_RUN_CUDA_TESTS") == "1",
                     "set MATMUL_INSPECTOR_RUN_CUDA_TESTS=1 for RTX 4060 Ti validation")
class TritonHardwareTests(unittest.TestCase):
    def test_packaged_fc1_kernel_on_sm89(self):
        result = TritonGemmEvaluator(
            TritonEvaluatorConfig(warmups=5, iterations_per_sample=10, samples=3),
            runner=CudaTritonRunner(),
        ).evaluate(candidate())
        self.assertEqual(result.status, EvaluationStatus.COMPLETED, result.failure)
        self.assertTrue(result.correctness.passed)
        self.assertGreater(result.benchmark.latency, 0)
        self.assertFalse(result.benchmark.synthetic)


if __name__ == "__main__":
    unittest.main()
