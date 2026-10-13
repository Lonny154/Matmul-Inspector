#!/usr/bin/env python3
"""CPU-only tests for evaluation contracts and the mock evaluator."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from matmul_inspector.evaluation import (
    EvaluationLifecycle, EvaluationRecord, EvaluationStatus, FailureCategory,
    MockEvaluator, MockEvaluatorConfig, MockOutcome, read_evaluation_jsonl,
    write_evaluation_jsonl,
)
from matmul_inspector.sampling import (
    SearchContext, TritonGemmSearchSpace, UniformSampler, make_sampling_records,
)


FIXED_TIME = datetime(2026, 10, 13, 12, 0, tzinfo=timezone.utc)


def sampled_candidate():
    space = TritonGemmSearchSpace(
        {"block_m": [32], "block_n": [64], "block_k": [16],
         "num_warps": [4], "num_stages": [2]},
        SearchContext(512, 3072, 768, "fp16", hardware_target="sm_89"),
        identifier="evaluation-test",
    )
    sampler = UniformSampler(space, seed=7)
    return make_sampling_records(sampler.sample(), sampler,
                                 timestamp="2026-10-13T11:00:00Z")[0]


def evaluator(outcome=MockOutcome.SUCCESS, identifier="evaluation-fixed"):
    return MockEvaluator(
        MockEvaluatorConfig(outcome=outcome, synthetic_latency=2.5,
                            measurement_type="synthetic_mock_latency",
                            unit="synthetic_us", repetitions=20),
        id_factory=lambda: identifier, clock=lambda: FIXED_TIME,
    )


class EvaluationTests(unittest.TestCase):
    def test_lifecycle_accepts_valid_transitions_and_terminal_state(self):
        lifecycle = EvaluationLifecycle()
        for status in (EvaluationStatus.COMPILATION, EvaluationStatus.CORRECTNESS_VALIDATION,
                       EvaluationStatus.BENCHMARKING, EvaluationStatus.COMPLETED):
            lifecycle.transition(status)
        self.assertTrue(lifecycle.terminal)
        self.assertEqual(lifecycle.history[0], EvaluationStatus.PENDING)

    def test_lifecycle_rejects_skips_and_terminal_transitions(self):
        lifecycle = EvaluationLifecycle()
        with self.assertRaisesRegex(ValueError, "pending -> benchmarking"):
            lifecycle.transition(EvaluationStatus.BENCHMARKING)
        lifecycle.transition(EvaluationStatus.FAILED)
        with self.assertRaisesRegex(ValueError, "failed -> compilation"):
            lifecycle.transition(EvaluationStatus.COMPILATION)

    def test_successful_mock_evaluation_has_synthetic_measurement_contract(self):
        result = evaluator().evaluate(sampled_candidate())
        self.assertEqual(result.status, EvaluationStatus.COMPLETED)
        self.assertTrue(result.compilation.succeeded)
        self.assertTrue(result.correctness.passed)
        self.assertEqual(result.benchmark.latency, 2.5)
        self.assertEqual(result.benchmark.measurement_type, "synthetic_mock_latency")
        self.assertEqual(result.benchmark.unit, "synthetic_us")
        self.assertEqual(result.benchmark.repetitions, 20)
        self.assertTrue(result.benchmark.synthetic)

    def test_each_controlled_failure_has_no_benchmark_latency(self):
        outcomes = {
            MockOutcome.COMPILATION_FAILURE: FailureCategory.COMPILATION,
            MockOutcome.CORRECTNESS_FAILURE: FailureCategory.CORRECTNESS,
            MockOutcome.BENCHMARK_FAILURE: FailureCategory.BENCHMARK,
            MockOutcome.INFRASTRUCTURE_ERROR: FailureCategory.INFRASTRUCTURE,
        }
        for outcome, category in outcomes.items():
            with self.subTest(outcome=outcome):
                result = evaluator(outcome).evaluate(sampled_candidate())
                self.assertEqual(result.status, EvaluationStatus.FAILED)
                self.assertEqual(result.failure.category, category)
                self.assertIsNone(result.benchmark)

    def test_mock_behavior_is_deterministic_with_injected_identity_and_clock(self):
        candidate = sampled_candidate()
        self.assertEqual(evaluator().evaluate(candidate), evaluator().evaluate(candidate))

    def test_default_evaluation_identities_are_unique_for_same_candidate(self):
        candidate = sampled_candidate()
        mock = MockEvaluator()
        first = mock.evaluate(candidate)
        second = mock.evaluate(candidate)
        self.assertEqual(first.candidate_id, second.candidate_id)
        self.assertNotEqual(first.evaluation_id, second.evaluation_id)

    def test_evaluator_does_not_modify_candidate(self):
        candidate = sampled_candidate()
        before = candidate.to_dict()
        evaluator().evaluate(candidate)
        self.assertEqual(candidate.to_dict(), before)
        with self.assertRaises(FrozenInstanceError):
            candidate.sampling_order = 99

    def test_persistence_round_trip_preserves_multiple_evaluations_and_extensions(self):
        candidate = sampled_candidate()
        records = [evaluator(identifier=f"evaluation-{index}").evaluate(candidate)
                   for index in range(2)]
        value = records[0].to_dict()
        value["extensions"]["future"] = {"value": 1}
        records[0] = EvaluationRecord.from_dict(value)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evaluations.jsonl"
            write_evaluation_jsonl(path, records)
            loaded = read_evaluation_jsonl(path)
        self.assertEqual(loaded, records)
        self.assertEqual(loaded[0].candidate_id, loaded[1].candidate_id)
        self.assertNotEqual(loaded[0].evaluation_id, loaded[1].evaluation_id)

    def test_duplicate_evaluation_ids_do_not_replace_existing_jsonl(self):
        record = evaluator().evaluate(sampled_candidate())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evaluations.jsonl"
            path.write_text("preserved\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate evaluation IDs"):
                write_evaluation_jsonl(path, [record, record])
            self.assertEqual(path.read_text(encoding="utf-8"), "preserved\n")

    def test_schema_validation_rejects_invalid_records(self):
        value = evaluator().evaluate(sampled_candidate()).to_dict()
        cases = []
        missing = dict(value)
        missing.pop("candidate_id")
        cases.append((missing, "missing required fields: candidate_id"))
        cases.append(({**value, "unknown": 1}, "unknown version-1"))
        cases.append(({**value, "schema_version": 2}, "unsupported evaluation schema"))
        cases.append(({**value, "metadata": {"backend": "triton"}},
                      "evaluation metadata is missing required fields"))
        failed_with_latency = dict(value)
        failed_with_latency.update({
            "status": "failed",
            "lifecycle": ["pending", "compilation", "correctness_validation",
                          "benchmarking", "failed"],
            "failure": {"category": "benchmark_failure", "description": "failed"},
        })
        cases.append((failed_with_latency, "cannot contain a successful benchmark"))
        for malformed, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                EvaluationRecord.from_dict(malformed)

    def test_jsonl_reports_malformed_record_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            path.write_text(json.dumps({"schema_version": 99}) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "line 1.*unsupported evaluation schema"):
                read_evaluation_jsonl(path)


if __name__ == "__main__":
    unittest.main()
