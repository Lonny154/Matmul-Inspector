#!/usr/bin/env python3
"""CPU-only tests for controlled interleaved benchmarking."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from matmul_inspector.controlled_benchmark import (
    BASELINE_ID, BenchmarkPreparation, ControlledBenchmarkConfig,
    ControlledBenchmarkRunner, PreparedCandidate, TimingObservation,
    build_round_schedule, comparison_statistics, latency_statistics,
    read_controlled_manifest, render_markdown_report, validate_manifest,
    write_controlled_benchmark,
)
from matmul_inspector.evaluation import (
    CandidateIdentity, CompilationResult, CorrectnessResult, EvaluationIdentity,
    EvaluationMetadata, EvaluationRecord, EvaluationStatus, FailureCategory, FailureInfo,
    read_evaluation_jsonl,
)
from matmul_inspector.sampling import (
    SearchContext, TritonGemmSearchSpace, UniformSampler, make_sampling_records,
)


FIXED_TIME = datetime(2026, 10, 13, 12, 0, tzinfo=timezone.utc)


def candidates(count=2):
    space = TritonGemmSearchSpace(
        {"block_m": [32, 64], "block_n": [64, 128], "block_k": [32],
         "num_warps": [4], "num_stages": [2]},
        SearchContext(512, 3072, 768, "fp16", hardware_target="sm_89"),
        identifier="controlled-test",
    )
    sampler = UniformSampler(space, seed=2)
    return make_sampling_records(sampler.sample(count, unique=True), sampler,
                                 timestamp="2026-10-13T11:00:00Z")


class FakeBackend:
    def __init__(self, sampled, *, failed=(), timing=None, fail_measurement=None):
        self.sampled = sampled
        self.failed = set(failed)
        self.timing = timing or {}
        self.fail_measurement = fail_measurement
        self.warmups = []
        self.calls = []

    def prepare(self, sampled, config):
        eligible, failures = [], []
        environment = {
            "device": {"name": "Mock RTX 4060 Ti", "compute_capability": "8.9"},
            "versions": {"torch": "mock", "triton": "mock", "cuda_runtime": "mock"},
        }
        for index, item in enumerate(sampled):
            if item.candidate_id in self.failed:
                failures.append(EvaluationRecord(
                    evaluation_id=EvaluationIdentity(f"failed-{index}"),
                    candidate_id=CandidateIdentity(item.candidate_id),
                    status=EvaluationStatus.FAILED,
                    lifecycle=(EvaluationStatus.PENDING, EvaluationStatus.COMPILATION,
                               EvaluationStatus.CORRECTNESS_VALIDATION, EvaluationStatus.FAILED),
                    metadata=EvaluationMetadata("sm_89", "triton", (512, 3072, 768),
                                                "fp16", "2026-10-13T12:00:00Z"),
                    compilation=CompilationResult(True), correctness=CorrectnessResult(False),
                    failure=FailureInfo(FailureCategory.CORRECTNESS, "controlled failure"),
                ))
            else:
                eligible.append(PreparedCandidate(
                    item, f"evaluation-{index}", CompilationResult(True, "mock"),
                    CorrectnessResult(True), {"passed": True, "max_absolute_error": 0.0},
                    {"candidate_parameters": item.parameters.to_dict()},
                ))
        return BenchmarkPreparation(
            environment, {"operation": "mock torch.matmul out", "output_allocation": "preallocated"},
            tuple(eligible), tuple(failures),
        )

    def warmup(self, implementation_ids, launches):
        self.warmups.append((tuple(implementation_ids), launches))

    def measure(self, implementation_id, launches):
        self.calls.append(implementation_id)
        if implementation_id == self.fail_measurement:
            raise RuntimeError("controlled measurement failure")
        value = self.timing.get(implementation_id, 100.0)
        return TimingObservation(value, value * 0.8, value * 1.1)


def run(sampled=None, backend=None, **overrides):
    sampled = sampled or candidates()
    backend = backend or FakeBackend(sampled)
    config = ControlledBenchmarkConfig(
        rounds=overrides.get("rounds", 3), samples_per_round=overrides.get("samples", 2),
        launches_per_sample=4, warmups=2, scheduling_seed=overrides.get("seed", 9),
    )
    result = ControlledBenchmarkRunner(
        backend, id_factory=lambda: "experiment-test", clock=lambda: FIXED_TIME,
    ).run(sampled, config)
    return result, backend


class ControlledBenchmarkTests(unittest.TestCase):
    def test_schedule_is_deterministic_and_balanced_per_round(self):
        identifiers = ["a", "b", "c"]
        first = build_round_schedule(identifiers, 6, 17)
        second = build_round_schedule(identifiers, 6, 17)
        self.assertEqual(first, second)
        self.assertNotEqual(first, build_round_schedule(identifiers, 6, 18))
        for order in first:
            self.assertCountEqual(order, [BASELINE_ID, *identifiers])

    def test_latency_statistics_and_paired_comparison(self):
        summary = latency_statistics([1.0, 2.0, 3.0])
        self.assertEqual(summary["median_us"], 2.0)
        self.assertEqual(summary["mean_us"], 2.0)
        self.assertEqual(summary["minimum_us"], 1.0)
        self.assertEqual(summary["maximum_us"], 3.0)
        self.assertEqual(summary["sample_standard_deviation_us"], 1.0)
        self.assertEqual(summary["observation_count"], 3)
        comparison = comparison_statistics([100.0, 110.0], [50.0, 55.0])
        self.assertEqual(comparison["speedup_baseline_over_candidate"], 2.0)
        self.assertEqual(comparison["latency_reduction_percent"], 50.0)
        self.assertEqual(comparison["paired_round_speedups"], [2.0, 2.0])

    def test_invalid_measurement_sets_are_rejected(self):
        for values in ([], [0.0], [-1.0], [float("nan")], [float("inf")]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                latency_statistics(values)
        with self.assertRaises(ValueError):
            comparison_statistics([1.0], [1.0, 2.0])

    def test_interleaved_run_uses_injected_timings_and_records_orders(self):
        sampled = candidates()
        timing = {BASELINE_ID: 100.0, sampled[0].candidate_id: 80.0,
                  sampled[1].candidate_id: 125.0}
        result, backend = run(sampled, FakeBackend(sampled, timing=timing))
        manifest = result["manifest"]
        self.assertEqual(len(manifest["round_execution_order"]), 3)
        self.assertEqual(manifest["sampling"]["seeds"], [2])
        self.assertEqual([item["candidate_id"] for item in manifest["candidates"]],
                         [item.candidate_id for item in sampled])
        self.assertEqual(len(manifest["evaluation_records"]), 2)
        self.assertEqual(backend.warmups[0][1], 2)
        self.assertEqual(len(backend.calls), 3 * 3 * 2)
        first = manifest["implementations"][sampled[0].candidate_id]
        self.assertEqual(first["summary"]["median_us"], 80.0)
        self.assertEqual(first["per_round_median_us"], [80.0, 80.0, 80.0])
        self.assertEqual(first["comparison_to_baseline"]["speedup_baseline_over_candidate"], 1.25)
        self.assertAlmostEqual(
            first["comparison_to_baseline"]["latency_reduction_percent"], 20.0)
        self.assertEqual(len(result["evaluations"]), 2)

    def test_correctness_failure_is_gated_out_of_schedule_and_ranking(self):
        sampled = candidates()
        failed_id = sampled[0].candidate_id
        result, backend = run(sampled, FakeBackend(sampled, failed={failed_id}))
        manifest = result["manifest"]
        self.assertTrue(all(failed_id not in order for order in manifest["round_execution_order"]))
        self.assertNotIn(failed_id, backend.calls)
        failure = next(item for item in result["evaluations"]
                       if item.candidate_id.value == failed_id)
        self.assertEqual(failure.failure.category, FailureCategory.CORRECTNESS)
        self.assertIsNone(failure.benchmark)

    def test_benchmark_failure_has_no_score_or_latency(self):
        sampled = candidates()
        failed_id = sampled[0].candidate_id
        result, _ = run(sampled, FakeBackend(sampled, fail_measurement=failed_id))
        implementation = result["manifest"]["implementations"][failed_id]
        self.assertEqual(implementation["status"], "failed")
        self.assertIsNone(implementation["comparison_to_baseline"])
        evaluation = next(item for item in result["evaluations"]
                          if item.candidate_id.value == failed_id)
        self.assertEqual(evaluation.failure.category, FailureCategory.BENCHMARK)
        self.assertIsNone(evaluation.benchmark)

    def test_manifest_persistence_and_reports_round_trip(self):
        result, _ = run()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "controlled"
            write_controlled_benchmark(output, result)
            loaded = read_controlled_manifest(output / "experiment.json")
            evaluations = read_evaluation_jsonl(output / "evaluations.jsonl")
            report = (output / "report.md").read_text(encoding="utf-8")
        self.assertEqual(loaded, result["manifest"])
        self.assertEqual(evaluations, result["evaluations"])
        self.assertIn("PyTorch/cuBLAS", report)
        self.assertIn("Fastest measured candidate", report)
        self.assertIn("not a global-optimum claim", report)

    def test_schema_and_report_validation(self):
        result, _ = run()
        manifest = result["manifest"]
        self.assertIn("Median latency", render_markdown_report(manifest))
        with self.assertRaisesRegex(ValueError, "unsupported controlled benchmark schema"):
            validate_manifest({**manifest, "schema_version": 2})
        with self.assertRaisesRegex(ValueError, "unknown controlled benchmark"):
            validate_manifest({**manifest, "unexpected": True})
        missing = dict(manifest)
        missing.pop("baseline")
        with self.assertRaisesRegex(ValueError, "missing fields: baseline"):
            validate_manifest(missing)

    def test_report_explains_deduplicated_search_winner_provenance(self):
        result, _ = run()
        candidate_id = result["manifest"]["candidate_ids"][0]
        result["manifest"]["extensions"]["winner_selection"] = [{
            "candidate_id": candidate_id, "parameters": {},
            "discoveries": [
                {"trial_id": "trial-a", "arm": "uniform", "logical_budget": 12,
                 "search_latency_us": 60.25},
                {"trial_id": "trial-b", "arm": "adaptive", "logical_budget": 9,
                 "search_latency_us": 60.75},
            ],
        }]
        report = render_markdown_report(result["manifest"])
        self.assertIn("Search-time discovery versus confirmation", report)
        self.assertIn("not confirmation measurements", report)
        self.assertIn("confirmed once", report)
        self.assertIn("trial-a uniform at budget 12 (60.250 us exploratory)", report)


@unittest.skipUnless(os.environ.get("MATMUL_INSPECTOR_RUN_CUDA_TESTS") == "1",
                     "set MATMUL_INSPECTOR_RUN_CUDA_TESTS=1 for RTX 4060 Ti validation")
class ControlledBenchmarkHardwareTests(unittest.TestCase):
    def test_one_candidate_matched_smoke_run(self):
        from matmul_inspector.controlled_benchmark import CudaControlledBenchmarkBackend

        sampled = candidates(1)
        config = ControlledBenchmarkConfig(
            rounds=2, samples_per_round=1, launches_per_sample=5, warmups=5)
        result = ControlledBenchmarkRunner(CudaControlledBenchmarkBackend()).run(sampled, config)
        manifest = result["manifest"]
        self.assertIsNotNone(manifest["implementations"][BASELINE_ID]["summary"])
        self.assertEqual(manifest["implementations"][sampled[0].candidate_id]["status"],
                         "completed")


if __name__ == "__main__":
    unittest.main()
