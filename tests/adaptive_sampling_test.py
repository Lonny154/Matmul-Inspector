#!/usr/bin/env python3
"""CPU-only tests for offline adaptive candidate sampling."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from matmul_inspector.adaptive_sampling import (
    AdaptiveSampler, read_adaptive_state, write_adaptive_state,
)
from matmul_inspector.evaluation import (
    BenchmarkResult, CandidateIdentity, CompilationResult, CorrectnessResult,
    EvaluationIdentity, EvaluationMetadata, EvaluationRecord, EvaluationStatus,
    FailureCategory, FailureInfo,
)
from matmul_inspector.sampling import (
    SearchContext, SearchSpaceExhausted, TritonGemmConfig, TritonGemmSearchSpace,
    candidate_identifier,
)


def space():
    return TritonGemmSearchSpace(
        {"block_m": [32, 64], "block_n": [64, 128], "block_k": [32],
         "num_warps": [4, 8], "num_stages": [2, 3]},
        SearchContext(512, 3072, 768, "fp16", hardware_target="sm_89"),
        identifier="adaptive-test",
    )


def evaluation(config, latency=70.0, *, identifier=None, synthetic=False,
               status=EvaluationStatus.COMPLETED, context=None, unit="us",
               measurement="interleaved median CUDA-event kernel latency",
               software="torch=x; triton=y; cuda_runtime=z"):
    context = context or space().context
    candidate_id = candidate_identifier(config, context)
    metadata = EvaluationMetadata(
        context.hardware_target, context.backend, (context.m, context.n, context.k),
        context.precision, "2026-10-13T12:00:00Z", software,
    )
    if status is EvaluationStatus.FAILED:
        return EvaluationRecord(
            evaluation_id=EvaluationIdentity(identifier or f"evaluation-{candidate_id}"),
            candidate_id=CandidateIdentity(candidate_id), status=status,
            lifecycle=(EvaluationStatus.PENDING, EvaluationStatus.COMPILATION,
                       EvaluationStatus.FAILED), metadata=metadata,
            compilation=CompilationResult(False),
            failure=FailureInfo(FailureCategory.COMPILATION, "controlled failure"),
            extensions={"candidate_parameters": config.to_dict()},
        )
    return EvaluationRecord(
        evaluation_id=EvaluationIdentity(identifier or f"evaluation-{candidate_id}"),
        candidate_id=CandidateIdentity(candidate_id), status=EvaluationStatus.COMPLETED,
        lifecycle=(EvaluationStatus.PENDING, EvaluationStatus.COMPILATION,
                   EvaluationStatus.CORRECTNESS_VALIDATION, EvaluationStatus.BENCHMARKING,
                   EvaluationStatus.COMPLETED), metadata=metadata,
        compilation=CompilationResult(True), correctness=CorrectnessResult(True),
        benchmark=BenchmarkResult(latency, measurement, unit, 100, synthetic),
        extensions={"candidate_parameters": config.to_dict()},
    )


class AdaptiveSamplingTests(unittest.TestCase):
    def test_initial_distribution_is_uniform(self):
        sampler = AdaptiveSampler(space(), seed=5)
        probabilities = list(sampler.candidate_probabilities().values())
        self.assertEqual(len(probabilities), 16)
        self.assertTrue(all(value == probabilities[0] for value in probabilities))
        self.assertAlmostEqual(sum(probabilities), 1.0)

    def test_lower_latency_increases_relative_probability(self):
        search = space()
        candidates = search.valid_candidates()
        fast = next(item for item in candidates if item.block_m == 64 and item.num_warps == 4)
        slow = next(item for item in candidates if item.block_m == 32 and item.num_warps == 8)
        sampler = AdaptiveSampler(search, seed=3, epsilon=0.2, exclude_observed_from_unique=False)
        diagnostics = sampler.update(
            [evaluation(fast, 50.0), evaluation(slow, 100.0)], source_id="experiment-a")
        probabilities = sampler.candidate_probabilities()
        self.assertGreater(probabilities[fast], probabilities[slow])
        self.assertGreater(sampler.parameter_scores()["block_m"][64],
                           sampler.parameter_scores()["block_m"][32])
        self.assertEqual(diagnostics["eligible_candidate_count"], 2)
        self.assertIn("parameter_scores_before", diagnostics)
        self.assertIn("parameter_scores_after", diagnostics)

    def test_epsilon_preserves_candidate_level_exploration(self):
        search = space()
        candidates = search.valid_candidates()
        sampler = AdaptiveSampler(search, seed=1, epsilon=0.4,
                                  exclude_observed_from_unique=False)
        sampler.update([evaluation(candidates[0], 10.0), evaluation(candidates[-1], 1000.0)],
                       source_id="experiment-a")
        lower_bound = 0.4 / len(candidates)
        self.assertTrue(all(probability >= lower_bound - 1e-15
                            for probability in sampler.candidate_probabilities().values()))

    def test_fixed_seed_unique_sampling_and_exhaustion(self):
        first = AdaptiveSampler(space(), seed=11, exclude_observed_from_unique=False)
        second = AdaptiveSampler(space(), seed=11, exclude_observed_from_unique=False)
        self.assertEqual(first.sample(16, unique=True), second.sample(16, unique=True))
        with self.assertRaises(SearchSpaceExhausted):
            first.sample(unique=True)

    def test_no_eligible_feedback_retains_uniform_distribution(self):
        search = space()
        config = search.valid_candidates()[0]
        sampler = AdaptiveSampler(search, seed=1)
        diagnostics = sampler.update(
            [evaluation(config, synthetic=True), evaluation(config, status=EvaluationStatus.FAILED,
                                                            identifier="failed")],
            source_id="experiment-a",
        )
        probabilities = list(sampler.candidate_probabilities().values())
        self.assertTrue(all(value == probabilities[0] for value in probabilities))
        self.assertEqual(diagnostics["eligible_evaluation_count"], 0)
        self.assertEqual(diagnostics["exclusion_reasons"]["synthetic_measurement"], 1)
        self.assertEqual(diagnostics["failure_categories"]["compilation_failure"], 1)

    def test_context_measurement_and_identity_exclusions(self):
        search = space()
        config = search.valid_candidates()[0]
        wrong_context = SearchContext(256, 3072, 768, "fp16", hardware_target="sm_89")
        records = [
            evaluation(config, context=wrong_context, identifier="wrong-workload"),
            evaluation(config, unit="ms", identifier="wrong-unit"),
            evaluation(config, measurement="host wall latency", identifier="wrong-method"),
        ]
        tampered = evaluation(config, identifier="tampered")
        object.__setattr__(tampered, "candidate_id", CandidateIdentity("candidate-tampered"))
        records.append(tampered)
        sampler = AdaptiveSampler(search, seed=1)
        diagnostics = sampler.update(records, source_id="experiment-a")
        reasons = diagnostics["exclusion_reasons"]
        self.assertEqual(reasons["workload_mismatch"], 1)
        self.assertEqual(reasons["unsupported_measurement_unit"], 1)
        self.assertEqual(reasons["unsupported_measurement_type"], 1)
        self.assertEqual(reasons["candidate_identity_mismatch"], 1)

    def test_backend_precision_and_hardware_mismatches_are_excluded(self):
        search = space()
        config = search.valid_candidates()[0]
        contexts = (
            (SearchContext(512, 3072, 768, "bf16", hardware_target="sm_89"),
             "precision_mismatch"),
            (SearchContext(512, 3072, 768, "fp16", backend="cuda",
                           hardware_target="sm_89"), "backend_mismatch"),
            (SearchContext(512, 3072, 768, "fp16", hardware_target="sm_80"),
             "hardware_target_mismatch"),
        )
        records = [evaluation(config, context=context, identifier=f"mismatch-{index}")
                   for index, (context, _) in enumerate(contexts)]
        diagnostics = AdaptiveSampler(search, seed=1).update(
            records, source_id="experiment-a")
        for _, reason in contexts:
            self.assertEqual(diagnostics["exclusion_reasons"][reason], 1)

    def test_repeated_evaluations_are_aggregated_once_and_ids_deduplicated(self):
        search = space()
        config = search.valid_candidates()[0]
        first = evaluation(config, 40.0, identifier="one")
        second = evaluation(config, 80.0, identifier="two")
        sampler = AdaptiveSampler(search, seed=1)
        diagnostics = sampler.update([first, first, second], source_id="experiment-a")
        self.assertEqual(diagnostics["eligible_evaluation_count"], 2)
        self.assertEqual(diagnostics["eligible_candidate_count"], 1)
        self.assertEqual(diagnostics["exclusion_reasons"]["duplicate_evaluation_id"], 1)
        self.assertEqual(next(iter(diagnostics["candidate_latencies_us"].values())), 60.0)

    def test_unmatched_sources_and_software_environments_are_not_combined(self):
        search = space()
        one, two = search.valid_candidates()[:2]
        sampler = AdaptiveSampler(search, seed=1)
        diagnostics = sampler.update(
            [evaluation(one, software="environment-a", identifier="a"),
             evaluation(two, software="environment-b", identifier="b")],
            source_id="experiment-a",
        )
        self.assertEqual(diagnostics["eligible_evaluation_count"], 1)
        self.assertEqual(diagnostics["exclusion_reasons"]["software_environment_mismatch"], 1)
        with self.assertRaisesRegex(ValueError, "unmatched feedback sources"):
            sampler.update([evaluation(one, identifier="c")], source_id="experiment-b")

    def test_second_update_from_same_source_is_explicitly_rejected(self):
        search = space()
        record = evaluation(search.valid_candidates()[0])
        sampler = AdaptiveSampler(search, seed=1)
        sampler.update([record], source_id="experiment-a")
        with self.assertRaisesRegex(ValueError, "one offline feedback update"):
            sampler.update([record], source_id="experiment-a")

    def test_state_round_trip_preserves_exact_rng_and_duplicate_state(self):
        search = space()
        observed = search.valid_candidates()[:2]
        sampler = AdaptiveSampler(search, seed=19)
        sampler.update([evaluation(observed[0], 50.0), evaluation(observed[1], 80.0)],
                       source_id="experiment-a")
        sampler.sample(3, unique=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            write_adaptive_state(path, sampler)
            restored = read_adaptive_state(path)
        self.assertEqual(sampler.to_dict(), restored.to_dict())
        self.assertEqual(sampler.sample(5, unique=True), restored.sample(5, unique=True))

    def test_feedback_update_does_not_mutate_records(self):
        config = space().valid_candidates()[0]
        record = evaluation(config)
        before = deepcopy(record.to_dict())
        AdaptiveSampler(space(), seed=1).update([record], source_id="experiment-a")
        self.assertEqual(record.to_dict(), before)

    def test_interaction_pattern_exposes_parameter_model_limitation(self):
        search = TritonGemmSearchSpace(
            {"block_m": [32, 64], "block_n": [64, 128], "block_k": [32],
             "num_warps": [4], "num_stages": [2]}, space().context,
            identifier="interaction-test",
        )
        configs = {(_config.block_m, _config.block_n): _config
                   for _config in search.valid_candidates()}
        records = [
            evaluation(configs[(32, 64)], 50.0, identifier="a"),
            evaluation(configs[(64, 128)], 50.0, identifier="b"),
            evaluation(configs[(32, 128)], 100.0, identifier="c"),
            evaluation(configs[(64, 64)], 100.0, identifier="d"),
        ]
        sampler = AdaptiveSampler(search, seed=1, exclude_observed_from_unique=False)
        sampler.update(records, source_id="experiment-a")
        probabilities = list(sampler.candidate_probabilities().values())
        self.assertTrue(all(abs(value - probabilities[0]) < 1e-15 for value in probabilities))


if __name__ == "__main__":
    unittest.main()
