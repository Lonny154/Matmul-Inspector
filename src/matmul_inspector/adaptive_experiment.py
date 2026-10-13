"""CPU-only paired experiments comparing uniform and adaptive sampling."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence
import uuid

from .adaptive_sampling import AdaptiveSampler
from .evaluation import (
    BenchmarkResult, CandidateIdentity, CompilationResult, CorrectnessResult,
    EvaluationIdentity, EvaluationLifecycle, EvaluationMetadata, EvaluationRecord,
    EvaluationStatus, FailureCategory, FailureInfo,
)
from .sampling import (
    PARAMETER_NAMES, SamplingRecord, SearchSpaceExhausted, TritonGemmConfig,
    TritonGemmSearchSpace, UniformSampler, candidate_identifier,
)


TRIAL_SCHEMA_VERSION = 1
ARM_NAMES = ("uniform", "adaptive")
LANDSCAPE_KINDS = ("structured", "interaction", "flat")
_FIXED_TIMESTAMP = "2000-01-01T00:00:00Z"


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def derive_seed(master_seed: int, *parts: object) -> int:
    """Derive an order-independent 64-bit seed and preserve its provenance."""
    if isinstance(master_seed, bool) or not isinstance(master_seed, int) or master_seed < 0:
        raise ValueError("master_seed must be a non-negative integer")
    payload = _canonical([master_seed, *parts])
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


@dataclass(frozen=True)
class SimulatedLandscape:
    """Deterministic latency surface with optional order-independent noise/failures."""

    kind: str = "structured"
    seed: int = 0
    noise_us: float = 0.0
    failure_modulus: int | None = None
    failure_category: FailureCategory = FailureCategory.COMPILATION

    def __post_init__(self) -> None:
        if self.kind not in LANDSCAPE_KINDS:
            raise ValueError(f"landscape kind must be one of: {', '.join(LANDSCAPE_KINDS)}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("landscape seed must be a non-negative integer")
        if (isinstance(self.noise_us, bool) or not isinstance(self.noise_us, (int, float))
                or not math.isfinite(self.noise_us) or self.noise_us < 0):
            raise ValueError("noise_us must be finite and non-negative")
        if self.failure_modulus is not None and (
                isinstance(self.failure_modulus, bool) or not isinstance(self.failure_modulus, int)
                or self.failure_modulus < 2):
            raise ValueError("failure_modulus must be null or an integer of at least two")
        if not isinstance(self.failure_category, FailureCategory):
            raise ValueError("failure_category must be a FailureCategory")

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "seed": self.seed, "noise_us": float(self.noise_us),
                "failure_modulus": self.failure_modulus,
                "failure_category": self.failure_category.value}

    def _fraction(self, label: str, config: TritonGemmConfig) -> float:
        raw = _digest([self.seed, label, config.to_dict()])
        return int(raw[:16], 16) / float(2**64 - 1)

    def latency(self, config: TritonGemmConfig) -> float:
        if self.kind == "flat":
            base = 100.0
        elif self.kind == "interaction":
            aligned = (config.block_m <= 32) == (config.block_n <= 64)
            base = 70.0 if aligned else 115.0
            base += abs(config.num_warps - 4) * 1.5 + abs(config.num_stages - 2)
        else:
            base = (145.0 - 0.30 * config.block_m - 0.12 * config.block_n
                    + 0.10 * config.block_k + abs(config.num_warps - 4) * 3.0
                    + abs(config.num_stages - 3) * 2.0)
        noise = (2.0 * self._fraction("noise", config) - 1.0) * float(self.noise_us)
        return max(1e-9, base + noise)

    def failure_for(self, config: TritonGemmConfig) -> FailureCategory | None:
        if self.failure_modulus is None:
            return None
        bucket = int(_digest([self.seed, "failure", config.to_dict()])[:16], 16)
        return self.failure_category if bucket % self.failure_modulus == 0 else None

    def ranked_candidates(
        self, search_space: TritonGemmSearchSpace,
    ) -> tuple[TritonGemmConfig, ...]:
        return tuple(sorted(search_space.valid_candidates(),
                            key=lambda item: (self.latency(item), tuple(item.to_dict().values()))))


class SimulatedEvaluator:
    """Existing-contract evaluator whose outcomes depend only on candidate and landscape."""

    def __init__(self, landscape: SimulatedLandscape, *, trial_id: str, arm_id: str) -> None:
        self.landscape = landscape
        self.trial_id = trial_id
        self.arm_id = arm_id

    def evaluate(self, candidate: SamplingRecord) -> EvaluationRecord:
        if not isinstance(candidate, SamplingRecord):
            raise TypeError("candidate must be a SamplingRecord")
        lifecycle = EvaluationLifecycle()
        failure = self.landscape.failure_for(candidate.parameters)
        metadata = EvaluationMetadata(
            candidate.context.hardware_target, candidate.context.backend,
            (candidate.context.m, candidate.context.n, candidate.context.k),
            candidate.context.precision, _FIXED_TIMESTAMP, "simulated-evaluator-1",
        )
        evaluation_id = EvaluationIdentity(
            "evaluation-" + _digest([self.trial_id, self.arm_id,
                                      candidate.sampling_order, candidate.candidate_id])[:24]
        )
        common = dict(evaluation_id=evaluation_id,
                      candidate_id=CandidateIdentity(candidate.candidate_id), metadata=metadata,
                      extensions={"candidate_parameters": candidate.parameters.to_dict(),
                                  "measurement_provenance": "cpu_simulation",
                                  "landscape": self.landscape.to_dict()})
        if failure is FailureCategory.INFRASTRUCTURE:
            lifecycle.transition(EvaluationStatus.FAILED)
            return EvaluationRecord(status=lifecycle.status, lifecycle=lifecycle.history,
                                    failure=FailureInfo(failure, "simulated infrastructure failure"),
                                    **common)
        lifecycle.transition(EvaluationStatus.COMPILATION)
        if failure in (FailureCategory.COMPILATION, FailureCategory.INVALID_CANDIDATE):
            lifecycle.transition(EvaluationStatus.FAILED)
            return EvaluationRecord(
                status=lifecycle.status, lifecycle=lifecycle.history,
                compilation=CompilationResult(False, "simulated-evaluator-1",
                                              "simulated compilation failure"),
                failure=FailureInfo(failure, "simulated candidate failure"), **common)
        compilation = CompilationResult(True, "simulated-evaluator-1",
                                        "simulated compilation succeeded")
        lifecycle.transition(EvaluationStatus.CORRECTNESS_VALIDATION)
        if failure is FailureCategory.CORRECTNESS:
            lifecycle.transition(EvaluationStatus.FAILED)
            return EvaluationRecord(
                status=lifecycle.status, lifecycle=lifecycle.history, compilation=compilation,
                correctness=CorrectnessResult(False, "simulated mismatch"),
                failure=FailureInfo(failure, "simulated correctness failure"), **common)
        correctness = CorrectnessResult(True, "simulated correctness passed")
        lifecycle.transition(EvaluationStatus.BENCHMARKING)
        if failure is FailureCategory.BENCHMARK:
            lifecycle.transition(EvaluationStatus.FAILED)
            return EvaluationRecord(
                status=lifecycle.status, lifecycle=lifecycle.history, compilation=compilation,
                correctness=correctness,
                failure=FailureInfo(failure, "simulated benchmark failure"), **common)
        lifecycle.transition(EvaluationStatus.COMPLETED)
        return EvaluationRecord(
            status=lifecycle.status, lifecycle=lifecycle.history, compilation=compilation,
            correctness=correctness,
            benchmark=BenchmarkResult(self.landscape.latency(candidate.parameters),
                                      "simulated objective latency", "us", 1, True), **common)


@dataclass(frozen=True)
class TrialConfig:
    evaluation_budget: int
    batch_size: int
    replicates: int
    master_seed: int
    epsilon: float = 0.25
    prior_strength: float = 2.0

    def __post_init__(self) -> None:
        for name in ("evaluation_budget", "batch_size", "replicates"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.batch_size > self.evaluation_budget:
            raise ValueError("batch_size cannot exceed evaluation_budget")
        if isinstance(self.master_seed, bool) or not isinstance(self.master_seed, int) \
                or self.master_seed < 0:
            raise ValueError("master_seed must be a non-negative integer")


@dataclass
class _ArmState:
    name: str
    records: list[EvaluationRecord] = field(default_factory=list)
    candidates: list[TritonGemmConfig] = field(default_factory=list)
    metrics: list[dict[str, Any]] = field(default_factory=list)
    rebuilds: list[dict[str, Any]] = field(default_factory=list)
    proposal_attempts: int = 0
    duplicates: int = 0
    invalid: int = 0
    completed_rounds: int = 0
    termination: str = "budget_exhausted"


class PairedTrialRunner:
    """Run paired arms with an equal submitted-evaluation budget."""

    def __init__(self, search_space: TritonGemmSearchSpace,
                 landscape: SimulatedLandscape, config: TrialConfig) -> None:
        self.search_space = search_space
        self.landscape = landscape
        self.config = config
        if config.evaluation_budget > len(search_space.valid_candidates()):
            raise ValueError("evaluation_budget exceeds the unique valid candidate count")

    def run(self) -> dict[str, Any]:
        trial_id = "trial-" + _digest({
            "space": self._space_definition(), "landscape": self.landscape.to_dict(),
            "config": asdict(self.config),
        })[:20]
        replicates = [self._run_replicate(trial_id, index)
                      for index in range(self.config.replicates)]
        return {
            "schema_version": TRIAL_SCHEMA_VERSION,
            "trial_id": trial_id,
            "experiment_type": "paired_uniform_vs_adaptive_cpu_simulation",
            "search_space": self._space_definition(),
            "landscape": self.landscape.to_dict(),
            "config": asdict(self.config),
            "replicates": replicates,
            "aggregate": self._aggregate(replicates),
            "extensions": {},
        }

    def _run_replicate(self, trial_id: str, replicate: int) -> dict[str, Any]:
        seeds = {
            "initialization": derive_seed(self.config.master_seed, replicate, "initialization"),
            "uniform": derive_seed(self.config.master_seed, replicate, "uniform"),
            "adaptive_rounds": [derive_seed(self.config.master_seed, replicate, "adaptive", round_index)
                                for round_index in range(math.ceil(
                                    self.config.evaluation_budget / self.config.batch_size))],
        }
        uniform_sampler = UniformSampler(self.search_space, seed=seeds["uniform"])
        initial_sampler = UniformSampler(self.search_space, seed=seeds["initialization"])
        initial_count = min(self.config.batch_size, self.config.evaluation_budget)
        initial = initial_sampler.sample(initial_count, unique=True)
        # Advance the uniform arm through its own deterministic sequence while excluding
        # the paired initialization candidates.
        uniform_sampler._seen.update(initial)
        states = {name: _ArmState(name) for name in ARM_NAMES}
        next_batches = {"uniform": list(initial), "adaptive": list(initial)}
        round_index = 0
        infrastructure_stop = False
        while len(states["uniform"].records) < self.config.evaluation_budget:
            remaining = self.config.evaluation_budget - len(states["uniform"].records)
            count = min(self.config.batch_size, remaining)
            if round_index > 0:
                next_batches["uniform"] = uniform_sampler.sample(count, unique=True)
                next_batches["adaptive"] = self._adaptive_batch(
                    states["adaptive"], seeds["adaptive_rounds"][round_index], count,
                    trial_id, replicate, round_index,
                )
            for offset in range(count):
                for arm_name in ARM_NAMES:
                    state = states[arm_name]
                    config = next_batches[arm_name][offset]
                    state.proposal_attempts += 1
                    if config in state.candidates:
                        state.duplicates += 1
                        raise RuntimeError("a sampler proposed a duplicate unique candidate")
                    if self.search_space.validation_errors(config):
                        state.invalid += 1
                        raise RuntimeError("a sampler proposed an invalid candidate")
                    proposal_seed = seeds["initialization"] if round_index == 0 else (
                        seeds["uniform"] if arm_name == "uniform"
                        else seeds["adaptive_rounds"][round_index]
                    )
                    sampling = self._sampling_record(config, state, proposal_seed, arm_name)
                    result = SimulatedEvaluator(
                        self.landscape, trial_id=f"{trial_id}-r{replicate}", arm_id=arm_name,
                    ).evaluate(sampling)
                    state.candidates.append(config)
                    state.records.append(result)
                    self._append_metric(state)
                    if result.failure and result.failure.category is FailureCategory.INFRASTRUCTURE:
                        infrastructure_stop = True
                if infrastructure_stop:
                    break
            for state in states.values():
                state.completed_rounds += 1
                if infrastructure_stop:
                    state.termination = "paired_infrastructure_stop"
            if infrastructure_stop:
                break
            round_index += 1
        return {
            "replicate": replicate, "seeds": seeds,
            "arms": {name: self._arm_result(state) for name, state in states.items()},
            "paired_budget_equal": (len(states["uniform"].records)
                                    == len(states["adaptive"].records)),
        }

    def _adaptive_batch(self, state: _ArmState, seed: int, count: int,
                        trial_id: str, replicate: int, round_index: int) -> list[TritonGemmConfig]:
        sampler = AdaptiveSampler(
            self.search_space, seed=seed, epsilon=self.config.epsilon,
            prior_strength=self.config.prior_strength, exclude_observed_from_unique=True,
            allow_synthetic_feedback=True,
        )
        source_records = self._deduplicate(state.records)
        source_id = f"{trial_id}-r{replicate}-adaptive-cumulative-round-{round_index}"
        diagnostics = sampler.update(source_records, source_id=source_id)
        # Failed evaluations consume budget and are not eligible feedback, but they
        # still must not be proposed again within this unique-search trial.
        sampler._seen.update(state.candidates)
        ids = [record.evaluation_id.value for record in source_records]
        state.rebuilds.append({
            "round": round_index, "source_id": source_id,
            "source_evaluation_ids": ids,
            "source_candidate_ids": [record.candidate_id.value for record in source_records],
            "feedback_count": diagnostics["eligible_evaluation_count"],
            "feedback_set_digest": _digest([record.to_dict() for record in source_records]),
            "sampler_config": {"epsilon": self.config.epsilon,
                               "prior_strength": self.config.prior_strength,
                               "exclude_observed_from_unique": True,
                               "allow_synthetic_feedback": True},
            "proposal_seed": seed,
            "proposal_rng_state_before": repr(sampler._random.getstate()),
        })
        return sampler.sample(count, unique=True)

    @staticmethod
    def _deduplicate(records: Sequence[EvaluationRecord]) -> list[EvaluationRecord]:
        by_id: dict[str, EvaluationRecord] = {}
        for record in records:
            by_id.setdefault(record.evaluation_id.value, record)
        return list(by_id.values())

    def _sampling_record(self, config: TritonGemmConfig, state: _ArmState,
                         seed: int, arm_name: str) -> SamplingRecord:
        return SamplingRecord(
            candidate_identifier(config, self.search_space.context), self.search_space.identifier,
            arm_name, seed, config, self.search_space.context, len(state.records), _FIXED_TIMESTAMP,
            {"simulated": True},
        )

    def _append_metric(self, state: _ArmState) -> None:
        successful = [record for record in state.records
                      if record.status is EvaluationStatus.COMPLETED]
        observed_ids = {record.candidate_id.value for record in successful}
        ranked = self.landscape.ranked_candidates(self.search_space)
        rank_by_id = {candidate_identifier(config, self.search_space.context): index + 1
                      for index, config in enumerate(ranked)}
        best = min(successful, key=lambda item: item.benchmark.latency) if successful else None
        best_latency = best.benchmark.latency if best else None
        optimum = self.landscape.latency(ranked[0])
        first_optimum = next((item["evaluations_consumed"] for item in state.metrics
                              if item.get("best_observed_rank") == 1), None)
        if best is not None and rank_by_id[best.candidate_id.value] == 1 and first_optimum is None:
            first_optimum = len(state.records)
        state.metrics.append({
            "evaluations_consumed": len(state.records),
            "best_observed_latency_us": best_latency,
            "simple_regret_us": None if best_latency is None else best_latency - optimum,
            "best_observed_rank": None if best is None else rank_by_id[best.candidate_id.value],
            "evaluations_to_known_optimum": first_optimum,
            "unique_coverage": len(set(state.candidates)) / len(ranked),
            "successful_evaluations": len(successful),
            "eligible_feedback_count": len(observed_ids),
        })

    def _arm_result(self, state: _ArmState) -> dict[str, Any]:
        failures = Counter(record.failure.category.value for record in state.records
                           if record.failure is not None)
        successful = sum(record.status is EvaluationStatus.COMPLETED for record in state.records)
        return {
            "arm_id": state.name,
            "budget": {
                "proposal_attempts": state.proposal_attempts,
                "duplicate_proposals": state.duplicates,
                "invalid_proposals": state.invalid,
                "submitted_evaluations": len(state.records),
                "successful_evaluations": successful,
                "failed_evaluations": len(state.records) - successful,
                "failures_by_category": dict(sorted(failures.items())),
                "unique_candidates_evaluated": len(set(state.candidates)),
                "eligible_feedback_records": successful,
                "completed_rounds": state.completed_rounds,
            },
            "termination": state.termination,
            "candidate_ids": [candidate_identifier(item, self.search_space.context)
                              for item in state.candidates],
            "evaluations": [record.to_dict() for record in state.records],
            "metrics": state.metrics,
            "rebuilds": state.rebuilds,
        }

    def _space_definition(self) -> dict[str, Any]:
        return {
            "identifier": self.search_space.identifier,
            "values": {name: list(self.search_space.values[name]) for name in PARAMETER_NAMES},
            "context": asdict(self.search_space.context),
            "valid_candidate_ids": [candidate_identifier(item, self.search_space.context)
                                    for item in self.search_space.valid_candidates()],
        }

    @staticmethod
    def _aggregate(replicates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {"replicate_count": len(replicates), "arms": {}}
        for arm in ARM_NAMES:
            final = [item["arms"][arm]["metrics"][-1] for item in replicates]
            latencies = [item["best_observed_latency_us"] for item in final
                         if item["best_observed_latency_us"] is not None]
            regrets = [item["simple_regret_us"] for item in final
                       if item["simple_regret_us"] is not None]
            result["arms"][arm] = {
                "completed_replicates": len(latencies),
                "mean_best_observed_latency_us": statistics.fmean(latencies) if latencies else None,
                "median_best_observed_latency_us": statistics.median(latencies) if latencies else None,
                "mean_simple_regret_us": statistics.fmean(regrets) if regrets else None,
                "raw_final_metrics": final,
            }
        paired = []
        for item in replicates:
            uniform = item["arms"]["uniform"]["metrics"][-1]
            adaptive = item["arms"]["adaptive"]["metrics"][-1]
            if (uniform["best_observed_latency_us"] is None
                    or adaptive["best_observed_latency_us"] is None):
                delta = None
            else:
                delta = (adaptive["best_observed_latency_us"]
                         - uniform["best_observed_latency_us"])
            paired.append({"replicate": item["replicate"],
                           "adaptive_minus_uniform_best_latency_us": delta})
        deltas = [item["adaptive_minus_uniform_best_latency_us"] for item in paired
                  if item["adaptive_minus_uniform_best_latency_us"] is not None]
        result["paired"] = {
            "raw_deltas": paired,
            "mean_adaptive_minus_uniform_best_latency_us": (
                statistics.fmean(deltas) if deltas else None),
            "adaptive_lower_count": sum(delta < 0 for delta in deltas),
            "uniform_lower_count": sum(delta > 0 for delta in deltas),
            "tie_count": sum(delta == 0 for delta in deltas),
        }
        return result


def validate_trial_result(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("trial result must be an object")
    expected = {"schema_version", "trial_id", "experiment_type", "search_space",
                "landscape", "config", "replicates", "aggregate", "extensions"}
    missing, unknown = sorted(expected - set(value)), sorted(set(value) - expected)
    if missing:
        raise ValueError(f"version-1 trial result missing fields: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"unknown version-1 trial result fields: {', '.join(unknown)}")
    if value["schema_version"] != TRIAL_SCHEMA_VERSION:
        raise ValueError(f"unsupported trial result schema: {value['schema_version']!r}")
    if not isinstance(value["extensions"], Mapping):
        raise ValueError("trial result extensions must be an object")
    if not isinstance(value["replicates"], list) or not value["replicates"]:
        raise ValueError("trial result replicates must be a non-empty array")
    for replicate in value["replicates"]:
        if not isinstance(replicate, Mapping) or set(replicate) != {
                "replicate", "seeds", "arms", "paired_budget_equal"}:
            raise ValueError("malformed trial replicate")
        if set(replicate["arms"]) != set(ARM_NAMES):
            raise ValueError("trial replicate must contain uniform and adaptive arms")
        for arm in replicate["arms"].values():
            for raw in arm.get("evaluations", []):
                EvaluationRecord.from_dict(raw)
    return json.loads(json.dumps(value, allow_nan=False))


def write_trial_result(path: str | Path, value: Mapping[str, Any]) -> None:
    validated = validate_trial_result(value)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(validated, indent=2, sort_keys=True,
                                        allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_trial_result(path: str | Path) -> dict[str, Any]:
    try:
        return validate_trial_result(json.loads(Path(path).read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read trial result: {error}") from error
