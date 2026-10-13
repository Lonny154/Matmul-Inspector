"""Reproducible paired hardware search orchestration with a safe dry-run default."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Callable, Mapping, Protocol, Sequence
import uuid

from .adaptive_experiment import derive_seed
from .adaptive_sampling import AdaptiveSampler
from .controlled_benchmark import (
    ControlledBenchmarkBackend, ControlledBenchmarkConfig, ControlledBenchmarkRunner,
    write_controlled_benchmark,
)
from .evaluation import (
    EvaluationRecord, EvaluationStatus, FailureCategory, write_evaluation_jsonl,
)
from .sampling import (
    PARAMETER_NAMES, SamplingRecord, SearchContext, TritonGemmConfig,
    TritonGemmSearchSpace, UniformSampler, candidate_identifier,
)


HARDWARE_SEARCH_SCHEMA_VERSION = 1
ARM_NAMES = ("uniform", "adaptive")
_MANIFEST_FIELDS = {
    "schema_version", "experiment_id", "mode", "status", "created_at", "search_space",
    "config", "dependency_check", "trial_plans", "trials", "accounting",
    "environment", "artifacts", "limitations", "extensions",
}


class CandidateEvaluator(Protocol):
    def evaluate(self, candidate: SamplingRecord) -> EvaluationRecord: ...


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def default_fc1_search_space() -> TritonGemmSearchSpace:
    """Return the bounded configuration space used by the initial SM 8.9 study."""
    return TritonGemmSearchSpace(
        {"block_m": [32, 64, 128], "block_n": [64, 128], "block_k": [32, 64],
         "num_warps": [4, 8], "num_stages": [2, 3]},
        SearchContext(512, 3072, 768, "fp16", hardware_target="sm_89"),
        identifier="fc1-adaptive-search-sm89-v1",
    )


@dataclass(frozen=True)
class HardwareSearchConfig:
    trial_seeds: tuple[int, ...] = (101, 202, 303)
    per_arm_budget: int = 16
    shared_initial_size: int = 5
    adaptive_batch_size: int = 2
    epsilon: float = 0.25
    prior_strength: float = 2.0
    latency_threshold_us: float | None = 62.0

    def __post_init__(self) -> None:
        if (not isinstance(self.trial_seeds, tuple) or not self.trial_seeds
                or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
                       for seed in self.trial_seeds)
                or len(set(self.trial_seeds)) != len(self.trial_seeds)):
            raise ValueError("trial_seeds must be a non-empty tuple of unique non-negative integers")
        for name in ("per_arm_budget", "shared_initial_size", "adaptive_batch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.shared_initial_size > self.per_arm_budget:
            raise ValueError("shared_initial_size cannot exceed per_arm_budget")
        for name in ("epsilon", "prior_strength"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.epsilon > 1:
            raise ValueError("epsilon cannot exceed one")
        if self.latency_threshold_us is not None and (
                isinstance(self.latency_threshold_us, bool)
                or not isinstance(self.latency_threshold_us, (int, float))
                or not math.isfinite(self.latency_threshold_us)
                or self.latency_threshold_us <= 0):
            raise ValueError("latency_threshold_us must be null or finite and positive")


def optional_gpu_dependency_check() -> dict[str, Any]:
    """Inspect module availability without importing GPU libraries or touching CUDA."""
    modules = {name: importlib.util.find_spec(name) is not None for name in ("torch", "triton")}
    return {
        "method": "importlib.util.find_spec only; modules were not imported",
        "modules": modules,
        "cuda_queried": False,
        "ready_for_execution_check": all(modules.values()),
        "note": "GPU availability and compute capability are checked only with --execute",
    }


def _space_dict(search_space: TritonGemmSearchSpace) -> dict[str, Any]:
    valid = search_space.valid_candidates()
    return {
        "identifier": search_space.identifier,
        "context": asdict(search_space.context),
        "values": {name: list(search_space.values[name]) for name in PARAMETER_NAMES},
        "valid_candidate_count": len(valid),
        "valid_candidate_ids": [candidate_identifier(item, search_space.context)
                                for item in valid],
    }


def _sample_record(search_space: TritonGemmSearchSpace, config: TritonGemmConfig,
                   *, sampler: str, seed: int, order: int, trial_id: str,
                   logical_arms: Sequence[str]) -> SamplingRecord:
    return SamplingRecord(
        candidate_id=candidate_identifier(config, search_space.context),
        search_space_id=search_space.identifier, sampler=sampler, seed=seed,
        parameters=config, context=search_space.context, sampling_order=order,
        timestamp="2000-01-01T00:00:00Z",
        extensions={"trial_id": trial_id, "logical_arms": list(logical_arms)},
    )


class HardwareSearchExperiment:
    """Plan or execute paired search while keeping GPU access behind ``execute``."""

    def __init__(self, search_space: TritonGemmSearchSpace,
                 config: HardwareSearchConfig = HardwareSearchConfig(), *,
                 clock: Callable[[], datetime] | None = None,
                 timer: Callable[[], float] | None = None) -> None:
        self.search_space = search_space
        self.config = config
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._timer = timer or time.perf_counter
        valid_count = len(search_space.valid_candidates())
        if config.per_arm_budget > valid_count:
            raise ValueError("per_arm_budget exceeds the valid candidate count")

    def dry_run(self) -> dict[str, Any]:
        return self._base_manifest("dry_run", "planned", self._trial_plans(), trials=[])

    def execute(self, evaluator_factory: Callable[[int, str], CandidateEvaluator]) -> dict[str, Any]:
        """Execute only after the caller has made an explicit hardware-mode choice."""
        if not callable(evaluator_factory):
            raise TypeError("evaluator_factory must be callable")
        plans = self._trial_plans()
        trials = []
        all_physical_records: list[EvaluationRecord] = []
        started = self._timer()
        for plan in plans:
            trial, physical = self._execute_trial(plan, evaluator_factory)
            trials.append(trial)
            all_physical_records.extend(physical)
        manifest = self._base_manifest("hardware", "completed", plans, trials)
        manifest["accounting"] = self._execution_accounting(trials)
        manifest["environment"] = _environment_from_records(all_physical_records)
        manifest["extensions"]["search_wall_clock_seconds"] = self._timer() - started
        return {"manifest": manifest, "evaluations": all_physical_records}

    def _trial_plans(self) -> list[dict[str, Any]]:
        plans = []
        for index, trial_seed in enumerate(self.config.trial_seeds):
            trial_id = f"trial-{index}-" + _digest([trial_seed, self.search_space.identifier])[:16]
            seeds = {
                "trial": trial_seed,
                "initial": derive_seed(trial_seed, "shared-initial"),
                "uniform": derive_seed(trial_seed, "uniform"),
                "schedule": derive_seed(trial_seed, "physical-schedule"),
                "adaptive_rounds": [
                    derive_seed(trial_seed, "adaptive", round_index)
                    for round_index in range(self._adaptive_round_count())
                ],
            }
            sampler = UniformSampler(self.search_space, seed=seeds["initial"])
            initial = sampler.sample(self.config.shared_initial_size, unique=True)
            boundaries = list(range(
                self.config.shared_initial_size,
                self.config.per_arm_budget,
                self.config.adaptive_batch_size,
            ))
            plans.append({
                "trial_index": index, "trial_id": trial_id, "seeds": seeds,
                "shared_initial_candidates": [
                    {"candidate_id": candidate_identifier(item, self.search_space.context),
                     "parameters": item.to_dict()} for item in initial
                ],
                "logical_budget_per_arm": self.config.per_arm_budget,
                "shared_logical_evaluations_per_arm": self.config.shared_initial_size,
                "shared_physical_evaluations": self.config.shared_initial_size,
                "maximum_physical_evaluations": (
                    2 * self.config.per_arm_budget - self.config.shared_initial_size),
                "adaptive_rebuild_after_logical_counts": boundaries,
                "future_adaptive_decisions": [
                    {"round": round_index + 1, "status": "pending_hardware_feedback"}
                    for round_index in range(self._adaptive_round_count())
                ],
            })
        return plans

    def _adaptive_round_count(self) -> int:
        remaining = self.config.per_arm_budget - self.config.shared_initial_size
        return math.ceil(remaining / self.config.adaptive_batch_size)

    def _base_manifest(self, mode: str, status: str, plans: list[dict[str, Any]],
                       trials: list[dict[str, Any]]) -> dict[str, Any]:
        created = self._clock().astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        experiment_id = "hardware-search-" + _digest({
            "space": _space_dict(self.search_space), "config": asdict(self.config),
        })[:20]
        logical = len(plans) * 2 * self.config.per_arm_budget
        physical = sum(plan["maximum_physical_evaluations"] for plan in plans)
        return {
            "schema_version": HARDWARE_SEARCH_SCHEMA_VERSION,
            "experiment_id": experiment_id, "mode": mode, "status": status,
            "created_at": created, "search_space": _space_dict(self.search_space),
            "config": {**asdict(self.config), "trial_seeds": list(self.config.trial_seeds)},
            "dependency_check": optional_gpu_dependency_check(),
            "trial_plans": plans, "trials": trials,
            "accounting": {
                "planned_logical_evaluations": logical,
                "maximum_physical_evaluations": physical,
                "physical_savings_from_shared_initial": (
                    len(plans) * self.config.shared_initial_size),
                "completed_logical_evaluations": 0,
                "completed_physical_evaluations": 0,
            },
            "environment": None,
            "artifacts": {"manifest": "experiment.json", "evaluations": "evaluations.jsonl",
                          "report": "report.md", "progress": "progress.json"},
            "limitations": [
                "dry-run does not query CUDA or execute kernels",
                "future adaptive choices depend on eligible hardware feedback",
                "search is exploratory; winners require separate controlled confirmation",
            ],
            "extensions": {},
        }

    def _execute_trial(self, plan: Mapping[str, Any], evaluator_factory: Callable[
            [int, str], CandidateEvaluator]) -> tuple[dict[str, Any], list[EvaluationRecord]]:
        trial_started = self._timer()
        histories: dict[str, list[dict[str, Any]]] = {name: [] for name in ARM_NAMES}
        feedback: dict[str, list[EvaluationRecord]] = {name: [] for name in ARM_NAMES}
        seen: dict[str, set[TritonGemmConfig]] = {name: set() for name in ARM_NAMES}
        physical: list[EvaluationRecord] = []
        execution_order: list[dict[str, Any]] = []
        rebuilds: list[dict[str, Any]] = []
        trial_seed = plan["seeds"]["trial"]
        evaluator = evaluator_factory(trial_seed, plan["trial_id"])
        initial = [TritonGemmConfig.from_dict(item["parameters"])
                   for item in plan["shared_initial_candidates"]]
        schedule_rng = random.Random(plan["seeds"]["schedule"])
        initial_order = list(initial)
        schedule_rng.shuffle(initial_order)
        pair_orientation = schedule_rng.randrange(2)
        paired_step = 0
        terminated = None
        for config in initial_order:
            sampled = _sample_record(
                self.search_space, config, sampler="shared_initial",
                seed=plan["seeds"]["initial"], order=len(physical),
                trial_id=plan["trial_id"], logical_arms=ARM_NAMES,
            )
            record = evaluator.evaluate(sampled)
            physical.append(record)
            execution_order.append(_execution_entry(len(physical) - 1, record, ARM_NAMES))
            for arm in ARM_NAMES:
                seen[arm].add(config)
                feedback[arm].append(record)
                histories[arm].append(_history_entry(config, record, shared=True))
            if _infrastructure_failure(record):
                terminated = "shared_initial_infrastructure_failure"
                break

        uniform_sampler = UniformSampler(self.search_space, seed=plan["seeds"]["uniform"])
        uniform_sampler._seen.update(seen["uniform"])
        round_index = 0
        while terminated is None and len(histories["uniform"]) < self.config.per_arm_budget:
            remaining = self.config.per_arm_budget - len(histories["uniform"])
            count = min(self.config.adaptive_batch_size, remaining)
            uniform_batch = uniform_sampler.sample(count, unique=True)
            adaptive_seed = plan["seeds"]["adaptive_rounds"][round_index]
            adaptive_sampler = AdaptiveSampler(
                self.search_space, seed=adaptive_seed, epsilon=self.config.epsilon,
                prior_strength=self.config.prior_strength,
                exclude_observed_from_unique=True,
            )
            unique_feedback = _deduplicate_records(feedback["adaptive"])
            source_id = f"{plan['trial_id']}-adaptive-cumulative-{round_index + 1}"
            diagnostics = adaptive_sampler.update(unique_feedback, source_id=source_id)
            adaptive_sampler._seen.update(seen["adaptive"])
            adaptive_batch = adaptive_sampler.sample(count, unique=True)
            rebuilds.append({
                "round": round_index + 1, "source_id": source_id,
                "source_evaluation_ids": [item.evaluation_id.value for item in unique_feedback],
                "feedback_digest": _digest([item.to_dict() for item in unique_feedback]),
                "eligible_feedback_count": diagnostics["eligible_evaluation_count"],
                "proposal_seed": adaptive_seed, "diagnostics": diagnostics,
            })
            for offset in range(count):
                paired = [("uniform", uniform_batch[offset]),
                          ("adaptive", adaptive_batch[offset])]
                if (pair_orientation + paired_step) % 2:
                    paired.reverse()
                paired_step += 1
                for arm, config in paired:
                    sampled = _sample_record(
                        self.search_space, config, sampler=arm,
                        seed=(plan["seeds"]["uniform"] if arm == "uniform" else adaptive_seed),
                        order=len(histories[arm]), trial_id=plan["trial_id"],
                        logical_arms=(arm,),
                    )
                    record = evaluator.evaluate(sampled)
                    physical.append(record)
                    execution_order.append(_execution_entry(len(physical) - 1, record, (arm,)))
                    seen[arm].add(config)
                    feedback[arm].append(record)
                    histories[arm].append(_history_entry(config, record, shared=False))
                    if _infrastructure_failure(record):
                        terminated = "infrastructure_failure"
                        break
                if terminated is not None:
                    break
            round_index += 1
        trial = {
            "trial_id": plan["trial_id"], "trial_seed": trial_seed,
            "status": terminated or "budget_exhausted",
            "arms": {arm: _arm_summary(histories[arm], self.search_space,
                                        self.config.latency_threshold_us)
                     for arm in ARM_NAMES},
            "shared_evaluation_ids": [item.evaluation_id.value
                                      for item in physical[:min(len(physical), len(initial_order))]],
            "physical_execution_order": execution_order,
            "physical_evaluation_count": len(physical),
            "adaptive_rebuilds": rebuilds,
            "paired_progress": _paired_progress(histories),
            "search_wall_clock_seconds": self._timer() - trial_started,
        }
        return trial, physical

    def _execution_accounting(self, trials: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        logical = sum(sum(trial["arms"][arm]["submitted_evaluations"] for arm in ARM_NAMES)
                      for trial in trials)
        physical = sum(trial["physical_evaluation_count"] for trial in trials)
        return {
            "planned_logical_evaluations": (
                len(self.config.trial_seeds) * 2 * self.config.per_arm_budget),
            "maximum_physical_evaluations": (
                len(self.config.trial_seeds)
                * (2 * self.config.per_arm_budget - self.config.shared_initial_size)),
            "physical_savings_from_shared_initial": logical - physical,
            "completed_logical_evaluations": logical,
            "completed_physical_evaluations": physical,
        }


def _deduplicate_records(records: Sequence[EvaluationRecord]) -> list[EvaluationRecord]:
    unique: dict[str, EvaluationRecord] = {}
    for record in records:
        unique.setdefault(record.evaluation_id.value, record)
    return list(unique.values())


def _infrastructure_failure(record: EvaluationRecord) -> bool:
    return bool(record.failure and record.failure.category is FailureCategory.INFRASTRUCTURE)


def _history_entry(config: TritonGemmConfig, record: EvaluationRecord,
                   *, shared: bool) -> dict[str, Any]:
    return {
        "logical_order": None, "candidate_id": record.candidate_id.value,
        "parameters": config.to_dict(), "evaluation_id": record.evaluation_id.value,
        "shared_physical_evaluation": shared, "status": record.status.value,
        "failure_category": record.failure.category.value if record.failure else None,
        "latency_us": record.benchmark.latency if record.benchmark else None,
    }


def _execution_entry(index: int, record: EvaluationRecord,
                     logical_arms: Sequence[str]) -> dict[str, Any]:
    return {"physical_order": index, "evaluation_id": record.evaluation_id.value,
            "candidate_id": record.candidate_id.value, "logical_arms": list(logical_arms),
            "status": record.status.value}


def _arm_summary(history: list[dict[str, Any]], search_space: TritonGemmSearchSpace,
                 threshold: float | None) -> dict[str, Any]:
    for index, item in enumerate(history):
        item["logical_order"] = index
    best = None
    progress = []
    failures = Counter()
    threshold_at = None
    for index, item in enumerate(history, 1):
        if item["failure_category"]:
            failures[item["failure_category"]] += 1
        if item["latency_us"] is not None:
            best = item["latency_us"] if best is None else min(best, item["latency_us"])
        if threshold is not None and best is not None and best <= threshold and threshold_at is None:
            threshold_at = index
        progress.append({"submitted_evaluations": index, "best_valid_latency_us": best})
    return {
        "submitted_evaluations": len(history),
        "successful_evaluations": sum(item["status"] == EvaluationStatus.COMPLETED.value
                                      for item in history),
        "failures_by_category": dict(sorted(failures.items())),
        "unique_candidate_count": len({item["candidate_id"] for item in history}),
        "unique_coverage": len({item["candidate_id"] for item in history})
                           / len(search_space.valid_candidates()),
        "evaluations_to_threshold": threshold_at,
        "best_valid_latency_us": best, "history": history, "progress": progress,
    }


def _paired_progress(histories: Mapping[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    count = min(len(histories["uniform"]), len(histories["adaptive"]))
    uniform_best = None
    adaptive_best = None
    result = []
    for index in range(count):
        uniform_latency = histories["uniform"][index]["latency_us"]
        adaptive_latency = histories["adaptive"][index]["latency_us"]
        if uniform_latency is not None:
            uniform_best = (uniform_latency if uniform_best is None
                            else min(uniform_best, uniform_latency))
        if adaptive_latency is not None:
            adaptive_best = (adaptive_latency if adaptive_best is None
                             else min(adaptive_best, adaptive_latency))
        delta = (None if uniform_best is None or adaptive_best is None
                 else adaptive_best - uniform_best)
        result.append({
            "equal_logical_budget": index + 1,
            "uniform_best_latency_us": uniform_best,
            "adaptive_best_latency_us": adaptive_best,
            "adaptive_minus_uniform_best_latency_us": delta,
        })
    return result


def _environment_from_records(records: Sequence[EvaluationRecord]) -> Mapping[str, Any] | None:
    for record in records:
        environment = record.extensions.get("environment")
        if environment:
            return environment
    return None


def validate_hardware_search_manifest(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("hardware search manifest must be an object")
    if value.get("schema_version") != HARDWARE_SEARCH_SCHEMA_VERSION:
        raise ValueError(f"unsupported hardware search schema: {value.get('schema_version')!r}")
    missing, unknown = sorted(_MANIFEST_FIELDS - set(value)), sorted(set(value) - _MANIFEST_FIELDS)
    if missing:
        raise ValueError(f"hardware search manifest missing fields: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"unknown hardware search manifest fields: {', '.join(unknown)}")
    if value["mode"] not in {"dry_run", "hardware"}:
        raise ValueError("hardware search mode must be dry_run or hardware")
    if not isinstance(value["extensions"], Mapping):
        raise ValueError("hardware search extensions must be an object")
    if not isinstance(value["trial_plans"], list) or not value["trial_plans"]:
        raise ValueError("hardware search trial_plans must be a non-empty array")
    return json.loads(json.dumps(value, allow_nan=False))


def render_hardware_search_report(manifest: Mapping[str, Any]) -> str:
    value = validate_hardware_search_manifest(manifest)
    lines = [f"# Hardware adaptive-search experiment `{value['experiment_id']}`", ""]
    if value["mode"] == "dry_run":
        lines.extend([
            "**DRY RUN: no benchmark was run and CUDA was not queried.**", "",
            f"Valid configurations: {value['search_space']['valid_candidate_count']}",
            f"Trial replicates: {len(value['trial_plans'])}",
            f"Logical budget: {value['config']['per_arm_budget']} per arm",
            f"Shared initial evaluations: {value['config']['shared_initial_size']} per trial",
            f"Maximum physical evaluations: {value['accounting']['maximum_physical_evaluations']}",
            "", "Future adaptive batches are pending hardware feedback.",
        ])
    else:
        lines.extend(["This is exploratory hardware search evidence, not confirmation.", ""])
        for trial in value["trials"]:
            lines.append(f"## `{trial['trial_id']}`")
            lines.append("")
            lines.append("| Arm | Submitted | Best valid latency (us) |")
            lines.append("|---|---:|---:|")
            for arm in ARM_NAMES:
                row = trial["arms"][arm]
                best = "n/a" if row["best_valid_latency_us"] is None else f"{row['best_valid_latency_us']:.3f}"
                lines.append(f"| {arm} | {row['submitted_evaluations']} | {best} |")
            lines.append("")
    lines.extend(["", "A separate randomized interleaved confirmation is required for winners.", ""])
    return "\n".join(lines)


def write_hardware_search(directory: str | Path, result: Mapping[str, Any]) -> Path:
    destination = Path(directory)
    if destination.exists():
        raise FileExistsError(f"output directory already exists: {destination}")
    manifest = validate_hardware_search_manifest(result["manifest"])
    evaluations = tuple(result.get("evaluations", ()))
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    temporary.mkdir(parents=True)
    try:
        (temporary / "experiment.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        write_evaluation_jsonl(temporary / "evaluations.jsonl", evaluations)
        (temporary / "progress.json").write_text(
            json.dumps([{trial["trial_id"]: {arm: trial["arms"][arm]["progress"]
                                             for arm in ARM_NAMES}}
                        for trial in manifest["trials"]], indent=2, sort_keys=True,
                       allow_nan=False) + "\n", encoding="utf-8",
        )
        (temporary / "report.md").write_text(
            render_hardware_search_report(manifest), encoding="utf-8")
        temporary.replace(destination)
    except BaseException:
        for path in temporary.iterdir() if temporary.exists() else ():
            path.unlink()
        if temporary.exists():
            temporary.rmdir()
        raise
    return destination


def read_hardware_search_manifest(path: str | Path) -> dict[str, Any]:
    try:
        return validate_hardware_search_manifest(json.loads(Path(path).read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read hardware search manifest: {error}") from error


def confirm_search_winners(
    manifest: Mapping[str, Any], backend: ControlledBenchmarkBackend,
    output: str | Path, config: ControlledBenchmarkConfig,
) -> Path:
    """Separately confirm unique arm winners with the controlled benchmark runner."""
    winners = search_winner_records(manifest)
    result = ControlledBenchmarkRunner(backend).run(winners, config)
    return write_controlled_benchmark(output, result)


def search_winner_records(manifest: Mapping[str, Any]) -> tuple[SamplingRecord, ...]:
    """Recover and deduplicate the best successful candidate from every trial arm."""
    value = validate_hardware_search_manifest(manifest)
    if value["mode"] != "hardware":
        raise ValueError("confirmation requires a completed hardware search manifest")
    winners: dict[str, SamplingRecord] = {}
    context = SearchContext(**value["search_space"]["context"])
    for trial in value["trials"]:
        for arm in ARM_NAMES:
            successful = [item for item in trial["arms"][arm]["history"]
                          if item["latency_us"] is not None]
            if not successful:
                continue
            winner = min(successful, key=lambda item: item["latency_us"])
            parameters = TritonGemmConfig.from_dict(winner["parameters"])
            winners.setdefault(winner["candidate_id"], SamplingRecord(
                winner["candidate_id"], value["search_space"]["identifier"],
                "hardware_search_winner", 0, parameters, context, len(winners),
                "2000-01-01T00:00:00Z", {"source_experiment_id": value["experiment_id"]},
            ))
    if not winners:
        raise ValueError("hardware search has no successful candidates to confirm")
    return tuple(winners.values())
