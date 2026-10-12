"""Sequential, reproducible batches built on the single-experiment runner."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import statistics
import sys
import traceback
from typing import Any, Callable, Mapping, Sequence
import uuid

from .experiment import DEFAULT_CONFIG, run_experiment, validate_config


BATCH_SCHEMA_VERSION = 1
BATCH_EXIT_SUCCESS = 0
BATCH_EXIT_FAILURE = 1
BATCH_EXIT_CONFIGURATION = 2
BATCH_EXIT_INTERRUPTED = 130
_SHARED_FIELDS = {
    "workload", "seed", "correctness", "benchmark", "timeout_seconds", "baseline",
}


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def validate_batch_config(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Validate batch structure and shared settings, leaving candidates independent."""
    if not isinstance(raw, Mapping):
        raise ValueError("batch configuration must be a JSON object")
    expected = {"shared", "candidates", "scheduling_seed", "repetitions"}
    unknown = sorted(set(raw) - expected)
    missing = sorted(expected - set(raw))
    if unknown:
        raise ValueError(f"unknown batch configuration fields: {', '.join(unknown)}")
    if missing:
        raise ValueError(f"missing batch configuration fields: {', '.join(missing)}")
    config = json.loads(json.dumps(raw))
    shared = config["shared"]
    if not isinstance(shared, dict):
        raise ValueError("shared must be an object")
    shared_unknown = sorted(set(shared) - _SHARED_FIELDS)
    shared_missing = sorted(_SHARED_FIELDS - set(shared))
    if shared_unknown:
        raise ValueError(f"unknown shared fields: {', '.join(shared_unknown)}")
    if shared_missing:
        raise ValueError(f"missing shared fields: {', '.join(shared_missing)}")
    # Reuse the single-run schema for every shared numerical and benchmark field.
    placeholder_candidate = json.loads(json.dumps(DEFAULT_CONFIG["candidate"]))
    # BLOCK_K=1 prevents shared-workload validation from depending on the
    # repository's default candidate; each real candidate is validated below.
    placeholder_candidate["parameters"]["block_k"] = 1
    resolved = validate_config({**shared, "candidate": placeholder_candidate})
    config["shared"] = {field: resolved[field] for field in _SHARED_FIELDS}

    seed = config["scheduling_seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("scheduling_seed must be a non-negative integer")
    _positive_int(config["repetitions"], "repetitions")
    candidates = config["candidates"]
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("candidates must be a non-empty list")
    identifiers: list[str] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise ValueError(f"candidates[{index}] must be an object")
        identifier = candidate.get("id")
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError(f"candidates[{index}].id must be a non-empty string")
        identifiers.append(identifier)
    duplicates = sorted({identifier for identifier in identifiers if identifiers.count(identifier) > 1})
    if duplicates:
        raise ValueError(f"duplicate candidate IDs: {', '.join(duplicates)}")
    return config


def _candidate_config(shared: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    expected = {"id", "implementation", "parameters"}
    if set(candidate) != expected:
        missing = sorted(expected - set(candidate))
        unknown = sorted(set(candidate) - expected)
        details = []
        if missing:
            details.append(f"missing fields: {', '.join(missing)}")
        if unknown:
            details.append(f"unknown fields: {', '.join(unknown)}")
        raise ValueError("candidate " + "; ".join(details))
    return validate_config({
        **shared,
        "candidate": {
            "implementation": candidate.get("implementation"),
            "parameters": candidate.get("parameters"),
        },
    })


def _atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _entry(candidate_id: str, repetition: int | None, result: Mapping[str, Any]) -> dict[str, Any]:
    correctness = result.get("correctness") or {}
    benchmark = result.get("benchmark") or {}
    baseline = benchmark.get("baseline_summary") or {}
    candidate = benchmark.get("candidate_summary") or {}
    failure = result.get("failure") or {}
    status = result.get("status", "failed")
    if status == "success" and not (
            correctness.get("passed") is True
            and baseline.get("median_ms") is not None
            and candidate.get("median_ms") is not None
            and benchmark.get("speedup_baseline_over_candidate") is not None):
        status = "failed"
        failure = {
            "stage": "result", "exception_type": "MalformedExperimentResult",
            "message": "successful experiment omitted correctness or benchmark measurements",
            "traceback": None,
        }
    run_directory = result.get("run_directory")
    return {
        "repetition": repetition,
        "candidate_id": candidate_id,
        "status": status,
        "correctness_passed": correctness.get("passed"),
        "baseline_median_ms": baseline.get("median_ms"),
        "candidate_median_ms": candidate.get("median_ms"),
        "paired_speedup_baseline_over_candidate": benchmark.get("speedup_baseline_over_candidate"),
        "failure_stage": failure.get("stage"),
        "failure": failure or None,
        "experiment_run_id": result.get("run_id"),
        "result_path": str(Path(run_directory) / "result.json") if run_directory else None,
    }


def _failure_entry(candidate_id: str, repetition: int | None, stage: str,
                   error: BaseException) -> dict[str, Any]:
    return {
        "repetition": repetition, "candidate_id": candidate_id, "status": "failed",
        "correctness_passed": None, "baseline_median_ms": None,
        "candidate_median_ms": None, "paired_speedup_baseline_over_candidate": None,
        "failure_stage": stage,
        "failure": {"stage": stage, "exception_type": type(error).__name__,
                    "message": str(error), "traceback": traceback.format_exc()},
        "experiment_run_id": None, "result_path": None,
    }


def _aggregate(config: Mapping[str, Any], validations: Mapping[str, Mapping[str, Any]],
               entries: Sequence[Mapping[str, Any]], planned_count: int,
               interrupted: bool) -> dict[str, Any]:
    repetitions = config["repetitions"]
    aggregates = []
    for candidate in config["candidates"]:
        identifier = candidate["id"]
        validation = validations[identifier]
        runs = [entry for entry in entries
                if entry["candidate_id"] == identifier and entry["repetition"] is not None]
        successful = [entry for entry in runs if entry["status"] == "success"]
        speeds = [entry["paired_speedup_baseline_over_candidate"] for entry in successful]
        median_speedup = statistics.median(speeds) if speeds else None
        mad = (statistics.median(abs(value - median_speedup) for value in speeds)
               if speeds else None)
        complete = validation["valid"] and len(runs) == repetitions and len(successful) == repetitions
        aggregates.append({
            "candidate_id": identifier, "implementation": candidate.get("implementation"),
            "parameters": candidate.get("parameters"),
            "configuration_valid": validation["valid"],
            "configuration_error": validation.get("error"),
            "scheduled_runs_completed": len(runs), "successful_runs": len(successful),
            "failed_runs": len(runs) - len(successful), "expected_runs": repetitions,
            "complete": complete, "median_run_speedup": median_speedup,
            "speedup_median_absolute_deviation": mad,
            "speedup_spread_definition": (
                "median absolute deviation of successful run-level baseline/candidate speedups"),
        })
    ranked = sorted(
        (item for item in aggregates if item["complete"]),
        key=lambda item: (-item["median_run_speedup"], item["candidate_id"]),
    )
    rankings = [{"rank": rank, "candidate_id": item["candidate_id"],
                 "median_run_speedup": item["median_run_speedup"]}
                for rank, item in enumerate(ranked, 1)]
    scheduled_entries = [entry for entry in entries if entry["repetition"] is not None]
    invalid_count = sum(not validation["valid"] for validation in validations.values())
    successful_count = sum(entry["status"] == "success" for entry in scheduled_entries)
    failed_count = len(scheduled_entries) - successful_count
    if interrupted:
        status = "interrupted"
    elif invalid_count or failed_count or len(scheduled_entries) != planned_count:
        status = "completed_with_failures"
    else:
        status = "success"
    return {
        "status": status,
        "counts": {
            "valid_candidates": len(validations) - invalid_count,
            "invalid_candidates": invalid_count, "planned_experiments": planned_count,
            "completed_experiments": len(scheduled_entries),
            "successful_experiments": successful_count, "failed_experiments": failed_count,
            "remaining_experiments": planned_count - len(scheduled_entries),
        },
        "candidate_aggregates": aggregates,
        "experiments": list(entries),
        "complete_candidate_ranking": rankings,
        "ranking_note": (
            "Descriptive ranking by median successful run-level speedup; only candidates "
            "with all repetitions successful are included. No statistical significance is claimed."
        ),
    }


_CSV_FIELDS = (
    "repetition", "candidate_id", "status", "correctness_passed",
    "baseline_median_ms", "candidate_median_ms",
    "paired_speedup_baseline_over_candidate", "failure_stage",
    "experiment_run_id", "result_path", "parameters_json",
)


def _write_csv(path: Path, entries: Sequence[Mapping[str, Any]],
               candidates: Mapping[str, Mapping[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for entry in entries:
            writer.writerow({
                **{field: entry.get(field) for field in _CSV_FIELDS if field != "parameters_json"},
                "parameters_json": json.dumps(candidates[entry["candidate_id"]].get("parameters"),
                                               sort_keys=True),
            })
    temporary.replace(path)


def _persist(directory: Path, batch: dict[str, Any], summary: dict[str, Any],
             candidates: Mapping[str, Mapping[str, Any]]) -> None:
    summary = {"schema_version": BATCH_SCHEMA_VERSION, "batch_id": batch["batch_id"], **summary}
    batch["summary"] = summary
    _atomic_write_json(directory / "batch_results.json", batch)
    _atomic_write_json(directory / "summary.json", summary)
    _write_csv(directory / "summary.csv", batch["entries"], candidates)


ExperimentExecutor = Callable[..., dict[str, Any]]


def run_batch(raw_config: Mapping[str, Any], *,
              output_root: Path | str = "artifacts/experiment-batches",
              experiment_executor: ExperimentExecutor = run_experiment) -> dict[str, Any]:
    """Validate, schedule, run, and incrementally persist a candidate batch."""
    created = datetime.now(timezone.utc)
    batch_id = f"{created.strftime('%Y%m%dT%H%M%S.%fZ')}-{uuid.uuid4().hex[:8]}"
    directory = Path(output_root).resolve() / batch_id
    directory.mkdir(parents=True, exist_ok=False)
    _atomic_write_json(directory / "submitted_batch_config.json", raw_config)
    batch: dict[str, Any] = {
        "schema_version": BATCH_SCHEMA_VERSION, "batch_id": batch_id,
        "created_at_utc": created.isoformat(), "status": "configuration_failed",
        "batch_directory": str(directory), "entries": [],
    }
    try:
        config = validate_batch_config(raw_config)
    except BaseException as error:
        batch["failure"] = {
            "stage": "configuration", "exception_type": type(error).__name__,
            "message": str(error), "traceback": traceback.format_exc(),
        }
        batch["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        _atomic_write_json(directory / "batch_results.json", batch)
        _atomic_write_json(directory / "summary.json", {
            "schema_version": BATCH_SCHEMA_VERSION, "batch_id": batch_id,
            "status": "configuration_failed", "counts": None,
            "candidate_aggregates": [], "complete_candidate_ranking": [],
        })
        _write_csv(directory / "summary.csv", [], {})
        return batch

    _atomic_write_json(directory / "resolved_batch_config.json", {
        "schema_version": BATCH_SCHEMA_VERSION, "config": config,
    })
    candidates = {candidate["id"]: candidate for candidate in config["candidates"]}
    validations: dict[str, dict[str, Any]] = {}
    valid_configs: dict[str, dict[str, Any]] = {}
    for candidate in config["candidates"]:
        identifier = candidate["id"]
        try:
            valid_configs[identifier] = _candidate_config(config["shared"], candidate)
            validations[identifier] = {"valid": True, "error": None}
        except BaseException as error:
            validations[identifier] = {"valid": False, "error": str(error)}
            invalid_entry = _failure_entry(identifier, None, "configuration", error)
            invalid_entry["parameters"] = candidate.get("parameters")
            batch["entries"].append(invalid_entry)

    rng = random.Random(config["scheduling_seed"])
    schedule = []
    sequence = 1
    for repetition in range(1, config["repetitions"] + 1):
        order = list(valid_configs)
        rng.shuffle(order)
        for identifier in order:
            schedule.append({"sequence": sequence, "repetition": repetition,
                             "candidate_id": identifier})
            sequence += 1
    _atomic_write_json(directory / "planned_schedule.json", {
        "scheduling_seed": config["scheduling_seed"], "entries": schedule,
    })
    batch.update({"status": "running", "config": config, "candidate_validations": validations,
                  "planned_schedule": schedule})
    summary = _aggregate(config, validations, batch["entries"], len(schedule), False)
    _persist(directory, batch, summary, candidates)

    interrupted = False
    child_root = directory / "experiments"
    for scheduled in schedule:
        identifier = scheduled["candidate_id"]
        try:
            result = experiment_executor(valid_configs[identifier], output_root=child_root)
            entry = _entry(identifier, scheduled["repetition"], result)
        except KeyboardInterrupt as error:
            entry = _failure_entry(identifier, scheduled["repetition"], "interruption", error)
            interrupted = True
        except BaseException as error:
            entry = _failure_entry(identifier, scheduled["repetition"], "orchestration", error)
        entry["parameters"] = candidates[identifier].get("parameters")
        batch["entries"].append({"sequence": scheduled["sequence"], **entry})
        if entry["failure_stage"] == "interruption":
            interrupted = True
        summary = _aggregate(config, validations, batch["entries"], len(schedule), interrupted)
        batch["status"] = summary["status"] if interrupted else "running"
        _persist(directory, batch, summary, candidates)
        if interrupted:
            break

    summary = _aggregate(config, validations, batch["entries"], len(schedule), interrupted)
    batch["status"] = summary["status"]
    batch["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    _persist(directory, batch, summary, candidates)
    return batch


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Run a sequential batch of predefined kernel candidates")
    result.add_argument("--config", type=Path, required=True, help="batch JSON configuration")
    result.add_argument("--output-root", type=Path, default=Path("artifacts/experiment-batches"))
    return result


def main(arguments: Sequence[str] | None = None) -> int:
    args = parser().parse_args(arguments)
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(f"batch configuration error: {error}", file=sys.stderr)
        return BATCH_EXIT_CONFIGURATION
    batch = run_batch(config, output_root=args.output_root)
    print(f"batch directory: {batch['batch_directory']}")
    summary = batch.get("summary")
    if summary:
        counts = summary["counts"]
        print(f"status: {summary['status']}")
        print(f"experiments: {counts['successful_experiments']} succeeded, "
              f"{counts['failed_experiments']} failed, {counts['remaining_experiments']} remaining")
        for ranked in summary["complete_candidate_ranking"]:
            print(f"rank {ranked['rank']}: {ranked['candidate_id']} "
                  f"median speedup={ranked['median_run_speedup']:.6f}x")
    else:
        print(f"configuration failure: {batch['failure']['message']}", file=sys.stderr)
    if batch["status"] == "success":
        return BATCH_EXIT_SUCCESS
    if batch["status"] == "interrupted":
        return BATCH_EXIT_INTERRUPTED
    if batch["status"] == "configuration_failed":
        return BATCH_EXIT_CONFIGURATION
    return BATCH_EXIT_FAILURE


if __name__ == "__main__":
    raise SystemExit(main())
