"""CPU-safe selection and evidence helpers for the selected GPU schedule study."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import statistics

from .tensor_game import RankOneAction
from .tensor_game_schedule_model import (
    BlockConfig, LaunchGroup, Schedule, ScheduledOp, build_execution_dag,
    evaluate_schedule, reference_dag, validate_schedule,
)


SELECTION = (
    ("conventional_reference", "conventional", 4, "0", "reference", "APPROXIMATELY_IMPLEMENTABLE"),
    ("strassen_reference", "strassen", 4, "0", "reference", "APPROXIMATELY_IMPLEMENTABLE"),
    ("rank8_stored", "project-rank8", 4, "0", "stored_form", "APPROXIMATELY_IMPLEMENTABLE"),
    ("rank8_recomputed", "project-rank8", 4, "1", "recomputed_form", "APPROXIMATELY_IMPLEMENTABLE"),
    ("rank8_capacity_efficient", "project-rank8", 2, "0", None, "NOT_IMPLEMENTABLE_IN_CURRENT_LOWERING"),
)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def verify_artifact_hashes(directory: Path) -> dict[str, str]:
    manifest = json.loads((directory / "hash_manifest.json").read_text())
    for filename, expected in manifest.items():
        actual = hashlib.sha256((directory / filename).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"artifact hash mismatch: {filename}")
    return manifest


def _schedule(payload: dict) -> Schedule:
    groups = tuple(LaunchGroup(group["group_id"],
                               tuple(ScheduledOp(**op) for op in group["operations"]))
                   for group in payload["groups"])
    return Schedule(payload["schedule_id"], payload["policy"], payload["dag_hash"],
                    groups, tuple(payload["recompute_allowed"]), payload["schedule_hash"])


def _dag(algorithm: str, config: BlockConfig, frozen_actions: tuple[RankOneAction, ...]):
    if algorithm == "project-rank8":
        return build_execution_dag(algorithm, frozen_actions, config)
    return reference_dag(algorithm, config)


def load_selected_points(ha2_directory: Path, frozen_artifact: Path) -> list[dict]:
    """Verify retained static evidence before mapping any schedule to GPU code."""
    verify_artifact_hashes(ha2_directory)
    config = json.loads((ha2_directory / "config.json").read_text())
    frozen = json.loads(frozen_artifact.read_text().splitlines()[0])
    if frozen.get("exact_reconstruction_status") != "PASS" or frozen.get("rank") != 8:
        raise ValueError("frozen project candidate is not an exact rank-8 record")
    actions = tuple(RankOneAction(**action) for action in frozen["actions"])
    runs = _read_jsonl(ha2_directory / "solver_runs.jsonl")
    schedules = {(row["algorithm"], row["matrix_size"], row["schedule"]["schedule_hash"]):
                 row["schedule"] for row in _read_jsonl(ha2_directory / "schedules.jsonl")}
    metrics = {(row["run_id"], row["matrix_size"]): row for row in
               _read_jsonl(ha2_directory / "schedule_metrics.jsonl")}
    selected = []
    for name, algorithm, capacity, budget, mode, classification in SELECTION:
        matches = [row for row in runs if row["algorithm"] == algorithm
                   and row["capacity_block_units"] == capacity
                   and row["recomputation_budget_block_units"] == budget
                   and row["objective"] == "LEX_TEMP_RECOMPUTE_LAUNCH"]
        if len(matches) != 1 or matches[0]["status"] != "OPTIMAL" or matches[0]["agreement"] != "PASS":
            raise ValueError(f"selected HA-Schedule-2 point is missing or invalid: {name}")
        run = matches[0]
        per_size = {}
        for size in config["matrix_sizes"]:
            row = metrics[(run["run_id"], size)]
            key = (algorithm, size, row["schedule_hash"])
            schedule = _schedule(schedules[key])
            dag = _dag(algorithm, BlockConfig(size, config["input_dtype"],
                                              config["accumulation_dtype"], config["output_dtype"]), actions)
            validation = validate_schedule(dag, schedule)
            if not validation.valid:
                raise ValueError(f"retained schedule invalid for {name}, size {size}: {validation.issues}")
            evaluated = evaluate_schedule(dag, schedule)
            if evaluated != row["metrics"]:
                raise ValueError(f"retained metric mismatch for {name}, size {size}")
            per_size[size] = {"schedule_hash": schedule.schedule_hash,
                              "schedule": schedules[key], "metrics": row["metrics"]}
        recompute = tuple(per_size[config["matrix_sizes"][0]]["schedule"]["recompute_allowed"])
        if name == "rank8_recomputed" and recompute != ("A_FORM00",):
            raise ValueError("selected recompute point does not recompute A_FORM00")
        if name == "rank8_stored" and recompute:
            raise ValueError("selected store point unexpectedly recomputes a form")
        deviations = (
            ["Each product is written to FP32 HBM and one reconstruction kernel reads it; "
             "abstract within-group output accumulation is not directly fused.",
             "Other singleton input forms are fused in the GEMM load path even if the static schedule materializes them."]
            if algorithm == "project-rank8" else
            ["Each product is written to FP32 HBM and one reconstruction kernel reads it; "
             "abstract within-group output accumulation is not directly fused."]
        )
        if name == "rank8_capacity_efficient":
            deviations = ["Its lower-capacity intermediate placements require additional selective HBM kernels "
                          "not provided by the existing fused path; timing a substitute would mislabel the point."]
        selected.append({"name": name, "algorithm": algorithm, "run_id": run["run_id"],
                         "capacity_block_units": capacity, "recomputation_budget_blocks": budget,
                         "classification": classification, "gpu_mode": mode,
                         "static_per_size": per_size, "mapping_deviations": deviations,
                         "source_artifact_hash_count": len(json.loads((ha2_directory / "hash_manifest.json").read_text()))})
    return selected


def build_lowering_map(selected: list[dict]) -> list[dict]:
    """Describe source kernel sequences without importing optional GPU packages."""
    mapping = []
    for item in selected:
        mode = item["gpu_mode"]
        rank = {"conventional": 8, "strassen": 7, "project-rank8": 8}[item["algorithm"]]
        if mode == "reference":
            kernels = ["_fused_input_gemm_kernel"] * rank + ["_fused_reconstruction_kernel"]
        elif mode in {"stored_form", "recomputed_form"}:
            kernels = (["_materialize_u_form"] if mode == "stored_form" else []) + [
                "_shared_form_gemm"] * 2 + ["_fused_input_gemm_kernel"] * (rank - 2) + [
                    "_fused_reconstruction_kernel"]
        else:
            kernels = []
        mapping.append({"candidate": item["name"], "algorithm": item["algorithm"],
                        "run_id": item["run_id"], "classification": item["classification"],
                        "gpu_mode": mode, "deviations": item["mapping_deviations"],
                        "source_kernel_sequence": kernels,
                        "schedule_hash_by_size": {size: data["schedule_hash"]
                                                  for size, data in item["static_per_size"].items()},
                        "modeled_launch_groups_by_size": {size: data["metrics"]["logical_launch_groups"]
                                                          for size, data in item["static_per_size"].items()},
                        "modeled_hbm_materializations_by_size": {
                            size: data["metrics"]["number_of_materialized_intermediates"]
                            for size, data in item["static_per_size"].items()},
                        "modeled_recomputation_by_size": {
                            size: data["metrics"]["number_of_recomputed_intermediates"]
                            for size, data in item["static_per_size"].items()},
                        "modeled_peak_on_chip_bytes_by_size": {
                            size: data["metrics"]["peak_on_chip_live_bytes"]
                            for size, data in item["static_per_size"].items()},
                        "abstract_capacity_block_units": item["capacity_block_units"]})
    return mapping


def join_static_measurements(selected: list[dict], sizes: list[int],
                             summary: list[dict], launch_records: list[dict]) -> list[dict]:
    """Join only correctness-gated measurements to independently replayed metrics."""
    by_measurement = {(row["matrix_size"], row["candidate"]): row for row in summary}
    by_launch = {(row["matrix_size"], row["candidate"]): row for row in launch_records}
    rows = []
    for item in selected:
        for size in sizes:
            data = item["static_per_size"][size]
            static = data["metrics"]
            measurement = by_measurement.get((size, item["name"]))
            launches = by_launch.get((size, item["name"]))
            rows.append({"candidate": item["name"], "algorithm": item["algorithm"],
                         "matrix_size": size, "schedule_hash": data["schedule_hash"],
                         "modeled_temporary_hbm_traffic_bytes": static["temporary_hbm_traffic_bytes"],
                         "modeled_recompute_operations": static["recomputation_arithmetic_operations"],
                         "modeled_extra_source_read_bytes": static["extra_source_read_bytes"],
                         "modeled_peak_on_chip_bytes": static["peak_on_chip_live_bytes"],
                         "modeled_launch_groups": static["logical_launch_groups"],
                         "actual_kernel_launches": launches["actual_kernel_launches"] if launches else None,
                         "measured_complete_operation_median_ms": measurement["median_ms"] if measurement else None,
                         "measurement_status": ("PASS" if measurement else
                                                "EXCLUDED_IMPLEMENTABILITY" if item["gpu_mode"] is None else
                                                "EXCLUDED_CORRECTNESS_FAILURE")})
    return rows


def timing_distribution(round_medians_ms: list[float]) -> dict:
    if not round_medians_ms:
        raise ValueError("timing summary needs at least one round")
    mean = statistics.mean(round_medians_ms)
    std = statistics.stdev(round_medians_ms) if len(round_medians_ms) > 1 else 0.0
    return {"median_ms": statistics.median(round_medians_ms), "mean_ms": mean,
            "stdev_ms": std, "min_ms": min(round_medians_ms), "max_ms": max(round_medians_ms),
            "coefficient_of_variation": std / mean if mean else 0.0,
            "round_medians_ms": round_medians_ms}


def summarize_timings(raw_rows: list[dict], passed_names: set[str]) -> list[dict]:
    """Use round medians as replicates; exclude failed correctness candidates."""
    grouped: dict[tuple[int, str, int], list[float]] = {}
    for row in raw_rows:
        if row["candidate"] in passed_names:
            grouped.setdefault((row["matrix_size"], row["candidate"], row["round"]), []).append(row["elapsed_ms"])
    by_candidate: dict[tuple[int, str], list[tuple[int, float]]] = {}
    for (size, name, round_index), values in grouped.items():
        by_candidate.setdefault((size, name), []).append((round_index, statistics.median(values)))
    result = []
    for (size, name), pairs in sorted(by_candidate.items()):
        medians = [value for _, value in sorted(pairs)]
        result.append({"matrix_size": size, "candidate": name,
                       "sample_count": sum(len(grouped[(size, name, round_index)]) for round_index, _ in pairs),
                       **timing_distribution(medians)})
    return result
