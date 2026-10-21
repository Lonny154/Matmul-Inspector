"""CPU-only, evidence-scoped schedule ranking for HA-Schedule-5.

An ordinal calibrated cost is a pair preference, never predicted GPU latency.
Schedules without a validated lowering map remain uncalibrated.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Iterable

from .tensor_game_schedule_model import ExecutionDag, Schedule, validate_schedule


MODES = ("STATIC", "CALIBRATED_EAGER", "CALIBRATED_GRAPH")
EXECUTION_MODES = ("EAGER", "CUDA_GRAPH")
CONFIDENCE = ("STRONG", "MODERATE", "WEAK", "PARITY", "UNSTABLE")
KNOWN_LOWERINGS = {
    "conventional_reference": ("conventional", 9, "REFERENCE"),
    "strassen_reference": ("strassen", 8, "REFERENCE"),
    "rank8_stored": ("project-rank8", 10, "STORED"),
    "rank8_recomputed": ("project-rank8", 9, "RECOMPUTED"),
}


def stable_hash(value: object) -> str:
    return hashlib.sha256((json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()).hexdigest()


def pair_direction(pair: dict) -> str:
    low, high = pair["ratio_bootstrap_95pct"]
    if low > 1:
        return "RECOMPUTED"
    if high < 1:
        return "STORED"
    return "PARITY"


def eager_confidence(a: dict, b: dict) -> tuple[str, str]:
    first, second = pair_direction(a), pair_direction(b)
    if first == second == "PARITY":
        return "PARITY", "PARITY"
    if first != second:
        if "PARITY" in (first, second):
            return "WEAK", "PARITY"
        return "UNSTABLE", "PARITY"
    overlap = max(a["ratio_bootstrap_95pct"][0], b["ratio_bootstrap_95pct"][0]) <= min(
        a["ratio_bootstrap_95pct"][1], b["ratio_bootstrap_95pct"][1])
    return ("STRONG" if overlap else "MODERATE"), first


def graph_confidence(pair: dict) -> tuple[str, str]:
    direction = pair_direction(pair)
    return ("PARITY" if direction == "PARITY" else "MODERATE"), direction


def validate_profile(profile: dict, *, hardware: dict, software: dict,
                     matrix_size: int, execution_mode: str) -> None:
    required = ("profile_id", "execution_mode", "hardware", "software", "matrix_size", "dtype",
                "source_artifact", "source_hashes", "session_ids", "session_count",
                "materialization_stage_ms", "stored_consumer_stage_ms",
                "recomputed_consumer_stage_ms", "recompute_consumer_delta_ms",
                "launch_diagnostic_range_ms", "actual_launch_count", "compiler_resources",
                "paired_evidence", "confidence", "preference", "notes")
    missing = [name for name in required if name not in profile]
    if missing:
        raise ValueError(f"calibration profile missing: {missing}")
    if execution_mode not in EXECUTION_MODES or profile["execution_mode"] != execution_mode:
        raise ValueError("execution mode mismatch")
    if profile["matrix_size"] != matrix_size:
        raise ValueError("calibration size mismatch; interpolation is not supported")
    if profile["hardware"] != hardware or profile["software"] != software:
        raise ValueError("calibration hardware/software provenance mismatch")
    if not profile["source_artifact"] or not profile["source_hashes"]:
        raise ValueError("calibration source provenance is missing")
    if any(len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
           for value in profile["source_hashes"].values()):
        raise ValueError("invalid calibration source hash")
    if profile["session_count"] < 1 or len(profile["session_ids"]) != profile["session_count"]:
        raise ValueError("invalid calibration session provenance")
    if profile["confidence"] not in CONFIDENCE or profile["preference"] not in ("STORED", "RECOMPUTED", "PARITY"):
        raise ValueError("invalid calibration uncertainty category")
    if profile["actual_launch_count"] != {"STORED": 10, "RECOMPUTED": 9}:
        raise ValueError("rank-8 physical topology is not the retained topology")
    if profile["dtype"] != "fp16-input-output/fp32-products":
        raise ValueError("calibration dtype mismatch")
    if profile["profile_id"] != stable_hash({key: value for key, value in profile.items() if key != "profile_id"}):
        raise ValueError("calibration profile content hash mismatch")


def verify_profile_sources(profile: dict, repository_root: Path) -> None:
    """Resolve retained file hashes without allowing a profile to escape the repo."""
    source = (repository_root / profile["source_artifact"]).resolve()
    if not source.is_relative_to(repository_root.resolve()):
        raise ValueError("calibration source path escapes repository")
    for name, expected in profile["source_hashes"].items():
        path = (source / name).resolve()
        if not path.is_relative_to(source) or not path.is_file():
            raise ValueError(f"calibration source is missing or unsafe: {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"calibration source hash mismatch: {name}")


def build_profiles(timing: dict, stage_rows: list[dict], launch_rows: list[dict],
                   hardware: dict, software: dict, source_artifact: str,
                   source_hashes: dict[str, str], topology: dict,
                   compiler_resources: dict) -> list[dict]:
    complete = {(row["matrix_size"], row["session"]): row for row in timing["complete_pairs"]}
    graph = {row["matrix_size"]: row for row in timing["graph_pairs"]}
    consumer = {row["matrix_size"]: row for row in timing["consumer_pairs"]}
    stage = {(row["matrix_size"], row["variant"]): row for row in stage_rows
             if row["phase"] == "stage_component" or row["phase"] == "stage_consumer"}
    tiny = {row["matrix_size"]: row for row in launch_rows if row["variant"] == "tiny_dummy_DIAGNOSTIC_ONLY"}
    launches = {"STORED": len(topology["rank8_stored"]),
                "RECOMPUTED": len(topology["rank8_recomputed"])}
    profiles = []
    for size in (128, 256, 512, 1024):
        a, b = complete[(size, "A")], complete[(size, "B")]
        for mode in EXECUTION_MODES:
            confidence, preference = (eager_confidence(a, b) if mode == "EAGER"
                                      else graph_confidence(graph[size]))
            evidence = [a, b] if mode == "EAGER" else [graph[size]]
            tiny_row = tiny.get(size)
            launch_range = ([tiny_row["min_ms"], tiny_row["max_ms"]] if tiny_row else None)
            payload = {
                "execution_mode": mode, "hardware": hardware, "software": software,
                "matrix_size": size, "dtype": "fp16-input-output/fp32-products",
                "source_artifact": source_artifact, "source_hashes": source_hashes,
                "session_ids": ["A", "B"] if mode == "EAGER" else ["A"],
                "session_count": 2 if mode == "EAGER" else 1,
                "materialization_stage_ms": stage[(size, "materialization")]["median_ms"],
                "stored_consumer_stage_ms": stage[(size, "rank8_stored")]["median_ms"],
                "recomputed_consumer_stage_ms": stage[(size, "rank8_recomputed")]["median_ms"],
                "recompute_consumer_delta_ms": consumer[size]["median_recomputed_minus_stored_ms"],
                "launch_diagnostic_range_ms": launch_range,
                "actual_launch_count": launches,
                "compiler_resources": compiler_resources,
                "paired_evidence": [{"session": row["session"],
                                     "stored_over_recomputed_ratio": row["median_stored_over_recomputed_ratio"],
                                     "ratio_bootstrap_95pct": row["ratio_bootstrap_95pct"]}
                                    for row in evidence],
                "confidence": confidence, "preference": preference,
                "notes": ["Ordinal pair evidence; stage events are not additive device costs.",
                          "Graph evidence has one session; eager evidence has two.",
                          "Dummy-launch event timing is context-dependent and not a coefficient."],
            }
            profile = {"profile_id": stable_hash(payload), **payload}
            validate_profile(profile, hardware=hardware, software=software,
                             matrix_size=size, execution_mode=mode)
            profiles.append(profile)
    return profiles


def schedule_features(dag: ExecutionDag, schedule: Schedule, metrics: dict,
                      lowering_name: str | None) -> dict:
    validation = validate_schedule(dag, schedule)
    if not validation.valid or metrics.get("validation_status") != "PASS":
        raise ValueError("calibrated ranking cannot change or bypass schedule legality")
    nodes = {node.node_id: node for node in dag.nodes}
    form_uses = Counter(source for term in dag.terms for source in (term.u_source, term.v_source)
                        if nodes[source].operation == "LINEAR_FORM")
    materialized_forms = {op.node_id for group in schedule.groups for op in group.operations
                          if op.location == "HBM" and nodes[op.node_id].operation == "LINEAR_FORM"}
    expected = KNOWN_LOWERINGS[lowering_name][1] if lowering_name in KNOWN_LOWERINGS else None
    consumer = KNOWN_LOWERINGS[lowering_name][2] if lowering_name in KNOWN_LOWERINGS else "UNKNOWN"
    actual_materializations = (1 if lowering_name == "rank8_stored" else
                               0 if lowering_name in KNOWN_LOWERINGS else None)
    return {
        "logical_launch_groups": metrics["logical_launch_groups"],
        "expected_actual_launch_count": expected,
        "abstract_hbm_input_form_count": len(materialized_forms),
        "expected_actual_materialization_stage_count": actual_materializations,
        "recomputed_form_count": metrics["number_of_recomputed_intermediates"],
        "recomputation_operation_count": metrics["recomputation_arithmetic_operations"],
        "shared_input_form_count": sum(uses > 1 for uses in form_uses.values()),
        "consumer_variant_class": consumer,
        "abstract_temporary_hbm_bytes": metrics["temporary_hbm_traffic_bytes"],
        "abstract_on_chip_peak_bytes": metrics["peak_on_chip_live_bytes"],
        "peak_abstract_live_bytes": metrics["peak_abstract_local_bytes"],
    }


def static_sort_key(candidate: dict) -> tuple:
    metrics = candidate["metrics"]
    return (metrics["temporary_hbm_traffic_bytes"],
            metrics["recomputation_arithmetic_operations"],
            metrics["logical_launch_groups"], candidate["schedule_hash"])


def static_rank(candidates: Iterable[dict]) -> list[dict]:
    ordered = sorted(candidates, key=static_sort_key)
    return [{"schedule_hash": row["schedule_hash"], "algorithm": row["algorithm"],
             "matrix_size": row["matrix_size"], "static_rank": index,
             "lowering_name": row["lowering_name"], "static_vector": static_sort_key(row)[:3]}
            for index, row in enumerate(ordered, 1)]


def calibrated_rank(candidates: Iterable[dict], profile: dict, *, hardware: dict,
                    software: dict) -> list[dict]:
    mode = profile["execution_mode"]
    validate_profile(profile, hardware=hardware, software=software,
                     matrix_size=profile["matrix_size"], execution_mode=mode)
    rows = []
    for candidate in candidates:
        if candidate["matrix_size"] != profile["matrix_size"]:
            raise ValueError("candidate and calibration size differ")
        name = candidate["lowering_name"]
        if name not in KNOWN_LOWERINGS:
            rows.append({"schedule_hash": candidate["schedule_hash"], "calibrated_rank": None,
                         "calibrated_schedule_cost": None, "status": "UNCALIBRATED_LOWERING",
                         "profile_id": None, "source_hashes": None})
            continue
        cls = KNOWN_LOWERINGS[name][2]
        if KNOWN_LOWERINGS[name][0] != candidate["algorithm"]:
            raise ValueError("validated lowering belongs to a different algorithm")
        if cls == "REFERENCE":
            rows.append({"schedule_hash": candidate["schedule_hash"], "calibrated_rank": 1,
                         "calibrated_schedule_cost": None,
                         "status": "SUPPORTED_ANCHOR_NO_COMPARATIVE_CALIBRATION",
                         "profile_id": None, "execution_mode": mode,
                         "source_hashes": None, "confidence": "NOT_APPLICABLE"})
            continue
        preferred = profile["preference"] if profile["confidence"] in ("STRONG", "MODERATE") else "PARITY"
        cost = int(cls in ("STORED", "RECOMPUTED") and preferred != "PARITY" and cls != preferred)
        rows.append({"schedule_hash": candidate["schedule_hash"], "calibrated_rank": 1 + cost,
                     "calibrated_schedule_cost": cost, "status": "CALIBRATED_ORDINAL",
                     "profile_id": profile["profile_id"], "execution_mode": mode,
                     "source_artifact": profile["source_artifact"],
                     "source_hashes": profile["source_hashes"], "session_ids": profile["session_ids"],
                     "fresh_validation_sha256": profile.get("fresh_validation_sha256"),
                     "retained_profile_id": profile.get("retained_profile_id", profile["profile_id"]),
                     "compiler_resources": profile["compiler_resources"],
                     "confidence": profile["confidence"], "preference": preferred})
    return sorted(rows, key=lambda row: (row["calibrated_rank"] is None,
                                        row["calibrated_rank"] if row["calibrated_rank"] is not None else 99,
                                        row["schedule_hash"]))


def leave_one_session_out(complete_pairs: list[dict]) -> list[dict]:
    by_size: dict[int, dict[str, dict]] = {}
    for pair in complete_pairs:
        by_size.setdefault(pair["matrix_size"], {})[pair["session"]] = pair
    rows = []
    for size, sessions in sorted(by_size.items()):
        if set(sessions) != {"A", "B"}:
            raise ValueError("leave-one-session-out requires both retained sessions")
        for source, held_out in (("A", "B"), ("B", "A")):
            predicted = pair_direction(sessions[source])
            observed = pair_direction(sessions[held_out])
            rows.append({"matrix_size": size, "calibration_session": source,
                         "held_out_session": held_out, "predicted_direction": predicted,
                         "observed_direction": observed, "direction_agrees": predicted == observed})
    return rows


def validation_assessment(profile: dict, fresh_pair: dict | None) -> dict:
    if fresh_pair is None:
        return {"status": "NOT_VALIDATED", "confidence_after_validation": "UNSTABLE"}
    observed = pair_direction(fresh_pair)
    expected = profile["preference"]
    if expected != "PARITY" and observed not in (expected, "PARITY"):
        return {"status": "CONTRADICTED", "observed": observed,
                "confidence_after_validation": "UNSTABLE"}
    if expected != "PARITY" and observed == "PARITY":
        return {"status": "NOT_REPRODUCED", "observed": observed,
                "confidence_after_validation": "WEAK"}
    if expected == "PARITY" and observed != "PARITY":
        return {"status": "NEW_DIRECTION_FROM_PARITY", "observed": observed,
                "confidence_after_validation": "WEAK"}
    return {"status": "DIRECTION_AGREES", "observed": observed,
            "confidence_after_validation": profile["confidence"]}


def validated_profile(profile: dict, assessment: dict, fresh_validation_sha256: str) -> dict:
    """Derive, rather than rewrite, a retained profile after fresh validation."""
    updated = deepcopy(profile)
    updated["retained_profile_id"] = profile["profile_id"]
    updated["fresh_validation_sha256"] = fresh_validation_sha256
    updated["confidence"] = assessment["confidence_after_validation"]
    if updated["confidence"] in ("WEAK", "PARITY", "UNSTABLE"):
        updated["preference"] = "PARITY"
    updated["notes"] = [*updated["notes"], f"Fresh validation: {assessment['status']}."]
    updated["profile_id"] = stable_hash({key: value for key, value in updated.items() if key != "profile_id"})
    return updated


def pareto_hashes(candidates: list[dict]) -> list[str]:
    fields = ("temporary_hbm_traffic_bytes", "recomputation_arithmetic_operations",
              "peak_abstract_local_bytes", "logical_launch_groups")
    def dominates(left: dict, right: dict) -> bool:
        return all(left["metrics"][key] <= right["metrics"][key] for key in fields) and any(
            left["metrics"][key] < right["metrics"][key] for key in fields)
    return sorted(row["schedule_hash"] for row in candidates
                  if not any(dominates(other, row) for other in candidates if other is not row))
