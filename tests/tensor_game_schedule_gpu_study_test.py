"""CPU-only checks for retained HA-Schedule-3 selection and summaries."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from matmul_inspector.tensor_game_schedule_gpu_study import (
    build_lowering_map, join_static_measurements, load_selected_points,
    summarize_timings, timing_distribution,
    verify_artifact_hashes,
)


ROOT = Path(__file__).resolve().parents[1]
HA2 = ROOT / "artifacts/tensor-game/hardware-aware-schedule-search-v1"
FROZEN = ROOT / "artifacts/tensor-game/hardware-aware-rank89-v1/frozen-rank8/solutions.jsonl"


def test_selected_points_replay_retained_static_metrics_and_mapping():
    selected = load_selected_points(HA2, FROZEN)
    assert [(item["name"], item["run_id"]) for item in selected] == [
        ("conventional_reference", "R0013"),
        ("strassen_reference", "R0037"),
        ("rank8_stored", "R0061"),
        ("rank8_recomputed", "R0063"),
        ("rank8_capacity_efficient", "R0053"),
    ]
    assert all(len(item["static_per_size"]) == 4 for item in selected)
    assert all(len(data["schedule_hash"]) == 64 for item in selected
               for data in item["static_per_size"].values())
    assert [item["gpu_mode"] for item in selected] == [
        "reference", "reference", "stored_form", "recomputed_form", None,
    ]
    assert selected[-1]["classification"] == "NOT_IMPLEMENTABLE_IN_CURRENT_LOWERING"
    assert all(item["classification"] == "APPROXIMATELY_IMPLEMENTABLE"
               for item in selected[:-1])
    assert [item["static_per_size"][128]["metrics"]["temporary_hbm_traffic_bytes"]
            for item in selected] == [131072, 262144, 393216, 368640, 524288]
    assert selected[3]["static_per_size"][128]["metrics"]["recomputation_arithmetic_operations"] == 4096
    assert selected[3]["static_per_size"][128]["metrics"]["extra_source_read_bytes"] == 16384
    assert selected[3]["static_per_size"][128]["schedule"]["recompute_allowed"] == ["A_FORM00"]


def test_lowering_map_and_static_measurement_join():
    selected = load_selected_points(HA2, FROZEN)
    mapping = build_lowering_map(selected)
    assert [len(item["source_kernel_sequence"]) for item in mapping] == [9, 8, 10, 9, 0]
    assert mapping[2]["source_kernel_sequence"][0] == "_materialize_u_form"
    assert mapping[3]["source_kernel_sequence"][0] == "_shared_form_gemm"
    assert mapping[3]["schedule_hash_by_size"][128] == selected[3]["static_per_size"][128]["schedule_hash"]
    rows = join_static_measurements(
        selected, [128],
        [{"matrix_size": 128, "candidate": "rank8_recomputed", "median_ms": 0.125}],
        [{"matrix_size": 128, "candidate": "rank8_recomputed", "actual_kernel_launches": 9}],
    )
    assert len(rows) == 5
    assert rows[3]["measurement_status"] == "PASS"
    assert rows[3]["modeled_temporary_hbm_traffic_bytes"] == 368640
    assert rows[3]["actual_kernel_launches"] == 9
    assert rows[3]["measured_complete_operation_median_ms"] == 0.125
    assert rows[2]["measurement_status"] == "EXCLUDED_CORRECTNESS_FAILURE"
    assert rows[4]["measurement_status"] == "EXCLUDED_IMPLEMENTABILITY"


def test_retained_artifact_hashes_are_checked(tmp_path):
    data = b"original\n"
    (tmp_path / "data.json").write_bytes(data)
    (tmp_path / "hash_manifest.json").write_text(json.dumps({"data.json": hashlib.sha256(data).hexdigest()}))
    assert verify_artifact_hashes(tmp_path)["data.json"] == hashlib.sha256(data).hexdigest()
    (tmp_path / "data.json").write_bytes(b"changed\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_artifact_hashes(tmp_path)


def test_timing_summary_uses_round_medians_and_excludes_failed_candidates():
    rows = [
        {"matrix_size": 128, "candidate": name, "round": round_index, "elapsed_ms": duration}
        for name, round_index, duration in [
            ("good", 0, 1.0), ("good", 0, 3.0),
            ("good", 1, 2.0), ("good", 1, 4.0),
            ("failed", 0, 0.1), ("failed", 1, 0.1),
        ]
    ]
    assert summarize_timings(rows, {"good"}) == [
        {"matrix_size": 128, "candidate": "good", "sample_count": 4,
         "median_ms": 2.5, "mean_ms": 2.5, "stdev_ms": pytest.approx(0.7071067811865476),
         "min_ms": 2.0, "max_ms": 3.0,
         "coefficient_of_variation": pytest.approx(0.282842712474619),
         "round_medians_ms": [2.0, 3.0]},
    ]
    with pytest.raises(ValueError, match="at least one round"):
        timing_distribution([])
