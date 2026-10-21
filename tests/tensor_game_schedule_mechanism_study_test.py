"""CPU-only checks for HA-Schedule-4 evidence accounting."""

from pathlib import Path

import pytest

from matmul_inspector.tensor_game_schedule_gpu_study import load_selected_points
from matmul_inspector.tensor_game_schedule_mechanism_study import (
    COMMON, DUMMY, MECHANISM_LABELS, RECOMPUTED, STORED, classify_kernel_roles,
    classify_mechanism, compare_sessions, compiler_diff_summary,
    expected_topology, join_static_measured, normalize_sass, paired_summary,
    summarize_raw, validate_topology,
)


ROOT = Path(__file__).resolve().parents[1]
HA2 = ROOT / "artifacts/tensor-game/hardware-aware-schedule-search-v1"
FROZEN = ROOT / "artifacts/tensor-game/hardware-aware-rank89-v1/frozen-rank8/solutions.jsonl"


def test_topology_and_common_kernel_classification():
    stored = validate_topology(STORED, [row["kernel_name"] for row in expected_topology(STORED)])
    recomputed = validate_topology(RECOMPUTED, [row["kernel_name"] for row in expected_topology(RECOMPUTED)])
    roles = classify_kernel_roles(stored, recomputed)
    assert roles == {"stored_only": ["materialization"],
                     "differing": ["consumer_0", "consumer_1"],
                     "common": list(COMMON), "stored_launches": 10,
                     "recomputed_launches": 9}
    assert len(expected_topology(DUMMY)) == 10
    assert expected_topology(DUMMY)[0]["role"] == "diagnostic_dummy"
    with pytest.raises(ValueError, match="topology mismatch"):
        validate_topology(STORED, ["wrong"])
    with pytest.raises(ValueError, match="unknown mechanism mode"):
        expected_topology("other")


def test_round_medians_and_session_comparison_do_not_pool_processes():
    rows = []
    for session, stored_values, recomputed_values in (
        ("A", [[3.0, 5.0], [4.0, 6.0]], [[2.0, 4.0], [3.0, 5.0]]),
        ("B", [[6.0, 8.0], [7.0, 9.0]], [[4.0, 6.0], [5.0, 7.0]]),
    ):
        for variant, groups in ((STORED, stored_values), (RECOMPUTED, recomputed_values)):
            for round_index, values in enumerate(groups):
                rows.extend({"session": session, "matrix_size": 128, "phase": "complete",
                             "variant": variant, "round": round_index, "elapsed_ms": value}
                            for value in values)
    summary = summarize_raw(rows)
    assert len(summary) == 4
    assert next(row for row in summary if row["session"] == "A" and row["variant"] == STORED)["round_medians_ms"] == [4.0, 5.0]
    pairs = paired_summary(summary, "complete")
    assert len(pairs) == 2
    assert pairs[0]["median_recomputed_minus_stored_ms"] == -1.0
    assert compare_sessions(pairs)[0]["direction_agrees"]
    with pytest.raises(ValueError, match="both independent sessions"):
        compare_sessions(pairs[:1])


def test_compiler_diff_is_bounded_and_preserves_meaningful_mnemonics():
    stored = {"cubin_sha256": "a" * 64, "registers_per_thread": 40,
              "static_shared_memory_bytes": 8192}
    recomputed = {"cubin_sha256": "b" * 64, "registers_per_thread": 40,
                  "static_shared_memory_bytes": 10240}
    left = "/*00a0*/ LDGSTS.E [R2], [R4];\n/*00b0*/ FADD R1, R2, R3;"
    right = "/*0100*/ LDGSTS.E [R2], [R4];\n/*0110*/ FADD R1, R2, R3;\n/*0120*/ FADD R5, R6, R7;"
    assert normalize_sass(left)[0] == normalize_sass(right)[0]
    result = compiler_diff_summary(stored, recomputed, left, right, max_diff_lines=4)
    assert not result["same_cubin"]
    assert result["registers_per_thread"] == [40, 40]
    assert result["static_shared_memory_bytes"] == [8192, 10240]
    assert result["static_sass_mnemonic_counts"]["stored"]["FADD"] == 1
    assert result["static_sass_mnemonic_counts"]["recomputed"]["FADD"] == 2
    assert len(result["bounded_normalized_sass_diff"]) <= 4


def test_static_join_preserves_abstract_and_measured_categories():
    selected = load_selected_points(HA2, FROZEN)
    topology = {name: validate_topology(name, [row["kernel_name"] for row in expected_topology(name)])
                for name in (STORED, RECOMPUTED)}
    complete = [{"session": session, "matrix_size": 128, "variant": variant, "median_ms": value}
                for session in ("A", "B") for variant, value in ((STORED, 0.2), (RECOMPUTED, 0.18))]
    stage = [{"session": "A", "matrix_size": 128, "variant": STORED, "median_ms": 0.04}]
    rows = join_static_measured(selected, [128], complete, stage, topology,
                                {(128, STORED): [{"dram__bytes_read.sum": 10.0}]})
    assert len(rows) == 2
    assert rows[0]["modeled_temporary_hbm_bytes"] == 393216
    assert rows[1]["modeled_temporary_hbm_bytes"] == 368640
    assert rows[0]["actual_launches"] == 10
    assert rows[1]["actual_launches"] == 9
    assert rows[0]["complete_median_ms_session_b"] == 0.2
    assert rows[0]["profiler"] == [{"dram__bytes_read.sum": 10.0}]


def test_mechanism_labels_require_matched_support():
    sessions = [{"matrix_size": size, "direction_agrees": True} for size in (128, 256)]
    complete = [{"session": session, "matrix_size": size,
                 "ratio_bootstrap_95pct": [1.03, 1.12]}
                for session in ("A", "B") for size in (128, 256)]
    consumer = [{"matrix_size": size, "delta_bootstrap_95pct_ms": [0.002, 0.005],
                 "median_recomputed_minus_stored_ms": 0.003} for size in (128, 256)]
    prep = [{"matrix_size": size, "median_ms": 0.008} for size in (128, 256)]
    result = classify_mechanism(sessions, complete, consumer, prep, [], [])
    assert result["primary"] == "RECOMPUTE_COSTS_BUT_MATERIALIZATION_COSTS_MORE"
    assert result["decision"] == "MECHANISM_IDENTIFIED"
    assert classify_mechanism([{**sessions[0], "direction_agrees": False}],
                              complete, consumer, prep, [], [])["primary"] == "SESSION_UNSTABLE"
    faster = [{**row, "delta_bootstrap_95pct_ms": [-0.006, -0.001]}
              for row in consumer]
    assert classify_mechanism(sessions, complete, faster, prep, [], [])["primary"] == "RECOMPUTE_CONSUMER_ITSELF_FASTER"
    mixed = classify_mechanism(sessions, complete, [], prep, [{}], [{}])
    assert mixed["primary"] == "MIXED_MECHANISM"
    assert DUMMY.endswith("DIAGNOSTIC_ONLY")
    assert "MEMORY_TRAFFIC_SIGNAL" in MECHANISM_LABELS
    graph_eager = [{"matrix_size": size, "median_recomputed_minus_stored_ms": -0.02}
                   for size in (128, 256)]
    graph = [{"matrix_size": size, "median_recomputed_minus_stored_ms": -0.005}
             for size in (128, 256)]
    dummy = [{"matrix_size": size, "median_stored_over_recomputed_ratio": 1.01}
             for size in (128, 256)]
    launch = classify_mechanism(sessions, complete, [], prep, graph, dummy, graph_eager)
    assert launch["primary"] == "LAUNCH_REDUCTION_DOMINANT_SIGNAL"
    assert classify_mechanism([{**sessions[0], "material_disagreement": True}],
                              complete, [], prep, [], [])["primary"] == "SESSION_UNSTABLE"
