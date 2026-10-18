"""CPU-only tests for evidence-scoped schedule calibration."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from matmul_inspector.tensor_game_schedule_calibration import (
    calibrated_rank, eager_confidence, leave_one_session_out, pair_direction,
    stable_hash, static_rank, validate_profile, validated_profile, validation_assessment,
    verify_profile_sources,
)
from scripts.run_hardware_aware_schedule_calibration import (
    REPO, calibration_profiles, collect_candidates, rank_study, source_data,
)


@pytest.fixture(scope="module")
def retained():
    source = source_data()
    return source, calibration_profiles(source), collect_candidates(source)


def test_profiles_parse_and_verify_real_provenance(retained):
    _, profiles, _ = retained
    assert len(profiles) == 8
    for profile in profiles:
        validate_profile(profile, hardware=profile["hardware"], software=profile["software"],
                         matrix_size=profile["matrix_size"], execution_mode=profile["execution_mode"])
        verify_profile_sources(profile, REPO)
        assert profile["profile_id"] == stable_hash({k: v for k, v in profile.items() if k != "profile_id"})


def test_missing_and_wrong_provenance_rejected(retained, tmp_path: Path):
    profile = deepcopy(retained[1][0])
    profile["source_hashes"] = {}
    with pytest.raises(ValueError, match="provenance"):
        validate_profile(profile, hardware=profile["hardware"], software=profile["software"],
                         matrix_size=128, execution_mode="EAGER")
    profile = deepcopy(retained[1][0])
    profile["source_artifact"] = "../../outside"
    with pytest.raises(ValueError, match="escapes repository"):
        verify_profile_sources(profile, tmp_path)


def test_execution_modes_sizes_and_hardware_are_explicit(retained):
    profiles = retained[1]
    eager = next(p for p in profiles if p["matrix_size"] == 256 and p["execution_mode"] == "EAGER")
    graph = next(p for p in profiles if p["matrix_size"] == 256 and p["execution_mode"] == "CUDA_GRAPH")
    assert eager["profile_id"] != graph["profile_id"]
    assert eager["session_count"] == 2 and graph["session_count"] == 1
    for size, mode, hardware in ((512, "EAGER", eager["hardware"]),
                                 (256, "CUDA_GRAPH", eager["hardware"]),
                                 (256, "EAGER", {**eager["hardware"], "gpu_name": "wrong"})):
        with pytest.raises(ValueError):
            validate_profile(eager, hardware=hardware, software=eager["software"],
                             matrix_size=size, execution_mode=mode)


def test_static_rank_is_existing_lexicographic_objective(retained):
    candidates = [row for row in retained[2] if row["algorithm"] == "project-rank8" and row["matrix_size"] == 256]
    ranked = static_rank(candidates)
    keys = {row["schedule_hash"]: (row["metrics"]["temporary_hbm_traffic_bytes"],
                                   row["metrics"]["recomputation_arithmetic_operations"],
                                   row["metrics"]["logical_launch_groups"], row["schedule_hash"])
            for row in candidates}
    assert [row["schedule_hash"] for row in ranked] == sorted(keys, key=keys.get)


def test_known_pair_eager_graph_and_parity_controls(retained):
    _, profiles, candidates = retained
    by_profile = {(p["matrix_size"], p["execution_mode"]): p for p in profiles}
    for size in (128, 256, 512, 1024):
        pair = [c for c in candidates if c["algorithm"] == "project-rank8"
                and c["matrix_size"] == size and c["lowering_name"] in ("rank8_stored", "rank8_recomputed")]
        eager = by_profile[(size, "EAGER")]
        graph = by_profile[(size, "CUDA_GRAPH")]
        e = calibrated_rank(pair, eager, hardware=eager["hardware"], software=eager["software"])
        g = calibrated_rank(pair, graph, hardware=graph["hardware"], software=graph["software"])
        e_by = {next(c["lowering_name"] for c in pair if c["schedule_hash"] == r["schedule_hash"]): r for r in e}
        g_by = {next(c["lowering_name"] for c in pair if c["schedule_hash"] == r["schedule_hash"]): r for r in g}
        if size <= 512:
            assert e_by["rank8_recomputed"]["calibrated_rank"] < e_by["rank8_stored"]["calibrated_rank"]
        else:
            assert e_by["rank8_recomputed"]["calibrated_rank"] == e_by["rank8_stored"]["calibrated_rank"]
        if size in (128, 256):
            assert g_by["rank8_recomputed"]["calibrated_rank"] == g_by["rank8_stored"]["calibrated_rank"]
        else:
            assert g_by["rank8_stored"]["calibrated_rank"] < g_by["rank8_recomputed"]["calibrated_rank"]


def test_uncertainty_prevents_hard_preference(retained):
    profile = deepcopy(next(p for p in retained[1] if p["matrix_size"] == 256 and p["execution_mode"] == "EAGER"))
    profile["confidence"] = "UNSTABLE"
    profile["profile_id"] = stable_hash({k: v for k, v in profile.items() if k != "profile_id"})
    pair = [c for c in retained[2] if c["algorithm"] == "project-rank8" and c["matrix_size"] == 256
            and c["lowering_name"] in ("rank8_stored", "rank8_recomputed")]
    rows = calibrated_rank(pair, profile, hardware=profile["hardware"], software=profile["software"])
    assert {row["calibrated_rank"] for row in rows} == {1}


def test_candidate_hashes_legality_and_unmapped_status(retained):
    _, profiles, candidates = retained
    assert len(candidates) == 156
    assert len({(c["algorithm"], c["matrix_size"], c["schedule_hash"]) for c in candidates}) == 156
    assert all(c["legality"] == "PASS" and c["schedule_hash"] == c["metrics"]["schedule_hash"] for c in candidates)
    assert all(c["features"]["expected_actual_launch_count"] is None for c in candidates
               if c["lowering_name"] is None)
    p = profiles[0]
    group = [c for c in candidates if c["algorithm"] == "project-rank8" and c["matrix_size"] == 128]
    ranked = calibrated_rank(group, p, hardware=p["hardware"], software=p["software"])
    assert sum(row["status"] == "UNCALIBRATED_LOWERING" for row in ranked) == 15


def test_static_and_calibrated_can_disagree(retained):
    ranking = rank_study(retained[2], retained[1])
    row = next(x for x in ranking["comparisons"] if x["algorithm"] == "project-rank8" and x["matrix_size"] == 512)
    assert row["static_best_mapped"] == row["eager_best_mapped"]
    assert row["graph_best_mapped"] != row["static_best_mapped"]
    assert row["candidate_count"] == 17 and row["calibrated_lowering_count"] == 2


def test_leave_one_session_out_direction_and_failed_validation(retained):
    from scripts.run_hardware_aware_schedule_calibration import HA4, read_json
    pairs = read_json(HA4 / "timing_summary.json")["complete_pairs"]
    checks = leave_one_session_out(pairs)
    assert len(checks) == 8 and all(row["direction_agrees"] for row in checks)
    profile = next(p for p in retained[1] if p["matrix_size"] == 256 and p["execution_mode"] == "EAGER")
    contradictory = {"ratio_bootstrap_95pct": [0.8, 0.9]}
    assert pair_direction(contradictory) == "STORED"
    assert validation_assessment(profile, contradictory)["confidence_after_validation"] == "UNSTABLE"


def test_one_session_confidence_is_not_two_session_confidence(retained):
    profile = next(p for p in retained[1] if p["matrix_size"] == 512 and p["execution_mode"] == "EAGER")
    a, b = profile["paired_evidence"]
    rows = [{"ratio_bootstrap_95pct": row["ratio_bootstrap_95pct"]} for row in (a, b)]
    confidence, direction = eager_confidence(*rows)
    assert (confidence, direction) == ("MODERATE", "RECOMPUTED")


def test_fresh_parity_downgrades_rank_without_changing_retained_profile(retained):
    profile = next(p for p in retained[1] if p["matrix_size"] == 512 and p["execution_mode"] == "EAGER")
    assessed = validation_assessment(profile, {"ratio_bootstrap_95pct": [0.99, 1.11]})
    updated = validated_profile(profile, assessed, "a" * 64)
    assert profile["confidence"] == "MODERATE" and profile["preference"] == "RECOMPUTED"
    assert updated["confidence"] == "WEAK" and updated["preference"] == "PARITY"
    assert updated["retained_profile_id"] == profile["profile_id"]
    pair = [c for c in retained[2] if c["algorithm"] == "project-rank8" and c["matrix_size"] == 512
            and c["lowering_name"] in ("rank8_stored", "rank8_recomputed")]
    ranked = calibrated_rank(pair, updated, hardware=updated["hardware"], software=updated["software"])
    assert {row["calibrated_rank"] for row in ranked} == {1}
    assert all(row["fresh_validation_sha256"] == "a" * 64 for row in ranked)
