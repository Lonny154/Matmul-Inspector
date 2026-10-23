"""Offline replay of the prospective screening evidence and freeze audit."""

import pytest

from scripts import analyze_mi_research_10 as study
from scripts import run_mi_research_10 as gpu
from scripts import time_mi_research_10 as timing


def _historical_evidence_paths():
    return (study.ANALYSIS, study.MANIFEST, gpu.OUT, timing.OUT)


def _require_historical_evidence() -> None:
    if not any(path.exists() or path.is_symlink() for path in _historical_evidence_paths()):
        pytest.skip("Original MI-Research-10 analysis.json, hash manifest, GPU validation, and paired timing evidence are unavailable; historical replay is unverified")


def test_evidence_and_frozen_prediction_audit() -> None:
    _require_historical_evidence()
    assert study.verify()["status"] == "PASS"
    record = study.analyze()
    assert record["plan_audit"]["status"] == "CONTRADICTS_FROZEN_DESCRIPTOR"
    assert record["plan_audit"]["descriptor_58_minus_55_live_bytes"] == 4096
    assert record["compiler_candidate_minus_control"]["static_counts"]["ttir_dot"] == 0
    assert record["compiler_candidate_minus_control"]["static_counts"]["ptx_mma_sync"] == 0
    assert record["timing"]["decision"] == "INCONCLUSIVE"
    assert record["timing"]["ranking_supported"] is False


@pytest.fixture
def isolated_evidence_paths(tmp_path, monkeypatch):
    analysis = tmp_path / "analysis.json"
    manifest = tmp_path / "hash_manifest.json"
    gpu_path = tmp_path / "gpu-validation-v1"
    timing_path = tmp_path / "graph-pair-s1"
    monkeypatch.setattr(study, "ANALYSIS", analysis)
    monkeypatch.setattr(study, "MANIFEST", manifest)
    monkeypatch.setattr(gpu, "OUT", gpu_path)
    monkeypatch.setattr(timing, "OUT", timing_path)
    return analysis, manifest, gpu_path, timing_path


def test_fully_missing_historical_evidence_skips(isolated_evidence_paths):
    with pytest.raises(pytest.skip.Exception, match="historical replay is unverified"):
        test_evidence_and_frozen_prediction_audit()


def test_complete_evidence_enters_strict_verifier(isolated_evidence_paths, monkeypatch):
    analysis, manifest, gpu, timing = isolated_evidence_paths
    analysis.touch()
    manifest.touch()
    gpu.mkdir()
    timing.mkdir()

    def strict_verify():
        raise ValueError("strict historical verification called")

    monkeypatch.setattr(study, "verify", strict_verify)
    with pytest.raises(ValueError, match="strict historical verification called"):
        test_evidence_and_frozen_prediction_audit()


@pytest.mark.parametrize("present", ["analysis", "manifest", "gpu", "timing"])
def test_partial_historical_evidence_fails(isolated_evidence_paths, present):
    analysis, manifest, gpu, timing = isolated_evidence_paths
    paths = {"analysis": analysis, "manifest": manifest, "gpu": gpu, "timing": timing}
    path = paths[present]
    if present in ("gpu", "timing"):
        path.mkdir()
    else:
        path.touch()
    with pytest.raises((OSError, ValueError)):
        test_evidence_and_frozen_prediction_audit()


def test_corrupt_historical_analysis_fails(isolated_evidence_paths):
    analysis = isolated_evidence_paths[0]
    analysis.write_text("not JSON")
    with pytest.raises(ValueError):
        test_evidence_and_frozen_prediction_audit()


def test_hash_mismatched_historical_manifest_fails(isolated_evidence_paths, monkeypatch):
    analysis, manifest, _, _ = isolated_evidence_paths
    analysis.write_text("{}")
    manifest.write_text("{}")
    monkeypatch.setattr(study, "OUT", analysis.parent)
    monkeypatch.setattr(study, "analyze", lambda: {})
    with pytest.raises(ValueError, match="MI-Research-10 hash manifest mismatch"):
        test_evidence_and_frozen_prediction_audit()
