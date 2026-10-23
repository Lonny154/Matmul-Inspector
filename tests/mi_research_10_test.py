"""Offline replay of the prospective screening evidence and freeze audit."""

from scripts import analyze_mi_research_10 as study


def test_evidence_and_frozen_prediction_audit() -> None:
    assert study.verify()["status"] == "PASS"
    record = study.analyze()
    assert record["plan_audit"]["status"] == "CONTRADICTS_FROZEN_DESCRIPTOR"
    assert record["plan_audit"]["descriptor_58_minus_55_live_bytes"] == 4096
    assert record["compiler_candidate_minus_control"]["static_counts"]["ttir_dot"] == 0
    assert record["compiler_candidate_minus_control"]["static_counts"]["ptx_mma_sync"] == 0
    assert record["timing"]["decision"] == "INCONCLUSIVE"
    assert record["timing"]["ranking_supported"] is False
