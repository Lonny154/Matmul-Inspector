"""CPU-only numerical diagnosis and offline evidence checks."""

import copy
import json

import pytest
import torch

from scripts import run_mi_research_3b_correctness as study


def test_error_counts_coordinates_and_relative_domain():
    reference = torch.tensor([[0.0, 2.0], [4.0, -1.0]], dtype=torch.float16)
    actual = torch.tensor([[0.0, 2.5], [4.0, -1.0]], dtype=torch.float16)
    result = study.summarize_error(torch, actual, reference)
    assert not result["passed"]
    assert result["failing_elements"] == 1
    assert result["failing_fraction"] == 0.25
    assert result["max_absolute_error"] == 0.5
    assert result["max_relative_error_reference_abs_ge_1"] == 0.25
    assert result["representatives"][0]["row"] == 0
    assert result["representatives"][0]["col"] == 1
    assert result["representatives"][0]["failed"]


def test_candidate_failure_overlap_and_numerical_agreement():
    reference = torch.zeros((2, 2), dtype=torch.float16)
    a = torch.tensor([[0.02, 0.0], [0.0, 0.0]], dtype=torch.float16)
    b = torch.tensor([[0.02, 0.0], [0.0, 0.02]], dtype=torch.float16)
    result = study.compare_candidates(torch, a, b, reference)
    assert result["a_failure_count"] == 1
    assert result["b_failure_count"] == 2
    assert result["same_failing_coordinates"] == 1
    assert result["failure_jaccard"] == 0.5


def fixture_raw():
    cases = []
    for m in study.MS:
        for seed in range(3):
            cases.append({"m": m, "seed": seed,
                          "eager": {label: {"passed": m < 256, "failing_elements": int(m >= 256)}
                                    for label in ("A", "B")},
                          **({"graph": {label: {"passed": m < 256} for label in ("A", "B")},
                              "graph_equals_eager": {"A": True, "B": True}} if seed == 0 else {})})
    return {"schema_version": 1, "m_values": list(study.MS), "k": study.K, "n": study.N,
            "tolerance": study.TOLERANCE, "shape_seed_cases": cases,
            "controlled_prefix_cases": [{"m": m, "reference_vs_m512_prefix": {"passed": True},
                                         "A_exact_match_fraction_vs_m512_A_prefix": 1.0,
                                         "B_exact_match_fraction_vs_m512_B_prefix": 1.0}
                                        for m in study.MS]}


def test_summary_preserves_gate_and_rejects_missing_case():
    raw = fixture_raw()
    result = study.summarize(raw)
    assert result["shape_results"]["256"]["eager_gate_by_seed"]["0"] == {"A": False, "B": False}
    bad = copy.deepcopy(raw)
    bad["shape_seed_cases"].pop()
    with pytest.raises(ValueError, match="configuration"):
        study.summarize(bad)
    legacy = copy.deepcopy(raw)
    for case in legacy["controlled_prefix_cases"]:
        case.pop("A_exact_match_fraction_vs_m512_A_prefix")
        case.pop("B_exact_match_fraction_vs_m512_B_prefix")
    assert "controlled_prefix_candidate_agreement" not in study.summarize(legacy)


def test_manifest_and_offline_replay(tmp_path):
    raw = fixture_raw()
    for key, name in (("source_sha256", "source.py"),
                      ("conventional_source_sha256", "conventional_source.py"),
                      ("script_sha256", "diagnostic_script.py")):
        (tmp_path / name).write_bytes(study.SOURCE.read_bytes() if name == "source.py" else name.encode())
        raw[key] = study.digest(tmp_path / name)
    study.write_json(tmp_path / "raw.json", raw)
    study.write_json(tmp_path / "summary.json", study.summarize(raw))
    study.write_json(tmp_path / "hash_manifest.json", {
        str(p.relative_to(tmp_path)): study.digest(p) for p in tmp_path.iterdir()
        if p.is_file() and p.name != "hash_manifest.json"})
    assert study.verify(tmp_path)["status"] == "PASS"
    (tmp_path / "summary.json").write_text("{}")
    with pytest.raises(ValueError, match="manifest"):
        study.verify(tmp_path)


def test_retained_reference_mode_finding_replays_without_gpu():
    output = study.ROOT / "artifacts/mi-research-3b/evidence-v1/diagnostic-v2"
    assert study.verify(output)["status"] == "PASS"
    raw = json.loads((output / "raw.json").read_text())
    for case in raw["shape_seed_cases"]:
        expected = case["m"] < 256
        assert all(case["eager"][label]["passed"] is expected for label in ("A", "B"))
        if case["seed"] == 0:
            assert case["graph_equals_eager"] == {"A": True, "B": True}
            assert case["scratch_after_graph"]["nonfinite_elements"] == 0
            assert case["scratch_after_graph"]["reduction_matches_output_exactly"]
            if not expected:
                controls = case["reference_controls"]
                assert not controls["fp16_cublas_vs_fp32_rounded"]["passed"]
                assert controls["no_reduced_precision_vs_fp32_rounded"]["passed"]
                assert controls["A_vs_fp32_rounded"]["passed"]
                assert controls["B_vs_fp32_rounded"]["passed"]
    for case in raw["controlled_prefix_cases"]:
        assert case["A_exact_match_fraction_vs_m512_A_prefix"] == 1.0
        assert case["B_exact_match_fraction_vs_m512_B_prefix"] == 1.0
