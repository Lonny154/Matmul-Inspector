"""CPU-only exactness and offline-replay tests for the 3x3 feasibility study."""

import json
import shutil

import pytest

from matmul_inspector.tensor_game import RankOneAction, matrix_multiplication_tensor, sum_actions
from scripts import run_mi_research_6_feasibility as study
from scripts import run_mi_research_6_gpu_probe as gpu


def test_authoritative_import_and_all_tensor_entries():
    source = study.SOURCE.read_text()
    laderman = study.parse_laderman(source)
    ordinary = study.conventional()
    assert (len(laderman), len(ordinary)) == (23, 27)
    target = matrix_multiplication_tensor(3, 3, 3)
    assert sum_actions(laderman, study.SHAPE) == target
    assert sum_actions(ordinary, study.SHAPE) == target
    # The Maple source stores Transpose(W); omitting that convention fails.
    wrong_output = tuple(RankOneAction(a.u, a.v,
                        (a.w[3 * c + r] for r in range(3) for c in range(3)))
                         for a in laderman)
    assert sum_actions(wrong_output, study.SHAPE) != target
    altered = source.replace("[0,1,-l]", "[0,1,l]", 1)
    assert sum_actions(study.parse_laderman(altered), study.SHAPE) != target
    with pytest.raises(ValueError, match="unsupported coefficient"):
        study.parse_laderman(source.replace("[0,1,-l]", "[0,1,secret]", 1))


@pytest.mark.parametrize("candidate", [study.conventional, lambda: study.parse_laderman(study.SOURCE.read_text())])
def test_noncommutative_block_cases_and_both_schedules(candidate):
    actions = candidate()
    natural = tuple(range(len(actions)))
    liveness = study.liveness_order(actions)
    assert sorted(liveness) == list(natural)
    assert study.block_matmul([[0, 1], [0, 0]], [[0, 0], [1, 0]]) != study.block_matmul(
        [[0, 0], [1, 0]], [[0, 1], [0, 0]])
    for _label, left, right in study.block_cases():
        expected = study.direct_blocks(left, right)
        assert study.evaluate_blocks(actions, natural, left, right) == expected
        assert study.evaluate_blocks(actions, liveness, left, right) == expected


def test_dags_and_costs_are_consistent():
    bundle = study.build()
    results = json.loads(bundle["exact_and_block_results.json"])
    dags = json.loads(bundle["execution_dags.json"])
    assert set(results["algorithm_records"]) == set(dags)
    assert results["algorithm_records"]["conventional_27"]["static_cost"]["pre_additions"] == 0
    assert results["algorithm_records"]["laderman_23_l_equals_1"]["static_cost"]["pre_additions"] == 56
    assert results["algorithm_records"]["laderman_23_l_equals_1"]["static_cost"]["post_additions"] == 42
    for name, graph in dags.items():
        seen = set()
        assert graph["product_count"] == results["algorithm_records"][name]["rank"]
        for node in graph["nodes"]:
            assert set(node["deps"]) <= seen
            seen.add(node["id"])
        assert len(seen) == graph["node_count"]
        assert sum(n["kind"] == "OUTPUT_WRITE" for n in graph["nodes"]) == 9
        assert len(results["algorithm_records"][name]["direct_matrix_block_cases"]) == 18


def test_replay_rejects_modified_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "OUT", tmp_path)
    monkeypatch.setattr(study, "SOURCE", tmp_path / study.SOURCE.name)
    study.SOURCE.write_bytes((study.ROOT / "artifacts/mi-research-6/laderman_3x3x3_source.mpl").read_bytes())
    study.run(False)
    study.run(True)
    path = tmp_path / "coefficients.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="offline replay mismatch"):
        study.run(True)


def test_gpu_topology_and_correctness_gate_are_strict():
    raw = {
        "schema_version": 1, "size": 96, "ranks": gpu.RANKS,
        "case_names": list(gpu.CASES), "tolerance": gpu.TOLERANCE,
        "cases": [{"case": name, "checks": {label: {"passed": True, "failing_elements": 0}
                                              for label in gpu.LABELS}} for name in gpu.CASES],
        "topology": {label: {"product_events": gpu.RANKS[label],
                              "reconstruction_events": 1,
                              "kernel_events": gpu.RANKS[label] + 1,
                              "driver_launch_calls": gpu.RANKS[label] + 1,
                              "kernel_names": ["_block_product"] * gpu.RANKS[label]
                                              + ["_reconstruct"]} for label in gpu.LABELS},
        "compiler": {label: {phase: {"ptx_sha256": "sha", "metadata": {"n_spills": 0}}
                             for phase in ("product", "reconstruction")} for label in gpu.LABELS},
        "timing": None,
    }
    assert gpu.summarize(raw)["gpu_execution_feasible"]
    raw["topology"]["B"]["product_events"] -= 1
    assert not gpu.summarize(raw)["gpu_execution_feasible"]
    raw["topology"]["B"]["product_events"] += 1
    raw["topology"]["B"]["kernel_names"] = ["_reconstruct"] + ["_block_product"] * gpu.RANKS["B"]
    assert not gpu.summarize(raw)["gpu_execution_feasible"]
    raw["topology"]["B"]["kernel_names"] = ["_block_product"] * gpu.RANKS["B"] + ["_reconstruct"]
    raw["cases"][0]["checks"]["B"]["passed"] = False
    assert not gpu.summarize(raw)["gpu_execution_feasible"]


def test_retained_gpu_evidence_replays_offline(tmp_path):
    bundle = study.OUT / "gpu-probe-n96-eager-s1"
    result = gpu.verify(bundle)
    assert result["status"] == "PASS"
    assert result["correctness_passed"]
    assert result["topology_passed"]
    assert result["timing_decision"] == "EXPLORATORY_SINGLE_SESSION_NO_SPEEDUP_CLAIM"
    copy = tmp_path / "tampered_gpu_bundle"
    shutil.copytree(bundle, copy)
    ptx = copy / "B_product.ptx"
    ptx.write_bytes(ptx.read_bytes() + b"// tampered\n")
    with pytest.raises(ValueError, match="manifest mismatch"):
        gpu.verify(copy)
