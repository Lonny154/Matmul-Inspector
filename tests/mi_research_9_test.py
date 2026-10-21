"""CPU-only regression checks for output-slice dependency and cost analysis."""

import json

import pytest

from scripts import analyze_mi_research_9 as study
from scripts import run_mi_research_6c as exact


@pytest.fixture(scope="module")
def cases():
    source = json.loads(exact.SOURCE.read_text())
    cert = json.loads(exact.CERT.read_text())
    retained = json.loads((exact.OUT / "analysis.json").read_text())["configurations"]
    return {label: (graph, retained[f"perminov_{label}"]["schedules"]["liveness"]["order"])
            for label in ("55", "58")
            for graph, _ in [exact.build(source, cert, optimized=label == "55")]}


def test_dependency_traversal_and_output_ownership(cases):
    graph, order = cases["55"]
    roles = study.mapping_roles("thirty_six_cta")
    study.validate_ownership(roles, graph)
    assert len(roles) == 9
    assert sum(len(role["tile_positions"]) for role in roles) == 36
    selected = study.closure(graph, ("C4",))
    assert {name for name in selected if graph.by_id[name]["kind"] == "PRODUCT"} == {"M12", "M14", "M21"}
    assert study.role_cost(graph, order, ("C4",))["gate_instances"] == 2
    assert all(set(graph.by_id[name]["deps"]) <= selected for name in selected)
    assert study.closure(graph, tuple(f"C{i}" for i in range(9))) == set(graph.by_id)


@pytest.mark.parametrize("label,dots,gates,extra_dots,extra_gates", [
    ("55", 256, 704, 164, 484), ("58", 232, 568, 140, 336)])
def test_counts_duplication_and_order(cases, label, dots, gates, extra_dots, extra_gates):
    graph, order = cases[label]
    result = study.mapping_cost(graph, order, "thirty_six_cta")
    vector = result["cost_vector"]
    assert (vector["per_grid_dot_instances"], vector["per_grid_gate_instances"]) == (dots, gates)
    assert (vector["per_grid_extra_dots_due_to_output_slicing"],
            vector["per_grid_extra_gates_due_to_output_slicing"]) == (extra_dots, extra_gates)
    assert sum(item["extra_instances_per_grid"] for item in result["cross_output_shared_expressions"]
               if item["kind"] == "PRODUCT") == extra_dots
    assert result == study.mapping_cost(graph, order, "thirty_six_cta")
    assert result["roles"]["output_4"]["products"] == ["M12", "M14", "M21"]


def test_four_cta_and_lifetime(cases):
    graph, order = cases["55"]
    result = study.mapping_cost(graph, order, "four_cta")
    vector = result["cost_vector"]
    assert (vector["cta_count"], vector["per_grid_dot_instances"],
            vector["per_grid_gate_instances"]) == (4, 92, 220)
    role = result["roles"]["all_outputs"]
    assert role["peak_live_intermediates"] == 17
    assert role["dependency_path_depth_operations"] == 15
    assert role["peak_estimated_live_bytes_including_inputs"] == 59392
    for name, life in role["lifetimes"].items():
        assert life["created_at"] < life["last_used_at"]
        assert life["last_used_at"] == max(role["execution_order"].index(use)
                                            for use in life["consumers"])
    # Stores are uses of their reconstruction nodes, with no new array.
    assert all(name not in role["lifetimes"] for name in (f"C{i}" for i in range(9)))


def test_local_sharing_and_byte_model(cases):
    graph, order = cases["55"]
    role = study.role_cost(graph, order, ("C3",))
    assert role["locally_shared_expressions"]
    assert all(len(role["lifetimes"][name]["consumers"]) > 1
               for name in role["locally_shared_expressions"])
    assert role["peak_estimated_live_bytes_including_inputs"] == 4 * role["peak_logical_fp32_elements_including_inputs"]
    assert role["peak_live_intermediates"] <= role["peak_live_values_including_inputs"]


def test_invalid_mapping_and_artifact_replay(cases, tmp_path, monkeypatch):
    graph, order = cases["55"]
    with pytest.raises(ValueError, match="unknown output mapping"):
        study.mapping_cost(graph, order, "invented")
    with pytest.raises(ValueError, match="nonempty and unique"):
        study.closure(graph, ())
    with pytest.raises(ValueError, match="unknown or non-output"):
        study.closure(graph, ("M00",))
    roles = study.mapping_roles("thirty_six_cta")
    roles[0]["outputs"] = ["C1"]
    with pytest.raises(ValueError, match="exactly once"):
        study.validate_ownership(roles, graph)
    assert study.verify()["status"] == "PASS"
    copy = tmp_path / "analysis.json"
    copy.write_bytes(study.ANALYSIS.read_bytes())
    monkeypatch.setattr(study, "ANALYSIS", copy)
    copy.write_text(copy.read_text().replace('"schema_version": 1', '"schema_version": 2', 1))
    with pytest.raises(ValueError, match="hash manifest mismatch"):
        study.verify()


def test_retained_stable_cost_records():
    record = json.loads(study.ANALYSIS.read_text())
    assert record["tensor"] == {"checked": 729, "mismatches": 0, "unit": 27, "zero": 702}
    assert record["identical_factor_maps"]
    assert record["pareto_comparisons"]["thirty_six_cta"]["58_dominates_55"]
    assert not record["pareto_comparisons"]["four_cta"]["58_dominates_55"]
    assert record["historical_timing_decisions"] == ["INCONCLUSIVE", "INCONCLUSIVE"]
    assert record["circuits"]["55"]["thirty_six_cta"]["cost_vector"]["per_grid_zero_net_product_dependency_instances"] == 24
    assert record["circuits"]["58"]["thirty_six_cta"]["cost_vector"]["per_grid_zero_net_product_dependency_instances"] == 0
    assert record["circuits"]["55"]["thirty_six_cta"]["roles"]["output_6"]["zero_net_product_dependencies"] == ["M05", "M10", "M11", "M18"]
