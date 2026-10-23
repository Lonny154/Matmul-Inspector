"""CPU-only fixtures for the preregistration consistency gate."""

from copy import deepcopy
import hashlib
import json
import re
import sys

import pytest

from scripts import freeze_mi_research_10 as frozen
from scripts import mi_research_10_preregistration as gate
from scripts import run_mi_research_10 as runner


def _new_plan() -> dict:
    plan = json.loads(frozen.PLAN.read_text())
    del plan["screening_hypotheses"]
    plan["descriptor_claims"] = gate.build_claims(
        plan, "perminov55_four_cta", "perminov58_four_cta")
    return plan


def test_validated_summary_covers_every_numeric_descriptor_without_mutation() -> None:
    plan = _new_plan()
    original = deepcopy(plan)
    comparison = gate.validate(plan)
    assert plan == original
    assert set(comparison) == set(plan["descriptor_claims"]["fields"])
    assert len(comparison) >= 15
    for field in ("whole_circuit_additions", "per_grid_gate_instances",
                  "per_grid_dot_instances", "per_grid_extra_dots_due_to_output_slicing",
                  "max_estimated_live_bytes_per_cta", "max_dependency_path_depth_operations",
                  "per_grid_gates_by_phase.U", "cta_count"):
        assert field in comparison
    assert "max_estimated_live_bytes_per_cta: candidate 63488 vs control 59392 (+4096; more)" in gate.summary(plan)


def test_exact_historical_4096_byte_contradiction_is_rejected() -> None:
    plan = _new_plan()
    field = "max_estimated_live_bytes_per_cta"
    plan["descriptor_claims"]["fields"][field] = {"direction": "fewer", "delta": -4096}
    before = deepcopy(plan)
    with pytest.raises(ValueError, match=r"max_estimated_live_bytes_per_cta: expected more .*delta \+4096.*stated fewer \(delta -4096\)"):
        gate.validate(plan)
    assert plan == before  # Failed validation must not fix the preregistration.


@pytest.mark.parametrize("field,wrong_direction,wrong_delta", [
    ("per_grid_gate_instances", "fewer", -12),
    ("per_grid_dot_instances", "more", 1),
    ("max_dependency_path_depth_operations", "more", 3),
    ("per_grid_gates_by_phase.U", "equal", 0),
    ("per_grid_extra_dots_due_to_output_slicing", "fewer", -1),
])
def test_other_directional_claims_are_checked(field: str, wrong_direction: str, wrong_delta: int) -> None:
    plan = _new_plan()
    plan["descriptor_claims"]["fields"][field] = {"direction": wrong_direction, "delta": wrong_delta}
    with pytest.raises(ValueError, match=f"{re.escape(field)}: expected"):
        gate.validate(plan)


def test_exact_delta_and_complete_coverage_are_required() -> None:
    plan = _new_plan()
    plan["descriptor_claims"]["fields"]["max_estimated_live_bytes_per_cta"]["delta"] = 2048
    with pytest.raises(ValueError, match=r"expected more .*delta \+4096.*stated more \(delta \+2048\)"):
        gate.validate(plan)
    del plan["descriptor_claims"]["fields"]["per_grid_dot_instances"]
    with pytest.raises(ValueError, match=r"missing=.*per_grid_dot_instances"):
        gate.validate(plan)


def test_historical_plan_is_immutable_and_cannot_be_recompiled(monkeypatch, tmp_path) -> None:
    original_hash = hashlib.sha256(frozen.PLAN.read_bytes()).hexdigest()
    historical = json.loads(frozen.PLAN.read_text())
    with pytest.raises(ValueError, match="missing descriptor_claims"):
        gate.validate(historical)
    monkeypatch.setattr(runner.frozen, "main", lambda: None)
    monkeypatch.setattr(runner, "OUT", tmp_path / "unused-gpu-output")
    with pytest.raises(ValueError, match="missing descriptor_claims"):
        runner.run()  # Fails before importing torch or creating the output directory.
    assert not runner.OUT.exists()
    assert hashlib.sha256(frozen.PLAN.read_bytes()).hexdigest() == original_hash


def test_structured_claims_cannot_hide_legacy_prose() -> None:
    plan = _new_plan()
    plan["screening_hypotheses"] = ["58 has 4096 fewer modeled live bytes"]
    with pytest.raises(ValueError, match="legacy screening_hypotheses"):
        gate.validate(plan)


def test_other_mapping_recomputes_directions_from_its_own_descriptors() -> None:
    prior = json.loads((frozen.costs.OUT / "analysis.json").read_text())
    plan = {"candidates": [
        {"id": "55_36", "role": "retained_compiled_control", "mapping": "thirty_six_cta",
         "descriptors": prior["circuits"]["55"]["thirty_six_cta"]["cost_vector"]},
        {"id": "58_36", "role": "prospective_unbenchmarked_candidate", "mapping": "thirty_six_cta",
         "descriptors": prior["circuits"]["58"]["thirty_six_cta"]["cost_vector"]},
    ]}
    plan["descriptor_claims"] = gate.build_claims(plan, "55_36", "58_36")
    compared = gate.validate(plan)
    assert compared["max_estimated_live_bytes_per_cta"]["direction"] == "fewer"
    assert compared["max_estimated_live_bytes_per_cta"]["delta"] == -2048
    assert compared["per_grid_dot_instances"]["delta"] == -24


def test_cli_validates_offline_fixture(tmp_path, monkeypatch, capsys) -> None:
    path = tmp_path / "prospective_plan.json"
    path.write_text(json.dumps(_new_plan()))
    monkeypatch.setattr(sys, "argv", ["mi_research_10_preregistration.py", str(path)])
    gate.main()
    assert "max_estimated_live_bytes_per_cta: candidate 63488 vs control 59392 (+4096; more)" in capsys.readouterr().out


def test_cli_rejects_historical_plan_without_a_traceback(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["mi_research_10_preregistration.py", str(frozen.PLAN)])
    with pytest.raises(SystemExit) as result:
        gate.main()
    assert result.value.code == 2
    assert "preregistration rejected: missing descriptor_claims" in capsys.readouterr().err
