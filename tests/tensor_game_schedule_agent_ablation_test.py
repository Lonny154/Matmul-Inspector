"""CPU-only checks for the matched Live-2 model decision ablation."""

from __future__ import annotations

import io
import json

import pytest

import matmul_inspector.tensor_game_schedule_agent as agent
from matmul_inspector.tensor_game_schedule_agent_ablation import (
    assess_proposal, canonical, common_context, compare_trials, condition_feedback,
    static_catalog,
)
from scripts import run_schedule_agent_decision_ablation as study


def test_verified_control_has_no_batch1_measurement_results():
    inputs = study.load_inputs()
    control, treatment = (inputs["conditions"][name] for name in ("control", "treatment"))
    assert common_context(control) == common_context(treatment)
    assert control["batch1_evidence"] == {"status": "WITHHELD"}
    control_text = canonical(control).decode()
    assert '"median_ms"' not in control_text
    assert '"round_medians_ms"' not in control_text
    assert '"coefficient_of_variation"' not in control_text
    assert '"pairs"' not in control_text
    assert '"interpretation"' not in control_text
    assert all(str(row["median_ms"]) not in control_text for row in
               inputs["feedback"]["measurements"]["timings"])
    assert len(control["candidate_catalog"]["schedules"]) == 39
    assert len(treatment["batch1_evidence"]["timings"]) == 16
    assert len(treatment["batch1_evidence"]["pairs"]) == 16
    assert treatment["batch1_evidence"]["interpretation"]["indeterminate_pairs"] == 7


def test_family_identity_is_checked_against_corpus():
    inputs = study.load_inputs()
    catalog = static_catalog(inputs["candidates"], inputs["batch1"])
    bad = [dict(row) for row in catalog]
    bad[0] = {**bad[0], "algorithm": "project-rank8" if bad[0]["algorithm"] !=
              "project-rank8" else "strassen"}
    with pytest.raises(ValueError, match="identity differs"):
        condition_feedback(inputs["feedback"], bad, "control")


def mocked_responses(monkeypatch, proposals):
    requests = []
    def fake_urlopen(request, timeout):
        body = json.loads(request.data)
        requests.append(body)
        proposal = proposals[len(requests) - 1]
        response = {"id": f"resp_mock_{len(requests)}", "status": "completed",
                    "usage": {"input_tokens": 1800, "output_tokens": 200,
                              "total_tokens": 2000},
                    "output": [{"content": [{"type": "output_text",
                                             "text": json.dumps(proposal)}]}]}
        return io.BytesIO(json.dumps(response).encode())
    monkeypatch.setattr(agent, "urlopen", fake_urlopen)
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-secret-never-retain")
    return requests


def proposal(scope: list[str], *, reason: str = "Compare structure") -> dict:
    result = agent.fallback_proposal(2)
    result["candidate_hashes"] = scope
    result["reason"] = reason
    return result


def test_three_matched_pairs_six_calls_and_offline_replay(tmp_path, monkeypatch):
    inputs = study.load_inputs()
    eligible = inputs["feedback"]["eligible_batch2_hashes"]
    control = proposal(eligible[:4])
    treatment = proposal(eligible[4:8], reason="0.091584 ms and unstable timing need review")
    requests = mocked_responses(monkeypatch, [control, treatment, treatment,
                                               control, control, treatment])
    output = tmp_path / "ablation"
    result = study.run("gpt-4.1-mini", 3, output)
    assert result["status"] == "COMPLETED"
    assert result["api_calls"] == len(requests) == 6
    assert result["reported_tokens"] == 12_000
    assert result["summary"]["valid_pairs"] == 3
    assert result["summary"]["consistent_decision_difference"] is True
    assert result["summary"]["causal_effect_or_performance_improvement_established"] is False
    schemas = [request["text"]["format"]["schema"] for request in requests]
    assert all(schema == schemas[0] for schema in schemas)
    assert all(len(canonical(request)) < study.MAX_REQUEST_BYTES for request in requests)
    prompts = [json.loads(request["input"][0]["content"])["feedback"]
               for request in requests]
    assert all(common_context(prompt) == common_context(prompts[0]) for prompt in prompts)
    assert [prompt["batch1_evidence"]["status"] for prompt in prompts] == [
        "WITHHELD", "AVAILABLE", "AVAILABLE", "WITHHELD", "WITHHELD", "AVAILABLE"]
    assert all('"median_ms"' not in canonical(prompt).decode() for prompt in prompts
               if prompt["batch1_evidence"]["status"] == "WITHHELD")
    assert "test-only-secret-never-retain" not in "".join(
        item.read_text() for item in output.iterdir())
    assert study.replay(output)["selection_replay_exact"] is True
    monkeypatch.delenv("OPENAI_API_KEY")
    monkeypatch.setattr(agent, "urlopen", lambda *args, **kwargs: pytest.fail("replay called API"))
    assert study.replay(output)["records"] == 6


def test_invalid_proposal_is_retained_without_repair(tmp_path, monkeypatch):
    invalid = agent.fallback_proposal(2)
    invalid["candidate_hashes"] = ["runtime"]
    requests = mocked_responses(monkeypatch, [invalid, agent.fallback_proposal(2)])
    output = tmp_path / "invalid"
    result = study.run("test-model", 1, output)
    assert len(requests) == result["api_calls"] == 2
    record = json.loads((output / "trial1_control.json").read_text())
    assert record["proposal"]["candidate_hashes"] == ["runtime"]
    assert record["validation"]["valid"] is False
    assert result["summary"]["valid_pairs"] == 0
    assert study.replay(output)["records"] == 2


def test_grounding_flags_mixed_family_and_real_timing_reference():
    inputs = study.load_inputs()
    original = json.loads((study.EVIDENCE / "batch2_proposal.json").read_text())["proposal"]
    proposal_text = dict(original)
    proposal_text["reason"] = "All selected project-rank8 schedules had 0.091584 ms; timing is unstable"
    assessment = assess_proposal(proposal_text, inputs["candidates"], inputs["batch1"],
                                 inputs["observations"], inputs["feedback"])
    assert set(assessment["selected_families"].values()) == {"project-rank8", "strassen"}
    assert assessment["family_claim_review_required"] is True
    assert assessment["rationale_ms_values_matching_a_batch1_median"] == [0.091584]
    assert assessment["mentions_uncertainty"] is True
    assert assessment["free_text_claims_require_manual_review"] is True


def test_measurement_citation_requires_matching_hash_mode_and_value():
    inputs = study.load_inputs()
    original = json.loads((study.EVIDENCE / "batch2_proposal.json").read_text())["proposal"]
    first = next(row for row in inputs["feedback"]["measurements"]["timings"]
                 if row["mode"] == "CUDA_GRAPH")
    claim = dict(original)
    claim["reason"] = (f'{first["schedule_hash"][:8]} in CUDA_GRAPH measured '
                       f'{first["median_ms"]:.6f} ms; uncertainty remains')
    assessment = assess_proposal(claim, inputs["candidates"], inputs["batch1"],
                                 inputs["observations"], inputs["feedback"])
    assert assessment["checkable_timing_citations"][0]["matches_saved_median"] is True
    claim["reason"] = f'{first["schedule_hash"][:8]} in EAGER measured 0.000001 ms'
    assessment = assess_proposal(claim, inputs["candidates"], inputs["batch1"],
                                 inputs["observations"], inputs["feedback"])
    assert assessment["checkable_timing_citations"][0]["matches_saved_median"] is False


def test_budget_and_hash_tamper_stop_offline(tmp_path, monkeypatch):
    mocked_responses(monkeypatch, [])
    with pytest.raises(ValueError, match="one through three"):
        study.run("test-model", 4, tmp_path / "too-many")
    assert not (tmp_path / "too-many").exists()
    mocked_responses(monkeypatch, [agent.fallback_proposal(2), agent.fallback_proposal(2)])
    output = tmp_path / "valid"
    study.run("test-model", 1, output)
    (output / "trial1_control.json").write_text("{}\n")
    with pytest.raises((ValueError, AssertionError)):
        study.replay(output)


def test_repeated_model_variability_is_not_called_consistent():
    records = []
    for trial in (1, 2):
        for condition in ("control", "treatment"):
            choice = "a" if condition == "control" else ("b" if trial == 1 else "c")
            records.append({"trial": trial, "condition": condition,
                            "validation": {"valid": True},
                            "assessment": {"substantive_fields": {
                                "experiment_family": "STRUCTURAL_DIVERSITY",
                                "descriptors": [], "execution_mode": choice,
                                "candidate_hashes": [], "comparison_preferences": []},
                                           "selected_hashes": [choice],
                                           "mentions_uncertainty": False}})
    result = compare_trials(records)
    assert result["consistent_decision_difference"] is False
    assert result["within_condition_decision_variability_observed"] is True
    assert result["treatment_unique_decisions"] == 2


def test_provider_failure_keeps_attempt_metadata_without_credential(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-secret-never-retain")
    calls = []
    def failing_urlopen(request, timeout):
        calls.append(1)
        raise TimeoutError("sensitive provider text must not be retained")
    monkeypatch.setattr(agent, "urlopen", failing_urlopen)
    output = tmp_path / "provider-failed"
    with pytest.raises(TimeoutError):
        study.run("test-model", 1, output)
    assert len(calls) == 1
    attempt = json.loads((output / "trial1_control.json").read_text())
    assert attempt["provider_error_kind"] == "TimeoutError"
    assert attempt["request_body"]["model"] == "test-model"
    assert json.loads((output / "failure.json").read_text())["api_calls"] == 1
    assert "test-only-secret-never-retain" not in "".join(
        item.read_text() for item in output.iterdir())
