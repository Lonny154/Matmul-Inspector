"""CPU-only validation of the bounded controller and saved HA-10 replay."""

from __future__ import annotations

from argparse import Namespace
from copy import deepcopy
import io
import json
from pathlib import Path
import re
import time

import pytest

from matmul_inspector.tensor_game_schedule_agent import (
    DESCRIPTORS, FAMILIES, FIELDS, LIMITS, FallbackPlanner, Limits, OpenAIPlanner,
    PlannerBudgetExceeded,
    ProposalError, ReplayPlanner,
    fallback_proposal, feedback_summary, selections, validate_proposal,
)
from matmul_inspector.tensor_game_schedule_reselection import (
    MODES, load_retained_corpus, make_observation, select_batch1,
)
from scripts import run_schedule_agent_prototype as runner
from scripts.audit_schedule_agent_bundle import audit

REPO = Path(__file__).resolve().parents[1]
HA10 = REPO / "artifacts/ha-schedule-10/gpu-v1"


@pytest.fixture(scope="module")
def corpus():
    candidates, _ = load_retained_corpus(REPO)
    return candidates, select_batch1(candidates)


def args(output, *, proposals=None, gpu=False):
    return Namespace(output=output, gpu=gpu, replay_ha10=HA10 if not gpu else None,
                     proposals=proposals, provider="fallback", model=None)


def live_feedback(batch):
    candidates, _ = load_retained_corpus(REPO)
    batch1 = select_batch1(candidates)
    if batch == 1:
        return {"frozen_batch1_hashes": [row["schedule_hash"] for row in batch1],
                "remaining_candidate_budget": LIMITS.max_gpu_candidates}
    return feedback_summary(batch1, [], [], [], candidates)


def mocked_openai_responses(monkeypatch, proposals):
    """Return request bodies while serving raw proposals without schema repair."""
    import matmul_inspector.tensor_game_schedule_agent as agent
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-secret")
    requests = []
    def fake_urlopen(request, timeout):
        requests.append(json.loads(request.data))
        proposal = proposals[len(requests) - 1]
        data = {"id": f"mock-{len(requests)}", "status": "completed",
                "usage": {"input_tokens": 80, "output_tokens": 20, "total_tokens": 100},
                "output": [{"content": [{"type": "output_text", "text": json.dumps(proposal)}]}]}
        return io.BytesIO(json.dumps(data).encode())
    monkeypatch.setattr(agent, "urlopen", fake_urlopen)
    return requests


def test_valid_proposals_and_stable_hashes(corpus):
    candidates, batch1 = corpus
    first = validate_proposal(fallback_proposal(1), batch=1,
                              candidates=candidates, batch1=batch1)
    second = validate_proposal(fallback_proposal(2), batch=2,
                               candidates=candidates, batch1=batch1)
    assert first["requested_configurations"] == 8
    assert second["requested_configurations"] == 4
    assert len({row["schedule_hash"] for row in batch1}) == 8
    assert all(len(row["schedule_hash"]) == 64 for row in batch1)
    assert FallbackPlanner().source != ReplayPlanner([first, second]).source


@pytest.mark.parametrize("key,value,reason", [
    ("experiment_family", "UNRESTRICTED", "family"),
    ("execution_mode", "MIXED", "mode"),
    ("requested_configurations", 5, "budget"),
    ("descriptors", ["imaginary"], "descriptor"),
    ("candidate_hashes", ["a" * 64], "ineligible"),
])
def test_invalid_proposal_rejected(corpus, key, value, reason):
    candidates, batch1 = corpus
    proposal = fallback_proposal(2)
    proposal[key] = value
    with pytest.raises(ProposalError, match=reason):
        validate_proposal(proposal, batch=2, candidates=candidates, batch1=batch1)


def test_extra_fields_and_frozen_batch1_scope_rejected(corpus):
    candidates, batch1 = corpus
    proposal = fallback_proposal(1)
    proposal["override_correctness"] = True
    with pytest.raises(ProposalError, match="exactly"):
        validate_proposal(proposal, batch=1, candidates=candidates, batch1=batch1)
    del proposal["override_correctness"]
    proposal["candidate_hashes"] = [batch1[0]["schedule_hash"]]
    with pytest.raises(ProposalError, match="frozen"):
        validate_proposal(proposal, batch=1, candidates=candidates, batch1=batch1)


def test_failed_and_small_scope_excluded(corpus):
    candidates, batch1 = corpus
    failed = next(row for row in candidates if row["retained_correctness"] != "PASS")
    proposal = fallback_proposal(2)
    proposal["candidate_hashes"] = [failed["schedule_hash"]]
    with pytest.raises(ProposalError, match="ineligible"):
        validate_proposal(proposal, batch=2, candidates=candidates, batch1=batch1)
    proposal["candidate_hashes"] = [next(row["schedule_hash"] for row in candidates
                                    if row["matrix_size"] == 256 and
                                    row["schedule_hash"] not in {b["schedule_hash"] for b in batch1})]
    with pytest.raises(ProposalError, match="at least four"):
        validate_proposal(proposal, batch=2, candidates=candidates, batch1=batch1)
    proposal = fallback_proposal(2)
    proposal["comparison_preferences"] = [failed["schedule_hash"]]
    with pytest.raises(ProposalError, match="ineligible"):
        validate_proposal(proposal, batch=2, candidates=candidates, batch1=batch1)


def test_mode_isolation_missing_and_contradictory_evidence(corpus):
    candidates, batch1 = corpus
    left, right = batch1[:2]
    def pair(left_ms, right_ms):
        return make_observation(left, right, mode="EAGER", left_ms=left_ms,
            right_ms=right_ms, provenance={"evidence_kind": "SYNTHETIC_TEST_FIXTURE"},
            benchmark={"rounds": 5}, correctness={left["schedule_hash"]: "PASS",
                                                right["schedule_hash"]: "PASS"})
    first = pair([1.0] * 5, [2.0] * 5)
    opposite = pair([2.0] * 5, [1.0] * 5)
    summary = feedback_summary(batch1, [], [], [first, opposite], candidates)
    assert summary["interpretation"]["contradictory_pairs"] == 1
    assert summary["interpretation"]["conflicted_descriptor_tokens"]["EAGER"]
    empty = selections(candidates, batch1, [], fallback_proposal(2))
    contradicted = selections(candidates, batch1, [first, opposite], fallback_proposal(2))
    assert contradicted["calibrated"] == empty["frozen"]
    eager_only = selections(candidates, batch1, [first], fallback_proposal(2))
    assert eager_only["calibrated"]["CUDA_GRAPH"] == empty["frozen"]["CUDA_GRAPH"]
    assert eager_only["calibrated"]["EAGER"] != empty["frozen"]["EAGER"]
    tampered = deepcopy(first)
    tampered["preferred"] = right["schedule_hash"]
    with pytest.raises(ValueError, match="hash"):
        selections(candidates, batch1, [tampered], fallback_proposal(2))


def test_candidate_scope_changes_only_guided_selection(corpus):
    candidates, batch1 = corpus
    selected = {row["schedule_hash"] for row in batch1}
    scope = [row["schedule_hash"] for row in candidates if row["matrix_size"] == 256 and
             row["retained_correctness"] == "PASS" and row["schedule_hash"] not in selected][-4:]
    proposal = fallback_proposal(2)
    proposal["candidate_hashes"] = scope
    validate_proposal(proposal, batch=2, candidates=candidates, batch1=batch1)
    decision = selections(candidates, batch1, [], proposal)
    assert {row["schedule_hash"] for row in decision["physical_batch2"]} == set(scope)
    assert decision["frozen"] == decision["calibrated"]
    assert decision["agent_changed_scope"]


def test_cpu_replay_is_deterministic_and_hashed(tmp_path):
    left, right = tmp_path / "a", tmp_path / "b"
    assert runner.run(args(left))["classification"] == "B"
    assert runner.run(args(right))["classification"] == "B"
    for name in ("batch1_selection.json", "batch1_feedback.json",
                 "batch2_controls_and_guidance.json", "control_comparisons.json"):
        assert (left / name).read_bytes() == (right / name).read_bytes()
    for directory in (left, right):
        manifest = json.loads((directory / "hash_manifest.json").read_text())
        assert manifest and all(runner.digest(directory / name) == value
                                for name, value in manifest.items())
        assert json.loads((directory / "resource_usage.json").read_text())["gpu_candidates"] == 0


def test_saved_proposal_can_prioritize_graph(tmp_path):
    saved = [fallback_proposal(1), fallback_proposal(2)]
    saved[1]["proposal_id"] = "saved-graph-priority"
    saved[1]["execution_mode"] = "CUDA_GRAPH"
    proposals = tmp_path / "proposals.json"
    proposals.write_text(json.dumps(saved))
    output = tmp_path / "run"
    assert runner.run(args(output, proposals=proposals))["classification"] == "B"
    selection = json.loads((output / "batch2_controls_and_guidance.json").read_text())
    assert selection["physical_mode"] == "CUDA_GRAPH"
    assert selection["agent_changed_mode"]
    assert selection["physical_batch2"] == selection["calibrated"]["CUDA_GRAPH"]
    assert json.loads((output / "batch2_proposal.json").read_text())["source"] == "SAVED_PROPOSAL_REPLAY"


def test_gpu_bundle_audits_and_replays_without_cuda(tmp_path):
    source = REPO / "artifacts/schedule-agent/gpu-v1"
    if not source.exists():
        pytest.skip("retained GPU prototype bundle not present")
    assert audit(source)["status"] == "PASS"
    output = tmp_path / "replay"
    proposals = REPO / "artifacts/schedule-agent/replay_proposals_v1.json"
    request = args(output, proposals=proposals)
    request.replay_ha10 = None
    request.replay_bundle = source
    assert runner.run(request)["classification"] == "B"
    assert json.loads((output / "replay_comparison.json").read_text())["batch2_selection_exact_match"]


def test_bundle_replay_loads_hashed_proposals_without_provider(tmp_path, monkeypatch):
    import matmul_inspector.tensor_game_schedule_agent as agent
    source = REPO / "artifacts/schedule-agent/gpu-v1"
    if not source.exists():
        pytest.skip("retained GPU prototype bundle not present")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(agent, "urlopen", lambda *args, **kwargs: pytest.fail("network in replay"))
    request = args(tmp_path / "replay")
    request.replay_ha10 = None
    request.replay_bundle = source
    assert runner.run(request)["classification"] == "B"
    assert json.loads((request.output / "replay_comparison.json").read_text())["batch2_selection_exact_match"]
    assert json.loads((request.output / "resource_usage.json").read_text())["provider_api_calls"] == 0


def test_artifact_size_budget_is_terminal(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "LIMITS", runner.LIMITS.__class__(max_artifact_bytes=50))
    bundle = runner.Bundle(tmp_path / "small")
    with pytest.raises(RuntimeError, match="artifact size budget"):
        bundle.write("large.json", {"payload": "x" * 100})
    assert not (bundle.path / "large.json").exists()


def test_malformed_saved_proposal_is_terminal_and_preserved(tmp_path):
    proposals = tmp_path / "bad.json"
    proposals.write_text(json.dumps([fallback_proposal(1), {"proposal_id": "bad"}]))
    output = tmp_path / "run"
    result = runner.run(args(output, proposals=proposals))
    assert result["status"] == "FAILED_VALIDATION"
    assert (output / "batch1_feedback.json").exists()
    assert not (output / "batch2_requested_hashes.json").exists()
    assert json.loads((output / "batch2_validation.json").read_text())["valid"] is False


def test_provider_missing_credentials_and_timeout_fallback(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ProposalError, match="required"):
        OpenAIPlanner("test-model")
    class TimedOut:
        source = "LIVE_TEST"
        def propose(self, batch, feedback):
            raise TimeoutError("provider timeout")
    candidates, batch1 = load_retained_corpus(REPO)[0], None
    batch1 = select_batch1(candidates)
    bundle = runner.Bundle(tmp_path / "bundle")
    proposal, source = runner.get_proposal(TimedOut(), 1, {}, bundle, candidates, batch1)
    assert source == "DETERMINISTIC_FALLBACK_AFTER_PROVIDER_ERROR"
    assert proposal["requested_configurations"] == LIMITS.batch1
    assert json.loads((bundle.path / "batch1_provider_error.json").read_text())["fallback_used"]


def test_live_adapter_parses_structured_response_without_network(monkeypatch):
    import matmul_inspector.tensor_game_schedule_agent as agent
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    proposal = fallback_proposal(1)
    response = {"id": "resp_test", "status": "completed",
                "usage": {"input_tokens": 80, "output_tokens": 20, "total_tokens": 100},
                "output": [{"content": [{"type": "output_text", "text": json.dumps(proposal)}]}]}
    class Response(io.BytesIO):
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.close()
    calls = []
    def fake_urlopen(request, timeout):
        calls.append((request.full_url, timeout, json.loads(request.data)))
        return Response(json.dumps(response).encode())
    monkeypatch.setattr(agent, "urlopen", fake_urlopen)
    planner = OpenAIPlanner("test-model")
    assert planner.propose(1, live_feedback(1)) == proposal
    assert planner.last_metadata["response_id"] == "resp_test"
    assert calls[0][:2] == ("https://api.openai.com/v1/responses", LIMITS.planner_timeout_seconds)
    assert calls[0][2]["max_output_tokens"] == LIMITS.max_planner_output_tokens_per_call
    assert calls[0][2]["store"] is False
    fields = calls[0][2]["text"]["format"]["schema"]["properties"]
    assert fields["candidate_hashes"]["maxItems"] == 0
    assert fields["requested_configurations"]["enum"] == [LIMITS.batch1]
    assert set(fields["comparison_preferences"]["items"]["enum"]) == set(
        live_feedback(1)["frozen_batch1_hashes"])
    assert planner.total_tokens == 100


def test_live_api_call_order_and_budget_are_enforced_without_network(monkeypatch):
    import matmul_inspector.tensor_game_schedule_agent as agent
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-secret")
    calls = []
    def fake_urlopen(request, timeout):
        calls.append(request)
        proposal = fallback_proposal(len(calls))
        data = {"id": f"resp_{len(calls)}", "status": "completed",
                "usage": {"input_tokens": 80, "output_tokens": 20, "total_tokens": 100},
                "output": [{"content": [{"type": "output_text", "text": json.dumps(proposal)}]}]}
        return io.BytesIO(json.dumps(data).encode())
    monkeypatch.setattr(agent, "urlopen", fake_urlopen)
    planner = OpenAIPlanner("test-model")
    with pytest.raises(ProposalError, match="two-batch order"):
        planner.propose(2, {})
    assert planner.propose(1, live_feedback(1)) == fallback_proposal(1)
    assert planner.propose(2, live_feedback(2)) == fallback_proposal(2)
    with pytest.raises(PlannerBudgetExceeded, match="API-call budget"):
        planner.propose(3, {})
    assert len(calls) == 2
    assert planner.total_tokens == 200


def test_retained_batch1_feedback_fits_live_request_budget(monkeypatch):
    import matmul_inspector.tensor_game_schedule_agent as agent
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-secret")
    requests = []
    def fake_urlopen(request, timeout):
        requests.append(request.data)
        data = {"status": "completed", "usage": {"input_tokens": 100,
                "output_tokens": 20, "total_tokens": 120},
                "output": [{"content": [{"type": "output_text",
                            "text": json.dumps(fallback_proposal(len(requests)))}]}]}
        return io.BytesIO(json.dumps(data).encode())
    monkeypatch.setattr(agent, "urlopen", fake_urlopen)
    planner = OpenAIPlanner("test-model")
    planner.propose(1, live_feedback(1))
    feedback = json.loads((REPO / "artifacts/schedule-agent/gpu-v1/batch1_feedback.json").read_text())
    planner.propose(2, feedback)
    assert len(requests) == 2
    assert len(requests[1]) < LIMITS.max_planner_request_bytes
    request = json.loads(requests[1])
    fields = request["text"]["format"]["schema"]["properties"]
    assert "eligible_batch2_hashes" not in json.loads(request["input"][0]["content"])["feedback"]
    assert len(feedback["eligible_batch2_hashes"]) == 31
    scope = fields["candidate_hashes"]["anyOf"]
    assert scope[0]["maxItems"] == 0
    assert scope[1]["minItems"] == LIMITS.batch2
    assert scope[1]["maxItems"] == 16
    assert scope[1]["items"] == {"$ref": "#/$defs/eligible_hash"}
    assert set(request["text"]["format"]["schema"]["$defs"]["eligible_hash"]["enum"]) == set(
        feedback["eligible_batch2_hashes"])
    assert fields["requested_configurations"]["enum"] == [LIMITS.batch2]
    assert set(fields["comparison_preferences"]["items"]["enum"]) == set(
        feedback["eligible_batch2_hashes"]) | {
            row["schedule_hash"] for row in feedback["measurements"]["candidates"]}


def test_live_schema_audit_matches_controller_choices_in_both_batches(monkeypatch):
    feedback2 = json.loads((REPO / "artifacts/schedule-agent/gpu-v1/batch1_feedback.json").read_text())
    requests = mocked_openai_responses(monkeypatch, [fallback_proposal(1), fallback_proposal(2)])
    planner = OpenAIPlanner("test-model")
    planner.propose(1, live_feedback(1))
    planner.propose(2, feedback2)
    for batch, fixed in ((1, 8), (2, 4)):
        schema = requests[batch - 1]["text"]["format"]["schema"]
        fields = schema["properties"]
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(fields) == FIELDS
        for key in ("proposal_id", "hypothesis", "reason"):
            pattern = fields[key]["pattern"]
            assert fields[key]["type"] == "string"
            assert re.fullmatch(pattern, "a" * 400)
            assert not re.fullmatch(pattern, "a" * 401)
            assert not re.fullmatch(pattern, "")
            assert not re.fullmatch(pattern, "   ")
        assert set(fields["experiment_family"]["enum"]) == set(FAMILIES)
        assert set(fields["execution_mode"]["enum"]) == set(MODES)
        assert set(fields["descriptors"]["items"]["enum"]) == set(DESCRIPTORS)
        assert fields["descriptors"]["maxItems"] == len(DESCRIPTORS)
        assert fields["requested_configurations"]["enum"] == [fixed]
        assert fields["comparison_preferences"]["maxItems"] == 4
        if batch == 1:
            assert fields["candidate_hashes"]["maxItems"] == 0
            allowed = live_feedback(1)["frozen_batch1_hashes"]
        else:
            alternatives = fields["candidate_hashes"]["anyOf"]
            assert [(part.get("minItems"), part["maxItems"]) for part in alternatives] == [
                (None, 0), (4, 16)]
            assert alternatives[1]["items"] == {"$ref": "#/$defs/eligible_hash"}
            assert set(schema["$defs"]["eligible_hash"]["enum"]) == set(
                feedback2["eligible_batch2_hashes"])
            allowed = feedback2["eligible_batch2_hashes"] + [
                row["schedule_hash"] for row in feedback2["measurements"]["candidates"]]
        assert set(fields["comparison_preferences"]["items"]["enum"]) == set(allowed)
    assert len(requests) == 2


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("key,value,reason", [
    ("proposal_id", "", "proposal_id"),
    ("proposal_id", None, "proposal_id"),
    ("hypothesis", "   ", "hypothesis"),
    ("hypothesis", "h" * 401, "hypothesis"),
    ("reason", "r" * 401, "reason"),
    ("experiment_family", "UNKNOWN", "family"),
    ("execution_mode", "UNKNOWN", "mode"),
    ("descriptors", ["unknown"], "descriptor"),
    ("descriptors", "launch_groups", "unique bounded list"),
    ("descriptors", ["launch_groups", "launch_groups"], "unique"),
    ("comparison_preferences", ["runtime"], "ineligible hash"),
    ("comparison_preferences", [None], "contain strings"),
    ("comparison_preferences", ["a" * 64] * 2, "unique"),
    ("comparison_preferences", ["a" * 64] * 5, "unique bounded"),
    ("requested_configurations", 5, "fixed batch budget"),
    ("requested_configurations", True, "fixed batch budget"),
])
def test_live_proposal_validator_rejects_malformed_and_out_of_range_fields(
        corpus, batch, key, value, reason):
    candidates, batch1 = corpus
    proposal = fallback_proposal(batch)
    proposal[key] = value
    with pytest.raises(ProposalError, match=reason):
        validate_proposal(proposal, batch=batch, candidates=candidates, batch1=batch1)


@pytest.mark.parametrize("batch", [1, 2])
def test_live_text_boundaries_and_required_fields(corpus, batch):
    candidates, batch1 = corpus
    proposal = fallback_proposal(batch)
    for key in ("proposal_id", "hypothesis", "reason"):
        proposal[key] = "x" * 400
        assert validate_proposal(proposal, batch=batch, candidates=candidates,
                                 batch1=batch1)[key] == "x" * 400
        proposal[key] = "x" * 401
        with pytest.raises(ProposalError, match=key):
            validate_proposal(proposal, batch=batch, candidates=candidates, batch1=batch1)
        proposal[key] = fallback_proposal(batch)[key]
    del proposal["reason"]
    with pytest.raises(ProposalError, match="exactly the approved fields"):
        validate_proposal(proposal, batch=batch, candidates=candidates, batch1=batch1)


def test_live_batch_specific_hash_scope_and_preferences(corpus):
    candidates, batch1 = corpus
    feedback = live_feedback(2)
    eligible = feedback["eligible_batch2_hashes"]
    first = fallback_proposal(1)
    first["comparison_preferences"] = [batch1[0]["schedule_hash"]]
    assert validate_proposal(first, batch=1, candidates=candidates, batch1=batch1)
    first["candidate_hashes"] = [eligible[0]]
    with pytest.raises(ProposalError, match="frozen"):
        validate_proposal(first, batch=1, candidates=candidates, batch1=batch1)
    second = fallback_proposal(2)
    second["comparison_preferences"] = [batch1[0]["schedule_hash"], eligible[0]]
    second["candidate_hashes"] = eligible[:4]
    assert validate_proposal(second, batch=2, candidates=candidates, batch1=batch1)
    second["candidate_hashes"] = eligible[:3]
    with pytest.raises(ProposalError, match="at least four"):
        validate_proposal(second, batch=2, candidates=candidates, batch1=batch1)
    second["candidate_hashes"] = eligible[:17]
    with pytest.raises(ProposalError, match="unique bounded list"):
        validate_proposal(second, batch=2, candidates=candidates, batch1=batch1)
    second["candidate_hashes"] = [batch1[0]["schedule_hash"]] + eligible[:3]
    with pytest.raises(ProposalError, match="ineligible hash"):
        validate_proposal(second, batch=2, candidates=candidates, batch1=batch1)


def test_v3_length_failure_is_preserved_by_mocked_batch2_validation(
        tmp_path, monkeypatch, corpus):
    candidates, batch1 = corpus
    invalid = fallback_proposal(2)
    invalid["hypothesis"] = "h" * 445  # Observed v3 length; no model text is copied.
    invalid["candidate_hashes"] = live_feedback(2)["eligible_batch2_hashes"][:1]
    requests = mocked_openai_responses(monkeypatch, [fallback_proposal(1), invalid])
    planner = OpenAIPlanner("test-model")
    bundle = runner.Bundle(tmp_path / "v3-length-regression")
    runner.get_proposal(planner, 1, live_feedback(1), bundle, candidates, batch1)
    with pytest.raises(ProposalError, match="hypothesis must be a nonempty string"):
        runner.get_proposal(planner, 2, live_feedback(2), bundle, candidates, batch1)
    assert len(requests) == 2
    pattern = requests[1]["text"]["format"]["schema"]["properties"]["hypothesis"]["pattern"]
    assert not re.fullmatch(pattern, invalid["hypothesis"])
    assert json.loads((bundle.path / "batch2_proposal.json").read_text())["proposal"] == invalid
    assert json.loads((bundle.path / "batch2_validation.json").read_text()) == {
        "valid": False, "reason": "hypothesis must be a nonempty string of at most 400 characters"}
    latent = {**invalid, "hypothesis": "valid test fixture"}
    with pytest.raises(ProposalError, match="at least four"):
        validate_proposal(latent, batch=2, candidates=candidates, batch1=batch1)


def test_live_batch1_runtime_preference_is_rejected_and_schema_excludes_it(tmp_path, monkeypatch):
    import matmul_inspector.tensor_game_schedule_agent as agent
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-secret")
    candidates, _ = load_retained_corpus(REPO)
    batch1 = select_batch1(candidates)
    invalid = fallback_proposal(1)
    invalid["comparison_preferences"] = ["runtime", "memory_usage"]
    with pytest.raises(ProposalError, match="ineligible hash: runtime"):
        validate_proposal(invalid, batch=1, candidates=candidates, batch1=batch1)
    requests = []
    def fake_urlopen(request, timeout):
        requests.append(json.loads(request.data))
        data = {"status": "completed", "usage": {"input_tokens": 100,
                "output_tokens": 20, "total_tokens": 120},
                "output": [{"content": [{"type": "output_text", "text": json.dumps(invalid)}]}]}
        return io.BytesIO(json.dumps(data).encode())
    monkeypatch.setattr(agent, "urlopen", fake_urlopen)
    bundle = runner.Bundle(tmp_path / "rejected")
    with pytest.raises(ProposalError, match="ineligible hash: runtime"):
        runner.get_proposal(OpenAIPlanner("test-model"), 1, live_feedback(1),
                            bundle, candidates, batch1)
    allowed = requests[0]["text"]["format"]["schema"]["properties"][
        "comparison_preferences"]["items"]["enum"]
    assert "runtime" not in allowed and "memory_usage" not in allowed
    assert set(allowed) == {row["schedule_hash"] for row in batch1}
    assert json.loads((bundle.path / "batch1_proposal.json").read_text())["proposal"] == invalid
    assert json.loads((bundle.path / "batch1_validation.json").read_text())["valid"] is False


def test_live_batch1_requested_five_is_rejected_and_preserved(tmp_path, monkeypatch, corpus):
    import matmul_inspector.tensor_game_schedule_agent as agent
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-secret")
    candidates, batch1 = corpus
    invalid = fallback_proposal(1)
    invalid["requested_configurations"] = 5  # Observed in the second live attempt.
    requests = []
    def fake_urlopen(request, timeout):
        requests.append(json.loads(request.data))
        data = {"status": "completed", "usage": {"input_tokens": 80,
                "output_tokens": 20, "total_tokens": 100},
                "output": [{"content": [{"type": "output_text", "text": json.dumps(invalid)}]}]}
        return io.BytesIO(json.dumps(data).encode())
    monkeypatch.setattr(agent, "urlopen", fake_urlopen)
    bundle = runner.Bundle(tmp_path / "rejected-budget")
    with pytest.raises(ProposalError, match="requested_configurations exceeds or changes"):
        runner.get_proposal(OpenAIPlanner("test-model"), 1, live_feedback(1),
                            bundle, candidates, batch1)
    fields = requests[0]["text"]["format"]["schema"]["properties"]
    assert fields["requested_configurations"]["enum"] == [8]
    assert json.loads((bundle.path / "batch1_proposal.json").read_text())["proposal"] == invalid
    validation = json.loads((bundle.path / "batch1_validation.json").read_text())
    assert validation == {"valid": False,
        "reason": "requested_configurations exceeds or changes the fixed batch budget"}


def test_live_batch2_requested_five_is_rejected_and_preserved(tmp_path, monkeypatch, corpus):
    import matmul_inspector.tensor_game_schedule_agent as agent
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-secret")
    candidates, batch1 = corpus
    invalid = fallback_proposal(2)
    invalid["requested_configurations"] = 5
    requests = []
    def fake_urlopen(request, timeout):
        requests.append(json.loads(request.data))
        proposal = fallback_proposal(1) if len(requests) == 1 else invalid
        data = {"status": "completed", "usage": {"input_tokens": 80,
                "output_tokens": 20, "total_tokens": 100},
                "output": [{"content": [{"type": "output_text", "text": json.dumps(proposal)}]}]}
        return io.BytesIO(json.dumps(data).encode())
    monkeypatch.setattr(agent, "urlopen", fake_urlopen)
    bundle = runner.Bundle(tmp_path / "rejected-batch2-budget")
    planner = OpenAIPlanner("test-model")
    runner.get_proposal(planner, 1, live_feedback(1), bundle, candidates, batch1)
    with pytest.raises(ProposalError, match="requested_configurations exceeds or changes"):
        runner.get_proposal(planner, 2, live_feedback(2), bundle, candidates, batch1)
    fields = requests[1]["text"]["format"]["schema"]["properties"]
    assert fields["requested_configurations"]["enum"] == [4]
    assert json.loads((bundle.path / "batch2_proposal.json").read_text())["proposal"] == invalid
    assert json.loads((bundle.path / "batch2_validation.json").read_text())["valid"] is False


def test_live_request_and_reported_token_budgets_fail_closed(monkeypatch):
    import matmul_inspector.tensor_game_schedule_agent as agent
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-secret")
    called = []
    def fake_urlopen(request, timeout):
        called.append(1)
        data = {"status": "completed", "usage": {"input_tokens": 32_001,
                "output_tokens": 20, "total_tokens": 32_021},
                "output": [{"content": [{"type": "output_text",
                            "text": json.dumps(fallback_proposal(1))}]}]}
        return io.BytesIO(json.dumps(data).encode())
    monkeypatch.setattr(agent, "urlopen", fake_urlopen)
    planner = OpenAIPlanner("test-model", Limits(max_planner_request_bytes=100))
    with pytest.raises(PlannerBudgetExceeded, match="request-size"):
        planner.propose(1, live_feedback(1))
    assert called == []
    with pytest.raises(PlannerBudgetExceeded, match="token budget"):
        OpenAIPlanner("test-model").propose(1, live_feedback(1))
    assert called == [1]


def test_live_cumulative_token_budget_blocks_second_call(monkeypatch):
    import matmul_inspector.tensor_game_schedule_agent as agent
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-secret")
    calls = []
    def fake_urlopen(request, timeout):
        calls.append(1)
        data = {"status": "completed", "usage": {"input_tokens": 980,
                "output_tokens": 20, "total_tokens": 1000},
                "output": [{"content": [{"type": "output_text",
                            "text": json.dumps(fallback_proposal(1))}]}]}
        return io.BytesIO(json.dumps(data).encode())
    monkeypatch.setattr(agent, "urlopen", fake_urlopen)
    planner = OpenAIPlanner("test-model", Limits(max_planner_total_tokens=33_000))
    planner.propose(1, live_feedback(1))
    with pytest.raises(PlannerBudgetExceeded, match="token budget"):
        planner.propose(2, live_feedback(2))
    assert calls == [1]


def test_live_missing_usage_and_deadline_fail_closed(monkeypatch):
    import matmul_inspector.tensor_game_schedule_agent as agent
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-secret")
    monkeypatch.setattr(agent, "urlopen", lambda request, timeout: io.BytesIO(
        json.dumps({"status": "completed", "usage": None}).encode()))
    with pytest.raises(PlannerBudgetExceeded, match="valid token usage"):
        OpenAIPlanner("test-model").propose(1, live_feedback(1))
    def slow_urlopen(request, timeout):
        time.sleep(0.1)
        raise AssertionError("deadline failed")
    monkeypatch.setattr(agent, "urlopen", slow_urlopen)
    with pytest.raises(TimeoutError, match="deadline"):
        OpenAIPlanner("test-model", Limits(planner_timeout_seconds=0.01)).propose(
            1, live_feedback(1))


def test_live_credential_never_enters_artifacts_and_offline_replay(tmp_path, monkeypatch):
    import matmul_inspector.tensor_game_schedule_agent as agent
    secret = "test-only-\\secret-é-never-in-artifacts"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    def fake_urlopen(request, timeout):
        proposal = fallback_proposal(1)
        proposal["reason"] = secret
        data = {"status": "completed", "usage": {"input_tokens": 80,
                "output_tokens": 20, "total_tokens": 100},
                "output": [{"content": [{"type": "output_text", "text": json.dumps(proposal)}]}]}
        return io.BytesIO(json.dumps(data).encode())
    monkeypatch.setattr(agent, "urlopen", fake_urlopen)
    candidates, _ = load_retained_corpus(REPO)
    batch1 = select_batch1(candidates)
    bundle = runner.Bundle(tmp_path / "live-artifacts")
    proposal, source = runner.get_proposal(OpenAIPlanner("test-model"), 1, live_feedback(1),
                                           bundle, candidates, batch1)
    assert source == "DETERMINISTIC_FALLBACK_AFTER_PROVIDER_ERROR"
    assert proposal == fallback_proposal(1)
    assert all(secret not in path.read_text() for path in bundle.path.iterdir())
    saved = [fallback_proposal(1), fallback_proposal(2)]
    proposal_path = tmp_path / "saved.json"
    proposal_path.write_text(json.dumps(saved))
    monkeypatch.setattr(agent, "urlopen", lambda *args, **kwargs: pytest.fail("network in replay"))
    output = tmp_path / "offline-replay"
    assert runner.run(args(output, proposals=proposal_path))["classification"] == "B"
    assert json.loads((output / "resource_usage.json").read_text())["provider_api_calls"] == 0


def test_live_budget_exhaustion_is_terminal_before_gpu(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-secret")
    candidates, _ = load_retained_corpus(REPO)
    batch1 = select_batch1(candidates)
    bundle = runner.Bundle(tmp_path / "budget-artifacts")
    planner = OpenAIPlanner("test-model", Limits(max_planner_request_bytes=100))
    with pytest.raises(PlannerBudgetExceeded, match="request-size"):
        runner.get_proposal(planner, 1, live_feedback(1), bundle, candidates, batch1)
    error = json.loads((bundle.path / "batch1_provider_error.json").read_text())
    assert error["fallback_used"] is False
    assert not (bundle.path / "batch1_proposal.json").exists()


def test_live_provider_is_never_called_by_cpu_replay(tmp_path, monkeypatch):
    import matmul_inspector.tensor_game_schedule_agent as agent
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-secret")
    monkeypatch.setattr(agent, "urlopen", lambda *args, **kwargs: pytest.fail("network in replay"))
    request = args(tmp_path / "offline", gpu=False)
    request.provider = "openai"
    request.model = "test-model"
    result = runner.run(request)
    assert result["classification"] == "D"
    assert result["status"] == "FAILED_VALIDATION"


def test_proposal_preflight_validates_one_live_batch1_call_without_gpu(tmp_path, monkeypatch):
    import matmul_inspector.tensor_game_schedule_agent as agent
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-secret")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runner, "cpu_gate", lambda *args: pytest.fail("CPU gate in preflight"))
    monkeypatch.setattr(runner, "execute_gpu_batch", lambda *args: pytest.fail("GPU batch in preflight"))
    comparison = live_feedback(1)["frozen_batch1_hashes"][0]
    proposal = fallback_proposal(1)
    proposal["comparison_preferences"] = [comparison]
    calls = []
    def fake_urlopen(request, timeout):
        calls.append(json.loads(request.data))
        data = {"status": "completed", "usage": {"input_tokens": 80,
                "output_tokens": 20, "total_tokens": 100},
                "output": [{"content": [{"type": "output_text", "text": json.dumps(proposal)}]}]}
        return io.BytesIO(json.dumps(data).encode())
    monkeypatch.setattr(agent, "urlopen", fake_urlopen)
    result = runner.run_proposal_preflight("test-model")
    assert result == {"status": "VALID_PROPOSAL", "validation_passed": True,
                      "model": "test-model", "comparison_hashes": [comparison],
                      "invalid_comparison_entries": 0,
                      "token_usage": {"input_tokens": 80, "output_tokens": 20,
                                      "total_tokens": 100}, "api_calls": 1}
    assert len(calls) == 1
    assert json.loads(calls[0]["input"][0]["content"])["batch"] == 1
    assert calls[0]["text"]["format"]["schema"]["properties"][
        "requested_configurations"]["enum"] == [8]
    assert list(tmp_path.iterdir()) == []


def test_proposal_preflight_reports_requested_five_without_repair(monkeypatch):
    import matmul_inspector.tensor_game_schedule_agent as agent
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-secret")
    invalid = fallback_proposal(1)
    invalid["requested_configurations"] = 5
    calls = []
    def fake_urlopen(request, timeout):
        calls.append(json.loads(request.data))
        data = {"status": "completed", "usage": {"input_tokens": 80,
                "output_tokens": 20, "total_tokens": 100},
                "output": [{"content": [{"type": "output_text", "text": json.dumps(invalid)}]}]}
        return io.BytesIO(json.dumps(data).encode())
    monkeypatch.setattr(agent, "urlopen", fake_urlopen)
    result = runner.run_proposal_preflight("test-model")
    assert result["status"] == "INVALID_PROPOSAL"
    assert result["validation_passed"] is False
    assert result["validation_error"] == (
        "requested_configurations exceeds or changes the fixed batch budget")
    assert result["api_calls"] == len(calls) == 1
    assert calls[0]["text"]["format"]["schema"]["properties"][
        "requested_configurations"]["enum"] == [8]


def test_full_live_proposal_path_uses_fixed_batch_budgets_without_gpu(tmp_path, monkeypatch):
    import matmul_inspector.tensor_game_schedule_agent as agent
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-secret")
    requests = []
    def fake_urlopen(request, timeout):
        requests.append(json.loads(request.data))
        data = {"status": "completed", "usage": {"input_tokens": 80,
                "output_tokens": 20, "total_tokens": 100},
                "output": [{"content": [{"type": "output_text",
                            "text": json.dumps(fallback_proposal(len(requests)))}]}]}
        return io.BytesIO(json.dumps(data).encode())
    monkeypatch.setattr(agent, "urlopen", fake_urlopen)
    batches = []
    def fake_gpu(bundle, label, selected, deadline):
        batches.append((label, len(selected)))
        return {"correctness": [], "topology": [], "raw": [],
                "summaries": [], "observations": []}
    monkeypatch.setattr(runner, "execute_gpu_batch", fake_gpu)
    request = args(tmp_path / "mocked-live-run", gpu=True)
    request.provider = "openai"
    request.model = "test-model"
    result = runner.run(request)
    assert result["status"] == "COMPLETED_WITH_LIMITATIONS"
    assert batches == [("batch1", 8), ("batch2", 4)]
    assert len(requests) == 2
    for batch, fixed in ((1, 8), (2, 4)):
        fields = requests[batch - 1]["text"]["format"]["schema"]["properties"]
        assert fields["requested_configurations"]["enum"] == [fixed]
        bundle = request.output
        assert json.loads((bundle / f"batch{batch}_proposal.json").read_text())[
            "proposal"]["requested_configurations"] == fixed
        assert json.loads((bundle / f"batch{batch}_validation.json").read_text())["valid"]
    usage = json.loads((request.output / "resource_usage.json").read_text())
    assert usage["provider_api_calls"] == 2
    assert usage["provider_total_tokens"] == 200


def test_full_two_proposal_path_uses_retained_batch1_feedback_without_gpu(tmp_path, monkeypatch):
    source = REPO / "artifacts/schedule-agent/gpu-v1"
    saved_feedback = json.loads((source / "batch1_feedback.json").read_text())
    proposal1 = fallback_proposal(1)
    proposal1["reason"] = "r" * 400
    proposal2 = fallback_proposal(2)
    proposal2["hypothesis"] = "h" * 400
    proposal2["candidate_hashes"] = saved_feedback["eligible_batch2_hashes"][:4]
    proposal2["comparison_preferences"] = [
        saved_feedback["measurements"]["candidates"][0]["schedule_hash"],
        saved_feedback["eligible_batch2_hashes"][0]]
    proposal2["execution_mode"] = "CUDA_GRAPH"
    requests = mocked_openai_responses(monkeypatch, [proposal1, proposal2])
    batches = []
    def fake_gpu(bundle, label, selected, deadline):
        batches.append(label)
        if label == "batch2":
            return {"correctness": [], "topology": [], "raw": [],
                    "summaries": [], "observations": []}
        return {"correctness": runner.read_rows(source / "batch1_gpu_correctness.jsonl"),
                "topology": runner.read_rows(source / "batch1_gpu_topology.jsonl"),
                "raw": runner.read_rows(source / "batch1_raw_device_timing.jsonl"),
                "summaries": runner.read_rows(source / "batch1_timing_summary.jsonl"),
                "observations": runner.read_rows(source / "batch1_observations.jsonl")}
    monkeypatch.setattr(runner, "execute_gpu_batch", fake_gpu)
    request = args(tmp_path / "saved-batch1-mocked-provider", gpu=True)
    request.provider = "openai"
    request.model = "test-model"
    result = runner.run(request)
    assert result["status"] == "COMPLETED_WITH_LIMITATIONS"
    assert batches == ["batch1", "batch2"]
    generated_feedback = json.loads((request.output / "batch1_feedback.json").read_text())
    assert generated_feedback["measurements"] == saved_feedback["measurements"]
    assert generated_feedback["eligible_batch2_hashes"] == saved_feedback["eligible_batch2_hashes"]
    assert all(generated_feedback["interpretation"].get(key) == value for key, value in
               saved_feedback["interpretation"].items())
    assert all(json.loads((request.output / f"batch{batch}_validation.json").read_text())[
        "valid"] for batch in (1, 2))
    second_prompt = json.loads(requests[1]["input"][0]["content"])
    assert second_prompt["feedback"]["measurements"] == generated_feedback["measurements"]
    assert second_prompt["feedback"]["interpretation"] == generated_feedback["interpretation"]
    assert len(requests) == 2


def test_batch2_proposal_preflight_uses_saved_evidence_and_one_mock_call(tmp_path, monkeypatch):
    source = REPO / "artifacts/schedule-agent/gpu-v1"
    feedback = json.loads((source / "batch1_feedback.json").read_text())
    proposal = fallback_proposal(2)
    proposal["candidate_hashes"] = feedback["eligible_batch2_hashes"][:4]
    proposal["comparison_preferences"] = [
        feedback["measurements"]["candidates"][0]["schedule_hash"]]
    requests = mocked_openai_responses(monkeypatch, [proposal])
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runner, "cpu_gate", lambda *args: pytest.fail("CPU gate in preflight"))
    monkeypatch.setattr(runner, "execute_gpu_batch", lambda *args: pytest.fail("GPU in preflight"))
    result = runner.run_proposal_preflight("test-model", batch2_bundle=source)
    assert result["status"] == "VALID_PROPOSAL"
    assert result["validation_passed"] is True
    assert result["batch"] == 2
    assert result["candidate_hashes"] == proposal["candidate_hashes"]
    assert result["comparison_hashes"] == proposal["comparison_preferences"]
    assert result["api_calls"] == len(requests) == 1
    assert json.loads(requests[0]["input"][0]["content"])["batch"] == 2
    assert requests[0]["text"]["format"]["schema"]["properties"][
        "requested_configurations"]["enum"] == [4]
    assert list(tmp_path.iterdir()) == []


def test_batch2_preflight_rejects_tampered_feedback_before_api(tmp_path, monkeypatch):
    import shutil
    source = REPO / "artifacts/schedule-agent/gpu-v1"
    copied = tmp_path / "tampered"
    shutil.copytree(source, copied)
    (copied / "batch1_feedback.json").write_text("{}\n")
    requests = mocked_openai_responses(monkeypatch, [fallback_proposal(2)])
    result = runner.run_proposal_preflight("test-model", batch2_bundle=copied)
    assert result["status"] == "PROVIDER_ERROR"
    assert result["validation_passed"] is False
    assert result["api_calls"] == 0
    assert requests == []


def test_batch2_preflight_reports_overlong_hypothesis_without_repair(monkeypatch):
    source = REPO / "artifacts/schedule-agent/gpu-v1"
    proposal = fallback_proposal(2)
    proposal["hypothesis"] = "h" * 445
    requests = mocked_openai_responses(monkeypatch, [proposal])
    result = runner.run_proposal_preflight("test-model", batch2_bundle=source)
    assert result["status"] == "INVALID_PROPOSAL"
    assert result["validation_passed"] is False
    assert result["validation_error"] == (
        "hypothesis must be a nonempty string of at most 400 characters")
    assert result["api_calls"] == len(requests) == 1
    assert result["candidate_hashes"] == []


def test_proposal_preflight_reports_invalid_runtime_without_silent_repair(monkeypatch):
    import matmul_inspector.tensor_game_schedule_agent as agent
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-secret")
    proposal = fallback_proposal(1)
    proposal["comparison_preferences"] = ["runtime", "memory_usage"]
    calls = []
    def fake_urlopen(request, timeout):
        calls.append(1)
        data = {"status": "completed", "usage": {"input_tokens": 80,
                "output_tokens": 20, "total_tokens": 100},
                "output": [{"content": [{"type": "output_text", "text": json.dumps(proposal)}]}]}
        return io.BytesIO(json.dumps(data).encode())
    monkeypatch.setattr(agent, "urlopen", fake_urlopen)
    result = runner.run_proposal_preflight("test-model")
    assert result["status"] == "INVALID_PROPOSAL"
    assert result["validation_passed"] is False
    assert result["comparison_hashes"] == []
    assert result["invalid_comparison_entries"] == 2
    assert result["validation_error"] == "comparison_preferences contains an ineligible hash"
    assert result["api_calls"] == len(calls) == 1


def test_proposal_preflight_does_not_print_echoed_credential(monkeypatch):
    import matmul_inspector.tensor_game_schedule_agent as agent
    secret = "test-only-secret-never-print"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    proposal = fallback_proposal(1)
    proposal["reason"] = secret
    def fake_urlopen(request, timeout):
        data = {"status": "completed", "usage": {"input_tokens": 80,
                "output_tokens": 20, "total_tokens": 100},
                "output": [{"content": [{"type": "output_text", "text": json.dumps(proposal)}]}]}
        return io.BytesIO(json.dumps(data).encode())
    monkeypatch.setattr(agent, "urlopen", fake_urlopen)
    result = runner.run_proposal_preflight("test-model")
    assert result["status"] == "INVALID_PROPOSAL"
    assert result["validation_passed"] is False
    assert secret not in json.dumps(result)


def test_proposal_preflight_reports_usage_after_incomplete_response(monkeypatch):
    import matmul_inspector.tensor_game_schedule_agent as agent
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-secret")
    calls = []
    def fake_urlopen(request, timeout):
        calls.append(1)
        return io.BytesIO(json.dumps({"status": "incomplete", "usage": {
            "input_tokens": 80, "output_tokens": 20, "total_tokens": 100}}).encode())
    monkeypatch.setattr(agent, "urlopen", fake_urlopen)
    result = runner.run_proposal_preflight("test-model")
    assert result["status"] == "PROVIDER_ERROR"
    assert result["validation_passed"] is False
    assert result["api_calls"] == len(calls) == 1
    assert result["token_usage"] == {"input_tokens": 80, "output_tokens": 20,
                                     "total_tokens": 100}


def test_proposal_preflight_cli_requires_exclusive_mode_without_output(monkeypatch, capsys):
    monkeypatch.setattr(runner, "run_proposal_preflight", lambda model, **kwargs: {
        "status": "VALID_PROPOSAL", "validation_passed": True, "model": model,
        "batch": 2 if kwargs else 1,
        "comparison_hashes": [], "token_usage": None, "api_calls": 1})
    monkeypatch.setattr("sys.argv", ["schedule-agent", "--proposal-preflight",
                                   "--provider", "openai", "--model", "test-model"])
    runner.main()
    assert json.loads(capsys.readouterr().out)["validation_passed"] is True
    monkeypatch.setattr("sys.argv", ["schedule-agent", "--proposal-preflight",
                                   "--provider", "openai", "--model", "test-model",
                                   "--batch2-evidence", "saved-batch1"])
    runner.main()
    assert json.loads(capsys.readouterr().out)["batch"] == 2
    monkeypatch.setattr("sys.argv", ["schedule-agent", "--proposal-preflight",
                                   "--provider", "openai", "--model", "test-model", "--gpu"])
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2
    monkeypatch.setattr("sys.argv", ["schedule-agent", "--batch2-evidence", "saved-batch1",
                                   "--output", "unused", "--gpu"])
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2
    monkeypatch.setattr(runner, "run_gpu_worker", lambda *args: pytest.fail("worker in preflight"))
    monkeypatch.setattr("sys.argv", ["schedule-agent", "--proposal-preflight",
                                   "--provider", "openai", "--model", "test-model",
                                   "--worker-batch", "batch1", "--output", "unused"])
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2


def test_mid_run_failure_preserves_completed_evidence(tmp_path, monkeypatch):
    calls = []
    observations = json.loads((HA10 / "calibration_after.json").read_text())["observations"]
    def worker(bundle, label, selected, deadline):
        calls.append(label)
        if label == "batch2":
            raise TimeoutError("simulated batch 2 deadline")
        bundle.write("batch1_raw_device_timing.jsonl", [{"sample": 1}], rows=True)
        return {"correctness": [], "topology": [], "raw": [{"sample": 1}],
                "summaries": [], "observations": observations}
    monkeypatch.setattr(runner, "execute_gpu_batch", worker)
    output = tmp_path / "run"
    result = runner.run(args(output, gpu=True))
    assert calls == ["batch1", "batch2"]
    assert result["status"] == "BUDGET_EXHAUSTED"
    assert (output / "batch1_raw_device_timing.jsonl").exists()
    assert (output / "batch1_feedback.json").exists()
    assert (output / "batch2_controls_and_guidance.json").exists()
    manifest = json.loads((output / "hash_manifest.json").read_text())
    assert manifest["batch1_raw_device_timing.jsonl"] == runner.digest(output / "batch1_raw_device_timing.jsonl")


def test_feedback_separates_measurements_and_interpretation(corpus):
    candidates, batch1 = corpus
    feedback = feedback_summary(batch1, [], [], [], candidates)
    assert feedback["measurements"]["timings"] == []
    assert feedback["interpretation"]["indeterminate_pairs"] == 0
    assert feedback["remaining_candidate_budget"] == 4
    assert all(item not in {row["schedule_hash"] for row in batch1}
               for item in feedback["eligible_batch2_hashes"])
