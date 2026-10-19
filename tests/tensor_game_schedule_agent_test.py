"""CPU-only validation of the bounded controller and saved HA-10 replay."""

from __future__ import annotations

from argparse import Namespace
from copy import deepcopy
import io
import json
from pathlib import Path

import pytest

from matmul_inspector.tensor_game_schedule_agent import (
    LIMITS, FallbackPlanner, OpenAIPlanner, ProposalError, ReplayPlanner,
    fallback_proposal, feedback_summary, selections, validate_proposal,
)
from matmul_inspector.tensor_game_schedule_reselection import (
    load_retained_corpus, make_observation, select_batch1,
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
    response = {"id": "resp_test", "status": "completed", "usage": {"output_tokens": 20},
                "output": [{"content": [{"type": "output_text", "text": json.dumps(proposal)}]}]}
    class Response(io.BytesIO):
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.close()
    calls = []
    def fake_urlopen(request, timeout):
        calls.append((request.full_url, timeout))
        return Response(json.dumps(response).encode())
    monkeypatch.setattr(agent, "urlopen", fake_urlopen)
    planner = OpenAIPlanner("test-model")
    assert planner.propose(1, {}) == proposal
    assert planner.last_metadata["response_id"] == "resp_test"
    assert calls == [("https://api.openai.com/v1/responses", LIMITS.planner_timeout_seconds)]


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
