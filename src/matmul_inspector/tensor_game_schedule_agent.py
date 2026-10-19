"""Bounded proposal validation and feedback for the HA-10 schedule controller.

Planner output is advisory. Candidate identities, observations, and budgets are
checked here and by the existing deterministic selector before GPU execution.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
import signal
import time
from typing import Protocol
from urllib.request import Request, urlopen

from .tensor_game_schedule_reselection import (
    MODES, evidence_votes, select_batch2, validate_observations,
)

FAMILIES = ("STRUCTURAL_DIVERSITY", "RANK8_FORM_TRADEOFF", "MATERIALIZATION_TOPOLOGY")
DESCRIPTORS = ("stored_forms", "recomputed_forms", "hbm_products", "hbm_partials",
               "partial_updates", "reconstruction_in_gemm_group", "launch_groups",
               "max_group_operations")


@dataclass(frozen=True)
class Limits:
    max_batches: int = 2
    batch1: int = 8
    batch2: int = 4
    matrix_size: int = 256
    max_gpu_candidates: int = 12
    max_profiler_invocations: int = 12
    max_timing_samples: int = 600
    max_rounds: int = 5
    max_iterations_per_round: int = 5
    max_wall_seconds: int = 300
    max_batch_seconds: int = 120
    max_artifact_bytes: int = 2_000_000
    max_planner_calls: int = 2
    planner_timeout_seconds: int = 15
    max_planner_request_bytes: int = 32_000
    max_planner_response_bytes: int = 16_000
    max_planner_output_tokens_per_call: int = 650
    max_planner_total_tokens: int = 66_000


LIMITS = Limits()
FIELDS = frozenset(("proposal_id", "hypothesis", "experiment_family", "descriptors",
                    "execution_mode", "candidate_hashes", "comparison_preferences",
                    "requested_configurations", "reason"))


class Planner(Protocol):
    source: str

    def propose(self, batch: int, feedback: dict) -> dict: ...


class ProposalError(ValueError):
    pass


class PlannerBudgetExceeded(RuntimeError):
    """A live planning limit was reached; the controller must stop."""


def fallback_proposal(batch: int) -> dict:
    return {"proposal_id": f"fallback-batch-{batch}",
            "hypothesis": "Measured structural preferences may change the next selection",
            "experiment_family": "STRUCTURAL_DIVERSITY", "descriptors": ["launch_groups"],
            "execution_mode": "EAGER", "candidate_hashes": [],
            "comparison_preferences": [],
            "requested_configurations": LIMITS.batch1 if batch == 1 else LIMITS.batch2,
            "reason": "Use the frozen Batch 1 and the deterministic calibrated selector"}


class FallbackPlanner:
    source = "DETERMINISTIC_FALLBACK"

    def propose(self, batch: int, feedback: dict) -> dict:
        return fallback_proposal(batch)


class ReplayPlanner:
    source = "SAVED_PROPOSAL_REPLAY"

    def __init__(self, proposals: list[dict]):
        self.proposals = proposals

    def propose(self, batch: int, feedback: dict) -> dict:
        if len(self.proposals) < batch:
            raise ProposalError(f"missing saved proposal for Batch {batch}")
        return self.proposals[batch - 1]


_SCHEDULE_HASH = re.compile(r"[0-9a-f]{64}\Z")


def live_hash_allowlists(batch: int, feedback: dict, limits: Limits = LIMITS) -> tuple[list[str], list[str]]:
    """Derive schema choices from the controller's frozen and eligible records."""
    if batch == 1:
        comparison = feedback.get("frozen_batch1_hashes")
        scope = []
        if not isinstance(comparison, list) or len(comparison) != limits.batch1:
            raise ProposalError("live Batch 1 requires the frozen schedule hashes")
    elif batch == 2:
        scope = feedback.get("eligible_batch2_hashes")
        measured = feedback.get("measurements", {}).get("candidates", [])
        if not isinstance(scope, list) or len(scope) < limits.batch2 or not isinstance(measured, list):
            raise ProposalError("live Batch 2 requires eligible schedule hashes")
        comparison = [row.get("schedule_hash") for row in measured if isinstance(row, dict)] + scope
        if len(measured) != limits.batch1 or len(comparison) != len(measured) + len(scope):
            raise ProposalError("live Batch 2 requires measured Batch 1 hashes")
    else:
        raise ProposalError("live proposal batch is outside the fixed experiment")
    for hashes in (scope, comparison):
        if (any(not isinstance(value, str) or _SCHEDULE_HASH.fullmatch(value) is None
                for value in hashes) or len(set(hashes)) != len(hashes)):
            raise ProposalError("live proposal hash allowlist is invalid")
    return sorted(scope), sorted(comparison)


class OpenAIPlanner:
    """Optional Responses API adapter; no SDK or credential is needed for replay."""

    source = "LIVE_OPENAI_RESPONSES"

    def __init__(self, model: str, limits: Limits = LIMITS, *, first_batch: int = 1):
        if not model or not os.environ.get("OPENAI_API_KEY"):
            raise ProposalError("model and OPENAI_API_KEY are required for live planning")
        if first_batch not in (1, 2):
            raise ProposalError("live proposal batch is outside the fixed experiment")
        self.model = model
        self.limits = limits
        self.first_batch = first_batch
        self.calls = 0
        self.total_tokens = 0
        self.last_metadata: dict = {}

    def propose(self, batch: int, feedback: dict) -> dict:
        if self.calls >= self.limits.max_planner_calls:
            raise PlannerBudgetExceeded("planner API-call budget exhausted")
        if batch != self.first_batch + self.calls or batch > self.limits.max_batches:
            raise ProposalError("live proposals must follow the two-batch order")
        scope_hashes, comparison_hashes = live_hash_allowlists(batch, feedback, self.limits)
        fixed_configurations = self.limits.batch1 if batch == 1 else self.limits.batch2
        # The response format reduces malformed output; validate_proposal is authoritative.
        # A supported pattern bounds all three free-text fields to 400 characters
        # and requires non-whitespace content. The validator remains authoritative.
        short_text = {"type": "string", "pattern": r"^\S[\s\S]{0,399}$",
                      "description": "Start with a non-whitespace character; at most 400 characters"}
        string = {"type": "string"}
        eligible_hash = {"type": "string", "enum": scope_hashes}
        # Reuse the enum across both length branches to stay under the fixed
        # request-size budget. Definitions and references are supported here.
        scope_item = {"$ref": "#/$defs/eligible_hash"}
        batch2_scope = {"anyOf": [
            {"type": "array", "items": scope_item, "maxItems": 0},
            {"type": "array", "items": scope_item,
             "minItems": self.limits.batch2, "maxItems": 16}]}
        schema = {"type": "object", "additionalProperties": False,
                  "required": sorted(FIELDS), "properties": {
                      "proposal_id": short_text, "hypothesis": short_text,
                      "experiment_family": {"type": "string", "enum": FAMILIES},
                      "descriptors": {"type": "array", "items":
                          {"type": "string", "enum": DESCRIPTORS},
                          "maxItems": len(DESCRIPTORS)},
                      "execution_mode": {"type": "string", "enum": MODES},
                      "candidate_hashes": (batch2_scope if batch == 2 else
                          {"type": "array", "items": string, "maxItems": 0,
                           "description": "Empty because Batch 1 is frozen"}),
                      "comparison_preferences": {"type": "array", "items":
                          {"type": "string", "enum": comparison_hashes}, "maxItems": 4,
                          "description": "Schedule hashes to compare, never metric or descriptor names"},
                      "requested_configurations": {"type": "integer",
                          "enum": [fixed_configurations],
                          "description": "Fixed by the controller; not a model-selected budget"},
                      "reason": short_text}}
        if batch == 2:
            schema["$defs"] = {"eligible_hash": eligible_hash}
        # The Batch 2 eligible list is already present in the schema enum.
        # Keep the full controller feedback artifact while avoiding a duplicate prompt list.
        prompt_feedback = ({key: value for key, value in feedback.items()
                            if key != "eligible_batch2_hashes"} if batch == 2 else feedback)
        prompt = {"batch": batch, "limits": self.limits.__dict__, "feedback": prompt_feedback,
                  "instruction": "Propose one bounded schedule experiment. candidate_hashes and "
                                 "comparison_preferences contain only full schedule hashes allowed by "
                                 "their schema enums. comparison_preferences names schedules, not "
                                 "metrics such as runtime or memory_usage; use [] if none. Batch 1 "
                                 "candidate_hashes must be empty. requested_configurations is "
                                 "fixed by the controller at the schema's only allowed value. "
                                 "For Batch 2 candidate_hashes, choose [] or at least four "
                                 "distinct eligible hashes. Keep proposal_id, hypothesis, and "
                                 "reason within 400 characters with no leading whitespace. "
                                 "Do not repeat descriptors or hash choices. Never invent measurements.",
                  "allowed_families": FAMILIES, "allowed_descriptors": DESCRIPTORS,
                  "allowed_modes": MODES}
        body = {"model": self.model, "input": [{"role": "user", "content":
                json.dumps(prompt, sort_keys=True, separators=(",", ":"))}],
                "max_output_tokens": self.limits.max_planner_output_tokens_per_call,
                "store": False,
                "text": {"format": {"type": "json_schema", "name": "schedule_proposal",
                                    "strict": True, "schema": schema}}}
        payload = json.dumps(body, separators=(",", ":")).encode()
        if len(payload) > self.limits.max_planner_request_bytes:
            raise PlannerBudgetExceeded("planner request-size budget exhausted")
        # Reserve the full request-byte and output-token allowances before a call.
        # Actual server-side token counts are checked from the returned usage.
        reserved = (self.limits.max_planner_request_bytes +
                    self.limits.max_planner_output_tokens_per_call)
        if self.total_tokens + reserved > self.limits.max_planner_total_tokens:
            raise PlannerBudgetExceeded("planner token budget exhausted")
        request = Request("https://api.openai.com/v1/responses", data=payload,
                          headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
                                   "Content-Type": "application/json"}, method="POST")
        if signal.getsignal(signal.SIGALRM) != signal.SIG_DFL:
            raise ProposalError("live planner requires an unused process alarm")
        self.calls += 1  # Failed or timed-out requests still spend an API-call attempt.
        started = time.monotonic()
        def deadline_reached(signum, frame):
            raise TimeoutError("planner response deadline exceeded")
        previous_handler = signal.signal(signal.SIGALRM, deadline_reached)
        try:
            signal.setitimer(signal.ITIMER_REAL, self.limits.planner_timeout_seconds)
            with urlopen(request, timeout=self.limits.planner_timeout_seconds) as response:
                raw = response.read(self.limits.max_planner_response_bytes + 1)
            if len(raw) > self.limits.max_planner_response_bytes:
                raise PlannerBudgetExceeded("planner response-size budget exhausted")
            data = json.loads(raw)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)
        if time.monotonic() - started > self.limits.planner_timeout_seconds:
            raise TimeoutError("planner response deadline exceeded")
        usage = data.get("usage")
        if not isinstance(usage, dict) or any(type(usage.get(key)) is not int or usage[key] < 0
                for key in ("input_tokens", "output_tokens", "total_tokens")) or (
                usage["input_tokens"] + usage["output_tokens"] != usage["total_tokens"]):
            raise PlannerBudgetExceeded("provider did not report valid token usage")
        self.total_tokens += usage["total_tokens"]
        self.last_metadata = {"response_id": data.get("id"), "status": data.get("status"),
                              "usage": usage, "cumulative_tokens": self.total_tokens,
                              "api_calls": self.calls}
        if (usage["output_tokens"] > self.limits.max_planner_output_tokens_per_call or
                usage["input_tokens"] > self.limits.max_planner_request_bytes or
                self.total_tokens > self.limits.max_planner_total_tokens):
            raise PlannerBudgetExceeded("planner token budget exhausted")
        if data.get("status") != "completed":
            raise ProposalError(f"provider returned status {data.get('status')}")
        parts = [part["text"] for item in data.get("output", [])
                 for part in item.get("content", []) if part.get("type") == "output_text"]
        if len(parts) != 1:
            raise ProposalError("provider returned no single structured proposal")
        return json.loads(parts[0])


def validate_proposal(raw: dict, *, batch: int, candidates: list[dict],
                      batch1: list[dict], limits: Limits = LIMITS) -> dict:
    if not isinstance(raw, dict) or set(raw) != FIELDS:
        raise ProposalError("proposal must contain exactly the approved fields")
    for key in ("proposal_id", "hypothesis", "reason"):
        if not isinstance(raw[key], str) or not 1 <= len(raw[key].strip()) <= 400:
            raise ProposalError(f"{key} must be a nonempty string of at most 400 characters")
    if raw["experiment_family"] not in FAMILIES:
        raise ProposalError("experiment_family is not approved")
    if raw["execution_mode"] not in MODES:
        raise ProposalError("execution_mode is not approved")
    for key, allowed, maximum in (("descriptors", DESCRIPTORS, len(DESCRIPTORS)),
                                  ("comparison_preferences", None, 4),
                                  ("candidate_hashes", None, 16)):
        value = raw[key]
        if not isinstance(value, list) or len(value) > maximum or len(set(map(str, value))) != len(value):
            raise ProposalError(f"{key} must be a unique bounded list")
        if any(not isinstance(item, str) for item in value):
            raise ProposalError(f"{key} must contain strings")
        if allowed is not None and any(item not in allowed for item in value):
            raise ProposalError(f"{key} contains an unapproved descriptor")
    if batch not in (1, 2) or type(raw["requested_configurations"]) is not int or (
            raw["requested_configurations"] != (limits.batch1 if batch == 1 else limits.batch2)):
        raise ProposalError("requested_configurations exceeds or changes the fixed batch budget")
    known = {row["schedule_hash"]: row for row in candidates}
    selected = {row["schedule_hash"] for row in batch1}
    if batch == 1 and raw["candidate_hashes"]:
        raise ProposalError("Batch 1 is frozen; candidate_hashes must be empty")
    for schedule_hash in raw["candidate_hashes"]:
        row = known.get(schedule_hash)
        if row is None or row["matrix_size"] != limits.matrix_size or (
                row["retained_correctness"] != "PASS" or schedule_hash in selected):
            raise ProposalError(f"candidate_hashes contains an ineligible hash: {schedule_hash}")
    if batch == 2 and raw["candidate_hashes"] and len(raw["candidate_hashes"]) < limits.batch2:
        raise ProposalError("Batch 2 scope must contain at least four eligible candidates")
    for schedule_hash in raw["comparison_preferences"]:
        row = known.get(schedule_hash)
        if row is None or row["matrix_size"] != limits.matrix_size or (
                row["retained_correctness"] != "PASS" or
                (batch == 1 and schedule_hash not in selected)):
            raise ProposalError(f"comparison_preferences contains an ineligible hash: {schedule_hash}")
    return json.loads(json.dumps(raw, sort_keys=True))


def selections(candidates: list[dict], batch1: list[dict], observations: list[dict],
               proposal: dict) -> dict:
    for mode in MODES:
        validate_observations(observations, candidates, mode)
    frozen = {mode: select_batch2(candidates, batch1, [], mode=mode) for mode in MODES}
    calibrated = {mode: select_batch2(candidates, batch1, observations, mode=mode) for mode in MODES}
    scope = set(proposal["candidate_hashes"])
    scoped_candidates = ([row for row in candidates if row["schedule_hash"] in scope or
                          row["schedule_hash"] in {b["schedule_hash"] for b in batch1}]
                         if scope else candidates)
    guided = {mode: select_batch2(scoped_candidates, batch1, observations, mode=mode) for mode in MODES}
    mode = proposal["execution_mode"]
    return {"frozen": frozen, "calibrated": calibrated, "guided": guided,
            "physical_mode": mode, "physical_batch2": guided[mode],
            "scope_hashes": sorted(scope),
            "agent_changed_mode": mode != "EAGER",
            "agent_changed_scope": any(
                {x["schedule_hash"] for x in guided[m]} !=
                {x["schedule_hash"] for x in calibrated[m]} for m in MODES)}


def feedback_summary(batch1: list[dict], correctness: list[dict], summaries: list[dict],
                     observations: list[dict], candidates: list[dict]) -> dict:
    """Small measured record for a planner; interpretations are separate."""
    measured = [{"schedule_hash": row["schedule_hash"], "algorithm": row["algorithm"],
                 "descriptors": row["descriptors"]} for row in batch1]
    evidence = [{"schedule_hash": row["variant"], "mode": row["execution_mode"],
                 "median_ms": row.get("median_ms"),
                 "coefficient_of_variation": row.get("coefficient_of_variation"),
                 "round_medians_ms": row.get("round_medians_ms")}
                for row in summaries]
    pairs = [{"observation_id": row["observation_id"], "mode": row["mode"],
              "left": row["left"], "right": row["right"], "preferred": row["preferred"],
              "confidence": row["confidence"], "delta_range_ms": row["delta_range_ms"]}
             for row in observations]
    failed = sorted({row["schedule_hash"] for row in correctness if not row["passed"]})
    pair_preferences: dict[tuple, set[str]] = {}
    for row in observations:
        if row["preferred"] is not None:
            key = (row["mode"], *sorted((row["left"], row["right"])))
            pair_preferences.setdefault(key, set()).add(row["preferred"])
    conflicted_tokens = {mode: sorted(token for token, vote in
        evidence_votes(observations, candidates, mode).items() if vote["conflicted"])
        for mode in MODES}
    return {"measurements": {"candidates": measured, "timings": evidence,
                             "correctness": correctness, "pairs": pairs},
            "interpretation": {"directional_pairs": sum(p["preferred"] is not None for p in pairs),
                               "indeterminate_pairs": sum(p["preferred"] is None for p in pairs),
                               "contradictory_pairs": sum(len(values) > 1 for values in pair_preferences.values()),
                               "conflicted_descriptor_tokens": conflicted_tokens,
                               "failed_hashes": failed,
                               "uncertainty": "All-same-sign round screen; no confidence interval or speedup claim"},
            "eligible_batch2_hashes": sorted(row["schedule_hash"] for row in candidates
                                              if row["matrix_size"] == 256 and
                                              row["retained_correctness"] == "PASS" and
                                              row["schedule_hash"] not in {b["schedule_hash"] for b in batch1}),
            "remaining_candidate_budget": LIMITS.batch2}
