"""Bounded proposal validation and feedback for the HA-10 schedule controller.

Planner output is advisory. Candidate identities, observations, and budgets are
checked here and by the existing deterministic selector before GPU execution.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Protocol
from urllib.request import Request, urlopen

from .tensor_game_schedule_reselection import MODES, select_batch2, validate_observations

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


LIMITS = Limits()
FIELDS = frozenset(("proposal_id", "hypothesis", "experiment_family", "descriptors",
                    "execution_mode", "candidate_hashes", "comparison_preferences",
                    "requested_configurations", "reason"))


class Planner(Protocol):
    source: str

    def propose(self, batch: int, feedback: dict) -> dict: ...


class ProposalError(ValueError):
    pass


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


class OpenAIPlanner:
    """Optional Responses API adapter; no SDK or credential is needed for replay."""

    source = "LIVE_OPENAI_RESPONSES"

    def __init__(self, model: str):
        if not model or not os.environ.get("OPENAI_API_KEY"):
            raise ProposalError("model and OPENAI_API_KEY are required for live planning")
        self.model = model

    def propose(self, batch: int, feedback: dict) -> dict:
        # The response format reduces malformed output; validate_proposal is authoritative.
        string = {"type": "string"}
        schema = {"type": "object", "additionalProperties": False,
                  "required": sorted(FIELDS), "properties": {
                      "proposal_id": string, "hypothesis": string, "experiment_family": string,
                      "descriptors": {"type": "array", "items": string},
                      "execution_mode": string,
                      "candidate_hashes": {"type": "array", "items": string},
                      "comparison_preferences": {"type": "array", "items": string},
                      "requested_configurations": {"type": "integer"}, "reason": string}}
        prompt = {"batch": batch, "limits": LIMITS.__dict__, "feedback": feedback,
                  "instruction": "Propose one bounded schedule experiment. Use only listed candidate hashes. "
                                 "Batch 1 candidate_hashes must be empty. Never invent measurements.",
                  "allowed_families": FAMILIES, "allowed_descriptors": DESCRIPTORS,
                  "allowed_modes": MODES}
        body = {"model": self.model, "input": [{"role": "user", "content": json.dumps(prompt)}],
                "max_output_tokens": 650,
                "text": {"format": {"type": "json_schema", "name": "schedule_proposal",
                                    "strict": True, "schema": schema}}}
        request = Request("https://api.openai.com/v1/responses", data=json.dumps(body).encode(),
                          headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
                                   "Content-Type": "application/json"}, method="POST")
        with urlopen(request, timeout=LIMITS.planner_timeout_seconds) as response:
            data = json.load(response)
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
        if schedule_hash not in known or known[schedule_hash]["matrix_size"] != limits.matrix_size:
            raise ProposalError(f"comparison_preferences contains an unknown N=256 hash: {schedule_hash}")
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
    return {"measurements": {"candidates": measured, "timings": evidence,
                             "correctness": correctness, "pairs": pairs},
            "interpretation": {"directional_pairs": sum(p["preferred"] is not None for p in pairs),
                               "indeterminate_pairs": sum(p["preferred"] is None for p in pairs),
                               "failed_hashes": failed,
                               "uncertainty": "All-same-sign round screen; no confidence interval or speedup claim"},
            "eligible_batch2_hashes": sorted(row["schedule_hash"] for row in candidates
                                              if row["matrix_size"] == 256 and
                                              row["retained_correctness"] == "PASS" and
                                              row["schedule_hash"] not in {b["schedule_hash"] for b in batch1}),
            "remaining_candidate_budget": LIMITS.batch2}
