"""CPU-only, matched Batch 2 decision inputs and conservative analysis."""

from __future__ import annotations

import json
import re

from .tensor_game_schedule_agent import DESCRIPTORS, LIMITS, selections, validate_proposal
from .tensor_game_schedule_reselection import MODES

CONDITIONS = ("control", "treatment")
SUBSTANTIVE_FIELDS = ("experiment_family", "descriptors", "execution_mode",
                      "candidate_hashes", "comparison_preferences")
_MS = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*ms\b", re.I)
_HASH = re.compile(r"(?<![0-9a-f])([0-9a-f]{8,64})(?![0-9a-f])", re.I)
_UNCERTAINTY = re.compile(r"\b(?:unstable|stability|variable|variability|variance|"
                          r"uncertain|uncertainty|noisy|noise|indeterminate|conflicted|"
                          r"coefficient of variation|cv)\b", re.I)


def static_catalog(candidates: list[dict], batch1: list[dict]) -> list[dict]:
    """Verify labels and identities directly from the retained candidate corpus."""
    frozen = {row["schedule_hash"] for row in batch1}
    rows = [row for row in candidates if row["matrix_size"] == LIMITS.matrix_size
            and row["retained_correctness"] == "PASS"]
    if len(frozen) != LIMITS.batch1 or not frozen <= {row["schedule_hash"] for row in rows}:
        raise ValueError("frozen Batch 1 identities differ from retained corpus")
    for row in rows:
        if row["candidate_id"] != f'{row["algorithm"]}:{LIMITS.matrix_size}:{row["schedule_hash"]}':
            raise ValueError("candidate family or hash conflicts with retained identity")
    catalog = [{"schedule_hash": row["schedule_hash"], "algorithm": row["algorithm"],
                "descriptors": {key: row["descriptors"][key] for key in DESCRIPTORS},
                "frozen_batch1": row["schedule_hash"] in frozen}
               for row in rows]
    return sorted(catalog, key=lambda row: row["schedule_hash"])


def condition_feedback(verified: dict, catalog: list[dict], condition: str) -> dict:
    """Build identical decision context, changing only the evidence slot."""
    if condition not in CONDITIONS:
        raise ValueError("unknown ablation condition")
    frozen = [row for row in catalog if row["frozen_batch1"]]
    eligible = [row["schedule_hash"] for row in catalog if not row["frozen_batch1"]]
    if (len(frozen) != LIMITS.batch1 or
            sorted(verified["eligible_batch2_hashes"]) != eligible or
            {row["schedule_hash"] for row in verified["measurements"]["candidates"]} !=
            {row["schedule_hash"] for row in frozen}):
        raise ValueError("verified feedback differs from static candidate catalog")
    for row in verified["measurements"]["candidates"]:
        static = next(item for item in frozen if item["schedule_hash"] == row["schedule_hash"])
        if row["algorithm"] != static["algorithm"] or row["descriptors"] != static["descriptors"]:
            raise ValueError("Batch 1 algorithm or descriptor identity differs from corpus")
    evidence = {"status": "WITHHELD"}
    if condition == "treatment":
        frozen_index = {row["schedule_hash"]: index for index, row in enumerate(frozen)}
        evidence = {"status": "AVAILABLE",
                    "timings": [{"hash": row["schedule_hash"], "mode": row["mode"],
                                 "median_ms": row["median_ms"],
                                 "cv": row["coefficient_of_variation"],
                                 "round_medians_ms": row["round_medians_ms"]}
                                for row in verified["measurements"]["timings"]],
                    "pair_index_basis": "indices into measurements.candidates",
                    "pairs": [{"mode": row["mode"],
                               "left_index": frozen_index[row["left"]],
                               "right_index": frozen_index[row["right"]],
                               "preferred_index": (None if row["preferred"] is None else
                                                   frozen_index[row["preferred"]]),
                               "confidence": row["confidence"],
                               "delta_range_ms": row["delta_range_ms"]}
                              for row in verified["measurements"]["pairs"]],
                    "correctness_passed": not any(
                        not row["passed"] for row in verified["measurements"]["correctness"]),
                    "interpretation": verified["interpretation"]}
    return {"eligible_batch2_hashes": eligible,
            # This static list is required for the existing hash-enum schema.
            "measurements": {"candidates": [
                {"schedule_hash": row["schedule_hash"]} for row in frozen]},
            "candidate_catalog": {
                "descriptor_order": list(DESCRIPTORS),
                "schedules": [{"hash": row["schedule_hash"],
                               "family": row["algorithm"],
                               "d": [row["descriptors"][key] for key in DESCRIPTORS],
                               "batch1": row["frozen_batch1"]} for row in catalog]},
            "remaining_candidate_budget": LIMITS.batch2,
            "research_task": (
                "Choose one Batch 2 schedule experiment at N=256. The catalog is the "
                "authority for algorithm family and descriptors; each d array follows "
                "descriptor_order. Cite a schedule hash and "
                "execution mode for any specific timing claim. Treat uncertain or unstable "
                "timings as uncertain. If measurements are withheld, do not claim to have "
                "observed them. The controller decides legality, ranking, and resources."),
            "batch1_evidence": evidence}


def common_context(feedback: dict) -> dict:
    return {key: value for key, value in feedback.items() if key != "batch1_evidence"}


def assess_proposal(proposal: dict, candidates: list[dict], batch1: list[dict],
                    observations: list[dict], verified: dict) -> dict:
    """Record checkable facts; leave free-text causal interpretation for review."""
    validate_proposal(proposal, batch=2, candidates=candidates, batch1=batch1)
    selection = selections(candidates, batch1, observations, proposal)
    known = {row["schedule_hash"]: row for row in candidates}
    chosen = [row["schedule_hash"] for row in selection["physical_batch2"]]
    scope = proposal["candidate_hashes"] or chosen
    rationale = proposal["hypothesis"] + " " + proposal["reason"]
    mentioned = sorted({hash_value for hash_value in known if
                        any(hash_value.startswith(hit.group(1).lower()) for hit in _HASH.finditer(rationale))})
    unknown_prefixes = sorted({hit.group(1).lower() for hit in _HASH.finditer(rationale)
                               if not any(value.startswith(hit.group(1).lower()) for value in known)})
    actual_ms = [row["median_ms"] for row in verified["measurements"]["timings"]]
    claims = [float(hit.group(1)) for hit in _MS.finditer(rationale)]
    matches = [value for value in claims if any(abs(value - actual) <= 0.001
               for actual in actual_ms)]
    timing_by_key = {(row["schedule_hash"], row["mode"]): row["median_ms"]
                     for row in verified["measurements"]["timings"]}
    citations = []
    for clause in re.split(r"(?<=[.!?])\s+|[;\n]", rationale):
        cited_hashes = {value for value in known if any(value.startswith(hit.group(1).lower())
                        for hit in _HASH.finditer(clause))}
        cited_modes = {mode for mode in MODES if mode.replace("_", " ").lower() in
                       clause.replace("_", " ").lower()}
        cited_ms = [float(hit.group(1)) for hit in _MS.finditer(clause)]
        if len(cited_hashes) == len(cited_modes) == len(cited_ms) == 1:
            hash_value, cited_mode, value = next(iter(cited_hashes)), next(iter(cited_modes)), cited_ms[0]
            actual = timing_by_key.get((hash_value, cited_mode))
            citations.append({"schedule_hash": hash_value, "mode": cited_mode,
                              "claimed_ms": value, "saved_batch1_median_ms": actual,
                              "matches_saved_median": actual is not None and
                                  abs(value - actual) <= 0.001})
    family_mentions = sorted({family for family in ("project-rank8", "strassen", "conventional")
                              if family in rationale.lower()})
    selected_families = {hash_value: known[hash_value]["algorithm"] for hash_value in chosen}
    mode = proposal["execution_mode"]
    return {"selected_hashes": chosen, "selected_families": selected_families,
            "scope_hashes": sorted(scope), "execution_mode": mode,
            "frozen_hashes": [row["schedule_hash"] for row in selection["frozen"][mode]],
            "calibrated_hashes": [row["schedule_hash"] for row in selection["calibrated"][mode]],
            "frozen_by_mode": {item: [row["schedule_hash"] for row in selection["frozen"][item]]
                               for item in MODES},
            "calibrated_by_mode": {item: [row["schedule_hash"] for row in selection["calibrated"][item]]
                                   for item in MODES},
            "guided_by_mode": {item: [row["schedule_hash"] for row in selection["guided"][item]]
                               for item in MODES},
            "substantive_fields": {key: proposal[key] for key in SUBSTANTIVE_FIELDS},
            "rationale_hashes_recognized": mentioned,
            "rationale_hash_prefixes_unknown": unknown_prefixes,
            "rationale_family_mentions": family_mentions,
            "family_claim_review_required": bool(family_mentions and
                set(selected_families.values()) - set(family_mentions)),
            "rationale_ms_values": claims,
            "rationale_ms_values_matching_a_batch1_median": matches,
            "checkable_timing_citations": citations,
            "mentions_uncertainty": bool(_UNCERTAINTY.search(rationale)),
            "free_text_claims_require_manual_review": True}


def compare_trials(records: list[dict]) -> dict:
    """Describe matched differences without declaring evidence causality."""
    pairs = []
    for trial in sorted({row["trial"] for row in records}):
        rows = {row["condition"]: row for row in records if row["trial"] == trial}
        if set(rows) != set(CONDITIONS) or not all(row["validation"]["valid"] for row in rows.values()):
            pairs.append({"trial": trial, "complete_valid_pair": False})
            continue
        left, right = (rows[condition]["assessment"] for condition in CONDITIONS)
        fields = [key for key in SUBSTANTIVE_FIELDS
                  if left["substantive_fields"][key] != right["substantive_fields"][key]]
        pairs.append({"trial": trial, "complete_valid_pair": True,
                      "changed_fields": fields,
                      "selected_set_changed": set(left["selected_hashes"]) != set(right["selected_hashes"]),
                      "selected_order_changed": left["selected_hashes"] != right["selected_hashes"],
                      "control_decision": left["substantive_fields"],
                      "treatment_decision": right["substantive_fields"],
                      "control_selected": left["selected_hashes"],
                      "treatment_selected": right["selected_hashes"],
                      "control_mentions_uncertainty": left["mentions_uncertainty"],
                      "treatment_mentions_uncertainty": right["mentions_uncertainty"]})
    valid = [row for row in pairs if row["complete_valid_pair"]]
    consistent = (len(valid) >= 2 and len(valid) == len(pairs) and
                  bool(valid[0]["changed_fields"] or valid[0]["selected_set_changed"]) and
                  all(row["changed_fields"] == valid[0]["changed_fields"] and
                      row["control_decision"] == valid[0]["control_decision"] and
                      row["treatment_decision"] == valid[0]["treatment_decision"] and
                      row["control_selected"] == valid[0]["control_selected"] and
                      row["treatment_selected"] == valid[0]["treatment_selected"]
                      for row in valid))
    control_unique = len({tuple(row["control_selected"]) for row in valid})
    treatment_unique = len({tuple(row["treatment_selected"]) for row in valid})
    control_decisions = len({canonical(row["control_decision"]) for row in valid})
    treatment_decisions = len({canonical(row["treatment_decision"]) for row in valid})
    return {"pairs": pairs, "valid_pairs": len(valid),
            "consistent_decision_difference": consistent,
            "control_unique_decisions": control_decisions,
            "treatment_unique_decisions": treatment_decisions,
            "control_unique_selections": control_unique,
            "treatment_unique_selections": treatment_unique,
            "within_condition_selection_variability_observed":
                control_unique > 1 or treatment_unique > 1,
            "within_condition_decision_variability_observed":
                control_decisions > 1 or treatment_decisions > 1,
            "causal_effect_or_performance_improvement_established": False,
            "note": "Repeated matched choices support review; free-text grounding and causality require independent judgment."}


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
