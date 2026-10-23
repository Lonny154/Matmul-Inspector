"""CPU-only consistency gate for structured MI-Research-10 source-cost claims.

Claims compare candidate minus control. Historical prose and evidence are never
parsed, edited, or silently corrected by this module.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _numeric_fields(value: dict, prefix: str = "") -> dict[str, int]:
    fields = {}
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            fields.update(_numeric_fields(item, path))
        elif type(item) is int:
            fields[path] = item
        elif isinstance(item, (float, bool)):
            raise ValueError(f"unsupported numeric descriptor {path}: expected an integer")
    return fields


def _candidates(plan: dict, control_id: str, candidate_id: str) -> tuple[dict, dict]:
    rows = plan.get("candidates")
    if (not isinstance(rows, list) or not rows or
            any(not isinstance(row, dict) or not isinstance(row.get("id"), str) for row in rows) or
            len({row["id"] for row in rows}) != len(rows)):
        raise ValueError("plan must contain candidates with unique IDs")
    by_id = {row["id"]: row for row in rows}
    if (not isinstance(control_id, str) or not isinstance(candidate_id, str) or
            control_id == candidate_id or control_id not in by_id or candidate_id not in by_id):
        raise ValueError("descriptor_claims must name distinct, existing control_id and candidate_id")
    control, candidate = by_id[control_id], by_id[candidate_id]
    if control.get("role") != "retained_compiled_control" or candidate.get("role") != "prospective_unbenchmarked_candidate":
        raise ValueError("descriptor_claims IDs must identify the retained control and prospective candidate")
    if control.get("mapping") != candidate.get("mapping"):
        raise ValueError("directional claims require the same output mapping")
    if not isinstance(control.get("descriptors"), dict) or not isinstance(candidate.get("descriptors"), dict):
        raise ValueError("control and candidate need descriptor mappings")
    return control, candidate


def _comparison(plan: dict, control_id: str, candidate_id: str) -> dict[str, dict]:
    control, candidate = _candidates(plan, control_id, candidate_id)
    a, b = _numeric_fields(control["descriptors"]), _numeric_fields(candidate["descriptors"])
    if not a or a.keys() != b.keys():
        raise ValueError("control and candidate must have the same nonempty numeric descriptor fields")
    result = {}
    for field in sorted(a):
        delta = b[field] - a[field]
        result[field] = {"control": a[field], "candidate": b[field],
                         "delta": delta,
                         "direction": "more" if delta > 0 else "fewer" if delta < 0 else "equal"}
    return result


def build_claims(plan: dict, control_id: str, candidate_id: str) -> dict:
    """Explicitly create a new structured claim block from descriptor data."""
    comparison = _comparison(plan, control_id, candidate_id)
    return {"control_id": control_id, "candidate_id": candidate_id,
            "fields": {field: {"direction": row["direction"], "delta": row["delta"]}
                       for field, row in comparison.items()}}


def validate(plan: dict) -> dict[str, dict]:
    """Reject missing, incomplete, or contradictory claims before compilation."""
    claims = plan.get("descriptor_claims")
    if not isinstance(claims, dict):
        raise ValueError("missing descriptor_claims; add control_id, candidate_id, and "
                         "structured fields to a new plan before compilation. "
                         "The historical frozen plan must remain unchanged")
    if plan.get("screening_hypotheses"):
        raise ValueError("legacy screening_hypotheses cannot be checked for directional "
                         "consistency; create a new plan with descriptor_claims and derive "
                         "source comparisons from the validated summary")
    comparison = _comparison(plan, claims.get("control_id"), claims.get("candidate_id"))
    fields = claims.get("fields")
    if not isinstance(fields, dict):
        raise ValueError("descriptor_claims.fields must be a mapping of numeric descriptor paths")
    missing, extra = set(comparison) - set(fields), set(fields) - set(comparison)
    if missing or extra:
        raise ValueError(f"descriptor_claims must cover every numeric field; missing={sorted(missing)}, "
                         f"unknown={sorted(extra)}")
    for field, expected in comparison.items():
        stated = fields[field]
        if not isinstance(stated, dict) or set(stated) != {"direction", "delta"}:
            raise ValueError(f"{field}: state direction and signed delta")
        direction, delta = stated["direction"], stated["delta"]
        if direction not in ("more", "fewer", "equal") or type(delta) is not int:
            raise ValueError(f"{field}: direction must be more/fewer/equal and delta an integer")
        if direction != expected["direction"] or delta != expected["delta"]:
            raise ValueError(
                f"{field}: expected {expected['direction']} (candidate {expected['candidate']} "
                f"vs control {expected['control']}; delta {expected['delta']:+d}), "
                f"stated {direction} (delta {delta:+d}). Correct the new claim or its descriptor "
                "provenance before compilation; do not edit historical frozen evidence")
    return comparison


def summary(plan: dict) -> list[str]:
    """Render comparisons only after all structured claims pass validation."""
    return [f"{field}: candidate {row['candidate']} vs control {row['control']} "
            f"({row['delta']:+d}; {row['direction']})"
            for field, row in validate(plan).items()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path, help="new preregistration JSON with descriptor_claims")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    try:
        lines = summary(plan)
    except ValueError as error:
        parser.exit(2, f"preregistration rejected: {error}\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
