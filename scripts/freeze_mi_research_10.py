"""Freeze the sole prospective circuit/mapping candidate before GPU compilation."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

try:
    from scripts import analyze_mi_research_9 as costs
    from scripts import generate_mi_research_10_kernel as candidate
    from scripts import run_mi_research_6c as exact
except ModuleNotFoundError:
    import analyze_mi_research_9 as costs
    import generate_mi_research_10_kernel as candidate
    import run_mi_research_6c as exact

OUT = candidate.TARGET.parent
PLAN = OUT / "frozen_plan.json"


def compute() -> dict:
    if costs.verify()["status"] != "PASS":
        raise ValueError("MI-Research-9 replay failed")
    if candidate.TARGET.read_text() != candidate.generate():
        raise ValueError("candidate source changed")
    analysis = json.loads(costs.ANALYSIS.read_text())
    records = []
    for label in ("55", "58"):
        vector = analysis["circuits"][label]["four_cta"]["cost_vector"]
        source = exact.PROBE_SOURCE if label == "55" else candidate.TARGET
        records.append({
            "id": f"perminov{label}_four_cta",
            "role": "retained_compiled_control" if label == "55" else "prospective_unbenchmarked_candidate",
            "circuit": f"perminov_{label}", "mapping": "four_cta",
            "source": str(source.relative_to(costs.ROOT)),
            "source_sha256": costs.digest(source),
            "descriptors": vector,
        })
    return {
        "schema_version": 1, "study": "MI-Research-10",
        "base_commit": "ff55666ea0dff33bdb5ee19061c5cd950ee60be9",
        "prior_evidence_sha256": {str(path.relative_to(costs.ROOT)): costs.digest(path)
                                  for path in (costs.ANALYSIS, costs.MANIFEST, exact.SOURCE,
                                               exact.CERT, exact.OUT / "analysis.json")},
        "tensor_identity": {
            "ordered_block_tensor_shape": [3, 3, 3],
            "rank": 23,
            "source": str(exact.SOURCE.relative_to(costs.ROOT)),
            "certificate": str(exact.CERT.relative_to(costs.ROOT)),
            "factor_maps_identical": analysis["identical_factor_maps"],
            "exact_tensor_check": analysis["tensor"],
        },
        "workload": {"N": 96, "input_dtype": "float16", "output_dtype": "float16",
                     "dot_input_precision": "tf32x3", "output_tile": [16, 16],
                     "grid": [2, 2], "cta_count": 4, "warps_per_cta": 4,
                     "output_ownership": "one CTA owns one local tile in each of nine output blocks"},
        "candidates": records,
        "execution_set": ["perminov58_four_cta"],
        "control_evidence": "artifacts/mi-research-6c/gpu-probe-n96",
        "screening_hypotheses": [
            "Both four-CTA circuits contain 23 dots per CTA and 92 dots per grid; compiled TTIR dot and PTX MMA counts should agree if dot lowering is stable.",
            "The 58 circuit has 12 more source gate instances per grid; compiled arithmetic may differ in count and need not preserve this exact delta.",
            "The 58 circuit has 4096 fewer modeled live bytes per CTA and a dependency path shorter by three operations; test whether reported registers or spills differ, without equating model bytes to allocation.",
            "The 58 circuit is a work/storage/depth tradeoff against the 55 control; no latency direction is predicted.",
        ],
        "candidate_selection_rule": "Use the only previously uncompiled combination of the two certified circuits and the two established output ownership mappings; no latency data used.",
        "prior_benchmarked_combinations": ["perminov55_four_cta", "perminov55_thirty_six_cta", "perminov58_thirty_six_cta"],
        "compilation_status_at_freeze": "not_started",
    }


def main() -> None:
    record = compute()
    if PLAN.exists():
        retained = json.loads(PLAN.read_text())
        retained.pop("frozen_at_utc")
        if retained != record:
            raise ValueError("frozen plan differs from replay")
        print(f"PASS {costs.digest(PLAN)}")
        return
    record["frozen_at_utc"] = datetime.now(timezone.utc).isoformat()
    PLAN.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"FROZEN {costs.digest(PLAN)}")


if __name__ == "__main__":
    main()
