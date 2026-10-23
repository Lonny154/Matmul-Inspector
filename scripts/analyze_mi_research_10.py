"""Replay MI-Research-10 source, compiler, correctness, and timing evidence."""

from __future__ import annotations

import json
from pathlib import Path
import sys

try:
    from scripts import freeze_mi_research_10 as frozen
    from scripts import run_mi_research_10 as study
    from scripts import time_mi_research_10 as timing
    from scripts import run_mi_research_2c_graph as evidence
except ModuleNotFoundError:
    import freeze_mi_research_10 as frozen
    import run_mi_research_10 as study
    import time_mi_research_10 as timing
    import run_mi_research_2c_graph as evidence

OUT = frozen.OUT
ANALYSIS = OUT / "analysis.json"
MANIFEST = OUT / "hash_manifest.json"
CONTROL = frozen.candidate.ROOT / "artifacts/mi-research-7b/primary-s1"


def analyze() -> dict:
    if study.verify()["status"] != "PASS" or timing.verify()["status"] != "PASS":
        raise ValueError("GPU validation or timing evidence did not verify")
    plan = json.loads(frozen.PLAN.read_text())
    vectors = {row["circuit"]: row["descriptors"] for row in plan["candidates"]}
    old_raw = json.loads((CONTROL / "raw.json").read_text())
    new_raw = json.loads((study.OUT / "raw.json").read_text())
    pair_raw = json.loads((timing.OUT / "raw.json").read_text())
    old_counts = study.static_counts((CONTROL / "A.ttir").read_text(), (CONTROL / "A.ptx").read_text())
    new_counts = new_raw["static_counts"]
    count_delta = {name: new_counts[name] - old_counts[name] for name in old_counts}
    old_meta, new_meta = old_raw["compiler"]["A"]["metadata"], new_raw["compiler"]["metadata"]
    bytes_delta = (vectors["perminov_58"]["max_estimated_live_bytes_per_cta"] -
                   vectors["perminov_55"]["max_estimated_live_bytes_per_cta"])
    if bytes_delta != 4096:
        raise ValueError("expected model descriptor changed")
    if old_raw["compiler"]["A"]["ptx_sha256"] != evidence.digest(CONTROL / "A.ptx"):
        raise ValueError("historical compiler control changed")
    return {
        "schema_version": 1, "study": "MI-Research-10", "classification": "B",
        "freeze_commit": "5df762ae9e6732a19a899d1076a67d0d02b07ab0",
        "frozen_plan_sha256": evidence.digest(frozen.PLAN),
        "plan_audit": {"descriptor_58_minus_55_live_bytes": bytes_delta,
                       "frozen_prose_claim": "58 has 4096 fewer modeled live bytes per CTA",
                       "status": "CONTRADICTS_FROZEN_DESCRIPTOR",
                       "disposition": "preserve frozen plan; do not count prose liveness claim as a valid prospective prediction"},
        "source_descriptors": vectors,
        "compiler_control": {"source": "artifacts/mi-research-7b/primary-s1/A_source.py",
                             "source_sha256": old_raw["source_sha256"]["A"],
                             "compiler_artifact_sha256": {"ttir": evidence.digest(CONTROL / "A.ttir"),
                                                          "ptx": evidence.digest(CONTROL / "A.ptx")},
                             "static_counts": old_counts, "metadata": old_meta},
        "compiler_candidate": {"source_sha256": new_raw["candidate_source_sha256"],
                               "static_counts": new_counts, "metadata": new_meta},
        "compiler_candidate_minus_control": {"static_counts": count_delta,
                                             "registers_per_thread": new_meta["n_regs"] - old_meta["n_regs"],
                                             "reported_spills": new_meta["n_spills"] - old_meta["n_spills"],
                                             "shared_bytes": new_meta["shared"] - old_meta["shared"]},
        "correctness": {"candidate_cases_passed": len(new_raw["correctness"]),
                        "candidate_repeats_per_case": 2,
                        "graph_pair_cases_passed": len(pair_raw["correctness"]),
                        "graph_pair_repeats_per_case_and_mode": 2,
                        "tolerance": new_raw["tolerance"]},
        "topology": new_raw["topology"],
        "timing": {"mode": "fixed_address_cuda_graph", "fresh_paired_sessions": 1,
                   "operations_per_replay": pair_raw["graph_operations_per_replay"],
                   "rounds": len(pair_raw["pair"]["rounds"]),
                   "median_us_per_operation": pair_raw["event_median_us"],
                   "paired_median_A_minus_B_us": pair_raw["paired_median_A_minus_B_us"],
                   "aa_max_absolute_gap_us": pair_raw["decision"]["aa_control"]["max_absolute_difference_ms"] * 1000,
                   "clock_status": pair_raw["host_clock_preflight"]["status"],
                   "decision": pair_raw["decision"]["decision"],
                   "ranking_supported": pair_raw["decision"]["ranking_supported"]},
        "interpretation": {
            "source_work": "Four-CTA 58 has 12 more gate instances per grid and equal dot instances; no duplicated products in either mapping.",
            "compiler": "Equal TTIR dot and PTX MMA counts; more total PTX FP32 add/sub text for 58. Reported spills rise by 28 despite equal 255 registers and 1024 shared bytes.",
            "liveness": "The descriptor predicts 4096 more logical live bytes for 58. The frozen prose says the opposite, so liveness-to-resource prediction is not cleanly preregistered.",
            "performance": "One paired CUDA Graph session is inconclusive; no speedup claimed.",
        },
    }


def manifest_entries() -> dict:
    return {str(path.relative_to(OUT)): evidence.digest(path) for path in sorted(OUT.rglob("*"))
            if path.is_file() and path != MANIFEST}


def verify() -> dict:
    if json.loads(ANALYSIS.read_text()) != analyze():
        raise ValueError("analysis replay mismatch")
    if json.loads(MANIFEST.read_text()) != manifest_entries():
        raise ValueError("MI-Research-10 hash manifest mismatch")
    return {"status": "PASS", "classification": "B", "plan_audit": "CONTRADICTS_FROZEN_DESCRIPTOR"}


if __name__ == "__main__":
    if "--verify" in sys.argv:
        print(json.dumps(verify(), sort_keys=True))
    else:
        if ANALYSIS.exists() or MANIFEST.exists():
            raise FileExistsError("analysis already exists")
        evidence.write_json(ANALYSIS, analyze())
        evidence.write_json(MANIFEST, manifest_entries())
        print(json.dumps(verify(), sort_keys=True))
