"""Replay the MI-Research-7 four-session decision matrix without a GPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts import run_mi_research_7 as study
    from scripts import run_mi_research_2c_graph as evidence
except ModuleNotFoundError:
    import run_mi_research_7 as study
    import run_mi_research_2c_graph as evidence


BASE = study.BASE
ANALYSIS = BASE / "analysis.json"
MANIFEST = BASE / "analysis_manifest.json"
SESSION_NAMES = ("eager-s1", "eager-s2", "cuda_graph-s1", "cuda_graph-s2")
MANIFEST_FILES = ("protocol.json", "analysis.json", *(
    f"{name}/hash_manifest.json" for name in SESSION_NAMES))


def analyze() -> dict:
    study.check_protocol()
    sessions = {}
    for name in SESSION_NAMES:
        bundle = BASE / name
        if study.verify(bundle)["status"] != "PASS":
            raise ValueError(f"MI-Research-7 session failed verification: {name}")
        raw = json.loads((bundle / "raw.json").read_text())
        summary = study.summarize(raw)
        if name != f"{raw['mode']}-s{raw['session']}":
            raise ValueError("MI-Research-7 mode/session label mismatch")
        sessions[name] = {
            "manifest_sha256": evidence.digest(bundle / "hash_manifest.json"),
            "correctness_passed": summary["correctness_passed"],
            "capture_or_initialization_order": raw["capture_or_initialization_order"],
            "pair_order": raw["pair_order"],
            "first_call_ms": raw["first_call_ms"],
            "capture_setup_ms": raw["capture_setup_ms"],
            "host_clock_preflight_status": raw["host_clock_preflight"]["status"],
            "topology": raw["topology"],
            "compiler": raw["compiler"],
            "precision_policy": raw["precision_policy"],
            "pairs": {other: {
                "decision": result["decision"]["decision"],
                "reason": result["decision"]["reason"],
                "warnings": result["decision"]["warnings"],
                "event_median_us": result["event_median_us"],
                "event_us_per_operation": result["event_us_per_operation"],
                "paired_P_minus_baseline_us": result["paired_P_minus_baseline_us"],
                "paired_median_us": result["paired_median_us"],
                "paired_mad_us": result["paired_mad_us"],
                "perminov_faster_rounds": result["perminov_faster_rounds"],
                "baseline_faster_rounds": result["baseline_faster_rounds"],
                "aa_max_absolute_gap_us": result["aa_max_absolute_gap_us"],
                "paired_host_ms": result["decision"].get("paired_host_A_minus_B_ms"),
                "paired_raw_ms": result["decision"].get("paired_raw_A_minus_B_ms")}
                for other, result in summary["pairs"].items()}}
    decisions = {mode: {other: [sessions[f"{mode}-s{number}"]["pairs"][other]["decision"]
                                for number in (1, 2)] for other in ("T", "C")}
                 for mode in ("eager", "cuda_graph")}
    if any(values == ["A_FASTER", "A_FASTER"]
           for pairs in decisions.values() for values in pairs.values()):
        classification = "A — Practical Perminov advantage demonstrated"
    elif any(values == ["B_FASTER", "B_FASTER"]
             for pairs in decisions.values() for values in pairs.values()):
        classification = "B — Valid comparison without a supported advantage"
    elif all(value == "INCONCLUSIVE" for pairs in decisions.values()
             for values in pairs.values() for value in values):
        classification = "C — Timing variability prevents interpretation"
    else:
        classification = "B — Valid comparison without a supported advantage"
    return {"schema_version": 1, "protocol_sha256": study.PROTOCOL_SHA256,
            "base_commit": study.BASE_COMMIT,
            "source_sha256": {"P": study.FUSED_SHA256, "T": study.CONVENTIONAL_SHA256},
            "sessions": sessions, "decisions": decisions,
            "classification": classification,
            "eager_graph_rankings_separate": True,
            "interpretation": "Both graph comparisons support the conventional baseline in two fresh sessions; eager comparisons are inconclusive under the unchanged rule. Compiler resources and four configured Perminov CTAs do not by themselves establish the cause."}


def write() -> dict:
    if ANALYSIS.exists() or MANIFEST.exists():
        raise FileExistsError("MI-Research-7 analysis already exists")
    result = analyze()
    evidence.write_json(ANALYSIS, result)
    evidence.write_json(MANIFEST, {name: evidence.digest(BASE / name)
                                   for name in MANIFEST_FILES})
    return verify()


def verify() -> dict:
    manifest = json.loads(MANIFEST.read_text())
    if set(manifest) != set(MANIFEST_FILES) or any(
            evidence.digest(BASE / name) != digest for name, digest in manifest.items()):
        raise ValueError("MI-Research-7 analysis manifest mismatch")
    current = analyze()
    if current != json.loads(ANALYSIS.read_text()):
        raise ValueError("MI-Research-7 cross-session analysis changed")
    return {"status": "PASS", "classification": current["classification"],
            "sessions": len(current["sessions"]), "decisions": current["decisions"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify() if args.verify else write(), sort_keys=True))


if __name__ == "__main__":
    main()
