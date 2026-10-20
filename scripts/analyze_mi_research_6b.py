"""CPU-only cross-session replay for the bounded Laderman graph study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

try:
    from scripts import run_mi_research_2c_graph as artifacts
    from scripts import run_mi_research_6b_graph as study
    from scripts import run_mi_research_6_gpu_probe as eager
except ModuleNotFoundError:
    import run_mi_research_2c_graph as artifacts
    import run_mi_research_6b_graph as study
    import run_mi_research_6_gpu_probe as eager

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "artifacts/mi-research-6b"
EAGER = ROOT / "artifacts/mi-research-6/gpu-probe-n96-eager-s1"


def _stats(values: list[float]) -> dict:
    return {"median_us": statistics.median(values), "min_us": min(values),
            "max_us": max(values), "cv": statistics.stdev(values) / statistics.fmean(values)}


def analyze() -> dict:
    if eager.verify(EAGER)["status"] != "PASS":
        raise ValueError("historical eager bundle failed verification")
    eager_raw = json.loads((EAGER / "raw.json").read_text())
    eager_event = {label: [row["measurements"][label]["cuda_event_elapsed_ms_per_operation"] * 1000
                           for row in eager_raw["timing"]["rounds"]] for label in study.LABELS}
    sessions = []
    for number in (1, 2):
        path = EVIDENCE / f"graph-s{number}"
        if study.verify(path)["status"] != "PASS":
            raise ValueError(f"graph session {number} failed verification")
        raw = json.loads((path / "raw.json").read_text())
        summary = json.loads((path / "summary.json").read_text())
        if raw["session"] != number or raw["source_sha256"] != study.SOURCE_SHA256:
            raise ValueError("session or kernel identity changed")
        sessions.append({"session": number, "manifest_sha256": artifacts.digest(path / "hash_manifest.json"),
                         "capture_order": raw["capture_order"],
                         "capture_setup_ms": raw["capture_setup_ms"],
                         "first_compile_ms": raw["first_compile_ms"],
                         "correctness_passed": summary["correctness_passed"],
                         "topology_passed": summary["topology_passed"],
                         "physical_kernels_per_replay": {label: raw["graph_profile"][label]["kernel_events"]
                                                         for label in study.LABELS},
                         "decision": summary["decision"]["decision"],
                         "decision_reason": summary["decision"]["reason"],
                         "warnings": summary["decision"].get("warnings", []),
                         "event_us": {label: {key: summary["event_us"][label][key]
                                               for key in ("median_us", "min_us", "max_us", "mad_us", "cv")}
                                      for label in study.LABELS},
                         "paired_median_us": summary["paired_median_us"],
                         "paired_mad_us": summary["paired_mad_us"],
                         "paired_A_minus_B_us": summary["paired_A_minus_B_us"],
                         "aa_max_absolute_gap_us": summary["aa_max_absolute_gap_us"],
                         "host_clock_preflight": raw["benchmark"]["host_clock_preflight"]["status"]})
    if all(item["decision"] == "B_FASTER" for item in sessions):
        classification = "A"
    elif all(item["decision"] == "INCONCLUSIVE" and
             "AB_WITHIN_AA_VARIABILITY" in item["warnings"] for item in sessions):
        classification = "C"
    else:
        classification = "B"
    return {"schema_version": 1, "milestone": "MI-Research-6B",
            "source_sha256": study.SOURCE_SHA256,
            "coefficients_sha256": study.COEFFICIENTS_SHA256,
            "historical_eager_manifest_sha256": artifacts.digest(EAGER / "hash_manifest.json"),
            "mode": "fixed_address_cuda_graph", "operations_per_replay": study.OPERATIONS,
            "historical_eager": {"mode": "eager", "matched_to_graph": False,
                                 "event_us": {label: _stats(eager_event[label]) for label in study.LABELS},
                                 "decision": "INCONCLUSIVE"},
            "sessions": sessions, "classification": classification,
            "performance_improvement_demonstrated": classification == "A",
            "comparison_limit": "Historical eager and new graph sessions differ in process, execution mode, and measurement time. Their medians do not establish a controlled mode speedup or a causal mechanism.",
            "external_anchor_status": "NOT_MEASURED"}


def run(verify: bool) -> dict:
    expected = analyze()
    path = EVIDENCE / "analysis.json"
    manifest_path = EVIDENCE / "analysis_manifest.json"
    if verify:
        if json.loads(path.read_text()) != expected:
            raise ValueError("cross-session analysis does not replay")
        manifest = json.loads(manifest_path.read_text())
        if manifest != {"analysis.json": artifacts.digest(path),
                        "graph-s1/hash_manifest.json": artifacts.digest(EVIDENCE / "graph-s1/hash_manifest.json"),
                        "graph-s2/hash_manifest.json": artifacts.digest(EVIDENCE / "graph-s2/hash_manifest.json")}:
            raise ValueError("cross-session manifest mismatch")
    else:
        if path.exists() or manifest_path.exists():
            raise FileExistsError("cross-session analysis already exists")
        artifacts.write_json(path, expected)
        artifacts.write_json(manifest_path, {
            "analysis.json": artifacts.digest(path),
            "graph-s1/hash_manifest.json": artifacts.digest(EVIDENCE / "graph-s1/hash_manifest.json"),
            "graph-s2/hash_manifest.json": artifacts.digest(EVIDENCE / "graph-s2/hash_manifest.json")})
    return {"status": "PASS", "classification": expected["classification"],
            "session_decisions": [row["decision"] for row in expected["sessions"]]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    print(json.dumps(run(parser.parse_args().verify), sort_keys=True))
