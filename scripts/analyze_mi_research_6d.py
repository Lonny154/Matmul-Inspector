"""CPU-only replay and cross-session decision for MI-Research-6D."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts import run_mi_research_2c_graph as artifacts
    from scripts import run_mi_research_6d_graph as study
except ModuleNotFoundError:
    import run_mi_research_2c_graph as artifacts
    import run_mi_research_6d_graph as study

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "artifacts/mi-research-6d"


def analyze(root: Path = EVIDENCE) -> dict:
    sessions = []
    for number in (1, 2):
        bundle = root / f"graph-s{number}"
        if study.verify(bundle)["status"] != "PASS":
            raise ValueError(f"session {number} failed verification")
        raw = json.loads((bundle / "raw.json").read_text())
        summary = json.loads((bundle / "summary.json").read_text())
        if raw["session"] != number or not raw["precision"]["restored"]:
            raise ValueError("session identity or precision restoration changed")
        sessions.append({
            "session": number, "manifest_sha256": artifacts.digest(bundle / "hash_manifest.json"),
            "capture_order": raw["capture_order"], "capture_setup_ms": raw["capture_setup_ms"],
            "first_compile_ms": raw["first_compile_ms"],
            "physical_kernels_per_operation": {label: raw["graph_profile"][label]["kernels_per_operation"]
                                               for label in study.LABELS},
            "decision": summary["decision"]["decision"],
            "reason": summary["decision"]["reason"],
            "warnings": summary["decision"].get("warnings", []),
            "event_us": {label: {key: summary["event_us"][label][key]
                                 for key in ("median_us", "min_us", "max_us", "mad_us", "cv")}
                         for label in study.LABELS},
            "paired_A_minus_B_us": summary["paired_A_minus_B_us"],
            "paired_median_us": summary["paired_median_us"],
            "paired_mad_us": summary["paired_mad_us"],
            "aa_differences_us": summary["aa_differences_us"],
            "aa_max_absolute_gap_us": summary["aa_max_absolute_gap_us"],
            "host_clock_preflight": raw["benchmark"]["host_clock_preflight"]["status"],
        })
    if all(row["decision"] == "B_FASTER" for row in sessions):
        classification = "A"
    elif all(row["decision"] == "INCONCLUSIVE" and
             "AB_WITHIN_AA_VARIABILITY" in row["warnings"] for row in sessions):
        classification = "C"
    else:
        classification = "B"
    return {"schema_version": 1, "milestone": "MI-Research-6D",
            "mode": "fixed_address_cuda_graph", "operations_per_replay": study.OPERATIONS,
            "staged_source_sha256": study.previous.SOURCE_SHA256,
            "fused_source_sha256": study.FUSED_SHA256,
            "sessions": sessions, "classification": classification,
            "performance_advantage_demonstrated": classification == "A",
            "comparison_limit": "The candidates differ in algebraic representation and GPU execution design. No causal contribution of addition count, fusion, launch boundaries, product scratch, registers, or CTA count is isolated."}


def run(*, verify: bool = False, root: Path = EVIDENCE) -> dict:
    result = analyze(root)
    path = root / "analysis.json"
    manifest = root / "analysis_manifest.json"
    if verify:
        if (result != json.loads(path.read_text()) or
                json.loads(manifest.read_text()) != {
                    "analysis.json": artifacts.digest(path),
                    "graph-s1/hash_manifest.json": artifacts.digest(root / "graph-s1/hash_manifest.json"),
                    "graph-s2/hash_manifest.json": artifacts.digest(root / "graph-s2/hash_manifest.json")}):
            raise ValueError("cross-session analysis replay mismatch")
        return {"status": "PASS", "classification": result["classification"],
                "session_decisions": [row["decision"] for row in result["sessions"]]}
    if path.exists() or manifest.exists():
        raise FileExistsError("6D analysis already exists")
    artifacts.write_json(path, result)
    artifacts.write_json(manifest, {
        "analysis.json": artifacts.digest(path),
        "graph-s1/hash_manifest.json": artifacts.digest(root / "graph-s1/hash_manifest.json"),
        "graph-s2/hash_manifest.json": artifacts.digest(root / "graph-s2/hash_manifest.json")})
    return {"status": "PASS", "classification": result["classification"],
            "session_decisions": [row["decision"] for row in result["sessions"]]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--root", type=Path, default=EVIDENCE)
    args = parser.parse_args()
    print(json.dumps(run(verify=args.verify, root=args.root), sort_keys=True))


if __name__ == "__main__":
    main()
