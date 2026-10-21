"""Offline exact, numerical, compiler, and paired-decision replay for MI-Research-8."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import re

try:
    from scripts import run_mi_research_2c_graph as evidence
    from scripts import run_mi_research_8 as study
    from scripts import run_mi_research_6c as exact
except ModuleNotFoundError:
    import run_mi_research_2c_graph as evidence
    import run_mi_research_8 as study
    import run_mi_research_6c as exact

BASE = study.BASE
ANALYSIS = BASE / "analysis.json"
MANIFEST = BASE / "analysis_manifest.json"
SESSION_NAMES = ("graph-s1", "graph-s2")
MANIFEST_FILES = ("protocol.json", "mapping_plan.json", "analysis.json", *(
    f"{name}/hash_manifest.json" for name in SESSION_NAMES))


def source_liveness(path: Path) -> dict:
    """Count named arrays used by later statements in each branch, not registers."""
    tree = ast.parse(path.read_text())
    kernel = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_fused")
    branches = [node for node in kernel.body if isinstance(node, ast.If)]
    if len(branches) != 9:
        raise ValueError("expected nine output branches")
    peaks = []
    for branch in branches:
        seq = [node for node in branch.body if isinstance(node, ast.Assign) and
               node.targets[0].id.startswith(("A", "B", "Ug", "Vg", "Wg", "M"))]
        seq += [node for node in branch.body if isinstance(node, ast.Expr)]
        peak = 0
        for i in range(len(seq)):
            future = {item.id for later in seq[i+1:] for item in ast.walk(later)
                      if isinstance(item, ast.Name)}
            defined = {node.targets[0].id for node in seq[:i+1] if isinstance(node, ast.Assign)}
            peak = max(peak, len(future & defined))
        peaks.append(peak)
    return {"peak_live_named_arrays_by_output_block": peaks,
            "max_peak_live_named_arrays": max(peaks),
            "model": "AST names assigned and used by a later statement inside each output branch; excludes index arrays, pointer values, compiler temporaries, and layout replication"}


def analyze() -> dict:
    p = study.check_protocol()
    plan = json.loads(study.PLAN.read_text())
    retained = json.loads((exact.OUT / "analysis.json").read_text())["configurations"]
    f55 = retained["perminov_55"]["factor_coefficients"]
    f58 = retained["perminov_58"]["factor_coefficients"]
    if (f55 != f58 or retained["perminov_55"]["tensor"] !=
        retained["perminov_58"]["tensor"] or
        retained["perminov_55"]["tensor"] !=
        {"checked": 729, "mismatches": 0, "unit": 27, "zero": 702} or
        sum(retained["perminov_55"]["addition_counts"].values()) != 55 or
        sum(retained["perminov_58"]["addition_counts"].values()) != 58 or
        plan["implementation_identity"]["new_58_source_sha256"] != evidence.digest(study.SOURCES["B"])):
        raise ValueError("exact circuit identity or frozen plan changed")
    sessions = {}
    for number, name in enumerate(SESSION_NAMES, 1):
        path = BASE / name
        if study.verify(path)["status"] != "PASS":
            raise ValueError(f"session failed: {name}")
        raw = json.loads((path / "raw.json").read_text())
        summary = study.summarize(raw)
        if (raw["session"] != number or raw["initialization_order"] !=
            (["A", "B"] if number == 1 else ["B", "A"]) or
            raw["base_commit"] != p["base_commit"] or
            raw["mapping_plan_sha256"] != evidence.digest(study.PLAN) or
            raw["precision_policy"]["active_allow_fp16_reduced_precision_reduction"] is not False or
            raw["environment"]["compute_capability"] != "8.9"):
            raise ValueError("session provenance or execution policy changed")
        numerical = {}
        for case in raw["correctness"]:
            row = {}
            for label in ("A", "B"):
                checks = case["checks"][label]
                if (not checks["graph_eager_bitwise_equal"] or
                    len(checks["eager"]) != 2 or len(checks["cuda_graph"]) != 2 or any(
                        not check["passed"] or check["failing_elements"] or
                        check["candidate_nonfinite_count"] for phase in ("eager", "cuda_graph")
                        for check in checks[phase])):
                    raise ValueError(f"numerical gate failed: {name}, {case['case']}, {label}")
                row[label] = {phase: [{"max_absolute_error": check["max_absolute_error"],
                                      "mean_absolute_error": check["mean_absolute_error"],
                                      "failing_elements": check["failing_elements"],
                                      "candidate_nonfinite_count": check["candidate_nonfinite_count"]}
                                     for check in checks[phase]] for phase in ("eager", "cuda_graph")}
            numerical[case["case"]] = row
        compiler = {}
        for label in ("A", "B"):
            record = raw["compiler"][label]
            ttir, ptx = (path / f"{label}.ttir").read_text(), (path / f"{label}.ptx").read_text()
            compiler[label] = {"metadata": record["metadata"],
                               "ttir_dot_count": ttir.count("tt.dot"),
                               "ttir_addf_count": ttir.count("arith.addf"),
                               "ttir_subf_count": ttir.count("arith.subf"),
                               "ptx_mma_sync_text_count": ptx.count("mma.sync"),
                               "ptx_add_f32_text_count": len(re.findall(r"\badd\.f32\b", ptx)),
                               "ptx_sub_f32_text_count": len(re.findall(r"\bsub\.f32\b", ptx)),
                               "ptx_ld_local_text_count": ptx.count("ld.local"),
                               "ptx_st_local_text_count": ptx.count("st.local"),
                               "ptx_sha256": record["ptx_sha256"],
                               "ir_sha256": record["ir_sha256"]}
        if [(compiler[l]["ttir_dot_count"], compiler[l]["ptx_mma_sync_text_count"])
            for l in ("A", "B")] != [(64, 1536), (58, 1392)]:
            raise ValueError("frozen compiled dot work changed")
        sessions[name] = {"manifest_sha256": evidence.digest(path / "hash_manifest.json"),
                          "initialization_order": raw["initialization_order"],
                          "environment": raw["environment"],
                          "correctness": numerical, "topology": raw["topology"],
                          "compiler": compiler,
                          "first_call_ms": raw["first_call_ms"],
                          "capture_setup_ms": raw["capture_setup_ms"],
                          "host_clock_preflight_status": raw["host_clock_preflight"]["status"],
                          "decision": summary["decision"],
                          "event_us_per_operation": summary["event_us_per_operation"],
                          "event_median_us": summary["event_median_us"],
                          "paired_A55_minus_B58_us": summary["paired_A55_minus_B58_us"],
                          "paired_median_us": summary["paired_median_us"],
                          "paired_mad_us": summary["paired_mad_us"],
                          "aa_max_absolute_gap_us": summary["aa_max_absolute_gap_us"]}
    decisions = [sessions[name]["decision"]["decision"] for name in SESSION_NAMES]
    classification = ("A — Reproducible 55-addition circuit advantage" if decisions == ["A_FASTER"] * 2 else
                      "B — Reproducible 58-addition circuit advantage" if decisions == ["B_FASTER"] * 2 else
                      "C — Feasible comparison without a supported latency ranking")
    return {"schema_version": 1, "base_commit": p["base_commit"],
            "protocol_sha256": evidence.digest(study.PROTOCOL),
            "mapping_plan_sha256": evidence.digest(study.PLAN),
            "source_sha256": p["source_sha256"],
            "exact_6c_manifest_sha256": evidence.digest(exact.OUT / "hash_manifest.json"),
            "identical_rank_23_factor_maps": True,
            "tensor": retained["perminov_55"]["tensor"],
            "whole_circuit": {label: {"addition_counts": retained[f"perminov_{label}"]["addition_counts"],
                                      "dependency_depth": retained[f"perminov_{label}"]["schedules"]["liveness"]["peak_dependency_depth_operations"],
                                      "reordered_peak_live_intermediates": retained[f"perminov_{label}"]["schedules"]["liveness"]["peak_live_intermediates"]}
                              for label in ("55", "58")},
            "source_slice_liveness": {label: source_liveness(study.SOURCES[label])
                                      for label in ("A", "B")},
            "sessions": sessions, "decisions": decisions,
            "classification": classification,
            "interpretation": "At fixed 36-CTA ownership, the 55-gate whole circuit's longer and more shared dependency graph duplicates more gates and dots across independent output owners. The 58 variant emits fewer FP32 arithmetic and MMA instructions and reports fewer spills, but both paired sessions are inconclusive. This does not isolate a three-addition latency effect."}


def write() -> dict:
    if ANALYSIS.exists() or MANIFEST.exists():
        raise FileExistsError("MI-Research-8 analysis already exists")
    evidence.write_json(ANALYSIS, analyze())
    evidence.write_json(MANIFEST, {name: evidence.digest(BASE / name) for name in MANIFEST_FILES})
    return verify()


def verify() -> dict:
    manifest = json.loads(MANIFEST.read_text())
    if set(manifest) != set(MANIFEST_FILES) or any(
        evidence.digest(BASE / name) != sha for name, sha in manifest.items()):
        raise ValueError("MI-Research-8 analysis manifest mismatch")
    result = analyze()
    if result != json.loads(ANALYSIS.read_text()):
        raise ValueError("MI-Research-8 analysis did not replay")
    return {"status": "PASS", "classification": result["classification"],
            "decisions": result["decisions"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify() if args.verify else write(), sort_keys=True))


if __name__ == "__main__":
    main()
