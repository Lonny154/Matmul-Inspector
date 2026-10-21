"""Offline cross-session verification and analysis for MI-Research-7B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

try:
    from scripts import run_mi_research_7b as study
    from scripts import run_mi_research_2c_graph as evidence
    from scripts import analyze_mi_research_7 as prior
except ModuleNotFoundError:
    import run_mi_research_7b as study
    import run_mi_research_2c_graph as evidence
    import analyze_mi_research_7 as prior

BASE = study.BASE
ANALYSIS = BASE / "analysis.json"
MANIFEST = BASE / "analysis_manifest.json"
SESSIONS = ("primary-s1", "primary-s2", "cublas-s1")
MANIFEST_FILES = ("protocol.json", "mapping_plan.json", "analysis.json", *(
    f"{name}/hash_manifest.json" for name in SESSIONS))


def analyze() -> dict:
    study.check_protocol()
    if prior.verify()["status"] != "PASS":
        raise ValueError("historical MI-Research-7 evidence changed")
    sessions = {}
    for name in SESSIONS:
        bundle = BASE / name
        if study.verify(bundle)["status"] != "PASS":
            raise ValueError(f"7B session failed verification: {name}")
        raw = json.loads((bundle / "raw.json").read_text())
        summary = study.summarize(raw)
        if raw["protocol_sha256"] != evidence.digest(BASE / "protocol.json") or (
            raw["mapping_plan_sha256"] != evidence.digest(BASE / "mapping_plan.json") or
            raw["source_sha256"] != study.check_protocol()["source_sha256"] or
            [row["case"] for row in raw["correctness"]] != list(study.cases.CASE_NAMES) or
            raw["tolerance"] != study.TOLERANCE or raw["size"] != study.SIZE or
            not raw["fixed_addresses_unchanged"] or
            raw["precision_policy"]["active_allow_fp16_reduced_precision_reduction"] is not False):
            raise ValueError(f"7B workload or provenance changed: {name}")
        for case in raw["correctness"]:
            for label in raw["physical_labels"]:
                c = case["checks"][label]
                if not c["graph_eager_bitwise_equal"] or any(
                    not x["passed"] or x["failing_elements"] or x["candidate_nonfinite_count"]
                    for phase in ("eager", "cuda_graph") for x in c[phase]):
                    raise ValueError(f"7B numerical check changed: {name}, {case['case']}")
        for label in raw["physical_labels"]:
            if raw["topology"][label]["physical_kernel_events"] != study.OPS:
                raise ValueError(f"7B launch count changed: {name}, {label}")
        pair = raw["pair"]
        for i, row in enumerate(pair["rounds"], 1):
            if row["round"] != i or row["order"] != study.experiment_pair.pair_order(i):
                raise ValueError("7B round order changed")
            for sample in row["measurements"].values():
                if sample["complete_operations_per_block"] != study.OPS or sample[
                    "host_submission_calls_per_block"] != 1:
                    raise ValueError("7B timing operation count changed")
        compiler = {}
        for label, record in raw["compiler"].items():
            ptx = (bundle / f"{label}.ptx").read_text()
            compiler[label] = {"metadata": record["metadata"],
                               "static_ptx_mma_sync_occurrences": ptx.count("mma.sync"),
                               "ptx_ld_local_text_occurrences": ptx.count("ld.local"),
                               "ptx_st_local_text_occurrences": ptx.count("st.local"),
                               "ptx_sha256": record["ptx_sha256"],
                               "ir_sha256": record["ir_sha256"]}
        gaps = summary["paired_A_minus_B_us"]
        sessions[name] = {"manifest_sha256": evidence.digest(bundle / "hash_manifest.json"),
                          "physical_labels": raw["physical_labels"],
                          "initialization_order": raw["initialization_order"],
                          "decision": summary["decision"]["decision"],
                          "reason": summary["decision"]["reason"],
                          "warnings": summary["decision"]["warnings"],
                          "event_us_per_operation": summary["event_us_per_operation"],
                          "event_median_us": summary["event_median_us"],
                          "paired_A_minus_B_us": gaps,
                          "paired_median_us": statistics.median(gaps),
                          "aa_max_absolute_gap_us": summary["aa_max_absolute_gap_us"],
                          "compiler": compiler, "topology": raw["topology"],
                          "host_clock_preflight_status": raw["host_clock_preflight"]["status"],
                          "environment": raw["environment"]}
    if [sessions[f"primary-s{i}"]["decision"] for i in (1, 2)] != ["B_FASTER"] * 2:
        raise ValueError("7B conditional cuBLAS comparison lacked primary trigger")
    if (sessions["cublas-s1"]["physical_labels"] != ["B", "C"] or
        sessions["cublas-s1"]["decision"] != "B_FASTER"):
        raise ValueError("7B cuBLAS labels changed")
    return {"schema_version": 1, "base_commit": study.check_protocol()["base_commit"],
            "protocol_sha256": evidence.digest(study.PROTOCOL),
            "mapping_plan_sha256": evidence.digest(study.MAPPING),
            "source_sha256": study.check_protocol()["source_sha256"],
            "historical_7_analysis_manifest_sha256": evidence.digest(prior.MANIFEST),
            "sessions": sessions,
            "classification": "A — Higher-parallelism performance improvement",
            "cuBLAS_qualification": "No practical cuBLAS advantage: C faster in the single matched comparison",
            "interpretation": "The 36-CTA mapping is reproducibly faster than the four-CTA original despite 2.78x source-level dot duplication. This does not isolate a single causal mechanism or prove hardware spill traffic. Both remain slower than cuBLAS in the matched comparison."}


def write() -> dict:
    if ANALYSIS.exists() or MANIFEST.exists():
        raise FileExistsError("7B analysis already exists")
    result = analyze()
    evidence.write_json(ANALYSIS, result)
    evidence.write_json(MANIFEST, {name: evidence.digest(BASE / name) for name in MANIFEST_FILES})
    return verify()


def verify() -> dict:
    manifest = json.loads(MANIFEST.read_text())
    if set(manifest) != set(MANIFEST_FILES) or any(
        evidence.digest(BASE / name) != sha for name, sha in manifest.items()):
        raise ValueError("7B analysis manifest mismatch")
    if analyze() != json.loads(ANALYSIS.read_text()):
        raise ValueError("7B analysis did not replay")
    return {"status": "PASS", "classification": analyze()["classification"],
            "sessions": len(SESSIONS)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify() if args.verify else write(), sort_keys=True))


if __name__ == "__main__":
    main()
