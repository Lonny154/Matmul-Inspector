"""CPU-only verification and bounded counter analysis for MI-Research-7C."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

try:
    from scripts import run_mi_research_2c_graph as evidence
    from scripts import run_mi_research_7c_profile as target
    from scripts import analyze_mi_research_7b as prior
except ModuleNotFoundError:
    import run_mi_research_2c_graph as evidence
    import run_mi_research_7c_profile as target
    import analyze_mi_research_7b as prior

BASE = target.BASE
ANALYSIS = BASE / "analysis.json"
FILES = ("protocol.json", "targeted_capture_2.json", "environment.json", "analysis.json",
         *(f"{label}_{kind}.json" for label in "AB" for kind in ("preflight", "profile", "profile2")),
         *(f"{label}_ncu{suffix}_console.txt" for label in "AB" for suffix in ("", "_targeted")))


def _rows(path: Path) -> tuple[str, list[dict]]:
    content = path.read_text()
    if '"ID","Process ID"' not in content:
        raise ValueError(f"no Nsight CSV in {path.name}")
    text = content[content.index('"ID","Process ID"'):]
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows or "==ERROR==" in content:
        raise ValueError(f"failed Nsight capture {path.name}")
    return content, rows


def _check_capture(label: str, suffix: str, required: set[str]) -> dict:
    path = BASE / f"{label}_ncu{suffix}_console.txt"
    content, rows = _rows(path)
    grid = "(2, 2, 1)" if label == "A" else "(2, 2, 9)"
    marker = f"PROFILE_TARGET label={label} grid={'2x2x1' if label == 'A' else '2x2x9'} one_graph_node"
    if content.count(marker) != 1 or content.count('"status": "PASS"') != 1:
        raise ValueError(f"target process marker changed: {path.name}")
    if any(row["ID"] != "0" or row["Kernel Name"] != "_fused" or
           row["Grid Size"] != grid or row["Block Size"] != "(128, 1, 1)" or
           row["CC"] != "8.9" for row in rows):
        raise ValueError(f"wrong profiled kernel: {path.name}")
    metrics = {row["Metric Name"]: row["Metric Value"] for row in rows
               if row["Section Name"] == "Command line profiler metrics"}
    if set(metrics) != required:
        raise ValueError(f"profiled metric set changed: {path.name}")
    if (content.count("==WARNING==") != 2 or
        "uncontrolled GPU caches" not in content or "unmodified GPU clocks" not in content):
        raise ValueError(f"cache or clock policy warning changed: {path.name}")
    return {"grid": grid, "kernel": "_fused", "rows": len(rows),
            "metrics": {key: float(value.replace(",", "")) for key, value in metrics.items()},
            "console_sha256": evidence.digest(path)}


def _check_record(label: str, kind: str, base_commit: str) -> dict:
    path = BASE / f"{label}_{kind}.json"
    raw = json.loads(path.read_text())
    if (raw["label"] != label or raw["source_sha256"] != target.SOURCE_HASHES[label] or
        raw["git_head"] != base_commit or raw["size"] != 96 or
        raw["tolerance"] != target.TOLERANCE or raw["graph_nodes"] != 1 or
        raw["profiled_graph_node"] != (kind != "preflight") or
        raw["expected_grid"] != ([2, 2, 1] if label == "A" else [2, 2, 9]) or
        raw["warmup_eager"] != 10 or raw["warmup_graph"] != 10 or
        not raw["fixed_addresses_unchanged"] or
        raw["precision_policy"]["active_allow_fp16_reduced_precision_reduction"] is not False or
        [x["case"] for x in raw["correctness"]] != list(target.cases.CASE_NAMES)):
        raise ValueError(f"profile target workload changed: {path.name}")
    for case in raw["correctness"]:
        if not case["graph_eager_bitwise_equal"] or any(
            not case[phase]["passed"] or case[phase]["failing_elements"] or
            case[phase]["candidate_nonfinite_count"] for phase in ("eager", "graph")):
            raise ValueError(f"numerical gate changed: {path.name}, {case['case']}")
    expected = {"A": (255, 422), "B": (255, 166)}[label]
    metadata = raw["compiler_metadata"]
    if (metadata["n_regs"], metadata["n_spills"]) != expected or metadata["num_warps"] != 4:
        raise ValueError(f"compiler configuration changed: {path.name}")
    return {"sha256": evidence.digest(path), "compiler_metadata": metadata,
            "correctness_cases": len(raw["correctness"]), "environment": raw["environment"]}


def analyze() -> dict:
    if prior.verify()["status"] != "PASS":
        raise ValueError("7B historical evidence changed")
    protocol = json.loads((BASE / "protocol.json").read_text())
    selected = json.loads((BASE / "targeted_capture_2.json").read_text())
    environment = json.loads((BASE / "environment.json").read_text())
    if (protocol["base_commit"] != "d146df92e1ab9a4c908c0182cdd26caf9d9674e7" or
        protocol["source_sha256"] != target.SOURCE_HASHES or
        protocol["correctness_cases"] != list(target.cases.CASE_NAMES) or
        protocol["tolerance"] != target.TOLERANCE or
        protocol["capture_budget"] != "at most two targeted Nsight Compute captures per kernel; stop if hardware-counter access is denied" or
        not selected["recorded_before_second_captures"] or
        environment["sm_count_from_torch"] != 34 or
        environment["compute_capability"] != "8.9"):
        raise ValueError("7C protocol or environment changed")
    sessions = {}
    for label in "AB":
        records = {kind: _check_record(label, kind, protocol["base_commit"])
                   for kind in ("preflight", "profile", "profile2")}
        first = _check_capture(label, "", {protocol["permission_probe_metric"]})
        second = _check_capture(label, "_targeted", set(selected["metrics"]))
        occupation_rows = _rows(BASE / f"{label}_ncu_targeted_console.txt")[1]
        occupancy = {r["Metric Name"]: float(r["Metric Value"])
                     for r in occupation_rows if r["Section Name"] == "Occupancy" and
                     r["Metric Name"] in ("Theoretical Occupancy", "Achieved Occupancy",
                                            "Achieved Active Warps Per SM", "Block Limit Registers")}
        if set(occupancy) != {"Theoretical Occupancy", "Achieved Occupancy",
                              "Achieved Active Warps Per SM", "Block Limit Registers"}:
            raise ValueError(f"occupancy section incomplete: {label}")
        if abs(first["metrics"][protocol["permission_probe_metric"]] - occupancy["Achieved Occupancy"]) > 0.5:
            raise ValueError(f"separate occupancy captures disagree: {label}")
        sessions[label] = {"records": records, "permission_probe": first,
                           "targeted_capture": second, "occupancy": occupancy}
    a = sessions["A"]["targeted_capture"]["metrics"]
    b = sessions["B"]["targeted_capture"]["metrics"]
    if ([a["sm__ctas_launched.sum"], a["sm__ctas_launched.min"], a["sm__ctas_launched.max"]] != [4, 0, 1] or
        [b["sm__ctas_launched.sum"], b["sm__ctas_launched.min"], b["sm__ctas_launched.max"]] != [36, 1, 2]):
        raise ValueError("CTA coverage interpretation changed")
    static = {}
    for label in "AB":
        base = prior.BASE / "primary-s1"
        ptx, ttir = (base / f"{label}.ptx"), (base / f"{label}.ttir")
        static[label] = {"ttir_dot_count": ttir.read_text().count("tt.dot"),
                         "ptx_mma_sync_text_count": ptx.read_text().count("mma.sync"),
                         "ttir_sha256": evidence.digest(ttir), "ptx_sha256": evidence.digest(ptx)}
    if [(static[label]["ttir_dot_count"], static[label]["ptx_mma_sync_text_count"])
        for label in "AB"] != [(23, 552), (64, 1536)]:
        raise ValueError("7B static compiler evidence changed")
    return {"schema_version": 1, "protocol_sha256": evidence.digest(BASE / "protocol.json"),
            "targeted_capture_2_sha256": evidence.digest(BASE / "targeted_capture_2.json"),
            "environment_sha256": evidence.digest(BASE / "environment.json"),
            "historical_7b_analysis_manifest_sha256": evidence.digest(prior.MANIFEST),
            "sessions": sessions, "static_compiler_evidence": static,
            "measured_active_sm_coverage": {"A": 4, "B": 34},
            "classification": "B — Multiple contributors identified",
            "interpretation": "B used all 34 SMs rather than four, with nearly unchanged occupancy on active SMs. It executed fewer local-memory instructions and requested fewer L1 bytes, while executing more total and tensor-pipe instructions and reading more DRAM bytes. These one-node counter captures do not isolate the latency contribution of each mechanism."}


def write() -> dict:
    if ANALYSIS.exists() or (BASE / "hash_manifest.json").exists():
        raise FileExistsError("7C analysis already exists")
    evidence.write_json(ANALYSIS, analyze())
    evidence.manifest(BASE)
    return verify()


def verify() -> dict:
    manifest = json.loads((BASE / "hash_manifest.json").read_text())
    files = {str(p.relative_to(BASE)) for p in BASE.rglob("*")
             if p.is_file() and p.name != "hash_manifest.json"}
    if files != set(FILES) or set(manifest) != files or any(
        evidence.digest(BASE / name) != sha for name, sha in manifest.items()):
        raise ValueError("7C evidence manifest mismatch")
    if analyze() != json.loads(ANALYSIS.read_text()):
        raise ValueError("7C analysis did not replay")
    return {"status": "PASS", "classification": analyze()["classification"],
            "targeted_captures": 4}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify() if args.verify else write(), sort_keys=True))


if __name__ == "__main__":
    main()
