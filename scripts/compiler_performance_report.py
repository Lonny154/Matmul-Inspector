#!/usr/bin/env python3
"""Combine saved PTX, SASS, and benchmark evidence without rerunning tools."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Sequence


SCHEMA_VERSION = 1
LAYERS = ("ptx", "sass", "benchmark")
SIDES = ("reference", "candidate")
REQUIRED_BENCHMARK_FILES = ("metadata.json", "summary.csv")
OPTIONAL_BENCHMARK_FILES = ("pairwise.csv", "timing_statistics.csv", "timing_samples.csv", "mismatches.csv")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def integer(value: Any) -> int | None:
    try:
        return int(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def number(value: Any) -> float | None:
    try:
        result = float(value) if value not in (None, "") else None
        return result if result is None or result == result else None
    except (TypeError, ValueError):
        return None


def boolean(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    return None


def finding(severity: str, code: str, message: str, evidence: Any = None) -> dict[str, Any]:
    result = {"severity": severity, "code": code, "message": message}
    if evidence is not None:
        result["evidence"] = evidence
    return result


def parse_mapping(text: str | None) -> dict[str, str]:
    if text is None:
        return {}
    result: dict[str, str] = {}
    for item in text.split(","):
        if "=" not in item:
            raise ValueError("identity maps use layer=name entries separated by commas")
        layer, name = (part.strip() for part in item.split("=", 1))
        if layer not in LAYERS or not name or layer in result:
            raise ValueError(f"invalid or duplicate identity mapping entry: {item!r}")
        result[layer] = name
    return result


def delta_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("name")): {key: row.get(key) for key in ("baseline", "candidate", "delta")}
            for row in rows if row.get("name") is not None}


def load_ptx(path: Path) -> dict[str, Any]:
    data = load_json(path)
    if data.get("tool") != "ptx-inspector" or data.get("mode") != "compare" or data.get("schema_version") != 1:
        raise ValueError(f"unsupported PTX comparison schema: {path}")
    comparison = data.get("comparison", {})
    selected = comparison.get("kernel", {})
    baseline, candidate = selected.get("baseline", {}), selected.get("candidate", {})
    return {
        "status": "available", "source": artifact(path),
        "input_artifacts": {side: data.get("baseline" if side == "reference" else "candidate", {}).get("file")
                            for side in SIDES},
        "target": data.get("baseline", {}).get("analysis", {}).get("target"),
        "ptx_version": data.get("baseline", {}).get("analysis", {}).get("ptx_version"),
        "address_size": data.get("baseline", {}).get("analysis", {}).get("address_size"),
        "text_comparison": {"raw": comparison.get("raw"), "normalized": comparison.get("normalized")},
        "selected_scope": selected.get("scope"),
        "reference": summarize_ptx_symbol(baseline),
        "candidate": summarize_ptx_symbol(candidate),
        "category_deltas": delta_map(selected.get("instruction_categories", [])),
        "mnemonic_deltas": delta_map(selected.get("instruction_mnemonics", [])),
        "register_deltas": delta_map(selected.get("ptx_declared_registers", [])),
        "compatibility": comparison.get("compatibility", {}),
    }


def summarize_ptx_symbol(symbol: dict[str, Any]) -> dict[str, Any]:
    mnemonics = symbol.get("instruction_mnemonics", {})
    arithmetic = {name: count for name, count in sorted(mnemonics.items())
                  if str(name).split(".")[0] in {"add", "sub", "mul", "mad", "fma", "div"}}
    return {
        "name": symbol.get("name"), "kind": symbol.get("kind"),
        "instruction_count": symbol.get("instruction_count"),
        "instruction_categories": symbol.get("instruction_categories", {}),
        "arithmetic_operations": arithmetic,
        "ptx_declared_registers_by_type": symbol.get("ptx_declared_registers_by_type", {}),
        "ptx_declared_static_shared_bytes": symbol.get("ptx_declared_static_shared_bytes"),
        "has_dynamic_shared_memory": symbol.get("has_dynamic_shared_memory"),
    }


def load_sass(path: Path) -> dict[str, Any]:
    data = load_json(path)
    if data.get("tool") != "sass-inspector" or data.get("mode") != "compare" or data.get("schema_version") != 1:
        raise ValueError(f"unsupported SASS comparison schema: {path}")
    comparison = data.get("comparison", {})
    selected = comparison.get("function", {})
    return {
        "status": "available", "source": artifact(path),
        "input_artifacts": {side: data.get("baseline" if side == "reference" else "candidate", {}).get("file")
                            for side in SIDES},
        "target": data.get("baseline", {}).get("analysis", {}).get("target"),
        "boundary_ambiguities": {
            side: data.get("baseline" if side == "reference" else "candidate", {}).get("analysis", {}).get("ambiguities", [])
            for side in SIDES
        },
        "text_comparison": {"raw": comparison.get("raw"),
                            "normalized": comparison.get("module_normalized")},
        "selected_scope": selected.get("scope"),
        "selected_normalized_comparison": selected.get("normalized"),
        "reference": summarize_sass_function(selected.get("baseline", {})),
        "candidate": summarize_sass_function(selected.get("candidate", {})),
        "category_deltas": delta_map(selected.get("instruction_categories", [])),
        "mnemonic_deltas": delta_map(selected.get("instruction_mnemonics", [])),
        "base_operation_deltas": delta_map(selected.get("base_operations", [])),
    }


def summarize_sass_function(function: dict[str, Any]) -> dict[str, Any]:
    operations = function.get("base_operations", {})
    return {
        "name": function.get("name"), "role": function.get("role"),
        "instruction_count": function.get("instruction_count"),
        "instruction_categories": function.get("instruction_categories", {}),
        "arithmetic_operations": {name: operations.get(name, 0) for name in ("FFMA", "FMUL", "FADD")},
        "allocated_registers_per_thread": function.get("allocated_registers_per_thread"),
        "helper_calls": function.get("calls", []),
    }


def benchmark_file_records(directory: Path) -> dict[str, Any]:
    records = {}
    for name in REQUIRED_BENCHMARK_FILES + OPTIONAL_BENCHMARK_FILES:
        path = directory / name
        records[name] = artifact(path) if path.is_file() else {"path": str(path), "status": "absent"}
    return records


def load_benchmark(directory: Path, mapped_reference: str | None,
                   mapped_candidate: str | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    findings: list[dict[str, Any]] = []
    if not directory.is_dir():
        raise ValueError(f"benchmark input is not a directory: {directory}")
    files = benchmark_file_records(directory)
    missing = [name for name in REQUIRED_BENCHMARK_FILES if files[name].get("status") == "absent"]
    if missing:
        findings.append(finding("incompatible", "benchmark_required_files_missing",
                                "Benchmark data is incomplete; required files are missing.", missing))
    metadata = load_json(directory / "metadata.json") if "metadata.json" not in missing else {}
    summary = load_csv(directory / "summary.csv") if "summary.csv" not in missing else []
    config = metadata.get("config", {}) if isinstance(metadata.get("config"), dict) else {}
    reference = mapped_reference or config.get("reference")
    candidate = mapped_candidate or config.get("candidate")
    names = sorted({row.get("kernel", "") for row in summary if row.get("kernel")})
    if not reference or not candidate:
        findings.append(finding("incompatible", "benchmark_identity_ambiguous",
                                "Benchmark reference/candidate could not be selected.", names))
    elif reference not in names or candidate not in names:
        findings.append(finding("incompatible", "benchmark_identity_not_found",
                                "Mapped benchmark implementations are absent from summary.csv.",
                                {"reference": reference, "candidate": candidate, "available": names}))
    selected = [row for row in summary if row.get("kernel") in {reference, candidate}]
    rows = [benchmark_row(row) for row in selected]
    pairwise = load_csv(directory / "pairwise.csv") if files["pairwise.csv"].get("status") != "absent" else []
    pairs = [benchmark_pair(row) for row in pairwise
             if row.get("reference_kernel") == reference and row.get("candidate_kernel") == candidate]
    stats = load_csv(directory / "timing_statistics.csv") if files["timing_statistics.csv"].get("status") != "absent" else []
    overall_stats = {row.get("row_id"): row for row in stats if row.get("scope") == "overall"}
    for row in rows:
        stat = overall_stats.get(str(row.get("row_id")))
        if stat:
            row["measurement_count"] = integer(stat.get("measurement_count"))
            row["warmup_count"] = integer(stat.get("warmup_count"))
            row["trial_count"] = integer(stat.get("trial_count"))
    completion = metadata.get("status") or ("incomplete" if missing else "unavailable")
    completeness = "incomplete" if missing else "complete"
    tolerances = [row["tolerance_pass"] for row in rows if row.get("kernel") == candidate]
    if not rows or not tolerances or any(value is None for value in tolerances):
        correctness = "unavailable"
    elif all(tolerances):
        correctness = "passed"
    else:
        correctness = "failed"
    if completion == "tolerance_failed" or correctness == "failed":
        findings.append(finding("warning", "benchmark_numerical_failure",
                                "The candidate fails numerical acceptance for at least one reported configuration."))
    if metadata.get("git_dirty") or metadata.get("build_git_dirty") or metadata.get("runtime_git_dirty"):
        findings.append(finding("warning", "benchmark_dirty_tree",
                                "The benchmark records a dirty source tree."))
    unavailable = sorted({key for row in rows for key in ("median_ms", "reported_gflops", "median_ci_low_ms",
                                                           "median_ci_high_ms") if row.get(key) is None})
    if unavailable:
        findings.append(finding("informational", "benchmark_metrics_unavailable",
                                "Some benchmark metrics are unavailable.", unavailable))
    return {
        "status": "available" if not missing else "incomplete",
        "directory": str(directory), "files": files,
        "completion_status": completion, "data_completeness": completeness,
        "correctness_status": correctness,
        "reference_implementation": reference, "candidate_implementation": candidate,
        "available_implementations": names, "rows": rows, "pairwise": pairs,
        "provenance": benchmark_provenance(metadata), "methodology": benchmark_methodology(metadata),
    }, findings


def benchmark_row(row: dict[str, str]) -> dict[str, Any]:
    return {
        "row_id": integer(row.get("row_id")), "M": integer(row.get("M")),
        "N": integer(row.get("N")), "K": integer(row.get("K")),
        "kernel": row.get("kernel"), "backend": row.get("backend"),
        "timing_mode": row.get("timing_mode"), "median_ms": number(row.get("median_ms")),
        "reported_gflops": number(row.get("gflops")),
        "median_ci_low_ms": number(row.get("median_ci_low_ms")),
        "median_ci_high_ms": number(row.get("median_ci_high_ms")),
        "median_speedup": number(row.get("median_speedup")),
        "stability_warnings": [item for item in row.get("stability_warnings", "").split(";") if item],
        "bitwise_equal": boolean(row.get("bitwise_equal")),
        "divergent_count": integer(row.get("divergent_count")),
        "tolerance_pass": boolean(row.get("tolerance_pass")),
        "tolerance_failures": integer(row.get("tolerance_failures")),
        "max_ulp": integer(row.get("max_ulp")),
        "mean_divergent_ulp": number(row.get("mean_divergent_ulp")),
        "max_absolute_error": number(row.get("max_absolute_error")),
        "max_relative_error": number(row.get("max_relative_error")),
        "measurement_count": None, "warmup_count": None, "trial_count": None,
    }


def benchmark_pair(row: dict[str, str]) -> dict[str, Any]:
    return {
        "M": integer(row.get("M")), "N": integer(row.get("N")), "K": integer(row.get("K")),
        "reference_median_ms": number(row.get("reference_median_ms")),
        "candidate_median_ms": number(row.get("candidate_median_ms")),
        "candidate_reference_latency_ratio": number(row.get("latency_ratio")),
        "percent_difference": number(row.get("percent_difference")),
        "median_based_speedup": number(row.get("speedup")),
        "median_ci_overlap": boolean(row.get("median_ci_overlap")),
        "tolerance_pass": boolean(row.get("tolerance_pass")),
        "reference_timing_mode": row.get("reference_timing_mode"),
        "candidate_timing_mode": row.get("candidate_timing_mode"),
    }


def benchmark_provenance(meta: dict[str, Any]) -> dict[str, Any]:
    return {key: meta.get(key) for key in (
        "timestamp_utc", "git_commit", "git_dirty", "runtime_git_commit", "runtime_git_dirty",
        "build_git_dirty", "gpu_name", "gpu_compute_capability", "cuda_compiler",
        "cuda_runtime_version", "cuda_architectures", "cuda_flags", "build_type",
        "fp_verification_library_sha256", "fp_verified", "fp_verification_method")}


def benchmark_methodology(meta: dict[str, Any]) -> dict[str, Any]:
    config = meta.get("config", {}) if isinstance(meta.get("config"), dict) else {}
    return {
        "timing_methodology": meta.get("timing_methodology"), "trial_method": meta.get("trial_method"),
        "warmup_method": meta.get("warmup_method"), "primary_latency_statistic": meta.get("primary_latency_statistic"),
        "bootstrap_method": meta.get("bootstrap_method"), "dtype": config.get("dtype"),
        "input": config.get("input"), "generator": config.get("generator"),
        "seed": config.get("seed"), "seed_b": config.get("seed_b"),
        "warmups": config.get("warmups"), "iterations": config.get("iterations"),
        "trials": config.get("trials"), "atol": config.get("atol"), "rtol": config.get("rtol"),
    }


def identity_mapping(reference: dict[str, str], candidate: dict[str, str],
                     evidence: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    present = [layer for layer in LAYERS if evidence.get(layer, {}).get("status") in ("available", "incomplete")]
    findings = []
    if len(present) > 1:
        for side, mapping in (("reference", reference), ("candidate", candidate)):
            missing = [layer for layer in present if layer not in mapping]
            if missing:
                raise ValueError(f"--{side}-map must explicitly map all present layers: missing {missing}")
    actual = {
        "reference": {layer: evidence[layer].get("reference", {}).get("name") if layer != "benchmark"
                      else evidence[layer].get("reference_implementation") for layer in present},
        "candidate": {layer: evidence[layer].get("candidate", {}).get("name") if layer != "benchmark"
                      else evidence[layer].get("candidate_implementation") for layer in present},
    }
    requested = {"reference": reference, "candidate": candidate}
    for side in SIDES:
        for layer, expected in requested[side].items():
            if layer not in present:
                raise ValueError(f"{side} mapping names absent layer {layer!r}")
            if actual[side].get(layer) != expected:
                raise ValueError(f"{side} {layer} mapping {expected!r} does not match selected artifact identity {actual[side].get(layer)!r}")
    status = "explicit" if len(present) > 1 else "not_required"
    if status == "explicit":
        findings.append(finding("informational", "identity_mapping_explicit",
                                "Cross-layer reference/candidate correspondence was supplied by the user; it is not build verification."))
    return {"status": status, "present_layers": present, "reference": actual["reference"],
            "candidate": actual["candidate"], "meaning": "intended correspondence only; not compilation provenance"}, findings


def architecture_key(value: Any) -> str | None:
    if value is None:
        return None
    digits = "".join(re.findall(r"\d", str(value)))
    return digits or None


def strong_identifiers(layer: str, evidence: dict[str, Any]) -> dict[str, str]:
    # Inspector input hashes describe text files, not linked binaries, and are intentionally excluded.
    if layer != "benchmark":
        return {}
    value = evidence.get("provenance", {}).get("fp_verification_library_sha256")
    return {"binary_sha256": value} if isinstance(value, str) and value else {}


def compilation_provenance(evidence: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    present = [layer for layer in LAYERS if evidence.get(layer, {}).get("status") in ("available", "incomplete")]
    identifiers = {layer: strong_identifiers(layer, evidence[layer]) for layer in present}
    relationships = []
    findings: list[dict[str, Any]] = []
    targets = {
        "ptx": evidence.get("ptx", {}).get("target"), "sass": evidence.get("sass", {}).get("target"),
        "benchmark": evidence.get("benchmark", {}).get("provenance", {}).get("gpu_compute_capability"),
    }
    for index, left in enumerate(present):
        for right in present[index + 1:]:
            common = sorted(set(identifiers[left]) & set(identifiers[right]))
            matching = [key for key in common if identifiers[left][key] == identifiers[right][key]]
            conflicting = [key for key in common if identifiers[left][key] != identifiers[right][key]]
            arch_left, arch_right = architecture_key(targets.get(left)), architecture_key(targets.get(right))
            architecture_compatible = None if not arch_left or not arch_right else arch_left == arch_right
            if conflicting or architecture_compatible is False:
                status = "incompatible"
                reason = "Contradictory strong identifiers or compilation targets were recorded."
                severity = "incompatible"
            elif matching:
                status = "verified"
                reason = "A matching strong build or binary identifier links these artifacts."
                severity = "informational"
            else:
                status = "unverified"
                reason = "No matching build or binary identifier links these artifacts."
                severity = "warning"
            record = {"layers": [left, right], "status": status, "reason": reason,
                      "matching_identifiers": matching, "conflicting_identifiers": conflicting,
                      "architecture_compatible": architecture_compatible,
                      "architecture_values": {left: targets.get(left), right: targets.get(right)}}
            relationships.append(record)
            findings.append(finding(severity, f"compilation_identity_{status}",
                                    f"{left.upper()} ↔ {right.upper()}: {reason}", record))
    overall = "absent" if len(present) < 2 else (
        "incompatible" if any(item["status"] == "incompatible" for item in relationships) else
        "verified" if relationships and all(item["status"] == "verified" for item in relationships) else "unverified")
    return {"status": overall, "relationships": relationships, "strong_identifiers": identifiers,
            "meaning": "build linkage based only on strong artifact identifiers; mappings and matching names are excluded"}, findings


def absent(layer: str) -> dict[str, Any]:
    return {"status": "absent", "reason": f"No {layer.upper()} artifact was supplied."}


def build_experiment(args: argparse.Namespace) -> dict[str, Any]:
    if not any((args.ptx_comparison, args.sass_comparison, args.benchmark)):
        raise ValueError("at least one evidence input is required")
    reference_map, candidate_map = parse_mapping(args.reference_map), parse_mapping(args.candidate_map)
    evidence: dict[str, Any] = {layer: absent(layer) for layer in LAYERS}
    findings: list[dict[str, Any]] = []
    if args.ptx_comparison:
        evidence["ptx"] = load_ptx(args.ptx_comparison)
    if args.sass_comparison:
        evidence["sass"] = load_sass(args.sass_comparison)
    if args.benchmark:
        evidence["benchmark"], benchmark_findings = load_benchmark(
            args.benchmark, reference_map.get("benchmark"), candidate_map.get("benchmark"))
        findings.extend(benchmark_findings)
    mapping, mapping_findings = identity_mapping(reference_map, candidate_map, evidence)
    provenance, provenance_findings = compilation_provenance(evidence)
    findings.extend(mapping_findings + provenance_findings)
    severity_order = {"incompatible": 0, "warning": 1, "informational": 2}
    findings.sort(key=lambda item: (severity_order[item["severity"]], item["code"], item["message"]))
    return {"schema_version": SCHEMA_VERSION, "tool": "compiler-performance-report",
            "identity_mapping": mapping, "compilation_provenance": provenance,
            "evidence": evidence, "findings": findings,
            "interpretation_limits": [
                "PTX and SASS static instructions are not dynamic execution counts.",
                "PTX virtual registers are distinct from SASS physical register allocation.",
                "PTX instructions do not map one-to-one to SASS instructions.",
                "Static evidence does not establish scheduling, occupancy, stalls, bandwidth, or performance causality.",
                "Bootstrap confidence-interval overlap is descriptive, not a formal hypothesis test.",
            ]}


def display(value: Any, digits: int = 6) -> str:
    if value is None or value == "":
        return "unavailable"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def text_result(value: dict[str, Any] | None) -> str:
    if not value:
        return "unavailable"
    return "identical" if value.get("identical") else "different"


def counts_text(values: dict[str, Any]) -> str:
    return ", ".join(f"`{name}`={count}" for name, count in sorted(values.items())) or "none"


def render_report(result: dict[str, Any]) -> str:
    evidence, mapping, provenance = result["evidence"], result["identity_mapping"], result["compilation_provenance"]
    lines = ["# Unified compiler-to-performance experiment report", "", "## 1. Experiment overview", "",
             f"- Evidence layers: {', '.join(mapping['present_layers']) or 'none'}",
             f"- Cross-layer identity mapping: **{mapping['status']}**",
             f"- Shared compilation provenance: **{provenance['status']}**", ""]
    benchmark = evidence["benchmark"]
    if benchmark["status"] != "absent":
        lines += [f"- Benchmark completion status: **{benchmark['completion_status']}**",
                  f"- Benchmark data completeness: **{benchmark['data_completeness']}**",
                  f"- Benchmark correctness status: **{benchmark['correctness_status']}**", ""]
        captured = benchmark["provenance"]
        lines += ["| Benchmark provenance | Recorded value |", "|---|---|",
                  f"| Execution timestamp | {display(captured.get('timestamp_utc'))} |",
                  f"| Git commit | `{display(captured.get('git_commit'))}` |",
                  f"| Dirty tree | {display(captured.get('git_dirty'))} |",
                  f"| GPU | {display(captured.get('gpu_name'))} |",
                  f"| Compute capability | {display(captured.get('gpu_compute_capability'))} |",
                  f"| CUDA compiler | {display(captured.get('cuda_compiler'))} |",
                  f"| CUDA architecture | {display(captured.get('cuda_architectures'))} |",
                  f"| CUDA flags | `{display(captured.get('cuda_flags'))}` |", ""]
    lines += ["Identity mapping records the user's intended reference/candidate correspondence. It does not verify that compiler artifacts and the benchmark executable came from the same compilation.", "",
              "## 2. Evidence and provenance status", "", "| Severity | Finding |", "|---|---|"]
    lines += [f"| {item['severity']} | {item['message']} |" for item in result["findings"]] or ["| informational | No findings. |"]
    lines += ["", "## 3. Kernel identity mapping", "", "| Side | PTX | SASS | Benchmark |", "|---|---|---|---|"]
    for side in SIDES:
        values = mapping[side]
        lines.append(f"| {side} | `{display(values.get('ptx'))}` | `{display(values.get('sass'))}` | `{display(values.get('benchmark'))}` |")
    lines += ["", "These mappings establish correspondence only. Compilation linkage is evaluated separately:", ""]
    if provenance["relationships"]:
        lines += ["| Layers | Status | Basis |", "|---|---|---|"]
        for item in provenance["relationships"]:
            lines.append(f"| {' ↔ '.join(item['layers'])} | {item['status']} | {item['reason']} |")
    else:
        lines.append("Only one evidence layer is present; cross-layer compilation linkage is absent.")
    lines += [""]
    render_ptx(lines, evidence["ptx"])
    render_sass(lines, evidence["sass"])
    render_benchmark(lines, benchmark)
    lines += ["## 8. Reliability and experimental limitations", ""]
    for item in result["interpretation_limits"]:
        lines.append(f"- {item}")
    lines += [""]
    if benchmark["status"] != "absent":
        method = benchmark["methodology"]
        lines += [f"- Timing boundaries: {display(method.get('timing_methodology'))}",
                  f"- Trial/order method: {display(method.get('trial_method'))}", ""]
    lines += ["## 9. Conclusions supported by the evidence", ""]
    if evidence["sass"]["status"] == "available":
        ref, cand = evidence["sass"]["reference"], evidence["sass"]["candidate"]
        lines.append(f"- In the selected SASS functions, `{ref['name']}` has {ref['arithmetic_operations']['FFMA']} static FFMA instructions; `{cand['name']}` has {cand['arithmetic_operations']['FMUL']} FMUL and {cand['arithmetic_operations']['FADD']} FADD instructions.")
    if benchmark["status"] != "absent":
        if benchmark["correctness_status"] == "failed":
            lines.append("- The benchmark candidate fails numerical acceptance for at least one shape; it must not be presented as a successful optimization.")
        lines.append("- Recorded latency differences describe this benchmark capture only and vary by configuration.")
    if provenance["status"] != "verified" and len(mapping["present_layers"]) > 1:
        lines.append("- Compiler and benchmark observations are reported as separate artifacts because shared compilation identity is not verified.")
    lines += ["- Static instruction differences do not establish the cause of measured performance differences.", "",
              "## 10. Source artifact references and hashes", "", "| Artifact | Path | SHA-256 |", "|---|---|---|"]
    for layer in LAYERS:
        item = evidence[layer]
        if item.get("source"):
            lines.append(f"| {layer} comparison | `{item['source']['path']}` | `{item['source']['sha256']}` |")
        for side, record in item.get("input_artifacts", {}).items():
            if isinstance(record, dict) and record.get("sha256"):
                lines.append(f"| {layer} {side} input | `{display(record.get('path'))}` | `{record['sha256']}` |")
        for name, record in item.get("files", {}).items():
            if record.get("sha256"):
                lines.append(f"| benchmark `{name}` | `{record['path']}` | `{record['sha256']}` |")
    lines.append("")
    return "\n".join(lines)


def render_ptx(lines: list[str], ptx: dict[str, Any]) -> None:
    lines += ["## 4. PTX structural comparison", ""]
    if ptx["status"] == "absent":
        lines += ["PTX evidence was not supplied.", ""]
        return
    lines += [f"Full-module raw text is **{text_result(ptx['text_comparison']['raw'])}**; normalized full-module text is **{text_result(ptx['text_comparison']['normalized'])}**. Selected statistics use scope `{ptx['selected_scope']}`.", "",
              "| Metric | Reference | Candidate |", "|---|---:|---:|",
              f"| Symbol | `{ptx['reference']['name']}` | `{ptx['candidate']['name']}` |",
              f"| Static PTX instructions | {display(ptx['reference']['instruction_count'])} | {display(ptx['candidate']['instruction_count'])} |",
              f"| Arithmetic mnemonics | {counts_text(ptx['reference']['arithmetic_operations'])} | {counts_text(ptx['candidate']['arithmetic_operations'])} |",
              f"| Declared virtual registers | {sum(ptx['reference']['ptx_declared_registers_by_type'].values())} | {sum(ptx['candidate']['ptx_declared_registers_by_type'].values())} |", "",
              "These register counts are PTX declarations, not physical allocation or occupancy data.", ""]


def render_sass(lines: list[str], sass: dict[str, Any]) -> None:
    lines += ["## 5. SASS machine-code comparison", ""]
    if sass["status"] == "absent":
        lines += ["SASS evidence was not supplied.", ""]
        return
    ref, cand = sass["reference"], sass["candidate"]
    ref_targets = ", ".join(f"`{call.get('target') or 'unresolved'}`" for call in ref["helper_calls"]) or "none"
    cand_targets = ", ".join(f"`{call.get('target') or 'unresolved'}`" for call in cand["helper_calls"]) or "none"
    lines += [f"Full-file raw text is **{text_result(sass['text_comparison']['raw'])}**; normalized full-file text is **{text_result(sass['text_comparison']['normalized'])}**. Selected statistics use scope `{sass['selected_scope']}`.", "",
              "| Metric | Reference | Candidate |", "|---|---:|---:|",
              f"| Function | `{ref['name']}` | `{cand['name']}` |",
              f"| Static SASS instructions | {display(ref['instruction_count'])} | {display(cand['instruction_count'])} |",
              f"| FFMA | {ref['arithmetic_operations']['FFMA']} | {cand['arithmetic_operations']['FFMA']} |",
              f"| FMUL | {ref['arithmetic_operations']['FMUL']} | {cand['arithmetic_operations']['FMUL']} |",
              f"| FADD | {ref['arithmetic_operations']['FADD']} | {cand['arithmetic_operations']['FADD']} |",
              f"| Physical registers/thread | {display(ref['allocated_registers_per_thread'])} | {display(cand['allocated_registers_per_thread'])} |",
              f"| Helper calls | {len(ref['helper_calls'])} | {len(cand['helper_calls'])} |", "",
              f"Helper targets: reference {ref_targets}; candidate {cand_targets}.", "",
              "Counts are static selected-function observations. Called helper bodies and dynamic execution counts are excluded.", ""]


def render_benchmark(lines: list[str], benchmark: dict[str, Any]) -> None:
    lines += ["## 6. GPU benchmark measurements", ""]
    if benchmark["status"] == "absent":
        lines += ["Benchmark evidence was not supplied.", "", "## 7. Numerical correctness", "", "Numerical evidence was not supplied.", ""]
        return
    lines += [f"Completion: **{benchmark['completion_status']}**. Data completeness: **{benchmark['data_completeness']}**. Correctness: **{benchmark['correctness_status']}**.", ""]
    if benchmark["completion_status"] != "complete" or benchmark["correctness_status"] != "passed":
        lines += ["**This run is not presented as a successful optimization.**", ""]
    lines += ["| Shape | Reference median ms | Candidate median ms | Candidate/reference | Median speedup | Numerical acceptance |", "|---|---:|---:|---:|---:|---|"]
    for pair in benchmark["pairwise"]:
        shape = f"{pair['M']}×{pair['N']}×{pair['K']}"
        lines.append(f"| {shape} | {display(pair['reference_median_ms'])} | {display(pair['candidate_median_ms'])} | {display(pair['candidate_reference_latency_ratio'])} | {display(pair['median_based_speedup'])}× | {'PASS' if pair['tolerance_pass'] else 'FAIL' if pair['tolerance_pass'] is False else 'unavailable'} |")
    if not benchmark["pairwise"]:
        lines.append("| — | unavailable | unavailable | unavailable | unavailable | unavailable |")
    lines += ["", "All listed timings retain their recorded timing modes; no mean-based speedup is substituted for the median-based pairwise speedup.", "",
              "### Recorded timing detail", "",
              "| Shape / implementation | Mode | Median ms (CI) | Reported GFLOP/s | Samples / trials | Reliability flags |",
              "|---|---|---:|---:|---:|---|"]
    for row in benchmark["rows"]:
        shape = f"{row['M']}×{row['N']}×{row['K']} / `{row['kernel']}`"
        interval = (f"{display(row['median_ms'])} ({display(row['median_ci_low_ms'])}–"
                    f"{display(row['median_ci_high_ms'])})")
        counts = f"{display(row['measurement_count'])} / {display(row['trial_count'])}"
        flags = "; ".join(row["stability_warnings"]) or "none"
        lines.append(f"| {shape} | {display(row['timing_mode'])} | {interval} | {display(row['reported_gflops'])} | {counts} | {flags} |")
    if not benchmark["rows"]:
        lines.append("| — | unavailable | unavailable | unavailable | unavailable | unavailable |")
    lines += ["", "GFLOP/s is preserved exactly as reported by `summary.csv`; it is not relabeled as median-derived throughput.", "",
              "## 7. Numerical correctness", "", "| Shape | Candidate | Bitwise equal | Divergent | Tolerance failures | Max abs error | Max rel error |", "|---|---|---|---:|---:|---:|---:|"]
    candidate = benchmark.get("candidate_implementation")
    for row in benchmark["rows"]:
        if row["kernel"] != candidate:
            continue
        lines.append(f"| {row['M']}×{row['N']}×{row['K']} | {'PASS' if row['tolerance_pass'] else 'FAIL' if row['tolerance_pass'] is False else 'unavailable'} | {display(row['bitwise_equal'])} | {display(row['divergent_count'])} | {display(row['tolerance_failures'])} | {display(row['max_absolute_error'])} | {display(row['max_relative_error'])} |")
    lines.append("")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ptx-comparison", type=Path)
    parser.add_argument("--sass-comparison", type=Path)
    parser.add_argument("--benchmark", type=Path)
    parser.add_argument("--reference-map", help="Explicit layer=name pairs, e.g. ptx=k,sass=k,benchmark=impl")
    parser.add_argument("--candidate-map", help="Explicit layer=name pairs, e.g. ptx=k,sass=k,benchmark=impl")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def run(arguments: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(arguments)
    if args.output.exists():
        raise FileExistsError(f"output path already exists: {args.output}")
    result = build_experiment(args)
    args.output.mkdir(parents=True)
    (args.output / "experiment.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    (args.output / "report.md").write_text(render_report(result), encoding="utf-8")
    return result


def main(arguments: Sequence[str] | None = None) -> int:
    try:
        result = run(arguments)
        benchmark = result["evidence"]["benchmark"]
        print(f"compiler-performance report: provenance={result['compilation_provenance']['status']} "
              f"benchmark={benchmark.get('completion_status', 'absent')} correctness={benchmark.get('correctness_status', 'absent')}")
        return 0
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        print(f"compiler-performance-report: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
