#!/usr/bin/env python3
"""Compare two saved NKI compiler-artifact directories without NKI or hardware."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Sequence


SCHEMA_VERSION = 1
MLIR_NAME = "module.mlir"
NEFF_NAME = "kernel.neff"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_record(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"status": "missing"}
    data = path.read_bytes()
    return {"status": "present", "path": str(path), "size_bytes": len(data),
            "sha256": sha256_bytes(data)}


def _skip_quoted(text: str, start: int) -> int:
    quote = text[start]
    index = start + 1
    while index < len(text):
        if text[index] == "\\":
            index += 2
        elif text[index] == quote:
            return index + 1
        else:
            index += 1
    return index


def strip_mlir_locations(text: str) -> str:
    """Remove balanced MLIR loc(...) annotations while preserving other text."""
    output: list[str] = []
    index = 0
    while index < len(text):
        match = re.match(r"loc\s*\(", text[index:])
        if match and (index == 0 or not (text[index - 1].isalnum() or text[index - 1] == "_")):
            depth = 1
            index += match.end()
            while index < len(text) and depth:
                if text[index] in "\"'":
                    index = _skip_quoted(text, index)
                    continue
                if text[index] == "(":
                    depth += 1
                elif text[index] == ")":
                    depth -= 1
                index += 1
            if depth:
                raise ValueError("unterminated MLIR loc(...) expression")
            while index < len(text) and text[index] in " \t":
                index += 1
            continue
        output.append(text[index])
        index += 1
    return "".join(output)


def normalize_mlir(text: str) -> str:
    stripped = strip_mlir_locations(text)
    entry = re.search(r"\bfunc\.func\s+@(?P<name>[A-Za-z0-9_.$-]+)", stripped)
    if entry:
        name = entry.group("name")
        stripped = re.sub(r"@" + re.escape(name) + r"(?=[^A-Za-z0-9_.$-]|$)",
                          "@__nki_entry", stripped)
    lines = [line.rstrip() for line in stripped.splitlines()]
    return "\n".join(lines).strip() + "\n"


def diff_stats(left: str, right: str) -> dict[str, int | bool]:
    diff = list(difflib.unified_diff(left.splitlines(), right.splitlines(), lineterm=""))
    additions = sum(line.startswith("+") and not line.startswith("+++") for line in diff)
    deletions = sum(line.startswith("-") and not line.startswith("---") for line in diff)
    return {"identical": left == right, "changed_lines": additions + deletions,
            "additions": additions, "deletions": deletions}


def unified_diff(left: str, right: str, limit: int) -> tuple[str, bool, int]:
    lines = list(difflib.unified_diff(left.splitlines(), right.splitlines(),
                                      fromfile="baseline", tofile="candidate", lineterm=""))
    truncated = len(lines) > limit
    shown = lines[:limit]
    if truncated:
        shown.append(f"... diff truncated; {len(lines) - limit} lines omitted ...")
    return "\n".join(shown) + ("\n" if shown else ""), truncated, len(lines)


def _table_rows(text: str) -> list[list[str]]:
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not (line.startswith("│") and line.endswith("│")):
            continue
        cells = [cell.strip() for cell in line[1:-1].split("│")]
        if cells:
            rows.append(cells)
    return rows


def parse_instruction_stats(text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for cells in _table_rows(text):
        if len(cells) != 2 or cells[0].lower() == "opcode":
            continue
        try:
            counts[cells[0]] = int(cells[1].replace(",", ""))
        except ValueError:
            continue
    if not counts:
        raise ValueError("no opcode rows found in instruction statistics")
    return counts


def parse_dma_stats(text: str) -> dict[str, Any]:
    sections: dict[str, list[dict[str, str]]] = {}
    heading = "unknown"
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        if stripped.endswith(":"):
            heading = stripped[:-1].lower().replace(" ", "_")
        if stripped.startswith("│"):
            table = []
            while index < len(lines) and (lines[index].strip().startswith(("│", "├", "└"))):
                if lines[index].strip().startswith("│"):
                    table.append([cell.strip() for cell in lines[index].strip()[1:-1].split("│")])
                index += 1
            if table:
                header, *body = table
                sections[heading] = [dict(zip(header, row)) for row in body if len(row) == len(header)]
            continue
        index += 1
    total = re.search(r"Total descriptors:\s*([0-9,]+)\s*\(([^)]+)\)", text)
    return {
        "sections": sections,
        "total_descriptors": int(total.group(1).replace(",", "")) if total else None,
        "total_size_reported": total.group(2) if total else None,
        "normalized_sha256": sha256_bytes(("\n".join(line.rstrip() for line in lines).strip() + "\n").encode()),
    }


def _choose_subgraph(root: Path, name: str, requested: str | None) -> Path | None:
    matches = sorted(root.glob(f"sg*/{name}"))
    if requested:
        path = root / requested / name
        return path if path.is_file() else None
    if len(matches) > 1:
        raise ValueError(f"{root}: multiple {name} files; select one with --subgraph")
    return matches[0] if matches else None


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON {path}: {error}") from error


def provenance(root: Path, mlir: str | None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    info = root / "info.json"
    if info.is_file():
        data = _load_json(info)
        result.update({key: data.get(key) for key in ("tool_version", "nki_version") if data.get(key) is not None})
    neff_json = root / "neff.json"
    if neff_json.is_file():
        data = _load_json(neff_json)
        attrs = data.get("attrs", {})
        result["tensor_shapes"] = attrs.get("shape", [None, None])[-1]
        result["tensor_dtypes"] = attrs.get("dltype", [None, None])[-1]
    kernel_info = root / "kernel_info.json"
    if kernel_info.is_file():
        data = _load_json(kernel_info)
        result["kernel_symbols"] = sorted(data.get("global", {}).get("kernels", {}))
        result["subgraph_count"] = data.get("global", {}).get("summary", {}).get("total_subgraphs")
    if mlir:
        target = re.search(r"nisa\.target\s*=\s*#nisa\.target<([^>]+)>", mlir)
        if target:
            result["target"] = target.group(1)
    return result


def read_benchmark(root: Path | None) -> dict[str, Any]:
    if root is None:
        return {"status": "not_requested"}
    matches = sorted(root.rglob("info.json"))
    candidates = []
    for path in matches:
        data = _load_json(path)
        if "latency" in data or "nc_latency" in data:
            candidates.append((path, data))
    if not candidates:
        return {"status": "missing", "path": str(root)}
    if len(candidates) != 1:
        raise ValueError(f"{root}: expected one neuron-bench info.json, found {len(candidates)}")
    path, data = candidates[0]
    return {"status": "present", "path": str(path), "run_date": data.get("run_date"),
            "instance_type": data.get("instance_type"), "nc_count": data.get("nc_node_count"),
            "software_info": data.get("software_info"), "nc_latency_us": data.get("nc_latency"),
            "runtime_latency_us": data.get("latency"), "throughput_inf_s": data.get("throughput")}


def collect(root: Path, subgraph: str | None) -> dict[str, Any]:
    if not root.is_dir():
        raise ValueError(f"artifact directory does not exist: {root}")
    mlir_path = root / MLIR_NAME
    mlir = mlir_path.read_text(encoding="utf-8") if mlir_path.is_file() else None
    instruction_path = _choose_subgraph(root, "instruction_stats.txt", subgraph)
    dma_path = _choose_subgraph(root, "dma_stats.txt", subgraph)
    result = {
        "root": str(root), "mlir_file": file_record(mlir_path if mlir is not None else None),
        "neff_file": file_record(root / NEFF_NAME if (root / NEFF_NAME).is_file() else None),
        "instruction_file": file_record(instruction_path), "dma_file": file_record(dma_path),
        "provenance": provenance(root, mlir),
    }
    if instruction_path:
        result["instruction_counts"] = parse_instruction_stats(instruction_path.read_text(encoding="utf-8"))
    if dma_path:
        result["dma_statistics"] = parse_dma_stats(dma_path.read_text(encoding="utf-8"))
    if mlir is not None:
        result["_mlir"] = mlir
        result["_normalized_mlir"] = normalize_mlir(mlir)
    return result


def mapping_delta(left: dict[str, int], right: dict[str, int]) -> list[dict[str, Any]]:
    rows = []
    for name in sorted(set(left) | set(right)):
        baseline, candidate = left.get(name, 0), right.get(name, 0)
        rows.append({"name": name, "baseline": baseline, "candidate": candidate,
                     "delta": candidate - baseline,
                     "percent_delta": ((candidate - baseline) / baseline * 100) if baseline else None})
    return rows


def compare(baseline: dict[str, Any], candidate: dict[str, Any], diff_limit: int) -> tuple[dict[str, Any], dict[str, str]]:
    generated: dict[str, str] = {}
    if "_mlir" in baseline and "_mlir" in candidate:
        raw_left, raw_right = baseline["_mlir"], candidate["_mlir"]
        norm_left, norm_right = baseline["_normalized_mlir"], candidate["_normalized_mlir"]
        diff, truncated, total = unified_diff(norm_left, norm_right, diff_limit)
        generated = {"baseline.normalized.mlir": norm_left, "candidate.normalized.mlir": norm_right,
                     "normalized.diff": diff}
        mlir_result: dict[str, Any] = {
            "status": "compared", "raw": {"status": "compared", **diff_stats(raw_left, raw_right)},
            "canonical": {"status": "compared", **diff_stats(norm_left, norm_right),
                          "baseline_sha256": sha256_bytes(norm_left.encode()),
                          "candidate_sha256": sha256_bytes(norm_right.encode()),
                          "diff_total_lines": total, "diff_truncated": truncated},
        }
    else:
        mlir_result = {"status": "unavailable"}
    left_counts = baseline.get("instruction_counts")
    right_counts = candidate.get("instruction_counts")
    instructions = ({"status": "compared", "identical": left_counts == right_counts,
                     "baseline_total": sum(left_counts.values()), "candidate_total": sum(right_counts.values()),
                     "opcodes": mapping_delta(left_counts, right_counts)}
                    if left_counts is not None and right_counts is not None else {"status": "unavailable"})
    left_dma, right_dma = baseline.get("dma_statistics"), candidate.get("dma_statistics")
    dma = ({"status": "compared", "identical": left_dma == right_dma,
            "baseline": left_dma, "candidate": right_dma}
           if left_dma is not None and right_dma is not None else {"status": "unavailable"})
    left_neff, right_neff = baseline["neff_file"], candidate["neff_file"]
    neff = ({"status": "compared", "byte_identical": left_neff["sha256"] == right_neff["sha256"],
             "baseline": left_neff, "candidate": right_neff}
            if left_neff["status"] == right_neff["status"] == "present" else {"status": "unavailable"})
    prov_left, prov_right = baseline["provenance"], candidate["provenance"]
    mismatches = {key: {"baseline": prov_left.get(key), "candidate": prov_right.get(key)}
                  for key in sorted(set(prov_left) | set(prov_right))
                  if prov_left.get(key) != prov_right.get(key) and key != "kernel_symbols"}
    return ({"mlir": mlir_result, "instructions": instructions, "dma": dma, "neff": neff,
             "provenance": {"compatible": not mismatches, "mismatches": mismatches}}, generated)


def _state(section: dict[str, Any], key: str = "identical") -> str:
    if section.get("status") != "compared":
        return "unavailable"
    return "identical" if section.get(key) else "different"


def render_report(result: dict[str, Any]) -> str:
    comp = result["comparison"]
    lines = ["# NKI compiler-artifact comparison", "",
             f"- Baseline: `{result['baseline']['root']}`",
             f"- Candidate: `{result['candidate']['root']}`",
             f"- Provenance compatible: **{str(comp['provenance']['compatible']).lower()}**", "",
             "## Evidence summary", "",
             "| Layer | Result |", "|---|---|",
             f"| Raw MLIR | {_state(comp['mlir'].get('raw', {})) if comp['mlir'].get('status') == 'compared' else 'unavailable'} |",
             f"| Canonical MLIR | {_state(comp['mlir'].get('canonical', {})) if comp['mlir'].get('status') == 'compared' else 'unavailable'} |",
             f"| Instruction counts | {_state(comp['instructions'])} |",
             f"| DMA statistics | {_state(comp['dma'])} |",
             f"| NEFF bytes | {_state(comp['neff'], 'byte_identical')} |", ""]
    if comp["mlir"].get("status") == "compared":
        raw = comp["mlir"]["raw"]
        canonical = comp["mlir"]["canonical"]
        lines += ["## MLIR comparison", "",
                  "| Representation | Result | Changed lines | Additions | Deletions |",
                  "|---|---|---:|---:|---:|",
                  f"| Raw | {_state(raw)} | {raw['changed_lines']} | {raw['additions']} | {raw['deletions']} |",
                  f"| Canonical | {_state(canonical)} | {canonical['changed_lines']} | {canonical['additions']} | {canonical['deletions']} |", ""]
    if comp["instructions"].get("status") == "compared":
        lines += ["## Instruction counts", "", "| Opcode | Baseline | Candidate | Delta |",
                  "|---|---:|---:|---:|"]
        lines += [f"| {row['name']} | {row['baseline']} | {row['candidate']} | {row['delta']:+d} |"
                  for row in comp["instructions"]["opcodes"]]
        lines.append("")
    if comp["dma"].get("status") == "compared":
        baseline_dma = comp["dma"]["baseline"]
        candidate_dma = comp["dma"]["candidate"]
        baseline_rows = sum(len(rows) for rows in baseline_dma["sections"].values())
        candidate_rows = sum(len(rows) for rows in candidate_dma["sections"].values())
        baseline_descriptor_rows = sum(
            len(rows) for name, rows in baseline_dma["sections"].items() if "descriptors" in name
        )
        candidate_descriptor_rows = sum(
            len(rows) for name, rows in candidate_dma["sections"].items() if "descriptors" in name
        )
        lines += ["## DMA comparison", "",
                  f"DMA equality: **{_state(comp['dma'])}**. This result compares the parsed section rows, "
                  "total descriptor fields, reported total size, and normalized full-text SHA-256.", "",
                  "| Metric | Baseline | Candidate |", "|---|---:|---:|",
                  f"| Parsed table rows | {baseline_rows} | {candidate_rows} |",
                  f"| Parsed descriptor rows | {baseline_descriptor_rows} | {candidate_descriptor_rows} |",
                  f"| Total descriptors | {baseline_dma['total_descriptors']} | {candidate_dma['total_descriptors']} |",
                  f"| Reported total size | {baseline_dma['total_size_reported']} | {candidate_dma['total_size_reported']} |",
                  f"| Normalized text SHA-256 | `{baseline_dma['normalized_sha256']}` | `{candidate_dma['normalized_sha256']}` |", ""]
        if (baseline_descriptor_rows == candidate_descriptor_rows == 0
                and baseline_dma["total_descriptors"] == candidate_dma["total_descriptors"] == 0):
            lines += ["Both DMA reports contained valid parsed descriptor tables with no descriptor data rows. Equality therefore means "
                      "that the empty descriptor summaries, zero totals, other parsed table rows, and normalized report text matched; "
                      "it is not a claim that no data movement occurred.", ""]
    if comp["neff"].get("status") == "compared":
        baseline_neff = comp["neff"]["baseline"]
        candidate_neff = comp["neff"]["candidate"]
        lines += ["## NEFF files", "", "| Capture | Size (bytes) | SHA-256 |",
                  "|---|---:|---|",
                  f"| Baseline | {baseline_neff['size_bytes']} | `{baseline_neff['sha256']}` |",
                  f"| Candidate | {candidate_neff['size_bytes']} | `{candidate_neff['sha256']}` |", ""]
    baseline_bench = result["baseline"]["benchmark"]
    candidate_bench = result["candidate"]["benchmark"]
    if baseline_bench.get("status") == "present" or candidate_bench.get("status") == "present":
        def percentile(record: dict[str, Any], field: str, value: str) -> Any:
            return (record.get(field) or {}).get(value, "unavailable")
        lines += ["## Optional benchmark evidence", "",
                  "| Metric | Baseline | Candidate |", "|---|---:|---:|",
                  f"| NeuronCore p50 (µs) | {percentile(baseline_bench, 'nc_latency_us', '50')} | {percentile(candidate_bench, 'nc_latency_us', '50')} |",
                  f"| NeuronCore p99 (µs) | {percentile(baseline_bench, 'nc_latency_us', '99')} | {percentile(candidate_bench, 'nc_latency_us', '99')} |",
                  f"| Runtime p50 (µs) | {percentile(baseline_bench, 'runtime_latency_us', '50')} | {percentile(candidate_bench, 'runtime_latency_us', '50')} |", ""]
    conclusions = []
    if comp["mlir"].get("status") == "compared":
        raw_identical = comp["mlir"]["raw"]["identical"]
        canonical_identical = comp["mlir"]["canonical"]["identical"]
        if raw_identical == canonical_identical:
            verb = "is identical" if raw_identical else "differs"
            conclusions.append(
                f"The raw MLIR {verb}, and the canonical MLIR {verb} under the documented normalization."
            )
        else:
            conclusions.append(
                f"The raw MLIR {_state(comp['mlir']['raw'])}, while the canonical MLIR is "
                f"{_state(comp['mlir']['canonical'])} under the documented normalization."
            )
    if comp["instructions"].get("status") == "compared" and comp["dma"].get("status") == "compared":
        conclusions.append(
            f"Aggregate instruction counts are {_state(comp['instructions'])}, and the captured DMA summaries are "
            f"{_state(comp['dma'])}; these observations do not establish identical execution schedules."
        )
    if comp["neff"].get("status") == "compared":
        conclusions.append(
            f"The NEFF files are {_state(comp['neff'], 'byte_identical')} at the byte level; this alone does not "
            "identify which executable instructions or metadata differ."
        )
    if not conclusions:
        conclusions.append("The available artifacts are insufficient for a cross-layer conclusion.")
    lines += ["## Conclusions", ""] + [f"- {conclusion}" for conclusion in conclusions] + ["",
              "## Interpretation limits", "",
              "Canonical MLIR equality applies only after removing source locations and normalizing the entry-point symbol. "
              "Aggregate opcode and DMA equality does not establish identical instruction ordering, scheduling, or critical paths. "
              "NEFF comparison establishes only byte identity or difference; it is not a machine-code disassembly. "
              "NeuronCore and runtime latency are separate measurement windows, and a single benchmark summary does not establish statistical significance.", ""]
    return "\n".join(lines)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("baseline", type=Path)
    result.add_argument("candidate", type=Path)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--subgraph", help="Subgraph directory such as sg00")
    result.add_argument("--baseline-benchmark", type=Path)
    result.add_argument("--candidate-benchmark", type=Path)
    result.add_argument("--diff-lines", type=int, default=400,
                        help="Maximum normalized unified-diff lines to save (default: 400)")
    return result


def run(arguments: Sequence[str] | None = None) -> dict[str, Any]:
    args = parser().parse_args(arguments)
    if args.diff_lines < 0:
        raise ValueError("--diff-lines must be nonnegative")
    if args.output.exists():
        raise FileExistsError(f"output path already exists: {args.output}")
    baseline = collect(args.baseline, args.subgraph)
    candidate = collect(args.candidate, args.subgraph)
    comparison, generated = compare(baseline, candidate, args.diff_lines)
    baseline["benchmark"] = read_benchmark(args.baseline_benchmark)
    candidate["benchmark"] = read_benchmark(args.candidate_benchmark)
    for capture in (baseline, candidate):
        capture.pop("_mlir", None)
        capture.pop("_normalized_mlir", None)
    result = {"schema_version": SCHEMA_VERSION, "tool": "nki-compiler-artifact-comparison",
              "baseline": baseline, "candidate": candidate, "comparison": comparison}
    args.output.mkdir(parents=True)
    mlir_dir = args.output / "mlir"
    if generated:
        mlir_dir.mkdir()
        for name, content in generated.items():
            (mlir_dir / name).write_text(content, encoding="utf-8")
    (args.output / "comparison.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output / "report.md").write_text(render_report(result), encoding="utf-8")
    return result


def main(arguments: Sequence[str] | None = None) -> int:
    try:
        result = run(arguments)
        comp = result["comparison"]
        canonical = comp["mlir"].get("canonical", {})
        print(f"Canonical MLIR: {_state(canonical)}")
        print(f"Instruction counts: {_state(comp['instructions'])}")
        print(f"DMA statistics: {_state(comp['dma'])}")
        print(f"NEFF bytes: {_state(comp['neff'], 'byte_identical')}")
        return 0
    except (OSError, ValueError) as error:
        print(f"nki-compare-artifacts: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
