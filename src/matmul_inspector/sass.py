#!/usr/bin/env python3
"""SASS inspection and comparison implementation."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, field
import difflib
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Sequence


SCHEMA_VERSION = 1
NORMALIZATION = (
    "Normalize line endings/trailing whitespace and omit hexadecimal instruction-address "
    "comments. Preserve function identities, labels, predicates, mnemonics, operands, "
    "constants, branch targets, and instruction order."
)


@dataclass
class SassInstruction:
    address: str
    address_value: int
    predicate: str | None
    mnemonic: str
    base: str
    modifiers: list[str]
    operands: str
    category: str
    call_target: str | None = None


@dataclass
class SassFunction:
    name: str
    section: str
    role: str
    instructions: list[SassInstruction] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    body_events: list[tuple[str, Any]] = field(default_factory=list)
    allocated_registers_per_thread: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SassModule:
    source: str
    target: str | None
    functions: list[SassFunction]
    ambiguities: list[dict[str, Any]]
    text_sections: list[str]


INSTRUCTION_RE = re.compile(
    r"^\s*/\*([0-9A-Fa-f]+)\*/\s+"
    r"(?:(@!?[A-Za-z0-9_]+)\s+)?"
    r"([A-Za-z][A-Za-z0-9_.]*)"
    r"(?:\s+(.*?))?\s*;\s*$"
)
FUNCTION_TYPE_RE = re.compile(
    r"(?m)^\s*\.type\s+([^,\s]+)\s*,\s*@function\s*$"
)
SECTION_RE = re.compile(r"(?m)^\s*\.section\s+([^,\s]+)(?:,[^\n]*)?$")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize_space(text: str) -> str:
    return " ".join(text.split())


def classify_instruction(mnemonic: str) -> str:
    base = mnemonic.upper().split(".")[0]
    if base.startswith(("HMMA", "IMMA", "MMA", "WGMMA")):
        return "tensor"
    if base.startswith(("FFMA", "FMUL", "FADD", "FSEL", "FCHK", "DADD", "DMUL", "DFMA")):
        return "floating_point_arithmetic"
    if base.startswith(("IMAD", "IADD", "UIADD", "UIMAD", "LEA", "LOP", "ULOP", "SHF", "USHF", "ISCADD")):
        return "integer_address_arithmetic"
    if base in {"LDG", "STG", "ATOM", "ATOMG", "RED", "REDG", "CCTL", "CCTLL"}:
        return "memory_global"
    if base in {"LDS", "STS", "ATOMS", "REDS"}:
        return "memory_shared"
    if base in {"LDL", "STL"}:
        return "memory_local"
    if base in {"LDC", "ULDC"}:
        return "memory_constant"
    if base.startswith(("LD", "ST")):
        return "memory_other"
    if base.startswith(("ISETP", "FSET", "DSETP", "PLOP", "UPLOP", "PSET", "CSET")):
        return "comparison_predicate"
    if base in {"BRA", "BRX", "JMP", "JMX", "BSSY", "BSYNC", "BREAK", "CONT", "SSY", "SYNC"}:
        return "control_flow"
    if base.startswith("CALL"):
        return "call"
    if base in {"RET", "EXIT", "KILL", "NANOSLEEP", "YIELD"}:
        return "exit_return"
    if base in {"BAR", "WARPSYNC", "MEMBAR", "DEPBAR", "ERRBAR", "CGAERRBAR"}:
        return "synchronization"
    if base.startswith(("F2I", "F2F", "I2F", "I2I", "I2IP", "FRND")):
        return "conversion"
    if base.startswith(("MUFU", "RRO", "POPC", "FLO", "BFE", "BFI", "BREV")):
        return "special_function"
    return "other"


def extract_call_target(mnemonic: str, operands: str) -> str | None:
    if not mnemonic.upper().startswith("CALL"):
        return None
    target = re.search(r"`\(\s*([^()\s]+)\s*\)", operands)
    if not target:
        target = re.search(r"`?([$_A-Za-z][\w$@.]*)", operands)
    return target.group(1) if target else None


def parse_instruction(line: str) -> SassInstruction | None:
    match = INSTRUCTION_RE.match(line)
    if not match:
        return None
    address, predicate, mnemonic, operands = match.groups()
    operands = normalize_space(operands or "")
    parts = mnemonic.split(".")
    return SassInstruction(
        address=address.lower(), address_value=int(address, 16), predicate=predicate,
        mnemonic=mnemonic, base=parts[0], modifiers=parts[1:], operands=operands,
        category=classify_instruction(mnemonic),
        call_target=extract_call_target(mnemonic, operands),
    )


def section_ranges(source: str) -> list[tuple[str, int, int, str]]:
    matches = list(SECTION_RE.finditer(source))
    return [
        (match.group(1), match.start(), matches[index + 1].start() if index + 1 < len(matches) else len(source),
         source[match.start():matches[index + 1].start() if index + 1 < len(matches) else len(source)])
        for index, match in enumerate(matches)
    ]


def symbol_label_matches(section_text: str, symbol: str, start: int, end: int) -> list[re.Match[str]]:
    pattern = re.compile(r"(?m)^\s*" + re.escape(symbol) + r":\s*$")
    return list(pattern.finditer(section_text, start, end))


def parse_text_section(name: str, text: str) -> tuple[list[SassFunction], list[dict[str, Any]]]:
    declarations = list(FUNCTION_TYPE_RE.finditer(text))
    ambiguities: list[dict[str, Any]] = []
    functions: list[SassFunction] = []
    primary_name = name[len(".text."):] if name.startswith(".text.") else None
    register_match = re.search(r'\.sectioninfo\s+@"[^"]*\bSHI_REGISTERS=(\d+)\b[^"]*"', text)
    section_registers = int(register_match.group(1)) if register_match else None
    owned_ranges: list[tuple[int, int]] = []

    if not declarations:
        instruction_lines = [line for line in text.splitlines() if parse_instruction(line)]
        if instruction_lines:
            ambiguities.append({"section": name, "reason": "instructions_without_function_symbols",
                                "instruction_count": len(instruction_lines)})
        return functions, ambiguities

    for index, declaration in enumerate(declarations):
        symbol = declaration.group(1)
        boundary = declarations[index + 1].start() if index + 1 < len(declarations) else len(text)
        labels = symbol_label_matches(text, symbol, declaration.end(), boundary)
        if len(labels) != 1:
            ambiguities.append({"section": name, "symbol": symbol,
                                "reason": "missing_or_ambiguous_exact_symbol_label",
                                "matching_labels": len(labels)})
            continue
        body_start, body_end = labels[0].end(), boundary
        owned_ranges.append((body_start, body_end))
        role = "primary" if symbol == primary_name else "helper"
        function = SassFunction(
            name=symbol, section=name, role=role,
            allocated_registers_per_thread=section_registers if role == "primary" else None,
            metadata={"section_registers": section_registers if role == "primary" else None},
        )
        for line_number, line in enumerate(text[body_start:body_end].splitlines(), 1):
            instruction = parse_instruction(line)
            if instruction:
                function.instructions.append(instruction)
                function.body_events.append(("instruction", instruction))
                continue
            stripped = line.strip()
            if re.fullmatch(r"[.$A-Za-z_][\w$@.]*:", stripped):
                label = stripped[:-1]
                if label != name:
                    function.labels.append(label)
                    function.body_events.append(("label", label))
            elif re.match(r"^/\*[0-9A-Fa-f]+\*/", stripped):
                ambiguities.append({"section": name, "symbol": symbol,
                                    "reason": "unparsed_address_bearing_line",
                                    "line_in_function": line_number, "text": stripped})
        functions.append(function)

    # Any instruction outside a confidently owned function body is not attributed.
    for match in re.finditer(r"(?m)^.*?/\*[0-9A-Fa-f]+\*/.*$", text):
        if parse_instruction(match.group(0)) and not any(start <= match.start() < end for start, end in owned_ranges):
            ambiguities.append({"section": name, "reason": "orphan_instruction_region",
                                "text": match.group(0).strip()})
    if primary_name and not any(function.role == "primary" for function in functions):
        ambiguities.append({"section": name, "symbol": primary_name,
                            "reason": "section_primary_function_not_resolved"})
    return functions, ambiguities


def parse_module(source: str) -> SassModule:
    target_match = re.search(r"(?m)^\s*\.target\s+([^\s]+)", source)
    functions: list[SassFunction] = []
    ambiguities: list[dict[str, Any]] = []
    text_sections = []
    for name, _start, _end, text in section_ranges(source):
        if not name.startswith(".text."):
            continue
        text_sections.append(name)
        parsed, section_ambiguities = parse_text_section(name, text)
        functions.extend(parsed)
        ambiguities.extend(section_ambiguities)
    if not text_sections:
        raise ValueError("no executable .text.<symbol> sections found")
    names = [function.name for function in functions]
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    if duplicates:
        ambiguities.append({"reason": "duplicate_function_symbols", "symbols": duplicates})
    return SassModule(source=source, target=target_match.group(1) if target_match else None,
                      functions=functions, ambiguities=ambiguities, text_sections=text_sections)


def normalize_module(source: str) -> str:
    source = source.replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for line in source.splitlines():
        line = re.sub(r"/\*[0-9A-Fa-f]+\*/", "", line)
        lines.append(" ".join(line.split()))
    return "\n".join(line for line in lines if line).strip() + "\n"


def normalize_function(function: SassFunction) -> str:
    lines = [f"function {function.name}"]
    for kind, value in function.body_events:
        if kind == "label":
            lines.append(f"{value}:")
            continue
        instruction = value
        prefix = f"{instruction.predicate} " if instruction.predicate else ""
        suffix = f" {instruction.operands}" if instruction.operands else ""
        lines.append(f"{prefix}{instruction.mnemonic}{suffix} ;")
    return "\n".join(lines) + "\n"


def function_summary(function: SassFunction) -> dict[str, Any]:
    categories = Counter(instruction.category for instruction in function.instructions)
    mnemonics = Counter(instruction.mnemonic for instruction in function.instructions)
    bases = Counter(instruction.base for instruction in function.instructions)
    calls = [
        {"address": instruction.address, "address_value": instruction.address_value,
         "mnemonic": instruction.mnemonic, "target": instruction.call_target,
         "operands": instruction.operands}
        for instruction in function.instructions if instruction.category == "call"
    ]
    return {
        "name": function.name, "section": function.section, "role": function.role,
        "instruction_count": len(function.instructions),
        "instructions": [asdict(instruction) for instruction in function.instructions],
        "instruction_categories": dict(sorted(categories.items())),
        "instruction_mnemonics": dict(sorted(mnemonics.items())),
        "base_operations": dict(sorted(bases.items())),
        "predicated_instruction_count": sum(instruction.predicate is not None for instruction in function.instructions),
        "labels": function.labels, "calls": calls,
        "allocated_registers_per_thread": function.allocated_registers_per_thread,
        "resource_metadata": function.metadata,
        "resource_note": (
            "SHI_REGISTERS is compiler-recorded physical 32-bit register allocation per thread for the primary "
            "section function. It is not a PTX virtual-register count or an occupancy estimate."
        ),
    }


def module_summary(module: SassModule, selected: list[SassFunction]) -> dict[str, Any]:
    return {
        "target": module.target, "text_sections": module.text_sections,
        "discovered_primary_functions": [function.name for function in module.functions if function.role == "primary"],
        "discovered_helper_functions": [function.name for function in module.functions if function.role == "helper"],
        "ambiguities": module.ambiguities,
        "selected_functions": [function_summary(function) for function in selected],
    }


def select_inspection(module: SassModule, name: str | None) -> list[SassFunction]:
    if name is None:
        return module.functions
    matches = [function for function in module.functions if function.name == name]
    if len(matches) != 1:
        raise ValueError(f"function {name!r} not found uniquely; discovered: {[item.name for item in module.functions]}")
    return matches


def select_comparison(left: SassModule, right: SassModule, same: str | None,
                      left_name: str | None, right_name: str | None) -> tuple[SassFunction, SassFunction]:
    if same and (left_name or right_name):
        raise ValueError("--kernel cannot be combined with paired kernel selection")
    if bool(left_name) != bool(right_name):
        raise ValueError("--baseline-kernel and --candidate-kernel must be supplied together")
    if same:
        left_name = right_name = same
    if left_name and right_name:
        left_matches = [function for function in left.functions if function.name == left_name]
        right_matches = [function for function in right.functions if function.name == right_name]
        if len(left_matches) != 1 or len(right_matches) != 1:
            raise ValueError("selected functions were not found uniquely")
        return left_matches[0], right_matches[0]
    left_primary = {function.name: function for function in left.functions if function.role == "primary"}
    right_primary = {function.name: function for function in right.functions if function.role == "primary"}
    common = sorted(set(left_primary) & set(right_primary))
    if len(common) != 1:
        raise ValueError(f"comparison requires one common primary function or explicit selection; common: {common}")
    return left_primary[common[0]], right_primary[common[0]]


def diff_stats(left: str, right: str) -> dict[str, Any]:
    diff = list(difflib.unified_diff(left.splitlines(), right.splitlines(), lineterm=""))
    additions = sum(line.startswith("+") and not line.startswith("+++") for line in diff)
    deletions = sum(line.startswith("-") and not line.startswith("---") for line in diff)
    return {"identical": left == right, "changed_lines": additions + deletions,
            "additions": additions, "deletions": deletions}


def bounded_diff(left: str, right: str, limit: int) -> tuple[str, bool, int]:
    lines = list(difflib.unified_diff(left.splitlines(), right.splitlines(),
                                      fromfile="baseline", tofile="candidate", lineterm=""))
    truncated = len(lines) > limit
    shown = lines[:limit]
    if truncated:
        shown.append(f"... diff truncated; {len(lines) - limit} lines omitted ...")
    return "\n".join(shown) + ("\n" if shown else ""), truncated, len(lines)


def count_delta(left: dict[str, int], right: dict[str, int]) -> list[dict[str, Any]]:
    return [{"name": name, "baseline": left.get(name, 0), "candidate": right.get(name, 0),
             "delta": right.get(name, 0) - left.get(name, 0)}
            for name in sorted(set(left) | set(right))]


def compare_functions(left: SassFunction, right: SassFunction) -> dict[str, Any]:
    left_summary, right_summary = function_summary(left), function_summary(right)
    left_normalized, right_normalized = normalize_function(left), normalize_function(right)
    return {
        "scope": "selected_function_only", "baseline_name": left.name, "candidate_name": right.name,
        "normalized": {**diff_stats(left_normalized, right_normalized),
                       "baseline_sha256": sha256(left_normalized.encode()),
                       "candidate_sha256": sha256(right_normalized.encode())},
        "instruction_categories": count_delta(left_summary["instruction_categories"], right_summary["instruction_categories"]),
        "instruction_mnemonics": count_delta(left_summary["instruction_mnemonics"], right_summary["instruction_mnemonics"]),
        "base_operations": count_delta(left_summary["base_operations"], right_summary["base_operations"]),
        "allocated_registers_per_thread": {
            "baseline": left.allocated_registers_per_thread,
            "candidate": right.allocated_registers_per_thread,
        },
        "calls": {"baseline": left_summary["calls"], "candidate": right_summary["calls"]},
        "baseline": left_summary, "candidate": right_summary,
    }


def file_record(path: Path, data: bytes) -> dict[str, Any]:
    return {"path": str(path), "size_bytes": len(data), "sha256": sha256(data)}


def render_inspection(result: dict[str, Any]) -> str:
    analysis = result["analysis"]
    lines = ["# NVIDIA SASS inspection", "", f"- Input: `{result['input']['path']}`",
             f"- Target: `{analysis['target'] or 'unavailable'}`",
             f"- Primary functions: {len(analysis['discovered_primary_functions'])}",
             f"- Helper functions: {len(analysis['discovered_helper_functions'])}",
             f"- Boundary ambiguities: {len(analysis['ambiguities'])}", ""]
    for function in analysis["selected_functions"]:
        registers = function["allocated_registers_per_thread"]
        lines += [f"## {function['role']} `{function['name']}`", "",
                  f"- Instructions: {function['instruction_count']}",
                  f"- Predicated instructions: {function['predicated_instruction_count']}",
                  f"- Allocated 32-bit registers/thread: {registers if registers is not None else 'unavailable'}",
                  f"- Calls: {len(function['calls'])}", "",
                  "### Instruction categories", "", "| Category | Count |", "|---|---:|"]
        lines += [f"| {name} | {count} |" for name, count in function["instruction_categories"].items()]
        lines += ["", "### Full mnemonics", "", "| Mnemonic | Count |", "|---|---:|"]
        lines += [f"| `{name}` | {count} |" for name, count in function["instruction_mnemonics"].items()]
        lines += ["", "### Helper calls", ""]
        lines += ([f"- `{call['mnemonic']}` at `0x{call['address']}` → `{call['target'] or 'unresolved'}`"
                   for call in function["calls"]] or ["None detected."])
        lines.append("")
    if analysis["ambiguities"]:
        lines += ["## Boundary ambiguities", ""]
        lines += [f"- `{json.dumps(item, sort_keys=True)}`" for item in analysis["ambiguities"]]
        lines.append("")
    lines += ["## Resource interpretation", "",
              "`SHI_REGISTERS` is compiler-recorded physical 32-bit register allocation per thread for a primary "
              "section function. Missing helper metadata remains unavailable. Register allocation alone does not establish occupancy.", "",
              "## Limitations", "",
              "This is structural analysis of nvdisasm text. It does not reconstruct instruction encodings, scheduling dependencies, "
              "occupancy, runtime behavior, or performance. Unknown instructions are preserved, and ambiguous instruction ownership is not guessed.", ""]
    return "\n".join(lines)


def render_comparison(result: dict[str, Any]) -> str:
    comparison = result["comparison"]
    function = comparison["function"]
    lines = ["# NVIDIA SASS comparison", "", f"- Baseline: `{result['baseline']['file']['path']}`",
             f"- Candidate: `{result['candidate']['file']['path']}`",
             f"- Functions: `{function['baseline_name']}` → `{function['candidate_name']}`", "",
             "## Full-file comparison", "",
             "| Layer | Result | Changed lines |", "|---|---|---:|",
             f"| Raw | {'identical' if comparison['raw']['identical'] else 'different'} | {comparison['raw']['changed_lines']} |",
             f"| Normalized presentation | {'identical' if comparison['module_normalized']['identical'] else 'different'} | {comparison['module_normalized']['changed_lines']} |", "",
             "## Selected-function comparison", "",
             "Address comments are omitted only from normalized text. Original addresses remain in structured instruction records. "
             "Normalized similarity is neither binary nor semantic equivalence.", "",
             f"- Normalized instruction text: {'identical' if function['normalized']['identical'] else 'different'}",
             f"- Allocated registers/thread: {function['allocated_registers_per_thread']['baseline']} → {function['allocated_registers_per_thread']['candidate']}", "",
             "### Instruction categories", "", "| Category | Baseline | Candidate | Delta |", "|---|---:|---:|---:|"]
    lines += [f"| {row['name']} | {row['baseline']} | {row['candidate']} | {row['delta']:+d} |"
              for row in function["instruction_categories"]]
    lines += ["", "### Full mnemonics", "", "| Mnemonic | Baseline | Candidate | Delta |", "|---|---:|---:|---:|"]
    lines += [f"| `{row['name']}` | {row['baseline']} | {row['candidate']} | {row['delta']:+d} |"
              for row in function["instruction_mnemonics"]]
    lines += ["", "### Helper calls", "",
              "| Capture | Address | Mnemonic | Target |", "|---|---|---|---|"]
    for side in ("baseline", "candidate"):
        for call in function["calls"][side]:
            lines.append(f"| {side} | `0x{call['address']}` | `{call['mnemonic']}` | `{call['target'] or 'unresolved'}` |")
    if not function["calls"]["baseline"] and not function["calls"]["candidate"]:
        lines.append("| — | — | — | No calls detected |")
    lines += ["", "## Conclusions", "",
              "- Differences above are static machine-instruction and compiler-metadata observations for the selected functions.",
              "- Static counts do not explain or predict measured runtime behavior.",
              "- Normalized textual agreement would not prove binary identity or semantic equivalence.", "",
              "## Limitations", "",
              "The parser consumes nvdisasm presentation text, not raw instruction encodings. It does not infer occupancy, "
              "scheduling, dependencies, memory traffic, or performance. Embedded helper bodies are excluded from primary-function counts.", ""]
    return "\n".join(lines)


def ensure_new_output(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"output path already exists: {path}")
    path.mkdir(parents=True)


def inspect_command(args: argparse.Namespace) -> dict[str, Any]:
    data = args.input.read_bytes()
    source = data.decode("utf-8")
    module = parse_module(source)
    selected = select_inspection(module, args.kernel)
    normalized = normalize_module(source)
    result = {"schema_version": SCHEMA_VERSION, "tool": "sass-inspector", "mode": "inspect",
              "normalization": NORMALIZATION, "input": file_record(args.input, data),
              "normalized_sha256": sha256(normalized.encode()),
              "analysis": module_summary(module, selected)}
    ensure_new_output(args.output)
    sass_dir = args.output / "sass"
    sass_dir.mkdir()
    (sass_dir / "input.normalized.sass").write_text(normalized, encoding="utf-8")
    (args.output / "analysis.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output / "report.md").write_text(render_inspection(result), encoding="utf-8")
    return result


def compare_command(args: argparse.Namespace) -> dict[str, Any]:
    left_data, right_data = args.baseline.read_bytes(), args.candidate.read_bytes()
    left_source, right_source = left_data.decode("utf-8"), right_data.decode("utf-8")
    left_module, right_module = parse_module(left_source), parse_module(right_source)
    left_function, right_function = select_comparison(
        left_module, right_module, args.kernel, args.baseline_kernel, args.candidate_kernel
    )
    left_module_normalized, right_module_normalized = normalize_module(left_source), normalize_module(right_source)
    left_function_normalized, right_function_normalized = normalize_function(left_function), normalize_function(right_function)
    diff, truncated, diff_lines = bounded_diff(left_function_normalized, right_function_normalized, args.diff_lines)
    function_comparison = compare_functions(left_function, right_function)
    result = {
        "schema_version": SCHEMA_VERSION, "tool": "sass-inspector", "mode": "compare",
        "normalization": NORMALIZATION,
        "baseline": {"file": file_record(args.baseline, left_data),
                     "analysis": module_summary(left_module, [left_function])},
        "candidate": {"file": file_record(args.candidate, right_data),
                      "analysis": module_summary(right_module, [right_function])},
        "comparison": {
            "raw": {"scope": "full_file", **diff_stats(left_source, right_source)},
            "module_normalized": {"scope": "full_file", **diff_stats(left_module_normalized, right_module_normalized),
                                  "baseline_sha256": sha256(left_module_normalized.encode()),
                                  "candidate_sha256": sha256(right_module_normalized.encode())},
            "function": {**function_comparison, "diff_truncated": truncated, "diff_total_lines": diff_lines},
        },
    }
    ensure_new_output(args.output)
    sass_dir = args.output / "sass"
    sass_dir.mkdir()
    (sass_dir / "baseline.normalized.sass").write_text(left_function_normalized, encoding="utf-8")
    (sass_dir / "candidate.normalized.sass").write_text(right_function_normalized, encoding="utf-8")
    (sass_dir / "normalized.diff").write_text(diff, encoding="utf-8")
    (args.output / "comparison.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output / "report.md").write_text(render_comparison(result), encoding="utf-8")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("--input", type=Path, required=True)
    inspect.add_argument("--kernel")
    inspect.add_argument("--output", type=Path, required=True)
    compare = commands.add_parser("compare")
    compare.add_argument("--baseline", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--kernel")
    compare.add_argument("--baseline-kernel")
    compare.add_argument("--candidate-kernel")
    compare.add_argument("--diff-lines", type=int, default=400)
    compare.add_argument("--output", type=Path, required=True)
    return parser


def run(arguments: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(arguments)
    if getattr(args, "diff_lines", 0) < 0:
        raise ValueError("--diff-lines must be nonnegative")
    return inspect_command(args) if args.command == "inspect" else compare_command(args)


def main(arguments: Sequence[str] | None = None) -> int:
    try:
        result = run(arguments)
        if result["mode"] == "inspect":
            analysis = result["analysis"]
            print(f"SASS target={analysis['target']} primary={len(analysis['discovered_primary_functions'])} helpers={len(analysis['discovered_helper_functions'])}")
        else:
            comparison = result["comparison"]["function"]
            print(f"Selected SASS: {'identical' if comparison['normalized']['identical'] else 'different'}")
        return 0
    except (OSError, UnicodeError, ValueError) as error:
        print(f"sass-inspector: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
