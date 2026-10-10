#!/usr/bin/env python3
"""Inspect and compare NVIDIA PTX using only the Python standard library."""

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
NORMALIZATION = "Normalize line endings/trailing whitespace and omit .file/.loc directives; preserve comments and code."


@dataclass
class Instruction:
    mnemonic: str
    base: str
    modifiers: list[str]
    predicate: str | None
    operands: str
    category: str


@dataclass
class RegisterDeclaration:
    type: str
    count: int
    text: str


@dataclass
class SharedDeclaration:
    name: str
    type: str | None
    elements: int | None
    static_bytes: int | None
    dynamic: bool
    text: str


@dataclass
class PtxSymbol:
    name: str
    kind: str
    interface: str
    parameter_types: list[str]
    instructions: list[Instruction] = field(default_factory=list)
    registers: list[RegisterDeclaration] = field(default_factory=list)
    shared_memory: list[SharedDeclaration] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    directives: list[str] = field(default_factory=list)


@dataclass
class PtxModule:
    source: str
    version: str | None
    target: str | None
    address_size: int | None
    symbols: list[PtxSymbol]
    module_shared_memory: list[SharedDeclaration]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def strip_comments_for_parsing(text: str) -> str:
    """Replace comment contents with spaces while preserving offsets/newlines."""
    output = list(text)
    index = 0
    while index < len(text):
        if text.startswith("//", index):
            end = text.find("\n", index)
            end = len(text) if end < 0 else end
            for position in range(index, end):
                output[position] = " "
            index = end
        elif text.startswith("/*", index):
            end = text.find("*/", index + 2)
            if end < 0:
                raise ValueError("unterminated PTX block comment")
            for position in range(index, end + 2):
                if output[position] != "\n":
                    output[position] = " "
            index = end + 2
        elif text[index] == '"':
            index = skip_string(text, index)
        else:
            index += 1
    return "".join(output)


def skip_string(text: str, start: int) -> int:
    index = start + 1
    while index < len(text):
        if text[index] == "\\":
            index += 2
        elif text[index] == '"':
            return index + 1
        else:
            index += 1
    raise ValueError("unterminated PTX string")


def matching_delimiter(text: str, start: int, opening: str, closing: str) -> int:
    depth = 1
    index = start + 1
    while index < len(text):
        if text[index] == '"':
            index = skip_string(text, index)
            continue
        if text[index] == opening:
            depth += 1
        elif text[index] == closing:
            depth -= 1
            if depth == 0:
                return index
        index += 1
    raise ValueError(f"unbalanced PTX delimiter {opening}{closing}")


def normalize_space(text: str) -> str:
    return " ".join(text.split())


def split_body(body: str) -> tuple[list[str], list[str]]:
    """Split semicolon statements and labels with balanced operand delimiters."""
    statements: list[str] = []
    labels: list[str] = []
    buffer: list[str] = []
    depths = {"(": 0, "[": 0, "{": 0}
    pairs = {")": "(", "]": "[", "}": "{"}
    index = 0
    while index < len(body):
        char = body[index]
        if char == '"':
            end = skip_string(body, index)
            buffer.append(body[index:end])
            index = end
            continue
        if char in depths:
            depths[char] += 1
        elif char in pairs:
            opening = pairs[char]
            if depths[opening] == 0:
                raise ValueError(f"unmatched {char} in PTX function body")
            depths[opening] -= 1
        buffer.append(char)
        at_top = not any(depths.values())
        if char == "\n" and at_top:
            candidate = "".join(buffer).strip()
            if re.match(r"\.loc\b", candidate):
                statements.append(candidate)
                buffer.clear()
        elif char == ":" and at_top:
            candidate = "".join(buffer[:-1]).strip()
            if re.fullmatch(r"[$A-Za-z_.$][\w$.$]*", candidate):
                labels.append(candidate)
                buffer.clear()
        elif char == ";" and at_top:
            statement = "".join(buffer).strip()
            if statement:
                statements.append(statement)
            buffer.clear()
        index += 1
    if any(depths.values()):
        raise ValueError("unbalanced operand delimiter in PTX function body")
    remainder = "".join(buffer).strip()
    if remainder and remainder not in ("{", "}"):
        raise ValueError(f"unterminated PTX statement: {remainder[:80]}")
    return statements, labels


def classify_instruction(mnemonic: str) -> str:
    parts = mnemonic.lower().split(".")
    base = parts[0]
    modifiers = set(parts[1:])
    if base in {"mma", "wmma", "wgmma"}:
        return "tensor"
    if base in {"bar", "barrier", "mbarrier", "membar", "fence"}:
        return "synchronization"
    if base in {"bra", "brx", "call", "ret", "exit", "trap", "brkpt"}:
        return "control_flow"
    if base in {"cvt", "cvta"}:
        return "conversion"
    if base in {"set", "setp", "selp", "slct"}:
        return "comparison_predication"
    if base in {"ld", "ldu", "st", "atom", "red", "prefetch", "prefetchu", "suld", "sust", "suq"}:
        for space in ("global", "shared", "local", "const", "param", "generic"):
            if space in modifiers:
                return f"memory_{space}"
        if base in {"ld", "ldu", "st"}:
            return "memory_generic"
        return "memory_other"
    if base in {
        "abs", "add", "addc", "and", "bfe", "bfi", "brev", "clz", "copysign", "cos",
        "div", "ex2", "fma", "lg2", "mad", "madc", "max", "min", "mov", "mul", "neg",
        "not", "or", "popc", "rcp", "rem", "rsqrt", "sad", "shf", "shl", "shr", "sin",
        "sqrt", "sub", "subc", "tanh", "testp", "xor",
    }:
        return "arithmetic"
    return "other"


def parse_register(statement: str) -> RegisterDeclaration | None:
    if not re.search(r"(?:^|\s)\.reg\b", statement):
        return None
    type_match = re.search(r"\.reg\s+\.(pred|[busf]\d+)\b", statement)
    if not type_match:
        raise ValueError(f"unsupported register declaration: {normalize_space(statement)}")
    tail = statement[type_match.end():].rstrip(";").strip()
    count = 0
    for item in tail.split(","):
        item = item.strip()
        group = re.search(r"<\s*(\d+)\s*>", item)
        count += int(group.group(1)) if group else 1
    return RegisterDeclaration(type=type_match.group(1), count=count, text=normalize_space(statement))


def parse_shared(statement: str) -> SharedDeclaration | None:
    if not re.search(r"(?:^|\s)\.shared\b", statement):
        return None
    dynamic = bool(re.search(r"(?:^|\s)\.extern\b", statement))
    type_match = re.search(r"\.(b|u|s|f)(\d+)\b", statement)
    array_match = re.search(r"([A-Za-z_$][\w$.$]*)\s*\[\s*(\d*)\s*\]\s*;?$", statement)
    scalar_match = re.search(r"([A-Za-z_$][\w$.$]*)\s*;?$", statement)
    name_match = array_match or scalar_match
    if not name_match:
        raise ValueError(f"unsupported shared-memory declaration: {normalize_space(statement)}")
    elements = int(array_match.group(2)) if array_match and array_match.group(2) else None
    dynamic = dynamic or bool(array_match and not array_match.group(2))
    static_bytes = None
    if not dynamic and type_match:
        static_bytes = (elements if elements is not None else 1) * int(type_match.group(2)) // 8
    return SharedDeclaration(
        name=name_match.group(1),
        type=(type_match.group(1) + type_match.group(2)) if type_match else None,
        elements=elements, static_bytes=static_bytes, dynamic=dynamic,
        text=normalize_space(statement),
    )


def parse_instruction(statement: str) -> Instruction:
    text = statement.rstrip(";").strip()
    predicate = None
    predicate_match = re.match(r"@(!?%[A-Za-z0-9_$]+)\s+", text)
    if predicate_match:
        predicate = predicate_match.group(1)
        text = text[predicate_match.end():]
    match = re.match(r"([A-Za-z_][A-Za-z0-9_.]*)\b(?:\s+(.*))?$", text, re.DOTALL)
    if not match:
        raise ValueError(f"cannot parse PTX instruction: {normalize_space(statement)}")
    mnemonic = match.group(1)
    operands = normalize_space(match.group(2) or "")
    parts = mnemonic.split(".")
    return Instruction(mnemonic=mnemonic, base=parts[0], modifiers=parts[1:],
                       predicate=predicate, operands=operands,
                       category=classify_instruction(mnemonic))


def parse_symbol(name: str, kind: str, interface: str, body: str) -> PtxSymbol:
    statements, labels = split_body(body)
    symbol = PtxSymbol(
        name=name, kind=kind, interface=normalize_space(interface),
        parameter_types=re.findall(r"\.param(?:\s+\.align\s+\d+)?\s+\.([busf]\d+)", interface),
        labels=labels,
    )
    for statement in statements:
        register = parse_register(statement)
        if register:
            symbol.registers.append(register)
            continue
        shared = parse_shared(statement)
        if shared:
            symbol.shared_memory.append(shared)
            continue
        if statement.lstrip().startswith("."):
            symbol.directives.append(normalize_space(statement))
            continue
        symbol.instructions.append(parse_instruction(statement))
    return symbol


def find_symbols(clean: str) -> tuple[list[PtxSymbol], list[tuple[int, int]]]:
    symbols = []
    ranges = []
    pattern = re.compile(r"\.(entry|func)\b")
    for match in pattern.finditer(clean):
        kind = "entry" if match.group(1) == "entry" else "func"
        index = match.end()
        while index < len(clean) and clean[index].isspace():
            index += 1
        if kind == "func" and index < len(clean) and clean[index] == "(":
            index = matching_delimiter(clean, index, "(", ")") + 1
            while index < len(clean) and clean[index].isspace():
                index += 1
        name_match = re.match(r"[A-Za-z_$][\w$.$]*", clean[index:])
        if not name_match:
            raise ValueError(f"missing PTX {kind} name near offset {match.start()}")
        name = name_match.group(0)
        search = index + name_match.end()
        paren = bracket = 0
        body_start = None
        while search < len(clean):
            char = clean[search]
            if char == '"':
                search = skip_string(clean, search)
                continue
            if char == "(":
                paren += 1
            elif char == ")":
                paren -= 1
            elif char == "[":
                bracket += 1
            elif char == "]":
                bracket -= 1
            elif char == ";" and paren == bracket == 0:
                break  # function declaration without a body
            elif char == "{" and paren == bracket == 0:
                body_start = search
                break
            search += 1
        if body_start is None:
            continue
        body_end = matching_delimiter(clean, body_start, "{", "}")
        interface = clean[match.start():body_start]
        symbols.append(parse_symbol(name, kind, interface, clean[body_start + 1:body_end]))
        ranges.append((match.start(), body_end + 1))
    return symbols, ranges


def parse_module(source: str) -> PtxModule:
    clean = strip_comments_for_parsing(source)
    version_match = re.search(r"(?m)^\s*\.version\s+([^\s;]+)", clean)
    target_match = re.search(r"(?m)^\s*\.target\s+([^\s,;]+)", clean)
    address_match = re.search(r"(?m)^\s*\.address_size\s+(\d+)", clean)
    if not version_match or not target_match or not address_match:
        raise ValueError("PTX must declare .version, .target, and .address_size")
    symbols, ranges = find_symbols(clean)
    masked = list(clean)
    for start, end in ranges:
        for index in range(start, end):
            if masked[index] != "\n":
                masked[index] = " "
    module_shared = []
    for statement in re.findall(r"(?m)^[^\n;]*\.shared\b[^;]*;", "".join(masked)):
        shared = parse_shared(statement.strip())
        if shared:
            module_shared.append(shared)
    return PtxModule(source=source, version=version_match.group(1), target=target_match.group(1),
                     address_size=int(address_match.group(1)), symbols=symbols,
                     module_shared_memory=module_shared)


def normalize_ptx(source: str) -> str:
    normalized = source.replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for line in normalized.splitlines():
        stripped = line.lstrip()
        if re.match(r"\.(file|loc)\b", stripped):
            continue
        lines.append(line.rstrip())
    return "\n".join(lines).strip() + "\n"


def symbol_summary(symbol: PtxSymbol) -> dict[str, Any]:
    categories = Counter(instruction.category for instruction in symbol.instructions)
    mnemonics = Counter(instruction.mnemonic for instruction in symbol.instructions)
    register_types = Counter()
    for declaration in symbol.registers:
        register_types[declaration.type] += declaration.count
    static_shared = sum(item.static_bytes or 0 for item in symbol.shared_memory if not item.dynamic)
    return {
        "name": symbol.name, "kind": symbol.kind, "interface": symbol.interface,
        "parameter_types": symbol.parameter_types,
        "instruction_count": len(symbol.instructions),
        "instruction_categories": dict(sorted(categories.items())),
        "instruction_mnemonics": dict(sorted(mnemonics.items())),
        "predicated_instruction_count": sum(item.predicate is not None for item in symbol.instructions),
        "labels": symbol.labels, "label_count": len(symbol.labels),
        "register_declarations": [asdict(item) for item in symbol.registers],
        "ptx_declared_registers_by_type": dict(sorted(register_types.items())),
        "shared_memory_declarations": [asdict(item) for item in symbol.shared_memory],
        "ptx_declared_static_shared_bytes": static_shared,
        "has_dynamic_shared_memory": any(item.dynamic for item in symbol.shared_memory),
        "directives": symbol.directives,
    }


def module_summary(module: PtxModule, selected: list[PtxSymbol]) -> dict[str, Any]:
    return {
        "ptx_version": module.version, "target": module.target, "address_size": module.address_size,
        "discovered_entries": [item.name for item in module.symbols if item.kind == "entry"],
        "discovered_functions": [item.name for item in module.symbols if item.kind == "func"],
        "selected_symbols": [symbol_summary(item) for item in selected],
        "module_shared_memory_declarations": [asdict(item) for item in module.module_shared_memory],
        "resource_note": (
            "Register and shared-memory values are PTX declarations only. They are not final physical register "
            "allocation, final hardware resource usage, or an occupancy estimate."
        ),
    }


def select_inspection(module: PtxModule, kernel: str | None) -> list[PtxSymbol]:
    if kernel is None:
        return module.symbols
    matches = [item for item in module.symbols if item.name == kernel]
    if not matches:
        raise ValueError(f"kernel/function {kernel!r} not found; discovered: {', '.join(item.name for item in module.symbols)}")
    return matches


def select_comparison(left: PtxModule, right: PtxModule, same: str | None,
                      left_name: str | None, right_name: str | None) -> tuple[PtxSymbol, PtxSymbol]:
    if same and (left_name or right_name):
        raise ValueError("--kernel cannot be combined with --baseline-kernel/--candidate-kernel")
    if bool(left_name) != bool(right_name):
        raise ValueError("--baseline-kernel and --candidate-kernel must be supplied together")
    if same:
        left_name = right_name = same
    if left_name and right_name:
        left_matches = [item for item in left.symbols if item.name == left_name and item.kind == "entry"]
        right_matches = [item for item in right.symbols if item.name == right_name and item.kind == "entry"]
        if len(left_matches) != 1 or len(right_matches) != 1:
            raise ValueError("selected entry point was not found uniquely in both PTX files")
        return left_matches[0], right_matches[0]
    left_entries = {item.name: item for item in left.symbols if item.kind == "entry"}
    right_entries = {item.name: item for item in right.symbols if item.kind == "entry"}
    common = sorted(set(left_entries) & set(right_entries))
    if len(common) != 1:
        raise ValueError(f"comparison requires one unambiguous common entry or explicit selection; common entries: {common}")
    return left_entries[common[0]], right_entries[common[0]]


def diff_stats(left: str, right: str) -> dict[str, Any]:
    lines = list(difflib.unified_diff(left.splitlines(), right.splitlines(), lineterm=""))
    additions = sum(line.startswith("+") and not line.startswith("+++") for line in lines)
    deletions = sum(line.startswith("-") and not line.startswith("---") for line in lines)
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


def compare_symbols(left: PtxSymbol, right: PtxSymbol) -> dict[str, Any]:
    left_summary, right_summary = symbol_summary(left), symbol_summary(right)
    return {
        "scope": "selected_entry_only",
        "baseline_name": left.name, "candidate_name": right.name,
        "interface_compatible": left.parameter_types == right.parameter_types,
        "baseline_interface": left.interface, "candidate_interface": right.interface,
        "instruction_categories": count_delta(left_summary["instruction_categories"], right_summary["instruction_categories"]),
        "instruction_mnemonics": count_delta(left_summary["instruction_mnemonics"], right_summary["instruction_mnemonics"]),
        "ptx_declared_registers": count_delta(left_summary["ptx_declared_registers_by_type"], right_summary["ptx_declared_registers_by_type"]),
        "static_shared_bytes": {"baseline": left_summary["ptx_declared_static_shared_bytes"],
                                "candidate": right_summary["ptx_declared_static_shared_bytes"]},
        "dynamic_shared_memory": {"baseline": left_summary["has_dynamic_shared_memory"],
                                  "candidate": right_summary["has_dynamic_shared_memory"]},
        "baseline": left_summary, "candidate": right_summary,
    }


def file_summary(path: Path, data: bytes) -> dict[str, Any]:
    return {"path": str(path), "size_bytes": len(data), "sha256": sha256(data)}


def render_inspection(result: dict[str, Any]) -> str:
    analysis = result["analysis"]
    def shared_line(item: dict[str, Any]) -> str:
        extent = "dynamic/extern" if item["dynamic"] else f"{item['static_bytes']} static bytes"
        return f"- `{item['text']}` ({extent})"

    lines = ["# PTX inspection", "", f"- Input: `{result['input']['path']}`",
             f"- PTX version: `{analysis['ptx_version']}`", f"- Target: `{analysis['target']}`",
             f"- Address size: {analysis['address_size']}", "", "## Discovered symbols", "",
             f"- Entry points: {', '.join(f'`{name}`' for name in analysis['discovered_entries']) or 'none'}",
             f"- Device functions: {', '.join(f'`{name}`' for name in analysis['discovered_functions']) or 'none'}", ""]
    if analysis["module_shared_memory_declarations"]:
        lines += ["### Module-level PTX shared-memory declarations", ""]
        lines += [shared_line(item) for item in analysis["module_shared_memory_declarations"]]
        lines.append("")
    for symbol in analysis["selected_symbols"]:
        lines += [f"## {symbol['kind']} `{symbol['name']}`", "",
                  f"- Instructions: {symbol['instruction_count']}",
                  f"- Predicated instructions: {symbol['predicated_instruction_count']}",
                  f"- Labels: {symbol['label_count']}",
                  f"- PTX-declared static shared memory: {symbol['ptx_declared_static_shared_bytes']} bytes",
                  f"- Dynamic shared memory declaration: {str(symbol['has_dynamic_shared_memory']).lower()}", "",
                  "### Instruction categories", "", "| Category | Count |", "|---|---:|"]
        lines += [f"| {name} | {count} |" for name, count in symbol["instruction_categories"].items()]
        lines += ["", "### PTX-declared registers", "", "| Type | Declared names |", "|---|---:|"]
        lines += [f"| {name} | {count} |" for name, count in symbol["ptx_declared_registers_by_type"].items()]
        lines += ["", "### PTX shared-memory declarations", ""]
        lines += ([shared_line(item) for item in symbol["shared_memory_declarations"]]
                  or ["None in this symbol."])
        lines.append("")
    lines += ["## Resource interpretation", "", analysis["resource_note"], "",
              "## Limitations", "",
              "This is conservative static PTX analysis. It does not inspect SASS, final instruction scheduling, "
              "physical register allocation, occupancy, runtime memory traffic, or performance. Per-entry statistics "
              "exclude instructions in called device functions.", ""]
    return "\n".join(lines)


def render_comparison(result: dict[str, Any]) -> str:
    comparison = result["comparison"]
    structural = comparison["kernel"]
    lines = ["# PTX comparison", "", f"- Baseline: `{result['baseline']['file']['path']}`",
             f"- Candidate: `{result['candidate']['file']['path']}`",
             f"- Kernels: `{structural['baseline_name']}` → `{structural['candidate_name']}`", "",
             "## Compatibility", "", "| Field | Baseline | Candidate | Match |", "|---|---|---|---|",
             f"| PTX version | {result['baseline']['analysis']['ptx_version']} | {result['candidate']['analysis']['ptx_version']} | {comparison['compatibility']['ptx_version']} |",
             f"| Target | {result['baseline']['analysis']['target']} | {result['candidate']['analysis']['target']} | {comparison['compatibility']['target']} |",
             f"| Address size | {result['baseline']['analysis']['address_size']} | {result['candidate']['analysis']['address_size']} | {comparison['compatibility']['address_size']} |",
             f"| Kernel interface | — | — | {structural['interface_compatible']} |", "",
             "## Full-module text comparison", "",
             "These raw and normalized results compare the complete PTX modules, not only the selected entry point.", "",
             "| Layer | Result | Changed lines |", "|---|---|---:|",
             f"| Raw | {'identical' if comparison['raw']['identical'] else 'different'} | {comparison['raw']['changed_lines']} |",
             f"| Normalized | {'identical' if comparison['normalized']['identical'] else 'different'} | {comparison['normalized']['changed_lines']} |", "",
             "## Selected-entry structural comparison", "",
             "The instruction and resource statistics below cover only the selected entry point; called device-function bodies are excluded.", "",
             "### Instruction categories", "", "| Category | Baseline | Candidate | Delta |", "|---|---:|---:|---:|"]
    lines += [f"| {row['name']} | {row['baseline']} | {row['candidate']} | {row['delta']:+d} |"
              for row in structural["instruction_categories"]]
    lines += ["", "### Full mnemonics", "", "| Mnemonic | Baseline | Candidate | Delta |", "|---|---:|---:|---:|"]
    lines += [f"| `{row['name']}` | {row['baseline']} | {row['candidate']} | {row['delta']:+d} |"
              for row in structural["instruction_mnemonics"]]
    lines += ["", "### PTX-declared resources", "",
              "These are virtual register and static/dynamic shared-memory declarations in PTX. They are not final "
              "physical register allocation, hardware resource usage, or occupancy estimates.", "",
              "| Register type | Baseline | Candidate | Delta |", "|---|---:|---:|---:|"]
    lines += [f"| {row['name']} | {row['baseline']} | {row['candidate']} | {row['delta']:+d} |"
              for row in structural["ptx_declared_registers"]]
    lines += ["", f"- PTX-declared static shared bytes: {structural['static_shared_bytes']['baseline']} → {structural['static_shared_bytes']['candidate']}",
              f"- Dynamic shared declaration: {structural['dynamic_shared_memory']['baseline']} → {structural['dynamic_shared_memory']['candidate']}", "",
              "### Baseline shared-memory declarations", ""]
    lines += ([f"- `{item['text']}`" for item in structural["baseline"]["shared_memory_declarations"]]
              or ["None in the selected entry."])
    lines += ["", "### Candidate shared-memory declarations", ""]
    lines += ([f"- `{item['text']}`" for item in structural["candidate"]["shared_memory_declarations"]]
              or ["None in the selected entry."])
    lines += ["",
              "## Conclusions", "",
              f"- The input PTX files are {'identical' if comparison['normalized']['identical'] else 'different'} after the documented conservative normalization.",
              "- Instruction and declaration differences are static PTX observations; they do not establish SASS changes or runtime performance effects.", "",
              "## Limitations", "",
              "Normalization removes only `.file`/`.loc` source annotations and whitespace differences described in the JSON report. "
              "Comments, predicates, constants, operands, labels, symbols, and instruction order are retained. Per-entry statistics "
              "exclude called device-function bodies. PTX instructions do not map one-to-one to SASS instructions.", ""]
    return "\n".join(lines)


def write_new_directory(output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"output path already exists: {output}")
    output.mkdir(parents=True)


def inspect_command(args: argparse.Namespace) -> dict[str, Any]:
    input_bytes = args.input.read_bytes()
    source = input_bytes.decode("utf-8")
    module = parse_module(source)
    selected = select_inspection(module, args.kernel)
    normalized = normalize_ptx(source)
    result = {"schema_version": SCHEMA_VERSION, "tool": "ptx-inspector", "mode": "inspect",
              "normalization": NORMALIZATION, "input": file_summary(args.input, input_bytes),
              "normalized_sha256": sha256(normalized.encode()),
              "analysis": module_summary(module, selected)}
    write_new_directory(args.output)
    ptx_dir = args.output / "ptx"
    ptx_dir.mkdir()
    (ptx_dir / "input.normalized.ptx").write_text(normalized, encoding="utf-8")
    (args.output / "analysis.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output / "report.md").write_text(render_inspection(result), encoding="utf-8")
    return result


def compare_command(args: argparse.Namespace) -> dict[str, Any]:
    left_bytes = args.baseline.read_bytes()
    right_bytes = args.candidate.read_bytes()
    left_source = left_bytes.decode("utf-8")
    right_source = right_bytes.decode("utf-8")
    left_module, right_module = parse_module(left_source), parse_module(right_source)
    left_symbol, right_symbol = select_comparison(
        left_module, right_module, args.kernel, args.baseline_kernel, args.candidate_kernel
    )
    left_normalized, right_normalized = normalize_ptx(left_source), normalize_ptx(right_source)
    diff, truncated, total_lines = bounded_diff(left_normalized, right_normalized, args.diff_lines)
    raw = diff_stats(left_source, right_source)
    normalized = {**diff_stats(left_normalized, right_normalized),
                  "baseline_sha256": sha256(left_normalized.encode()),
                  "candidate_sha256": sha256(right_normalized.encode()),
                  "diff_truncated": truncated, "diff_total_lines": total_lines}
    left_analysis = module_summary(left_module, [left_symbol])
    right_analysis = module_summary(right_module, [right_symbol])
    result = {
        "schema_version": SCHEMA_VERSION, "tool": "ptx-inspector", "mode": "compare",
        "normalization": NORMALIZATION,
        "baseline": {"file": file_summary(args.baseline, left_bytes), "analysis": left_analysis},
        "candidate": {"file": file_summary(args.candidate, right_bytes), "analysis": right_analysis},
        "comparison": {
            "text_scope": "full_module",
            "raw": {"scope": "full_module", **raw},
            "normalized": {"scope": "full_module", **normalized},
            "compatibility": {
                "ptx_version": left_module.version == right_module.version,
                "target": left_module.target == right_module.target,
                "address_size": left_module.address_size == right_module.address_size,
            },
            "kernel": compare_symbols(left_symbol, right_symbol),
        },
    }
    write_new_directory(args.output)
    ptx_dir = args.output / "ptx"
    ptx_dir.mkdir()
    (ptx_dir / "baseline.normalized.ptx").write_text(left_normalized, encoding="utf-8")
    (ptx_dir / "candidate.normalized.ptx").write_text(right_normalized, encoding="utf-8")
    (ptx_dir / "normalized.diff").write_text(diff, encoding="utf-8")
    (args.output / "comparison.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output / "report.md").write_text(render_comparison(result), encoding="utf-8")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="Inspect one PTX file")
    inspect.add_argument("--input", type=Path, required=True)
    inspect.add_argument("--output", type=Path, required=True)
    inspect.add_argument("--kernel")
    compare = commands.add_parser("compare", help="Compare two PTX files")
    compare.add_argument("--baseline", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--kernel")
    compare.add_argument("--baseline-kernel")
    compare.add_argument("--candidate-kernel")
    compare.add_argument("--diff-lines", type=int, default=400)
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
            print(f"PTX {analysis['ptx_version']} target={analysis['target']} entries={len(analysis['discovered_entries'])}")
        else:
            comparison = result["comparison"]
            print(f"Raw PTX: {'identical' if comparison['raw']['identical'] else 'different'}")
            print(f"Normalized PTX: {'identical' if comparison['normalized']['identical'] else 'different'}")
        return 0
    except (OSError, ValueError) as error:
        print(f"ptx-inspector: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
