"""Generate the matched 36-CTA Perminov-58 kernel from verified exact DAG data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

try:
    from scripts import run_mi_research_6c as exact
except ModuleNotFoundError:
    import run_mi_research_6c as exact

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "examples/kernels/mi_research_7b_fused_perminov55_output_block.py"
TARGET = ROOT / "examples/kernels/mi_research_8_fused_perminov58_output_block.py"
BASELINE_SHA256 = "10b94103a2dbf3d1649d1e3d79560d45830eeb5d960161905d69a6d5947db418"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _statement(node: dict) -> str:
    name = node["id"]
    if node["kind"] == "ADD":
        a, b = node["deps"]
        x, y = node["signs"]
        left = a if x == 1 else f"(-{a})"
        return f"{name} = {left} {'+' if y == 1 else '-'} {b}"
    if node["kind"] == "PRODUCT":
        a, b = node["deps"]
        x, y = node["signs"]
        left = a if x == 1 else f"(-{a})"
        right = b if y == 1 else f"(-{b})"
        return (f'{name} = tl.dot({left}, {right}, '
                'tl.zeros((16, 16), tl.float32), input_precision="tf32x3")')
    if node["kind"] == "OUTPUT":
        index = int(name[1:])
        block_row, block_col = divmod(index, 3)
        dep = node["deps"][0]
        value = dep if node["signs"][0] == 1 else f"(-{dep})"
        return (f"tl.store(c_ptr + ({block_row} * 32 + row[:, None]) * 96 + "
                f"{block_col} * 32 + col[None, :], {value})")
    raise ValueError(f"unexpected node kind: {node['kind']}")


def _load(name: str) -> str:
    side, index = name[0], int(name[1:])
    block_row, block_col = divmod(index, 3)
    if side == "A":
        address = (f"a_ptr + ({block_row} * 32 + row[:, None]) * 96 + "
                   f"{block_col} * 32 + inner[None, :]")
    else:
        address = (f"b_ptr + ({block_row} * 32 + inner[:, None]) * 96 + "
                   f"{block_col} * 32 + col[None, :]")
    return f"{name} = tl.load({address}).to(tl.float32)"


def render(graph: exact.Circuit, order: list[str], *, optimized: bool) -> tuple[str, dict]:
    additions = sum(n["kind"] == "ADD" for n in graph.nodes)
    if additions != (55 if optimized else 58) or sum(n["kind"] == "PRODUCT" for n in graph.nodes) != 23:
        raise ValueError("unexpected Perminov circuit")
    title = ("Generated output-block ownership of the frozen Perminov 55-gate circuit."
             if optimized else "Generated output-block ownership of the verified Perminov 58-gate circuit.")
    instruction = ("Run scripts/generate_mi_research_7b_kernel.py to check the source slices."
                   if optimized else "Run scripts/generate_mi_research_8_kernel.py to check the source slices.")
    lines = [f'"""{title}', "", instruction, '"""', "", "import torch", "import triton",
             "import triton.language as tl", "", "", "@triton.jit",
             "def _fused(a_ptr, b_ptr, c_ptr):",
             "    row = tl.program_id(0) * 16 + tl.arange(0, 16)",
             "    col = tl.program_id(1) * 16 + tl.arange(0, 16)",
             "    inner = tl.arange(0, 32)", "    output_block = tl.program_id(2)"]
    stats = {"dots_by_block": [], "gates_by_block": [], "loads_by_block": [],
             "output_blocks": 9, "grid": [2, 2, 9]}
    for index in range(9):
        needed = {f"C{index}"}
        pending = [f"C{index}"]
        while pending:
            name = pending.pop()
            for dependency in graph.by_id[name]["deps"]:
                if dependency not in needed:
                    needed.add(dependency)
                    pending.append(dependency)
        lines.append(f"    if output_block == {index}:")
        lines.extend(("        row = tl.program_id(0) * 16 + tl.arange(0, 16)",
                      "        col = tl.program_id(1) * 16 + tl.arange(0, 16)",
                      "        inner = tl.arange(0, 32)"))
        inputs = [f"{side}{i}" for side in ("A", "B") for i in range(9)
                  if f"{side}{i}" in needed]
        lines.extend("        " + _load(name) for name in inputs)
        selected = [graph.by_id[name] for name in order if name in needed]
        if selected[-1]["id"] != f"C{index}":
            raise ValueError(f"output order changed: {index}")
        lines.extend("        " + _statement(node) for node in selected)
        stats["loads_by_block"].append(len(inputs))
        stats["dots_by_block"].append(sum(n["kind"] == "PRODUCT" for n in selected))
        stats["gates_by_block"].append(sum(n["kind"] == "ADD" for n in selected))
    lines += ["", "", "def call(left, right, output):",
              "    if (left.shape != (96, 96) or right.shape != (96, 96) or",
              "            output.shape != (96, 96) or left.dtype != torch.float16 or",
              "            right.dtype != torch.float16 or output.dtype != torch.float16 or",
              "            not left.is_cuda or not right.is_cuda or not output.is_cuda or",
              "            not left.is_contiguous() or not right.is_contiguous() or",
              "            not output.is_contiguous()):",
              '        raise ValueError("expected contiguous CUDA FP16 96x96 inputs/output")',
              "    return _fused[(2, 2, 9)](left, right, output, num_warps=4, num_stages=2)"]
    return "\n".join(lines) + "\n", stats


def generate() -> tuple[str, dict]:
    if sha256(BASELINE) != BASELINE_SHA256 or exact.verify()["status"] != "PASS":
        raise ValueError("frozen 55 circuit or exact evidence changed")
    data = json.loads(exact.SOURCE.read_text())
    cert = json.loads(exact.CERT.read_text())
    retained = json.loads((exact.OUT / "analysis.json").read_text())["configurations"]
    circuits = {}
    for label, optimized in (("55", True), ("58", False)):
        graph, refs = exact.build(data, cert, optimized=optimized)
        if exact.factor_rows(graph, refs) != retained[f"perminov_{label}"]["factor_coefficients"]:
            raise ValueError(f"Perminov {label} factors changed")
        if exact.tensor_check(exact.factor_rows(graph, refs))["mismatches"]:
            raise ValueError(f"Perminov {label} tensor invalid")
        order = retained[f"perminov_{label}"]["schedules"]["liveness"]["order"]
        circuits[label] = render(graph, order, optimized=optimized)
    if circuits["55"][0] != BASELINE.read_text():
        raise ValueError("matched generator does not reproduce frozen 55 baseline byte-for-byte")
    if (circuits["55"][1]["dots_by_block"] != [8, 6, 6, 12, 3, 10, 9, 3, 7] or
        circuits["58"][1]["dots_by_block"] != [8, 6, 6, 11, 3, 10, 5, 3, 6] or
        sum(circuits["55"][1]["gates_by_block"]) != 176 or
        sum(circuits["58"][1]["gates_by_block"]) != 142):
        raise ValueError("frozen dependency-slice work changed")
    return circuits["58"]


if __name__ == "__main__":
    text, stats = generate()
    if TARGET.exists():
        if TARGET.read_text() != text:
            raise SystemExit("generated 58 kernel differs from verified source")
        print(f"PASS {sha256(TARGET)} {stats}")
    else:
        TARGET.write_text(text)
        print(f"WROTE {sha256(TARGET)} {stats}")
