"""Generate and check the single output-block-owned Perminov mapping."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "examples/kernels/mi_research_6c_fused_perminov55.py"
TARGET = ROOT / "examples/kernels/mi_research_7b_fused_perminov55_output_block.py"
ORIGINAL_SHA256 = "c9b2d61fc6a1b6a658f1198d1a3938fae154efce2029c463e439b7fabbfaf8db"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render() -> str:
    if digest(ORIGINAL) != ORIGINAL_SHA256:
        raise ValueError("frozen Perminov source changed")
    source = ORIGINAL.read_text()
    tree = ast.parse(source)
    kernel = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_fused")
    assignments = {node.targets[0].id: node for node in kernel.body
                   if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
    stores = [node for node in kernel.body if isinstance(node, ast.Expr)
              and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
              and node.value.func.attr == "store"]
    if len(stores) != 9 or len(assignments) != 99:
        raise ValueError("frozen circuit structure changed")
    by_block = {}
    for node in stores:
        statement = ast.get_source_segment(source, node)
        block = tuple(map(int, re.search(r"\(([012]) \* 32 \+ row.*\+ ([012]) \* 32 \+ col", statement).groups()))
        by_block[block] = node
    if set(by_block) != {(i, j) for i in range(3) for j in range(3)}:
        raise ValueError("output blocks changed")
    lines = ["\"\"\"Generated output-block ownership of the frozen Perminov 55-gate circuit.",
             "", "Run scripts/generate_mi_research_7b_kernel.py to check the source slices.",
             "\"\"\"", "", "import torch", "import triton", "import triton.language as tl", "",
             "", "@triton.jit", "def _fused(a_ptr, b_ptr, c_ptr):",
             "    row = tl.program_id(0) * 16 + tl.arange(0, 16)",
             "    col = tl.program_id(1) * 16 + tl.arange(0, 16)",
             "    inner = tl.arange(0, 32)", "    output_block = tl.program_id(2)"]
    counts = []
    for block in sorted(by_block):
        store = by_block[block]
        needed = {node.id for node in ast.walk(store) if isinstance(node, ast.Name)}
        selected = []
        for node in reversed(kernel.body):
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name in needed:
                    selected.append(node)
                    needed.update(child.id for child in ast.walk(node.value) if isinstance(child, ast.Name))
        selected.reverse()
        dots = sum("tl.dot" in ast.get_source_segment(source, node) for node in selected)
        gates = sum(node.targets[0].id.startswith(("Ug", "Vg", "Wg")) for node in selected)
        counts.append((dots, gates))
        lines.append(f"    if output_block == {3 * block[0] + block[1]}:")
        lines.extend("        " + ast.get_source_segment(source, node) for node in selected)
        lines.append("        " + ast.get_source_segment(source, store))
    if [x[0] for x in counts] != [8, 6, 6, 12, 3, 10, 9, 3, 7] or sum(x[1] for x in counts) != 176:
        raise ValueError(f"source slice counts changed: {counts}")
    lines += ["", "", "def call(left, right, output):",
              "    if (left.shape != (96, 96) or right.shape != (96, 96) or",
              "            output.shape != (96, 96) or left.dtype != torch.float16 or",
              "            right.dtype != torch.float16 or output.dtype != torch.float16 or",
              "            not left.is_cuda or not right.is_cuda or not output.is_cuda or",
              "            not left.is_contiguous() or not right.is_contiguous() or",
              "            not output.is_contiguous()):",
              "        raise ValueError(\"expected contiguous CUDA FP16 96x96 inputs/output\")",
              "    return _fused[(2, 2, 9)](left, right, output, num_warps=4, num_stages=2)"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    expected = render()
    if TARGET.exists():
        if TARGET.read_text() != expected:
            raise SystemExit("generated 7B kernel differs from frozen source slices")
        print(f"PASS {digest(TARGET)}")
    else:
        TARGET.write_text(expected)
        print(f"WROTE {digest(TARGET)}")
