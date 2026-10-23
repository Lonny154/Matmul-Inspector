"""Render the unmeasured four-CTA Perminov-58 circuit from the certified DAG."""

from __future__ import annotations

import json
from pathlib import Path

try:
    from scripts import run_mi_research_6c as exact
    from scripts import generate_mi_research_8_kernel as sliced
except ModuleNotFoundError:
    import run_mi_research_6c as exact
    import generate_mi_research_8_kernel as sliced

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "artifacts/mi-research-10/perminov58_four_cta.py"


def render(graph: exact.Circuit, order: list[str], label: str) -> str:
    if label not in ("55", "58"):
        raise ValueError("expected a certified Perminov circuit")
    if sum(node["kind"] == "ADD" for node in graph.nodes) != int(label):
        raise ValueError("addition count changed")
    if sum(node["kind"] == "PRODUCT" for node in graph.nodes) != 23:
        raise ValueError("rank changed")
    lines = [
        f'"""MI-Research-6C: one fixed N=96 output-owned Perminov {label}-gate probe.',
        "", "Generated from the independently checked frozen circuit. Four programs",
        "each own one local 16x16 tile of all nine output blocks. No atomics or",
        "rank-product buffers are used. This is a feasibility kernel, not a",
        "performance-tuned GEMM.", '"""', "", "import torch", "import triton",
        "import triton.language as tl", "", "", "@triton.jit",
        "def _fused(a_ptr, b_ptr, c_ptr):",
        "    row = tl.program_id(0) * 16 + tl.arange(0, 16)",
        "    col = tl.program_id(1) * 16 + tl.arange(0, 16)",
        "    inner = tl.arange(0, 32)",
    ]
    lines.extend("    " + sliced._load(f"{side}{index}")
                 for side in ("A", "B") for index in range(9))
    lines.extend("    " + sliced._statement(graph.by_id[name]) for name in order)
    lines += [
        "", "", "def call(left, right, output):",
        "    if (left.shape != (96, 96) or right.shape != (96, 96) or",
        "            output.shape != (96, 96) or left.dtype != torch.float16 or",
        "            right.dtype != torch.float16 or output.dtype != torch.float16 or",
        "            not left.is_cuda or not right.is_cuda or not output.is_cuda or",
        "            not left.is_contiguous() or not right.is_contiguous() or",
        "            not output.is_contiguous()):",
        '        raise ValueError("expected contiguous CUDA FP16 96x96 inputs/output")',
        "    return _fused[(2, 2)](left, right, output, num_warps=4, num_stages=2)",
        "",
    ]
    return "\n".join(lines)


def generate() -> str:
    if exact.verify()["status"] != "PASS":
        raise ValueError("MI-Research-6C evidence changed")
    source, cert = json.loads(exact.SOURCE.read_text()), json.loads(exact.CERT.read_text())
    retained = json.loads((exact.OUT / "analysis.json").read_text())["configurations"]
    texts = {}
    for label in ("55", "58"):
        graph, refs = exact.build(source, cert, optimized=label == "55")
        if exact.factor_rows(graph, refs) != retained[f"perminov_{label}"]["factor_coefficients"]:
            raise ValueError("factor map changed")
        order = retained[f"perminov_{label}"]["schedules"]["liveness"]["order"]
        texts[label] = render(graph, order, label)
    if texts["55"] != exact.PROBE_SOURCE.read_text():
        raise ValueError("generator does not reproduce historical four-CTA control")
    return texts["58"]


if __name__ == "__main__":
    text = generate()
    if TARGET.exists():
        if TARGET.read_text() != text:
            raise SystemExit("candidate source changed")
        print(f"PASS {sliced.sha256(TARGET)}")
    else:
        TARGET.parent.mkdir(parents=True, exist_ok=True)
        TARGET.write_text(text)
        print(f"WROTE {sliced.sha256(TARGET)}")
