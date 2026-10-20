"""Reproduce and diagnose MI-Research-3 numerical gates without timing kernels."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

from matmul_inspector import experiment_pair


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "examples/kernels/mi_research_2_split2.py"
CONVENTIONAL_SOURCE = ROOT / "src/matmul_inspector/triton_fc1.py"
PRIOR = ROOT / "artifacts/mi-research-3/evidence-v1"
MS = (32, 128, 256, 512)
K, N = 4096, 2048
PARAMETERS = {"block_m": 64, "block_n": 128, "block_k": 32,
              "num_warps": 4, "num_stages": 3, "group_size_m": 1}
TOLERANCE = {"atol": 0.01, "rtol": 0.01}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def representative_indices(torch, error, fail, n: int = 4) -> list[tuple[int, int]]:
    """First failing and largest-error coordinates, with stable deduplication."""
    indices = []
    first = torch.nonzero(fail, as_tuple=False)[:n].cpu().tolist()
    top = torch.topk(error.flatten(), k=min(n, error.numel())).indices.cpu().tolist()
    for row, col in first + [(flat // error.shape[1], flat % error.shape[1]) for flat in top]:
        pair = (int(row), int(col))
        if pair not in indices:
            indices.append(pair)
    return indices


def summarize_error(torch, actual, reference, *, a=None, b=None) -> dict:
    """The established allclose gate plus reproducible distribution diagnostics."""
    check = experiment_pair.check_output(torch, reference, actual, TOLERANCE)
    finite = torch.isfinite(actual) & torch.isfinite(reference)
    error = (actual.float() - reference.float()).abs()
    fail = (~finite) | (error > TOLERANCE["atol"] + TOLERANCE["rtol"] * reference.float().abs())
    relative_domain = finite & (reference.float().abs() >= 1.0)
    relative = error[relative_domain] / reference.float().abs()[relative_domain]
    locations = []
    for row, col in representative_indices(torch, error, fail):
        item = {"row": row, "col": col, "actual": float(actual[row, col].item()),
                "reference": float(reference[row, col].item()),
                "absolute_error": float(error[row, col].item()),
                "failed": bool(fail[row, col].item())}
        if a is not None and b is not None:
            exact = float(torch.dot(a[row].double(), b[:, col].double()).item())
            item["fp64_dot"] = exact
            item["fp64_dot_rounded_fp16"] = float(torch.tensor(exact, dtype=torch.float16).item())
        locations.append(item)
    return {**check, "failing_elements": int(fail.sum().item()),
            "failing_fraction": float(fail.float().mean().item()),
            "max_relative_error_reference_abs_ge_1":
                float(relative.max().item()) if relative.numel() else None,
            "representatives": locations}


def compare_candidates(torch, a_out, b_out, reference) -> dict:
    a_fail = (a_out.float() - reference.float()).abs() > (
        TOLERANCE["atol"] + TOLERANCE["rtol"] * reference.float().abs())
    b_fail = (b_out.float() - reference.float()).abs() > (
        TOLERANCE["atol"] + TOLERANCE["rtol"] * reference.float().abs())
    both = int((a_fail & b_fail).sum().item())
    either = int((a_fail | b_fail).sum().item())
    difference = (a_out.float() - b_out.float()).abs()
    return {"a_failure_count": int(a_fail.sum().item()),
            "b_failure_count": int(b_fail.sum().item()),
            "same_failing_coordinates": both,
            "union_failing_coordinates": either,
            "failure_jaccard": (both / either if either else None),
            "a_b_exact_equal_count": int((a_out == b_out).sum().item()),
            "a_b_max_absolute_difference": float(difference.max().item()),
            "a_b_mean_absolute_difference": float(difference.mean().item()),
            "a_b_allclose_existing_tolerance": bool(torch.allclose(
                a_out, b_out, atol=TOLERANCE["atol"], rtol=TOLERANCE["rtol"]))}


def verify(output: Path) -> dict:
    manifest = json.loads((output / "hash_manifest.json").read_text())
    if not isinstance(manifest, dict) or any(Path(p).is_absolute() or ".." in Path(p).parts for p in manifest):
        raise ValueError("invalid manifest")
    actual = {str(p.relative_to(output)) for p in output.rglob("*")
              if p.is_file() and p.name != "hash_manifest.json"}
    if actual != set(manifest) or any(digest(output / name) != value for name, value in manifest.items()):
        raise ValueError("manifest mismatch")
    raw = json.loads((output / "raw.json").read_text())
    for key, name in (("source_sha256", "source.py"),
                      ("conventional_source_sha256", "conventional_source.py"),
                      ("script_sha256", "diagnostic_script.py")):
        if raw[key] != digest(output / name):
            raise ValueError(f"{name} identity mismatch")
    if raw["source_sha256"] != "0b9144b81eed225c671518a1d7067f115865c32437ee66fe2db1b51309aa59e9":
        raise ValueError("Split-2 source differs from committed baseline")
    replay = summarize(raw)
    if replay != json.loads((output / "summary.json").read_text()):
        raise ValueError("summary does not replay")
    return {"status": "PASS", "cases": len(raw["shape_seed_cases"]),
            "controlled_shapes": len(raw["controlled_prefix_cases"])}


def summarize(raw: dict) -> dict:
    if (raw.get("schema_version") != 1 or raw.get("k") != K or raw.get("n") != N or
            raw.get("m_values") != list(MS) or raw.get("tolerance") != TOLERANCE or
            [(c["m"], c["seed"]) for c in raw.get("shape_seed_cases", [])] !=
            [(m, seed) for m in MS for seed in range(3)] or
            [c["m"] for c in raw.get("controlled_prefix_cases", [])] != list(MS)):
        raise ValueError("correctness study configuration differs")
    by_shape = {}
    for m in MS:
        cases = [c for c in raw["shape_seed_cases"] if c["m"] == m]
        by_shape[str(m)] = {
            "eager_gate_by_seed": {str(c["seed"]): {label: c["eager"][label]["passed"]
                                                  for label in ("A", "B")}
                                   for c in cases},
            "failing_elements_by_seed": {str(c["seed"]): {label: c["eager"][label]["failing_elements"]
                                                        for label in ("A", "B")}
                                         for c in cases},
            "graph_gate_seed0": {label: cases[0]["graph"][label]["passed"]
                                 for label in ("A", "B")},
            "graph_equals_eager_seed0": cases[0]["graph_equals_eager"]}
    return {"schema_version": 1, "shape_results": by_shape,
            "controlled_prefix_reference_difference": {
                str(c["m"]): c["reference_vs_m512_prefix"]
                for c in raw["controlled_prefix_cases"]}}


def run(output: Path) -> dict:
    if output.exists():
        raise ValueError("output already exists")
    if digest(SOURCE) != "0b9144b81eed225c671518a1d7067f115865c32437ee66fe2db1b51309aa59e9":
        raise ValueError("Split-2 source changed")
    output.mkdir(parents=True)
    for source, name in ((SOURCE, "source.py"), (CONVENTIONAL_SOURCE, "conventional_source.py"),
                         (Path(__file__), "diagnostic_script.py")):
        shutil.copyfile(source, output / name)
    import torch
    import triton
    from matmul_inspector import triton_fc1
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    module_spec = importlib.util.spec_from_file_location("mi_research_3b_split2", output / "source.py")
    module = importlib.util.module_from_spec(module_spec)
    old_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        module_spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = old_bytecode
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    reduced_precision_default = bool(torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)
    fp16_accumulation_default = getattr(torch.backends.cuda.matmul, "allow_fp16_accumulation", None)

    def shape_inputs(m, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        return (torch.randn((m, K), device="cuda", dtype=torch.float16),
                torch.randn((K, N), device="cuda", dtype=torch.float16))

    def kernels(a, b, scratch=None):
        m = a.shape[0]
        out_a = torch.empty((m, N), device="cuda", dtype=torch.float16)
        out_b = torch.empty_like(out_a)
        a_grid = (triton.cdiv(m, 64) * triton.cdiv(N, 128),)
        def conventional():
            return triton_fc1.fc1_kernel[a_grid](
                a, b, out_a, m, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                out_a.stride(0), out_a.stride(1), BLOCK_M=64, BLOCK_N=128, BLOCK_K=32,
                GROUP_SIZE_M=1, num_warps=4, num_stages=3)
        def split_eager():
            return module.matmul_into(a, b, out_b, PARAMETERS)
        def split_reuse():
            partial = module.split_k_partial[(a_grid[0], 2)](
                a, b, scratch, m, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                BLOCK_M=64, BLOCK_N=128, BLOCK_K=32, K_PART=K // 2,
                num_warps=4, num_stages=3)
            reduction = module.reduce_partials[(triton.cdiv(m * N, 1024),)](
                scratch, out_b, m, N, out_b.stride(0), out_b.stride(1), BLOCK=1024, num_warps=4)
            return partial, reduction
        return out_a, out_b, conventional, split_eager, split_reuse

    shape_cases = []
    for m in MS:
        for seed in range(3):
            a, b = shape_inputs(m, seed)
            reference = torch.matmul(a, b)
            out_a, out_b, conventional, split_eager, split_reuse = kernels(a, b)
            out_a.fill_(float("nan"))
            out_b.fill_(float("nan"))
            conventional()
            split_eager()
            torch.cuda.synchronize()
            eager_a, eager_b = out_a.clone(), out_b.clone()
            row = {"m": m, "seed": seed, "input_strides": {"a": list(a.stride()), "b": list(b.stride())},
                   "output_strides": {"a": list(out_a.stride()), "b": list(out_b.stride())},
                   "eager": {"A": summarize_error(torch, eager_a, reference, a=a, b=b),
                             "B": summarize_error(torch, eager_b, reference, a=a, b=b)},
                   "a_b": compare_candidates(torch, eager_a, eager_b, reference)}
            if seed == 0:
                # Diagnostic references do not replace the established FP16 gate.
                reference_fp32 = torch.matmul(a.float(), b.float()).half()
                row["reference_controls"] = {
                    "fp16_cublas_vs_fp32_rounded": summarize_error(torch, reference, reference_fp32),
                    "A_vs_fp32_rounded": summarize_error(torch, eager_a, reference_fp32),
                    "B_vs_fp32_rounded": summarize_error(torch, eager_b, reference_fp32)}
                torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
                try:
                    reference_no_reduced = torch.matmul(a, b)
                finally:
                    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = reduced_precision_default
                row["reference_controls"]["no_reduced_precision_vs_default"] = summarize_error(
                    torch, reference_no_reduced, reference)
                row["reference_controls"]["no_reduced_precision_vs_fp32_rounded"] = summarize_error(
                    torch, reference_no_reduced, reference_fp32)

                # One complete operation per graph. No timing is recorded.
                scratch = torch.empty((2, m, N), device="cuda", dtype=torch.float32)
                graph_a, graph_b, graph_conventional, _, graph_split_reuse = kernels(a, b, scratch)
                graphs = {}
                for label, function in (("A", graph_conventional), ("B", graph_split_reuse)):
                    if label == "B":
                        scratch.fill_(float("nan"))
                    function()  # compile/warm before capture
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        function()
                    graphs[label] = graph
                graph_a.fill_(float("nan"))
                graph_b.fill_(float("nan"))
                scratch.fill_(float("nan"))
                for label in ("A", "B"):
                    graphs[label].replay()
                torch.cuda.synchronize()
                row["graph"] = {"A": summarize_error(torch, graph_a, reference),
                                "B": summarize_error(torch, graph_b, reference)}
                row["graph_equals_eager"] = {"A": bool(torch.equal(graph_a, eager_a)),
                                              "B": bool(torch.equal(graph_b, eager_b))}
                row["scratch_after_graph"] = {
                    "nonfinite_elements": int((~torch.isfinite(scratch)).sum().item()),
                    "reduction_matches_output_exactly": bool(torch.equal(
                        (scratch[0] + scratch[1]).half(), graph_b))}
            shape_cases.append(row)

    # Same mathematical outputs across M: fixed B and nested prefixes of A.
    a_big, b_shared = shape_inputs(512, 13)
    full_reference = torch.matmul(a_big, b_shared)
    controlled = []
    for m in MS:
        a = a_big[:m]
        reference = torch.matmul(a, b_shared)
        out_a, out_b, conventional, split_eager, _ = kernels(a, b_shared)
        conventional()
        split_eager()
        torch.cuda.synchronize()
        controlled.append({"m": m, "input_strides": {"a": list(a.stride()), "b": list(b_shared.stride())},
                           "reference_vs_m512_prefix": summarize_error(torch, reference, full_reference[:m]),
                           "A_vs_m512_reference_prefix": summarize_error(torch, out_a, full_reference[:m]),
                           "B_vs_m512_reference_prefix": summarize_error(torch, out_b, full_reference[:m]),
                           "A_vs_own_reference": summarize_error(torch, out_a, reference),
                           "B_vs_own_reference": summarize_error(torch, out_b, reference)})
    props = torch.cuda.get_device_properties(0)
    raw = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
           "m_values": list(MS), "k": K, "n": N, "tolerance": TOLERANCE,
           "source_sha256": digest(SOURCE), "conventional_source_sha256": digest(CONVENTIONAL_SOURCE),
           "script_sha256": digest(Path(__file__)),
           "historical_mi_research_3_manifest_sha256": digest(PRIOR / "SHA256SUMS"),
           "environment": {"gpu": props.name, "sm_count": props.multi_processor_count,
                           "torch": torch.__version__, "triton": triton.__version__,
                           "cuda_runtime": torch.version.cuda,
                           "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                           "allow_fp16_reduced_precision_reduction_default": reduced_precision_default,
                           "allow_fp16_accumulation_default": fp16_accumulation_default},
           "shape_seed_cases": shape_cases, "controlled_prefix_seed": 13,
           "controlled_prefix_cases": controlled}
    write_json(output / "raw.json", raw)
    write_json(output / "summary.json", summarize(raw))
    write_json(output / "hash_manifest.json", {
        str(p.relative_to(output)): digest(p) for p in sorted(output.rglob("*"))
        if p.is_file() and p.name != "hash_manifest.json"})
    return verify(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    result = verify(args.output) if args.verify else run(args.output)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
