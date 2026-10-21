"""All-shape correctness gate for the explicit MI-Research-3C reference."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import shutil
import sys

from matmul_inspector import experiment_pair
try:
    from scripts import run_mi_research_2c_graph as artifacts
    from scripts import run_mi_research_3b_correctness as diagnostics
except ModuleNotFoundError:
    import run_mi_research_2c_graph as artifacts
    import run_mi_research_3b_correctness as diagnostics


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "examples/kernels/mi_research_2_split2.py"
CONVENTIONAL = ROOT / "src/matmul_inspector/triton_fc1.py"
EXPECTED_SOURCE = "0b9144b81eed225c671518a1d7067f115865c32437ee66fe2db1b51309aa59e9"
MS, K, N = (32, 128, 256, 512), 4096, 2048
TOLERANCE = {"atol": 0.01, "rtol": 0.01}


def summarize(raw: dict) -> dict:
    if (raw.get("schema_version") != 1 or raw.get("reference_policy") != "fp32_rounded_fp16" or
            raw.get("shape") != {"m_values": list(MS), "k": K, "n": N} or
            raw.get("tolerance") != TOLERANCE or
            [(c["m"], c["seed"]) for c in raw.get("cases", [])] !=
            [(m, seed) for m in MS for seed in range(3)]):
        raise ValueError("invalid reference preflight configuration")
    by_shape = {}
    for m in MS:
        cases = [row for row in raw["cases"] if row["m"] == m]
        by_shape[str(m)] = {
            str(row["seed"]): {label: row["checks"][label]["passed"] for label in ("A", "B", "C")}
            for row in cases}
    return {"schema_version": 1, "reference_policy": "fp32_rounded_fp16",
            "all_passed": all(all(case["checks"][label]["passed"] for label in ("A", "B", "C"))
                              for case in raw["cases"]), "by_shape": by_shape}


def verify(output: Path) -> dict:
    manifest = json.loads((output / "hash_manifest.json").read_text())
    actual = {str(p.relative_to(output)) for p in output.rglob("*")
              if p.is_file() and p.name != "hash_manifest.json"}
    if (not isinstance(manifest, dict) or any(Path(p).is_absolute() or ".." in Path(p).parts for p in manifest)
            or actual != set(manifest) or any(artifacts.digest(output / p) != value
                                             for p, value in manifest.items())):
        raise ValueError("reference preflight manifest mismatch")
    raw = json.loads((output / "raw.json").read_text())
    for key, name in (("source_sha256", "source.py"),
                      ("conventional_sha256", "conventional_source.py"),
                      ("script_sha256", "diagnostic_script.py")):
        if raw[key] != artifacts.digest(output / name):
            raise ValueError(f"{name} identity mismatch")
    if raw["source_sha256"] != EXPECTED_SOURCE:
        raise ValueError("Split-2 source is not the committed kernel")
    replay = summarize(raw)
    if replay != json.loads((output / "summary.json").read_text()):
        raise ValueError("reference preflight summary does not replay")
    return {"status": "PASS" if replay["all_passed"] else "CORRECTNESS_FAILED",
            "cases": len(raw["cases"])}


def run(output: Path) -> dict:
    if output.exists():
        raise ValueError("output already exists")
    if artifacts.digest(SOURCE) != EXPECTED_SOURCE:
        raise ValueError("Split-2 source changed")
    output.mkdir(parents=True)
    for src, name in ((SOURCE, "source.py"), (CONVENTIONAL, "conventional_source.py"),
                      (Path(__file__), "diagnostic_script.py")):
        shutil.copyfile(src, output / name)
    import torch
    import triton
    from matmul_inspector import triton_fc1
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    module_spec = importlib.util.spec_from_file_location("mi_research_3c_split2", output / "source.py")
    module = importlib.util.module_from_spec(module_spec)
    old_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        module_spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = old_bytecode
    original_reduced = bool(torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)
    original_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
    original_float32 = torch.get_float32_matmul_precision()
    cases = []
    try:
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        for m in MS:
            for seed in range(3):
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                a = torch.randn((m, K), device="cuda", dtype=torch.float16)
                b = torch.randn((K, N), device="cuda", dtype=torch.float16)
                reference = torch.matmul(a.float(), b.float()).half()
                cublas_no_reduced = torch.matmul(a, b)
                torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = original_reduced
                try:
                    cublas_default = torch.matmul(a, b)
                finally:
                    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
                out_a = torch.empty((m, N), device="cuda", dtype=torch.float16)
                out_b = torch.empty_like(out_a)
                grid = (triton.cdiv(m, 64) * triton.cdiv(N, 128),)
                triton_fc1.fc1_kernel[grid](
                    a, b, out_a, m, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                    out_a.stride(0), out_a.stride(1), BLOCK_M=64, BLOCK_N=128, BLOCK_K=32,
                    GROUP_SIZE_M=1, num_warps=4, num_stages=3)
                module.matmul_into(a, b, out_b, diagnostics.PARAMETERS)
                torch.cuda.synchronize()
                candidates = {"A": out_a, "B": out_b, "C": cublas_no_reduced,
                              "default_cublas": cublas_default}
                checks = {label: diagnostics.summarize_error(
                    torch, value, reference, a=a if label in ("A", "B", "default_cublas") else None,
                    b=b if label in ("A", "B", "default_cublas") else None)
                    for label, value in candidates.items()}
                cases.append({"m": m, "seed": seed, "checks": checks,
                              "input_strides": {"a": list(a.stride()), "b": list(b.stride())}})
    finally:
        torch.set_float32_matmul_precision(original_float32)
        torch.backends.cuda.matmul.allow_tf32 = original_tf32
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = original_reduced
    props = torch.cuda.get_device_properties(0)
    raw = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
           "reference_policy": "fp32_rounded_fp16", "tolerance": TOLERANCE,
           "shape": {"m_values": list(MS), "k": K, "n": N}, "cases": cases,
           "source_sha256": artifacts.digest(SOURCE),
           "conventional_sha256": artifacts.digest(CONVENTIONAL),
           "script_sha256": artifacts.digest(Path(__file__)),
           "historical_manifest_sha256": artifacts.digest(
               ROOT / "artifacts/mi-research-3/evidence-v1/SHA256SUMS"),
           "precision_settings": {"original_fp16_reduced_precision_reduction": original_reduced,
                                  "active_fp16_reduced_precision_reduction": False,
                                  "original_allow_tf32": original_tf32, "active_allow_tf32": False,
                                  "original_float32_matmul_precision": original_float32,
                                  "active_float32_matmul_precision": "highest",
                                  "restored_after_run": bool(
                                      torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction == original_reduced
                                      and torch.backends.cuda.matmul.allow_tf32 == original_tf32
                                      and torch.get_float32_matmul_precision() == original_float32)},
           "environment": {"gpu": props.name, "sm_count": props.multi_processor_count,
                           "torch": torch.__version__, "triton": triton.__version__,
                           "cuda_runtime": torch.version.cuda}}
    artifacts.write_json(output / "raw.json", raw)
    artifacts.write_json(output / "summary.json", summarize(raw))
    artifacts.manifest(output)
    return verify(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    result = verify(args.output) if args.verify else run(args.output)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
