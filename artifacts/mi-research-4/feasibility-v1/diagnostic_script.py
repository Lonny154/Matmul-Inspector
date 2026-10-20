"""Bounded correctness, compiler, and launch-topology gate for MI-Research-4.

This is not a benchmark. Performance uses the existing trusted-local pair CLI
only after this gate passes.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from matmul_inspector import experiment_pair
try:
    from scripts import run_mi_research_2c_graph as artifacts
    from scripts import run_mi_research_3b_correctness as diagnostics
except ModuleNotFoundError:
    import run_mi_research_2c_graph as artifacts
    import run_mi_research_3b_correctness as diagnostics


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "examples/kernels/mi_research_4_single_kernel_strassen.py"
CONVENTIONAL = ROOT / "src/matmul_inspector/triton_fc1.py"
SIZES = (64, 128)
CASE_NAMES = ("seed-0", "seed-1", "seed-2", "zero", "identity", "cancellation")
TOLERANCE = {"atol": 0.01, "rtol": 0.01}
BASELINE_PARAMETERS = {"block_m": 32, "block_n": 32, "block_k": 32,
                       "num_warps": 4, "num_stages": 2, "group_size_m": 1}


def profile_summary(trace: dict, label: str) -> dict:
    events = trace.get("traceEvents")
    if not isinstance(events, list):
        raise ValueError("profiler trace has no events")
    kernels = [row["name"] for row in sorted(
        (item for item in events if item.get("cat") == "kernel"),
        key=lambda item: item["ts"])]
    launches = sum(item.get("cat") == "cuda_runtime" and
                   item.get("name") in ("cudaLaunchKernel", "cudaLaunchKernelExC")
                   for item in events)
    expected = "fc1_kernel" if label == "A" else "_strassen_kernel"
    return {"kernel_names": kernels, "kernel_events": len(kernels),
            "cuda_launch_calls": launches, "expected_kernel_seen":
            len(kernels) == 1 and expected in kernels[0]}


def summarize(raw: dict) -> dict:
    if (raw.get("schema_version") != 1 or raw.get("sizes") != list(SIZES) or
            raw.get("case_names") != list(CASE_NAMES) or raw.get("tolerance") != TOLERANCE or
            [(row.get("size"), row.get("case")) for row in raw.get("cases", [])] !=
            [(size, name) for size in SIZES for name in CASE_NAMES] or
            raw.get("precision", {}).get("fp32_matmul_precision") != "highest" or
            raw.get("precision", {}).get("allow_tf32") is not False or
            raw.get("precision", {}).get("allow_fp16_reduced_precision_reduction") is not False):
        raise ValueError("invalid MI-Research-4 correctness protocol")
    checks = raw["cases"]
    passed = all(row["checks"][label]["passed"] is True for row in checks for label in ("A", "B"))
    topology = all(raw["profiles"][str(size)][label]["kernel_events"] == 1 and
                   raw["profiles"][str(size)][label]["cuda_launch_calls"] == 1 and
                   raw["profiles"][str(size)][label]["expected_kernel_seen"] is True
                   for size in SIZES for label in ("A", "B"))
    compiler = all(raw["compiler"][str(size)][label]["metadata"].get("n_spills") == 0 and
                   raw["compiler"][str(size)][label]["ptx_sha256"]
                   for size in SIZES for label in ("A", "B"))
    return {"schema_version": 1, "correctness_passed": passed,
            "topology_passed": topology, "zero_reported_spills": bool(compiler),
            "performance_gate_passed": bool(passed and topology and compiler),
            "cases": len(checks), "sizes": list(SIZES)}


def verify(output: Path) -> dict:
    manifest = json.loads((output / "hash_manifest.json").read_text())
    actual = {str(path.relative_to(output)) for path in output.rglob("*")
              if path.is_file() and path.name != "hash_manifest.json"}
    if (not isinstance(manifest, dict) or actual != set(manifest) or
            any(Path(name).is_absolute() or ".." in Path(name).parts or
                artifacts.digest(output / name) != digest for name, digest in manifest.items())):
        raise ValueError("MI-Research-4 manifest mismatch")
    if (output / "failure.json").exists():
        failure = json.loads((output / "failure.json").read_text())
        return {"status": "FAILURE_RETAINED", "type": failure["type"]}
    raw = json.loads((output / "raw.json").read_text())
    for field, name in (("source_sha256", "source.py"),
                        ("conventional_sha256", "conventional_source.py"),
                        ("script_sha256", "diagnostic_script.py")):
        if raw[field] != artifacts.digest(output / name):
            raise ValueError(f"{name} hash mismatch")
    for size in SIZES:
        for label in ("A", "B"):
            trace_name = f"n{size}_{label}_trace.json"
            trace = json.loads((output / trace_name).read_text())
            if (raw["profiles"][str(size)][label] != profile_summary(trace, label) or
                    raw["trace_sha256"][str(size)][label] != artifacts.digest(output / trace_name) or
                    raw["compiler"][str(size)][label]["ptx_sha256"] !=
                    artifacts.digest(output / f"n{size}_{label}.ptx")):
                raise ValueError("compiler or topology evidence mismatch")
    replay = summarize(raw)
    if replay != json.loads((output / "summary.json").read_text()):
        raise ValueError("MI-Research-4 summary does not replay")
    return {"status": "PASS" if replay["performance_gate_passed"] else "GATE_FAILED",
            **replay}


def _inputs(torch, size: int, name: str):
    if name.startswith("seed-"):
        seed = int(name.split("-")[1])
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        return (torch.randn((size, size), device="cuda", dtype=torch.float16),
                torch.randn((size, size), device="cuda", dtype=torch.float16))
    if name == "zero":
        return (torch.zeros((size, size), device="cuda", dtype=torch.float16),
                torch.zeros((size, size), device="cuda", dtype=torch.float16))
    torch.manual_seed(31)
    torch.cuda.manual_seed_all(31)
    if name == "identity":
        return (torch.eye(size, device="cuda", dtype=torch.float16),
                torch.randn((size, size), device="cuda", dtype=torch.float16))
    if name == "cancellation":
        half = size // 2
        x = torch.randn((half, half), device="cuda", dtype=torch.float16)
        y = torch.randn((half, half), device="cuda", dtype=torch.float16)
        a = torch.cat((torch.cat((x, -x), dim=1), torch.cat((x, -x), dim=1)), dim=0)
        b = torch.cat((torch.cat((y, -y), dim=1), torch.cat((-y, y), dim=1)), dim=0)
        return a, b
    raise ValueError("unknown correctness case")


def run(output: Path) -> dict:
    if output.exists():
        raise ValueError("output already exists")
    output.mkdir(parents=True)
    for src, name in ((SOURCE, "source.py"), (CONVENTIONAL, "conventional_source.py"),
                      (Path(__file__), "diagnostic_script.py")):
        shutil.copyfile(src, output / name)
    try:
        result = _run_gpu(output)
    except Exception as error:
        artifacts.write_json(output / "failure.json", {
            "type": type(error).__name__, "message": str(error),
            "source_sha256": artifacts.digest(output / "source.py"),
            "script_sha256": artifacts.digest(output / "diagnostic_script.py")})
        artifacts.manifest(output)
        raise
    return result


def _run_gpu(output: Path) -> dict:
    import torch
    import triton
    from matmul_inspector import triton_fc1
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    module_spec = importlib.util.spec_from_file_location("mi_research_4_strassen", output / "source.py")
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError("cannot import saved source")
    module = importlib.util.module_from_spec(module_spec)
    old_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        module_spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = old_bytecode
    original = {"allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                "allow_fp16_reduced_precision_reduction": bool(
                    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction),
                "float32_matmul_precision": torch.get_float32_matmul_precision()}
    cases, compiler, profiles, trace_hashes, first_launch_ms = [], {}, {}, {}, {}
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        for size in SIZES:
            a, b = _inputs(torch, size, "seed-0")
            outputs = {label: torch.empty((size, size), device="cuda", dtype=torch.float16)
                       for label in ("A", "B")}

            def conventional():
                out = outputs["A"]
                return triton_fc1.fc1_kernel[(triton.cdiv(size, 32) * triton.cdiv(size, 32),)](
                    a, b, out, size, size, size,
                    a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                    out.stride(0), out.stride(1), BLOCK_M=32, BLOCK_N=32, BLOCK_K=32,
                    GROUP_SIZE_M=1, num_warps=4, num_stages=2)

            def strassen():
                return module.matmul_into(a, b, outputs["B"], module.PARAMETERS)

            functions = {"A": conventional, "B": strassen}
            compiler[str(size)], first_launch_ms[str(size)] = {}, {}
            for label, function in functions.items():
                torch.cuda.synchronize()
                start = time.perf_counter_ns()
                launch = function()
                torch.cuda.synchronize()
                first_launch_ms[str(size)][label] = (time.perf_counter_ns() - start) / 1e6
                capture = experiment_pair._compiled_capture(launch)
                if capture is None or not isinstance(capture.get("ptx_text"), str):
                    raise RuntimeError(f"N={size} {label} compiler PTX unavailable")
                ptx_name = f"n{size}_{label}.ptx"
                (output / ptx_name).write_text(capture.pop("ptx_text"))
                capture.pop("sass_text", None)
                capture["ptx_sha256"] = artifacts.digest(output / ptx_name)
                compiler[str(size)][label] = capture
            for name in CASE_NAMES:
                left, right = _inputs(torch, size, name)
                a.copy_(left)
                b.copy_(right)
                reference = torch.matmul(a.float(), b.float()).half()
                cublas_no_reduced = torch.matmul(a, b)
                for out in outputs.values():
                    out.fill_(float("nan"))
                for function in functions.values():
                    function()
                torch.cuda.synchronize()
                checks = {label: diagnostics.summarize_error(torch, value, reference, a=a, b=b)
                          for label, value in outputs.items()}
                checks["cublas_no_reduced"] = diagnostics.summarize_error(
                    torch, cublas_no_reduced, reference)
                cases.append({"size": size, "case": name, "checks": checks})
            profiles[str(size)], trace_hashes[str(size)] = {}, {}
            a, b = _inputs(torch, size, "seed-0")
            for label, function in functions.items():
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                        torch.profiler.ProfilerActivity.CUDA]) as profiler:
                    function()
                    torch.cuda.synchronize()
                path = output / f"n{size}_{label}_trace.json"
                profiler.export_chrome_trace(str(path))
                profiles[str(size)][label] = profile_summary(json.loads(path.read_text()), label)
                trace_hashes[str(size)][label] = artifacts.digest(path)
    finally:
        torch.set_float32_matmul_precision(original["float32_matmul_precision"])
        torch.backends.cuda.matmul.allow_tf32 = original["allow_tf32"]
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = (
            original["allow_fp16_reduced_precision_reduction"])
    props = torch.cuda.get_device_properties(0)
    raw = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
           "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                               text=True).strip(),
           "sizes": list(SIZES), "case_names": list(CASE_NAMES), "tolerance": TOLERANCE,
           "baseline_parameters": BASELINE_PARAMETERS, "candidate_parameters": module.PARAMETERS,
           "precision": {"input_dtype": "float16", "output_dtype": "float16",
                         "operand_preparation": "float32", "product_accumulation": "float32",
                         "dot_input_precision": "tf32x3", "fp32_matmul_precision": "highest",
                         "allow_tf32": False, "allow_fp16_reduced_precision_reduction": False,
                         "original": original,
                         "restored_after_run": (
                             torch.get_float32_matmul_precision() == original["float32_matmul_precision"] and
                             bool(torch.backends.cuda.matmul.allow_tf32) == original["allow_tf32"] and
                             bool(torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction) ==
                             original["allow_fp16_reduced_precision_reduction"])},
           "cases": cases, "compiler": compiler, "profiles": profiles,
           "trace_sha256": trace_hashes, "first_launch_wall_ms": first_launch_ms,
           "source_sha256": artifacts.digest(output / "source.py"),
           "conventional_sha256": artifacts.digest(output / "conventional_source.py"),
           "script_sha256": artifacts.digest(output / "diagnostic_script.py"),
           "environment": {"gpu": props.name, "sm_count": props.multi_processor_count,
                           "compute_capability": f"{props.major}.{props.minor}",
                           "python": sys.version.split()[0], "torch": torch.__version__,
                           "triton": triton.__version__, "cuda_runtime": torch.version.cuda}}
    artifacts.write_json(output / "raw.json", raw)
    artifacts.write_json(output / "summary.json", summarize(raw))
    artifacts.manifest(output)
    return verify(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    try:
        result = verify(args.output) if args.verify else run(args.output)
    except Exception as error:
        print(f"MI-Research-4 feasibility failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
