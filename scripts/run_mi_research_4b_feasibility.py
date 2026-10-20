"""GPU correctness and one-launch gate for the matched fused rank-8/rank-7 study.

This command does not benchmark. Use the existing paired CLI only after PASS.
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
    from scripts import run_mi_research_4_feasibility as previous
except ModuleNotFoundError:
    import run_mi_research_2c_graph as artifacts
    import run_mi_research_3b_correctness as diagnostics
    import run_mi_research_4_feasibility as previous


ROOT = Path(__file__).resolve().parents[1]
RANK8 = ROOT / "examples/kernels/mi_research_4b_fused_rank8.py"
RANK7 = ROOT / "examples/kernels/mi_research_4_single_kernel_strassen.py"
RANK7_SHA256 = "0257dd59374f6abb4a008bd136db1dc5633c2fb993ea0706d044f12517a2398f"
SIZE = 128
CASES = previous.CASE_NAMES
TOLERANCE = previous.TOLERANCE
LABELS = ("A", "B")


def profile_summary(trace: dict, label: str) -> dict:
    events = trace.get("traceEvents")
    if not isinstance(events, list):
        raise ValueError("profiler trace has no events")
    kernels = [event["name"] for event in sorted(
        (event for event in events if event.get("cat") == "kernel"),
        key=lambda event: event["ts"])]
    driver = sum(event.get("cat") == "cuda_driver" and
                 event.get("name") in ("cuLaunchKernel", "cuLaunchKernelEx")
                 for event in events)
    runtime = sum(event.get("cat") == "cuda_runtime" and
                  event.get("name") in ("cudaLaunchKernel", "cudaLaunchKernelExC")
                  for event in events)
    expected = "_rank8_kernel" if label == "A" else "_strassen_kernel"
    return {"kernel_names": kernels, "kernel_events": len(kernels),
            "driver_launch_calls": driver, "runtime_launch_calls": runtime,
            "launch_calls": driver or runtime,
            "expected_kernel_seen": len(kernels) == 1 and expected in kernels[0]}


def summarize(raw: dict) -> dict:
    if (raw.get("schema_version") != 1 or raw.get("size") != SIZE or
            raw.get("case_names") != list(CASES) or raw.get("tolerance") != TOLERANCE or
            [row.get("case") for row in raw.get("cases", [])] != list(CASES) or
            raw.get("precision", {}).get("fp32_matmul_precision") != "highest" or
            raw.get("precision", {}).get("allow_tf32") is not False or
            raw.get("precision", {}).get("allow_fp16_reduced_precision_reduction") is not False or
            raw.get("rank7_sha256") != RANK7_SHA256):
        raise ValueError("invalid MI-Research-4B feasibility protocol")
    correctness = all(row["checks"][label]["passed"] is True and
                      row["checks"][label]["failing_elements"] == 0
                      for row in raw["cases"] for label in LABELS)
    topology = all(raw["profiles"][label]["kernel_events"] == 1 and
                   raw["profiles"][label]["launch_calls"] == 1 and
                   raw["profiles"][label]["expected_kernel_seen"] is True
                   for label in LABELS)
    compiler = all(raw["compiler"][label]["metadata"].get("n_spills") == 0 and
                   raw["compiler"][label]["metadata"].get("global_scratch_size") == 0 and
                   bool(raw["compiler"][label]["ptx_sha256"])
                   for label in LABELS)
    return {"schema_version": 1, "size": SIZE, "cases": len(raw["cases"]),
            "correctness_passed": correctness, "topology_passed": topology,
            "zero_reported_spills_and_scratch": bool(compiler),
            "performance_gate_passed": bool(correctness and topology and compiler)}


def verify(output: Path) -> dict:
    manifest = json.loads((output / "hash_manifest.json").read_text())
    actual = {str(path.relative_to(output)) for path in output.rglob("*")
              if path.is_file() and path.name != "hash_manifest.json"}
    if (not isinstance(manifest, dict) or actual != set(manifest) or
            any(Path(name).is_absolute() or ".." in Path(name).parts or
                artifacts.digest(output / name) != digest
                for name, digest in manifest.items())):
        raise ValueError("MI-Research-4B manifest mismatch")
    if (output / "failure.json").exists():
        failure = json.loads((output / "failure.json").read_text())
        return {"status": "FAILURE_RETAINED", "type": failure["type"]}
    raw = json.loads((output / "raw.json").read_text())
    if raw["rank7_sha256"] != artifacts.digest(output / "rank7_source.py"):
        raise ValueError("rank-7 source identity mismatch")
    for field, name in (("rank8_sha256", "rank8_source.py"),
                        ("script_sha256", "diagnostic_script.py")):
        if raw[field] != artifacts.digest(output / name):
            raise ValueError(f"{name} identity mismatch")
    for label in LABELS:
        trace = json.loads((output / f"{label}_trace.json").read_text())
        if (raw["profiles"][label] != profile_summary(trace, label) or
                raw["trace_sha256"][label] != artifacts.digest(output / f"{label}_trace.json")):
            raise ValueError("profiler topology does not replay")
        for kind, digest in raw["compiler"][label]["ir_sha256"].items():
            if digest != artifacts.digest(output / f"{label}.{kind}"):
                raise ValueError("compiler IR identity mismatch")
        if raw["compiler"][label]["ptx_sha256"] != artifacts.digest(output / f"{label}.ptx"):
            raise ValueError("PTX identity mismatch")
    replay = summarize(raw)
    if replay != json.loads((output / "summary.json").read_text()):
        raise ValueError("MI-Research-4B summary does not replay")
    return {"status": "PASS" if replay["performance_gate_passed"] else "GATE_FAILED", **replay}


def _load(path: Path, label: str):
    spec = importlib.util.spec_from_file_location(f"mi_research_4b_{label}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {label} source")
    module = importlib.util.module_from_spec(spec)
    previous_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous_bytecode
    return module


def run(output: Path) -> dict:
    if output.exists():
        raise ValueError("output already exists")
    if artifacts.digest(RANK7) != RANK7_SHA256:
        raise ValueError("committed rank-7 source identity changed")
    output.mkdir(parents=True)
    for source, name in ((RANK8, "rank8_source.py"), (RANK7, "rank7_source.py"),
                         (Path(__file__), "diagnostic_script.py")):
        shutil.copyfile(source, output / name)
    try:
        return _run_gpu(output)
    except Exception as error:
        artifacts.write_json(output / "failure.json", {
            "type": type(error).__name__, "message": str(error),
            "rank8_sha256": artifacts.digest(output / "rank8_source.py"),
            "rank7_sha256": artifacts.digest(output / "rank7_source.py")})
        artifacts.manifest(output)
        raise


def _run_gpu(output: Path) -> dict:
    import torch
    import triton

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    modules = {"A": _load(output / "rank8_source.py", "A"),
               "B": _load(output / "rank7_source.py", "B")}
    original = {"fp32_matmul_precision": torch.get_float32_matmul_precision(),
                "allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                "allow_fp16_reduced_precision_reduction": bool(
                    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)}
    cases, compiler, profiles, trace_hashes, first_launch_ms = [], {}, {}, {}, {}
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        a, b = previous._inputs(torch, SIZE, "seed-0")
        outputs = {label: torch.empty((SIZE, SIZE), device="cuda", dtype=torch.float16)
                   for label in LABELS}
        functions = {label: (lambda label=label: modules[label].matmul_into(
            a, b, outputs[label], modules[label].PARAMETERS)) for label in LABELS}
        for label in LABELS:
            torch.cuda.synchronize()
            start = time.perf_counter_ns()
            launch = functions[label]()
            torch.cuda.synchronize()
            first_launch_ms[label] = (time.perf_counter_ns() - start) / 1e6
            capture = experiment_pair._compiled_capture(launch)
            if capture is None or not isinstance(capture.get("ptx_text"), str):
                raise RuntimeError(f"{label} PTX unavailable")
            (output / f"{label}.ptx").write_text(capture["ptx_text"])
            ir_hashes = {}
            for kind in ("ttir", "ttgir", "llir"):
                source = launch.asm.get(kind)
                if isinstance(source, str):
                    (output / f"{label}.{kind}").write_text(source)
                    ir_hashes[kind] = artifacts.digest(output / f"{label}.{kind}")
            metadata = dict(capture["metadata"])
            metadata["global_scratch_size"] = getattr(launch.metadata, "global_scratch_size", None)
            compiler[label] = {"kernel_name": capture["kernel_name"], "metadata": metadata,
                               "ptx_sha256": artifacts.digest(output / f"{label}.ptx"),
                               "ir_sha256": ir_hashes,
                               "sass_status": "UNAVAILABLE" if not capture.get("sass_text") else
                                   "AVAILABLE_BUT_NOT_RETAINED"}
        for name in CASES:
            left, right = previous._inputs(torch, SIZE, name)
            a.copy_(left)
            b.copy_(right)
            reference = torch.matmul(a.float(), b.float()).half()
            cublas_no_reduced = torch.matmul(a, b)
            for target in outputs.values():
                target.fill_(float("nan"))
            for function in functions.values():
                function()
            torch.cuda.synchronize()
            checks = {label: diagnostics.summarize_error(torch, outputs[label], reference,
                                                          a=a, b=b) for label in LABELS}
            checks["cublas_no_reduced"] = diagnostics.summarize_error(
                torch, cublas_no_reduced, reference)
            cross = diagnostics.compare_candidates(torch, outputs["A"], outputs["B"], reference)
            cases.append({"case": name, "checks": checks, "rank8_vs_rank7": cross})
        a, b = previous._inputs(torch, SIZE, "seed-0")
        for label in LABELS:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as profiler:
                functions[label]()
                torch.cuda.synchronize()
            path = output / f"{label}_trace.json"
            profiler.export_chrome_trace(str(path))
            profiles[label] = profile_summary(json.loads(path.read_text()), label)
            trace_hashes[label] = artifacts.digest(path)
    finally:
        torch.set_float32_matmul_precision(original["fp32_matmul_precision"])
        torch.backends.cuda.matmul.allow_tf32 = original["allow_tf32"]
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = (
            original["allow_fp16_reduced_precision_reduction"])
    props = torch.cuda.get_device_properties(0)
    raw = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
           "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                               text=True).strip(),
           "size": SIZE, "case_names": list(CASES), "tolerance": TOLERANCE,
           "parameters": {label: modules[label].PARAMETERS for label in LABELS},
           "precision": {"input_dtype": "float16", "output_dtype": "float16",
                         "operand_preparation": "float32", "product_accumulation": "float32",
                         "dot_input_precision": "tf32x3", "fp32_matmul_precision": "highest",
                         "allow_tf32": False, "allow_fp16_reduced_precision_reduction": False,
                         "original": original,
                         "restored_after_run": (
                             torch.get_float32_matmul_precision() == original["fp32_matmul_precision"] and
                             bool(torch.backends.cuda.matmul.allow_tf32) == original["allow_tf32"] and
                             bool(torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction) ==
                             original["allow_fp16_reduced_precision_reduction"])},
           "cases": cases, "compiler": compiler, "profiles": profiles,
           "trace_sha256": trace_hashes, "first_launch_wall_ms": first_launch_ms,
           "rank8_sha256": artifacts.digest(output / "rank8_source.py"),
           "rank7_sha256": artifacts.digest(output / "rank7_source.py"),
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
        print(f"MI-Research-4B feasibility failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
