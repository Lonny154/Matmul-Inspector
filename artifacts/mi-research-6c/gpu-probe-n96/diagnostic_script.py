"""One bounded N=96 compilation and correctness probe for the 55-gate circuit.

This script does not benchmark. It records the complete single-kernel output,
compiler artifacts, and profiler topology before any timing decision.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
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
    from scripts import run_mi_research_4_feasibility as prior
    from scripts import run_mi_research_6_gpu_probe as historical
    from scripts import run_mi_research_6c as exact
except ModuleNotFoundError:
    import run_mi_research_2c_graph as artifacts
    import run_mi_research_3b_correctness as diagnostics
    import run_mi_research_4_feasibility as prior
    import run_mi_research_6_gpu_probe as historical
    import run_mi_research_6c as exact


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/mi-research-6c/gpu-probe-n96"
CASES = prior.CASE_NAMES
TOLERANCE = {"atol": 0.01, "rtol": 0.01}


def trace_topology(trace: dict) -> dict:
    events = trace.get("traceEvents", [])
    kernels = [event for event in events if event.get("cat") == "kernel"]
    driver = [event for event in events if event.get("cat") == "cuda_driver" and
              event.get("name") in ("cuLaunchKernel", "cuLaunchKernelEx")]
    if len(kernels) != 1 or "_fused" not in kernels[0]["name"] or len(driver) != 1:
        raise ValueError(f"expected one complete fused physical kernel: {len(kernels)} kernels, {len(driver)} launches")
    return {"physical_kernel_events": 1, "driver_launch_calls": 1,
            "kernel_name": kernels[0]["name"], "configured_ctas": 4}


def summarize(raw: dict) -> dict:
    if (raw.get("schema_version") != 1 or raw.get("size") != 96 or
            raw.get("mode") != "eager_single_kernel" or
            raw.get("cases") is None or
            [row.get("case") for row in raw["cases"]] != list(CASES) or
            raw.get("tolerance") != TOLERANCE):
        raise ValueError("GPU probe protocol changed")
    passed = all(len(row["checks"]) == 2 and all(
        check["passed"] is True and check["failing_elements"] == 0 and
        check["candidate_nonfinite_count"] == 0 for check in row["checks"])
        for row in raw["cases"])
    topology = raw["topology"]["physical_kernel_events"] == 1 and raw["topology"]["driver_launch_calls"] == 1
    compiler = bool(raw["compiler"]["ptx_sha256"] and raw["compiler"]["metadata"]["shared"] is not None)
    return {"schema_version": 1, "numerical_correctness_passed": passed,
            "topology_passed": topology, "compiler_artifacts_present": compiler,
            "gpu_execution_feasible": bool(passed and topology and compiler),
            "performance_status": "NOT_MEASURED"}


def verify(output: Path = OUT) -> dict:
    manifest = json.loads((output / "hash_manifest.json").read_text())
    files = {str(path.relative_to(output)) for path in output.rglob("*")
             if path.is_file() and path.name != "hash_manifest.json"}
    if files != set(manifest) or any(Path(name).is_absolute() or ".." in Path(name).parts or
                                     artifacts.digest(output / name) != value
                                     for name, value in manifest.items()):
        raise ValueError("GPU probe manifest mismatch")
    if (output / "failure.json").exists():
        failure = json.loads((output / "failure.json").read_text())
        return {"status": "FAILURE_RETAINED", "type": failure["type"],
                "stage": failure["stage"]}
    raw = json.loads((output / "raw.json").read_text())
    if (raw["source_sha256"] != artifacts.digest(output / "fused_source.py") or
            raw["source_sha256"] != json.loads((exact.OUT / "analysis.json").read_text())["probe_source_sha256"] or
            raw["pre_probe_gate_sha256"] != artifacts.digest(output / "pre_probe_gate.json") or
            raw["script_sha256"] != artifacts.digest(output / "diagnostic_script.py")):
        raise ValueError("probe source provenance mismatch")
    if raw["topology"] != trace_topology(json.loads((output / "trace.json").read_text())):
        raise ValueError("physical launch topology mismatch")
    for kind, sha in raw["compiler"]["ir_sha256"].items():
        if sha != artifacts.digest(output / f"fused.{kind}"):
            raise ValueError("compiler IR hash mismatch")
    if raw["compiler"]["ptx_sha256"] != artifacts.digest(output / "fused.ptx"):
        raise ValueError("PTX hash mismatch")
    result = summarize(raw)
    if result != json.loads((output / "summary.json").read_text()):
        raise ValueError("GPU summary did not replay")
    return {"status": "PASS" if result["gpu_execution_feasible"] else "GATE_FAILED", **result}


def run(output: Path = OUT) -> dict:
    if output.exists():
        raise FileExistsError("GPU probe evidence directory already exists")
    exact.verify()
    gate = json.loads((exact.OUT / "pre_probe_gate.json").read_text())
    if (gate["decision"] != "PASS_FOR_ONE_COMPILATION_AND_CORRECTNESS_PROBE" or
            gate["analysis_sha256"] != artifacts.digest(exact.OUT / "analysis.json") or
            gate["probe_source_sha256"] != artifacts.digest(exact.PROBE_SOURCE)):
        raise ValueError("pre-GPU feasibility decision is missing or stale")
    output.mkdir(parents=True)
    for source, name in ((exact.PROBE_SOURCE, "fused_source.py"),
                         (exact.OUT / "pre_probe_gate.json", "pre_probe_gate.json"),
                         (Path(__file__), "diagnostic_script.py")):
        shutil.copyfile(source, output / name)
    try:
        result = _run_gpu(output)
    except Exception as error:
        artifacts.write_json(output / "failure.json", {
            "type": type(error).__name__, "message": str(error),
            "stage": "compilation_or_correctness", "source_sha256": artifacts.digest(output / "fused_source.py")})
        artifacts.manifest(output)
        raise
    return result


def _run_gpu(output: Path) -> dict:
    import torch
    import triton

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    source = historical.load_snapshot(output / "fused_source.py")
    left, right = prior._inputs(torch, 96, "seed-0")
    result = torch.empty((96, 96), device="cuda", dtype=torch.float16)
    original = {"fp32_matmul_precision": torch.get_float32_matmul_precision(),
                "allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                "allow_fp16_reduced_precision_reduction": bool(
                    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)}
    cases = []
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.cuda.synchronize()
        started = time.perf_counter_ns()
        compiled = source.call(left, right, result)
        torch.cuda.synchronize()
        compile_ms = (time.perf_counter_ns() - started) / 1e6
        details = experiment_pair._compiled_capture(compiled)
        if details is None or not isinstance(details.get("ptx_text"), str):
            raise ValueError("compiler PTX unavailable")
        (output / "fused.ptx").write_text(details["ptx_text"])
        ir_hashes = {}
        for kind in ("ttir", "ttgir", "llir"):
            value = compiled.asm.get(kind)
            if isinstance(value, str):
                (output / f"fused.{kind}").write_text(value)
                ir_hashes[kind] = artifacts.digest(output / f"fused.{kind}")
        compiler = {"kernel_name": details["kernel_name"],
                    "ptx_sha256": artifacts.digest(output / "fused.ptx"),
                    "ir_sha256": ir_hashes,
                    "metadata": {**details["metadata"],
                                 "n_spills": getattr(compiled.metadata, "n_spills", None),
                                 "global_scratch_size": getattr(compiled.metadata, "global_scratch_size", None)},
                    "sass_status": "UNAVAILABLE" if not details.get("sass_text") else "AVAILABLE_NOT_RETAINED"}
        for case in CASES:
            a, b = prior._inputs(torch, 96, case)
            left.copy_(a)
            right.copy_(b)
            reference = torch.matmul(left.float(), right.float()).half()
            checks = []
            for _ in range(2):
                result.fill_(float("nan"))
                source.call(left, right, result)
                torch.cuda.synchronize()
                check = diagnostics.summarize_error(torch, result, reference, a=left, b=right)
                checks.append(check)
                if not check["passed"] or check["failing_elements"]:
                    raise ValueError(f"numerical gate failed: {case}")
            cases.append({"case": case, "checks": checks,
                          "output_poisoned_before_each_execution": True})
            artifacts.write_json(output / "correctness_progress.json", cases)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as profiler:
            source.call(left, right, result)
            torch.cuda.synchronize()
        profiler.export_chrome_trace(str(output / "trace.json"))
        topology = trace_topology(json.loads((output / "trace.json").read_text()))
    finally:
        torch.set_float32_matmul_precision(original["fp32_matmul_precision"])
        torch.backends.cuda.matmul.allow_tf32 = original["allow_tf32"]
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = (
            original["allow_fp16_reduced_precision_reduction"])
    props = torch.cuda.get_device_properties(0)
    raw = {"schema_version": 1, "mode": "eager_single_kernel", "size": 96,
           "created_at_utc": datetime.now(timezone.utc).isoformat(),
           "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                               cwd=ROOT, text=True).strip(),
           "source_sha256": artifacts.digest(output / "fused_source.py"),
           "script_sha256": artifacts.digest(output / "diagnostic_script.py"),
           "pre_probe_gate_sha256": artifacts.digest(output / "pre_probe_gate.json"),
           "environment": {"gpu": props.name, "sm_count": props.multi_processor_count,
                           "compute_capability": f"{props.major}.{props.minor}",
                           "python": sys.version.split()[0], "torch": torch.__version__,
                           "triton": triton.__version__, "cuda_runtime": torch.version.cuda},
           "precision": {"input": "fp16", "output": "fp16", "intermediate": "fp32",
                         "dot": "tf32x3", "reference": "torch.matmul(left.float(), right.float()).half()",
                         "active_allow_tf32": False,
                         "active_allow_fp16_reduced_precision_reduction": False,
                         "original": original,
                         "restored": bool(torch.get_float32_matmul_precision() == original["fp32_matmul_precision"]
                                          and bool(torch.backends.cuda.matmul.allow_tf32) == original["allow_tf32"]
                                          and bool(torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)
                                          == original["allow_fp16_reduced_precision_reduction"])},
           "tolerance": TOLERANCE, "compile_and_first_call_ms": compile_ms,
           "explicit_product_scratch_bytes": 0, "configured_grid": [2, 2],
           "cases": cases, "compiler": compiler, "topology": topology}
    artifacts.write_json(output / "raw.json", raw)
    artifacts.write_json(output / "summary.json", summarize(raw))
    artifacts.manifest(output)
    return verify(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    print(json.dumps(verify(args.output) if args.verify else run(args.output), sort_keys=True))


if __name__ == "__main__":
    main()
