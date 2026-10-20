"""Bounded GPU correctness, compiler, and launch gate for MI-Research-5.

No timing is performed here. Use the existing paired CLI after PASS.
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
    from scripts import generate_mi_research_5_candidates as selection
    from scripts import run_mi_research_2c_graph as artifacts
    from scripts import run_mi_research_3b_correctness as diagnostics
    from scripts import run_mi_research_4_feasibility as inputs
except ModuleNotFoundError:
    import generate_mi_research_5_candidates as selection
    import run_mi_research_2c_graph as artifacts
    import run_mi_research_3b_correctness as diagnostics
    import run_mi_research_4_feasibility as inputs


ROOT = Path(__file__).resolve().parents[1]
LABELS = ("A", "B")
CASES = inputs.CASE_NAMES
TOLERANCE = inputs.TOLERANCE
SIZES = (128, 256)


def profile_summary(trace: dict, label: str) -> dict:
    events = trace.get("traceEvents", [])
    kernels = [event["name"] for event in events if event.get("cat") == "kernel"]
    driver = sum(event.get("cat") == "cuda_driver" and
                 event.get("name") in ("cuLaunchKernel", "cuLaunchKernelEx")
                 for event in events)
    runtime = sum(event.get("cat") == "cuda_runtime" and
                  event.get("name") in ("cudaLaunchKernel", "cudaLaunchKernelExC")
                  for event in events)
    expected = "_strassen_kernel" if label == "A" else "_variant_kernel"
    return {"kernel_names": kernels, "kernel_events": len(kernels),
            "driver_launch_calls": driver, "runtime_launch_calls": runtime,
            "launch_calls": driver or runtime,
            "expected_kernel_seen": len(kernels) == 1 and expected in kernels[0]}


def summarize(raw: dict) -> dict:
    if (raw.get("schema_version") != 1 or raw.get("size") not in SIZES or
            raw.get("label") not in selection.CHOICES or
            raw.get("cases_expected") != list(CASES) or
            raw.get("tolerance") != TOLERANCE or
            raw.get("grid") != [raw["size"] // 32, raw["size"] // 32] or
            raw.get("configured_ctas") != (raw["size"] // 32) ** 2 or
            raw.get("baseline_sha256") != selection.BASELINE_SHA256 or
            raw.get("candidate_sha256") != selection.verify_source_hash(raw["label"]) or
            [row["case"] for row in raw.get("cases", [])] != list(CASES) or
            raw.get("precision", {}).get("fp32_matmul_precision") != "highest" or
            raw.get("precision", {}).get("allow_tf32") is not False or
            raw.get("precision", {}).get("allow_fp16_reduced_precision_reduction") is not False):
        raise ValueError("invalid MI-Research-5 feasibility protocol")
    correctness = all(row["checks"][label]["passed"] is True and
                      row["checks"][label]["failing_elements"] == 0 and
                      row["checks"][label]["candidate_nonfinite_count"] == 0
                      for row in raw["cases"] for label in LABELS)
    topology = all(raw["profiles"][label]["kernel_events"] == 1 and
                   raw["profiles"][label]["launch_calls"] == 1 and
                   raw["profiles"][label]["expected_kernel_seen"] is True
                   for label in LABELS)
    compiler = all(raw["compiler"][label]["metadata"].get("n_spills") == 0 and
                   raw["compiler"][label]["metadata"].get("global_scratch_size") == 0 and
                   bool(raw["compiler"][label]["ptx_sha256"])
                   for label in LABELS)
    return {"schema_version": 1, "label": raw["label"], "size": raw["size"],
            "cases": len(raw["cases"]), "correctness_passed": correctness,
            "topology_passed": topology, "zero_reported_spills_and_scratch": compiler,
            "performance_gate_passed": bool(correctness and topology and compiler)}


def verify(output: Path) -> dict:
    selection.verify()
    manifest = json.loads((output / "hash_manifest.json").read_text())
    actual = {str(path.relative_to(output)) for path in output.rglob("*")
              if path.is_file() and path.name != "hash_manifest.json"}
    if (actual != set(manifest) or
            any(Path(name).is_absolute() or ".." in Path(name).parts or
                artifacts.digest(output / name) != digest
                for name, digest in manifest.items())):
        raise ValueError("MI-Research-5 manifest mismatch")
    if (output / "failure.json").exists():
        failure = json.loads((output / "failure.json").read_text())
        return {"status": "FAILURE_RETAINED", "type": failure["type"]}
    raw = json.loads((output / "raw.json").read_text())
    for field, filename in (("baseline_sha256", "baseline_source.py"),
                            ("candidate_sha256", "candidate_source.py"),
                            ("script_sha256", "diagnostic_script.py")):
        if raw[field] != artifacts.digest(output / filename):
            raise ValueError(f"{filename} source identity mismatch")
    for label in LABELS:
        trace = json.loads((output / f"{label}_trace.json").read_text())
        if (raw["profiles"][label] != profile_summary(trace, label) or
                raw["trace_sha256"][label] != artifacts.digest(output / f"{label}_trace.json")):
            raise ValueError("profiler topology replay mismatch")
        for kind, expected in raw["compiler"][label]["ir_sha256"].items():
            if expected != artifacts.digest(output / f"{label}.{kind}"):
                raise ValueError("compiler IR hash mismatch")
        if raw["compiler"][label]["ptx_sha256"] != artifacts.digest(output / f"{label}.ptx"):
            raise ValueError("PTX hash mismatch")
    replay = summarize(raw)
    if replay != json.loads((output / "summary.json").read_text()):
        raise ValueError("summary replay mismatch")
    return {"status": "PASS" if replay["performance_gate_passed"] else "GATE_FAILED", **replay}


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(f"mi_research_5_{name}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load frozen source")
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def run(output: Path, label: str, size: int) -> dict:
    if label not in selection.CHOICES or size not in SIZES:
        raise ValueError("unsupported frozen label or size")
    if output.exists():
        raise FileExistsError("output already exists")
    selection.verify()
    candidate = selection.SOURCES / f"mi_research_5_{label}.py"
    output.mkdir(parents=True)
    for source, name in ((selection.BASELINE, "baseline_source.py"),
                         (candidate, "candidate_source.py"),
                         (Path(__file__), "diagnostic_script.py")):
        shutil.copyfile(source, output / name)
    try:
        return _run_gpu(output, label, size)
    except Exception as error:
        artifacts.write_json(output / "failure.json", {
            "type": type(error).__name__, "message": str(error),
            "label": label, "size": size})
        artifacts.manifest(output)
        raise


def _run_gpu(output: Path, label: str, size: int) -> dict:
    import torch
    import triton

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    modules = {"A": _load(output / "baseline_source.py", "A"),
               "B": _load(output / "candidate_source.py", "B")}
    original = {"fp32_matmul_precision": torch.get_float32_matmul_precision(),
                "allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                "allow_fp16_reduced_precision_reduction": bool(
                    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)}
    cases, compiler, profiles, trace_hashes, first_launch_ms = [], {}, {}, {}, {}
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        a, b = inputs._inputs(torch, size, "seed-0")
        outputs = {key: torch.empty((size, size), device="cuda", dtype=torch.float16)
                   for key in LABELS}
        functions = {key: (lambda key=key: modules[key].matmul_into(
            a, b, outputs[key], modules[key].PARAMETERS)) for key in LABELS}
        for key in LABELS:
            torch.cuda.synchronize()
            start = time.perf_counter_ns()
            launch = functions[key]()
            torch.cuda.synchronize()
            first_launch_ms[key] = (time.perf_counter_ns() - start) / 1e6
            capture = experiment_pair._compiled_capture(launch)
            if capture is None or not isinstance(capture.get("ptx_text"), str):
                raise RuntimeError(f"{key} PTX unavailable")
            (output / f"{key}.ptx").write_text(capture["ptx_text"])
            ir_hashes = {}
            for kind in ("ttir", "ttgir", "llir"):
                ir = launch.asm.get(kind)
                if isinstance(ir, str):
                    (output / f"{key}.{kind}").write_text(ir)
                    ir_hashes[kind] = artifacts.digest(output / f"{key}.{kind}")
            metadata = dict(capture["metadata"])
            metadata["global_scratch_size"] = getattr(launch.metadata, "global_scratch_size", None)
            compiler[key] = {"kernel_name": capture["kernel_name"], "metadata": metadata,
                             "ptx_sha256": artifacts.digest(output / f"{key}.ptx"),
                             "ir_sha256": ir_hashes,
                             "sass_status": "UNAVAILABLE" if not capture.get("sass_text")
                                            else "AVAILABLE_BUT_NOT_RETAINED"}
        for case in CASES:
            left, right = inputs._inputs(torch, size, case)
            a.copy_(left)
            b.copy_(right)
            reference = torch.matmul(a.float(), b.float()).half()
            library = torch.matmul(a, b)
            for target in outputs.values():
                target.fill_(float("nan"))
            for function in functions.values():
                function()
            torch.cuda.synchronize()
            checks = {key: diagnostics.summarize_error(torch, outputs[key], reference,
                                                        a=a, b=b) for key in LABELS}
            checks["cublas_no_reduced"] = diagnostics.summarize_error(torch, library, reference)
            cross = diagnostics.compare_candidates(torch, outputs["A"], outputs["B"], reference)
            cases.append({"case": case, "checks": checks, "A_vs_B": cross})
        a, b = inputs._inputs(torch, size, "seed-0")
        for key in LABELS:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as profiler:
                functions[key]()
                torch.cuda.synchronize()
            path = output / f"{key}_trace.json"
            profiler.export_chrome_trace(str(path))
            profiles[key] = profile_summary(json.loads(path.read_text()), key)
            trace_hashes[key] = artifacts.digest(path)
    finally:
        torch.set_float32_matmul_precision(original["fp32_matmul_precision"])
        torch.backends.cuda.matmul.allow_tf32 = original["allow_tf32"]
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = (
            original["allow_fp16_reduced_precision_reduction"])
    props = torch.cuda.get_device_properties(0)
    raw = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
           "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                               text=True).strip(),
           "label": label, "size": size, "grid": [size // 32, size // 32],
           "configured_ctas": (size // 32) ** 2,
           "cases_expected": list(CASES), "tolerance": TOLERANCE,
           "parameters": {key: modules[key].PARAMETERS for key in LABELS},
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
           "baseline_sha256": artifacts.digest(output / "baseline_source.py"),
           "candidate_sha256": artifacts.digest(output / "candidate_source.py"),
           "script_sha256": artifacts.digest(output / "diagnostic_script.py"),
           "selection_sha256": artifacts.digest(selection.OUT / "selection.json"),
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
    parser.add_argument("--label", choices=tuple(selection.CHOICES))
    parser.add_argument("--size", type=int, choices=SIZES)
    args = parser.parse_args()
    if not args.verify and (args.label is None or args.size is None):
        parser.error("--label and --size are required for a new GPU gate")
    try:
        result = verify(args.output) if args.verify else run(args.output, args.label, args.size)
    except Exception as error:
        print(f"MI-Research-5 feasibility failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
