"""One bounded N=96 staged 3x3 GPU feasibility and paired diagnostic.

The ordinary 27-product and Laderman 23-product algorithms use the same
experimental Triton source and fixed-address scratch/output contract.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

from matmul_inspector import experiment, experiment_pair
try:
    from scripts import run_mi_research_2c_graph as artifacts
    from scripts import run_mi_research_3b_correctness as diagnostics
    from scripts import run_mi_research_4_feasibility as prior
    from scripts import run_mi_research_6_feasibility as exact
except ModuleNotFoundError:
    import run_mi_research_2c_graph as artifacts
    import run_mi_research_3b_correctness as diagnostics
    import run_mi_research_4_feasibility as prior
    import run_mi_research_6_feasibility as exact

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "examples/kernels/mi_research_6_staged_3x3.py"
SIZE = 96
CASES = prior.CASE_NAMES
TOLERANCE = {"atol": 0.01, "rtol": 0.01}
LABELS = ("A", "B")
RANKS = {"A": 27, "B": 23}
WARMUPS = 10
ITERATIONS = 20
ROUNDS = 8


def summarize(raw: dict) -> dict:
    if (raw.get("schema_version") != 1 or raw.get("size") != SIZE
            or raw.get("ranks") != RANKS or raw.get("case_names") != list(CASES)
            or raw.get("tolerance") != TOLERANCE
            or [x.get("case") for x in raw.get("cases", [])] != list(CASES)):
        raise ValueError("invalid 3x3 probe protocol")
    correct = all(row["checks"][label]["passed"] is True
                  and row["checks"][label]["failing_elements"] == 0
                  for row in raw["cases"] for label in LABELS)
    topology = all(raw["topology"][label]["product_events"] == RANKS[label]
                   and raw["topology"][label]["reconstruction_events"] == 1
                   and raw["topology"][label]["kernel_events"] == RANKS[label] + 1
                   and raw["topology"][label]["driver_launch_calls"] == RANKS[label] + 1
                   and len(raw["topology"][label]["kernel_names"]) == RANKS[label] + 1
                   and all("_block_product" in name for name in
                           raw["topology"][label]["kernel_names"][:-1])
                   and "_reconstruct" in raw["topology"][label]["kernel_names"][-1]
                   for label in LABELS)
    compiler = all(all(raw["compiler"][label][phase]["ptx_sha256"]
                       and raw["compiler"][label][phase]["metadata"].get("n_spills") in (0, None)
                       for phase in ("product", "reconstruction")) for label in LABELS)
    result = {"schema_version": 1, "correctness_passed": correct,
              "topology_passed": topology, "compiler_feasible": compiler,
              "gpu_execution_feasible": bool(correct and topology and compiler),
              "timing_decision": "NOT_MEASURED"}
    timing = raw.get("timing")
    if timing is None:
        return result
    if not result["gpu_execution_feasible"]:
        raise ValueError("timing was collected before feasibility gates")
    rounds = timing["rounds"]
    if (timing["warmups_per_kernel"] != WARMUPS or timing["iterations_per_block"] != ITERATIONS
            or len(rounds) != ROUNDS or len(timing["aa_rounds"]) != ROUNDS
            or timing["sequence"] != experiment_pair.expected_diagnostic_sequence(
                ROUNDS, ITERATIONS, "before")):
        raise ValueError("invalid paired timing protocol")
    event = {label: [] for label in LABELS}
    host = {label: [] for label in LABELS}
    raw_clock = {label: [] for label in LABELS}
    for index, row in enumerate(rounds, 1):
        if row["round"] != index or row["order"] != experiment_pair.pair_order(index):
            raise ValueError("invalid pair order")
        for label in LABELS:
            sample = row["measurements"][label]
            if abs(sample["cuda_event_elapsed_ms_per_block"] / ITERATIONS
                   - sample["cuda_event_elapsed_ms_per_operation"]) > 1e-8:
                raise ValueError("event normalization mismatch")
            event[label].append(sample["cuda_event_elapsed_ms_per_operation"] * 1000)
            host[label].append(sample["host_wall_ms_per_block"] / ITERATIONS * 1000)
            raw_clock[label].append(sample["monotonic_raw_ms_per_block"] / ITERATIONS * 1000)
    paired = [a - b for a, b in zip(event["A"], event["B"])]
    aa = []
    for row in timing["aa_rounds"]:
        aa.append((row["measurements"]["X"]["cuda_event_elapsed_ms_per_operation"]
                   - row["measurements"]["Y"]["cuda_event_elapsed_ms_per_operation"]) * 1000)
    result["timing"] = {
        "cuda_event_us": {label: {"median": statistics.median(event[label]),
                                    "min": min(event[label]), "max": max(event[label])}
                          for label in LABELS},
        "perf_counter_us": {label: {"median": statistics.median(host[label]),
                                      "min": min(host[label]), "max": max(host[label])}
                            for label in LABELS},
        "monotonic_raw_us": {label: {"median": statistics.median(raw_clock[label]),
                                       "min": min(raw_clock[label]), "max": max(raw_clock[label])}
                             for label in LABELS},
        "paired_A_minus_B_us": paired,
        "paired_median_us": statistics.median(paired),
        "paired_mad_us": statistics.median(abs(x - statistics.median(paired)) for x in paired),
        "aa_differences_us": aa, "aa_max_absolute_gap_us": max(abs(x) for x in aa),
        "b_faster_rounds": sum(x > 0 for x in paired),
        "a_faster_rounds": sum(x < 0 for x in paired),
        "host_clock_preflight": timing["host_clock_preflight"]["status"],
    }
    # One session is a diagnostic, not a reproducible performance ranking.
    result["timing_decision"] = "EXPLORATORY_SINGLE_SESSION_NO_SPEEDUP_CLAIM"
    return result


def trace_topology(trace: dict, rank: int) -> dict:
    events = sorted((x for x in trace["traceEvents"] if x.get("cat") == "kernel"),
                    key=lambda x: x.get("ts", 0))
    names = [x["name"] for x in events]
    products = sum("_block_product" in name for name in names)
    reductions = sum("_reconstruct" in name for name in names)
    driver = sum(x.get("cat") == "cuda_driver" and x.get("name") in
                 ("cuLaunchKernel", "cuLaunchKernelEx") for x in trace["traceEvents"])
    return {"kernel_names": names, "kernel_events": len(names),
            "product_events": products, "reconstruction_events": reductions,
            "driver_launch_calls": driver, "expected_launches": rank + 1}


def verify(output: Path) -> dict:
    hashes = json.loads((output / "hash_manifest.json").read_text())
    actual = {str(path.relative_to(output)) for path in output.rglob("*")
              if path.is_file() and path.name != "hash_manifest.json"}
    if (actual != set(hashes) or any(Path(name).is_absolute() or ".." in Path(name).parts
                                    or artifacts.digest(output / name) != value
                                    for name, value in hashes.items())):
        raise ValueError("GPU bundle manifest mismatch")
    if (output / "failure.json").exists():
        failure = json.loads((output / "failure.json").read_text())
        return {"status": "FAILURE_RETAINED", "type": failure["type"]}
    raw = json.loads((output / "raw.json").read_text())
    for field, path in (("source_sha256", "staged_source.py"),
                        ("script_sha256", "diagnostic_script.py"),
                        ("coefficients_sha256", "coefficients.json")):
        if raw[field] != artifacts.digest(output / path):
            raise ValueError(f"source identity mismatch: {path}")
    for label in LABELS:
        trace = json.loads((output / f"{label}_trace.json").read_text())
        if raw["topology"][label] != trace_topology(trace, RANKS[label]):
            raise ValueError("topology replay mismatch")
        for phase in ("product", "reconstruction"):
            record = raw["compiler"][label][phase]
            if record["ptx_sha256"] != artifacts.digest(output / f"{label}_{phase}.ptx"):
                raise ValueError("PTX identity mismatch")
            for kind, value in record["ir_sha256"].items():
                if value != artifacts.digest(output / f"{label}_{phase}.{kind}"):
                    raise ValueError("compiler IR identity mismatch")
    result = summarize(raw)
    if result != json.loads((output / "summary.json").read_text()):
        raise ValueError("summary does not replay")
    return {"status": "PASS" if result["gpu_execution_feasible"] else "GATE_FAILED", **result}


def load_snapshot(path: Path):
    spec = importlib.util.spec_from_file_location("mi_research_6_staged_snapshot", path)
    module = importlib.util.module_from_spec(spec)
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = old
    return module


def run(output: Path) -> dict:
    if output.exists():
        raise FileExistsError("GPU evidence directory exists")
    output.mkdir(parents=True)
    for original, snapshot in ((SOURCE, "staged_source.py"), (Path(__file__), "diagnostic_script.py"),
                               (exact.OUT / "coefficients.json", "coefficients.json")):
        shutil.copyfile(original, output / snapshot)
    try:
        result = run_gpu(output)
    except Exception as error:
        artifacts.write_json(output / "failure.json", {
            "type": type(error).__name__, "message": str(error),
            "stage": "gpu_probe", "source_sha256": artifacts.digest(output / "staged_source.py")})
        artifacts.manifest(output)
        raise
    return result


def run_gpu(output: Path) -> dict:
    import torch
    import triton

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    source = load_snapshot(output / "staged_source.py")
    actions = {"A": exact.conventional(),
               "B": exact.parse_laderman(exact.SOURCE.read_text())}
    recorded = json.loads((output / "coefficients.json").read_text())
    if recorded != {"conventional_27": exact.coefficients(actions["A"]),
                    "laderman_23_l_equals_1": exact.coefficients(actions["B"])}:
        raise ValueError("GPU coefficient snapshot differs from exact study")
    buffers = {label: source.buffers(actions[label]) for label in LABELS}
    left, right = prior._inputs(torch, SIZE, "seed-0")
    launches = {}
    functions = {label: (lambda label=label: source.call(left, right, *buffers[label],
                                                         launches=launches.setdefault(label, {})))
                 for label in LABELS}
    original = {"fp32_matmul_precision": torch.get_float32_matmul_precision(),
                "allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                "allow_fp16_reduced_precision_reduction": bool(
                    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)}
    cases, compiler, topology = [], {}, {}
    timing = None
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        for label in LABELS:
            functions[label]()
        torch.cuda.synchronize()
        for label in LABELS:
            compiler[label] = {}
            for phase in ("product", "reconstruction"):
                kernel = launches[label][phase]
                capture = experiment_pair._compiled_capture(kernel)
                if capture is None or not capture.get("ptx_text"):
                    raise ValueError(f"{label} {phase} compiler capture unavailable")
                (output / f"{label}_{phase}.ptx").write_text(capture["ptx_text"])
                ir = {}
                for kind in ("ttir", "ttgir", "llir"):
                    value = kernel.asm.get(kind)
                    if isinstance(value, str):
                        (output / f"{label}_{phase}.{kind}").write_text(value)
                        ir[kind] = artifacts.digest(output / f"{label}_{phase}.{kind}")
                compiler[label][phase] = {
                    "name": capture["kernel_name"], "ptx_sha256": artifacts.digest(output / f"{label}_{phase}.ptx"),
                    "ir_sha256": ir, "metadata": {**capture["metadata"],
                                                  "n_spills": getattr(kernel.metadata, "n_spills", None),
                                                  "global_scratch_size": getattr(kernel.metadata, "global_scratch_size", None)},
                    "sass_status": "UNAVAILABLE" if not capture.get("sass_text") else "AVAILABLE_NOT_RETAINED",
                }
        for case in CASES:
            a, b = prior._inputs(torch, SIZE, case)
            left.copy_(a)
            right.copy_(b)
            reference = torch.matmul(left.float(), right.float()).half()
            for label in LABELS:
                buffers[label][2].fill_(float("nan"))
                functions[label]()
            torch.cuda.synchronize()
            checks = {label: diagnostics.summarize_error(torch, buffers[label][2], reference,
                                                         a=left, b=right) for label in LABELS}
            cases.append({"case": case, "checks": checks,
                          "candidate_difference": diagnostics.compare_candidates(
                              torch, buffers["A"][2], buffers["B"][2], reference)})
        if not all(row["checks"][label]["passed"] for row in cases for label in LABELS):
            raise ValueError("numerical correctness gate failed")
        a, b = prior._inputs(torch, SIZE, "seed-0")
        left.copy_(a)
        right.copy_(b)
        for label in LABELS:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as profiler:
                functions[label]()
                torch.cuda.synchronize()
            path = output / f"{label}_trace.json"
            profiler.export_chrome_trace(str(path))
            topology[label] = trace_topology(json.loads(path.read_text()), RANKS[label])
            if (topology[label]["product_events"] != RANKS[label]
                    or topology[label]["reconstruction_events"] != 1
                    or topology[label]["kernel_events"] != RANKS[label] + 1):
                raise ValueError(f"{label} observed launch topology failed")
        if not all(compiler[label][phase]["metadata"]["n_spills"] in (0, None)
                   for label in LABELS for phase in ("product", "reconstruction")):
            raise ValueError("compiler reports spills")
        clock_reader = experiment_pair.raw_clock()
        preflight = experiment_pair.host_clock_preflight(reader=clock_reader)
        for label in LABELS:
            for _ in range(WARMUPS):
                functions[label]()
            torch.cuda.synchronize()
        measure = experiment.CudaTritonBackend._measure
        opts = {"raw_clock": clock_reader}
        controls = {label: measure(torch, functions[label], ITERATIONS, **opts) for label in LABELS}
        aa, rounds, sequence = experiment_pair.measure_paired_rounds(
            measure, torch, functions, ITERATIONS, ROUNDS, opts,
            with_aa=True, aa_placement="before")
        timing = {"mode": "eager_preallocated_complete_operation", "warmups_per_kernel": WARMUPS,
                  "iterations_per_block": ITERATIONS, "rounds": rounds, "aa_rounds": aa,
                  "sequence": sequence, "unranked_controls": controls,
                  "host_clock_preflight": preflight,
                  "boundaries": "perf start; RAW start; CUDA start event; N complete calls; CUDA end event; synchronize end event; RAW end; perf end"}
    finally:
        torch.set_float32_matmul_precision(original["fp32_matmul_precision"])
        torch.backends.cuda.matmul.allow_tf32 = original["allow_tf32"]
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = (
            original["allow_fp16_reduced_precision_reduction"])
    props = torch.cuda.get_device_properties(0)
    raw = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
           "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
           "size": SIZE, "ranks": RANKS, "case_names": list(CASES), "tolerance": TOLERANCE,
           "precision": {"input": "fp16", "output": "fp16", "preparation": "fp32",
                         "partial_products": "fp32 tf32x3", "accumulation": "fp32",
                         "reference": "torch.matmul(input.float(), input.float()).half()",
                         "active_float32_matmul_precision": "highest", "active_allow_tf32": False,
                         "active_allow_fp16_reduced_precision_reduction": False,
                         "original": original,
                         "restored": bool(torch.get_float32_matmul_precision() == original["fp32_matmul_precision"]
                                          and bool(torch.backends.cuda.matmul.allow_tf32) == original["allow_tf32"]
                                          and bool(torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)
                                          == original["allow_fp16_reduced_precision_reduction"])},
           "buffer_semantics": {label: {"fp32_product_scratch_bytes": RANKS[label] * 32 * 32 * 4,
                                         "coefficient_bytes": RANKS[label] * 27 * 4,
                                         "output_bytes": SIZE * SIZE * 2,
                                         "allocation_inside_measured_operation": False} for label in LABELS},
           "configured_grid": {"product": [2, 2], "reconstruction": [9, 2, 2]},
           "cases": cases, "compiler": compiler, "topology": topology, "timing": timing,
           "source_sha256": artifacts.digest(output / "staged_source.py"),
           "script_sha256": artifacts.digest(output / "diagnostic_script.py"),
           "coefficients_sha256": artifacts.digest(output / "coefficients.json"),
           "cpu_evidence_manifest_sha256": artifacts.digest(exact.OUT / "hash_manifest.json"),
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
    result = verify(args.output) if args.verify else run(args.output)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
