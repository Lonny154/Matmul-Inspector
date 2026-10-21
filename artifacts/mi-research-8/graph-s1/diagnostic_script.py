"""Matched N=96, 36-CTA Perminov-55 versus Perminov-58 graph study."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

from matmul_inspector import experiment_pair
try:
    from scripts import run_mi_research_2c_graph as evidence
    from scripts import run_mi_research_3b_correctness as diagnostics
    from scripts import run_mi_research_4_feasibility as cases
    from scripts import run_mi_research_6_gpu_probe as loader
    from scripts import run_mi_research_6c as exact
    from scripts import run_mi_research_6d_graph as graph_prior
    from scripts import run_mi_research_7 as compiler_prior
    from scripts import analyze_mi_research_7b as prior_7b
    from scripts import analyze_mi_research_7c as prior_7c
    from scripts import generate_mi_research_8_kernel as generator
except ModuleNotFoundError:
    import run_mi_research_2c_graph as evidence
    import run_mi_research_3b_correctness as diagnostics
    import run_mi_research_4_feasibility as cases
    import run_mi_research_6_gpu_probe as loader
    import run_mi_research_6c as exact
    import run_mi_research_6d_graph as graph_prior
    import run_mi_research_7 as compiler_prior
    import analyze_mi_research_7b as prior_7b
    import analyze_mi_research_7c as prior_7c
    import generate_mi_research_8_kernel as generator

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/mi-research-8"
PROTOCOL = BASE / "protocol.json"
PLAN = BASE / "mapping_plan.json"
SIZE, OPS, ROUNDS, WARMUPS = 96, 20, 8, 10
TOLERANCE = {"atol": 0.01, "rtol": 0.01}
SOURCES = {"A": generator.BASELINE, "B": generator.TARGET}


def check_protocol() -> dict:
    p = json.loads(PROTOCOL.read_text())
    if (p["base_commit"] != "e08ca67bca9d8af8ef1d10868f69dfb111eebbdf" or
        not p["recorded_before_GPU_compilation"] or
        p["mapping_plan_sha256"] != evidence.digest(PLAN) or
        p["prior_6c_manifest_sha256"] != evidence.digest(exact.OUT / "hash_manifest.json") or
        p["prior_7b_analysis_manifest_sha256"] != evidence.digest(prior_7b.MANIFEST) or
        p["prior_7c_manifest_sha256"] != evidence.digest(prior_7c.BASE / "hash_manifest.json") or
        p["source_sha256"] != {"A_55": evidence.digest(SOURCES["A"]),
                               "B_58": evidence.digest(SOURCES["B"])} or
        p["correctness_cases"] != list(cases.CASE_NAMES) or
        p["tolerance"] != TOLERANCE or p["graph_operations_per_replay"] != OPS or
        p["paired_rounds_per_session"] != ROUNDS or p["warmups"] != WARMUPS or
        generator.generate()[0] != SOURCES["B"].read_text()):
        raise ValueError("frozen MI-Research-8 protocol or source changed")
    if exact.verify()["status"] != "PASS" or prior_7b.verify()["status"] != "PASS" or prior_7c.verify()["status"] != "PASS":
        raise ValueError("historical evidence changed")
    return p


def trace_topology(trace: dict) -> dict:
    events = trace["traceEvents"]
    kernels = [event for event in events if event.get("cat") == "kernel"]
    launches = sum(event.get("cat") == "cuda_runtime" and
                   event.get("name") == "cudaGraphLaunch" for event in events)
    if len(kernels) != OPS or launches != 1 or any(
        "_fused" not in k["name"] or k.get("args", {}).get("grid") != [2, 2, 9] or
        k.get("args", {}).get("block") != [128, 1, 1] for k in kernels):
        raise ValueError("matched graph launch topology changed")
    return {"physical_kernel_events": len(kernels), "kernels_per_operation": 1,
            "cuda_graph_launch_calls": launches, "grid": [2, 2, 9],
            "block": [128, 1, 1], "first_kernel_name": kernels[0]["name"]}


def capture_trace(torch, fn, path: Path) -> dict:
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as profiler:
        fn()
        torch.cuda.synchronize()
    profiler.export_chrome_trace(str(path))
    return trace_topology(json.loads(path.read_text()))


def summarize(raw: dict) -> dict:
    if (raw["protocol_sha256"] != evidence.digest(PROTOCOL) or
        raw["size"] != SIZE or raw["tolerance"] != TOLERANCE or
        [x["case"] for x in raw["correctness"]] != list(cases.CASE_NAMES) or
        not raw["correctness_passed"] or not raw["fixed_addresses_unchanged"]):
        raise ValueError("invalid MI-Research-8 raw record")
    pair = raw["pair"]
    if (pair["physical_labels"] != ["A", "B"] or len(pair["rounds"]) != ROUNDS or
        len(pair["aa_rounds"]) != ROUNDS or
        pair["measurement_sequence"] != experiment_pair.expected_diagnostic_sequence(ROUNDS, OPS, "before") or
        set(pair["unranked_timing_controls"]) != {"A", "B"} or
        pair["aa_kernel"] != "A" or pair["iterations_per_measurement"] != OPS or
        pair["aa_iterations_per_measurement"] != OPS):
        raise ValueError("paired graph protocol changed")
    all_samples = list(pair["unranked_timing_controls"].values())
    for i in range(1, ROUNDS + 1):
        row, aa = pair["rounds"][i-1], pair["aa_rounds"][i-1]
        if (row["round"] != i or row["order"] != experiment_pair.pair_order(i) or
            aa["round"] != i or aa["order"] != (["X", "Y"] if i % 2 else ["Y", "X"])):
            raise ValueError("paired graph order changed")
        all_samples.extend(row["measurements"].values())
        all_samples.extend(aa["measurements"].values())
    if any(s["complete_operations_per_block"] != OPS or
           s["host_submission_calls_per_block"] != 1 or
           not math.isclose(s["cuda_event_elapsed_ms_per_operation"] * OPS,
                            s["cuda_event_elapsed_ms_per_block"], abs_tol=1e-8)
           for s in all_samples):
        raise ValueError("complete-operation timing changed")
    first = raw["correctness"][0]["checks"]
    benchmark = {**pair, "host_clock_preflight": raw["host_clock_preflight"]}
    decision = experiment_pair.summarize_clock_aware(
        {"passed": True, "checks": {"A": first["A"]["cuda_graph"][0],
                                   "B": first["B"]["cuda_graph"][0]}},
        experiment_pair.timing_rows({"benchmark": benchmark}), None, benchmark)
    if decision["decision"] in ("FAILED", "REJECTED_INCORRECT"):
        raise ValueError("paired decision failed")
    samples = {label: [row["measurements"][label]["cuda_event_elapsed_ms_per_operation"] * 1000
                       for row in pair["rounds"]] for label in ("A", "B")}
    gaps = [a - b for a, b in zip(samples["A"], samples["B"])]
    return {"decision": decision, "event_us_per_operation": samples,
            "event_median_us": {label: statistics.median(values) for label, values in samples.items()},
            "paired_A55_minus_B58_us": gaps, "paired_median_us": statistics.median(gaps),
            "paired_mad_us": statistics.median(abs(x-statistics.median(gaps)) for x in gaps),
            "aa_max_absolute_gap_us": decision["aa_control"]["max_absolute_difference_ms"] * 1000}


def verify(output: Path) -> dict:
    check_protocol()
    manifest = json.loads((output / "hash_manifest.json").read_text())
    files = {str(path.relative_to(output)) for path in output.rglob("*")
             if path.is_file() and path.name != "hash_manifest.json"}
    if files != set(manifest) or any(evidence.digest(output / name) != sha for name, sha in manifest.items()):
        raise ValueError("MI-Research-8 manifest mismatch")
    if (output / "failure.json").exists():
        return {"status": "FAILURE_RETAINED", **json.loads((output / "failure.json").read_text())}
    raw = json.loads((output / "raw.json").read_text())
    p = json.loads((output / "protocol.json").read_text())
    if (raw["protocol_sha256"] != evidence.digest(output / "protocol.json") or
        p != json.loads(PROTOCOL.read_text()) or
        raw["source_sha256"] != {label: evidence.digest(output / f"{label}_source.py")
                                 for label in ("A", "B")} or
        raw["source_sha256"] != {label: evidence.digest(SOURCES[label]) for label in ("A", "B")} or
        raw["script_sha256"] != evidence.digest(output / "diagnostic_script.py") or
        raw["correctness"] != json.loads((output / "correctness_progress.json").read_text()) or
        raw["pair"] != json.loads((output / "pair_progress.json").read_text())):
        raise ValueError("MI-Research-8 provenance mismatch")
    for label in ("A", "B"):
        if raw["topology"][label] != trace_topology(json.loads((output / f"{label}_trace.json").read_text())):
            raise ValueError("graph topology changed")
        c = raw["compiler"][label]
        if c["ptx_sha256"] != evidence.digest(output / f"{label}.ptx") or any(
            sha != evidence.digest(output / f"{label}.{kind}") for kind, sha in c["ir_sha256"].items()):
            raise ValueError("compiler artifact changed")
    if summarize(raw) != json.loads((output / "summary.json").read_text()):
        raise ValueError("MI-Research-8 summary mismatch")
    return {"status": "PASS", "session": raw["session"],
            "decision": summarize(raw)["decision"]["decision"]}


def run(output: Path, session: int) -> dict:
    if output.exists() or session not in (1, 2):
        raise ValueError("invalid or existing output")
    protocol = check_protocol()
    output.mkdir(parents=True)
    for source, name in ((PROTOCOL, "protocol.json"), (PLAN, "mapping_plan.json"),
                         (SOURCES["A"], "A_source.py"), (SOURCES["B"], "B_source.py"),
                         (Path(__file__), "diagnostic_script.py")):
        shutil.copyfile(source, output / name)
    try:
        return _run_gpu(output, session, protocol)
    except Exception as error:
        evidence.write_json(output / "failure.json", {"type": type(error).__name__,
                                                       "message": str(error), "session": session})
        evidence.manifest(output)
        raise


def _run_gpu(output: Path, session: int, protocol: dict) -> dict:
    import torch
    import triton
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    modules = {label: loader.load_snapshot(output / f"{label}_source.py") for label in ("A", "B")}
    a0, b0 = cases._inputs(torch, SIZE, "seed-0")
    left, right = torch.empty_like(a0), torch.empty_like(b0)
    left.copy_(a0); right.copy_(b0)
    outputs = {label: torch.empty((SIZE, SIZE), device="cuda", dtype=torch.float16)
               for label in ("A", "B")}
    addresses = (left.data_ptr(), right.data_ptr(), outputs["A"].data_ptr(), outputs["B"].data_ptr())
    functions = {label: (lambda label=label: modules[label].call(left, right, outputs[label]))
                 for label in ("A", "B")}
    order = ["A", "B"] if session == 1 else ["B", "A"]
    original = {"precision": torch.get_float32_matmul_precision(),
                "tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                "reduced": bool(torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)}
    compiler, first_call, capture_setup, graphs, checks, topology = {}, {}, {}, {}, [], {}
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        for label in order:
            torch.cuda.synchronize(); start = time.perf_counter_ns()
            compiled = functions[label]()
            torch.cuda.synchronize()
            first_call[label] = (time.perf_counter_ns()-start)/1e6
            compiler[label] = compiler_prior._capture_compiler(compiled, output, label)
        for label in order:
            for _ in range(WARMUPS): functions[label]()
            torch.cuda.synchronize()
        for label in order:
            start = time.perf_counter_ns()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(OPS): functions[label]()
            torch.cuda.synchronize()
            graphs[label] = graph
            capture_setup[label] = (time.perf_counter_ns()-start)/1e6
        for case in cases.CASE_NAMES:
            aa, bb = cases._inputs(torch, SIZE, case)
            left.copy_(aa); right.copy_(bb)
            reference = torch.matmul(left.float(), right.float()).half()
            row = {}
            for label in ("A", "B"):
                eager, graph_checks = [], []
                for fn, destination in ((functions[label], eager), (graphs[label].replay, graph_checks)):
                    for _ in range(2):
                        outputs[label].fill_(float("nan")); fn(); torch.cuda.synchronize()
                        destination.append(diagnostics.summarize_error(
                            torch, outputs[label], reference, a=left, b=right))
                graph_result = outputs[label].clone()
                functions[label](); torch.cuda.synchronize()
                row[label] = {"eager": eager, "cuda_graph": graph_checks,
                              "graph_eager_bitwise_equal": bool(torch.equal(graph_result, outputs[label]))}
            checks.append({"case": case, "checks": row, "output_poisoned_before_each_check": True})
            evidence.write_json(output / "correctness_progress.json", checks)
            if any(not x["passed"] or x["failing_elements"] or x["candidate_nonfinite_count"]
                   for label in ("A", "B") for phase in ("eager", "cuda_graph") for x in row[label][phase]) or any(
                       not row[label]["graph_eager_bitwise_equal"] for label in ("A", "B")):
                raise ValueError(f"numerical gate failed: {case}")
        left.copy_(a0); right.copy_(b0); torch.cuda.synchronize()
        for label in order:
            topology[label] = capture_trace(torch, graphs[label].replay, output / f"{label}_trace.json")
        for label in order:
            for _ in range(WARMUPS): graphs[label].replay()
            torch.cuda.synchronize()
        reader = experiment_pair.raw_clock()
        clock = experiment_pair.host_clock_preflight(reader=reader)
        measure = graph_prior.previous.graph_measure
        mapped = {label: graphs[label].replay for label in ("A", "B")}
        controls = {label: measure(torch, mapped[label], OPS, raw_clock=reader) for label in ("A", "B")}
        aa_rows, rounds, sequence = experiment_pair.measure_paired_rounds(
            measure, torch, mapped, OPS, ROUNDS, {"raw_clock": reader}, with_aa=True,
            aa_placement="before")
        pair = {"physical_labels": ["A", "B"], "unranked_timing_controls": controls,
                "aa_rounds": aa_rows, "rounds": rounds, "measurement_sequence": sequence,
                "iterations_per_measurement": OPS, "aa_iterations_per_measurement": OPS,
                "aa_kernel": "A", "host_clock_preflight": clock}
        evidence.write_json(output / "pair_progress.json", pair)
        fixed = addresses == (left.data_ptr(), right.data_ptr(), outputs["A"].data_ptr(),
                              outputs["B"].data_ptr())
        if not fixed:
            raise ValueError("fixed tensor addresses changed")
    finally:
        torch.set_float32_matmul_precision(original["precision"])
        torch.backends.cuda.matmul.allow_tf32 = original["tf32"]
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = original["reduced"]
    props = torch.cuda.get_device_properties(0)
    raw = {"schema_version": 1, "session": session,
           "created_at_utc": datetime.now(timezone.utc).isoformat(),
           "base_commit": protocol["base_commit"],
           "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
           "protocol_sha256": evidence.digest(output / "protocol.json"),
           "mapping_plan_sha256": evidence.digest(output / "mapping_plan.json"),
           "source_sha256": {label: evidence.digest(output / f"{label}_source.py") for label in ("A", "B")},
           "script_sha256": evidence.digest(output / "diagnostic_script.py"),
           "size": SIZE, "tolerance": TOLERANCE, "initialization_order": order,
           "environment": {"gpu": props.name, "sm_count": props.multi_processor_count,
                           "compute_capability": f"{props.major}.{props.minor}",
                           "driver": subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version",
                                                                  "--format=csv,noheader"], text=True).strip(),
                           "python": sys.version.split()[0], "torch": torch.__version__,
                           "triton": triton.__version__, "cuda_runtime": torch.version.cuda},
           "precision_policy": {"reference": "FP32 matmul of exact FP16 inputs, rounded to FP16",
                                "active_float32_matmul_precision": "highest",
                                "active_allow_tf32": False,
                                "active_allow_fp16_reduced_precision_reduction": False,
                                "original": original},
           "first_call_ms": first_call, "capture_setup_ms": capture_setup,
           "correctness": checks, "correctness_passed": True,
           "fixed_addresses_unchanged": fixed, "topology": topology,
           "compiler": compiler, "host_clock_preflight": clock, "pair": pair}
    evidence.write_json(output / "raw.json", raw)
    evidence.write_json(output / "summary.json", summarize(raw))
    evidence.manifest(output)
    return verify(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--session", type=int, choices=(1, 2))
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if not args.verify and args.session is None:
        parser.error("--session required for a new run")
    print(json.dumps(verify(args.output) if args.verify else run(args.output, args.session), sort_keys=True))


if __name__ == "__main__":
    main()
