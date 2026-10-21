"""Bounded CUDA Graph study of output-block-owned Perminov at N=96."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
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
    from scripts import run_mi_research_6d_graph as graph_prior
    from scripts import run_mi_research_7 as prior
    from scripts import generate_mi_research_7b_kernel as generator
except ModuleNotFoundError:
    import run_mi_research_2c_graph as evidence
    import run_mi_research_3b_correctness as diagnostics
    import run_mi_research_4_feasibility as cases
    import run_mi_research_6_gpu_probe as loader
    import run_mi_research_6d_graph as graph_prior
    import run_mi_research_7 as prior
    import generate_mi_research_7b_kernel as generator

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/mi-research-7b"
PROTOCOL = BASE / "protocol.json"
MAPPING = BASE / "mapping_plan.json"
SIZE, OPS, ROUNDS, WARMUPS = 96, 20, 8, 10
TOLERANCE = {"atol": 0.01, "rtol": 0.01}


def check_protocol() -> dict:
    p = json.loads(PROTOCOL.read_text())
    if (p["base_commit"] != "0b6c741f92073b20ec9009ef808704225e47826c" or
        p["mapping_plan_sha256"] != evidence.digest(MAPPING) or
        p["source_sha256"] != {"A": evidence.digest(generator.ORIGINAL),
                               "B": evidence.digest(generator.TARGET)} or
        p["correctness_cases"] != list(cases.CASE_NAMES) or
        p["tolerance"] != TOLERANCE or p["graph_operations_per_replay"] != OPS or
        p["paired_rounds_per_session"] != ROUNDS or p["warmups"] != WARMUPS or
        generator.TARGET.read_text() != generator.render()):
        raise ValueError("7B frozen protocol or generated source changed")
    if prior.check_protocol()["base_commit"] != prior.BASE_COMMIT:
        raise ValueError("7 historical checkpoint changed")
    return p


def topology(trace: dict, label: str, graph: bool) -> dict:
    events = trace["traceEvents"]
    kernels = [x for x in events if x.get("cat") == "kernel"]
    expected = OPS if graph else 1
    if len(kernels) != expected:
        raise ValueError(f"{label} topology has {len(kernels)} kernels, expected {expected}")
    grid = {"A": [2, 2, 1], "B": [2, 2, 9], "C": None}[label]
    if label != "C" and any(x.get("args", {}).get("grid") != grid or
                            x.get("args", {}).get("block") != [128, 1, 1] or
                            "_fused" not in x.get("name", "") for x in kernels):
        raise ValueError(f"{label} grid or block changed")
    launches = sum(x.get("cat") == "cuda_runtime" and x.get("name") == "cudaGraphLaunch"
                   for x in events)
    if launches != int(graph):
        raise ValueError(f"{label} graph launch count changed")
    return {"physical_kernel_events": len(kernels), "kernels_per_operation": 1,
            "cuda_graph_launch_calls": launches, "grid": kernels[0].get("args", {}).get("grid"),
            "block": kernels[0].get("args", {}).get("block"),
            "first_kernel_name": kernels[0]["name"]}


def capture_trace(torch, fn, path: Path, label: str, graph: bool) -> dict:
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as profiler:
        fn()
        torch.cuda.synchronize()
    profiler.export_chrome_trace(str(path))
    return topology(json.loads(path.read_text()), label, graph)


def summarize(raw: dict) -> dict:
    if raw["protocol_sha256"] != evidence.digest(PROTOCOL) or not raw["correctness_passed"]:
        raise ValueError("invalid 7B raw record")
    p = raw["pair"]
    if (len(p["rounds"]) != ROUNDS or len(p["aa_rounds"]) != ROUNDS or
        p["measurement_sequence"] != experiment_pair.expected_diagnostic_sequence(ROUNDS, OPS, "before") or
        set(p["unranked_timing_controls"]) != {"A", "B"} or
        p["aa_kernel"] != "A" or p["iterations_per_measurement"] != OPS or
        p["aa_iterations_per_measurement"] != OPS):
        raise ValueError("7B timing shape changed")
    for i, row in enumerate(p["rounds"], 1):
        if row["round"] != i or row["order"] != experiment_pair.pair_order(i):
            raise ValueError("7B pair order changed")
    c0 = raw["correctness"][0]["checks"]
    benchmark = {**p, "host_clock_preflight": raw["host_clock_preflight"]}
    decision = experiment_pair.summarize_clock_aware(
        {"passed": True, "checks": {"A": c0[raw["physical_labels"][0]]["cuda_graph"][0],
                                   "B": c0[raw["physical_labels"][1]]["cuda_graph"][0]}},
        experiment_pair.timing_rows({"benchmark": benchmark}), None, benchmark)
    if decision["decision"] in ("FAILED", "REJECTED_INCORRECT"):
        raise ValueError("7B decision invalid")
    samples = {label: [row["measurements"][label]["cuda_event_elapsed_ms_per_operation"] * 1000
                       for row in p["rounds"]] for label in ("A", "B")}
    return {"decision": decision, "event_us_per_operation": samples,
            "event_median_us": {k: statistics.median(v) for k, v in samples.items()},
            "paired_A_minus_B_us": [a - b for a, b in zip(samples["A"], samples["B"])],
            "aa_max_absolute_gap_us": decision["aa_control"]["max_absolute_difference_ms"] * 1000}


def verify(output: Path) -> dict:
    check_protocol()
    manifest = json.loads((output / "hash_manifest.json").read_text())
    files = {str(x.relative_to(output)) for x in output.rglob("*")
             if x.is_file() and x.name != "hash_manifest.json"}
    if files != set(manifest) or any(evidence.digest(output / name) != sha for name, sha in manifest.items()):
        raise ValueError("7B manifest mismatch")
    if (output / "failure.json").exists():
        return {"status": "FAILURE_RETAINED", **json.loads((output / "failure.json").read_text())}
    raw = json.loads((output / "raw.json").read_text())
    if (raw["protocol_sha256"] != evidence.digest(output / "protocol.json") or
        raw["source_sha256"] != {"A": evidence.digest(output / "A_source.py"),
                                 "B": evidence.digest(output / "B_source.py")} or
        raw["script_sha256"] != evidence.digest(output / "diagnostic_script.py") or
        raw["correctness"] != json.loads((output / "correctness_progress.json").read_text()) or
        raw["pair"] != json.loads((output / "pair_progress.json").read_text())):
        raise ValueError("7B provenance mismatch")
    for label in raw["physical_labels"]:
        if raw["topology"][label] != topology(json.loads((output / f"{label}_trace.json").read_text()), label, True):
            raise ValueError("7B topology changed")
        if label != "C":
            c = raw["compiler"][label]
            if c["ptx_sha256"] != evidence.digest(output / f"{label}.ptx") or any(
                sha != evidence.digest(output / f"{label}.{kind}") for kind, sha in c["ir_sha256"].items()):
                raise ValueError("7B compiler artifact changed")
    if json.loads((output / "summary.json").read_text()) != summarize(raw):
        raise ValueError("7B summary changed")
    return {"status": "PASS", "session": raw["session"],
            "comparison": raw["comparison"], "decision": summarize(raw)["decision"]["decision"]}


def run(output: Path, session: int, comparison: str) -> dict:
    if output.exists() or session not in (1, 2) or comparison not in ("primary", "cublas"):
        raise ValueError("invalid or existing output")
    protocol = check_protocol()
    if comparison == "cublas":
        if session != 1 or any(verify(BASE / f"primary-s{i}")["decision"] != "B_FASTER" for i in (1, 2)):
            raise ValueError("conditional cuBLAS gate not met")
    output.mkdir(parents=True)
    for source, name in ((PROTOCOL, "protocol.json"), (MAPPING, "mapping_plan.json"),
                         (generator.ORIGINAL, "A_source.py"), (generator.TARGET, "B_source.py"),
                         (Path(__file__), "diagnostic_script.py")):
        shutil.copyfile(source, output / name)
    try:
        return _run_gpu(output, session, comparison, protocol)
    except Exception as error:
        evidence.write_json(output / "failure.json", {"type": type(error).__name__,
                                                       "message": str(error), "session": session,
                                                       "comparison": comparison})
        evidence.manifest(output)
        raise


def _run_gpu(output: Path, session: int, comparison: str, protocol: dict) -> dict:
    import torch
    import triton
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    modules = {"A": loader.load_snapshot(output / "A_source.py"),
               "B": loader.load_snapshot(output / "B_source.py")}
    a0, b0 = cases._inputs(torch, SIZE, "seed-0")
    left, right = torch.empty_like(a0), torch.empty_like(b0)
    left.copy_(a0); right.copy_(b0)
    labels = ("A", "B") if comparison == "primary" else ("B", "C")
    outputs = {label: torch.empty((SIZE, SIZE), device="cuda", dtype=torch.float16) for label in labels}
    functions = {label: (lambda label=label: modules[label].call(left, right, outputs[label]))
                 for label in labels if label != "C"}
    if "C" in labels:
        functions["C"] = lambda: torch.mm(left, right, out=outputs["C"])
    order = list(labels if session == 1 else reversed(labels))
    original = {"precision": torch.get_float32_matmul_precision(),
                "tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                "reduced": bool(torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)}
    compiler, first_ms, capture_ms, graphs, checks, topo = {}, {}, {}, {}, [], {}
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        for label in order:
            torch.cuda.synchronize(); start = time.perf_counter_ns()
            compiled = functions[label]()
            torch.cuda.synchronize()
            first_ms[label] = (time.perf_counter_ns() - start) / 1e6
            if label != "C":
                compiler[label] = prior._capture_compiler(compiled, output, label)
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
            capture_ms[label] = (time.perf_counter_ns() - start) / 1e6
        for case in cases.CASE_NAMES:
            aa, bb = cases._inputs(torch, SIZE, case)
            left.copy_(aa); right.copy_(bb)
            reference = torch.matmul(left.float(), right.float()).half()
            row = {}
            for label in labels:
                eager, graph_checks = [], []
                for fn, destination in ((functions[label], eager), (graphs[label].replay, graph_checks)):
                    for _ in range(2):
                        outputs[label].fill_(float("nan")); fn(); torch.cuda.synchronize()
                        destination.append(diagnostics.summarize_error(
                            torch, outputs[label], reference, a=left, b=right))
                saved = outputs[label].clone(); functions[label](); torch.cuda.synchronize()
                row[label] = {"eager": eager, "cuda_graph": graph_checks,
                              "graph_eager_bitwise_equal": bool(torch.equal(saved, outputs[label]))}
            checks.append({"case": case, "checks": row, "output_poisoned_before_each_check": True})
            evidence.write_json(output / "correctness_progress.json", checks)
            if any(not item["passed"] or item["failing_elements"] or item["candidate_nonfinite_count"]
                   for label in labels for phase in ("eager", "cuda_graph") for item in row[label][phase]) or any(
                       not row[label]["graph_eager_bitwise_equal"] for label in labels):
                raise ValueError(f"7B numerical gate failed: {case}")
        left.copy_(a0); right.copy_(b0); torch.cuda.synchronize()
        for label in order:
            topo[label] = capture_trace(torch, graphs[label].replay, output / f"{label}_trace.json", label, True)
        for label in order:
            for _ in range(WARMUPS): graphs[label].replay()
            torch.cuda.synchronize()
        reader = experiment_pair.raw_clock()
        clock = experiment_pair.host_clock_preflight(reader=reader)
        measure = graph_prior.previous.graph_measure
        mapped = {"A": graphs[labels[0]].replay, "B": graphs[labels[1]].replay}
        controls = {key: measure(torch, mapped[key], OPS, raw_clock=reader) for key in ("A", "B")}
        aa_rows, rounds, sequence = experiment_pair.measure_paired_rounds(
            measure, torch, mapped, OPS, ROUNDS, {"raw_clock": reader}, with_aa=True, aa_placement="before")
        pair = {"physical_labels": list(labels), "unranked_timing_controls": controls,
                "aa_rounds": aa_rows, "rounds": rounds, "measurement_sequence": sequence,
                "iterations_per_measurement": OPS, "aa_iterations_per_measurement": OPS,
                "aa_kernel": "A", "host_clock_preflight": clock}
        evidence.write_json(output / "pair_progress.json", pair)
    finally:
        torch.set_float32_matmul_precision(original["precision"])
        torch.backends.cuda.matmul.allow_tf32 = original["tf32"]
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = original["reduced"]
    props = torch.cuda.get_device_properties(0)
    raw = {"schema_version": 1, "session": session, "comparison": comparison,
           "created_at_utc": datetime.now(timezone.utc).isoformat(),
           "base_commit": protocol["base_commit"],
           "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
           "protocol_sha256": evidence.digest(output / "protocol.json"),
           "mapping_plan_sha256": evidence.digest(output / "mapping_plan.json"),
           "source_sha256": {key: evidence.digest(output / f"{key}_source.py") for key in ("A", "B")},
           "script_sha256": evidence.digest(output / "diagnostic_script.py"),
           "physical_labels": list(labels), "initialization_order": order,
           "size": SIZE, "tolerance": TOLERANCE,
           "environment": {"gpu": props.name, "sm_count": props.multi_processor_count,
                           "compute_capability": f"{props.major}.{props.minor}",
                           "driver": subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version",
                                                                  "--format=csv,noheader"], text=True).strip(),
                           "python": sys.version.split()[0], "torch": torch.__version__,
                           "triton": triton.__version__, "cuda_runtime": torch.version.cuda},
           "precision_policy": {"active_float32_matmul_precision": "highest",
                                "active_allow_tf32": False,
                                "active_allow_fp16_reduced_precision_reduction": False,
                                "original": original},
           "first_call_ms": first_ms, "capture_setup_ms": capture_ms,
           "correctness": checks, "correctness_passed": True,
           "fixed_addresses_unchanged": True, "topology": topo, "compiler": compiler,
           "host_clock_preflight": clock, "pair": pair}
    evidence.write_json(output / "raw.json", raw)
    evidence.write_json(output / "summary.json", summarize(raw))
    evidence.manifest(output)
    return verify(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--session", type=int, choices=(1, 2))
    parser.add_argument("--comparison", choices=("primary", "cublas"), default="primary")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if not args.verify and args.session is None:
        parser.error("--session required for a new run")
    print(json.dumps(verify(args.output) if args.verify else run(args.output, args.session, args.comparison),
                     sort_keys=True))


if __name__ == "__main__":
    main()
