"""Bounded complete-operation CUDA Graph pair for MI-Research-2C.

The normal paired CLI remains the eager control. This script only supplies
fixed-address graph capture and reuses its clocks, A/A schedule, and decision.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
from typing import Any

from matmul_inspector import experiment, experiment_pair


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "examples/kernels/mi_research_2_split2.py"
SPEC = ROOT / "examples/mi_research_2_low_output_split2.json"
PRIOR_SOURCE = ROOT / "artifacts/mi-research-2/evidence-v1/low-output-split2/sources/B.py"
OPERATIONS = 100
ROUNDS = 8
WARMUPS = 30


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def manifest(output: Path) -> None:
    write_json(output / "hash_manifest.json", {
        str(path.relative_to(output)): digest(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "hash_manifest.json"})


def normalize_replay_measurement(sample: dict[str, float], operations: int = OPERATIONS) -> dict[str, float | int]:
    """One graph replay contains `operations` complete GEMMs, not one GEMM."""
    if operations < 1 or not math.isclose(
            sample["cuda_event_elapsed_ms_per_operation"],
            sample["cuda_event_elapsed_ms_per_block"], abs_tol=1e-8):
        raise ValueError("graph sample must contain one measured replay")
    return {**sample,
            "cuda_event_elapsed_ms_per_operation": sample["cuda_event_elapsed_ms_per_block"] / operations,
            "complete_operations_per_block": operations,
            "host_submission_calls_per_block": 1}


def graph_measure(torch, function, iterations: int, *, raw_clock):
    if iterations != OPERATIONS:
        raise ValueError("graph replay operation count changed")
    return normalize_replay_measurement(
        experiment.CudaTritonBackend._measure(torch, function, 1, raw_clock=raw_clock), iterations)


def trace_sequence(trace: dict[str, Any]) -> tuple[list[str], int]:
    events = trace.get("traceEvents")
    if not isinstance(events, list):
        raise ValueError("profiler trace has no events")
    kernels = sorted((event for event in events if event.get("cat") == "kernel"),
                     key=lambda event: event["ts"])
    names = [event["name"] for event in kernels]
    graph_launches = sum(event.get("cat") == "cuda_runtime" and
                         event.get("name") == "cudaGraphLaunch" for event in events)
    return names, graph_launches


def require_complete_trace(trace: dict[str, Any], label: str, operations: int = OPERATIONS) -> dict[str, Any]:
    sequence, launches = trace_sequence(trace)
    expected = (["fc1_kernel"] * operations if label == "A" else
                [name for _ in range(operations) for name in ("split_k_partial", "reduce_partials")])
    if sequence != expected or launches != 1:
        raise ValueError(f"graph {label} profiler trace does not show the complete ordered operation")
    return {"kernel_sequence": sequence, "cuda_graph_launch_calls": launches,
            "kernel_nodes_observed": len(sequence), "complete_operations": operations}


def summarize(raw: dict[str, Any]) -> dict[str, Any]:
    if (raw.get("schema_version") != 1 or raw.get("mode") != "cuda_graph" or
            raw.get("operations_per_replay") != OPERATIONS or raw.get("rounds") != ROUNDS or
            raw.get("warmups_per_graph") != WARMUPS or raw.get("session") not in (1, 2)):
        raise ValueError("invalid graph diagnostic configuration")
    correctness = raw.get("correctness", {})
    if correctness.get("passed") is not True or len(correctness.get("seeds", [])) != 3 or any(
            case.get("seed") != seed or any(case.get("checks", {}).get(label, {}).get("passed") is not True
                                             for label in ("A", "B"))
            for seed, case in enumerate(correctness["seeds"])):
        raise ValueError("incomplete graph numerical correctness record")
    benchmark = raw.get("benchmark", {})
    expected_sequence = experiment_pair.expected_diagnostic_sequence(ROUNDS, OPERATIONS, "before")
    if (benchmark.get("measurement_sequence") != expected_sequence or
            benchmark.get("iterations_per_measurement") != OPERATIONS or
            benchmark.get("aa_kernel") != "A" or
            benchmark.get("aa_iterations_per_measurement") != OPERATIONS or
            len(benchmark.get("rounds", [])) != ROUNDS or
            len(benchmark.get("aa_rounds", [])) != ROUNDS or
            set(benchmark.get("unranked_timing_controls", {})) != {"A", "B"}):
        raise ValueError("graph benchmark sequence or controls changed")
    all_samples = list(benchmark["unranked_timing_controls"].values())
    for index in range(1, ROUNDS + 1):
        for phase, slots in (("rounds", ("A", "B")), ("aa_rounds", ("X", "Y"))):
            row = benchmark[phase][index - 1]
            expected_order = (experiment_pair.pair_order(index) if phase == "rounds" else
                              (["X", "Y"] if index % 2 else ["Y", "X"]))
            if (row.get("round") != index or row.get("order") != expected_order or
                    set(row.get("measurements", {})) != set(slots)):
                raise ValueError("graph paired order or round identity changed")
            all_samples.extend(row["measurements"].values())
    for sample in all_samples:
        if (sample.get("complete_operations_per_block") != OPERATIONS or
                sample.get("host_submission_calls_per_block") != 1 or
                not math.isclose(sample["cuda_event_elapsed_ms_per_operation"] * OPERATIONS,
                                 sample["cuda_event_elapsed_ms_per_block"], abs_tol=1e-8)):
            raise ValueError("graph operation count or normalization changed")
    decision = experiment_pair.summarize_clock_aware(
        {"passed": True, "checks": {label: correctness["seeds"][0]["checks"][label]
                                    for label in ("A", "B")}},
        experiment_pair.timing_rows(raw), None, benchmark)
    if decision.get("rounds") != ROUNDS or decision["decision"] in ("FAILED", "REJECTED_INCORRECT"):
        raise ValueError("graph decision could not replay")
    decision["timing_basis"] = "100 complete fixed-address GEMMs per CUDA Graph replay; one host submission per block"
    metrics = {}
    for label in ("A", "B"):
        rows = [row["measurements"][label] for row in benchmark["rounds"]]
        event_us = [row["cuda_event_elapsed_ms_per_operation"] * 1000 for row in rows]
        host_submission_ms = [row["host_submission_ms_per_block"] for row in rows]
        median = statistics.median(event_us)
        metrics[label] = {"event_us_per_operation_by_round": event_us,
                          "event_median_us": median,
                          "event_range_us": [min(event_us), max(event_us)],
                          "event_mad_us": statistics.median(abs(value - median) for value in event_us),
                          "host_submission_ms_per_block_by_round": host_submission_ms,
                          "host_submission_median_ms_per_block": statistics.median(host_submission_ms)}
    return {"schema_version": 1, "mode": "cuda_graph", "session": raw["session"],
            "decision": decision, "metrics": metrics,
            "graph_profile": raw["graph_profile"]}


def verify(output: Path) -> dict[str, Any]:
    saved = json.loads((output / "hash_manifest.json").read_text())
    if (not isinstance(saved, dict) or any(Path(name).is_absolute() or ".." in Path(name).parts
                                           for name in saved)):
        raise ValueError("invalid graph manifest")
    actual = {str(path.relative_to(output)) for path in output.rglob("*")
              if path.is_file() and path.name != "hash_manifest.json"}
    if actual != set(saved) or any(digest(output / name) != expected
                                   for name, expected in saved.items()):
        raise ValueError("graph manifest mismatch")
    raw = json.loads((output / "raw.json").read_text())
    for name, path in (("source_sha256", "source.py"), ("spec_sha256", "experiment.json"),
                       ("diagnostic_script_sha256", "diagnostic_script.py")):
        if raw[name] != digest(output / path):
            raise ValueError(f"{path} identity mismatch")
    for label, expected_names in (("A", ("conventional",)),
                                  ("B", ("partial", "reduction"))):
        trace_file = output / f"{label}_trace.json"
        profile = require_complete_trace(json.loads(trace_file.read_text()), label)
        if raw["graph_profile"][label] != profile or raw["graph_trace_sha256"][label] != digest(trace_file):
            raise ValueError(f"graph {label} trace identity mismatch")
        for name in expected_names:
            if raw["compiler"][name]["ptx_sha256"] != digest(output / f"{name}.ptx"):
                raise ValueError(f"{name} PTX identity mismatch")
    replayed = summarize(raw)
    if replayed != json.loads((output / "summary.json").read_text()):
        raise ValueError("graph summary does not replay")
    return {"status": "PASS", "session": raw["session"], "decision": replayed["decision"]["decision"],
            "files": len(saved), "profiler_kernel_nodes": {
                label: replayed["graph_profile"][label]["kernel_nodes_observed"] for label in ("A", "B")}}


def run(output: Path, session: int) -> dict[str, Any]:
    if output.exists():
        raise ValueError("graph output directory already exists")
    if session not in (1, 2):
        raise ValueError("session must be 1 or 2")
    if digest(SOURCE) != digest(PRIOR_SOURCE):
        raise ValueError("Split-2 source differs from the retained MI-Research-2 source")
    spec = experiment_pair.validate_spec(json.loads(SPEC.read_text()))
    if (spec["workload"] != {"m": 32, "k": 4096, "n": 2048, "dtype": "float16"} or
            spec["benchmark"]["warmups"] != WARMUPS or
            spec["benchmark"]["iterations"] != OPERATIONS or
            spec["benchmark"]["rounds"] != ROUNDS):
        raise ValueError("low-output experiment settings changed")
    output.mkdir(parents=True)
    shutil.copyfile(SOURCE, output / "source.py")
    shutil.copyfile(SPEC, output / "experiment.json")
    shutil.copyfile(Path(__file__), output / "diagnostic_script.py")

    import torch
    import triton
    from matmul_inspector import triton_fc1

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    module_spec = importlib.util.spec_from_file_location("mi_research_2c_source", output / "source.py")
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError("cannot import retained Split-2 source")
    module = importlib.util.module_from_spec(module_spec)
    old_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        module_spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = old_bytecode
    w, parameters = spec["workload"], spec["kernels"]["B"]["parameters"]
    a = torch.empty((w["m"], w["k"]), device="cuda", dtype=torch.float16)
    b = torch.empty((w["k"], w["n"]), device="cuda", dtype=torch.float16)
    outputs = {label: torch.empty((w["m"], w["n"]), device="cuda", dtype=torch.float16)
               for label in ("A", "B")}
    scratch = torch.empty((module.SPLIT_K, w["m"], w["n"]), device="cuda", dtype=torch.float32)
    a_grid = (triton.cdiv(w["m"], parameters["block_m"]) *
              triton.cdiv(w["n"], parameters["block_n"]),)
    partial_grid = (a_grid[0], module.SPLIT_K)
    reduction_grid = (triton.cdiv(w["m"] * w["n"], module.REDUCTION_BLOCK),)

    def conventional():
        out = outputs["A"]
        return triton_fc1.fc1_kernel[a_grid](
            a, b, out, w["m"], w["n"], w["k"], a.stride(0), a.stride(1),
            b.stride(0), b.stride(1), out.stride(0), out.stride(1),
            BLOCK_M=parameters["block_m"], BLOCK_N=parameters["block_n"],
            BLOCK_K=parameters["block_k"], GROUP_SIZE_M=parameters["group_size_m"],
            num_warps=parameters["num_warps"], num_stages=parameters["num_stages"])

    def split_complete():
        out = outputs["B"]
        partial = module.split_k_partial[partial_grid](
            a, b, scratch, w["m"], w["n"], w["k"], a.stride(0), a.stride(1),
            b.stride(0), b.stride(1), BLOCK_M=parameters["block_m"],
            BLOCK_N=parameters["block_n"], BLOCK_K=parameters["block_k"],
            K_PART=w["k"] // module.SPLIT_K,
            num_warps=parameters["num_warps"], num_stages=parameters["num_stages"])
        reduction = module.reduce_partials[reduction_grid](
            scratch, out, w["m"], w["n"], out.stride(0), out.stride(1),
            BLOCK=module.REDUCTION_BLOCK, num_warps=4)
        return partial, reduction

    def make_inputs(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        return (torch.randn((w["m"], w["k"]), device="cuda", dtype=torch.float16),
                torch.randn((w["k"], w["n"]), device="cuda", dtype=torch.float16))

    original_a, original_b = make_inputs(spec["seed"])
    a.copy_(original_a)
    b.copy_(original_b)
    conventional_launch = conventional()
    partial_launch, reduction_launch = split_complete()
    torch.cuda.synchronize()
    compiler = {}
    for name, launch in (("conventional", conventional_launch),
                         ("partial", partial_launch), ("reduction", reduction_launch)):
        capture = experiment_pair._compiled_capture(launch)
        if capture is None or not isinstance(capture.get("ptx_text"), str):
            raise RuntimeError(f"{name} PTX unavailable")
        (output / f"{name}.ptx").write_text(capture.pop("ptx_text"))
        capture.pop("sass_text", None)
        capture["ptx_sha256"] = digest(output / f"{name}.ptx")
        compiler[name] = capture
    eager_functions = {"A": conventional, "B": split_complete}
    for label in ("A", "B"):
        for _ in range(WARMUPS):
            eager_functions[label]()
        torch.cuda.synchronize()

    capture_order = ("A", "B") if session == 1 else ("B", "A")
    graphs = {}
    setup_ms = {}
    for label in capture_order:
        torch.cuda.synchronize()
        started = time.perf_counter_ns()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(OPERATIONS):
                eager_functions[label]()
        torch.cuda.synchronize()
        graphs[label] = graph
        setup_ms[label] = (time.perf_counter_ns() - started) / 1e6

    correctness_cases = []
    for seed in range(spec["seed"], spec["seed"] + 3):
        left, right = make_inputs(seed)
        a.copy_(left)
        b.copy_(right)
        reference = torch.matmul(a, b)
        for out in outputs.values():
            out.fill_(float("nan"))
        scratch.fill_(float("nan"))
        torch.cuda.synchronize()
        graphs["A"].replay()
        graphs["B"].replay()
        torch.cuda.synchronize()
        checks = {label: experiment_pair.check_output(torch, reference, outputs[label], spec["correctness"])
                  for label in ("A", "B")}
        correctness_cases.append({"seed": seed, "checks": checks})
        if not all(check["passed"] for check in checks.values()):
            raise RuntimeError(f"CUDA Graph numerical correctness failed for seed {seed}")
    a.copy_(original_a)
    b.copy_(original_b)
    torch.cuda.synchronize()

    graph_profiles = {}
    trace_hashes = {}
    for label in ("A", "B"):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as profiler:
            graphs[label].replay()
            torch.cuda.synchronize()
        trace_path = output / f"{label}_trace.json"
        profiler.export_chrome_trace(str(trace_path))
        graph_profiles[label] = require_complete_trace(json.loads(trace_path.read_text()), label)
        trace_hashes[label] = digest(trace_path)

    functions = {label: graphs[label].replay for label in ("A", "B")}
    for label in ("A", "B"):
        for _ in range(WARMUPS):
            functions[label]()
        torch.cuda.synchronize()
    clock_reader = experiment_pair.raw_clock()
    clock_preflight = experiment_pair.host_clock_preflight(reader=clock_reader)
    measure_options = {"raw_clock": clock_reader}
    controls = {label: graph_measure(torch, functions[label], OPERATIONS, **measure_options)
                for label in ("A", "B")}
    aa_rounds, rounds, sequence = experiment_pair.measure_paired_rounds(
        graph_measure, torch, functions, OPERATIONS, ROUNDS, measure_options,
        with_aa=True, aa_placement="before")
    props = torch.cuda.get_device_properties(0)
    git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                              capture_output=True, text=True).stdout.strip()
    raw = {"schema_version": 1, "mode": "cuda_graph", "session": session,
           "created_at_utc": datetime.now(timezone.utc).isoformat(), "git_head": git_head,
           "source_sha256": digest(SOURCE), "spec_sha256": digest(SPEC),
           "diagnostic_script_sha256": digest(Path(__file__)),
           "environment": {"gpu": props.name, "sm_count": props.multi_processor_count,
                           "python": sys.version.split()[0], "torch": torch.__version__,
                           "triton": triton.__version__, "cuda_runtime": torch.version.cuda},
           "operations_per_replay": OPERATIONS, "rounds": ROUNDS,
           "warmups_per_graph": WARMUPS, "capture_order": list(capture_order),
           "graph_setup_ms": setup_ms, "graph_profile": graph_profiles,
           "graph_trace_sha256": trace_hashes,
           "semantics": {"A": "100 fixed-address conventional GEMM nodes; output overwritten each node",
                         "B": "100 ordered partial-GEMM/reduction pairs; fixed FP32 scratch and output; "
                              "partial writes every scratch element before its reduction",
                         "scratch_bytes": scratch.numel() * scratch.element_size(),
                         "capture_excluded_from_timing": True,
                         "profiler_trace_excluded_from_timing": True,
                         "graph_replay_calls_per_block": 1},
           "compiler": compiler,
           "correctness": {"passed": True, "reference": "pytorch_matmul (cuBLAS)",
                           "tolerances": spec["correctness"], "seeds": correctness_cases},
           "benchmark": {"mode": "cuda_graph", "iterations_per_measurement": OPERATIONS,
                         "host_clock_preflight": clock_preflight,
                         "unranked_timing_controls": controls,
                         "aa_rounds": aa_rounds, "aa_kernel": "A",
                         "aa_iterations_per_measurement": OPERATIONS,
                         "rounds": rounds, "measurement_sequence": sequence,
                         "timing_boundaries": "perf start, RAW start, CUDA start event, one 100-operation "
                         "graph replay, CUDA end event, host submission timestamp, end-event sync, RAW end, perf end"}}
    write_json(output / "raw.json", raw)
    write_json(output / "summary.json", summarize(raw))
    manifest(output)
    return verify(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--session", type=int, choices=(1, 2))
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify == (args.session is not None):
        parser.error("choose exactly one of --verify or --session")
    try:
        result = verify(args.output) if args.verify else run(args.output, args.session)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        if not args.verify and args.output.is_dir():
            write_json(args.output / "failure.json", {"type": type(error).__name__, "message": str(error)})
        print(f"MI-Research-2C graph diagnostic failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
