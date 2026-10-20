"""Bounded fixed-address Laderman versus conventional CUDA Graph study.

This diagnostic reuses the committed staged source and Matmul Inspector's
existing clock, paired-order, A/A, and decision functions. One graph replay
contains 20 complete 3x3 matrix multiplications.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

from matmul_inspector import experiment, experiment_pair
try:
    from scripts import run_mi_research_2c_graph as artifacts
    from scripts import run_mi_research_3b_correctness as diagnostics
    from scripts import run_mi_research_4_feasibility as prior
    from scripts import run_mi_research_6_feasibility as exact
    from scripts import run_mi_research_6_gpu_probe as eager
except ModuleNotFoundError:
    import run_mi_research_2c_graph as artifacts
    import run_mi_research_3b_correctness as diagnostics
    import run_mi_research_4_feasibility as prior
    import run_mi_research_6_feasibility as exact
    import run_mi_research_6_gpu_probe as eager

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "examples/kernels/mi_research_6_staged_3x3.py"
COEFFICIENTS = ROOT / "artifacts/mi-research-6/coefficients.json"
PREVIOUS_BUNDLE = ROOT / "artifacts/mi-research-6/gpu-probe-n96-eager-s1"
SOURCE_SHA256 = "8b328950567868b2d119e021a0dd3bcc312d6e1c8138a1577b0e0491949fbca1"
COEFFICIENTS_SHA256 = "4abdcb3db4094b17dba24e8c23b17206fae2e82d769b61fa35952831480a68ef"
BASE_COMMIT = "c52b21e4f87cfdb3bb254b0964099ffa13148c5e"
SIZE = 96
RANKS = {"A": 27, "B": 23}
LABELS = ("A", "B")
CASES = prior.CASE_NAMES
TOLERANCE = {"atol": 0.01, "rtol": 0.01}
OPERATIONS = 20
ROUNDS = 8
WARMUPS = 10


def graph_measure(torch, function, iterations: int, *, raw_clock):
    """One timed replay contains exactly 20 complete operations."""
    if iterations != OPERATIONS:
        raise ValueError("graph operation count changed")
    sample = experiment.CudaTritonBackend._measure(torch, function, 1, raw_clock=raw_clock)
    if not math.isclose(sample["cuda_event_elapsed_ms_per_operation"],
                        sample["cuda_event_elapsed_ms_per_block"], abs_tol=1e-8):
        raise ValueError("unexpected replay normalization")
    return {**sample, "cuda_event_elapsed_ms_per_operation":
            sample["cuda_event_elapsed_ms_per_block"] / OPERATIONS,
            "complete_operations_per_block": OPERATIONS,
            "host_submission_calls_per_block": 1}


def trace_profile(trace: dict, rank: int) -> dict:
    events = trace.get("traceEvents")
    if not isinstance(events, list):
        raise ValueError("profiler trace missing events")
    kernels = sorted((event for event in events if event.get("cat") == "kernel"),
                     key=lambda event: event["ts"])
    names = [event["name"] for event in kernels]
    expected_count = OPERATIONS * (rank + 1)
    if len(names) != expected_count:
        raise ValueError(f"expected {expected_count} physical GPU kernel events; saw {len(names)}")
    for operation in range(OPERATIONS):
        block = names[operation * (rank + 1):(operation + 1) * (rank + 1)]
        if (not all("_block_product" in name for name in block[:-1])
                or "_reconstruct" not in block[-1]):
            raise ValueError(f"incomplete or reordered graph operation {operation}")
    launches = sum(event.get("cat") == "cuda_runtime" and
                   event.get("name") == "cudaGraphLaunch" for event in events)
    if launches != 1:
        raise ValueError(f"expected one graph submission; saw {launches}")
    return {"kernel_events": len(names), "product_events": OPERATIONS * rank,
            "reconstruction_events": OPERATIONS, "graph_launch_calls": launches,
            "complete_operations": OPERATIONS,
            "ordered_kernel_names_sha256": hashlib.sha256(("\n".join(names) + "\n").encode()).hexdigest()}


def _stats(values: list[float]) -> dict:
    median = statistics.median(values)
    return {"median_us": median, "min_us": min(values), "max_us": max(values),
            "mad_us": statistics.median(abs(value - median) for value in values),
            "cv": statistics.stdev(values) / statistics.fmean(values)}


def summarize(raw: dict) -> dict:
    if (raw.get("schema_version") != 1 or raw.get("mode") != "cuda_graph"
            or raw.get("session") not in (1, 2) or raw.get("size") != SIZE
            or raw.get("ranks") != RANKS or raw.get("operations_per_replay") != OPERATIONS
            or raw.get("warmups_per_graph") != WARMUPS or raw.get("tolerance") != TOLERANCE
            or raw.get("capture_order") != (["A", "B"] if raw["session"] == 1 else ["B", "A"])):
        raise ValueError("MI-Research-6B protocol changed")
    cases = raw.get("correctness", {}).get("cases", [])
    if [row.get("case") for row in cases] != list(CASES):
        raise ValueError("missing or reordered correctness case")
    for row in cases:
        for label in LABELS:
            if (len(row["replays"][label]) != 2 or
                    any(check["passed"] is not True or check["failing_elements"] != 0
                        or check["candidate_nonfinite_count"] != 0
                        for check in row["replays"][label]) or
                    row["eager_agreement"][label]["passed"] is not True):
                raise ValueError("graph numerical correctness or eager agreement failed")
    if raw["correctness"]["fixed_buffer_addresses_unchanged"] is not True:
        raise ValueError("graph addresses changed")
    for label in LABELS:
        profile = raw["graph_profile"][label]
        if (profile["kernel_events"] != OPERATIONS * (RANKS[label] + 1)
                or profile["product_events"] != OPERATIONS * RANKS[label]
                or profile["reconstruction_events"] != OPERATIONS
                or profile["graph_launch_calls"] != 1):
            raise ValueError("graph topology changed")
    benchmark = raw.get("benchmark")
    if not isinstance(benchmark, dict):
        raise ValueError("missing benchmark")
    if (benchmark.get("mode") != "cuda_graph_fixed_address"
            or benchmark.get("iterations_per_measurement") != OPERATIONS
            or benchmark.get("measurement_sequence") !=
                experiment_pair.expected_diagnostic_sequence(ROUNDS, OPERATIONS, "before")
            or len(benchmark.get("rounds", [])) != ROUNDS
            or len(benchmark.get("aa_rounds", [])) != ROUNDS
            or set(benchmark.get("unranked_timing_controls", {})) != set(LABELS)):
        raise ValueError("paired protocol or controls changed")
    samples = list(benchmark["unranked_timing_controls"].values())
    for index in range(1, ROUNDS + 1):
        for key, names in (("rounds", ("A", "B")), ("aa_rounds", ("X", "Y"))):
            row = benchmark[key][index - 1]
            order = experiment_pair.pair_order(index) if key == "rounds" else (
                ["X", "Y"] if index % 2 else ["Y", "X"])
            if (row.get("round") != index or row.get("order") != order
                    or set(row.get("measurements", {})) != set(names)):
                raise ValueError("paired round order changed")
            samples.extend(row["measurements"].values())
    for sample in samples:
        if (sample.get("complete_operations_per_block") != OPERATIONS
                or sample.get("host_submission_calls_per_block") != 1
                or not math.isclose(sample["cuda_event_elapsed_ms_per_operation"] * OPERATIONS,
                                    sample["cuda_event_elapsed_ms_per_block"], abs_tol=1e-8)):
            raise ValueError("graph operation count or normalization changed")
    decision = experiment_pair.summarize_clock_aware(
        {"passed": True, "checks": {label: cases[0]["replays"][label][0] for label in LABELS}},
        experiment_pair.timing_rows(raw), None, benchmark)
    if decision.get("rounds") != ROUNDS or decision["decision"] in ("FAILED", "REJECTED_INCORRECT"):
        raise ValueError("existing paired decision did not replay")
    decision["timing_basis"] = "one fixed-address CUDA Graph replay of 20 complete staged GEMMs per block"
    timing = {}
    for label in LABELS:
        values = [round_row["measurements"][label]["cuda_event_elapsed_ms_per_operation"] * 1000
                  for round_row in benchmark["rounds"]]
        timing[label] = {**_stats(values), "rounds_us": values}
    paired = [a - b for a, b in zip(timing["A"]["rounds_us"], timing["B"]["rounds_us"])]
    aa = [(round_row["measurements"]["X"]["cuda_event_elapsed_ms_per_operation"] -
           round_row["measurements"]["Y"]["cuda_event_elapsed_ms_per_operation"]) * 1000
          for round_row in benchmark["aa_rounds"]]
    return {"schema_version": 1, "session": raw["session"],
            "correctness_passed": True, "topology_passed": True,
            "decision": decision, "event_us": timing,
            "paired_A_minus_B_us": paired, "paired_median_us": statistics.median(paired),
            "paired_mad_us": statistics.median(abs(value - statistics.median(paired)) for value in paired),
            "aa_differences_us": aa, "aa_max_absolute_gap_us": max(abs(x) for x in aa),
            "b_faster_rounds": sum(x > 0 for x in paired),
            "a_faster_rounds": sum(x < 0 for x in paired)}


def verify(output: Path) -> dict:
    hashes = json.loads((output / "hash_manifest.json").read_text())
    actual = {str(path.relative_to(output)) for path in output.rglob("*")
              if path.is_file() and path.name != "hash_manifest.json"}
    if (actual != set(hashes) or any(Path(name).is_absolute() or ".." in Path(name).parts
                                    or artifacts.digest(output / name) != value
                                    for name, value in hashes.items())):
        raise ValueError("MI-Research-6B artifact manifest mismatch")
    if (output / "failure.json").exists():
        failure = json.loads((output / "failure.json").read_text())
        return {"status": "FAILURE_RETAINED", "type": failure["type"]}
    raw = json.loads((output / "raw.json").read_text())
    for key, name in (("source_sha256", "staged_source.py"),
                      ("coefficients_sha256", "coefficients.json"),
                      ("script_sha256", "diagnostic_script.py")):
        if raw[key] != artifacts.digest(output / name):
            raise ValueError(f"source identity changed: {name}")
    if raw["source_sha256"] != SOURCE_SHA256 or raw["coefficients_sha256"] != COEFFICIENTS_SHA256:
        raise ValueError("frozen MI-Research-6 identity changed")
    for label in LABELS:
        trace_path = output / f"{label}_graph_trace.json"
        trace = json.loads(trace_path.read_text())
        if (raw["graph_profile"][label] != trace_profile(trace, RANKS[label])
                or raw["graph_trace_sha256"][label] != artifacts.digest(trace_path)):
            raise ValueError("graph profiler topology differs from trace")
        for phase in ("product", "reconstruction"):
            record = raw["compiler"][label][phase]
            if record["ptx_sha256"] != artifacts.digest(output / f"{label}_{phase}.ptx"):
                raise ValueError("PTX hash mismatch")
            for ir, value in record["ir_sha256"].items():
                if value != artifacts.digest(output / f"{label}_{phase}.{ir}"):
                    raise ValueError("compiler IR hash mismatch")
    replay = summarize(raw)
    if replay != json.loads((output / "summary.json").read_text()):
        raise ValueError("graph summary differs from replay")
    return {"status": "PASS", "session": raw["session"],
            "decision": replay["decision"]["decision"],
            "kernel_events": {label: raw["graph_profile"][label]["kernel_events"]
                              for label in LABELS}}


def run(output: Path, session: int) -> dict:
    if output.exists():
        raise FileExistsError("graph evidence directory already exists")
    if session not in (1, 2):
        raise ValueError("session must be 1 or 2")
    if artifacts.digest(SOURCE) != SOURCE_SHA256 or artifacts.digest(COEFFICIENTS) != COEFFICIENTS_SHA256:
        raise ValueError("frozen source or coefficients changed")
    if eager.verify(PREVIOUS_BUNDLE)["status"] != "PASS":
        raise ValueError("prior eager evidence failed verification")
    output.mkdir(parents=True)
    for source, name in ((SOURCE, "staged_source.py"), (COEFFICIENTS, "coefficients.json"),
                         (Path(__file__), "diagnostic_script.py")):
        shutil.copyfile(source, output / name)
    try:
        return _run_gpu(output, session)
    except Exception as error:
        artifacts.write_json(output / "failure.json", {
            "type": type(error).__name__, "message": str(error), "stage": "graph_study"})
        artifacts.manifest(output)
        raise


def _run_gpu(output: Path, session: int) -> dict:
    import torch
    import triton

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    source = eager.load_snapshot(output / "staged_source.py")
    actions = {"A": exact.conventional(),
               "B": exact.parse_laderman(exact.SOURCE.read_text())}
    expected_coefficients = {"conventional_27": exact.coefficients(actions["A"]),
                             "laderman_23_l_equals_1": exact.coefficients(actions["B"])}
    if json.loads((output / "coefficients.json").read_text()) != expected_coefficients:
        raise ValueError("coefficient snapshot differs from exact decomposition")
    buffers = {label: source.buffers(actions[label]) for label in LABELS}
    left, right = prior._inputs(torch, SIZE, "seed-0")
    fixed_addresses = {"A": left.data_ptr(), "B": right.data_ptr(),
                       **{f"{label}_{kind}": tensor.data_ptr()
                          for label in LABELS for kind, tensor in zip(
                              ("coefficients", "scratch", "output"), buffers[label])}}
    def operation(label: str, capture=None):
        return source.call(left, right, *buffers[label], launches=capture)
    original = {"fp32_matmul_precision": torch.get_float32_matmul_precision(),
                "allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                "allow_fp16_reduced_precision_reduction": bool(
                    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)}
    compiler, setup_ms, graph_profiles, trace_hashes, correctness_cases = {}, {}, {}, {}, []
    timing = None
    capture_order = ("A", "B") if session == 1 else ("B", "A")
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        first_compile_ms = {}
        for label in LABELS:
            launch_capture = {}
            torch.cuda.synchronize()
            started = time.perf_counter_ns()
            operation(label, launch_capture)
            torch.cuda.synchronize()
            first_compile_ms[label] = (time.perf_counter_ns() - started) / 1e6
            compiler[label] = {}
            for phase in ("product", "reconstruction"):
                kernel = launch_capture[phase]
                details = experiment_pair._compiled_capture(kernel)
                if details is None or not isinstance(details.get("ptx_text"), str):
                    raise ValueError(f"{label} {phase} compiler evidence unavailable")
                (output / f"{label}_{phase}.ptx").write_text(details["ptx_text"])
                ir_hashes = {}
                for kind in ("ttir", "ttgir", "llir"):
                    value = kernel.asm.get(kind)
                    if isinstance(value, str):
                        (output / f"{label}_{phase}.{kind}").write_text(value)
                        ir_hashes[kind] = artifacts.digest(output / f"{label}_{phase}.{kind}")
                compiler[label][phase] = {
                    "kernel_name": details["kernel_name"],
                    "ptx_sha256": artifacts.digest(output / f"{label}_{phase}.ptx"),
                    "ir_sha256": ir_hashes,
                    "metadata": {**details["metadata"],
                                 "n_spills": getattr(kernel.metadata, "n_spills", None),
                                 "global_scratch_size": getattr(kernel.metadata, "global_scratch_size", None)},
                    "sass_status": "UNAVAILABLE" if not details.get("sass_text") else "AVAILABLE_NOT_RETAINED",
                }
        for label in LABELS:
            for _ in range(WARMUPS):
                operation(label)
            torch.cuda.synchronize()
        graphs = {}
        for label in capture_order:
            torch.cuda.synchronize()
            started = time.perf_counter_ns()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(OPERATIONS):
                    operation(label)
            torch.cuda.synchronize()
            setup_ms[label] = (time.perf_counter_ns() - started) / 1e6
            graphs[label] = graph
        artifacts.write_json(output / "capture_preflight.json", {
            "session": session, "capture_order": list(capture_order),
            "operations_per_graph": OPERATIONS, "setup_ms": setup_ms,
            "first_compile_ms": first_compile_ms,
            "fixed_addresses_unchanged": fixed_addresses == {
                "A": left.data_ptr(), "B": right.data_ptr(),
                **{f"{label}_{kind}": tensor.data_ptr()
                   for label in LABELS for kind, tensor in zip(
                       ("coefficients", "scratch", "output"), buffers[label])}},
        })
        for case in CASES:
            input_a, input_b = prior._inputs(torch, SIZE, case)
            left.copy_(input_a)
            right.copy_(input_b)
            reference = torch.matmul(left.float(), right.float()).half()
            records = {label: [] for label in LABELS}
            graph_outputs = {}
            for replay_index in range(2):
                for label in LABELS:
                    buffers[label][1].fill_(float("nan"))
                    buffers[label][2].fill_(float("nan"))
                    graphs[label].replay()
                    torch.cuda.synchronize()
                    check = diagnostics.summarize_error(
                        torch, buffers[label][2], reference, a=left, b=right)
                    records[label].append(check)
                    if not check["passed"] or check["failing_elements"] != 0:
                        raise ValueError(f"graph {label} failed correctness: {case}, replay {replay_index}")
                    if replay_index == 1:
                        graph_outputs[label] = buffers[label][2].clone()
            eager_agreement = {}
            for label in LABELS:
                operation(label)
                torch.cuda.synchronize()
                eager_agreement[label] = experiment_pair.check_output(
                    torch, graph_outputs[label], buffers[label][2], TOLERANCE)
                eager_agreement[label]["bitwise_equal"] = bool(torch.equal(
                    graph_outputs[label], buffers[label][2]))
                if not eager_agreement[label]["passed"]:
                    raise ValueError(f"graph/eager disagreement: {label}, {case}")
            correctness_cases.append({"case": case, "replays": records,
                                      "eager_agreement": eager_agreement,
                                      "scratch_poisoned_before_each_replay": True,
                                      "output_poisoned_before_each_replay": True})
            artifacts.write_json(output / "correctness_progress.json", correctness_cases)
        a, b = prior._inputs(torch, SIZE, "seed-0")
        left.copy_(a)
        right.copy_(b)
        for label in LABELS:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as profiler:
                graphs[label].replay()
                torch.cuda.synchronize()
            path = output / f"{label}_graph_trace.json"
            profiler.export_chrome_trace(str(path))
            trace = json.loads(path.read_text())
            graph_profiles[label] = trace_profile(trace, RANKS[label])
            trace_hashes[label] = artifacts.digest(path)
        artifacts.write_json(output / "topology_preflight.json", graph_profiles)
        functions = {label: graphs[label].replay for label in LABELS}
        for label in LABELS:
            for _ in range(WARMUPS):
                functions[label]()
            torch.cuda.synchronize()
        reader = experiment_pair.raw_clock()
        clock_preflight = experiment_pair.host_clock_preflight(reader=reader)
        options = {"raw_clock": reader}
        controls = {label: graph_measure(torch, functions[label], OPERATIONS, **options)
                    for label in LABELS}
        aa_rounds, rounds, sequence = experiment_pair.measure_paired_rounds(
            graph_measure, torch, functions, OPERATIONS, ROUNDS, options,
            with_aa=True, aa_placement="before")
        timing = {"mode": "cuda_graph_fixed_address", "iterations_per_measurement": OPERATIONS,
                  "host_clock_preflight": clock_preflight,
                  "unranked_timing_controls": controls, "aa_rounds": aa_rounds,
                  "aa_kernel": "A", "aa_iterations_per_measurement": OPERATIONS,
                  "rounds": rounds, "measurement_sequence": sequence,
                  "timing_boundaries": "perf start, RAW start, CUDA start event, one 20-operation graph replay, CUDA end event, host submission timestamp, end-event sync, RAW end, perf end"}
    finally:
        torch.set_float32_matmul_precision(original["fp32_matmul_precision"])
        torch.backends.cuda.matmul.allow_tf32 = original["allow_tf32"]
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = (
            original["allow_fp16_reduced_precision_reduction"])
    props = torch.cuda.get_device_properties(0)
    address_stability = fixed_addresses == {
        "A": left.data_ptr(), "B": right.data_ptr(),
        **{f"{label}_{kind}": tensor.data_ptr()
           for label in LABELS for kind, tensor in zip(
               ("coefficients", "scratch", "output"), buffers[label])}}
    raw = {"schema_version": 1, "mode": "cuda_graph", "session": session,
           "created_at_utc": datetime.now(timezone.utc).isoformat(),
           "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                               cwd=ROOT, text=True).strip(),
           "base_commit": BASE_COMMIT, "source_sha256": artifacts.digest(output / "staged_source.py"),
           "coefficients_sha256": artifacts.digest(output / "coefficients.json"),
           "script_sha256": artifacts.digest(output / "diagnostic_script.py"),
           "previous_eager_manifest_sha256": artifacts.digest(PREVIOUS_BUNDLE / "hash_manifest.json"),
           "environment": {"gpu": props.name, "sm_count": props.multi_processor_count,
                           "compute_capability": f"{props.major}.{props.minor}",
                           "python": sys.version.split()[0], "torch": torch.__version__,
                           "triton": triton.__version__, "cuda_runtime": torch.version.cuda},
           "size": SIZE, "ranks": RANKS, "operations_per_replay": OPERATIONS,
           "warmups_per_graph": WARMUPS, "capture_order": list(capture_order),
           "capture_setup_ms": setup_ms, "first_compile_ms": first_compile_ms,
           "scratch_bytes": {label: RANKS[label] * 32 * 32 * 4 for label in LABELS},
           "coefficient_bytes": {label: RANKS[label] * 27 * 4 for label in LABELS},
           "output_bytes": SIZE * SIZE * 2,
           "capture_semantics": "20 ordered complete staged calls, preallocated fixed-address inputs, coefficients, FP32 scratch, and FP16 outputs; no allocation or synchronization inside capture",
           "precision": {"input": "fp16", "output": "fp16", "preparation": "fp32",
                         "partial_products": "fp32 tf32x3", "accumulation": "fp32",
                         "reference": "torch.matmul(left.float(), right.float()).half()",
                         "active_float32_matmul_precision": "highest", "active_allow_tf32": False,
                         "active_allow_fp16_reduced_precision_reduction": False,
                         "original": original,
                         "restored": bool(torch.get_float32_matmul_precision() == original["fp32_matmul_precision"]
                                          and bool(torch.backends.cuda.matmul.allow_tf32) == original["allow_tf32"]
                                          and bool(torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)
                                          == original["allow_fp16_reduced_precision_reduction"])},
           "tolerance": TOLERANCE,
           "correctness": {"cases": correctness_cases,
                           "fixed_buffer_addresses_unchanged": address_stability,
                           "same_address_inputs_updated_across_cases": True},
           "graph_profile": graph_profiles, "graph_trace_sha256": trace_hashes,
           "compiler": compiler, "benchmark": timing}
    artifacts.write_json(output / "raw.json", raw)
    artifacts.write_json(output / "summary.json", summarize(raw))
    artifacts.manifest(output)
    return verify(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--session", type=int, choices=(1, 2))
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify == (args.session is not None):
        parser.error("choose exactly one of --verify or --session")
    result = verify(args.output) if args.verify else run(args.output, args.session)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
