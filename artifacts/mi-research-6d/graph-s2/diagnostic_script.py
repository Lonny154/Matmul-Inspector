"""Frozen N=96 staged Laderman versus fused Perminov CUDA Graph study."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

from matmul_inspector import experiment_pair
try:
    from scripts import run_mi_research_2c_graph as artifacts
    from scripts import run_mi_research_3b_correctness as diagnostics
    from scripts import run_mi_research_4_feasibility as prior
    from scripts import run_mi_research_6_feasibility as exact
    from scripts import run_mi_research_6_gpu_probe as eager
    from scripts import run_mi_research_6b_graph as previous
    from scripts import run_mi_research_6c as circuit
    from scripts import run_mi_research_6c_gpu_probe as fused_probe
except ModuleNotFoundError:
    import run_mi_research_2c_graph as artifacts
    import run_mi_research_3b_correctness as diagnostics
    import run_mi_research_4_feasibility as prior
    import run_mi_research_6_feasibility as exact
    import run_mi_research_6_gpu_probe as eager
    import run_mi_research_6b_graph as previous
    import run_mi_research_6c as circuit
    import run_mi_research_6c_gpu_probe as fused_probe

ROOT = Path(__file__).resolve().parents[1]
STAGED = previous.SOURCE
FUSED = circuit.PROBE_SOURCE
COEFFICIENTS = previous.COEFFICIENTS
FUSED_SHA256 = "c9b2d61fc6a1b6a658f1198d1a3938fae154efce2029c463e439b7fabbfaf8db"
BASE_COMMIT = "1b3ae45f5fb00f49f1901e3ea3e1fc8b17607c24"
LABELS = ("A", "B")  # A: staged Laderman; B: fused Perminov-55.
SIZE, OPERATIONS, ROUNDS, WARMUPS = 96, 20, 8, 10
CASES, TOLERANCE = prior.CASE_NAMES, {"atol": 0.01, "rtol": 0.01}


def trace_profile(trace: dict, label: str) -> dict:
    events = trace.get("traceEvents", [])
    kernels = sorted((event for event in events if event.get("cat") == "kernel"),
                     key=lambda event: event["ts"])
    names = [event["name"] for event in kernels]
    stride = 24 if label == "A" else 1
    if len(names) != OPERATIONS * stride:
        raise ValueError(f"{label}: expected {OPERATIONS * stride} kernels, saw {len(names)}")
    for index in range(OPERATIONS):
        block = names[index * stride:(index + 1) * stride]
        if label == "A":
            valid = all("_block_product" in name for name in block[:23]) and "_reconstruct" in block[-1]
        else:
            valid = "_fused" in block[0]
            args = kernels[index]["args"]
            valid = valid and args.get("grid") == [2, 2, 1] and args.get("block") == [128, 1, 1]
        if not valid:
            raise ValueError(f"{label}: incomplete or reordered operation {index}")
    submissions = sum(event.get("cat") == "cuda_runtime" and
                      event.get("name") == "cudaGraphLaunch" for event in events)
    if submissions != 1:
        raise ValueError(f"{label}: expected one graph submission, saw {submissions}")
    return {"kernel_events": len(names), "kernels_per_operation": stride,
            "graph_launch_calls": submissions, "complete_operations": OPERATIONS,
            "ordered_kernel_names_sha256": hashlib.sha256(("\n".join(names) + "\n").encode()).hexdigest()}


def summarize(raw: dict) -> dict:
    if (raw.get("schema_version") != 1 or raw.get("mode") != "cuda_graph" or
            raw.get("session") not in (1, 2) or raw.get("size") != SIZE or
            raw.get("operations_per_replay") != OPERATIONS or
            raw.get("warmups_per_graph") != WARMUPS or raw.get("tolerance") != TOLERANCE or
            raw.get("capture_order") != (["A", "B"] if raw["session"] == 1 else ["B", "A"])):
        raise ValueError("MI-Research-6D protocol changed")
    cases = raw.get("correctness", {}).get("cases", [])
    if [row.get("case") for row in cases] != list(CASES):
        raise ValueError("correctness cases changed")
    if raw["correctness"].get("fixed_buffer_addresses_unchanged") is not True:
        raise ValueError("buffer addresses changed")
    for row in cases:
        for label in LABELS:
            if (len(row["replays"][label]) != 2 or
                    any(check["passed"] is not True or check["failing_elements"] or
                        check["candidate_nonfinite_count"] for check in row["replays"][label]) or
                    row["eager_agreement"][label]["passed"] is not True):
                raise ValueError("graph numerical correctness failed")
    for label, stride in (("A", 24), ("B", 1)):
        profile = raw["graph_profile"][label]
        if (profile["kernel_events"] != OPERATIONS * stride or
                profile["kernels_per_operation"] != stride or profile["graph_launch_calls"] != 1):
            raise ValueError("graph topology changed")
    benchmark = raw.get("benchmark")
    if (not isinstance(benchmark, dict) or benchmark.get("mode") != "cuda_graph_fixed_address" or
            benchmark.get("iterations_per_measurement") != OPERATIONS or
            benchmark.get("measurement_sequence") !=
            experiment_pair.expected_diagnostic_sequence(ROUNDS, OPERATIONS, "before") or
            len(benchmark.get("rounds", [])) != ROUNDS or len(benchmark.get("aa_rounds", [])) != ROUNDS):
        raise ValueError("paired protocol changed")
    samples = list(benchmark["unranked_timing_controls"].values())
    for index in range(1, ROUNDS + 1):
        for key, expected in (("rounds", experiment_pair.pair_order(index)),
                              ("aa_rounds", ["X", "Y"] if index % 2 else ["Y", "X"])):
            row = benchmark[key][index - 1]
            if row["round"] != index or row["order"] != expected or set(row["measurements"]) != set(expected):
                raise ValueError("paired round order changed")
            samples.extend(row["measurements"].values())
    for sample in samples:
        if (sample.get("complete_operations_per_block") != OPERATIONS or
                sample.get("host_submission_calls_per_block") != 1 or
                abs(sample["cuda_event_elapsed_ms_per_operation"] * OPERATIONS -
                    sample["cuda_event_elapsed_ms_per_block"]) > 1e-8):
            raise ValueError("graph timing normalization changed")
    decision = experiment_pair.summarize_clock_aware(
        {"passed": True, "checks": {label: cases[0]["replays"][label][0] for label in LABELS}},
        experiment_pair.timing_rows(raw), None, benchmark)
    if decision.get("rounds") != ROUNDS or decision["decision"] in ("FAILED", "REJECTED_INCORRECT"):
        raise ValueError("paired decision replay failed")
    decision["timing_basis"] = "one fixed-address CUDA Graph replay of 20 complete GEMMs per block"
    timing = {}
    for label in LABELS:
        values = [row["measurements"][label]["cuda_event_elapsed_ms_per_operation"] * 1000
                  for row in benchmark["rounds"]]
        timing[label] = {**previous._stats(values), "rounds_us": values}
    paired = [a - b for a, b in zip(timing["A"]["rounds_us"], timing["B"]["rounds_us"])]
    aa = [(row["measurements"]["X"]["cuda_event_elapsed_ms_per_operation"] -
           row["measurements"]["Y"]["cuda_event_elapsed_ms_per_operation"]) * 1000
          for row in benchmark["aa_rounds"]]
    return {"schema_version": 1, "session": raw["session"], "decision": decision,
            "event_us": timing, "paired_A_minus_B_us": paired,
            "paired_median_us": statistics.median(paired),
            "paired_mad_us": statistics.median(abs(x - statistics.median(paired)) for x in paired),
            "aa_differences_us": aa, "aa_max_absolute_gap_us": max(abs(x) for x in aa),
            "b_faster_rounds": sum(x > 0 for x in paired)}


def verify(output: Path) -> dict:
    manifest = json.loads((output / "hash_manifest.json").read_text())
    files = {str(path.relative_to(output)) for path in output.rglob("*")
             if path.is_file() and path.name != "hash_manifest.json"}
    if files != set(manifest) or any(Path(name).is_absolute() or ".." in Path(name).parts or
                                     artifacts.digest(output / name) != value for name, value in manifest.items()):
        raise ValueError("6D manifest mismatch")
    if (output / "failure.json").exists():
        return {"status": "FAILURE_RETAINED", **json.loads((output / "failure.json").read_text())}
    raw = json.loads((output / "raw.json").read_text())
    if (raw["base_commit"] != BASE_COMMIT or
            raw["source_sha256"] != {"A": artifacts.digest(output / "staged_source.py"),
                                      "B": artifacts.digest(output / "fused_source.py")} or
            raw["source_sha256"] != {"A": previous.SOURCE_SHA256, "B": FUSED_SHA256} or
            raw["coefficients_sha256"] != artifacts.digest(output / "coefficients.json") or
            raw["script_sha256"] != artifacts.digest(output / "diagnostic_script.py") or
            raw["prior_manifest_sha256"] != {
                "A": artifacts.digest(ROOT / "artifacts/mi-research-6b/graph-s2/hash_manifest.json"),
                "B": artifacts.digest(fused_probe.OUT / "hash_manifest.json")}):
        raise ValueError("6D source provenance mismatch")
    if raw["correctness"]["cases"] != json.loads((output / "correctness_progress.json").read_text()):
        raise ValueError("correctness progress mismatch")
    if raw["graph_profile"] != json.loads((output / "topology_preflight.json").read_text()):
        raise ValueError("topology preflight mismatch")
    if (raw["capture_setup_ms"] != json.loads((output / "capture_preflight.json").read_text())["setup_ms"]):
        raise ValueError("capture setup mismatch")
    for label in LABELS:
        path = output / f"{label}_graph_trace.json"
        if (raw["graph_trace_sha256"][label] != artifacts.digest(path) or
                raw["graph_profile"][label] != trace_profile(json.loads(path.read_text()), label)):
            raise ValueError("trace topology mismatch")
        for phase, record in raw["compiler"][label].items():
            stem = f"{label}_{phase}"
            if record["ptx_sha256"] != artifacts.digest(output / f"{stem}.ptx"):
                raise ValueError("PTX identity mismatch")
            for kind, digest in record["ir_sha256"].items():
                if digest != artifacts.digest(output / f"{stem}.{kind}"):
                    raise ValueError("IR identity mismatch")
    if summarize(raw) != json.loads((output / "summary.json").read_text()):
        raise ValueError("summary replay mismatch")
    return {"status": "PASS", "session": raw["session"],
            "decision": summarize(raw)["decision"]["decision"]}


def run(output: Path, session: int) -> dict:
    if output.exists():
        raise FileExistsError("6D evidence directory already exists")
    if session not in (1, 2):
        raise ValueError("session must be 1 or 2")
    circuit.verify()
    if (previous.verify(ROOT / "artifacts/mi-research-6b/graph-s2")["status"] != "PASS" or
            fused_probe.verify()["status"] != "PASS"):
        raise ValueError("prior evidence failed verification")
    if (artifacts.digest(STAGED) != previous.SOURCE_SHA256 or
            artifacts.digest(FUSED) != FUSED_SHA256 or
            artifacts.digest(COEFFICIENTS) != previous.COEFFICIENTS_SHA256):
        raise ValueError("frozen source changed")
    output.mkdir(parents=True)
    for path, name in ((STAGED, "staged_source.py"), (FUSED, "fused_source.py"),
                       (COEFFICIENTS, "coefficients.json"), (Path(__file__), "diagnostic_script.py")):
        shutil.copyfile(path, output / name)
    try:
        return _run_gpu(output, session)
    except Exception as error:
        artifacts.write_json(output / "failure.json", {"type": type(error).__name__,
                       "message": str(error), "stage": "graph_study"})
        artifacts.manifest(output)
        raise


def _run_gpu(output: Path, session: int) -> dict:
    import torch
    import triton

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    staged = eager.load_snapshot(output / "staged_source.py")
    fused = eager.load_snapshot(output / "fused_source.py")
    actions = exact.parse_laderman(exact.SOURCE.read_text())
    if json.loads((output / "coefficients.json").read_text())["laderman_23_l_equals_1"] != exact.coefficients(actions):
        raise ValueError("Laderman coefficients changed")
    staged_buffers = staged.buffers(actions)
    fused_output = torch.empty((SIZE, SIZE), device="cuda", dtype=torch.float16)
    left, right = prior._inputs(torch, SIZE, "seed-0")
    addresses = {"left": left.data_ptr(), "right": right.data_ptr(),
                 "A_coefficients": staged_buffers[0].data_ptr(),
                 "A_scratch": staged_buffers[1].data_ptr(),
                 "A_output": staged_buffers[2].data_ptr(),
                 "B_output": fused_output.data_ptr()}
    def current_addresses():
        return {"left": left.data_ptr(), "right": right.data_ptr(),
                "A_coefficients": staged_buffers[0].data_ptr(),
                "A_scratch": staged_buffers[1].data_ptr(),
                "A_output": staged_buffers[2].data_ptr(), "B_output": fused_output.data_ptr()}
    def operation(label, capture=None):
        if label == "A":
            return staged.call(left, right, *staged_buffers, launches=capture)
        compiled = fused.call(left, right, fused_output)
        if capture is not None:
            capture["fused"] = compiled
        return compiled
    outputs = {"A": staged_buffers[2], "B": fused_output}
    original = {"fp32_matmul_precision": torch.get_float32_matmul_precision(),
                "allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                "allow_fp16_reduced_precision_reduction": bool(
                    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)}
    capture_order = ("A", "B") if session == 1 else ("B", "A")
    compiler, first_compile_ms, setup_ms, profiles, trace_hashes, correctness = {}, {}, {}, {}, {}, []
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        for label in LABELS:
            captures = {}
            torch.cuda.synchronize()
            started = time.perf_counter_ns()
            operation(label, captures)
            torch.cuda.synchronize()
            first_compile_ms[label] = (time.perf_counter_ns() - started) / 1e6
            compiler[label] = {}
            for phase, kernel in captures.items():
                details = experiment_pair._compiled_capture(kernel)
                if details is None or not isinstance(details.get("ptx_text"), str):
                    raise ValueError(f"{label} compiler PTX unavailable")
                stem = f"{label}_{phase}"
                (output / f"{stem}.ptx").write_text(details["ptx_text"])
                ir_hashes = {}
                for kind in ("ttir", "ttgir", "llir"):
                    value = kernel.asm.get(kind)
                    if isinstance(value, str):
                        (output / f"{stem}.{kind}").write_text(value)
                        ir_hashes[kind] = artifacts.digest(output / f"{stem}.{kind}")
                compiler[label][phase] = {
                    "kernel_name": details["kernel_name"], "ptx_sha256": artifacts.digest(output / f"{stem}.ptx"),
                    "ir_sha256": ir_hashes,
                    "metadata": {**details["metadata"], "n_spills": getattr(kernel.metadata, "n_spills", None),
                                 "global_scratch_size": getattr(kernel.metadata, "global_scratch_size", None)},
                    "sass_status": "UNAVAILABLE" if not details.get("sass_text") else "AVAILABLE_NOT_RETAINED"}
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
            "session": session, "capture_order": list(capture_order), "operations_per_graph": OPERATIONS,
            "setup_ms": setup_ms, "first_compile_ms": first_compile_ms,
            "fixed_addresses_unchanged": addresses == current_addresses()})
        for case in CASES:
            a, b = prior._inputs(torch, SIZE, case)
            left.copy_(a)
            right.copy_(b)
            reference = torch.matmul(left.float(), right.float()).half()
            checks = {label: [] for label in LABELS}
            graph_outputs = {}
            for replay in range(2):
                for label in LABELS:
                    if label == "A":
                        staged_buffers[1].fill_(float("nan"))
                    outputs[label].fill_(float("nan"))
                    graphs[label].replay()
                    torch.cuda.synchronize()
                    check = diagnostics.summarize_error(torch, outputs[label], reference, a=left, b=right)
                    checks[label].append(check)
                    if not check["passed"] or check["failing_elements"]:
                        raise ValueError(f"graph {label} failed correctness: {case}, replay {replay}")
                    if replay == 1:
                        graph_outputs[label] = outputs[label].clone()
            agreement = {}
            for label in LABELS:
                operation(label)
                torch.cuda.synchronize()
                agreement[label] = experiment_pair.check_output(
                    torch, graph_outputs[label], outputs[label], TOLERANCE)
                agreement[label]["bitwise_equal"] = bool(torch.equal(graph_outputs[label], outputs[label]))
                if not agreement[label]["passed"]:
                    raise ValueError(f"graph/eager disagreement: {label}, {case}")
            correctness.append({"case": case, "replays": checks, "eager_agreement": agreement,
                                "A_scratch_poisoned_before_each_replay": True,
                                "outputs_poisoned_before_each_replay": True})
            artifacts.write_json(output / "correctness_progress.json", correctness)
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
            profiles[label] = trace_profile(json.loads(path.read_text()), label)
            trace_hashes[label] = artifacts.digest(path)
        artifacts.write_json(output / "topology_preflight.json", profiles)
        functions = {label: graphs[label].replay for label in LABELS}
        for label in LABELS:
            for _ in range(WARMUPS):
                functions[label]()
            torch.cuda.synchronize()
        reader = experiment_pair.raw_clock()
        clock = experiment_pair.host_clock_preflight(reader=reader)
        options = {"raw_clock": reader}
        controls = {label: previous.graph_measure(torch, functions[label], OPERATIONS, **options)
                    for label in LABELS}
        aa, rounds, sequence = experiment_pair.measure_paired_rounds(
            previous.graph_measure, torch, functions, OPERATIONS, ROUNDS, options,
            with_aa=True, aa_placement="before")
        timing = {"mode": "cuda_graph_fixed_address", "iterations_per_measurement": OPERATIONS,
                  "host_clock_preflight": clock, "unranked_timing_controls": controls,
                  "aa_rounds": aa, "aa_kernel": "A", "aa_iterations_per_measurement": OPERATIONS,
                  "rounds": rounds, "measurement_sequence": sequence,
                  "timing_boundaries": "perf start, RAW start, CUDA start event, one 20-operation graph replay, CUDA end event, host submission timestamp, end-event sync, RAW end, perf end"}
    finally:
        torch.set_float32_matmul_precision(original["fp32_matmul_precision"])
        torch.backends.cuda.matmul.allow_tf32 = original["allow_tf32"]
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = (
            original["allow_fp16_reduced_precision_reduction"])
    props = torch.cuda.get_device_properties(0)
    raw = {"schema_version": 1, "mode": "cuda_graph", "session": session,
           "created_at_utc": datetime.now(timezone.utc).isoformat(),
           "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
           "base_commit": BASE_COMMIT,
           "source_sha256": {"A": artifacts.digest(output / "staged_source.py"),
                             "B": artifacts.digest(output / "fused_source.py")},
           "coefficients_sha256": artifacts.digest(output / "coefficients.json"),
           "script_sha256": artifacts.digest(output / "diagnostic_script.py"),
           "prior_manifest_sha256": {
               "A": artifacts.digest(ROOT / "artifacts/mi-research-6b/graph-s2/hash_manifest.json"),
               "B": artifacts.digest(fused_probe.OUT / "hash_manifest.json")},
           "environment": {"gpu": props.name, "sm_count": props.multi_processor_count,
                           "compute_capability": f"{props.major}.{props.minor}",
                           "python": sys.version.split()[0], "torch": torch.__version__,
                           "triton": triton.__version__, "cuda_runtime": torch.version.cuda,
                           "driver": subprocess.check_output([
                               "nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip()},
           "size": SIZE, "operations_per_replay": OPERATIONS, "warmups_per_graph": WARMUPS,
           "capture_order": list(capture_order), "capture_setup_ms": setup_ms,
           "first_compile_ms": first_compile_ms, "explicit_product_scratch_bytes": {"A": 94208, "B": 0},
           "precision": {"input": "fp16", "output": "fp16", "intermediate": "fp32",
                         "dot": "tf32x3", "reference": "torch.matmul(left.float(), right.float()).half()",
                         "active_float32_matmul_precision": "highest", "active_allow_tf32": False,
                         "active_allow_fp16_reduced_precision_reduction": False,
                         "original": original,
                         "restored": bool(torch.get_float32_matmul_precision() == original["fp32_matmul_precision"]
                                          and bool(torch.backends.cuda.matmul.allow_tf32) == original["allow_tf32"]
                                          and bool(torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)
                                          == original["allow_fp16_reduced_precision_reduction"])},
           "tolerance": TOLERANCE,
           "correctness": {"cases": correctness, "fixed_buffer_addresses_unchanged": addresses == current_addresses(),
                           "same_address_inputs_updated_across_cases": True},
           "graph_profile": profiles, "graph_trace_sha256": trace_hashes,
           "compiler": compiler, "benchmark": timing}
    artifacts.write_json(output / "raw.json", raw)
    artifacts.write_json(output / "summary.json", summarize(raw))
    artifacts.manifest(output)
    return verify(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
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
