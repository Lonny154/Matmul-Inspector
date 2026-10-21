"""Bounded N=96 Perminov versus conventional Triton and cuBLAS study."""

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

from matmul_inspector import experiment, experiment_pair
try:
    from scripts import run_mi_research_2c_graph as evidence
    from scripts import run_mi_research_3b_correctness as diagnostics
    from scripts import run_mi_research_4_feasibility as cases
    from scripts import run_mi_research_6_gpu_probe as loader
    from scripts import run_mi_research_6c_gpu_probe as fused_probe
    from scripts import run_mi_research_6d_graph as graph_prior
    from scripts import analyze_mi_research_6f as prior_6f
except ModuleNotFoundError:
    import run_mi_research_2c_graph as evidence
    import run_mi_research_3b_correctness as diagnostics
    import run_mi_research_4_feasibility as cases
    import run_mi_research_6_gpu_probe as loader
    import run_mi_research_6c_gpu_probe as fused_probe
    import run_mi_research_6d_graph as graph_prior
    import analyze_mi_research_6f as prior_6f


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/mi-research-7"
PROTOCOL = BASE / "protocol.json"
FUSED = ROOT / "examples/kernels/mi_research_6c_fused_perminov55.py"
CONVENTIONAL = ROOT / "src/matmul_inspector/triton_fc1.py"
BASE_COMMIT = "d095aa9939316414c8122ec5fa09a40839e10c60"
PROTOCOL_SHA256 = "c722f61b851faa31a1983f79057018b3e654bcc092fe35508da4dbc4b3874e98"
FUSED_SHA256 = "c9b2d61fc6a1b6a658f1198d1a3938fae154efce2029c463e439b7fabbfaf8db"
CONVENTIONAL_SHA256 = "a88226c46f335a9768f15b52eb275abf6066bfb57b1f7e012faa937a51768552"
SIZE, EAGER_OPERATIONS, GRAPH_OPERATIONS, ROUNDS, WARMUPS = 96, 20, 20, 8, 10
LABELS = ("P", "T", "C")
PAIRS = {"T": ("P", "T"), "C": ("P", "C")}
TOLERANCE = {"atol": 0.01, "rtol": 0.01}
CONFIG = {"block_m": 32, "block_n": 32, "block_k": 32,
          "num_warps": 4, "num_stages": 3, "group_size_m": 1}


def check_protocol() -> dict:
    if evidence.digest(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError("frozen MI-Research-7 protocol changed")
    protocol = json.loads(PROTOCOL.read_text())
    if (protocol["base_commit"] != BASE_COMMIT or
            protocol["rounds_per_pair"] != ROUNDS or
            protocol["operations_per_eager_block"] != EAGER_OPERATIONS or
            protocol["operations_per_graph_replay"] != GRAPH_OPERATIONS or
            protocol["correctness_cases"] != list(cases.CASE_NAMES) or
            protocol["tolerance"] != TOLERANCE):
        raise ValueError("MI-Research-7 workload changed")
    return protocol


def trace_topology(trace: dict, label: str, mode: str) -> dict:
    events = trace.get("traceEvents", [])
    kernels = sorted((item for item in events if item.get("cat") == "kernel"),
                     key=lambda item: item["ts"])
    expected = 1 if mode == "eager" else GRAPH_OPERATIONS
    name = {"P": "_fused", "T": "fc1_kernel", "C": None}[label]
    if len(kernels) != expected or (name is not None and
                                    any(name not in item["name"] for item in kernels)):
        raise ValueError(f"{label} {mode} physical kernel topology changed: {len(kernels)}")
    if label in ("P", "T"):
        grid = [2, 2, 1] if label == "P" else [9, 1, 1]
        if any(item.get("args", {}).get("grid") != grid or
               item.get("args", {}).get("block") != [128, 1, 1] for item in kernels):
            raise ValueError(f"{label} {mode} grid or thread block changed")
    graph_calls = sum(item.get("cat") == "cuda_runtime" and
                      item.get("name") == "cudaGraphLaunch" for item in events)
    if graph_calls != (1 if mode == "cuda_graph" else 0):
        raise ValueError(f"{label} {mode} graph-launch count changed")
    return {"physical_kernel_events": len(kernels),
            "kernels_per_operation": len(kernels) // expected,
            "cuda_graph_launch_calls": graph_calls,
            "first_kernel_name": kernels[0]["name"],
            "grid": kernels[0].get("args", {}).get("grid"),
            "block": kernels[0].get("args", {}).get("block"),
            "configured_ctas": (4 if label == "P" else 9 if label == "T" else None)}


def summarize(raw: dict) -> dict:
    mode, session = raw.get("mode"), raw.get("session")
    if (raw.get("schema_version") != 1 or mode not in ("eager", "cuda_graph") or
            session not in (1, 2) or raw.get("size") != SIZE or
            raw.get("tolerance") != TOLERANCE or
            raw.get("precision_policy", {}).get("active_allow_fp16_reduced_precision_reduction") is not False or
            raw.get("capture_or_initialization_order") !=
            (["P", "T", "C"] if session == 1 else ["C", "T", "P"]) or
            raw.get("pair_order") != (["T", "C"] if session == 1 else ["C", "T"])):
        raise ValueError("MI-Research-7 session protocol changed")
    checks = raw.get("correctness", {}).get("cases", [])
    if [item.get("case") for item in checks] != list(cases.CASE_NAMES):
        raise ValueError("MI-Research-7 correctness cases changed")
    if not raw["correctness"].get("fixed_addresses_unchanged"):
        raise ValueError("MI-Research-7 buffers changed address")
    for case in checks:
        for label in LABELS:
            row = case["checks"][label]
            if (len(row["eager"]) != 2 or
                    any(not check["passed"] or check["failing_elements"] or
                        check["candidate_nonfinite_count"] for check in row["eager"])):
                raise ValueError(f"{label} eager numerical gate failed")
            if mode == "cuda_graph" and (len(row.get("cuda_graph", [])) != 2 or
                    any(not check["passed"] or check["failing_elements"] or
                        check["candidate_nonfinite_count"] for check in row["cuda_graph"]) or
                    not row["graph_eager_agreement"]["bitwise_equal"]):
                raise ValueError(f"{label} graph numerical gate failed")
    ops = EAGER_OPERATIONS if mode == "eager" else GRAPH_OPERATIONS
    expected_sequence = experiment_pair.expected_diagnostic_sequence(ROUNDS, ops, "before")
    out = {"schema_version": 1, "mode": mode, "session": session,
           "correctness_passed": True, "pairs": {}}
    for other in raw["pair_order"]:
        pair = raw["pairs"][other]
        if (pair["physical_labels"] != ["P", other] or
                pair["measurement_sequence"] != expected_sequence or
                len(pair["rounds"]) != ROUNDS or len(pair["aa_rounds"]) != ROUNDS or
                set(pair["unranked_timing_controls"]) != {"A", "B"} or
                pair["iterations_per_measurement"] != ops or
                pair["aa_iterations_per_measurement"] != ops or
                pair["aa_kernel"] != "A"):
            raise ValueError(f"{other} pair timing protocol changed")
        samples = list(pair["unranked_timing_controls"].values())
        for index in range(1, ROUNDS + 1):
            for phase, slots in (("rounds", ("A", "B")), ("aa_rounds", ("X", "Y"))):
                row = pair[phase][index - 1]
                order = experiment_pair.pair_order(index) if phase == "rounds" else (
                    ["X", "Y"] if index % 2 else ["Y", "X"])
                if row["round"] != index or row["order"] != order or set(row["measurements"]) != set(slots):
                    raise ValueError(f"{other} pair round order changed")
                samples.extend(row["measurements"].values())
        for sample in samples:
            if (sample["complete_operations_per_block"] != ops or
                    sample["host_submission_calls_per_block"] != (ops if mode == "eager" else 1) or
                    not math.isclose(sample["cuda_event_elapsed_ms_per_operation"] * ops,
                                     sample["cuda_event_elapsed_ms_per_block"], abs_tol=1e-8)):
                raise ValueError("MI-Research-7 operation count changed")
        seed0 = checks[0]["checks"]
        benchmark = {**pair, "host_clock_preflight": raw["host_clock_preflight"]}
        decision = experiment_pair.summarize_clock_aware(
            {"passed": True, "checks": {"A": seed0["P"][mode][0],
                                          "B": seed0[other][mode][0]}},
            experiment_pair.timing_rows({"benchmark": benchmark}), None, benchmark)
        if decision["decision"] in ("FAILED", "REJECTED_INCORRECT") or decision.get("rounds") != ROUNDS:
            raise ValueError("MI-Research-7 decision did not replay")
        by_label = {label: [row["measurements"][label]["cuda_event_elapsed_ms_per_operation"] * 1000
                            for row in pair["rounds"]] for label in ("A", "B")}
        gaps = [a - b for a, b in zip(by_label["A"], by_label["B"])]
        out["pairs"][other] = {
            "decision": decision,
            "event_us_per_operation": {"P": by_label["A"], other: by_label["B"]},
            "event_median_us": {"P": statistics.median(by_label["A"]),
                                other: statistics.median(by_label["B"])},
            "paired_P_minus_baseline_us": gaps,
            "paired_median_us": statistics.median(gaps),
            "paired_mad_us": statistics.median(abs(x - statistics.median(gaps)) for x in gaps),
            "perminov_faster_rounds": sum(x < 0 for x in gaps),
            "baseline_faster_rounds": sum(x > 0 for x in gaps),
            "aa_max_absolute_gap_us": decision["aa_control"]["max_absolute_difference_ms"] * 1000}
    return out


def verify(output: Path) -> dict:
    check_protocol()
    manifest = json.loads((output / "hash_manifest.json").read_text())
    files = {str(path.relative_to(output)) for path in output.rglob("*")
             if path.is_file() and path.name != "hash_manifest.json"}
    if files != set(manifest) or any(Path(name).is_absolute() or ".." in Path(name).parts or
                                     evidence.digest(output / name) != value for name, value in manifest.items()):
        raise ValueError("MI-Research-7 manifest mismatch")
    if (output / "failure.json").exists():
        return {"status": "FAILURE_RETAINED", **json.loads((output / "failure.json").read_text())}
    raw = json.loads((output / "raw.json").read_text())
    if (raw["base_commit"] != BASE_COMMIT or
            raw["protocol_sha256"] != PROTOCOL_SHA256 or
            evidence.digest(output / "protocol.json") != PROTOCOL_SHA256 or
            raw["source_sha256"] != {"P": FUSED_SHA256, "T": CONVENTIONAL_SHA256} or
            evidence.digest(output / "perminov_source.py") != FUSED_SHA256 or
            evidence.digest(output / "triton_source.py") != CONVENTIONAL_SHA256 or
            raw["script_sha256"] != evidence.digest(output / "diagnostic_script.py") or
            raw["prior_6f_manifest_sha256"] != evidence.digest(
                ROOT / "artifacts/mi-research-6f/analysis_manifest.json")):
        raise ValueError("MI-Research-7 provenance changed")
    if raw["correctness"]["cases"] != json.loads((output / "correctness_progress.json").read_text()):
        raise ValueError("MI-Research-7 correctness progress mismatch")
    if raw["pairs"] != json.loads((output / "pairs_progress.json").read_text()):
        raise ValueError("MI-Research-7 paired progress mismatch")
    for label in LABELS:
        path = output / f"{label}_trace.json"
        if raw["topology"][label] != trace_topology(
                json.loads(path.read_text()), label, raw["mode"]):
            raise ValueError("MI-Research-7 topology changed")
        if label in ("P", "T"):
            record = raw["compiler"][label]
            if record["ptx_sha256"] != evidence.digest(output / f"{label}.ptx"):
                raise ValueError("MI-Research-7 PTX changed")
            for kind, sha in record["ir_sha256"].items():
                if sha != evidence.digest(output / f"{label}.{kind}"):
                    raise ValueError("MI-Research-7 IR changed")
    if summarize(raw) != json.loads((output / "summary.json").read_text()):
        raise ValueError("MI-Research-7 summary did not replay")
    return {"status": "PASS", "mode": raw["mode"], "session": raw["session"],
            "decisions": {other: value["decision"]["decision"]
                          for other, value in summarize(raw)["pairs"].items()}}


def _capture_trace(torch, function, path: Path, label: str, mode: str) -> dict:
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as profiler:
        function()
        torch.cuda.synchronize()
    profiler.export_chrome_trace(str(path))
    return trace_topology(json.loads(path.read_text()), label, mode)


def _capture_compiler(compiled, output: Path, label: str) -> dict:
    details = experiment_pair._compiled_capture(compiled)
    if details is None or not isinstance(details.get("ptx_text"), str):
        raise RuntimeError(f"{label} compiler PTX unavailable")
    (output / f"{label}.ptx").write_text(details["ptx_text"])
    ir_hashes = {}
    for kind in ("ttir", "ttgir", "llir"):
        text = compiled.asm.get(kind)
        if isinstance(text, str):
            (output / f"{label}.{kind}").write_text(text)
            ir_hashes[kind] = evidence.digest(output / f"{label}.{kind}")
    return {"kernel_name": details["kernel_name"],
            "ptx_sha256": evidence.digest(output / f"{label}.ptx"),
            "ir_sha256": ir_hashes,
            "metadata": {**details["metadata"],
                         "global_scratch_size": getattr(compiled.metadata, "global_scratch_size", None)},
            "sass_status": "UNAVAILABLE" if not details.get("sass_text") else "AVAILABLE_NOT_RETAINED"}


def run(output: Path, mode: str, session: int) -> dict:
    if output.exists():
        raise FileExistsError("MI-Research-7 output already exists")
    if mode not in ("eager", "cuda_graph") or session not in (1, 2):
        raise ValueError("invalid MI-Research-7 session")
    check_protocol()
    if (fused_probe.verify()["status"] != "PASS" or
            prior_6f.verify()["status"] != "PASS" or
            evidence.digest(FUSED) != FUSED_SHA256 or
            evidence.digest(CONVENTIONAL) != CONVENTIONAL_SHA256):
        raise ValueError("historical source or evidence changed")
    output.mkdir(parents=True)
    for source, name in ((PROTOCOL, "protocol.json"), (FUSED, "perminov_source.py"),
                         (CONVENTIONAL, "triton_source.py"),
                         (Path(__file__), "diagnostic_script.py")):
        shutil.copyfile(source, output / name)
    try:
        return _run_gpu(output, mode, session)
    except Exception as error:
        evidence.write_json(output / "failure.json", {
            "type": type(error).__name__, "message": str(error),
            "mode": mode, "session": session})
        evidence.manifest(output)
        raise


def _run_gpu(output: Path, mode: str, session: int) -> dict:
    import torch
    import triton
    from matmul_inspector import triton_fc1

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    fused = loader.load_snapshot(output / "perminov_source.py")
    initial_a, initial_b = cases._inputs(torch, SIZE, "seed-0")
    left = torch.empty_like(initial_a)
    right = torch.empty_like(initial_b)
    left.copy_(initial_a)
    right.copy_(initial_b)
    outputs = {label: torch.empty((SIZE, SIZE), device="cuda", dtype=torch.float16)
               for label in LABELS}
    addresses = {"left": left.data_ptr(), "right": right.data_ptr(),
                 **{label: outputs[label].data_ptr() for label in LABELS}}
    grid = (triton.cdiv(SIZE, CONFIG["block_m"]) * triton.cdiv(SIZE, CONFIG["block_n"]),)

    def perm():
        return fused.call(left, right, outputs["P"])

    def conventional():
        out = outputs["T"]
        return triton_fc1.fc1_kernel[grid](
            left, right, out, SIZE, SIZE, SIZE,
            left.stride(0), left.stride(1), right.stride(0), right.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_M=CONFIG["block_m"], BLOCK_N=CONFIG["block_n"],
            BLOCK_K=CONFIG["block_k"], GROUP_SIZE_M=CONFIG["group_size_m"],
            num_warps=CONFIG["num_warps"], num_stages=CONFIG["num_stages"])

    def cublas():
        return torch.mm(left, right, out=outputs["C"])

    functions = {"P": perm, "T": conventional, "C": cublas}
    order = ["P", "T", "C"] if session == 1 else ["C", "T", "P"]
    pair_order = ["T", "C"] if session == 1 else ["C", "T"]
    original = {"float32_matmul_precision": torch.get_float32_matmul_precision(),
                "allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                "allow_fp16_reduced_precision_reduction": bool(
                    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)}
    compiled, first_call_ms, setup_ms, graphs, correctness, topology, pairs = {}, {}, {}, {}, [], {}, {}
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        for label in order:
            torch.cuda.synchronize()
            start = time.perf_counter_ns()
            result = functions[label]()
            torch.cuda.synchronize()
            first_call_ms[label] = (time.perf_counter_ns() - start) / 1e6
            if label in ("P", "T"):
                compiled[label] = _capture_compiler(result, output, label)
        for label in order:
            for _ in range(WARMUPS):
                functions[label]()
            torch.cuda.synchronize()
        if mode == "cuda_graph":
            for label in order:
                torch.cuda.synchronize()
                start = time.perf_counter_ns()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(GRAPH_OPERATIONS):
                        functions[label]()
                torch.cuda.synchronize()
                graphs[label] = graph
                setup_ms[label] = (time.perf_counter_ns() - start) / 1e6
        for case in cases.CASE_NAMES:
            a, b = cases._inputs(torch, SIZE, case)
            left.copy_(a)
            right.copy_(b)
            reference = torch.matmul(left.float(), right.float()).half()
            checks = {}
            for label in LABELS:
                eager_checks = []
                for _ in range(2):
                    outputs[label].fill_(float("nan"))
                    functions[label]()
                    torch.cuda.synchronize()
                    check = diagnostics.summarize_error(
                        torch, outputs[label], reference, a=left, b=right)
                    eager_checks.append(check)
                checks[label] = {"eager": eager_checks}
                if mode == "cuda_graph":
                    graph_checks = []
                    for _ in range(2):
                        outputs[label].fill_(float("nan"))
                        graphs[label].replay()
                        torch.cuda.synchronize()
                        graph_checks.append(diagnostics.summarize_error(
                            torch, outputs[label], reference, a=left, b=right))
                    saved = outputs[label].clone()
                    functions[label]()
                    torch.cuda.synchronize()
                    checks[label]["cuda_graph"] = graph_checks
                    checks[label]["graph_eager_agreement"] = {
                        **experiment_pair.check_output(torch, saved, outputs[label], TOLERANCE),
                        "bitwise_equal": bool(torch.equal(saved, outputs[label]))}
            correctness.append({"case": case, "checks": checks,
                                "output_poisoned_before_each_check": True})
            evidence.write_json(output / "correctness_progress.json", correctness)
            if any(not check["passed"] or check["failing_elements"] or
                   check["candidate_nonfinite_count"]
                   for label in LABELS for phase in ("eager", "cuda_graph")
                   for check in checks[label].get(phase, [])) or (
                    mode == "cuda_graph" and any(
                        not checks[label]["graph_eager_agreement"]["bitwise_equal"]
                        for label in LABELS)):
                raise ValueError(f"MI-Research-7 numerical gate failed: {case}")
        left.copy_(initial_a)
        right.copy_(initial_b)
        torch.cuda.synchronize()
        profiled = {label: (graphs[label].replay if mode == "cuda_graph" else functions[label])
                    for label in LABELS}
        for label in order:
            topology[label] = _capture_trace(torch, profiled[label],
                                             output / f"{label}_trace.json", label, mode)
        for label in order:
            for _ in range(WARMUPS):
                profiled[label]()
            torch.cuda.synchronize()
        reader = experiment_pair.raw_clock()
        clock = experiment_pair.host_clock_preflight(reader=reader)
        ops = EAGER_OPERATIONS if mode == "eager" else GRAPH_OPERATIONS
        if mode == "cuda_graph":
            measure = graph_prior.previous.graph_measure
        else:
            def measure(torch_module, function, iterations, *, raw_clock):
                sample = experiment.CudaTritonBackend._measure(
                    torch_module, function, iterations, raw_clock=raw_clock)
                return {**sample, "complete_operations_per_block": iterations,
                        "host_submission_calls_per_block": iterations}
        for other in pair_order:
            mapped = {"A": profiled["P"], "B": profiled[other]}
            controls = {label: measure(torch, mapped[label], ops, raw_clock=reader)
                        for label in ("A", "B")}
            aa, rounds, sequence = experiment_pair.measure_paired_rounds(
                measure, torch, mapped, ops, ROUNDS, {"raw_clock": reader},
                with_aa=True, aa_placement="before")
            pairs[other] = {"physical_labels": ["P", other],
                            "unranked_timing_controls": controls,
                            "aa_rounds": aa, "rounds": rounds,
                            "measurement_sequence": sequence,
                            "iterations_per_measurement": ops,
                            "aa_iterations_per_measurement": ops,
                            "aa_kernel": "A",
                            "host_clock_preflight": clock}
            evidence.write_json(output / "pairs_progress.json", pairs)
    finally:
        torch.set_float32_matmul_precision(original["float32_matmul_precision"])
        torch.backends.cuda.matmul.allow_tf32 = original["allow_tf32"]
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = (
            original["allow_fp16_reduced_precision_reduction"])
    props = torch.cuda.get_device_properties(0)
    raw = {"schema_version": 1, "mode": mode, "session": session,
           "created_at_utc": datetime.now(timezone.utc).isoformat(),
           "base_commit": BASE_COMMIT,
           "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                               cwd=ROOT, text=True).strip(),
           "protocol_sha256": evidence.digest(output / "protocol.json"),
           "source_sha256": {"P": evidence.digest(output / "perminov_source.py"),
                             "T": evidence.digest(output / "triton_source.py")},
           "script_sha256": evidence.digest(output / "diagnostic_script.py"),
           "prior_6f_manifest_sha256": evidence.digest(
               ROOT / "artifacts/mi-research-6f/analysis_manifest.json"),
           "size": SIZE, "tolerance": TOLERANCE,
           "capture_or_initialization_order": order, "pair_order": pair_order,
           "environment": {"gpu": props.name, "sm_count": props.multi_processor_count,
                           "compute_capability": f"{props.major}.{props.minor}",
                           "driver": subprocess.check_output(
                               ["nvidia-smi", "--query-gpu=driver_version",
                                "--format=csv,noheader"], text=True).strip(),
                           "python": sys.version.split()[0], "torch": torch.__version__,
                           "triton": triton.__version__, "cuda_runtime": torch.version.cuda},
           "precision_policy": {"reference": "FP32 matmul of exact FP16 inputs, rounded to FP16",
                                "active_float32_matmul_precision": "highest",
                                "active_allow_tf32": False,
                                "active_allow_fp16_reduced_precision_reduction": False,
                                "original": original,
                                "restored": (torch.get_float32_matmul_precision() ==
                                             original["float32_matmul_precision"] and
                                             bool(torch.backends.cuda.matmul.allow_tf32) == original["allow_tf32"] and
                                             bool(torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)
                                             == original["allow_fp16_reduced_precision_reduction"])},
           "config": {"P": "frozen 6C source", "T": CONFIG,
                      "C": "torch.mm out= preallocated FP16 output, reduced FP16 reduction disabled"},
           "workload": {"input_shape": [SIZE, SIZE], "output_shape": [SIZE, SIZE],
                        "dtype": "float16", "strides": [SIZE, 1]},
           "measurement_window": ("20 eager preallocated calls inside one CUDA-event and host-clock block"
                                  if mode == "eager" else
                                  "one fixed-address CUDA Graph replay containing 20 complete operations"),
           "first_call_ms": first_call_ms, "capture_setup_ms": setup_ms,
           "correctness": {"cases": correctness,
                           "fixed_addresses_unchanged": addresses == {
                               "left": left.data_ptr(), "right": right.data_ptr(),
                               **{label: outputs[label].data_ptr() for label in LABELS}}},
           "topology": topology, "compiler": compiled,
           "host_clock_preflight": clock, "pairs": pairs}
    evidence.write_json(output / "raw.json", raw)
    evidence.write_json(output / "summary.json", summarize(raw))
    evidence.manifest(output)
    return verify(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=("eager", "cuda_graph"))
    parser.add_argument("--session", type=int, choices=(1, 2))
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify == (args.mode is not None or args.session is not None):
        parser.error("choose --verify or both --mode and --session")
    if not args.verify and (args.mode is None or args.session is None):
        parser.error("both --mode and --session are required")
    result = verify(args.output) if args.verify else run(args.output, args.mode, args.session)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
