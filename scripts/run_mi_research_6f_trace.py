"""Bounded NVTX-correlated CUDA Graph replay capture for staged Laderman."""

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
    from scripts import run_mi_research_6_feasibility as exact
    from scripts import run_mi_research_6_gpu_probe as eager
    from scripts import run_mi_research_6d_graph as historical
    from scripts import analyze_mi_research_6e as prior_analysis
except ModuleNotFoundError:
    import run_mi_research_2c_graph as artifacts
    import run_mi_research_3b_correctness as diagnostics
    import run_mi_research_4_feasibility as prior
    import run_mi_research_6_feasibility as exact
    import run_mi_research_6_gpu_probe as eager
    import run_mi_research_6d_graph as historical
    import analyze_mi_research_6e as prior_analysis

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "artifacts/mi-research-6f/protocol.json"
BASE_COMMIT = "5c914467985028a38e297cdb0c5d93186d75e35f"
SIZE, OPERATIONS, WARMUPS = 96, 20, 10
SESSIONS = {1: ("graph", 12), 2: ("node", 2)}
CASES = prior.CASE_NAMES
TOLERANCE = {"atol": 0.01, "rtol": 0.01}


def check_protocol(protocol: dict) -> None:
    if (protocol.get("schema_version") != 1 or protocol.get("base_commit") != BASE_COMMIT or
            protocol.get("size") != SIZE or protocol.get("tolerance") != TOLERANCE or
            protocol.get("complete_operations_per_graph") != OPERATIONS or
            protocol.get("eager_warmups") != WARMUPS or
            protocol.get("graph_warmups") != WARMUPS or
            protocol.get("sessions") != [
                {"session": 1, "trace_granularity": "graph", "measured_blocks": 12,
                 "node_level_captures": 0},
                {"session": 2, "trace_granularity": "node", "measured_blocks": 2,
                 "node_level_captures": 2}] or
            protocol.get("historical_unprofiled_normal_upper_us_per_operation") != 65.0 or
            protocol.get("historical_unprofiled_slow_lower_us_per_operation") != 80.0 or
            protocol.get("profiled_relative_anomaly_ratio") != 1.35):
        raise ValueError("frozen 6F protocol changed")


def summarize(raw: dict) -> dict:
    session = raw.get("session")
    if (raw.get("schema_version") != 1 or session not in SESSIONS or
            raw.get("trace_granularity") != SESSIONS[session][0] or
            raw.get("size") != SIZE or raw.get("operations_per_replay") != OPERATIONS or
            raw.get("tolerance") != TOLERANCE or
            raw.get("warmups") != {"eager": WARMUPS, "graph": WARMUPS}):
        raise ValueError("6F session protocol changed")
    cases = raw.get("correctness", {}).get("cases", [])
    if ([row.get("case") for row in cases] != list(CASES) or
            not raw["correctness"].get("fixed_addresses_unchanged")):
        raise ValueError("6F correctness setup changed")
    for case in cases:
        if (len(case["replays"]) != 2 or any(
                not check["passed"] or check["failing_elements"] or
                check["candidate_nonfinite_count"] for check in case["replays"]) or
                not case["eager_agreement"]["bitwise_equal"]):
            raise ValueError("6F graph numerical gate failed")
    samples = raw.get("measurements", [])
    if len(samples) != SESSIONS[session][1]:
        raise ValueError("6F measured block count changed")
    for index, sample in enumerate(samples, 1):
        expected = f"MI6F_S{session}_{SESSIONS[session][0].upper()}_B{index:02d}"
        if (sample["block"] != index or sample["replay_id"] != expected or
                sample["complete_operations_per_block"] != OPERATIONS or
                sample["host_submission_calls_per_block"] != 1 or
                abs(sample["cuda_event_elapsed_ms_per_operation"] * OPERATIONS -
                    sample["cuda_event_elapsed_ms_per_block"]) > 1e-8 or
                sample["outer_end_perf_ns"] < sample["outer_start_perf_ns"]):
            raise ValueError("6F replay ID or measurement normalization changed")
    return {"schema_version": 1, "session": session,
            "trace_granularity": SESSIONS[session][0],
            "correctness_passed": True,
            "measured_blocks": len(samples),
            "replay_ids": [sample["replay_id"] for sample in samples],
            "cuda_event_us_per_operation": [
                sample["cuda_event_elapsed_ms_per_operation"] * 1000 for sample in samples],
            "host_submission_us_per_operation": [
                sample["host_submission_ms_per_block"] * 1000 / OPERATIONS for sample in samples]}


def verify(output: Path) -> dict:
    manifest = json.loads((output / "hash_manifest.json").read_text())
    files = {str(path.relative_to(output)) for path in output.rglob("*")
             if path.is_file() and path.name != "hash_manifest.json"}
    if files != set(manifest) or any(Path(name).is_absolute() or ".." in Path(name).parts or
                                     artifacts.digest(output / name) != value for name, value in manifest.items()):
        raise ValueError("6F session manifest mismatch")
    if (output / "failure.json").exists():
        return {"status": "FAILURE_RETAINED", **json.loads((output / "failure.json").read_text())}
    raw = json.loads((output / "raw.json").read_text())
    if (raw["base_commit"] != BASE_COMMIT or
            raw["protocol_sha256"] != artifacts.digest(PROTOCOL) or
            raw["protocol_sha256"] != artifacts.digest(output / "protocol.json") or
            raw["source_sha256"] != historical.previous.SOURCE_SHA256 or
            raw["source_sha256"] != artifacts.digest(output / "staged_source.py") or
            raw["coefficients_sha256"] != historical.previous.COEFFICIENTS_SHA256 or
            raw["coefficients_sha256"] != artifacts.digest(output / "coefficients.json") or
            raw["script_sha256"] != artifacts.digest(output / "diagnostic_script.py") or
            raw["prior_6e_analysis_manifest_sha256"] != artifacts.digest(
                ROOT / "artifacts/mi-research-6e/analysis_manifest.json")):
        raise ValueError("6F provenance mismatch")
    if raw["correctness"]["cases"] != json.loads((output / "correctness_progress.json").read_text()):
        raise ValueError("6F correctness progress mismatch")
    result = summarize(raw)
    if result != json.loads((output / "summary.json").read_text()):
        raise ValueError("6F summary replay mismatch")
    return {"status": "PASS", "session": raw["session"],
            "trace_granularity": raw["trace_granularity"],
            "measured_blocks": len(raw["measurements"])}


def run(output: Path, session: int) -> dict:
    if output.exists():
        raise FileExistsError("6F session evidence already exists")
    if session not in SESSIONS:
        raise ValueError("session must be 1 or 2")
    check_protocol(json.loads(PROTOCOL.read_text()))
    for number in (1, 2):
        if historical.verify(ROOT / f"artifacts/mi-research-6d/graph-s{number}")["status"] != "PASS":
            raise ValueError("6D evidence failed verification")
    if prior_analysis.run(verify=True)["status"] != "PASS":
        raise ValueError("6E evidence failed verification")
    if artifacts.digest(historical.STAGED) != historical.previous.SOURCE_SHA256:
        raise ValueError("frozen staged source changed")
    output.mkdir(parents=True)
    for path, name in ((PROTOCOL, "protocol.json"), (historical.STAGED, "staged_source.py"),
                       (historical.COEFFICIENTS, "coefficients.json"),
                       (Path(__file__), "diagnostic_script.py")):
        shutil.copyfile(path, output / name)
    try:
        return _run_gpu(output, session)
    except Exception as error:
        artifacts.write_json(output / "failure.json", {
            "type": type(error).__name__, "message": str(error), "stage": "6f_trace"})
        artifacts.manifest(output)
        raise


def _telemetry() -> dict:
    command = ["nvidia-smi", "--query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu",
               "--format=csv,noheader,nounits"]
    try:
        return {"status": "AVAILABLE", "fields": ["clocks.sm_MHz", "clocks.mem_MHz",
                "temperature_C", "power_W", "utilization_percent"],
                "values": subprocess.check_output(command, text=True, timeout=5).strip().split(", ")}
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        return {"status": "UNAVAILABLE", "reason": type(error).__name__}


def _run_gpu(output: Path, session: int) -> dict:
    import torch
    import triton

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    source = eager.load_snapshot(output / "staged_source.py")
    actions = exact.parse_laderman(exact.SOURCE.read_text())
    if (json.loads((output / "coefficients.json").read_text())["laderman_23_l_equals_1"] !=
            exact.coefficients(actions)):
        raise ValueError("frozen Laderman factors changed")
    buffers = source.buffers(actions)
    left, right = prior._inputs(torch, SIZE, "seed-0")
    def addresses():
        return {"left": left.data_ptr(), "right": right.data_ptr(),
                "coefficients": buffers[0].data_ptr(), "scratch": buffers[1].data_ptr(),
                "output": buffers[2].data_ptr()}
    fixed_addresses = addresses()
    def operation():
        return source.call(left, right, *buffers)
    original = {"fp32_matmul_precision": torch.get_float32_matmul_precision(),
                "allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                "allow_fp16_reduced_precision_reduction": bool(
                    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)}
    correctness, measurements = [], []
    telemetry_before = _telemetry()
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        operation()
        torch.cuda.synchronize()
        first_call_ms = (time.perf_counter_ns() - start) / 1e6
        for _ in range(WARMUPS):
            operation()
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(OPERATIONS):
                operation()
        torch.cuda.synchronize()
        capture_setup_ms = (time.perf_counter_ns() - start) / 1e6
        for case in CASES:
            a, b = prior._inputs(torch, SIZE, case)
            left.copy_(a)
            right.copy_(b)
            reference = torch.matmul(left.float(), right.float()).half()
            checks = []
            for replay in range(2):
                buffers[1].fill_(float("nan"))
                buffers[2].fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize()
                check = diagnostics.summarize_error(torch, buffers[2], reference, a=left, b=right)
                checks.append(check)
                if not check["passed"] or check["failing_elements"]:
                    raise ValueError(f"6F numerical gate failed: {case}, replay {replay}")
            saved = buffers[2].clone()
            operation()
            torch.cuda.synchronize()
            agreement = experiment_pair.check_output(torch, saved, buffers[2], TOLERANCE)
            agreement["bitwise_equal"] = bool(torch.equal(saved, buffers[2]))
            if not agreement["bitwise_equal"]:
                raise ValueError(f"6F graph/eager disagreement: {case}")
            correctness.append({"case": case, "replays": checks, "eager_agreement": agreement,
                                "scratch_and_output_poisoned_before_each_replay": True})
            artifacts.write_json(output / "correctness_progress.json", correctness)
        a, b = prior._inputs(torch, SIZE, "seed-0")
        left.copy_(a)
        right.copy_(b)
        for _ in range(WARMUPS):
            graph.replay()
        torch.cuda.synchronize()
        reader = experiment_pair.raw_clock()
        clock = experiment_pair.host_clock_preflight(reader=reader)
        # Nsight 2026.1 did not start collection from the NVTX range in
        # session 1. The CUDA profiler API smoke test did. Keep nested NVTX
        # replay IDs for correlation inside this single bounded capture.
        if session == 2:
            torch.cuda.cudart().cudaProfilerStart()
        torch.cuda.nvtx.range_push("MI6F_TRACE")
        try:
            for index in range(1, SESSIONS[session][1] + 1):
                replay_id = f"MI6F_S{session}_{SESSIONS[session][0].upper()}_B{index:02d}"
                torch.cuda.nvtx.range_push(replay_id)
                try:
                    outer_start = time.perf_counter_ns()
                    sample = historical.previous.graph_measure(
                        torch, graph.replay, OPERATIONS, raw_clock=reader)
                    outer_end = time.perf_counter_ns()
                finally:
                    torch.cuda.nvtx.range_pop()
                measurements.append({**sample, "block": index, "replay_id": replay_id,
                                     "outer_start_perf_ns": outer_start,
                                     "outer_end_perf_ns": outer_end})
                artifacts.write_json(output / "measurement_progress.json", measurements)
        finally:
            torch.cuda.nvtx.range_pop()
            if session == 2:
                torch.cuda.cudart().cudaProfilerStop()
        telemetry_after = _telemetry()
    finally:
        torch.set_float32_matmul_precision(original["fp32_matmul_precision"])
        torch.backends.cuda.matmul.allow_tf32 = original["allow_tf32"]
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = (
            original["allow_fp16_reduced_precision_reduction"])
    props = torch.cuda.get_device_properties(0)
    raw = {"schema_version": 1, "session": session,
           "trace_granularity": SESSIONS[session][0],
           "created_at_utc": datetime.now(timezone.utc).isoformat(),
           "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
           "base_commit": BASE_COMMIT,
           "protocol_sha256": artifacts.digest(output / "protocol.json"),
           "source_sha256": artifacts.digest(output / "staged_source.py"),
           "coefficients_sha256": artifacts.digest(output / "coefficients.json"),
           "script_sha256": artifacts.digest(output / "diagnostic_script.py"),
           "prior_6e_analysis_manifest_sha256": artifacts.digest(
               ROOT / "artifacts/mi-research-6e/analysis_manifest.json"),
           "environment": {"gpu": props.name, "sm_count": props.multi_processor_count,
                           "compute_capability": f"{props.major}.{props.minor}",
                           "python": sys.version.split()[0], "torch": torch.__version__,
                           "triton": triton.__version__, "cuda_runtime": torch.version.cuda,
                           "driver": subprocess.check_output([
                               "nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip()},
           "size": SIZE, "operations_per_replay": OPERATIONS,
           "warmups": {"eager": WARMUPS, "graph": WARMUPS},
           "first_call_ms": first_call_ms, "capture_setup_ms": capture_setup_ms,
           "precision": {"reference": "torch.matmul(left.float(), right.float()).half()",
                         "input": "fp16", "output": "fp16", "preparation": "fp32", "dot": "tf32x3",
                         "original": original,
                         "restored": bool(torch.get_float32_matmul_precision() == original["fp32_matmul_precision"]
                                          and bool(torch.backends.cuda.matmul.allow_tf32) == original["allow_tf32"]
                                          and bool(torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)
                                          == original["allow_fp16_reduced_precision_reduction"])},
           "tolerance": TOLERANCE,
           "correctness": {"cases": correctness, "fixed_addresses_unchanged": fixed_addresses == addresses()},
           "host_clock_preflight": clock,
           "telemetry": {"before": telemetry_before, "after": telemetry_after,
                         "scope": "process-level snapshots; not per-block or causal telemetry"},
           "measurement_window": "Matmul Inspector CUDA start/end events around one graph replay of 20 complete staged GEMMs; host submission ends before end-event sync; RAW and perf end after sync; outer perf and NVTX range bracket whole measurement",
           "capture_trigger": "cudaProfilerApi" if session == 2 else "nvtx",
           "measurements": measurements}
    artifacts.write_json(output / "raw.json", raw)
    artifacts.write_json(output / "summary.json", summarize(raw))
    artifacts.manifest(output)
    return verify(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--session", type=int, choices=(1, 2))
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify == (args.session is not None):
        parser.error("choose exactly one of --verify or --session")
    result = verify(args.output) if args.verify else run(args.output, args.session)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
