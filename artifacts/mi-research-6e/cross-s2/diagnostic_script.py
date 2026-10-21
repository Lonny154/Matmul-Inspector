"""Bounded independent-graph controls for the MI-Research-6D timing result."""

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
    from scripts import run_mi_research_2c_graph as artifacts
    from scripts import run_mi_research_3b_correctness as diagnostics
    from scripts import run_mi_research_4_feasibility as prior
    from scripts import run_mi_research_6_feasibility as exact
    from scripts import run_mi_research_6_gpu_probe as eager
    from scripts import run_mi_research_6d_graph as historical
except ModuleNotFoundError:
    import run_mi_research_2c_graph as artifacts
    import run_mi_research_3b_correctness as diagnostics
    import run_mi_research_4_feasibility as prior
    import run_mi_research_6_feasibility as exact
    import run_mi_research_6_gpu_probe as eager
    import run_mi_research_6d_graph as historical

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "artifacts/mi-research-6e/protocol.json"
BASE_COMMIT = "d569e063859b09096b709eb9a6438c9d7bdb7b00"
CONDITIONS = {"staged_null": ("staged", "staged"),
              "fused_null": ("fused", "fused"), "cross": ("staged", "fused")}
SIZE, OPERATIONS, ROUNDS, WARMUPS = 96, 20, 8, 10
CASES, TOLERANCE = prior.CASE_NAMES, {"atol": 0.01, "rtol": 0.01}


def check_protocol(protocol: dict) -> None:
    if (protocol.get("schema_version") != 1 or protocol.get("base_commit") != BASE_COMMIT or
            protocol.get("conditions") != {k: list(v) for k, v in CONDITIONS.items()} or
            protocol.get("size") != SIZE or protocol.get("operations_per_replay") != OPERATIONS or
            protocol.get("warmups_per_instance") != WARMUPS or
            protocol.get("rounds_per_session") != ROUNDS or
            protocol.get("sessions_per_condition") != 2):
        raise ValueError("frozen 6E protocol changed")


def _stats(values: list[float]) -> dict:
    center = statistics.median(values)
    return {"median_us": center, "min_us": min(values), "max_us": max(values),
            "mad_us": statistics.median(abs(x - center) for x in values),
            "cv": statistics.stdev(values) / statistics.fmean(values)}


def summarize(raw: dict) -> dict:
    condition = raw.get("condition")
    session = raw.get("session")
    if (raw.get("schema_version") != 1 or condition not in CONDITIONS or session not in (1, 2) or
            raw.get("algorithms") != dict(zip(("X", "Y"), CONDITIONS[condition])) or
            raw.get("capture_order") != (["X", "Y"] if session == 1 else ["Y", "X"]) or
            raw.get("size") != SIZE or raw.get("operations_per_replay") != OPERATIONS or
            raw.get("warmups_per_instance") != WARMUPS or raw.get("tolerance") != TOLERANCE):
        raise ValueError("6E session protocol changed")
    cases = raw.get("correctness", {}).get("cases", [])
    if ([row.get("case") for row in cases] != list(CASES) or
            raw["correctness"].get("fixed_addresses_unchanged") is not True or
            raw["correctness"].get("distinct_input_and_output_addresses") is not True):
        raise ValueError("6E correctness setup changed")
    for case in cases:
        for slot in ("X", "Y"):
            checks = case["replays"][slot]
            if (len(checks) != 2 or any(not check["passed"] or check["failing_elements"] or
                                        check["candidate_nonfinite_count"] for check in checks) or
                    not case["eager_agreement"][slot]["bitwise_equal"]):
                raise ValueError("6E graph correctness failed")
        if condition != "cross" and not case["cross_instance_bitwise_equal"]:
            raise ValueError("equivalent graph instances disagreed")
    for slot, algorithm in zip(("X", "Y"), CONDITIONS[condition]):
        profile = raw["graph_profile"][slot]
        if (profile["complete_operations"] != OPERATIONS or
                profile["kernels_per_operation"] != (24 if algorithm == "staged" else 1) or
                profile["graph_launch_calls"] != 1):
            raise ValueError("6E graph topology failed")
    benchmark = raw["benchmark"]
    if (len(benchmark["rounds"]) != ROUNDS or benchmark["mode"] != "cuda_graph_fixed_address" or
            benchmark["iterations_per_measurement"] != OPERATIONS or
            len(benchmark["unranked_timing_controls"]) != 2 or
            (condition == "cross") != (len(benchmark["same_instance_aa_rounds"]) == ROUNDS)):
        raise ValueError("6E timing protocol changed")
    samples = list(benchmark["unranked_timing_controls"].values())
    for index, row in enumerate(benchmark["rounds"], 1):
        expected = ["X", "Y"] if index % 2 else ["Y", "X"]
        if (row["round"] != index or row["order"] != expected or
                set(row["measurements"]) != set(expected)):
            raise ValueError("6E paired order changed")
        samples.extend(row["measurements"].values())
    for index, row in enumerate(benchmark["same_instance_aa_rounds"], 1):
        expected = ["A1", "A2"] if index % 2 else ["A2", "A1"]
        if row["round"] != index or row["order"] != expected:
            raise ValueError("6E same-instance order changed")
        samples.extend(row["measurements"].values())
    if any(sample["complete_operations_per_block"] != OPERATIONS or
           sample["host_submission_calls_per_block"] != 1 or
           abs(sample["cuda_event_elapsed_ms_per_operation"] * OPERATIONS -
               sample["cuda_event_elapsed_ms_per_block"]) > 1e-8 or
           sample["outer_end_perf_ns"] < sample["outer_start_perf_ns"]
           for sample in samples):
        raise ValueError("6E timing normalization failed")
    values = {slot: [row["measurements"][slot]["cuda_event_elapsed_ms_per_operation"] * 1000
                     for row in benchmark["rounds"]] for slot in ("X", "Y")}
    gaps = [x - y for x, y in zip(values["X"], values["Y"])]
    result = {"schema_version": 1, "condition": condition, "session": session,
              "event_us": {slot: {**_stats(values[slot]), "rounds_us": values[slot]}
                           for slot in ("X", "Y")},
              "paired_X_minus_Y_us": gaps, "paired_median_us": statistics.median(gaps),
              "paired_abs_median_us": statistics.median(map(abs, gaps)),
              "paired_abs_max_us": max(map(abs, gaps)),
              "positive_rounds": sum(value > 0 for value in gaps),
              "negative_rounds": sum(value < 0 for value in gaps)}
    if condition == "cross":
        # The unchanged production function sees the original 6D schema and
        # the same-instance staged A/A control. This is a new 6E result only.
        rounds = [{**row, "measurements": {("A" if slot == "X" else "B"): sample
                                           for slot, sample in row["measurements"].items()},
                   "order": ["A" if slot == "X" else "B" for slot in row["order"]]}
                  for row in benchmark["rounds"]]
        aa = [{"round": row["round"], "order": ["X" if slot == "A1" else "Y" for slot in row["order"]],
               "measurements": {("X" if slot == "A1" else "Y"): sample
                                for slot, sample in row["measurements"].items()}}
              for row in benchmark["same_instance_aa_rounds"]]
        decision_input = {"benchmark": {"iterations_per_measurement": OPERATIONS,
            "unranked_timing_controls": {
            "A": benchmark["unranked_timing_controls"]["X"],
            "B": benchmark["unranked_timing_controls"]["Y"]},
            "host_clock_preflight": benchmark["host_clock_preflight"],
            "aa_rounds": aa, "aa_kernel": "A", "aa_iterations_per_measurement": OPERATIONS,
            "rounds": rounds}}
        result["existing_rule"] = experiment_pair.summarize_clock_aware(
            {"passed": True, "checks": {"A": cases[0]["replays"]["X"][0],
                                         "B": cases[0]["replays"]["Y"][0]}},
            experiment_pair.timing_rows(decision_input), None, decision_input["benchmark"])
        if result["existing_rule"]["decision"] in ("FAILED", "REJECTED_INCORRECT"):
            raise ValueError("existing decision could not replay")
        result["same_instance_aa_us"] = [
            (row["measurements"]["A1"]["cuda_event_elapsed_ms_per_operation"] -
             row["measurements"]["A2"]["cuda_event_elapsed_ms_per_operation"]) * 1000
            for row in benchmark["same_instance_aa_rounds"]]
    return result


def verify(output: Path) -> dict:
    manifest = json.loads((output / "hash_manifest.json").read_text())
    files = {str(path.relative_to(output)) for path in output.rglob("*")
             if path.is_file() and path.name != "hash_manifest.json"}
    if files != set(manifest) or any(Path(name).is_absolute() or ".." in Path(name).parts or
                                     artifacts.digest(output / name) != value for name, value in manifest.items()):
        raise ValueError("6E manifest mismatch")
    if (output / "failure.json").exists() and not (output / "recovery.json").exists():
        return {"status": "FAILURE_RETAINED", **json.loads((output / "failure.json").read_text())}
    if (output / "recovery.json").exists():
        recovery = json.loads((output / "recovery.json").read_text())
        if (not (output / "failure.json").exists() or recovery.get("reason") !=
                "summary adapter omitted iterations_per_measurement after complete GPU collection" or
                recovery.get("repair_script_sha256") != artifacts.digest(output / "analysis_repair_script.py") or
                json.loads((output / "failure.json").read_text()).get("type") != "KeyError"):
            raise ValueError("6E recovery provenance mismatch")
    raw = json.loads((output / "raw.json").read_text())
    if (raw["base_commit"] != BASE_COMMIT or
            raw["protocol_sha256"] != artifacts.digest(output / "protocol.json") or
            raw["protocol_sha256"] != artifacts.digest(PROTOCOL) or
            raw["script_sha256"] != artifacts.digest(output / "diagnostic_script.py") or
            raw["source_sha256"] != {
                "staged": artifacts.digest(output / "staged_source.py"),
                "fused": artifacts.digest(output / "fused_source.py")} or
            raw["source_sha256"] != {"staged": historical.previous.SOURCE_SHA256,
                                      "fused": historical.FUSED_SHA256} or
            raw["coefficients_sha256"] != artifacts.digest(output / "coefficients.json") or
            raw["coefficients_sha256"] != historical.previous.COEFFICIENTS_SHA256 or
            raw["prior_6d_manifests"] != {
                str(i): artifacts.digest(ROOT / f"artifacts/mi-research-6d/graph-s{i}/hash_manifest.json")
                for i in (1, 2)}):
        raise ValueError("6E provenance mismatch")
    if raw["correctness"]["cases"] != json.loads((output / "correctness_progress.json").read_text()):
        raise ValueError("correctness preflight mismatch")
    if raw["graph_profile"] != json.loads((output / "topology_preflight.json").read_text()):
        raise ValueError("topology preflight mismatch")
    for slot, algorithm in raw["algorithms"].items():
        path = output / f"{slot}_graph_trace.json"
        if (raw["graph_trace_sha256"][slot] != artifacts.digest(path) or
                raw["graph_profile"][slot] != historical.trace_profile(
                    json.loads(path.read_text()), "A" if algorithm == "staged" else "B")):
            raise ValueError("6E profiler trace mismatch")
    result = summarize(raw)
    if result != json.loads((output / "summary.json").read_text()):
        raise ValueError("6E offline summary mismatch")
    return {"status": "PASS", "condition": raw["condition"], "session": raw["session"],
            "existing_rule": result.get("existing_rule", {}).get("decision")}


def run(output: Path, condition: str, session: int) -> dict:
    if output.exists():
        raise FileExistsError("6E output directory already exists")
    if condition not in CONDITIONS or session not in (1, 2):
        raise ValueError("invalid bounded 6E condition or session")
    check_protocol(json.loads(PROTOCOL.read_text()))
    for number in (1, 2):
        if historical.verify(ROOT / f"artifacts/mi-research-6d/graph-s{number}")["status"] != "PASS":
            raise ValueError("6D evidence failed verification")
    output.mkdir(parents=True)
    for path, name in ((PROTOCOL, "protocol.json"), (historical.STAGED, "staged_source.py"),
                       (historical.FUSED, "fused_source.py"),
                       (historical.COEFFICIENTS, "coefficients.json"),
                       (Path(__file__), "diagnostic_script.py")):
        shutil.copyfile(path, output / name)
    try:
        return _run_gpu(output, condition, session)
    except Exception as error:
        artifacts.write_json(output / "failure.json", {
            "stage": "6e_control", "type": type(error).__name__, "message": str(error)})
        artifacts.manifest(output)
        raise


def recover_completed_summary(output: Path) -> dict:
    """Preserve the failed summary attempt and replay complete saved measurements."""
    existing = verify(output)
    if (existing.get("status") != "FAILURE_RETAINED" or
            existing.get("type") != "KeyError" or
            (output / "summary.json").exists() or (output / "recovery.json").exists()):
        raise ValueError("bundle is not the one complete summary-adapter failure")
    raw = json.loads((output / "raw.json").read_text())
    if raw.get("condition") != "cross" or raw.get("session") != 1:
        raise ValueError("unexpected recovered session")
    previous_manifest = artifacts.digest(output / "hash_manifest.json")
    repaired = summarize(raw)
    shutil.copyfile(Path(__file__), output / "analysis_repair_script.py")
    artifacts.write_json(output / "summary.json", repaired)
    artifacts.write_json(output / "recovery.json", {
        "reason": "summary adapter omitted iterations_per_measurement after complete GPU collection",
        "original_manifest_sha256": previous_manifest,
        "repair_script_sha256": artifacts.digest(output / "analysis_repair_script.py"),
        "measurement_repeated": False})
    artifacts.manifest(output)
    return verify(output)


def _run_gpu(output: Path, condition: str, session: int) -> dict:
    import torch
    import triton

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    staged = eager.load_snapshot(output / "staged_source.py")
    fused = eager.load_snapshot(output / "fused_source.py")
    algorithms = dict(zip(("X", "Y"), CONDITIONS[condition]))
    slots = {}
    actions = exact.parse_laderman(exact.SOURCE.read_text())
    if json.loads((output / "coefficients.json").read_text())["laderman_23_l_equals_1"] != exact.coefficients(actions):
        raise ValueError("Laderman coefficient identity changed")
    seed_a, seed_b = prior._inputs(torch, SIZE, "seed-0")
    for slot, algorithm in algorithms.items():
        left, right = seed_a.clone(), seed_b.clone()
        if algorithm == "staged":
            buffers = staged.buffers(actions)
            slots[slot] = {"left": left, "right": right, "output": buffers[2],
                           "scratch": buffers[1], "coefficients": buffers[0]}
        else:
            slots[slot] = {"left": left, "right": right,
                           "output": torch.empty((SIZE, SIZE), device="cuda", dtype=torch.float16)}
    def addresses():
        return {slot: {key: value.data_ptr() for key, value in state.items()}
                for slot, state in slots.items()}
    fixed_addresses = addresses()
    if any(fixed_addresses["X"][key] == fixed_addresses["Y"][key]
           for key in ("left", "right", "output")):
        raise ValueError("instances share an input or output address")
    def operation(slot):
        state = slots[slot]
        if algorithms[slot] == "staged":
            return staged.call(state["left"], state["right"], state["coefficients"],
                               state["scratch"], state["output"])
        return fused.call(state["left"], state["right"], state["output"])
    original = {"fp32_matmul_precision": torch.get_float32_matmul_precision(),
                "allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                "allow_fp16_reduced_precision_reduction": bool(
                    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)}
    capture_order = ["X", "Y"] if session == 1 else ["Y", "X"]
    compile_ms, setup_ms, correctness, profiles, trace_hashes = {}, {}, [], {}, {}
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        for slot in ("X", "Y"):
            torch.cuda.synchronize()
            start = time.perf_counter_ns()
            operation(slot)
            torch.cuda.synchronize()
            compile_ms[slot] = (time.perf_counter_ns() - start) / 1e6
            for _ in range(WARMUPS):
                operation(slot)
            torch.cuda.synchronize()
        graphs = {}
        for slot in capture_order:
            torch.cuda.synchronize()
            start = time.perf_counter_ns()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(OPERATIONS):
                    operation(slot)
            torch.cuda.synchronize()
            setup_ms[slot] = (time.perf_counter_ns() - start) / 1e6
            graphs[slot] = graph
        artifacts.write_json(output / "capture_preflight.json", {
            "capture_order": capture_order, "first_call_ms": compile_ms,
            "graph_setup_ms": setup_ms, "fixed_addresses_unchanged": fixed_addresses == addresses()})
        for case in CASES:
            a, b = prior._inputs(torch, SIZE, case)
            for state in slots.values():
                state["left"].copy_(a)
                state["right"].copy_(b)
            reference = torch.matmul(a.float(), b.float()).half()
            checks, agreement, graph_outputs = {}, {}, {}
            for slot in ("X", "Y"):
                checks[slot] = []
                for replay in range(2):
                    if "scratch" in slots[slot]:
                        slots[slot]["scratch"].fill_(float("nan"))
                    slots[slot]["output"].fill_(float("nan"))
                    graphs[slot].replay()
                    torch.cuda.synchronize()
                    check = diagnostics.summarize_error(
                        torch, slots[slot]["output"], reference, a=a, b=b)
                    checks[slot].append(check)
                    if not check["passed"] or check["failing_elements"]:
                        raise ValueError(f"graph correctness failed: {condition} {slot} {case} {replay}")
                graph_outputs[slot] = slots[slot]["output"].clone()
                operation(slot)
                torch.cuda.synchronize()
                agreement[slot] = experiment_pair.check_output(
                    torch, graph_outputs[slot], slots[slot]["output"], TOLERANCE)
                agreement[slot]["bitwise_equal"] = bool(torch.equal(
                    graph_outputs[slot], slots[slot]["output"]))
                if not agreement[slot]["bitwise_equal"]:
                    raise ValueError(f"graph/eager disagreement: {condition} {slot} {case}")
            identical = bool(torch.equal(graph_outputs["X"], graph_outputs["Y"]))
            if condition != "cross" and not identical:
                raise ValueError(f"equivalent graph instances disagree: {condition} {case}")
            correctness.append({"case": case, "replays": checks, "eager_agreement": agreement,
                                "cross_instance_bitwise_equal": identical,
                                "output_poisoned_before_each_replay": True,
                                "staged_scratch_poisoned_before_each_replay": True})
            artifacts.write_json(output / "correctness_progress.json", correctness)
        for slot, state in slots.items():
            state["left"].copy_(seed_a)
            state["right"].copy_(seed_b)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as profiler:
                graphs[slot].replay()
                torch.cuda.synchronize()
            path = output / f"{slot}_graph_trace.json"
            profiler.export_chrome_trace(str(path))
            profiles[slot] = historical.trace_profile(
                json.loads(path.read_text()), "A" if algorithms[slot] == "staged" else "B")
            trace_hashes[slot] = artifacts.digest(path)
        artifacts.write_json(output / "topology_preflight.json", profiles)
        functions = {slot: graphs[slot].replay for slot in ("X", "Y")}
        for slot in ("X", "Y"):
            for _ in range(WARMUPS):
                functions[slot]()
            torch.cuda.synchronize()
        reader = experiment_pair.raw_clock()
        clock = experiment_pair.host_clock_preflight(reader=reader)
        def measure(slot, round_number, phase, position):
            outer_start = time.perf_counter_ns()
            sample = historical.previous.graph_measure(
                torch, functions[slot], OPERATIONS, raw_clock=reader)
            outer_end = time.perf_counter_ns()
            return {**sample, "round": round_number, "phase": phase, "position": position,
                    "outer_start_perf_ns": outer_start, "outer_end_perf_ns": outer_end}
        controls = {slot: measure(slot, 0, "unranked", index)
                    for index, slot in enumerate(("X", "Y"), 1)}
        rounds, same_aa, sequence = [], [], []
        block_position = 0
        for index in range(1, ROUNDS + 1):
            if condition == "cross":
                order = ["A1", "A2"] if index % 2 else ["A2", "A1"]
                measurements = {}
                for label in order:
                    block_position += 1
                    measurements[label] = measure("X", index, "same_instance_aa", block_position)
                same_aa.append({"round": index, "order": order, "measurements": measurements})
                sequence.append({"round": index, "phase": "same_instance_aa", "order": order})
            order = ["X", "Y"] if index % 2 else ["Y", "X"]
            measurements = {}
            for slot in order:
                block_position += 1
                measurements[slot] = measure(slot, index, "pair", block_position)
            rounds.append({"round": index, "order": order, "measurements": measurements})
            sequence.append({"round": index, "phase": "pair", "order": order})
        benchmark = {"mode": "cuda_graph_fixed_address", "iterations_per_measurement": OPERATIONS,
                     "host_clock_preflight": clock, "unranked_timing_controls": controls,
                     "same_instance_aa_rounds": same_aa, "rounds": rounds, "measurement_sequence": sequence,
                     "timing_boundaries": "Matmul Inspector CUDA-event, host submission, synchronized perf_counter and optional RAW intervals; one graph replay of 20 complete operations per block"}
    finally:
        torch.set_float32_matmul_precision(original["fp32_matmul_precision"])
        torch.backends.cuda.matmul.allow_tf32 = original["allow_tf32"]
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = (
            original["allow_fp16_reduced_precision_reduction"])
    props = torch.cuda.get_device_properties(0)
    raw = {"schema_version": 1, "condition": condition, "session": session,
           "created_at_utc": datetime.now(timezone.utc).isoformat(),
           "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
           "base_commit": BASE_COMMIT, "protocol_sha256": artifacts.digest(output / "protocol.json"),
           "script_sha256": artifacts.digest(output / "diagnostic_script.py"),
           "source_sha256": {"staged": artifacts.digest(output / "staged_source.py"),
                             "fused": artifacts.digest(output / "fused_source.py")},
           "coefficients_sha256": artifacts.digest(output / "coefficients.json"),
           "prior_6d_manifests": {str(i): artifacts.digest(
               ROOT / f"artifacts/mi-research-6d/graph-s{i}/hash_manifest.json") for i in (1, 2)},
           "environment": {"gpu": props.name, "sm_count": props.multi_processor_count,
                           "compute_capability": f"{props.major}.{props.minor}",
                           "driver": subprocess.check_output([
                               "nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip(),
                           "python": sys.version.split()[0], "torch": torch.__version__,
                           "triton": triton.__version__, "cuda_runtime": torch.version.cuda},
           "algorithms": algorithms, "size": SIZE, "operations_per_replay": OPERATIONS,
           "warmups_per_instance": WARMUPS, "capture_order": capture_order,
           "first_call_ms": compile_ms, "graph_setup_ms": setup_ms,
           "precision": {"reference": "torch.matmul(left.float(), right.float()).half()",
                         "input": "fp16", "output": "fp16", "dot": "tf32x3",
                         "original": original,
                         "restored": bool(torch.get_float32_matmul_precision() == original["fp32_matmul_precision"]
                                          and bool(torch.backends.cuda.matmul.allow_tf32) == original["allow_tf32"]
                                          and bool(torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)
                                          == original["allow_fp16_reduced_precision_reduction"])},
           "tolerance": TOLERANCE,
           "correctness": {"cases": correctness, "fixed_addresses_unchanged": fixed_addresses == addresses(),
                           "distinct_input_and_output_addresses": True},
           "graph_profile": profiles, "graph_trace_sha256": trace_hashes,
           "benchmark": benchmark}
    artifacts.write_json(output / "raw.json", raw)
    artifacts.write_json(output / "summary.json", summarize(raw))
    artifacts.manifest(output)
    return verify(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--condition", choices=tuple(CONDITIONS))
    parser.add_argument("--session", type=int, choices=(1, 2))
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--recover-summary", action="store_true")
    args = parser.parse_args()
    modes = int(args.verify) + int(args.recover_summary) + int(
        args.condition is not None and args.session is not None)
    if modes != 1:
        parser.error("choose --verify, --recover-summary, or both --condition and --session")
    result = (verify(args.output) if args.verify else
              recover_completed_summary(args.output) if args.recover_summary else
              run(args.output, args.condition, args.session))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
