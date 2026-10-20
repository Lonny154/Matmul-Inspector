"""One bounded, paired comparison of two packaged Triton FC1 configurations."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import platform
import queue
import statistics
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

from . import experiment, experiment_compare, ptx


LABELS = ("A", "B")
FIELDS = experiment_compare.MEASUREMENT_FIELDS
PTX_DIFF_LINES = 300
REQUIRED = {"experiment.json", "raw_result.json", "correctness.json", "timings.jsonl",
            "comparison.json", "compiler.json", "provenance.json", "report.md", "candidate_source.py"}


def _write_json(path: Path, value: Any) -> None:
    experiment_compare._write_json(path, value)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_spec(raw: Mapping[str, Any], *, retained_root: Path | None = None) -> dict[str, Any]:
    """Validate versioned registered or trusted-source pairs without importing source."""
    if not isinstance(raw, dict) or set(raw) != {
            "schema_version", "workload", "seed", "correctness", "benchmark",
            "timeout_seconds", "kernels"}:
        raise ValueError("pair spec must contain exactly the versioned fields")
    version = raw["schema_version"]
    if type(version) is not int or version not in (1, 2, 3):
        raise ValueError("schema_version must be 1, 2, or 3")
    kernels = raw["kernels"]
    if not isinstance(kernels, dict) or set(kernels) != set(LABELS):
        raise ValueError("kernels must identify exactly A and B")
    if not isinstance(raw["benchmark"], dict) or set(raw["benchmark"]) != {
            "mode", "warmups", "iterations", "rounds"}:
        raise ValueError("benchmark must contain exactly the paired fields")
    shared = {key: value for key, value in raw.items() if key != "kernels"}
    normalized = {}
    for label in LABELS:
        trial = {**shared, "baseline": {"implementation": "pytorch_matmul"},
                 "candidate": kernels[label]}
        if version == 1:
            normalized[label] = experiment_compare.validate_spec(trial)
            continue
        candidate = copy.deepcopy(kernels[label])
        if not isinstance(candidate, dict):
            raise ValueError(f"kernel {label} must be an object")
        if candidate.get("implementation") == experiment.SOURCE_IMPLEMENTATION and retained_root is not None:
            source = candidate.get("source")
            if not isinstance(source, dict) or source.get("path") != f"sources/{label}.py":
                raise ValueError(f"retained kernel {label} must use sources/{label}.py")
            candidate["source"]["path"] = str((retained_root / source["path"]).resolve())
        checked = experiment.validate_config({key: value for key, value in
                                              {**trial, "candidate": candidate}.items()
                                              if key != "schema_version"})
        # The registered v1 validator owns the established finite/size/budget
        # limits. Its allocating-only check is applied to an equivalent bounds
        # view; v2 execution keeps the original preallocated mode.
        bounds = {**shared, "schema_version": 1,
                  "benchmark": {**shared["benchmark"], "mode": "allocating"},
                  "candidate": {"implementation": "triton_fc1",
                                "parameters": checked["candidate"]["parameters"]},
                  "baseline": {"implementation": "pytorch_matmul"}}
        experiment_compare.validate_spec(bounds)
        if checked["benchmark"]["mode"] != "preallocated":
            raise ValueError("v2/v3 pairs require preallocated mode")
        if retained_root is not None and candidate["implementation"] == experiment.SOURCE_IMPLEMENTATION:
            checked["candidate"]["source"]["path"] = f"sources/{label}.py"
        normalized[label] = {"schema_version": version, **checked}
    if normalized["A"]["candidate"] == normalized["B"]["candidate"]:
        raise ValueError("A and B must be distinct configurations")
    common = {key: value for key, value in normalized["A"].items()
              if key not in ("baseline", "candidate")}
    return {**common, "kernels": {label: normalized[label]["candidate"] for label in LABELS}}


def pair_order(round_index: int) -> list[str]:
    if type(round_index) is not int or round_index < 1:
        raise ValueError("round_index must be positive")
    return ["A", "B"] if round_index % 2 else ["B", "A"]


def measure_paired_rounds(measure, torch, functions, iterations: int, count: int,
                          options: Mapping[str, Any], *, with_aa: bool,
                          aa_placement: str = "before"):
    """Execute the existing balanced blocks with an optional diagnostic phase order."""
    if aa_placement not in ("before", "after"):
        raise ValueError("A/A placement must be before or after")
    aa_rounds, rounds, sequence = [], [], []
    for index in range(1, count + 1):
        if not with_aa:
            phases = ("ab",)
        else:
            phases = ("aa", "ab") if aa_placement == "before" else ("ab", "aa")
        for phase in phases:
            if phase == "aa":
                order = ["X", "Y"] if index % 2 else ["Y", "X"]
                aa_rounds.append({"round": index, "order": order,
                                  "measurements": {slot: measure(
                                      torch, functions["A"], iterations, **options) for slot in order}})
            else:
                order = pair_order(index)
                rounds.append({"round": index, "order": order,
                               "measurements": {label: measure(
                                   torch, functions[label], iterations, **options) for label in order}})
            sequence.append({"round": index, "phase": phase, "order": order,
                             "iterations_per_block": iterations})
    return aa_rounds, rounds, sequence


def expected_diagnostic_sequence(count: int, iterations: int, placement: str) -> list[dict[str, Any]]:
    if placement not in ("before", "after"):
        raise ValueError("A/A placement must be before or after")
    return [{"round": index, "phase": phase,
             "order": (["X", "Y"] if index % 2 else ["Y", "X"]) if phase == "aa" else pair_order(index),
             "iterations_per_block": iterations}
            for index in range(1, count + 1)
            for phase in (("aa", "ab") if placement == "before" else ("ab", "aa"))]


def raw_clock():
    """Return the optional Linux RAW reader; never substitute another clock."""
    if sys.platform.startswith("linux") and hasattr(time, "CLOCK_MONOTONIC_RAW"):
        return lambda: time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
    return None


def _preflight_summary(samples: Sequence[Mapping[str, float]]) -> dict[str, Any]:
    if len(samples) != 5 or any(
            type(sample.get(field)) not in (int, float) or not math.isfinite(sample[field])
            for sample in samples for field in (
                "perf_counter_ms", "monotonic_raw_ms", "raw_minus_perf_ms",
                "raw_over_perf", "read_span_bound_ms")):
        raise ValueError("invalid host clock preflight samples")
    if any(sample["perf_counter_ms"] <= 0 or sample["monotonic_raw_ms"] <= 0 or
           sample["read_span_bound_ms"] < 0 or not math.isclose(
               sample["raw_minus_perf_ms"], sample["monotonic_raw_ms"] - sample["perf_counter_ms"],
               abs_tol=1e-9) or not math.isclose(
               sample["raw_over_perf"], sample["monotonic_raw_ms"] / sample["perf_counter_ms"],
               abs_tol=1e-12) for sample in samples):
        raise ValueError("inconsistent host clock preflight samples")
    differences = [sample["raw_minus_perf_ms"] for sample in samples]
    span = max(sample["read_span_bound_ms"] for sample in samples)
    observed = (all(value > span for value in differences) or
                all(value < -span for value in differences))
    return {"status": "RATE_DIFFERENCE_OBSERVED" if observed else "NO_RESOLVED_DIFFERENCE",
            "samples": list(samples), "median_raw_over_perf": statistics.median(
                sample["raw_over_perf"] for sample in samples),
            "median_raw_minus_perf_ms": statistics.median(differences),
            "read_span_bound_ms": span,
            "rule": "all five differences have one sign and exceed the largest paired read span"}


def host_clock_preflight(*, reader=None, sleep=time.sleep) -> dict[str, Any]:
    """Compare five matched 20 ms host windows, retaining read-order uncertainty."""
    reader = raw_clock() if reader is None else reader
    if reader is None:
        return {"status": "UNAVAILABLE", "reason": "CLOCK_MONOTONIC_RAW is unavailable",
                "samples": []}
    samples = []
    for _ in range(5):
        perf_start = time.perf_counter_ns()
        raw_start = reader()
        perf_after_start = time.perf_counter_ns()
        sleep(0.02)
        perf_before_end = time.perf_counter_ns()
        raw_end = reader()
        perf_end = time.perf_counter_ns()
        perf_ms = (perf_end - perf_start) / 1e6
        raw_ms = (raw_end - raw_start) / 1e6
        samples.append({"perf_counter_ms": perf_ms, "monotonic_raw_ms": raw_ms,
                        "raw_minus_perf_ms": raw_ms - perf_ms,
                        "raw_over_perf": raw_ms / perf_ms,
                        "read_span_bound_ms": ((perf_after_start - perf_start) +
                                               (perf_end - perf_before_end)) / 1e6})
    return _preflight_summary(samples)


def _has_local(spec: Mapping[str, Any]) -> bool:
    return any(spec["kernels"][label]["implementation"] == experiment.SOURCE_IMPLEMENTATION
               for label in LABELS)


def load_trusted_entry(source: Mapping[str, Any], *, allowed: bool = False):
    """Import one explicitly selected, hashed source only after opt-in."""
    if not allowed:
        raise ValueError("trusted local source requires --allow-trusted-local-source")
    path = Path(source["path"])
    if not path.is_file():
        raise experiment.StageFailure("source_loading", "trusted source snapshot is missing")
    if _digest(path) != source["sha256"]:
        raise experiment.StageFailure("source_identity", "trusted source snapshot hash changed")
    previous_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        return experiment.CudaTritonBackend._load_source({"candidate": {"source": source}})
    finally:
        sys.dont_write_bytecode = previous_bytecode


def _compiled_capture(value: Any) -> dict[str, Any] | None:
    """Accept an optional Triton launch result, never a returned output tensor."""
    if value is None:
        return None
    metadata = getattr(value, "metadata", None)
    assembly = getattr(value, "asm", None)
    if metadata is None or not isinstance(assembly, dict):
        raise experiment.StageFailure(
            "interface", "matmul_into must return None or a Triton compiled launch result")
    result = {"kernel_name": str(getattr(value, "name", "UNAVAILABLE")),
              "metadata": {name: getattr(metadata, name, getattr(value, name, None))
                           for name in ("num_warps", "num_stages", "shared", "n_regs", "n_spills")},
              "ptx_text": assembly.get("ptx") if isinstance(assembly.get("ptx"), str) else None,
              "sass_text": assembly.get("sass") if isinstance(assembly.get("sass"), str) else None}
    return result


def check_output(torch, reference, actual, tolerance: Mapping[str, float]) -> dict[str, Any]:
    """Apply the same independent reference gate to either implementation."""
    shape_match = tuple(actual.shape) == tuple(reference.shape)
    dtype_match = actual.dtype == reference.dtype
    reference_nonfinite = int((~torch.isfinite(reference)).sum().item())
    candidate_nonfinite = int((~torch.isfinite(actual)).sum().item())
    if shape_match and not reference_nonfinite and not candidate_nonfinite:
        error = (reference.float() - actual.float()).abs()
        max_error = float(error.max().item())
        mean_error = float(error.mean().item())
    else:
        max_error = mean_error = None
    passed = bool(shape_match and dtype_match and not reference_nonfinite and
                  not candidate_nonfinite and torch.allclose(
                      actual, reference, atol=tolerance["atol"], rtol=tolerance["rtol"]))
    return {"passed": passed, "shape_match": shape_match, "dtype_match": dtype_match,
            "reference_nonfinite_count": reference_nonfinite,
            "candidate_nonfinite_count": candidate_nonfinite,
            "max_absolute_error": max_error, "mean_absolute_error": mean_error}


class PairBackend:
    """GPU-only adapter; importable on a CPU-only machine."""

    def run(self, spec: Mapping[str, Any], log) -> dict[str, Any]:
        torch, kernel = experiment.CudaTritonBackend()._imports("triton_fc1")
        import triton
        device = torch.cuda.get_device_properties(torch.cuda.current_device())
        environment = {
            "device": {"name": device.name, "compute_capability":
                       f"{device.major}.{device.minor}", "total_memory_bytes": device.total_memory},
            "versions": {"python": sys.version.split()[0], "torch": torch.__version__,
                         "triton": triton.__version__, "cuda_runtime": torch.version.cuda},
        }
        w = spec["workload"]
        torch.manual_seed(spec["seed"])
        torch.cuda.manual_seed_all(spec["seed"])
        a = torch.randn((w["m"], w["k"]), device="cuda", dtype=torch.float16)
        b = torch.randn((w["k"], w["n"]), device="cuda", dtype=torch.float16)
        mode = spec["benchmark"]["mode"]
        captured_local: dict[str, dict[str, Any]] = {}
        if mode == "allocating":
            functions = {
                label: (lambda config=kernel.TritonGemmConfig(**spec["kernels"][label]["parameters"]):
                        kernel.triton_gemm(a, b, config)) for label in LABELS
            }
            invoke = None
        else:
            outputs = {label: torch.empty((w["m"], w["n"]), device="cuda", dtype=torch.float16)
                       for label in LABELS}
            invoke = {}
            for label in LABELS:
                candidate = spec["kernels"][label]
                parameters = candidate["parameters"]
                target = outputs[label]
                if candidate["implementation"] == experiment.SOURCE_IMPLEMENTATION:
                    entry = load_trusted_entry(
                        candidate["source"], allowed=spec.get("_allow_trusted_local_source") is True)

                    def local_call(left, right, *, entry=entry, parameters=parameters,
                                   target=target, label=label):
                        try:
                            returned = entry(left, right, target, dict(parameters))
                            if label not in captured_local:
                                capture = _compiled_capture(returned)
                                if capture is not None:
                                    captured_local[label] = capture
                        except experiment.StageFailure:
                            raise
                        except Exception as error:
                            raise experiment.StageFailure(
                                "execution", f"trusted local {label} invocation failed ({type(error).__name__})") from None
                        return target

                    invoke[label] = local_call
                else:
                    grid = (triton.cdiv(w["m"], parameters["block_m"])
                            * triton.cdiv(w["n"], parameters["block_n"]),)

                    def registered_call(left, right, *, grid=grid, parameters=parameters,
                                        target=target):
                        kernel.fc1_kernel[grid](
                            left, right, target, w["m"], w["n"], w["k"],
                            left.stride(0), left.stride(1), right.stride(0), right.stride(1),
                            target.stride(0), target.stride(1),
                            BLOCK_M=parameters["block_m"], BLOCK_N=parameters["block_n"],
                            BLOCK_K=parameters["block_k"], GROUP_SIZE_M=parameters["group_size_m"],
                            num_warps=parameters["num_warps"], num_stages=parameters["num_stages"])
                        return target

                    invoke[label] = registered_call
            functions = {label: (lambda label=label: invoke[label](a, b)) for label in LABELS}
            for target in outputs.values():
                target.fill_(float("nan"))
        log("compilation: first synchronized invocation of reference, A, and B")
        setup_ms = {}
        outputs = {}
        for label, function in (("reference", lambda: torch.matmul(a, b)),
                                *[(name, functions[name]) for name in LABELS]):
            started = time.perf_counter()
            try:
                outputs[label] = function()
                torch.cuda.synchronize()
            except Exception as error:
                raise experiment.StageFailure("compilation", f"{label} invocation failed: {error}") from error
            setup_ms[label] = (time.perf_counter() - started) * 1000
        reference = outputs["reference"]
        checks = {}
        for label in LABELS:
            checks[label] = check_output(torch, reference, outputs[label], spec["correctness"])
        correctness = {"passed": all(checks[label]["passed"] for label in LABELS),
                       "reference": "pytorch_matmul (cuBLAS)", "seed": spec["seed"],
                       "atol": spec["correctness"]["atol"], "rtol": spec["correctness"]["rtol"],
                       "checks": checks}
        if mode == "preallocated" and correctness["passed"]:
            additional_cases = []
            for seed in (spec["seed"] + 1, spec["seed"] + 2):
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                validation_a = torch.randn(tuple(a.shape), device="cuda", dtype=a.dtype)
                validation_b = torch.randn(tuple(b.shape), device="cuda", dtype=b.dtype)
                validation_reference = torch.matmul(validation_a, validation_b)
                case = {"seed": seed, "checks": {}}
                for label in LABELS:
                    outputs[label].fill_(float("nan"))
                    actual = invoke[label](validation_a, validation_b)
                    torch.cuda.synchronize()
                    case["checks"][label] = check_output(
                        torch, validation_reference, actual, spec["correctness"])
                additional_cases.append(case)
            correctness["additional_cases"] = additional_cases
            correctness["passed"] = all(case["checks"][label]["passed"]
                                        for case in additional_cases for label in LABELS)
        partial = {"environment": environment, "setup_ms": setup_ms, "correctness": correctness}
        if not correctness["passed"]:
            return {**partial, "status": "failed", "failure": {
                "stage": "correctness", "message": "one or both candidates failed the independent reference"}}
        bench = spec["benchmark"]
        diagnostic_placement = spec.get("_diagnostic_aa_placement")
        if diagnostic_placement is not None and (
                spec["schema_version"] != 3 or diagnostic_placement not in ("before", "after")):
            raise experiment.StageFailure("benchmark", "invalid diagnostic A/A placement")
        log("benchmark: separate warmups, unranked timer controls, and balanced paired rounds")
        try:
            clock_reader = raw_clock() if spec["schema_version"] == 3 else None
            clock_check = host_clock_preflight(reader=clock_reader) if spec["schema_version"] == 3 else None
            for label in LABELS:
                for _ in range(bench["warmups"]):
                    functions[label]()
                torch.cuda.synchronize()
            measure = experiment.CudaTritonBackend._measure
            measure_options = {"raw_clock": clock_reader} if spec["schema_version"] == 3 else {}
            controls = {label: measure(torch, functions[label], bench["iterations"], **measure_options)
                        for label in LABELS}
            aa_rounds, rounds, sequence = measure_paired_rounds(
                measure, torch, functions, bench["iterations"], bench["rounds"],
                measure_options, with_aa=spec["schema_version"] == 3,
                aa_placement=diagnostic_placement or "before")
        except Exception as error:
            raise experiment.StageFailure("benchmark", f"paired measurement failed: {error}") from error
        return {**partial, "status": "success", "local_compiler_capture": captured_local,
                "benchmark": {
            "mode": mode, "warmups_per_kernel": bench["warmups"],
            "iterations_per_measurement": bench["iterations"],
            "wrapper_work": ("triton_gemm validates inputs, allocates output, builds grid and launches on every call"
                             if mode == "allocating" else
                             "registered and trusted wrappers launch into separate preallocated outputs"),
            "timing_boundaries": ("perf start, RAW start, start event, N calls, end event, "
                                  "end-event synchronization, RAW end, perf end; both host clocks "
                                  "enclose event completion" if spec["schema_version"] == 3 else
                                  "CUDA events bracket submissions; synchronized host wall brackets event completion"),
            "unranked_timing_controls": controls, "rounds": rounds,
            **({"host_clock_preflight": clock_check, "aa_rounds": aa_rounds,
                "aa_kernel": "A", "aa_iterations_per_measurement": bench["iterations"]}
               if spec["schema_version"] == 3 else {}),
            **({"diagnostic_aa_placement": diagnostic_placement,
                "measurement_sequence": sequence} if diagnostic_placement is not None else {})}}


def run_worker(spec: Mapping[str, Any], *, runtime_spec: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Use the established experiment worker and timeout pattern."""
    execution = dict(runtime_spec or spec)
    context = multiprocessing.get_context("fork" if sys.platform != "win32" else "spawn")
    messages = context.Queue()
    process = context.Process(target=experiment._worker, args=(execution, PairBackend, messages))
    process.start()
    try:
        process.join(float(spec["timeout_seconds"]))
        if process.is_alive():
            process.terminate()
            process.join(5)
            if process.is_alive():
                process.kill()
                process.join()
            payload = {"status": "failed", "failure": {"stage": "timeout",
                        "message": f"GPU worker exceeded {spec['timeout_seconds']} seconds"}}
            logs = []
        else:
            try:
                message = messages.get(timeout=1)
            except queue.Empty:
                payload = {"status": "failed", "failure": {"stage": "execution",
                            "message": f"worker exited {process.exitcode} without a result"}}
                logs = []
            else:
                payload, logs = message["payload"], message["logs"]
    finally:
        messages.close()
        messages.join_thread()
    if _has_local(spec) and "failure" in payload:
        stage = payload["failure"].get("stage", "execution")
        payload["failure"] = {"stage": stage, "exception_type": "TrustedSourceFailure",
                              "message": f"trusted local experiment failed at {stage}; details omitted from artifacts",
                              "traceback": None}
    result = {"spec": dict(spec), "git": experiment._git_metadata(Path(__file__).resolve().parents[2]),
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "logs": logs, **payload}
    if runtime_spec is not None:
        result["execution_source_paths"] = {
            label: execution["kernels"][label]["source"]["path"]
            for label in LABELS if execution["kernels"][label]["implementation"] == experiment.SOURCE_IMPLEMENTATION}
    return result


def timing_rows(raw: Mapping[str, Any]) -> list[dict[str, Any]]:
    benchmark = raw.get("benchmark")
    if not isinstance(benchmark, dict):
        return []
    return [{"round": round_row["round"], "order": round_row["order"],
             "kernel": label, "iterations": benchmark["iterations_per_measurement"],
             **round_row["measurements"][label]}
            for round_row in benchmark["rounds"] for label in round_row["order"]]


def _stats(values: list[float], *, signed: bool = False) -> dict[str, Any]:
    median = statistics.median(values)
    return {"median_ms": median, "min_ms": min(values), "max_ms": max(values),
            "stdev_ms": statistics.stdev(values),
            "cv": None if signed else statistics.stdev(values) / statistics.fmean(values),
            "median_absolute_deviation_ms": statistics.median(abs(x - median) for x in values)}


def summarize(correctness: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
              failure: Mapping[str, Any] | None = None,
              controls: Mapping[str, Any] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"decision": "INCONCLUSIVE", "reason": "comparable timings unavailable",
                              "ranking_supported": False, "speedup_A_over_B": None,
                              "timing_basis": "paired eager CUDA-event milliseconds per operation"}
    if failure:
        result.update(decision="REJECTED_INCORRECT" if failure.get("stage") == "correctness" else "FAILED",
                      reason=failure.get("message", "experiment failed"))
        return result
    if correctness.get("passed") is not True or any(
            correctness.get("checks", {}).get(label, {}).get("passed") is not True for label in LABELS):
        result.update(decision="REJECTED_INCORRECT",
                      reason="both candidates must pass the independent reference")
        return result
    rounds: dict[int, dict[str, Mapping[str, Any]]] = {}
    iterations_seen: set[int] = set()
    try:
        for row in rows:
            index, label = row["round"], row["kernel"]
            if type(index) is not int or label not in LABELS or row["order"] != pair_order(index):
                return result
            if type(row.get("iterations")) is not int or row["iterations"] < 1 or (
                    label in rounds.setdefault(index, {})):
                return result
            iterations_seen.add(row["iterations"])
            if any(type(row.get(field)) not in (int, float) or not math.isfinite(row[field])
                   for field in FIELDS):
                return result
            if row["cuda_event_elapsed_ms_per_operation"] <= 0 or row["host_wall_ms_per_block"] <= 0:
                return result
            rounds[index][label] = row
    except (KeyError, TypeError, ValueError):
        return result
    if len(iterations_seen) != 1 or len(rounds) < 5 or (
            set(rounds) != set(range(1, len(rounds) + 1))) or any(
            set(pair) != set(LABELS) for pair in rounds.values()):
        return result
    ordered = [rounds[index] for index in sorted(rounds)]
    values = {label: [pair[label]["cuda_event_elapsed_ms_per_operation"] for pair in ordered]
              for label in LABELS}
    host = {label: [pair[label]["host_wall_ms_per_block"] / pair[label]["iterations"]
                    for pair in ordered] for label in LABELS}
    delta = [a - b for a, b in zip(values["A"], values["B"])]
    host_delta = [a - b for a, b in zip(host["A"], host["B"])]
    result.update(rounds=len(ordered), cuda_event_ms={label: _stats(values[label]) for label in LABELS},
                  host_wall_ms={label: _stats(host[label]) for label in LABELS},
                  paired_A_minus_B_ms=delta, paired_host_A_minus_B_ms=host_delta,
                  paired_difference_ms=_stats(delta, signed=True),
                  paired_signs={"A_faster": sum(x < 0 for x in delta),
                                "B_faster": sum(x > 0 for x in delta),
                                "ties": sum(x == 0 for x in delta)})
    boundary_fail = any(row["cuda_event_minus_host_wall_ms_per_block"] > 0 for row in rows)
    if controls is None or set(controls) != set(LABELS):
        result["reason"] = "unranked timing controls are unavailable"
        return result
    result["timing_boundary_ms"] = {
        label: {
            "unranked_event_minus_wall": controls[label]["cuda_event_minus_host_wall_ms_per_block"],
            "round_event_minus_wall_range": [
                min(pair[label]["cuda_event_minus_host_wall_ms_per_block"] for pair in ordered),
                max(pair[label]["cuda_event_minus_host_wall_ms_per_block"] for pair in ordered)],
        } for label in LABELS}
    boundary_fail |= any(controls[label]["cuda_event_minus_host_wall_ms_per_block"] > 0
                         for label in LABELS)
    result["timing_boundary_discrepancy"] = boundary_fail
    if boundary_fail:
        result["reason"] = "CUDA-event time exceeded synchronized host wall in a round or control"
        return result
    if max(values["A"]) < min(values["B"]) and all(x < 0 for x in host_delta):
        winner = "A"
    elif max(values["B"]) < min(values["A"]) and all(x > 0 for x in host_delta):
        winner = "B"
    else:
        result["reason"] = "round ranges overlap or host-wall paired direction disagrees"
        return result
    result.update(decision=f"{winner}_FASTER", ranking_supported=True,
                  speedup_A_over_B=statistics.median(values["B"]) / statistics.median(values["A"]),
                  reason="all paired rounds agree and CUDA-event ranges separate; host-wall pairs agree")
    return result


def summarize_clock_aware(correctness: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
                          failure: Mapping[str, Any] | None,
                          benchmark: Mapping[str, Any] | None) -> dict[str, Any]:
    """Version 3: retain legacy statistics; decide using measured clock and A/A controls."""
    controls = (benchmark or {}).get("unranked_timing_controls")
    result = summarize(correctness, rows, failure, controls)
    result["decision_schema_version"] = 3
    if result["decision"] in ("FAILED", "REJECTED_INCORRECT"):
        return result
    result.update(decision="INCONCLUSIVE", ranking_supported=False, speedup_A_over_B=None)
    warnings = []
    preflight = (benchmark or {}).get("host_clock_preflight")
    aa_rounds = (benchmark or {}).get("aa_rounds")
    if ("cuda_event_ms" not in result or not isinstance(preflight, dict) or
            not isinstance(aa_rounds, list) or not isinstance(controls, dict) or
            set(controls) != set(LABELS)):
        result.update(reason="version 3 timing controls are incomplete", warnings=["MISSING_DIAGNOSTICS"])
        return result
    status = preflight.get("status")
    try:
        if status == "UNAVAILABLE":
            if preflight != {"status": "UNAVAILABLE",
                            "reason": "CLOCK_MONOTONIC_RAW is unavailable", "samples": []}:
                raise ValueError("invalid unavailable preflight")
        elif preflight != _preflight_summary(preflight["samples"]):
            raise ValueError("preflight summary differs from samples")
    except (KeyError, TypeError, ValueError):
        result.update(reason="host clock preflight does not replay", warnings=["INVALID_CLOCK_PREFLIGHT"])
        return result
    if status == "RATE_DIFFERENCE_OBSERVED":
        warnings.append("HOST_CLOCK_RATE_DIFFERENCE")
    elif status == "UNAVAILABLE":
        warnings.append("MONOTONIC_RAW_UNAVAILABLE")
    elif status != "NO_RESOLVED_DIFFERENCE":
        warnings.append("INVALID_CLOCK_PREFLIGHT")
    if result.get("timing_boundary_discrepancy"):
        warnings.append("CUDA_EVENT_EXCEEDS_PERF_COUNTER_WALL")
    raw_fields = ("monotonic_raw_ms_per_block", "cuda_event_minus_monotonic_raw_ms_per_block")
    all_measurements = list(rows) + list(controls.values() if isinstance(controls, dict) else [])
    aa_differences = []
    try:
        if len(aa_rounds) != result["rounds"] or (benchmark or {}).get("aa_kernel") != "A":
            raise ValueError("A/A round count or identity differs")
        if (benchmark or {}).get("aa_iterations_per_measurement") != rows[0]["iterations"]:
            raise ValueError("A/A operation count differs")
        for index, round_row in enumerate(aa_rounds, 1):
            order = ["X", "Y"] if index % 2 else ["Y", "X"]
            if round_row["round"] != index or round_row["order"] != order:
                raise ValueError("A/A order differs")
            measures = round_row["measurements"]
            if set(measures) != {"X", "Y"}:
                raise ValueError("A/A slot missing")
            all_measurements.extend(measures.values())
            aa_differences.append((measures["X"]["cuda_event_elapsed_ms_per_block"] -
                                   measures["Y"]["cuda_event_elapsed_ms_per_block"]) /
                                  rows[0]["iterations"])
        for sample in all_measurements:
            if any(type(sample.get(field)) not in (int, float) or
                   not math.isfinite(sample[field]) or sample[field] <= 0
                   for field in ("cuda_event_elapsed_ms_per_block", "host_wall_ms_per_block")):
                raise ValueError("invalid clock measurement")
            if status != "UNAVAILABLE" and any(type(sample.get(field)) not in (int, float)
                    or not math.isfinite(sample[field]) for field in raw_fields):
                raise ValueError("RAW measurement missing")
            event = sample["cuda_event_elapsed_ms_per_block"]
            perf = sample["host_wall_ms_per_block"]
            if not math.isclose(sample["cuda_event_elapsed_ms_per_operation"] * rows[0]["iterations"],
                                event, abs_tol=1e-8) or not math.isclose(
                    sample["cuda_event_minus_host_wall_ms_per_block"], event - perf, abs_tol=1e-8):
                raise ValueError("event or perf normalization differs")
            if status != "UNAVAILABLE" and (sample["monotonic_raw_ms_per_block"] <= 0 or
                    not math.isclose(sample[raw_fields[1]],
                                     event - sample[raw_fields[0]], abs_tol=1e-8)):
                raise ValueError("RAW difference differs")
    except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError):
        result.update(reason="version 3 timing controls are malformed", warnings=warnings + ["INVALID_DIAGNOSTICS"])
        return result
    result["aa_control"] = {"rounds": len(aa_rounds), "kernel": "A",
                            "paired_X_minus_Y_ms": aa_differences,
                            "paired_difference_ms": _stats(aa_differences, signed=True),
                            "max_absolute_difference_ms": max(abs(value) for value in aa_differences)}
    result["host_clock_preflight"] = preflight
    result["monotonic_raw_ms"] = ({label: _stats([
        row["monotonic_raw_ms_per_block"] / row["iterations"]
        for row in rows if row["kernel"] == label]) for label in LABELS}
        if status != "UNAVAILABLE" else None)
    result["raw_boundary_discrepancy"] = (any(sample[raw_fields[1]] > 0 for sample in all_measurements)
                                           if status != "UNAVAILABLE" else None)
    if result["raw_boundary_discrepancy"]:
        warnings.append("CUDA_EVENT_EXCEEDS_MONOTONIC_RAW_WALL")
    result["warnings"] = warnings
    if status == "UNAVAILABLE":
        result["reason"] = "RAW clock unavailable; clock-aware boundary cannot be assessed"
        return result
    if status not in ("RATE_DIFFERENCE_OBSERVED", "NO_RESOLVED_DIFFERENCE"):
        result["reason"] = "host clock preflight invalid"
        return result
    if result["raw_boundary_discrepancy"] or (status == "NO_RESOLVED_DIFFERENCE" and
                                               result.get("timing_boundary_discrepancy")):
        result["reason"] = "event time exceeds a host interval without a resolved clock-rate explanation"
        return result
    values = {label: [row["cuda_event_elapsed_ms_per_operation"] for row in rows
                      if row["kernel"] == label] for label in LABELS}
    host_delta = result["paired_host_A_minus_B_ms"]
    raw_by_round = {index: {row["kernel"]: row for row in rows if row["round"] == index}
                    for index in range(1, result["rounds"] + 1)}
    raw_delta = [(pair["A"]["monotonic_raw_ms_per_block"] -
                  pair["B"]["monotonic_raw_ms_per_block"]) / pair["A"]["iterations"]
                 for pair in raw_by_round.values()]
    result["paired_raw_A_minus_B_ms"] = raw_delta
    aa_floor = result["aa_control"]["max_absolute_difference_ms"]
    ab_delta = result["paired_A_minus_B_ms"]
    result["ab_exceeds_aa_variability"] = all(abs(value) > aa_floor for value in ab_delta)
    if not result["ab_exceeds_aa_variability"]:
        result["reason"] = "an A/B paired gap does not exceed the largest observed A/A gap"
        result["warnings"].append("AB_WITHIN_AA_VARIABILITY")
        return result
    if max(values["A"]) < min(values["B"]) and all(x < 0 for x in host_delta + raw_delta):
        winner = "A"
    elif max(values["B"]) < min(values["A"]) and all(x > 0 for x in host_delta + raw_delta):
        winner = "B"
    else:
        result["reason"] = "event ranges overlap or paired host directions disagree"
        return result
    result.update(decision=f"{winner}_FASTER", ranking_supported=True,
                  speedup_A_over_B=statistics.median(values["B"]) / statistics.median(values["A"]),
                  reason="event ranges separate, both host paired directions agree, and every A/B gap exceeds A/A")
    return result


def capture_compiler(spec: Mapping[str, Any], output: Path,
                     raw: Mapping[str, Any] | None = None) -> dict[str, Any]:
    captures = {}
    for label in LABELS:
        candidate = spec["kernels"][label]
        if candidate["implementation"] == experiment.SOURCE_IMPLEMENTATION:
            captured = (raw or {}).get("local_compiler_capture", {}).get(label)
            result = {"status": "UNAVAILABLE", "ptx": "UNAVAILABLE", "sass": "UNAVAILABLE",
                      "sass_reason": "trusted entrypoint did not expose decoded SASS"}
            if captured:
                result.update(status="CAPTURED", kernel_name=captured["kernel_name"],
                              metadata=captured["metadata"])
                for extension in ("ptx", "sass"):
                    value = captured.get(f"{extension}_text")
                    if isinstance(value, str):
                        (output / f"{label}.{extension}").write_text(value, encoding="utf-8")
                        result[extension] = "CAPTURED"
                if result["ptx"] == "CAPTURED":
                    try:
                        module = ptx.parse_module(captured["ptx_text"])
                        result["ptx_summary"] = ptx.module_summary(
                            module, [symbol for symbol in module.symbols if symbol.kind == "entry"])
                    except (ValueError, KeyError) as error:
                        result["ptx_summary_status"] = "UNAVAILABLE"
                        result["ptx_summary_reason"] = type(error).__name__
            else:
                result["reason"] = "matmul_into returned None; compiler result unavailable"
            captures[label] = result
        else:
            single = {**spec, "candidate": candidate}
            captures[label] = experiment_compare._compiler_capture(single, output)
            for extension in ("ptx", "sass"):
                source = output / f"candidate.{extension}"
                if source.exists():
                    source.rename(output / f"{label}.{extension}")
    comparison: dict[str, Any] = {"status": "UNAVAILABLE"}
    if all((output / f"{label}.ptx").is_file() for label in LABELS):
        def sole_entry(label: str) -> str | None:
            module = ptx.parse_module((output / f"{label}.ptx").read_text())
            entries = [symbol.name for symbol in module.symbols if symbol.kind == "entry"]
            return entries[0] if len(entries) == 1 else None

        try:
            args = argparse.Namespace(baseline=output / "A.ptx", candidate=output / "B.ptx",
                                      kernel=None, baseline_kernel=sole_entry("A"),
                                      candidate_kernel=sole_entry("B"),
                                      diff_lines=PTX_DIFF_LINES, output=output / "ptx_comparison")
            comparison = ptx.compare_command(args)
            comparison = {"status": "CAPTURED", "artifact": "ptx_comparison/comparison.json",
                          "normalized_equal": comparison["comparison"]["normalized"]["identical"],
                          "normalized_changed_lines": comparison["comparison"]["normalized"]["changed_lines"],
                          "normalized_A_sha256": comparison["comparison"]["normalized"]["baseline_sha256"],
                          "normalized_B_sha256": comparison["comparison"]["normalized"]["candidate_sha256"],
                          "A_ptx_sha256": comparison["baseline"]["file"]["sha256"],
                          "B_ptx_sha256": comparison["candidate"]["file"]["sha256"],
                          "kernel": comparison["comparison"]["kernel"]}
        except (OSError, ValueError, KeyError) as error:
            comparison = {"status": "UNAVAILABLE", "reason": f"PTX comparison failed: {type(error).__name__}"}
    return {"A": captures["A"], "B": captures["B"], "ptx_comparison": comparison,
            "machine_code_note": "PTX differences do not establish decoded SASS or final machine-code changes"}


def render_report(spec: Mapping[str, Any], correctness: Mapping[str, Any],
                  decision: Mapping[str, Any], compiler: Mapping[str, Any],
                  provenance: Mapping[str, Any]) -> str:
    w = spec["workload"]
    lines = ["# Paired FC1 A/B comparison", "",
             f"FP16 M={w['m']}, K={w['k']}, N={w['n']}; eager {spec['benchmark']['mode']} mode.",
             "Independent reference: PyTorch/cuBLAS `torch.matmul`.", ""]
    for label in LABELS:
        check = correctness.get("checks", {}).get(label, {})
        parameters = json.dumps(spec["kernels"][label]["parameters"], sort_keys=True)
        lines.append(f"- {label}: `{parameters}`; correctness "
                     f"{'PASS' if check.get('passed') else 'FAIL/UNAVAILABLE'}; "
                     f"max absolute error {check.get('max_absolute_error')}.")
    for case in correctness.get("additional_cases", []):
        labels = ", ".join(f"{label} {'PASS' if case['checks'][label]['passed'] else 'FAIL'}"
                           for label in LABELS)
        lines.append(f"- Additional seed {case['seed']}: {labels}.")
    lines += ["", f"**Decision: {decision['decision']}** — {decision['reason']}", ""]
    if "cuda_event_ms" in decision:
        for label in LABELS:
            event = decision["cuda_event_ms"][label]
            wall = decision["host_wall_ms"][label]
            lines.append(f"- {label}: CUDA-event median {event['median_ms']:.6f} ms/op "
                         f"(range {event['min_ms']:.6f}–{event['max_ms']:.6f}, CV {event['cv']:.2%}); "
                         f"host-wall median {wall['median_ms']:.6f} ms/op.")
        paired = decision["paired_difference_ms"]
        lines += [f"- Paired A−B: median {paired['median_ms']:.6f} ms/op; "
                  f"range {paired['min_ms']:.6f}–{paired['max_ms']:.6f}; "
                  f"MAD {paired['median_absolute_deviation_ms']:.6f} ms/op.",
                  f"- Paired signs: {decision['paired_signs']}. "
                  f"Timer discrepancy: {decision.get('timing_boundary_discrepancy')}.", ""]
        lines += ["| Round | CUDA-event A−B (ms/op) | Host-wall A−B (ms/op) |",
                  "|---:|---:|---:|"]
        for index, (event_delta, wall_delta) in enumerate(zip(
                decision["paired_A_minus_B_ms"], decision["paired_host_A_minus_B_ms"]), 1):
            lines.append(f"| {index} | {event_delta:+.6f} | {wall_delta:+.6f} |")
        lines.append("")
        if "timing_boundary_ms" in decision:
            for label in LABELS:
                boundary = decision["timing_boundary_ms"][label]
                lines.append(f"- {label} event−wall control: {boundary['unranked_event_minus_wall']:+.6f} ms/block; "
                             f"round range {boundary['round_event_minus_wall_range'][0]:+.6f} to "
                             f"{boundary['round_event_minus_wall_range'][1]:+.6f} ms/block.")
            lines.append("")
    if spec["schema_version"] == 3:
        lines += ["## Clock-aware diagnostics (v3)", "",
                  "The event interval brackets the configured calls on one CUDA stream. The perf and RAW "
                  "host intervals enclose end-event synchronization. Perf is sampled just outside RAW, "
                  "so the host windows differ by timestamp-read overhead. No clock is rescaled or treated "
                  "as a universal absolute reference.", ""]
        preflight = decision.get("host_clock_preflight") or {}
        if provenance.get("diagnostic_aa_placement"):
            lines.append(f"- Diagnostic A/A placement: {provenance['diagnostic_aa_placement']} "
                         "each A/B pair; exact phase order is in `raw_result.json`.")
        lines.append(f"- Host-clock preflight: {preflight.get('status', 'UNAVAILABLE')}; "
                     f"median RAW/perf rate ratio {preflight.get('median_raw_over_perf', 'UNAVAILABLE')}; "
                     f"five 20 ms windows when RAW is available.")
        if "aa_control" in decision:
            aa = decision["aa_control"]
            lines.append(f"- Same-kernel A/A: {aa['rounds']} balanced rounds; event X−Y "
                         f"median {aa['paired_difference_ms']['median_ms']:+.6f} ms/op; "
                         f"maximum absolute paired gap {aa['max_absolute_difference_ms']:.6f} ms/op.")
        if "cuda_event_ms" in decision:
            for label in LABELS:
                event = decision["cuda_event_ms"][label]["median_ms"]
                perf = decision["host_wall_ms"][label]["median_ms"]
                raw = ((decision.get("monotonic_raw_ms") or {}).get(label) or {}).get("median_ms")
                lines.append(f"- {label} median ms/op: CUDA event {event:.6f}; "
                             f"perf_counter wall {perf:.6f}; "
                             f"MONOTONIC_RAW {f'{raw:.6f}' if raw is not None else 'UNAVAILABLE'}.")
            lines.append(f"- RAW enclosing-boundary discrepancy: {decision.get('raw_boundary_discrepancy')}; "
                         f"A/B exceeds observed A/A gap: {decision.get('ab_exceeds_aa_variability')}.")
            if decision.get("paired_raw_A_minus_B_ms") is not None and "aa_control" in decision:
                lines += ["", "| Round | Event A−B | perf A−B | RAW A−B | A/A event X−Y |",
                          "|---:|---:|---:|---:|---:|"]
                for index, (event, perf, raw, aa) in enumerate(zip(
                        decision["paired_A_minus_B_ms"], decision["paired_host_A_minus_B_ms"],
                        decision["paired_raw_A_minus_B_ms"],
                        decision["aa_control"]["paired_X_minus_Y_ms"]), 1):
                    lines.append(f"| {index} | {event:+.6f} | {perf:+.6f} | "
                                 f"{raw:+.6f} | {aa:+.6f} |")
                lines += ["", "All table differences are ms per operation; raw block totals and "
                          "operation counts are retained in `raw_result.json`.", ""]
        lines.append(f"- Warnings: {', '.join(decision.get('warnings', [])) or 'none'}.")
        lines += ["- A/A is a local variability control, not a confidence interval. A supported ranking "
                  "requires correctness, complete controls, event range separation, both host paired "
                  "directions, and each A/B gap beyond the largest A/A gap. A RAW rate difference can "
                  "explain a perf boundary reversal but cannot prove RAW absolute accuracy.", ""]
    for label in LABELS:
        capture = compiler[label]
        metadata = capture.get("metadata", {})
        lines.append(f"- {label} compiler: shared={metadata.get('shared')} bytes; "
                     f"registers/thread={metadata.get('n_regs')}; spills={metadata.get('n_spills')}; "
                     f"warps={metadata.get('num_warps')}; stages={metadata.get('num_stages')}; "
                     f"PTX={capture.get('ptx')}; decoded SASS={capture.get('sass')}.")
    a_shared = compiler["A"].get("metadata", {}).get("shared")
    b_shared = compiler["B"].get("metadata", {}).get("shared")
    a_regs = compiler["A"].get("metadata", {}).get("n_regs")
    b_regs = compiler["B"].get("metadata", {}).get("n_regs")
    if type(a_shared) is int and type(b_shared) is int:
        lines.append(f"- Compiler shared-memory B−A: {b_shared - a_shared:+d} bytes per program.")
    if type(a_regs) is int and type(b_regs) is int:
        lines.append(f"- Compiler registers/thread B−A: {b_regs - a_regs:+d}.")
    ptx_difference = compiler["ptx_comparison"]
    lines += [f"- Normalized PTX equal: {ptx_difference.get('normalized_equal', 'UNAVAILABLE')}; "
              f"changed lines: {ptx_difference.get('normalized_changed_lines', 'UNAVAILABLE')}.",
              f"- A PTX SHA-256: `{ptx_difference.get('A_ptx_sha256', 'UNAVAILABLE')}`.",
              f"- B PTX SHA-256: `{ptx_difference.get('B_ptx_sha256', 'UNAVAILABLE')}`.",
              f"- PTX comparison: `{ptx_difference.get('artifact', 'UNAVAILABLE')}`.",
              "- PTX is not decoded SASS. Resource metadata alone does not prove a latency gain.",
              "", "## Provenance and reproduction", ""]
    environment = provenance.get("environment") or {}
    device = environment.get("device") or {}
    versions = environment.get("versions") or {}
    versions_text = (json.dumps(versions, sort_keys=True)
                     if spec["schema_version"] in (2, 3) else str(versions))
    command = provenance.get("command") or []
    lines += [f"GPU: {device.get('name', 'UNAVAILABLE')}; driver: {provenance.get('driver_version')}.",
              f"Software: {versions_text}.",
              f"Source SHA-256: `{provenance.get('candidate_source_sha256')}`; "
              f"Git: `{(provenance.get('git') or {}).get('commit')}` (dirty={(provenance.get('git') or {}).get('dirty')}).",
              f"Run: `{' '.join(command)}`."]
    if provenance.get("source_identities"):
        for label in LABELS:
            identity = provenance["source_identities"][label]
            lines.append(f"{label} source: `{identity['path']}`; entrypoint `{identity['entry_point']}`; "
                         f"SHA-256 `{identity['sha256']}`.")
    if "--output" in command:
        path = command[command.index("--output") + 1]
        lines.append(f"Verify: `matmul-inspector-python experiment pair --verify {path}`.")
    lines += ["Raw paired blocks, compiler files, source snapshot, and SHA-256 hashes are in this bundle.", ""]
    return "\n".join(lines)


def _manifest(output: Path) -> None:
    experiment_compare._manifest(output)


def verify_bundle(output: Path) -> dict[str, Any]:
    manifest = json.loads((output / "hash_manifest.json").read_text())
    if not isinstance(manifest, dict) or any(
            not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts
            for name in manifest):
        raise ValueError("invalid manifest paths")
    actual = {str(path.relative_to(output)) for path in output.rglob("*")
              if path.is_file() and path.name != "hash_manifest.json"}
    if set(manifest) != actual or any(_digest(output / name) != digest for name, digest in manifest.items()):
        raise ValueError("manifest does not match bundle files")
    if not REQUIRED <= actual:
        raise ValueError("required bundle file missing")
    saved_spec = json.loads((output / "experiment.json").read_text())
    spec = validate_spec(saved_spec, retained_root=output if saved_spec.get("schema_version") in (2, 3) else None)
    raw = json.loads((output / "raw_result.json").read_text())
    correctness = json.loads((output / "correctness.json").read_text())
    rows = [json.loads(line) for line in (output / "timings.jsonl").read_text().splitlines() if line]
    decision = json.loads((output / "comparison.json").read_text())
    compiler = json.loads((output / "compiler.json").read_text())
    provenance = json.loads((output / "provenance.json").read_text())
    controls = raw.get("benchmark", {}).get("unranked_timing_controls")
    if (raw.get("spec") != spec or raw.get("correctness", {"passed": None}) != correctness or
            (raw.get("benchmark") is not None and (
                raw["benchmark"].get("iterations_per_measurement") != spec["benchmark"]["iterations"] or
                raw["benchmark"].get("warmups_per_kernel") != spec["benchmark"]["warmups"] or
                raw["benchmark"].get("mode") != spec["benchmark"]["mode"] or
                len(raw["benchmark"].get("rounds", [])) != spec["benchmark"]["rounds"])) or
            timing_rows(raw) != rows or
            (summarize_clock_aware(correctness, rows, raw.get("failure"), raw.get("benchmark"))
             if spec["schema_version"] == 3 else
             summarize(correctness, rows, raw.get("failure"), controls)) != decision or
            render_report(spec, correctness, decision, compiler, provenance) != (output / "report.md").read_text() or
            provenance.get("candidate_source_sha256") != _digest(output / "candidate_source.py") or
            provenance.get("environment") != raw.get("environment") or
            provenance.get("git") != raw.get("git") or
            any((compiler[label].get("ptx") == "CAPTURED") != (f"{label}.ptx" in actual)
                or (compiler[label].get("sass") == "CAPTURED") != (f"{label}.sass" in actual)
                for label in LABELS)):
        raise ValueError("bundle does not reproduce from the raw result")
    if (spec["schema_version"] == 3 and raw.get("status") == "success" and
            set(decision.get("warnings", [])) & {
                "MISSING_DIAGNOSTICS", "INVALID_CLOCK_PREFLIGHT", "INVALID_DIAGNOSTICS"}):
        raise ValueError("successful v3 bundle has invalid timing diagnostics")
    diagnostic = raw.get("benchmark", {}).get("diagnostic_aa_placement")
    if diagnostic is not None:
        if (spec["schema_version"] != 3 or diagnostic not in ("before", "after") or
                provenance.get("diagnostic_aa_placement") != diagnostic or
                raw["benchmark"].get("measurement_sequence") != expected_diagnostic_sequence(
                    spec["benchmark"]["rounds"], spec["benchmark"]["iterations"], diagnostic)):
            raise ValueError("diagnostic phase order does not replay")
    elif (provenance.get("diagnostic_aa_placement") is not None or
          "measurement_sequence" in raw.get("benchmark", {})):
        raise ValueError("diagnostic placement metadata is incomplete")
    if _has_local(spec):
        if provenance.get("trusted_local_source_opt_in") is not True:
            raise ValueError("trusted source opt-in provenance missing")
        identities = provenance.get("source_identities") or {}
        execution_paths = raw.get("execution_source_paths") or {}
        for label in LABELS:
            candidate = spec["kernels"][label]
            identity = identities.get(label, {})
            expected_path = (candidate["source"]["path"]
                             if candidate["implementation"] == experiment.SOURCE_IMPLEMENTATION
                             else "candidate_source.py")
            expected_sha = (candidate["source"]["sha256"]
                            if candidate["implementation"] == experiment.SOURCE_IMPLEMENTATION
                            else _digest(output / "candidate_source.py"))
            if identity.get("path") != expected_path or identity.get("sha256") != expected_sha:
                raise ValueError("source identity does not match retained snapshot")
            if candidate["implementation"] == experiment.SOURCE_IMPLEMENTATION:
                if (identity.get("entry_point") != candidate["source"]["entry_point"] or
                        Path(execution_paths.get(label, "")).name != f"{label}.py"):
                    raise ValueError("executed source identity does not match snapshot")
                captured = raw.get("local_compiler_capture", {}).get(label)
                if captured:
                    if (compiler[label].get("metadata") != captured.get("metadata") or
                            (captured.get("ptx_text") is not None and
                             (output / f"{label}.ptx").read_text() != captured["ptx_text"]) or
                            (captured.get("sass_text") is not None and
                             (output / f"{label}.sass").read_text() != captured["sass_text"])):
                        raise ValueError("local compiler capture does not replay")
    if compiler["ptx_comparison"]["status"] == "CAPTURED":
        saved = json.loads((output / "ptx_comparison/comparison.json").read_text())
        recorded = compiler["ptx_comparison"]
        a_normalized = ptx.normalize_ptx((output / "A.ptx").read_text())
        b_normalized = ptx.normalize_ptx((output / "B.ptx").read_text())
        normalized_diff = ptx.bounded_diff(a_normalized, b_normalized, PTX_DIFF_LINES)[0]
        if (saved["baseline"]["file"]["sha256"] != _digest(output / "A.ptx") or
                saved["candidate"]["file"]["sha256"] != _digest(output / "B.ptx") or
                recorded["A_ptx_sha256"] != _digest(output / "A.ptx") or
                recorded["B_ptx_sha256"] != _digest(output / "B.ptx") or
                recorded["normalized_equal"] != saved["comparison"]["normalized"]["identical"] or
                recorded["normalized_changed_lines"] != saved["comparison"]["normalized"]["changed_lines"] or
                recorded["normalized_A_sha256"] != saved["comparison"]["normalized"]["baseline_sha256"] or
                recorded["normalized_B_sha256"] != saved["comparison"]["normalized"]["candidate_sha256"] or
                recorded["kernel"] != saved["comparison"]["kernel"] or
                saved["comparison"]["normalized"]["baseline_sha256"] !=
                hashlib.sha256(a_normalized.encode()).hexdigest() or
                saved["comparison"]["normalized"]["candidate_sha256"] !=
                hashlib.sha256(b_normalized.encode()).hexdigest() or
                (output / "ptx_comparison/ptx/baseline.normalized.ptx").read_text() != a_normalized or
                (output / "ptx_comparison/ptx/candidate.normalized.ptx").read_text() != b_normalized or
                (output / "ptx_comparison/ptx/normalized.diff").read_text() != normalized_diff or
                (output / "ptx_comparison/report.md").read_text() != ptx.render_comparison(saved)):
            raise ValueError("PTX comparison does not match captured artifacts")
    return {"status": "PASS", "decision": decision["decision"],
            "rounds": decision.get("rounds", 0), "files": len(actual)}


def run_pair(spec: Mapping[str, Any], output: Path, *, command: Sequence[str] | None = None,
             allow_trusted_local_source: bool = False,
             diagnostic_aa_placement: str | None = None,
             runner=run_worker, compiler_capture=capture_compiler) -> dict[str, Any]:
    validated = validate_spec(spec)
    if diagnostic_aa_placement is not None and (
            diagnostic_aa_placement not in ("before", "after") or validated["schema_version"] != 3):
        raise ValueError("diagnostic A/A placement requires a v3 pair spec and before/after")
    has_local = _has_local(validated)
    if has_local and not allow_trusted_local_source:
        raise ValueError("trusted local source requires --allow-trusted-local-source; this is not a sandbox")
    output = output.resolve()
    if output.exists():
        raise ValueError("output path already exists")
    output.mkdir(parents=True)
    public_spec = copy.deepcopy(validated)
    runtime_spec = copy.deepcopy(validated)
    if has_local:
        for label in LABELS:
            candidate = validated["kernels"][label]
            if candidate["implementation"] != experiment.SOURCE_IMPLEMENTATION:
                continue
            single = {key: value for key, value in validated.items()
                      if key not in ("schema_version", "kernels")}
            single.update(baseline={"implementation": "pytorch_matmul"}, candidate=candidate)
            preserved = experiment.snapshot_candidate_source(single, output / "sources" / f"{label}.py")
            runtime_spec["kernels"][label]["source"] = preserved["candidate"]["source"]
            public_spec["kernels"][label]["source"] = {
                **preserved["candidate"]["source"], "path": f"sources/{label}.py"}
        runtime_spec["_allow_trusted_local_source"] = True
    if diagnostic_aa_placement is not None:
        runtime_spec["_diagnostic_aa_placement"] = diagnostic_aa_placement
    _write_json(output / "experiment.json", public_spec)
    source = Path(__file__).with_name("triton_fc1.py")
    (output / "candidate_source.py").write_bytes(source.read_bytes())
    raw = (runner(public_spec, runtime_spec=runtime_spec)
           if has_local or diagnostic_aa_placement is not None else runner(public_spec))
    _write_json(output / "raw_result.json", raw)
    correctness = raw.get("correctness", {"passed": None})
    rows = timing_rows(raw)
    (output / "timings.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows), encoding="utf-8")
    decision = (summarize_clock_aware(correctness, rows, raw.get("failure"), raw.get("benchmark"))
                if validated["schema_version"] == 3 else
                summarize(correctness, rows, raw.get("failure"),
                          raw.get("benchmark", {}).get("unranked_timing_controls")))
    compiler = compiler_capture(public_spec, output, raw) if raw.get("status") == "success" and has_local else (
        compiler_capture(public_spec, output) if raw.get("status") == "success" else {
        "A": {"ptx": "UNAVAILABLE", "sass": "UNAVAILABLE"},
        "B": {"ptx": "UNAVAILABLE", "sass": "UNAVAILABLE"},
        "ptx_comparison": {"status": "UNAVAILABLE"}})
    _write_json(output / "correctness.json", correctness)
    _write_json(output / "comparison.json", decision)
    _write_json(output / "compiler.json", compiler)
    try:
        driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                capture_output=True, text=True, timeout=3)
        driver_version = driver.stdout.strip().splitlines()[0] if driver.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired, IndexError):
        driver_version = None
    if validated["schema_version"] == 3:
        if diagnostic_aa_placement is None:
            measurement = ("separate compile and warmup; A/A then alternating A/B blocks; "
                           "CUDA-event, perf_counter, optional MONOTONIC_RAW clocks")
        else:
            measurement = ("separate compile and warmup; "
                           f"A/A {diagnostic_aa_placement} each alternating A/B pair; "
                           "CUDA-event, perf_counter, optional MONOTONIC_RAW clocks")
    else:
        measurement = "separate compile and warmup; alternating A/B blocks with CUDA-event and host clocks"
    provenance = {
        "command": list(command or []), "python": platform.python_version(),
        "platform": platform.platform(), "driver_version": driver_version,
        "git": raw.get("git"), "environment": raw.get("environment"),
        "candidate_source": "matmul_inspector/triton_fc1.py",
        "candidate_source_sha256": _digest(output / "candidate_source.py"),
        "trusted_local_source_opt_in": True if has_local else None,
        "source_identities": {
            label: {"implementation": public_spec["kernels"][label]["implementation"],
                    "path": (public_spec["kernels"][label]["source"]["path"]
                             if public_spec["kernels"][label]["implementation"] == experiment.SOURCE_IMPLEMENTATION
                             else "candidate_source.py"),
                    "entry_point": (public_spec["kernels"][label]["source"]["entry_point"]
                                    if public_spec["kernels"][label]["implementation"] == experiment.SOURCE_IMPLEMENTATION
                                    else "fc1_kernel"),
                    "sha256": (public_spec["kernels"][label]["source"]["sha256"]
                               if public_spec["kernels"][label]["implementation"] == experiment.SOURCE_IMPLEMENTATION
                               else _digest(output / "candidate_source.py"))}
            for label in LABELS} if has_local else None,
        "measurement": measurement,
        **({"diagnostic_aa_placement": diagnostic_aa_placement}
           if diagnostic_aa_placement is not None else {})}
    _write_json(output / "provenance.json", provenance)
    (output / "report.md").write_text(
        render_report(validated, correctness, decision, compiler, provenance), encoding="utf-8")
    _manifest(output)
    verify_bundle(output)
    return decision


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="matmul-inspector-python experiment pair")
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--allow-trusted-local-source", action="store_true",
                        help="execute explicitly selected local Python source; this is not a sandbox")
    parser.add_argument("--diagnostic-aa-placement", choices=("before", "after"),
                        help="v3 diagnostic: place the same A/A blocks before or after each A/B pair")
    args = parser.parse_args(arguments)
    if args.verify:
        if args.spec or args.output or args.allow_trusted_local_source or args.diagnostic_aa_placement:
            parser.error("--verify cannot be combined with run options")
        try:
            print(json.dumps(verify_bundle(args.verify), sort_keys=True))
            return 0
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            print(f"verification failed: {error}", file=sys.stderr)
            return 1
    if not args.spec or not args.output:
        parser.error("--spec and --output are required")
    try:
        spec = json.loads(args.spec.read_text(encoding="utf-8"))
        command = ["matmul-inspector-python", "experiment", "pair", "--spec", str(args.spec),
                   "--output", str(args.output)]
        if args.allow_trusted_local_source:
            command.append("--allow-trusted-local-source")
        if args.diagnostic_aa_placement:
            command.extend(["--diagnostic-aa-placement", args.diagnostic_aa_placement])
        decision = run_pair(spec, args.output, command=command,
                            allow_trusted_local_source=args.allow_trusted_local_source,
                            diagnostic_aa_placement=args.diagnostic_aa_placement)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(f"paired comparison failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"output": str(args.output), **decision}, sort_keys=True))
    return 1 if decision["decision"] in ("REJECTED_INCORRECT", "FAILED") else 0
