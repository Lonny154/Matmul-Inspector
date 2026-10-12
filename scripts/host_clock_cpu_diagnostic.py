#!/usr/bin/env python3
"""Compare Linux monotonic clocks around fixed CPU work and sleep."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time
from typing import Any, Sequence
import uuid


DURATIONS_MS = (10, 100, 500)
REPETITIONS = 10


def _command(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, text=True, capture_output=True, check=False)


def _record(*args: str) -> dict[str, Any]:
    result = _command(*args)
    return {"command": " ".join(args), "returncode": result.returncode,
            "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}


def _git(root: Path, *args: str) -> str | None:
    result = _command("git", *args, cwd=root)
    return result.stdout.strip() if result.returncode == 0 else None


def _read(path: Path) -> dict[str, Any]:
    try:
        return {"path": str(path), "available": True,
                "value": path.read_text(encoding="utf-8").strip()}
    except OSError as error:
        return {"path": str(path), "available": False,
                "error": f"{type(error).__name__}: {error}"}


def _clock_info(value: Any) -> dict[str, Any]:
    return {"implementation": value.implementation, "monotonic": value.monotonic,
            "adjustable": value.adjustable, "resolution_seconds": value.resolution}


def _stats(values: Sequence[float]) -> dict[str, float | int]:
    median = statistics.median(values)
    return {"count": len(values), "minimum": min(values), "median": median,
            "maximum": max(values), "mean": statistics.fmean(values),
            "standard_deviation": statistics.stdev(values) if len(values) > 1 else 0.0,
            "median_absolute_deviation": statistics.median(
                abs(value - median) for value in values)}


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def _cpu_work(iterations: int) -> int:
    value = 0x12345678
    for index in range(iterations):
        value = ((value * 1664525 + 1013904223) ^ index) & 0xFFFFFFFF
    return value


def _calibrate_cpu_iterations() -> dict[int, int]:
    iterations = 10_000
    while True:
        started = time.perf_counter_ns()
        _cpu_work(iterations)
        elapsed_ns = time.perf_counter_ns() - started
        if elapsed_ns >= 50_000_000:
            break
        iterations *= 2
    iterations_per_ns = iterations / elapsed_ns
    return {duration: max(1, round(iterations_per_ns * duration * 1_000_000))
            for duration in DURATIONS_MS}


def _schedule() -> list[dict[str, Any]]:
    rotations = ((10, 100, 500), (100, 500, 10), (500, 10, 100))
    rows = []
    block_id = 0
    for repetition in range(1, REPETITIONS + 1):
        durations = rotations[(repetition - 1) % len(rotations)]
        kinds = ("cpu", "sleep") if repetition % 2 else ("sleep", "cpu")
        for duration_position, duration in enumerate(durations, 1):
            for kind_position, kind in enumerate(kinds, 1):
                block_id += 1
                rows.append({"block_id": block_id, "repetition": repetition,
                             "duration_position": duration_position,
                             "kind_position": kind_position, "work_kind": kind,
                             "target_duration_ms": duration})
    return rows


def run(output_root: Path) -> Path:
    root = Path(__file__).resolve().parents[1]
    raw_clock = getattr(time, "CLOCK_MONOTONIC_RAW", None)
    if raw_clock is None or not hasattr(time, "clock_gettime_ns"):
        raise RuntimeError(
            "CLOCK_MONOTONIC_RAW is unavailable; monotonic_ns was not substituted")
    time.clock_gettime_ns(raw_clock)

    compute_check = _record(
        "nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits")
    if compute_check["returncode"] != 0:
        raise RuntimeError("could not check competing GPU compute processes: "
                           + (compute_check["stderr"] or "unknown nvidia-smi error"))
    if compute_check["stdout"]:
        raise RuntimeError("competing GPU compute work is active; no diagnostic was launched: "
                           + compute_check["stdout"].replace("\n", "; "))

    created = datetime.now(timezone.utc)
    run_id = f"{created.strftime('%Y%m%dT%H%M%S.%fZ')}-{uuid.uuid4().hex[:8]}"
    destination = output_root.resolve() / run_id
    destination.mkdir(parents=True, exist_ok=False)

    kernel_release = platform.release()
    proc_version = _read(Path("/proc/version"))
    clocksource_root = Path("/sys/devices/system/clocksource/clocksource0")
    timedatectl = _record(
        "timedatectl", "show", "--property=NTPSynchronized", "--property=NTP",
        "--property=CanNTP", "--property=TimeUSec", "--property=RTCTimeUSec")
    command_availability = {
        command: _record("sh", "-c", f"command -v {command}")
        for command in ("adjtimex", "chronyc", "ntpq")}
    provenance = {
        "created_at_utc": created.isoformat(),
        "source_revision": _git(root, "rev-parse", "HEAD"),
        "source_branch": _git(root, "branch", "--show-current"),
        "source_status_porcelain": _git(root, "status", "--porcelain"),
        "os": platform.platform(), "kernel_release": kernel_release,
        "proc_version": proc_version, "python": platform.python_version(),
        "wsl": {"detected": "microsoft" in kernel_release.lower()
                or "microsoft" in str(proc_version.get("value", "")).lower(),
                "interop_environment_present": "WSL_INTEROP" in os.environ},
        "clocks": {
            "perf_counter": _clock_info(time.get_clock_info("perf_counter")),
            "clock_monotonic": {"clock_id": time.CLOCK_MONOTONIC,
                                "resolution_seconds": time.clock_getres(time.CLOCK_MONOTONIC)},
            "clock_monotonic_raw": {"supported": True, "clock_id": raw_clock,
                                    "resolution_seconds": time.clock_getres(raw_clock)}},
        "clocksource": {
            "current": _read(clocksource_root / "current_clocksource"),
            "available": _read(clocksource_root / "available_clocksource")},
        "time_adjustment_inspection": {
            "timedatectl": timedatectl, "command_availability": command_availability,
            "note": "Read-only inspection only. No setting was changed."},
        "competing_compute_process_check": compute_check,
    }
    cpu_iterations = _calibrate_cpu_iterations()
    config = {
        "purpose": "standalone CPU-only monotonic clock-rate diagnostic",
        "gpu_work": "none", "session_budget_impact": "none",
        "work_kinds": ["fixed deterministic Python integer loop", "time.sleep"],
        "target_durations_ms": list(DURATIONS_MS),
        "repetitions_per_kind_duration": REPETITIONS,
        "cpu_iterations_by_target_duration": cpu_iterations,
        "cpu_iteration_calibration": (
            "one pre-measurement calibration using perf_counter; measured CPU blocks use "
            "fixed iteration counts and do not stop on a clock threshold"),
        "schedule": "rotated durations and alternating CPU/sleep order",
        "timestamp_read_order": [
            "perf_start_ns", "monotonic_start_ns", "raw_start_ns",
            "perf_after_start_reads_ns", "CPU work or sleep", "perf_end_ns",
            "monotonic_end_ns", "raw_end_ns", "perf_after_end_reads_ns"],
        "comparison_rule": (
            "Only elapsed durations are compared. Absolute timestamps from different clocks "
            "are never compared. No correction factor is applied."),
    }
    _write_json(destination / "config.json", config)
    _write_json(destination / "provenance.json", provenance)

    checksum = 0
    for iterations in cpu_iterations.values():
        checksum ^= _cpu_work(max(1, iterations // 10))
    time.sleep(0.01)

    def measure(entry: dict[str, Any]) -> dict[str, Any]:
        perf_start = time.perf_counter_ns()
        monotonic_start = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
        raw_start = time.clock_gettime_ns(raw_clock)
        perf_after_start = time.perf_counter_ns()
        if entry["work_kind"] == "cpu":
            iterations = cpu_iterations[entry["target_duration_ms"]]
            work_result = _cpu_work(iterations)
        else:
            iterations = None
            work_result = None
            time.sleep(entry["target_duration_ms"] / 1000.0)
        perf_end = time.perf_counter_ns()
        monotonic_end = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
        raw_end = time.clock_gettime_ns(raw_clock)
        perf_after_end = time.perf_counter_ns()
        perf_ms = (perf_end - perf_start) / 1e6
        monotonic_ms = (monotonic_end - monotonic_start) / 1e6
        raw_ms = (raw_end - raw_start) / 1e6
        return {**entry, "cpu_iterations": iterations, "work_result": work_result,
                "perf_start_ns": perf_start, "monotonic_start_ns": monotonic_start,
                "raw_start_ns": raw_start, "perf_after_start_reads_ns": perf_after_start,
                "perf_end_ns": perf_end, "monotonic_end_ns": monotonic_end,
                "raw_end_ns": raw_end, "perf_after_end_reads_ns": perf_after_end,
                "start_clock_read_span_ns": perf_after_start - perf_start,
                "end_clock_read_span_ns": perf_after_end - perf_end,
                "perf_counter_ms": perf_ms, "clock_monotonic_ms": monotonic_ms,
                "clock_monotonic_raw_ms": raw_ms,
                "monotonic_minus_perf_ms": monotonic_ms - perf_ms,
                "monotonic_over_perf_ratio": monotonic_ms / perf_ms,
                "raw_minus_perf_ms": raw_ms - perf_ms,
                "raw_over_perf_ratio": raw_ms / perf_ms,
                "raw_minus_monotonic_ms": raw_ms - monotonic_ms,
                "raw_over_monotonic_ratio": raw_ms / monotonic_ms}

    schedule = _schedule()
    rows = [measure(entry) for entry in schedule]
    checksum ^= sum(int(row["work_result"] or 0) for row in rows) & 0xFFFFFFFF
    with (destination / "raw_measurements.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _write_json(destination / "raw_measurements.json", rows)

    groups = {}
    fields = ("perf_counter_ms", "clock_monotonic_ms", "clock_monotonic_raw_ms",
              "monotonic_minus_perf_ms", "monotonic_over_perf_ratio",
              "raw_minus_perf_ms", "raw_over_perf_ratio",
              "raw_minus_monotonic_ms", "raw_over_monotonic_ratio",
              "start_clock_read_span_ns", "end_clock_read_span_ns")
    for kind in ("cpu", "sleep"):
        groups[kind] = {}
        for duration in DURATIONS_MS:
            selected = [row for row in rows if row["work_kind"] == kind
                        and row["target_duration_ms"] == duration]
            groups[kind][str(duration)] = {
                field: _stats([row[field] for row in selected]) for field in fields}
    summary = {
        "groups": groups,
        "all_raw_over_perf_ratio": _stats([row["raw_over_perf_ratio"] for row in rows]),
        "all_raw_over_monotonic_ratio": _stats(
            [row["raw_over_monotonic_ratio"] for row in rows]),
        "all_monotonic_over_perf_ratio": _stats(
            [row["monotonic_over_perf_ratio"] for row in rows]),
        "checksum": checksum,
        "interpretation_note": (
            "Only elapsed durations are compared. Linux clock semantics make frequency "
            "adjustment relevant, but these measurements do not establish a cause or "
            "absolute accuracy. No correction is applied.")}
    _write_json(destination / "summary.json", summary)
    _write_json(destination / "result.json", {
        "schema_version": 1, "status": "success", "run_id": run_id,
        "run_directory": str(destination), "config": config,
        "provenance": provenance, "actual_execution_order": schedule,
        "summary": summary})
    return destination


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path,
                        default=Path("artifacts/host-clock-cpu-diagnostics"))
    args = parser.parse_args(arguments)
    try:
        destination = run(args.output_root)
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
