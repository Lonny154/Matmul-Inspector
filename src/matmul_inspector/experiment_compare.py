"""One bounded, inspectable PyTorch/cuBLAS versus packaged Triton GEMM comparison.

This command adapts the established experiment worker. It does not execute a
candidate supplied as a Python path and does not change the worker's timing loop.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import sys
from typing import Any, Mapping, Sequence

from . import experiment, ptx, sass


SCHEMA_VERSION = 1
FILES = ("experiment.json", "correctness.json", "timings.jsonl",
         "comparison.json", "report.md", "provenance.json", "compiler.json",
         "candidate_source.py")
MEASUREMENT_FIELDS = ("cuda_event_elapsed_ms_per_operation",
                      "cuda_event_elapsed_ms_per_block", "host_submission_ms_per_block",
                      "host_wall_ms_per_block", "cuda_event_minus_host_wall_ms_per_block")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def validate_spec(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Require an exact, finite, bounded v1 spec with registered code only."""
    if not isinstance(raw, dict) or type(raw.get("schema_version")) is not int or (
            raw["schema_version"] != SCHEMA_VERSION):
        raise ValueError("schema_version must be 1")
    expected = {"schema_version", *experiment.DEFAULT_CONFIG}
    if set(raw) != expected:
        raise ValueError("spec must contain exactly the v1 fields")
    candidate = raw.get("candidate")
    if not isinstance(candidate, dict) or candidate.get("implementation") != "triton_fc1":
        raise ValueError("v1 accepts only the packaged triton_fc1 candidate")
    config = experiment.validate_config({key: value for key, value in raw.items()
                                         if key != "schema_version"})
    if config["benchmark"]["mode"] != "allocating":
        raise ValueError("v1 accepts only eager allocating mode")
    if set(config["workload"]) != {"m", "n", "k", "dtype"} or (
            set(config["correctness"]) != {"atol", "rtol"} or
            set(config["benchmark"]) != {"mode", "warmups", "iterations", "rounds"} or
            set(config["baseline"]) != {"implementation"}):
        raise ValueError("nested spec fields must match the v1 schema")
    if any(not math.isfinite(config["correctness"][name]) for name in ("atol", "rtol")):
        raise ValueError("correctness tolerances must be finite")
    if not math.isfinite(config["timeout_seconds"]):
        raise ValueError("timeout_seconds must be finite")
    if any(config["workload"][name] > 8192 for name in ("m", "n", "k")) or (
            config["workload"]["m"] * config["workload"]["n"] > 16_777_216):
        raise ValueError("workload exceeds the v1 size limit")
    bench = config["benchmark"]
    if (bench["warmups"] > 100 or bench["iterations"] > 500 or
            not 5 <= bench["rounds"] <= 20 or config["timeout_seconds"] > 600):
        raise ValueError("benchmark exceeds the v1 measurement limits")
    return {"schema_version": SCHEMA_VERSION, **config}


def _timing_rows(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    benchmark = result.get("benchmark")
    if not isinstance(benchmark, dict):
        return []
    rows = []
    for round_row in benchmark["rounds"]:
        for implementation in round_row["order"]:
            rows.append({"round": round_row["round"], "order": round_row["order"],
                         "implementation": implementation, "iterations": benchmark["iterations_per_measurement"],
                         **{field: round_row[f"{implementation}_{field}"]
                            for field in MEASUREMENT_FIELDS}})
    return rows


def summarize(correctness: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
              failure: Mapping[str, Any] | None = None,
              boundary_check: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Only an all-round separated comparison supports a direction claim."""
    result: dict[str, Any] = {"schema_version": SCHEMA_VERSION,
                              "decision": "INCONCLUSIVE", "speedup_baseline_over_candidate": None,
                              "reason": "correctness or comparable timing is unavailable",
                              "timing_basis": "alternating eager CUDA-event milliseconds per operation"}
    if failure and failure.get("stage") == "correctness":
        result.update(decision="REJECTED_INCORRECT", reason="candidate failed numerical correctness")
        return result
    if failure:
        result.update(decision="FAILED", reason=f"experiment failed at {failure.get('stage', 'unknown')}")
        return result
    if correctness.get("passed") is not True:
        return result
    by_round: dict[int, dict[str, float]] = {}
    for row in rows:
        value = row.get("cuda_event_elapsed_ms_per_operation")
        if (type(value) not in (float, int) or not math.isfinite(value) or value <= 0 or
                row.get("implementation") not in ("baseline", "candidate")):
            return result
        bucket = by_round.setdefault(row["round"], {})
        if row["implementation"] in bucket:
            return result
        bucket[row["implementation"]] = float(value)
    if len(by_round) < 5 or any(set(values) != {"baseline", "candidate"}
                               for values in by_round.values()):
        return result
    baseline = [values["baseline"] for _, values in sorted(by_round.items())]
    candidate = [values["candidate"] for _, values in sorted(by_round.items())]
    base_median = statistics.median(baseline)
    candidate_median = statistics.median(candidate)
    result.update(
        rounds=len(by_round), baseline_median_ms=base_median,
        candidate_median_ms=candidate_median,
        baseline_range_ms=[min(baseline), max(baseline)],
        candidate_range_ms=[min(candidate), max(candidate)],
        baseline_cv=(statistics.stdev(baseline) / statistics.fmean(baseline)),
        candidate_cv=(statistics.stdev(candidate) / statistics.fmean(candidate)),
        paired_baseline_minus_candidate_ms=[a-b for a, b in zip(baseline, candidate)],
        speedup_baseline_over_candidate=base_median / candidate_median,
    )
    if boundary_check and any(
            item.get("cuda_event_not_greater_than_synchronized_host_wall") is False
            for item in boundary_check.get("measurements", {}).values()):
        result.update(decision="INCONCLUSIVE", speedup_baseline_over_candidate=None,
                      reason="unranked timing controls found a CUDA-event/host-wall discrepancy")
        return result
    if max(candidate) < min(baseline):
        result.update(decision="CANDIDATE_FASTER",
                      reason="all candidate round values are below all baseline round values")
    elif max(baseline) < min(candidate):
        result.update(decision="BASELINE_FASTER",
                      reason="all baseline round values are below all candidate round values")
    else:
        result.update(decision="INCONCLUSIVE",
                      reason="round ranges overlap; measured ratio is not a performance-win claim")
    return result


def _compiler_capture(spec: Mapping[str, Any], output: Path) -> dict[str, Any]:
    """Capture the registered kernel's PTX/SASS after timing, outside timed rounds."""
    result: dict[str, Any] = {"status": "UNAVAILABLE", "ptx": "UNAVAILABLE",
                              "sass": "UNAVAILABLE", "baseline_compiler": "UNAVAILABLE"}
    try:
        import torch
        import triton
        from . import triton_fc1
        w = spec["workload"]
        p = spec["candidate"]["parameters"]
        a = torch.empty((w["m"], w["k"]), device="cuda", dtype=torch.float16)
        b = torch.empty((w["k"], w["n"]), device="cuda", dtype=torch.float16)
        c = torch.empty((w["m"], w["n"]), device="cuda", dtype=torch.float16)
        grid = (triton.cdiv(w["m"], p["block_m"]) * triton.cdiv(w["n"], p["block_n"]),)
        compiled = triton_fc1.fc1_kernel[grid](
            a, b, c, w["m"], w["n"], w["k"], a.stride(0), a.stride(1),
            b.stride(0), b.stride(1), c.stride(0), c.stride(1),
            BLOCK_M=p["block_m"], BLOCK_N=p["block_n"], BLOCK_K=p["block_k"],
            GROUP_SIZE_M=p["group_size_m"], num_warps=p["num_warps"],
            num_stages=p["num_stages"])
        torch.cuda.synchronize()
        metadata = compiled.metadata
        result["status"] = "CAPTURED"
        result["kernel_name"] = compiled.name
        result["metadata"] = {name: getattr(metadata, name, None)
                              for name in ("num_warps", "num_stages", "shared", "global_scratch_size",
                                           "n_regs", "n_spills")}
        for name in ("n_regs", "n_spills"):
            if result["metadata"][name] is None:
                result["metadata"][name] = getattr(compiled, name, None)
        assembly = compiled.asm
        if isinstance(assembly.get("ptx"), str):
            (output / "candidate.ptx").write_text(assembly["ptx"], encoding="utf-8")
            result["ptx"] = "CAPTURED"
            module = ptx.parse_module(assembly["ptx"])
            selected = [item for item in module.symbols if item.kind == "entry"]
            result["ptx_summary"] = ptx.module_summary(module, selected)
        if isinstance(assembly.get("sass"), str):
            (output / "candidate.sass").write_text(assembly["sass"], encoding="utf-8")
            result["sass"] = "CAPTURED"
            module = sass.parse_module(assembly["sass"])
            result["sass_summary"] = sass.module_summary(module, module.functions)
        else:
            result["sass_reason"] = "Triton did not expose decoded SASS; nvdisasm was not required"
    except Exception as error:
        result["status"] = "PARTIAL" if result["ptx"] == "CAPTURED" else "UNAVAILABLE"
        result["capture_error_type"] = type(error).__name__
    return result


def _report(spec: Mapping[str, Any], correctness: Mapping[str, Any],
            comparison: Mapping[str, Any], compiler: Mapping[str, Any]) -> str:
    w = spec["workload"]
    lines = ["# GEMM comparison", "", f"Shape: M={w['m']}, N={w['n']}, K={w['k']}; {w['dtype']}; eager CUDA.",
             "Baseline: PyTorch/cuBLAS `torch.matmul`.",
             "Candidate: packaged Triton `triton_fc1.fc1_kernel`.", "",
             f"Correctness: {'PASS' if correctness.get('passed') else 'FAIL/UNAVAILABLE'}",
             f"Decision: **{comparison['decision']}** — {comparison['reason']}", ""]
    if comparison.get("baseline_median_ms") is not None:
        lines += [f"Baseline median: {comparison['baseline_median_ms']:.6f} ms/op.",
                  f"Candidate median: {comparison['candidate_median_ms']:.6f} ms/op.",
                  f"Baseline CV: {comparison['baseline_cv']:.2%}; candidate CV: {comparison['candidate_cv']:.2%}.", ""]
        if comparison["speedup_baseline_over_candidate"] is not None:
            lines.insert(-1, f"Measured baseline/candidate ratio: {comparison['speedup_baseline_over_candidate']:.4f}x.")
    lines += ["CUDA-event intervals include dispatch and possible device idle gaps.",
              "First synchronized invocation is reported separately in the preserved raw run.",
              f"Compiler: PTX {compiler['ptx']}; SASS {compiler['sass']}; cuBLAS machine code unavailable.",
              "Raw round timings and source hashes are in this bundle.", ""]
    return "\n".join(lines)


def _manifest(output: Path) -> None:
    paths = sorted(p for p in output.rglob("*") if p.is_file() and p.name != "hash_manifest.json")
    _write_json(output / "hash_manifest.json",
                {str(path.relative_to(output)): _digest(path) for path in paths})


def verify_bundle(output: Path) -> dict[str, Any]:
    manifest = json.loads((output / "hash_manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or any(
            not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts
            for name in manifest):
        raise ValueError("artifact manifest contains an invalid path")
    actual = {str(path.relative_to(output)) for path in output.rglob("*")
              if path.is_file() and path.name != "hash_manifest.json"}
    if set(manifest) != actual or any(_digest(output / name) != digest for name, digest in manifest.items()):
        raise ValueError("artifact manifest does not match bundle files")
    if not set(FILES) <= actual:
        raise ValueError("required bundle file is missing")
    spec = validate_spec(json.loads((output / "experiment.json").read_text()))
    correctness = json.loads((output / "correctness.json").read_text())
    rows = [json.loads(line) for line in (output / "timings.jsonl").read_text().splitlines() if line]
    comparison = json.loads((output / "comparison.json").read_text())
    compiler = json.loads((output / "compiler.json").read_text())
    provenance = json.loads((output / "provenance.json").read_text())
    raw_run = output / provenance["raw_run_result"]
    if (Path(provenance["raw_run_result"]).is_absolute() or
            ".." in Path(provenance["raw_run_result"]).parts):
        raise ValueError("raw result path is outside the bundle")
    raw = json.loads(raw_run.read_text())
    if (raw.get("config") != {key: value for key, value in spec.items() if key != "schema_version"} or
            raw.get("correctness", {"passed": None, "status": "UNAVAILABLE"}) != correctness or
            rows != _timing_rows(raw) or
            comparison != summarize(correctness, rows, raw.get("failure"),
                                    raw.get("benchmark", {}).get("timing_boundary_check")) or
            (output / "report.md").read_text() != _report(spec, correctness, comparison, compiler) or
            (compiler.get("ptx") == "CAPTURED") != ("candidate.ptx" in actual) or
            (compiler.get("sass") == "CAPTURED") != ("candidate.sass" in actual) or
            provenance["candidate_source_sha256"] != _digest(output / "candidate_source.py")):
        raise ValueError("bundle contents do not replay from the preserved raw result")
    return {"status": "PASS", "files": len(actual), "decision": comparison["decision"],
            "rounds": comparison.get("rounds", 0), "schema_version": spec["schema_version"]}


def run_compare(spec: Mapping[str, Any], output: Path,
                *, command: Sequence[str] | None = None,
                runner=experiment.run_experiment,
                compiler_capture=_compiler_capture) -> dict[str, Any]:
    validated = validate_spec(spec)
    output = output.resolve()
    if output.exists():
        raise ValueError("output path already exists")
    output.mkdir(parents=True)
    _write_json(output / "experiment.json", validated)
    source = Path(__file__).with_name("triton_fc1.py")
    (output / "candidate_source.py").write_bytes(source.read_bytes())
    raw = runner({key: value for key, value in validated.items() if key != "schema_version"},
                 output_root=output / "raw")
    correctness = raw.get("correctness", {"passed": None, "status": "UNAVAILABLE"})
    rows = _timing_rows(raw)
    (output / "timings.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8")
    comparison = summarize(correctness, rows, raw.get("failure"),
                           raw.get("benchmark", {}).get("timing_boundary_check"))
    compiler = (compiler_capture(validated, output) if raw.get("status") == "success"
                else {"status": "UNAVAILABLE", "ptx": "UNAVAILABLE", "sass": "UNAVAILABLE",
                      "baseline_compiler": "UNAVAILABLE", "reason": "no passing timed run"})
    _write_json(output / "correctness.json", correctness)
    _write_json(output / "comparison.json", comparison)
    _write_json(output / "compiler.json", compiler)
    (output / "report.md").write_text(_report(validated, correctness, comparison, compiler), encoding="utf-8")
    raw_path = Path(raw["run_directory"]) / "result.json"
    try:
        driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                capture_output=True, text=True, timeout=3)
        driver_version = driver.stdout.strip().splitlines()[0] if driver.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired, IndexError):
        driver_version = None
    _write_json(output / "provenance.json", {
        "schema_version": SCHEMA_VERSION, "command": list(command or []),
        "python": platform.python_version(), "platform": platform.platform(),
        "git": raw.get("git"), "environment": raw.get("environment"),
        "driver_version": driver_version,
        "candidate_source": "matmul_inspector/triton_fc1.py",
        "candidate_source_sha256": _digest(output / "candidate_source.py"),
        "raw_run_result": str(raw_path.relative_to(output)),
        "timing_separation": "first-invocation host setup is separate from repeated CUDA-event and host block timing",
    })
    _manifest(output)
    verify_bundle(output)
    return comparison


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="matmul-inspector-python experiment compare")
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args(arguments)
    if args.verify:
        if args.spec or args.output:
            parser.error("--verify cannot be combined with --spec or --output")
        try:
            print(json.dumps(verify_bundle(args.verify), sort_keys=True))
            return 0
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            print(f"verification failed: {error}", file=sys.stderr)
            return 1
    if not args.spec or not args.output:
        parser.error("--spec and --output are required")
    try:
        source = json.loads(args.spec.read_text(encoding="utf-8"))
        comparison = run_compare(source, args.output,
                                 command=["matmul-inspector-python", "experiment", "compare",
                                          "--spec", str(args.spec), "--output", str(args.output)])
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"comparison failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"output": str(args.output), **comparison}, sort_keys=True))
    return 1 if comparison["decision"] in ("REJECTED_INCORRECT", "FAILED") else 0
