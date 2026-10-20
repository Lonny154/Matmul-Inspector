"""Bounded Split-2 phase diagnostic; leaves the paired benchmark unchanged."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
from typing import Any

from matmul_inspector import experiment_pair


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "examples/kernels/mi_research_2_split2.py"
SPEC = ROOT / "examples/mi_research_2_low_output_split2.json"
PRIOR_SOURCE = ROOT / "artifacts/mi-research-2/evidence-v1/low-output-split2/sources/B.py"
ITERATIONS = 100
ROUNDS = 8
WARMUPS = 30
PROBE_BLOCKS = 8


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def complete_order(round_index: int, session: int) -> list[str]:
    if round_index < 1 or session not in (1, 2):
        raise ValueError("invalid diagnostic round or session")
    return (["alloc", "reuse"] if (round_index + session) % 2 == 0
            else ["reuse", "alloc"])


def phase_order(round_index: int) -> list[str]:
    if round_index < 1:
        raise ValueError("round_index must be positive")
    return ["partial", "reduction"] if round_index % 2 else ["reduction", "partial"]


def group_order(session: int) -> list[str]:
    if session not in (1, 2):
        raise ValueError("session must be 1 or 2")
    order = ["conventional", "complete_pair", "phase_pair"]
    return order if session == 1 else list(reversed(order))


def summarize(raw: dict[str, Any]) -> dict[str, Any]:
    if raw.get("schema_version") != 1 or raw.get("iterations_per_block") != ITERATIONS:
        raise ValueError("invalid diagnostic raw schema or operation count")
    if len(raw.get("rounds", [])) != ROUNDS or len(raw.get("instrumented_probe", [])) != PROBE_BLOCKS:
        raise ValueError("incomplete diagnostic rounds or probe")
    if raw.get("correctness", {}).get("passed") is not True:
        raise ValueError("diagnostic numerical correctness did not pass")
    session = raw["session"]
    expected_groups = group_order(session)
    for index, row in enumerate(raw["rounds"], 1):
        if (row.get("round") != index or row.get("group_order") != expected_groups or
                row.get("complete_order") != complete_order(index, session) or
                row.get("phase_order") != phase_order(index)):
            raise ValueError("diagnostic order does not replay")
        for key in ("conventional", "alloc", "reuse", "partial", "reduction"):
            sample = row["measurements"][key]
            for field in ("cuda_event_elapsed_ms_per_block", "host_wall_ms_per_block",
                          "monotonic_raw_ms_per_block", "host_submission_ms_per_block"):
                if not isinstance(sample.get(field), (int, float)) or sample[field] <= 0:
                    raise ValueError("invalid diagnostic timing field")
    for index, probe in enumerate(raw["instrumented_probe"], 1):
        if probe.get("block") != index or len(probe.get("per_call", [])) != ITERATIONS:
            raise ValueError("instrumented probe operation count does not replay")
        if any(any(type(call.get(field)) is not int or call[field] < 0
                   for field in ("scratch_allocation_ns", "partial_launch_submission_ns",
                                 "reduction_launch_submission_ns"))
               for call in probe["per_call"]):
            raise ValueError("invalid instrumented host phase duration")
    outcome = {}
    for key in ("conventional", "alloc", "reuse", "partial", "reduction"):
        rows = [row["measurements"][key] for row in raw["rounds"]]
        outcome[key] = {
            "event_median_ms_per_call": statistics.median(
                row["cuda_event_elapsed_ms_per_block"] / ITERATIONS for row in rows),
            "event_range_ms_per_call": [
                min(row["cuda_event_elapsed_ms_per_block"] for row in rows) / ITERATIONS,
                max(row["cuda_event_elapsed_ms_per_block"] for row in rows) / ITERATIONS],
            "host_submission_median_ms_per_block": statistics.median(
                row["host_submission_ms_per_block"] for row in rows),
            "event_blocks_ms": [row["cuda_event_elapsed_ms_per_block"] for row in rows],
            "host_submission_blocks_ms": [row["host_submission_ms_per_block"] for row in rows],
        }
    outcome["alloc_minus_reuse_event_ms_per_call"] = [
        (row["measurements"]["alloc"]["cuda_event_elapsed_ms_per_block"] -
         row["measurements"]["reuse"]["cuda_event_elapsed_ms_per_block"]) / ITERATIONS
        for row in raw["rounds"]]
    return {"schema_version": 1, "session": session, "rounds": ROUNDS,
            "iterations_per_block": ITERATIONS, "measurements": outcome,
            "correctness_passed": raw["correctness"]["passed"],
            "graph_phase_probe": raw["graph_phase_probe"]}


def manifest(output: Path) -> None:
    write_json(output / "hash_manifest.json", {
        str(path.relative_to(output)): digest(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "hash_manifest.json"})


def verify(output: Path) -> dict[str, Any]:
    saved = json.loads((output / "hash_manifest.json").read_text())
    if not isinstance(saved, dict) or any(
            Path(name).is_absolute() or ".." in Path(name).parts for name in saved):
        raise ValueError("invalid manifest")
    actual = {str(path.relative_to(output)) for path in output.rglob("*")
              if path.is_file() and path.name != "hash_manifest.json"}
    if actual != set(saved) or any(digest(output / name) != value for name, value in saved.items()):
        raise ValueError("diagnostic manifest mismatch")
    raw = json.loads((output / "raw.json").read_text())
    if digest(output / "source.py") != raw["source_sha256"]:
        raise ValueError("source snapshot mismatch")
    if (digest(output / "experiment.json") != raw["spec_sha256"] or
            digest(output / "diagnostic_script.py") != raw["diagnostic_script_sha256"]):
        raise ValueError("diagnostic source or spec identity mismatch")
    if digest(output / "partial.ptx") != raw["compiler"]["partial"]["ptx_sha256"] or digest(
            output / "reduction.ptx") != raw["compiler"]["reduction"]["ptx_sha256"]:
        raise ValueError("compiler PTX identity mismatch")
    if summarize(raw) != json.loads((output / "summary.json").read_text()):
        raise ValueError("diagnostic summary does not replay")
    return {"status": "PASS", "session": raw["session"], "rounds": ROUNDS,
            "files": len(saved)}


def graph_phase_probe(torch, functions: dict[str, Any]) -> dict[str, Any]:
    """Graph mode isolates host submission from a 100-node GPU phase block."""
    results = {}
    for name in ("partial", "reduction"):
        try:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(ITERATIONS):
                    functions[name]()
            torch.cuda.synchronize()
            for _ in range(2):
                graph.replay()
            torch.cuda.synchronize()
            samples = []
            for _ in range(4):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                wall_start = time.perf_counter_ns()
                start.record()
                graph.replay()
                end.record()
                submission_end = time.perf_counter_ns()
                end.synchronize()
                wall_end = time.perf_counter_ns()
                samples.append({"cuda_event_ms_per_100_nodes": start.elapsed_time(end),
                                "host_submission_ms_per_replay": (submission_end - wall_start) / 1e6,
                                "host_wall_ms_per_replay": (wall_end - wall_start) / 1e6})
            results[name] = {"status": "CAPTURED", "nodes_per_replay": ITERATIONS,
                             "execution_mode": "CUDA Graph diagnostic; not eager paired timing",
                             "samples": samples}
        except Exception as error:
            results[name] = {"status": "UNAVAILABLE", "error_type": type(error).__name__}
    return results


def run(output: Path, session: int) -> dict[str, Any]:
    if output.exists():
        raise ValueError("output directory already exists")
    if session not in (1, 2):
        raise ValueError("session must be 1 or 2")
    source_sha = digest(SOURCE)
    if source_sha != digest(PRIOR_SOURCE):
        raise ValueError("MI-Research-2 source differs from the retained snapshot")
    validated = experiment_pair.validate_spec(json.loads(SPEC.read_text()))
    output.mkdir(parents=True)
    shutil.copyfile(SOURCE, output / "source.py")
    shutil.copyfile(SPEC, output / "experiment.json")
    shutil.copyfile(Path(__file__), output / "diagnostic_script.py")

    import torch
    import triton
    from matmul_inspector import triton_fc1

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    spec = importlib.util.spec_from_file_location("mi_research_2b_source", output / "source.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import retained Split-2 source")
    module = importlib.util.module_from_spec(spec)
    previous_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous_bytecode

    w = validated["workload"]
    parameters = validated["kernels"]["B"]["parameters"]
    props = torch.cuda.get_device_properties(0)
    output_a = torch.empty((w["m"], w["n"]), device="cuda", dtype=torch.float16)
    output_alloc = torch.empty_like(output_a)
    output_reuse = torch.empty_like(output_a)
    scratch_reuse = torch.empty((2, w["m"], w["n"]), device="cuda", dtype=torch.float32)
    scratch_phase = torch.empty_like(scratch_reuse)
    partial_grid = (triton.cdiv(w["m"], parameters["block_m"]) *
                    triton.cdiv(w["n"], parameters["block_n"]), 2)
    reduction_grid = (triton.cdiv(w["m"] * w["n"], module.REDUCTION_BLOCK),)
    conventional_grid = (partial_grid[0],)

    def partial(a, b, scratch):
        return module.split_k_partial[partial_grid](
            a, b, scratch, w["m"], w["n"], w["k"],
            a.stride(0), a.stride(1), b.stride(0), b.stride(1),
            BLOCK_M=parameters["block_m"], BLOCK_N=parameters["block_n"],
            BLOCK_K=parameters["block_k"], K_PART=w["k"] // 2,
            num_warps=parameters["num_warps"], num_stages=parameters["num_stages"])

    def reduction(scratch, out):
        return module.reduce_partials[reduction_grid](
            scratch, out, w["m"], w["n"], out.stride(0), out.stride(1),
            BLOCK=module.REDUCTION_BLOCK, num_warps=4)

    def conventional(a, b):
        return triton_fc1.fc1_kernel[conventional_grid](
            a, b, output_a, w["m"], w["n"], w["k"],
            a.stride(0), a.stride(1), b.stride(0), b.stride(1),
            output_a.stride(0), output_a.stride(1),
            BLOCK_M=parameters["block_m"], BLOCK_N=parameters["block_n"],
            BLOCK_K=parameters["block_k"], GROUP_SIZE_M=parameters["group_size_m"],
            num_warps=parameters["num_warps"], num_stages=parameters["num_stages"])

    def make_inputs(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        return (torch.randn((w["m"], w["k"]), device="cuda", dtype=torch.float16),
                torch.randn((w["k"], w["n"]), device="cuda", dtype=torch.float16))

    def reuse_complete(a, b):
        # Mirror the committed matmul_into body, including parameter reads and
        # grid construction. Only torch.empty is replaced by saved scratch.
        m, k = a.shape
        n = b.shape[1]
        block_m = parameters["block_m"]
        block_n = parameters["block_n"]
        block_k = parameters["block_k"]
        if k % (module.SPLIT_K * block_k):
            raise ValueError("K must be divisible by both the partition count and block_k")
        partials = scratch_reuse
        grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n), module.SPLIT_K)
        module.split_k_partial[grid](
            a, b, partials, m, n, k,
            a.stride(0), a.stride(1), b.stride(0), b.stride(1),
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
            K_PART=k // module.SPLIT_K,
            num_warps=parameters["num_warps"], num_stages=parameters["num_stages"],
        )
        return module.reduce_partials[(triton.cdiv(m * n, module.REDUCTION_BLOCK),)](
            partials, output_reuse, m, n, output_reuse.stride(0), output_reuse.stride(1),
            BLOCK=module.REDUCTION_BLOCK, num_warps=4,
        )

    correctness = []
    compiler_launches = None
    for seed in (0, 1, 2):
        a, b = make_inputs(seed)
        reference = torch.matmul(a, b)
        for out in (output_a, output_alloc, output_reuse):
            out.fill_(float("nan"))
        conventional_launch = conventional(a, b)
        module.matmul_into(a, b, output_alloc, parameters)
        reuse_complete(a, b)
        torch.cuda.synchronize()
        checks = {"conventional": experiment_pair.check_output(
                      torch, reference, output_a, validated["correctness"]),
                  "alloc": experiment_pair.check_output(
                      torch, reference, output_alloc, validated["correctness"]),
                  "reuse": experiment_pair.check_output(
                      torch, reference, output_reuse, validated["correctness"])}
        correctness.append({"seed": seed, "checks": checks})
        if not all(check["passed"] for check in checks.values()):
            raise RuntimeError(f"numerical correctness failed for seed {seed}")
        if seed == 0:
            partial_launch = partial(a, b, scratch_phase)
            reduction_launch = reduction(scratch_phase, output_reuse)
            torch.cuda.synchronize()
            compiler_launches = {"conventional": conventional_launch,
                                 "partial": partial_launch, "reduction": reduction_launch}
    assert compiler_launches is not None

    compiler = {}
    for name, launch in compiler_launches.items():
        capture = experiment_pair._compiled_capture(launch)
        if capture is None or capture["ptx_text"] is None:
            raise RuntimeError(f"{name} PTX unavailable")
        (output / f"{name}.ptx").write_text(capture.pop("ptx_text"))
        capture.pop("sass_text", None)
        capture["ptx_sha256"] = digest(output / f"{name}.ptx")
        compiler[name] = capture

    a, b = make_inputs(0)

    functions = {
        "conventional": lambda: conventional(a, b),
        "alloc": lambda: module.matmul_into(a, b, output_alloc, parameters),
        "reuse": lambda: reuse_complete(a, b),
        "partial": lambda: partial(a, b, scratch_phase),
        "reduction": lambda: reduction(scratch_phase, output_reuse),
    }
    for name in functions:
        for _ in range(WARMUPS):
            functions[name]()
        torch.cuda.synchronize()
    functions["partial"]()
    torch.cuda.synchronize()
    raw_clock = experiment_pair.raw_clock()
    if raw_clock is None:
        raise RuntimeError("CLOCK_MONOTONIC_RAW is required for this diagnostic")
    preflight = experiment_pair.host_clock_preflight(reader=raw_clock)
    measure = experiment_pair.experiment.CudaTritonBackend._measure
    rounds = []
    for index in range(1, ROUNDS + 1):
        measured = {}
        groups = group_order(session)
        for group in groups:
            if group == "conventional":
                measured["conventional"] = measure(
                    torch, functions["conventional"], ITERATIONS, raw_clock=raw_clock)
            else:
                order = complete_order(index, session) if group == "complete_pair" else phase_order(index)
                for label in order:
                    measured[label] = measure(torch, functions[label], ITERATIONS, raw_clock=raw_clock)
        rounds.append({"round": index, "group_order": groups,
                       "complete_order": complete_order(index, session),
                       "phase_order": phase_order(index), "measurements": measured})

    # Instrumented calls use the same kernels but extra CPU timestamps; they
    # are diagnostic probes and must not replace the uninstrumented blocks.
    instrumented_probe = []
    for block in range(1, PROBE_BLOCKS + 1):
        per_call = []

        def instrumented():
            started = time.perf_counter_ns()
            scratch = torch.empty((2, w["m"], w["n"]), device="cuda", dtype=torch.float32)
            allocated = time.perf_counter_ns()
            partial(a, b, scratch)
            partial_submitted = time.perf_counter_ns()
            reduction(scratch, output_alloc)
            reduction_submitted = time.perf_counter_ns()
            per_call.append({"scratch_allocation_ns": allocated - started,
                             "partial_launch_submission_ns": partial_submitted - allocated,
                             "reduction_launch_submission_ns": reduction_submitted - partial_submitted})

        timing = measure(torch, instrumented, ITERATIONS, raw_clock=raw_clock)
        instrumented_probe.append({"block": block, "measurement": timing, "per_call": per_call})

    graph = graph_phase_probe(torch, functions)
    git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    raw = {"schema_version": 1, "session": session,
           "created_at_utc": datetime.now(timezone.utc).isoformat(),
           "source_sha256": source_sha, "spec_sha256": digest(SPEC),
           "diagnostic_script_sha256": digest(Path(__file__)), "git_head": git_head,
           "environment": {"gpu": props.name, "sms": props.multi_processor_count,
                           "python": sys.version.split()[0], "torch": torch.__version__,
                           "triton": triton.__version__, "cuda_runtime": torch.version.cuda},
           "measurement_windows": ("Existing CudaTritonBackend._measure: perf start, RAW start, CUDA start "
                                   "event, N complete calls, CUDA end event, perf submission timestamp, "
                                   "end-event sync, RAW end, perf end. Device event may contain host pacing gaps."),
           "iterations_per_block": ITERATIONS, "warmups_per_function": WARMUPS,
           "reuse_variant": "mirrors matmul_into except torch.empty is replaced by saved scratch",
           "correctness": {"passed": True, "seeds": correctness,
                           "reference": "pytorch_matmul (cuBLAS)",
                           "tolerances": validated["correctness"]},
           "host_clock_preflight": preflight,
           "grids": {"conventional": conventional_grid, "partial": partial_grid,
                     "reduction": reduction_grid},
           "compiler": compiler, "rounds": rounds,
           "instrumented_probe": instrumented_probe,
           "graph_phase_probe": graph}
    write_json(output / "raw.json", raw)
    write_json(output / "summary.json", summarize(raw))
    manifest(output)
    return verify(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--session", type=int, choices=(1, 2))
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify and args.session is not None:
        parser.error("--verify cannot be combined with --session")
    if not args.verify and args.session is None:
        parser.error("--session is required for a GPU run")
    try:
        result = verify(args.output) if args.verify else run(args.output, args.session)
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
        print(f"MI-Research-2B diagnostic failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
