"""Bounded shape study of complete conventional, Split-2, and cuBLAS graphs.

This narrow diagnostic reuses the MI-Research-2C clock and paired-decision code.
Each process measures one shape and one fresh session. No production timer or
kernel is changed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

from matmul_inspector import experiment_pair
try:
    from scripts import run_mi_research_2c_graph as prior
except ModuleNotFoundError:
    # Direct `python scripts/run_mi_research_3_graph.py` puts scripts/ on sys.path.
    import run_mi_research_2c_graph as prior


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "examples/kernels/mi_research_2_split2.py"
PRIOR_SOURCE = ROOT / "artifacts/mi-research-2/evidence-v1/low-output-split2/sources/B.py"
SPEC = ROOT / "examples/mi_research_2_low_output_split2.json"
SHAPES = (32, 128, 256, 512)
K, N = 4096, 2048
OPERATIONS, ROUNDS, WARMUPS = prior.OPERATIONS, prior.ROUNDS, prior.WARMUPS
PAIR_NAMES = {"AB": ("A", "B"), "AC": ("A", "C"), "BC": ("B", "C")}


def shape_geometry(m: int) -> dict:
    if m not in SHAPES:
        raise ValueError("shape is outside the frozen study")
    return {"conventional_ctas": math.ceil(m / 64) * math.ceil(N / 128),
            "split_partial_ctas": 2 * math.ceil(m / 64) * math.ceil(N / 128),
            "reduction_ctas": math.ceil(m * N / 1024),
            "scratch_bytes": 2 * m * N * 4}


def pair_phase_order(session: int) -> list[str]:
    if session not in (1, 2):
        raise ValueError("session must be 1 or 2")
    return ["AB", "AC", "BC"] if session == 1 else ["BC", "AC", "AB"]


def graph_profile(trace: dict, label: str) -> dict:
    names, launches = prior.trace_sequence(trace)
    if label in ("A", "B"):
        return prior.require_complete_trace(trace, label)
    if label != "C" or launches != 1 or len(names) < OPERATIONS or len(names) % OPERATIONS:
        raise ValueError("cuBLAS graph does not show a repeated complete operation")
    chunk = names[:len(names) // OPERATIONS]
    if names != chunk * OPERATIONS:
        raise ValueError("cuBLAS graph kernel pattern changed within replay")
    return {"kernel_sequence": names, "kernels_per_operation": len(chunk),
            "cuda_graph_launch_calls": launches, "kernel_nodes_observed": len(names),
            "complete_operations": OPERATIONS}


def _pair_benchmark(raw: dict, key: str) -> dict:
    pair = raw["pairs"][key]
    return {"mode": "cuda_graph", "iterations_per_measurement": OPERATIONS,
            "host_clock_preflight": raw["host_clock_preflight"],
            "unranked_timing_controls": pair["unranked_timing_controls"],
            "aa_rounds": pair["aa_rounds"], "aa_kernel": "A",
            "aa_iterations_per_measurement": OPERATIONS,
            "rounds": pair["rounds"], "measurement_sequence": pair["measurement_sequence"]}


def summarize(raw: dict) -> dict:
    if (raw.get("schema_version") != 1 or raw.get("mode") != "cuda_graph" or
            raw.get("m") not in SHAPES or raw.get("session") not in (1, 2) or
            raw.get("operations_per_replay") != OPERATIONS or
            raw.get("rounds") != ROUNDS or raw.get("warmups_per_graph") != WARMUPS or
            raw.get("phase_order") != pair_phase_order(raw["session"]) or
            raw.get("geometry") != shape_geometry(raw["m"])):
        raise ValueError("invalid shape-study configuration")
    correctness = raw["correctness"]
    if (correctness.get("passed") is not True or
            [case.get("seed") for case in correctness.get("seeds", [])] != [0, 1, 2] or
            any(case.get("checks", {}).get(label, {}).get("passed") is not True
                for case in correctness["seeds"] for label in ("A", "B", "C"))):
        raise ValueError("incomplete numerical gate")
    result = {"schema_version": 1, "m": raw["m"], "session": raw["session"],
              "geometry": raw["geometry"], "pairs": {}, "graph_setup_ms": raw["graph_setup_ms"]}
    expected = experiment_pair.expected_diagnostic_sequence(ROUNDS, OPERATIONS, "before")
    for key, (left, right) in PAIR_NAMES.items():
        pair = raw["pairs"][key]
        if (pair["physical_labels"] != [left, right] or
                pair["measurement_sequence"] != expected or
                len(pair["rounds"]) != ROUNDS or len(pair["aa_rounds"]) != ROUNDS or
                set(pair["unranked_timing_controls"]) != {"A", "B"}):
            raise ValueError(f"{key} pairing changed")
        samples = list(pair["unranked_timing_controls"].values())
        for index in range(1, ROUNDS + 1):
            for phase, slots in (("rounds", ("A", "B")), ("aa_rounds", ("X", "Y"))):
                row = pair[phase][index - 1]
                order = (experiment_pair.pair_order(index) if phase == "rounds" else
                         (["X", "Y"] if index % 2 else ["Y", "X"]))
                if row["round"] != index or row["order"] != order or set(row["measurements"]) != set(slots):
                    raise ValueError(f"{key} round order changed")
                samples.extend(row["measurements"].values())
        for sample in samples:
            if (sample.get("complete_operations_per_block") != OPERATIONS or
                    sample.get("host_submission_calls_per_block") != 1 or
                    not math.isclose(sample["cuda_event_elapsed_ms_per_operation"] * OPERATIONS,
                                     sample["cuda_event_elapsed_ms_per_block"], abs_tol=1e-8)):
                raise ValueError(f"{key} sample operation count changed")
        benchmark = _pair_benchmark(raw, key)
        decision = experiment_pair.summarize_clock_aware(
            {"passed": True, "checks": {"A": correctness["seeds"][0]["checks"][left],
                                          "B": correctness["seeds"][0]["checks"][right]}},
            experiment_pair.timing_rows({"benchmark": benchmark}), None, benchmark)
        if decision["decision"] in ("FAILED", "REJECTED_INCORRECT") or decision["rounds"] != ROUNDS:
            raise ValueError(f"{key} decision could not replay")
        values = {label: [row["measurements"][label]["cuda_event_elapsed_ms_per_operation"] * 1000
                          for row in pair["rounds"]] for label in ("A", "B")}
        result["pairs"][key] = {"physical_labels": [left, right], "decision": decision,
                                "event_median_us": {name: statistics.median(values[label])
                                                    for label, name in (("A", left), ("B", right))},
                                "event_range_us": {name: [min(values[label]), max(values[label])]
                                                   for label, name in (("A", left), ("B", right))}}
    return result


def verify(output: Path) -> dict:
    saved = json.loads((output / "hash_manifest.json").read_text())
    if not isinstance(saved, dict) or any(Path(name).is_absolute() or ".." in Path(name).parts for name in saved):
        raise ValueError("invalid manifest")
    actual = {str(p.relative_to(output)) for p in output.rglob("*") if p.is_file() and p.name != "hash_manifest.json"}
    if actual != set(saved) or any(prior.digest(output / name) != value for name, value in saved.items()):
        raise ValueError("manifest mismatch")
    raw = json.loads((output / "raw.json").read_text())
    for field, name in (("source_sha256", "source.py"), ("base_spec_sha256", "base_spec.json"),
                        ("diagnostic_script_sha256", "diagnostic_script.py"),
                        ("prior_script_sha256", "prior_script.py")):
        if raw[field] != prior.digest(output / name):
            raise ValueError(f"{name} identity mismatch")
    for label in ("A", "B", "C"):
        path = output / f"{label}_trace.json"
        profile = graph_profile(json.loads(path.read_text()), label)
        if raw["graph_profile"][label] != profile or raw["graph_trace_sha256"][label] != prior.digest(path):
            raise ValueError(f"{label} graph trace mismatch")
    for name in ("conventional", "partial", "reduction"):
        if raw["compiler"][name]["ptx_sha256"] != prior.digest(output / f"{name}.ptx"):
            raise ValueError(f"{name} PTX mismatch")
    replay = summarize(raw)
    if replay != json.loads((output / "summary.json").read_text()):
        raise ValueError("summary does not replay")
    return {"status": "PASS", "m": raw["m"], "session": raw["session"],
            "decisions": {key: value["decision"]["decision"] for key, value in replay["pairs"].items()},
            "files": len(saved)}


def run(output: Path, m: int, session: int) -> dict:
    if output.exists():
        raise ValueError("output already exists")
    geometry = shape_geometry(m)
    phase_order = pair_phase_order(session)
    if prior.digest(SOURCE) != prior.digest(PRIOR_SOURCE):
        raise ValueError("Split-2 source differs from retained source")
    base_spec = experiment_pair.validate_spec(json.loads(SPEC.read_text()))
    if (base_spec["workload"] != {"m": 32, "k": K, "n": N, "dtype": "float16"} or
            base_spec["benchmark"] != {"mode": "preallocated", "warmups": WARMUPS,
                                       "iterations": OPERATIONS, "rounds": ROUNDS}):
        raise ValueError("base spec changed")
    output.mkdir(parents=True)
    for src, dst in ((SOURCE, "source.py"), (SPEC, "base_spec.json"),
                     (Path(__file__), "diagnostic_script.py"), (Path(prior.__file__), "prior_script.py")):
        shutil.copyfile(src, output / dst)

    import torch
    import triton
    from matmul_inspector import triton_fc1
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    source_spec = importlib.util.spec_from_file_location("mi_research_3_split2", output / "source.py")
    if source_spec is None or source_spec.loader is None:
        raise RuntimeError("cannot import retained Split-2 source")
    module = importlib.util.module_from_spec(source_spec)
    old_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        source_spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = old_bytecode
    if module.SPLIT_K != 2 or module.REDUCTION_BLOCK != 1024:
        raise ValueError("Split-2 geometry changed")
    p = base_spec["kernels"]["B"]["parameters"]
    if p != {"block_m": 64, "block_n": 128, "block_k": 32, "num_warps": 4,
             "num_stages": 3, "group_size_m": 1}:
        raise ValueError("tile configuration changed")
    if geometry["scratch_bytes"] > 8 * 1024 * 1024:
        raise ValueError("scratch budget exceeded")
    a = torch.empty((m, K), device="cuda", dtype=torch.float16)
    b = torch.empty((K, N), device="cuda", dtype=torch.float16)
    outputs = {label: torch.empty((m, N), device="cuda", dtype=torch.float16) for label in ("A", "B", "C")}
    scratch = torch.empty((2, m, N), device="cuda", dtype=torch.float32)
    a_grid = (geometry["conventional_ctas"],)
    partial_grid = (geometry["conventional_ctas"], 2)
    reduction_grid = (geometry["reduction_ctas"],)

    def conventional():
        out = outputs["A"]
        return triton_fc1.fc1_kernel[a_grid](
            a, b, out, m, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
            out.stride(0), out.stride(1), BLOCK_M=p["block_m"], BLOCK_N=p["block_n"],
            BLOCK_K=p["block_k"], GROUP_SIZE_M=p["group_size_m"],
            num_warps=p["num_warps"], num_stages=p["num_stages"])

    def split_complete():
        out = outputs["B"]
        partial = module.split_k_partial[partial_grid](
            a, b, scratch, m, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
            BLOCK_M=p["block_m"], BLOCK_N=p["block_n"], BLOCK_K=p["block_k"],
            K_PART=K // 2, num_warps=p["num_warps"], num_stages=p["num_stages"])
        reduction = module.reduce_partials[reduction_grid](
            scratch, out, m, N, out.stride(0), out.stride(1),
            BLOCK=module.REDUCTION_BLOCK, num_warps=4)
        return partial, reduction

    def cublas():
        return torch.mm(a, b, out=outputs["C"])

    def inputs(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        return (torch.randn((m, K), device="cuda", dtype=torch.float16),
                torch.randn((K, N), device="cuda", dtype=torch.float16))

    original_a, original_b = inputs(0)
    a.copy_(original_a)
    b.copy_(original_b)
    conventional_launch = conventional()
    partial_launch, reduction_launch = split_complete()
    cublas()
    torch.cuda.synchronize()
    compiler = {}
    for name, launch in (("conventional", conventional_launch),
                         ("partial", partial_launch), ("reduction", reduction_launch)):
        capture = experiment_pair._compiled_capture(launch)
        if capture is None or not isinstance(capture.get("ptx_text"), str):
            raise RuntimeError(f"{name} PTX unavailable")
        (output / f"{name}.ptx").write_text(capture.pop("ptx_text"))
        capture.pop("sass_text", None)
        capture["ptx_sha256"] = prior.digest(output / f"{name}.ptx")
        compiler[name] = capture
    eager = {"A": conventional, "B": split_complete, "C": cublas}
    for label in eager:
        for _ in range(WARMUPS):
            eager[label]()
        torch.cuda.synchronize()
    capture_order = ["A", "B", "C"] if session == 1 else ["C", "B", "A"]
    graphs, setup_ms = {}, {}
    for label in capture_order:
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(OPERATIONS):
                eager[label]()
        torch.cuda.synchronize()
        graphs[label] = graph
        setup_ms[label] = (time.perf_counter_ns() - start) / 1e6
    cases = []
    for seed in range(3):
        left, right = inputs(seed)
        a.copy_(left)
        b.copy_(right)
        reference = torch.matmul(a, b)
        for out in outputs.values():
            out.fill_(float("nan"))
        scratch.fill_(float("nan"))
        torch.cuda.synchronize()
        for label in ("A", "B", "C"):
            graphs[label].replay()
        torch.cuda.synchronize()
        checks = {label: experiment_pair.check_output(torch, reference, outputs[label],
                                                       base_spec["correctness"]) for label in eager}
        cases.append({"seed": seed, "checks": checks})
        if not all(row["passed"] for row in checks.values()):
            prior.write_json(output / "correctness_failure.json",
                             {"m": m, "seed": seed, "checks": checks,
                              "tolerances": base_spec["correctness"],
                              "source_sha256": prior.digest(SOURCE)})
            raise RuntimeError(f"numerical correctness failed for seed {seed}")
    a.copy_(original_a)
    b.copy_(original_b)
    torch.cuda.synchronize()
    profiles, trace_hashes = {}, {}
    for label in eager:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as profiler:
            graphs[label].replay()
            torch.cuda.synchronize()
        path = output / f"{label}_trace.json"
        profiler.export_chrome_trace(str(path))
        profiles[label] = graph_profile(json.loads(path.read_text()), label)
        trace_hashes[label] = prior.digest(path)
    functions = {label: graphs[label].replay for label in eager}
    for label in eager:
        for _ in range(WARMUPS):
            functions[label]()
        torch.cuda.synchronize()
    raw_clock = experiment_pair.raw_clock()
    clock_preflight = experiment_pair.host_clock_preflight(reader=raw_clock)
    options = {"raw_clock": raw_clock}
    pairs = {}
    for key in phase_order:
        left, right = PAIR_NAMES[key]
        mapped = {"A": functions[left], "B": functions[right]}
        controls = {label: prior.graph_measure(torch, mapped[label], OPERATIONS, **options)
                    for label in ("A", "B")}
        aa, rounds, sequence = experiment_pair.measure_paired_rounds(
            prior.graph_measure, torch, mapped, OPERATIONS, ROUNDS, options,
            with_aa=True, aa_placement="before")
        pairs[key] = {"physical_labels": [left, right],
                      "unranked_timing_controls": controls, "aa_rounds": aa,
                      "rounds": rounds, "measurement_sequence": sequence}
    props = torch.cuda.get_device_properties(0)
    git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                              capture_output=True, text=True).stdout.strip()
    raw = {"schema_version": 1, "mode": "cuda_graph", "m": m, "k": K, "n": N,
           "session": session, "created_at_utc": datetime.now(timezone.utc).isoformat(),
           "git_head": git_head, "source_sha256": prior.digest(SOURCE),
           "base_spec_sha256": prior.digest(SPEC),
           "diagnostic_script_sha256": prior.digest(Path(__file__)),
           "prior_script_sha256": prior.digest(Path(prior.__file__)),
           "geometry": geometry, "operations_per_replay": OPERATIONS,
           "rounds": ROUNDS, "warmups_per_graph": WARMUPS,
           "capture_order": capture_order, "phase_order": phase_order,
           "graph_setup_ms": setup_ms, "graph_profile": profiles,
           "graph_trace_sha256": trace_hashes,
           "semantics": {"A": "fixed-address complete conventional GEMM; one kernel per operation",
                         "B": "fixed-address complete Split-2; partial plus reduction each operation; reusable scratch",
                         "C": "fixed-address torch.mm out= graph; matched replay semantics",
                         "graph_replays_per_measured_block": 1,
                         "capture_and_profiler_excluded_from_timing": True},
           "environment": {"gpu": props.name, "sm_count": props.multi_processor_count,
                           "compute_capability": f"{props.major}.{props.minor}",
                           "python": sys.version.split()[0], "torch": torch.__version__,
                           "triton": triton.__version__, "cuda_runtime": torch.version.cuda},
           "compiler": compiler,
           "correctness": {"passed": True, "reference": "pytorch_matmul (cuBLAS)",
                           "tolerances": base_spec["correctness"], "seeds": cases},
           "host_clock_preflight": clock_preflight, "pairs": pairs}
    prior.write_json(output / "raw.json", raw)
    prior.write_json(output / "summary.json", summarize(raw))
    prior.manifest(output)
    return verify(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--m", type=int, choices=SHAPES)
    parser.add_argument("--session", type=int, choices=(1, 2))
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify == (args.m is not None and args.session is not None):
        parser.error("choose --verify or both --m and --session")
    try:
        result = verify(args.output) if args.verify else run(args.output, args.m, args.session)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        if not args.verify and args.output.is_dir():
            prior.write_json(args.output / "failure.json", {"type": type(error).__name__, "message": str(error)})
        print(f"MI-Research-3 graph diagnostic failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
