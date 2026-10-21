"""Correctness-gated, single-node CUDA Graph target for MI-Research-7C."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from matmul_inspector import experiment_pair
try:
    from scripts import run_mi_research_2c_graph as evidence
    from scripts import run_mi_research_3b_correctness as diagnostics
    from scripts import run_mi_research_4_feasibility as cases
    from scripts import run_mi_research_6_gpu_probe as loader
    from scripts import run_mi_research_7b as previous
except ModuleNotFoundError:
    import run_mi_research_2c_graph as evidence
    import run_mi_research_3b_correctness as diagnostics
    import run_mi_research_4_feasibility as cases
    import run_mi_research_6_gpu_probe as loader
    import run_mi_research_7b as previous

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/mi-research-7c"
SOURCES = {"A": previous.generator.ORIGINAL, "B": previous.generator.TARGET}
SOURCE_HASHES = {"A": "c9b2d61fc6a1b6a658f1198d1a3938fae154efce2029c463e439b7fabbfaf8db",
                 "B": "10b94103a2dbf3d1649d1e3d79560d45830eeb5d960161905d69a6d5947db418"}
TOLERANCE = {"atol": 0.01, "rtol": 0.01}


def run(label: str, output: Path, profile: bool) -> dict:
    import torch
    import triton

    if output.exists():
        raise FileExistsError(output)
    previous.check_protocol()
    if label not in SOURCES or evidence.digest(SOURCES[label]) != SOURCE_HASHES[label]:
        raise ValueError("frozen source changed")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    module = loader.load_snapshot(SOURCES[label])
    initial_a, initial_b = cases._inputs(torch, 96, "seed-0")
    left, right = torch.empty_like(initial_a), torch.empty_like(initial_b)
    left.copy_(initial_a)
    right.copy_(initial_b)
    result = torch.empty((96, 96), dtype=torch.float16, device="cuda")
    addresses = (left.data_ptr(), right.data_ptr(), result.data_ptr())
    def kernel():
        return module.call(left, right, result)
    original = {"precision": torch.get_float32_matmul_precision(),
                "tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                "reduced": bool(torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)}
    checks = []
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        compiled = kernel()
        compiler = experiment_pair._compiled_capture(compiled)
        if compiler is None:
            raise RuntimeError("compiler metadata unavailable")
        for _ in range(10):
            kernel()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            kernel()
        for _ in range(10):
            graph.replay()
        torch.cuda.synchronize()
        for case in cases.CASE_NAMES:
            aa, bb = cases._inputs(torch, 96, case)
            left.copy_(aa); right.copy_(bb)
            reference = torch.matmul(left.float(), right.float()).half()
            result.fill_(float("nan")); kernel(); torch.cuda.synchronize()
            eager = diagnostics.summarize_error(torch, result, reference, a=left, b=right)
            result.fill_(float("nan")); graph.replay(); torch.cuda.synchronize()
            captured = result.clone()
            graph_check = diagnostics.summarize_error(torch, result, reference, a=left, b=right)
            kernel(); torch.cuda.synchronize()
            agreement = bool(torch.equal(result, captured))
            checks.append({"case": case, "eager": eager, "graph": graph_check,
                           "graph_eager_bitwise_equal": agreement})
            if (not eager["passed"] or not graph_check["passed"] or
                eager["failing_elements"] or graph_check["failing_elements"] or
                eager["candidate_nonfinite_count"] or graph_check["candidate_nonfinite_count"] or
                not agreement):
                raise ValueError(f"numerical gate failed: {case}")
        left.copy_(initial_a); right.copy_(initial_b)
        for _ in range(10):
            graph.replay()
        torch.cuda.synchronize()
        if profile:
            print(f"PROFILE_TARGET label={label} grid={'2x2x1' if label == 'A' else '2x2x9'} one_graph_node", flush=True)
            torch.cuda.profiler.start()
            graph.replay()
            torch.cuda.synchronize()
            torch.cuda.profiler.stop()
        else:
            result.fill_(float("nan")); graph.replay(); torch.cuda.synchronize()
            reference = torch.matmul(left.float(), right.float()).half()
            final = diagnostics.summarize_error(torch, result, reference, a=left, b=right)
            if not final["passed"] or final["failing_elements"]:
                raise ValueError("final graph replay failed")
        if addresses != (left.data_ptr(), right.data_ptr(), result.data_ptr()):
            raise ValueError("captured tensor addresses changed")
        props = torch.cuda.get_device_properties(0)
        raw = {"schema_version": 1, "label": label,
               "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "profiled_graph_node": profile,
               "source_sha256": SOURCE_HASHES[label],
               "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               "size": 96, "dtype": "float16", "tolerance": TOLERANCE,
               "correctness": checks, "fixed_addresses_unchanged": True,
               "graph_nodes": 1, "warmup_eager": 10, "warmup_graph": 10,
               "compiler_metadata": compiler["metadata"],
               "expected_grid": [2, 2, 1] if label == "A" else [2, 2, 9],
               "environment": {"gpu": props.name, "sm_count": props.multi_processor_count,
                               "compute_capability": f"{props.major}.{props.minor}",
                               "python": sys.version.split()[0], "torch": torch.__version__,
                               "triton": triton.__version__, "cuda_runtime": torch.version.cuda},
               "precision_policy": {"active_float32_matmul_precision": "highest",
                                    "active_allow_tf32": False,
                                    "active_allow_fp16_reduced_precision_reduction": False,
                                    "original": original}}
    finally:
        torch.set_float32_matmul_precision(original["precision"])
        torch.backends.cuda.matmul.allow_tf32 = original["tf32"]
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = original["reduced"]
    evidence.write_json(output, raw)
    return {"status": "PASS", "label": label, "cases": len(checks), "profiled_graph_node": profile}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", choices=("A", "B"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.kernel, args.output, args.profile), sort_keys=True))


if __name__ == "__main__":
    main()
