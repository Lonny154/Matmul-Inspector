"""One bounded matched CUDA Graph timing session for the frozen four-CTA pair."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import statistics
import sys

from matmul_inspector import experiment_pair
try:
    from scripts import freeze_mi_research_10 as frozen
    from scripts import run_mi_research_10 as study
    from scripts import run_mi_research_2c_graph as evidence
    from scripts import run_mi_research_3b_correctness as diagnostics
    from scripts import run_mi_research_4_feasibility as cases
    from scripts import run_mi_research_6_gpu_probe as loader
    from scripts import run_mi_research_6d_graph as graph_prior
    from scripts import run_mi_research_6c as exact
except ModuleNotFoundError:
    import freeze_mi_research_10 as frozen
    import run_mi_research_10 as study
    import run_mi_research_2c_graph as evidence
    import run_mi_research_3b_correctness as diagnostics
    import run_mi_research_4_feasibility as cases
    import run_mi_research_6_gpu_probe as loader
    import run_mi_research_6d_graph as graph_prior
    import run_mi_research_6c as exact

OUT = frozen.OUT / "graph-pair-s1"
OPS, ROUNDS, WARMUPS = 20, 8, 10


def verify() -> dict:
    frozen.main()
    manifest = json.loads((OUT / "hash_manifest.json").read_text())
    names = {str(p.relative_to(OUT)) for p in OUT.rglob("*") if p.is_file() and p.name != "hash_manifest.json"}
    if names != set(manifest) or any(evidence.digest(OUT / n) != h for n, h in manifest.items()):
        raise ValueError("timing manifest mismatch")
    raw = json.loads((OUT / "raw.json").read_text())
    if raw["frozen_plan_sha256"] != evidence.digest(frozen.PLAN) or raw["candidate_gpu_evidence_sha256"] != evidence.digest(study.OUT / "raw.json"):
        raise ValueError("timing provenance mismatch")
    if not raw["correctness_passed"] or not raw["fixed_addresses_unchanged"]:
        raise ValueError("timing gate failed")
    if raw["pair"]["measurement_sequence"] != experiment_pair.expected_diagnostic_sequence(ROUNDS, OPS, "before"):
        raise ValueError("paired sequence mismatch")
    for row in raw["correctness"]:
        for label in ("A", "B"):
            checks = row["checks"][label]
            if any(not x["passed"] or x["failing_elements"] for phase in ("eager", "cuda_graph") for x in checks[phase]):
                raise ValueError("correctness replay failed")
    return {"status": "PASS", "decision": raw["decision"]["decision"], "median_us": raw["event_median_us"],
            "paired_median_A_minus_B_us": raw["paired_median_A_minus_B_us"]}


def run() -> dict:
    if OUT.exists():
        raise FileExistsError("timing session exists")
    if study.verify()["status"] != "PASS" or exact.verify()["status"] != "PASS":
        raise ValueError("correctness or exact gate failed")
    import torch
    import triton
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    OUT.mkdir(parents=True)
    sources = {"A": exact.PROBE_SOURCE, "B": frozen.candidate.TARGET}
    for label, source in sources.items():
        shutil.copyfile(source, OUT / f"{label}_source.py")
    for source, name in ((frozen.PLAN, "frozen_plan.json"), (Path(__file__), "diagnostic_script.py")):
        shutil.copyfile(source, OUT / name)
    original = {"precision": torch.get_float32_matmul_precision(),
                "tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                "reduced": bool(torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)}
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        modules = {label: loader.load_snapshot(OUT / f"{label}_source.py") for label in ("A", "B")}
        a0, b0 = cases._inputs(torch, 96, "seed-0")
        left, right = torch.empty_like(a0), torch.empty_like(b0)
        left.copy_(a0); right.copy_(b0)
        outputs = {label: torch.empty((96, 96), device="cuda", dtype=torch.float16) for label in ("A", "B")}
        addresses = (left.data_ptr(), right.data_ptr(), outputs["A"].data_ptr(), outputs["B"].data_ptr())
        functions = {label: (lambda label=label: modules[label].call(left, right, outputs[label])) for label in ("A", "B")}
        for label in ("A", "B"):
            functions[label]()
            for _ in range(WARMUPS):
                functions[label]()
        torch.cuda.synchronize()
        graphs = {}
        for label in ("A", "B"):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(OPS):
                    functions[label]()
            graphs[label] = graph
        checks = []
        for name in cases.CASE_NAMES:
            aa, bb = cases._inputs(torch, 96, name)
            left.copy_(aa); right.copy_(bb)
            reference = torch.matmul(left.float(), right.float()).half()
            row = {}
            for label in ("A", "B"):
                row[label] = {"eager": [], "cuda_graph": []}
                for phase, fn in (("eager", functions[label]), ("cuda_graph", graphs[label].replay)):
                    for _ in range(2):
                        outputs[label].fill_(float("nan")); fn(); torch.cuda.synchronize()
                        row[label][phase].append(diagnostics.summarize_error(torch, outputs[label], reference, a=left, b=right))
            checks.append({"case": name, "checks": row})
        correct = all(x["passed"] and x["failing_elements"] == 0 and x["candidate_nonfinite_count"] == 0
                      for row in checks for label in ("A", "B") for phase in ("eager", "cuda_graph")
                      for x in row["checks"][label][phase])
        if not correct:
            raise ValueError("timing correctness gate failed")
        left.copy_(a0); right.copy_(b0)
        for label in ("A", "B"):
            for _ in range(WARMUPS):
                graphs[label].replay()
        torch.cuda.synchronize()
        reader = experiment_pair.raw_clock()
        clock = experiment_pair.host_clock_preflight(reader=reader)
        measure = graph_prior.previous.graph_measure
        mapped = {label: graphs[label].replay for label in ("A", "B")}
        controls = {label: measure(torch, mapped[label], OPS, raw_clock=reader) for label in ("A", "B")}
        aa_rows, rounds, sequence = experiment_pair.measure_paired_rounds(
            measure, torch, mapped, OPS, ROUNDS, {"raw_clock": reader}, with_aa=True,
            aa_placement="before")
        pair = {"physical_labels": ["A", "B"], "unranked_timing_controls": controls,
                "aa_rounds": aa_rows, "rounds": rounds, "measurement_sequence": sequence,
                "iterations_per_measurement": OPS, "aa_iterations_per_measurement": OPS,
                "aa_kernel": "A", "host_clock_preflight": clock}
        fixed = addresses == (left.data_ptr(), right.data_ptr(), outputs["A"].data_ptr(), outputs["B"].data_ptr())
        if not fixed:
            raise ValueError("tensor addresses changed")
        benchmark = {**pair, "host_clock_preflight": clock}
        decision = experiment_pair.summarize_clock_aware(
            {"passed": True, "checks": {"A": checks[0]["checks"]["A"]["cuda_graph"][0],
                                         "B": checks[0]["checks"]["B"]["cuda_graph"][0]}},
            experiment_pair.timing_rows({"benchmark": benchmark}), None, benchmark)
        times = {label: [row["measurements"][label]["cuda_event_elapsed_ms_per_operation"] * 1000
                         for row in rounds] for label in ("A", "B")}
        raw = {"schema_version": 1, "study": "MI-Research-10", "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "frozen_plan_sha256": evidence.digest(frozen.PLAN),
               "candidate_gpu_evidence_sha256": evidence.digest(study.OUT / "raw.json"),
               "source_sha256": {label: evidence.digest(OUT / f"{label}_source.py") for label in ("A", "B")},
               "script_sha256": evidence.digest(Path(__file__)),
               "environment": {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                               "triton": triton.__version__, "python": sys.version.split()[0],
                               "cuda_runtime": torch.version.cuda},
               "precision_policy": {"fp32_matmul_precision": "highest", "allow_tf32": False,
                                    "allow_fp16_reduced_precision_reduction": False, "original": original},
               "tolerance": study.TOLERANCE, "correctness": checks, "correctness_passed": correct,
               "fixed_addresses_unchanged": fixed, "graph_operations_per_replay": OPS,
               "host_clock_preflight": clock, "pair": pair, "decision": decision,
               "event_median_us": {label: statistics.median(values) for label, values in times.items()},
               "paired_median_A_minus_B_us": statistics.median(a-b for a, b in zip(times["A"], times["B"]))}
        evidence.write_json(OUT / "raw.json", raw)
        evidence.manifest(OUT)
        return verify()
    except Exception as error:
        evidence.write_json(OUT / "failure.json", {"type": type(error).__name__, "message": str(error)})
        evidence.manifest(OUT)
        raise
    finally:
        torch.set_float32_matmul_precision(original["precision"])
        torch.backends.cuda.matmul.allow_tf32 = original["tf32"]
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = original["reduced"]


if __name__ == "__main__":
    print(json.dumps(verify() if "--verify" in sys.argv else run(), sort_keys=True))
