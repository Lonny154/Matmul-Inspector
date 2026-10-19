#!/usr/bin/env python3
"""Run one bounded two-batch schedule-agent study or deterministic replay."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from matmul_inspector.tensor_game_schedule_agent import (
    LIMITS, FallbackPlanner, OpenAIPlanner, ProposalError, ReplayPlanner,
    fallback_proposal, feedback_summary, selections, validate_proposal,
)
from matmul_inspector.tensor_game_schedule_reselection import (
    MODES, compare_reselection, load_retained_corpus, select_batch1,
)
from matmul_inspector.tensor_game_schedule_gpu_study import verify_artifact_hashes
from scripts.run_ha_schedule_10 import CONFIG, cpu_gate, gpu_batch
from scripts.run_hardware_aware_schedule_lowering_study import (
    FROZEN, REPO, project_actions, source_rows,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


class Bundle:
    def __init__(self, path: Path):
        if path.exists():
            raise ValueError(f"output already exists: {path}")
        path.mkdir(parents=True)
        self.path = path

    def write(self, name: str, value, *, rows: bool = False) -> None:
        data = ("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                        for row in value) if rows else
                json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
        size = sum(file.stat().st_size for file in self.path.iterdir() if file.is_file() and
                   file.name != "hash_manifest.json" and file.name != name)
        reserve = 16_000 if name not in {"failure.json", "decision.json"} else 0
        if size + len(data.encode()) + reserve > LIMITS.max_artifact_bytes:
            raise RuntimeError("artifact size budget exhausted")
        temporary = self.path / (name + ".tmp")
        temporary.write_text(data)
        temporary.replace(self.path / name)
        self.manifest()

    def manifest(self) -> None:
        files = {file.name: digest(file) for file in sorted(self.path.iterdir())
                 if file.is_file() and file.name != "hash_manifest.json"}
        (self.path / "hash_manifest.json").write_text(
            json.dumps(files, sort_keys=True, separators=(",", ":")) + "\n")


def run_gpu_worker(output: Path, hashes: Path, label: str) -> None:
    candidates, _ = load_retained_corpus(REPO)
    by_hash = {row["schedule_hash"]: row for row in candidates}
    selected = [by_hash[item] for item in json.loads(hashes.read_text())]
    source = {row["schedule_hash"]: row for row in source_rows()}
    actions = project_actions(FROZEN)
    gates = cpu_gate(selected, source, actions)
    checks, topology, raw, summaries, observations = gpu_batch(selected, source, actions, label)
    output.write_text(json.dumps({"cpu_gates": gates, "correctness": checks,
                                  "topology": topology, "raw": raw,
                                  "summaries": summaries, "observations": observations},
                                 sort_keys=True, separators=(",", ":")) + "\n")


def execute_gpu_batch(bundle: Bundle, label: str, selected: list[dict], deadline: float) -> dict:
    hashes = [row["schedule_hash"] for row in selected]
    bundle.write(f"{label}_requested_hashes.json", hashes)
    worker_output = bundle.path / f"{label}_worker.json.tmp"
    remaining = int(min(LIMITS.max_batch_seconds, deadline - time.monotonic()))
    if remaining <= 0:
        raise TimeoutError("total wall-clock budget exhausted before GPU batch")
    command = [sys.executable, str(Path(__file__).resolve()), "--worker-batch", label,
               "--output", str(worker_output), "--hashes", str(bundle.path / f"{label}_requested_hashes.json")]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + str(REPO)
    try:
        result = subprocess.run(command, cwd=REPO, env=env, capture_output=True,
                                text=True, timeout=remaining, check=False)
    except subprocess.TimeoutExpired as error:
        raise TimeoutError(f"{label} GPU worker exceeded {remaining}s") from error
    if result.returncode:
        raise RuntimeError(f"{label} GPU worker exited {result.returncode}: {result.stderr[-3000:]}")
    if worker_output.stat().st_size > LIMITS.max_artifact_bytes:
        worker_output.unlink()
        raise RuntimeError("artifact size budget exhausted by GPU worker output")
    data = json.loads(worker_output.read_text())
    worker_output.unlink()
    if len(data["topology"]) > len(selected) or len(data["raw"]) > 50 * len(selected):
        raise RuntimeError("GPU profiler or timing sample budget exceeded")
    for key, filename in (("cpu_gates", "cpu_gates"), ("correctness", "gpu_correctness"),
                          ("topology", "gpu_topology"), ("raw", "raw_device_timing"),
                          ("summaries", "timing_summary"), ("observations", "observations")):
        bundle.write(f"{label}_{filename}.jsonl", data[key], rows=True)
    return data


def get_proposal(planner, batch: int, feedback: dict, bundle: Bundle,
                 candidates: list[dict], batch1: list[dict]) -> tuple[dict, str]:
    try:
        raw = planner.propose(batch, feedback)
        source = planner.source
    except Exception as error:
        if isinstance(planner, ReplayPlanner):
            raise
        bundle.write(f"batch{batch}_provider_error.json", {"kind": type(error).__name__,
                     "message": str(error)[:500], "fallback_used": True})
        raw = fallback_proposal(batch)
        source = "DETERMINISTIC_FALLBACK_AFTER_PROVIDER_ERROR"
    bundle.write(f"batch{batch}_proposal.json", {"source": source, "proposal": raw,
                                                  "model": planner.model if source == "LIVE_OPENAI_RESPONSES" else None,
                                                  "provider_metadata": planner.last_metadata
                                                  if source == "LIVE_OPENAI_RESPONSES" else None})
    try:
        proposal = validate_proposal(raw, batch=batch, candidates=candidates, batch1=batch1)
    except Exception as error:
        bundle.write(f"batch{batch}_validation.json", {"valid": False, "reason": str(error)})
        raise
    bundle.write(f"batch{batch}_validation.json", {"valid": True, "reason": "approved allowlist and hard limits"})
    return proposal, source


def run(args) -> dict:
    started = time.monotonic()
    deadline = started + LIMITS.max_wall_seconds
    bundle = Bundle(args.output.resolve())
    status = "BLOCKED"
    classification = "D"
    explanation = "controller did not complete"
    try:
        candidates, source_provenance = load_retained_corpus(REPO)
        source = {row["schedule_hash"]: row for row in source_rows()}
        actions = project_actions(FROZEN)
        batch1 = select_batch1(candidates, size=LIMITS.matrix_size)
        if len(batch1) != LIMITS.batch1 or CONFIG["rounds"] > LIMITS.max_rounds or (
                CONFIG["iterations_per_round"] > LIMITS.max_iterations_per_round):
            raise RuntimeError("frozen selection or benchmark settings exceed budget")
        bundle.write("limits.json", LIMITS.__dict__)
        bundle.write("benchmark_config.json", CONFIG)
        bundle.write("provenance.json", {**source_provenance,
            "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO, text=True).strip(),
            "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
            "created_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
            "runner_sha256": digest(Path(__file__)),
            "agent_module_sha256": digest(REPO / "src/matmul_inspector/tensor_game_schedule_agent.py"),
            "ha10_runner_sha256": digest(REPO / "scripts/run_ha_schedule_10.py"),
            "selector_sha256": digest(REPO / "src/matmul_inspector/tensor_game_schedule_reselection.py"),
            "execution": "FRESH_GPU" if args.gpu else "CPU_REPLAY"})
        bundle.write("candidate_universe.jsonl", candidates, rows=True)
        bundle.write("batch1_selection.json", batch1)
        frozen = {mode: selections(candidates, batch1, [], fallback_proposal(2))["frozen"][mode]
                  for mode in MODES}
        bundle.write("batch2_frozen_control.json", frozen)
        if args.provider == "openai":
            planner = OpenAIPlanner(args.model)
        elif args.proposals:
            planner = ReplayPlanner(json.loads(args.proposals.read_text()))
        else:
            planner = FallbackPlanner()
        proposal1, source1 = get_proposal(planner, 1,
            {"frozen_batch1_hashes": [row["schedule_hash"] for row in batch1],
             "remaining_candidate_budget": LIMITS.max_gpu_candidates}, bundle, candidates, batch1)
        gate1 = cpu_gate(batch1, source, actions)
        bundle.write("batch1_cpu_gates.jsonl", gate1, rows=True)
        if args.gpu:
            first = execute_gpu_batch(bundle, "batch1", batch1, deadline)
        else:
            replay_bundle = getattr(args, "replay_bundle", None)
            if args.replay_ha10 is None and replay_bundle is None:
                raise ValueError("CPU replay requires --replay-ha10 or --replay-bundle")
            if replay_bundle is not None:
                verify_artifact_hashes(replay_bundle)
                if read_rows(replay_bundle / "candidate_universe.jsonl") != candidates or (
                        json.loads((replay_bundle / "batch1_selection.json").read_text()) != batch1 or
                        json.loads((replay_bundle / "benchmark_config.json").read_text()) != json.loads(json.dumps(CONFIG))):
                    raise ValueError("replay bundle corpus, Batch 1, or benchmark config differs")
                first = {"correctness": read_rows(replay_bundle / "batch1_gpu_correctness.jsonl"),
                         "summaries": read_rows(replay_bundle / "batch1_timing_summary.jsonl"),
                         "observations": read_rows(replay_bundle / "batch1_observations.jsonl")}
                replay_source = replay_bundle
            else:
                verify_artifact_hashes(args.replay_ha10)
                first = {"correctness": [row for row in read_rows(args.replay_ha10 / "gpu_correctness.jsonl")
                                         if row["batch"] == "batch1"],
                         "summaries": [row for row in read_rows(args.replay_ha10 / "timing_summary.jsonl")
                                       if row["session"] == "batch1"],
                         "observations": [row for row in json.loads((args.replay_ha10 / "calibration_after.json").read_text())["observations"]
                                          if row["provenance"]["batch"] == "batch1"]}
                replay_source = args.replay_ha10
            bundle.write("replay_source.json", {"path": str(replay_source),
                                                "manifest_sha256": digest(replay_source / "hash_manifest.json")})
        observations = first["observations"]
        bundle.write("calibration_after_batch1.json", {"observations": observations,
            "evidence_kind": "FRESH_GPU" if args.gpu else "VERIFIED_HA10_REPLAY"})
        feedback = feedback_summary(batch1, first["correctness"], first["summaries"],
                                    observations, candidates)
        bundle.write("batch1_feedback.json", feedback)
        proposal2, source2 = get_proposal(planner, 2, feedback, bundle, candidates, batch1)
        selection = selections(candidates, batch1, observations, proposal2)
        bundle.write("batch2_controls_and_guidance.json", selection)
        replay_bundle = getattr(args, "replay_bundle", None)
        if replay_bundle is not None:
            original = json.loads((replay_bundle / "batch2_controls_and_guidance.json").read_text())
            if selection != original:
                raise ValueError("replay did not reproduce the observed Batch 2 selections")
            bundle.write("replay_comparison.json", {"batch2_selection_exact_match": True,
                                                    "source_selection_sha256": digest(
                                                        replay_bundle / "batch2_controls_and_guidance.json")})
        batch2 = [next(row for row in candidates if row["schedule_hash"] == item["schedule_hash"])
                  for item in selection["physical_batch2"]]
        if len(batch2) != LIMITS.batch2 or len(batch1) + len(batch2) > LIMITS.max_gpu_candidates:
            raise RuntimeError("GPU candidate budget exhausted")
        gate2 = cpu_gate(batch2, source, actions)
        bundle.write("batch2_cpu_gates.jsonl", gate2, rows=True)
        if args.gpu:
            second = execute_gpu_batch(bundle, "batch2", batch2, deadline)
            bundle.write("calibration_after_batch2.json", {"observations": observations + second["observations"],
                "evidence_kind": "FRESH_GPU", "selection_use": "retained for future replay; no Batch 3"})
            all_checks = first["correctness"] + second["correctness"]
            all_topology = first["topology"] + second["topology"]
            all_raw = first["raw"] + second["raw"]
            expected = {row["schedule_hash"] for row in batch1 + batch2}
            complete = (len(all_topology) == LIMITS.max_profiler_invocations and
                        len(all_raw) == LIMITS.max_timing_samples and
                        all(row["passed"] for row in all_checks) and
                        all(row["topology_agrees"] for row in all_topology) and
                        {row["schedule_hash"] for row in all_topology} == expected)
        else:
            bundle.write("batch2_replay_note.json", {"status": "CPU gates and selection only",
                "fresh_gpu_measurement": False, "historical_gpu_measurement_not_copied": True})
            complete = True
        comparisons = {mode: {"frozen_vs_calibrated": compare_reselection(
            selection["frozen"][mode], selection["calibrated"][mode]),
            "calibrated_vs_guided": compare_reselection(
                selection["calibrated"][mode], selection["guided"][mode])} for mode in MODES}
        bundle.write("control_comparisons.json", comparisons)
        if not complete:
            status, classification, explanation = "COMPLETED_WITH_LIMITATIONS", "C", "GPU gates or sample budget incomplete"
        elif args.gpu and source1 == source2 == "LIVE_OPENAI_RESPONSES":
            status, classification, explanation = "COMPLETED", "A", "two live proposals and two GPU batches completed"
        else:
            status, classification, explanation = "COMPLETED", "B", "two-batch controlled loop completed without a validated live model"
        bundle.write("resource_usage.json", {"batches": 2, "gpu_candidates": 12 if args.gpu else 0,
            "profiler_invocations": len(first["topology"] + second["topology"]) if args.gpu else 0,
            "timing_samples": len(first["raw"] + second["raw"]) if args.gpu else 0,
            "planner_calls": 2, "elapsed_seconds": time.monotonic() - started,
            "artifact_bytes": sum(p.stat().st_size for p in bundle.path.iterdir() if p.is_file()),
            "batch1_proposal_source": source1, "batch2_proposal_source": source2})
    except Exception as error:
        explanation = f"{type(error).__name__}: {error}"[:2000]
        if isinstance(error, (ProposalError, ValueError)):
            status, classification = "FAILED_VALIDATION", "D"
        elif isinstance(error, TimeoutError) or "budget" in str(error).lower():
            status, classification = "BUDGET_EXHAUSTED", "D"
        else:
            status, classification = "BLOCKED", "D"
        try:
            bundle.write("failure.json", {"status": status, "reason": explanation,
                                          "elapsed_seconds": time.monotonic() - started})
        except Exception:
            pass
    result = {"status": status, "classification": classification,
              "reason": explanation, "gpu": args.gpu,
              "elapsed_seconds": time.monotonic() - started}
    bundle.write("decision.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--replay-ha10", type=Path)
    parser.add_argument("--replay-bundle", type=Path)
    parser.add_argument("--proposals", type=Path)
    parser.add_argument("--provider", choices=("fallback", "openai"), default="fallback")
    parser.add_argument("--model")
    parser.add_argument("--worker-batch", choices=("batch1", "batch2"), help=argparse.SUPPRESS)
    parser.add_argument("--hashes", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_batch:
        run_gpu_worker(args.output, args.hashes, args.worker_batch)
        return
    if args.proposals and args.provider != "fallback":
        parser.error("--proposals and --provider openai are mutually exclusive")
    if sum((bool(args.gpu), bool(args.replay_ha10), bool(args.replay_bundle))) != 1:
        parser.error("choose exactly one of --gpu, --replay-ha10, or --replay-bundle")
    result = run(args)
    print(json.dumps({"output": str(args.output.resolve()), "decision": result}, sort_keys=True))
    if result["classification"] == "D":
        sys.exit(1)


if __name__ == "__main__":
    main()
