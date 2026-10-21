#!/usr/bin/env python3
"""Verify one completed Schedule Agent GPU bundle from retained evidence."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from matmul_inspector.tensor_game_schedule_mechanism_study import summarize_raw
from matmul_inspector.tensor_game_schedule_reselection import (
    MODES, load_retained_corpus, make_observation, select_batch1, validate_observations,
)
from matmul_inspector.tensor_game_schedule_agent import LIMITS, selections, validate_proposal
from matmul_inspector.tensor_game_schedule_gpu_study import verify_artifact_hashes
from scripts.run_ha_schedule_10 import CONFIG
from scripts.run_hardware_aware_schedule_lowering_study import REPO
from scripts.run_schedule_agent_prototype import digest, read_rows


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def audit(directory: Path) -> dict:
    manifest = verify_artifact_hashes(directory)
    actual = {path.name for path in directory.iterdir() if path.is_file()} - {"hash_manifest.json"}
    if set(manifest) != actual:
        raise ValueError("manifest omits or adds files")
    candidates, _ = load_retained_corpus(REPO)
    batch1 = select_batch1(candidates)
    assert read_rows(directory / "candidate_universe.jsonl") == candidates
    assert json.loads((directory / "batch1_selection.json").read_text()) == batch1
    assert json.loads((directory / "benchmark_config.json").read_text()) == json.loads(json.dumps(CONFIG))
    assert json.loads((directory / "decision.json").read_text())["status"] == "COMPLETED"
    checks, topology, raw, summaries, observations = [], [], [], [], []
    for batch, expected_count in (("batch1", 8), ("batch2", 4)):
        gate = read_rows(directory / f"{batch}_cpu_gates.jsonl")
        assert len(gate) == expected_count and all(row["status"] == "PASS" for row in gate)
        checks += read_rows(directory / f"{batch}_gpu_correctness.jsonl")
        topology += read_rows(directory / f"{batch}_gpu_topology.jsonl")
        raw += read_rows(directory / f"{batch}_raw_device_timing.jsonl")
        summaries += read_rows(directory / f"{batch}_timing_summary.jsonl")
        observations += read_rows(directory / f"{batch}_observations.jsonl")
    physical = json.loads((directory / "batch2_controls_and_guidance.json").read_text())
    batch2_hashes = [row["schedule_hash"] for row in physical["physical_batch2"]]
    expected = {row["schedule_hash"] for row in batch1} | set(batch2_hashes)
    assert len(expected) == LIMITS.max_gpu_candidates
    assert len(checks) == 48 and all(row["passed"] for row in checks)
    by_hash = defaultdict(list)
    for row in checks:
        by_hash[row["schedule_hash"]].append(row)
    assert set(by_hash) == expected
    assert all({row["seed"] for row in rows if row.get("mode") is None} == {11, 23, 37}
               and len([row for row in rows if row.get("mode") == "CUDA_GRAPH"]) == 1
               and all(row["rtol"] == CONFIG["rtol"] and row["atol"] == CONFIG["atol"] for row in rows)
               for rows in by_hash.values())
    assert len(topology) == LIMITS.max_profiler_invocations
    assert {row["schedule_hash"] for row in topology} == expected
    assert all(row["topology_agrees"] and row["workspace_audit"]["status"] == "PASS"
               for row in topology)
    assert len(raw) == LIMITS.max_timing_samples
    counts = Counter((row["session"], row["variant"], row["execution_mode"]) for row in raw)
    assert len(counts) == 24 and set(counts.values()) == {25}
    assert all(row["matrix_size"] == 256 and 0 <= row["round"] < 5 and
               0 <= row["iteration"] < 5 and row["elapsed_ms"] > 0 for row in raw)
    by_variant = {(row["session"], row["variant"], row["execution_mode"]): row for row in summaries}
    assert len(by_variant) == 24
    for key, row in by_variant.items():
        selected = [sample for sample in raw if (sample["session"], sample["variant"],
                    sample["execution_mode"]) == key]
        replay = summarize_raw(selected)
        assert len(replay) == 1
        assert replay[0]["round_medians_ms"] == row["round_medians_ms"]
        assert replay[0]["median_ms"] == row["median_ms"]
        assert replay[0]["coefficient_of_variation"] == row["coefficient_of_variation"]
    for mode in MODES:
        validate_observations(observations, candidates, mode)
    known = {row["schedule_hash"]: row for row in candidates}
    for row in observations:
        batch = row["provenance"]["batch"]
        left = by_variant[(batch, row["left"], row["mode"])]
        right = by_variant[(batch, row["right"], row["mode"])]
        rebuilt = make_observation(known[row["left"]], known[row["right"]], mode=row["mode"],
            left_ms=left["round_medians_ms"], right_ms=right["round_medians_ms"],
            provenance=row["provenance"], benchmark=row["benchmark"],
            correctness={row["left"]: "PASS", row["right"]: "PASS"})
        assert rebuilt == row
    first_observations = read_rows(directory / "batch1_observations.jsonl")
    proposal = json.loads((directory / "batch2_proposal.json").read_text())["proposal"]
    validate_proposal(proposal, batch=2, candidates=candidates, batch1=batch1)
    assert selections(candidates, batch1, first_observations, proposal) == physical
    source = observations[0]["provenance"]
    result = {"status": "PASS", "candidate_universe": len(candidates),
              "retained_correctness_pass": sum(row["retained_correctness"] == "PASS" for row in candidates),
              "gpu_candidates": len(expected), "seeded_eager_checks": 36, "graph_checks": 12,
              "topology_checks": len(topology), "raw_timing_samples": len(raw),
              "batch1_pair_observations": len(first_observations),
              "batch2_pair_observations": len(observations) - len(first_observations),
              "directional_batch1_pairs": sum(row["preferred"] is not None for row in first_observations),
              "indeterminate_batch1_pairs": sum(row["preferred"] is None for row in first_observations),
              "replayed_selection_exact_match": True,
              "hardware": {key: source[key] for key in ("device", "compute_capability",
                  "memory_bytes", "torch", "triton", "cuda_runtime")},
              "observation_ids_verified_against_raw_round_medians": len(observations)}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--retain", action="store_true", help="Append derived audit and final calibration")
    args = parser.parse_args()
    directory = args.directory.resolve()
    result = audit(directory)
    if args.retain:
        observations = (read_rows(directory / "batch1_observations.jsonl") +
                        read_rows(directory / "batch2_observations.jsonl"))
        calibration = directory / "calibration_after_batch2.json"
        if not calibration.exists():
            write_json(calibration, {"observations": observations, "evidence_kind": "FRESH_GPU",
                                     "selection_use": "retained for future replay; no Batch 3"})
        audit_path = directory / "validation_audit.json"
        write_json(audit_path, result)
        manifest = {path.name: digest(path) for path in sorted(directory.iterdir())
                    if path.is_file() and path.name != "hash_manifest.json"}
        write_json(directory / "hash_manifest.json", manifest)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
