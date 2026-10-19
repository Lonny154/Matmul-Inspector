"""CPU-only, bounded HA-Schedule-10 candidate selection and evidence replay.

Timing evidence is ordinal and scoped to one algorithm, size, dtype and mode.
No value in this module is a predicted latency.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics

from .tensor_game_schedule_gpu_study import verify_artifact_hashes

MODES = ("EAGER", "CUDA_GRAPH")
DTYPE = "fp16-input-output/fp32-products"
VERSION = "ha-schedule-10-reselection-v1"


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _index(rows: list[dict]) -> dict[str, dict]:
    result = {row["schedule_hash"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate schedule hash")
    return result


def structural_descriptors(ir: dict) -> dict:
    values = ir["values"]
    operations = ir["operations"]
    kinds = Counter(op["kind"] for op in operations)
    hbm = Counter(value["value_class"] for value in values if value["location"] == "HBM")
    group_sizes = [len(group["op_ids"]) for group in ir["groups"]]
    return {
        "stored_forms": hbm["PREPARED_FORM"],
        "recomputed_forms": kinds["RECOMPUTE_FORM"],
        "hbm_products": hbm["GEMM_PRODUCT"],
        "hbm_partials": hbm["PARTIAL_OUTPUT"],
        "partial_updates": sum("PARTIAL" in state for row in ir["output_transitions"]
                               for state in row["states"]),
        "reconstruction_in_gemm_group": sum(
            any(op["kind"] == "GEMM" for op in operations if op["op_id"] in group["op_ids"])
            and any(op["kind"] == "RECONSTRUCT" for op in operations if op["op_id"] in group["op_ids"])
            for group in ir["groups"]),
        "launch_groups": len(group_sizes),
        "max_group_operations": max(group_sizes),
    }


def load_retained_corpus(root: Path) -> tuple[list[dict], dict]:
    """Join hash-verified HA-5 and HA-9 artifacts without importing CUDA."""
    base = root / "artifacts/tensor-game"
    ha5 = base / "hardware-aware-schedule-calibration-v4"
    ha9 = base / "hardware-aware-general-lowering-ir-v1"
    verify_artifact_hashes(ha5)
    verify_artifact_hashes(ha9)
    source = _index(_rows(ha5 / "candidate_schedules.jsonl"))
    classified = _index(_rows(ha9 / "classification_after.jsonl"))
    irs = _index(_rows(ha9 / "schedule_ir.jsonl"))
    checks: dict[str, list[dict]] = defaultdict(list)
    for row in _rows(ha9 / "correctness.jsonl"):
        checks[row["schedule_hash"]].append(row)
    if not (len(source) == len(classified) == len(irs) == 156 and
            set(source) == set(classified) == set(irs)):
        raise ValueError("retained corpus identities differ")
    candidates = []
    for schedule_hash in sorted(source):
        old, current, ir = source[schedule_hash], classified[schedule_hash], irs[schedule_hash]
        if (old["algorithm"], old["matrix_size"]) != (current["algorithm"], current["matrix_size"]):
            raise ValueError("retained algorithm or size mismatch")
        if ir["ir_hash"] != current["ir_hash"] or ir["source_schedule_hash"] != schedule_hash:
            raise ValueError("retained IR identity mismatch")
        if current["ir_status"] != "IR_VALID" or current["mapping_status"] != "EXACT_MAPPING":
            raise ValueError("HA-9 structural checkpoint changed")
        passed = current["validation_status"] == "PASS"
        if len({row["seed"] for row in checks[schedule_hash]}) < 3 or any(
                row["passed"] != passed for row in checks[schedule_hash]):
            raise ValueError("retained numerical status mismatch")
        candidates.append({
            "candidate_id": f"{old['algorithm']}:{old['matrix_size']}:{schedule_hash}",
            "schedule_hash": schedule_hash, "ir_hash": ir["ir_hash"],
            "lowering_plan_hash": current["lowering_plan_hash"],
            "algorithm": old["algorithm"], "matrix_size": old["matrix_size"],
            "dtype": DTYPE, "policy": old["policy"],
            "descriptors": structural_descriptors(ir),
            "retained_correctness": current["validation_status"],
            "retained_seed_checks": len(checks[schedule_hash]),
        })
    if sum(row["retained_correctness"] == "PASS" for row in candidates) != 132:
        raise ValueError("HA-9 numerical checkpoint changed")
    provenance = {"ha5_manifest_sha256": _sha(ha5 / "hash_manifest.json"),
                  "ha9_manifest_sha256": _sha(ha9 / "hash_manifest.json"),
                  "ha9_correctness_sha256": _sha(ha9 / "correctness.jsonl")}
    return candidates, provenance


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tokens(row: dict) -> frozenset[str]:
    d = row["descriptors"]
    return frozenset((
        f"algorithm={row['algorithm']}",
        f"stored_forms={bool(d['stored_forms'])}",
        f"recomputed_forms={bool(d['recomputed_forms'])}",
        f"hbm_products={bool(d['hbm_products'])}",
        f"hbm_partials={bool(d['hbm_partials'])}",
        f"partial_updates={bool(d['partial_updates'])}",
        f"gemm_recon={bool(d['reconstruction_in_gemm_group'])}",
        f"launch_bin={min(d['launch_groups'] // 4, 5)}",
        f"group_bin={min(d['max_group_operations'], 4)}",
    ))


def _diverse(pool: list[dict], selected: list[dict], budget: int) -> list[dict]:
    seen = set().union(*(_tokens(row) for row in selected)) if selected else set()
    chosen = []
    while pool and len(chosen) < budget:
        row = min(pool, key=lambda item: (-len(_tokens(item) - seen), item["schedule_hash"]))
        chosen.append(row)
        seen.update(_tokens(row))
        pool.remove(row)
    return chosen


def select_batch1(candidates: list[dict], *, size: int = 256) -> list[dict]:
    """Eight configurations: four rank-8, two conventional, two Strassen."""
    result = []
    for algorithm, quota in (("project-rank8", 4), ("conventional", 2), ("strassen", 2)):
        pool = [row for row in candidates if row["matrix_size"] == size and
                row["algorithm"] == algorithm and row["retained_correctness"] == "PASS"]
        if len(pool) < quota:
            raise ValueError(f"insufficient correct {algorithm} candidates")
        result.extend(_diverse(pool, [], quota))
    return result


def make_observation(left: dict, right: dict, *, mode: str, left_ms: list[float],
                     right_ms: list[float], provenance: dict, benchmark: dict,
                     correctness: dict[str, str]) -> dict:
    """One matched comparison. All same-sign round differences support a direction."""
    if mode not in MODES:
        raise ValueError("invalid execution mode")
    if (left["algorithm"], left["matrix_size"], left["dtype"]) != (
            right["algorithm"], right["matrix_size"], right["dtype"]):
        raise ValueError("pair must share algorithm, size and dtype")
    if left["schedule_hash"] == right["schedule_hash"]:
        raise ValueError("pair must contain different candidates")
    if len(left_ms) != len(right_ms) or len(left_ms) < 3 or any(
            value <= 0 for value in (*left_ms, *right_ms)):
        raise ValueError("three or more positive matched round timings are required")
    if provenance.get("evidence_kind") not in (
            "FRESH_GPU", "RETAINED_GPU", "SYNTHETIC_TEST_FIXTURE", "SYNTHETIC_REPLAY_FIXTURE") or not benchmark:
        raise ValueError("observation requires provenance and benchmark configuration")
    if correctness.get(left["schedule_hash"]) != "PASS" or correctness.get(right["schedule_hash"]) != "PASS":
        raise ValueError("numerically failed candidates cannot provide timing preferences")
    differences = [a - b for a, b in zip(left_ms, right_ms)]
    if all(value > 0 for value in differences):
        preferred, confidence = right["schedule_hash"], "CONSISTENT_ROUNDS"
    elif all(value < 0 for value in differences):
        preferred, confidence = left["schedule_hash"], "CONSISTENT_ROUNDS"
    else:
        preferred, confidence = None, "INDETERMINATE"
    payload = {"version": VERSION, "left": left["schedule_hash"], "right": right["schedule_hash"],
               "algorithm": left["algorithm"], "matrix_size": left["matrix_size"],
               "dtype": left["dtype"], "mode": mode,
               "left_descriptors": left["descriptors"], "right_descriptors": right["descriptors"],
               "left_round_medians_ms": left_ms,
               "right_round_medians_ms": right_ms,
               "left_median_ms": statistics.median(left_ms),
               "right_median_ms": statistics.median(right_ms),
               "delta_range_ms": [min(differences), max(differences)],
               "preferred": preferred, "confidence": confidence,
               "reason": "all matched round medians agree" if preferred else
                         "matched rounds overlap or reverse direction",
               "benchmark": benchmark, "provenance": provenance}
    return {"observation_id": hashlib.sha256(json.dumps(payload, sort_keys=True,
            separators=(",", ":")).encode()).hexdigest(), **payload}


def validate_observations(observations: list[dict], candidates: list[dict], mode: str) -> None:
    known = {row["schedule_hash"]: row for row in candidates}
    if mode not in MODES:
        raise ValueError("invalid execution mode")
    ids = [item["observation_id"] for item in observations]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate observation ID")
    for item in observations:
        if item["mode"] != mode:
            continue
        left, right = known[item["left"]], known[item["right"]]
        if (left["algorithm"], left["matrix_size"], left["dtype"]) != (
                item["algorithm"], item["matrix_size"], item["dtype"]):
            raise ValueError("observation context mismatch")
        if left["retained_correctness"] != "PASS" or right["retained_correctness"] != "PASS":
            raise ValueError("failed candidate has observation")
        expected = make_observation(left, right, mode=mode,
            left_ms=item["left_round_medians_ms"], right_ms=item["right_round_medians_ms"],
            provenance=item["provenance"], benchmark=item["benchmark"],
            correctness={item["left"]: "PASS", item["right"]: "PASS"})
        if expected != item:
            raise ValueError("observation content or hash mismatch")


def evidence_votes(observations: list[dict], candidates: list[dict], mode: str) -> dict[str, dict]:
    """Net pair preferences on distinguishing structural tokens; conflicts cancel."""
    validate_observations(observations, candidates, mode)
    known = {row["schedule_hash"]: row for row in candidates}
    vote_count: Counter[str] = Counter()
    signs: dict[str, set[int]] = defaultdict(set)
    sources: dict[str, list[str]] = defaultdict(list)
    for item in observations:
        if item["mode"] != mode or item["preferred"] is None:
            continue
        left, right = known[item["left"]], known[item["right"]]
        winner = left if item["preferred"] == item["left"] else right
        loser = right if winner is left else left
        for token in _tokens(winner) - _tokens(loser):
            key = f"{winner['algorithm']}:{winner['matrix_size']}:{token}"
            vote_count[key] += 1
            signs[key].add(1)
            sources[key].append(item["observation_id"])
        for token in _tokens(loser) - _tokens(winner):
            key = f"{loser['algorithm']}:{loser['matrix_size']}:{token}"
            vote_count[key] -= 1
            signs[key].add(-1)
            sources[key].append(item["observation_id"])
    return {token: {"net_votes": count, "observation_ids": sorted(set(sources[token])),
                    "conflicted": len(signs[token]) > 1}
            for token, count in sorted(vote_count.items())}


def select_batch2(candidates: list[dict], batch1: list[dict], observations: list[dict],
                  *, mode: str, budget: int = 4) -> list[dict]:
    if not 0 <= budget <= 4 or mode not in MODES:
        raise ValueError("invalid batch budget or mode")
    allowed = {row["schedule_hash"] for row in batch1}
    if any(item["left"] not in allowed or item["right"] not in allowed for item in observations):
        raise ValueError("Batch 2 evidence must come from Batch 1")
    votes = evidence_votes(observations, candidates, mode)
    size = {row["matrix_size"] for row in batch1}
    if len(size) != 1:
        raise ValueError("Batch 1 must use one matrix size")
    selected_hashes = {row["schedule_hash"] for row in batch1}
    pool = [row for row in candidates if row["matrix_size"] in size and
            row["retained_correctness"] == "PASS" and row["schedule_hash"] not in selected_hashes]
    seen = set().union(*(_tokens(row) for row in batch1)) if batch1 else set()
    result = []
    while pool and len(result) < budget:
        def keyed_tokens(row):
            return {f"{row['algorithm']}:{row['matrix_size']}:{token}" for token in _tokens(row)}
        def key(row):
            support = sum(votes.get(token, {}).get("net_votes", 0) for token in keyed_tokens(row))
            return (-support, -len(_tokens(row) - seen), row["schedule_hash"])
        row = min(pool, key=key)
        support_tokens = {token: votes[token] for token in sorted(keyed_tokens(row))
                          if token in votes and votes[token]["net_votes"]}
        result.append({"candidate_id": row["candidate_id"], "schedule_hash": row["schedule_hash"],
                       "mode": mode, "ordinal_support": sum(v["net_votes"] for v in support_tokens.values()),
                       "new_descriptor_tokens": sorted(_tokens(row) - seen),
                       "supporting_tokens": support_tokens,
                       "reason": "mode-specific measured pair preferences" if support_tokens else
                                 "deterministic structural coverage and schedule hash"})
        seen.update(_tokens(row))
        pool.remove(row)
    return result


def compare_reselection(frozen: list[dict], updated: list[dict]) -> dict:
    old, new = {row["schedule_hash"] for row in frozen}, {row["schedule_hash"] for row in updated}
    return {"changed": old != new, "added": sorted(new - old), "displaced": sorted(old - new),
            "explanations": [row for row in updated if row["schedule_hash"] in new - old]}
