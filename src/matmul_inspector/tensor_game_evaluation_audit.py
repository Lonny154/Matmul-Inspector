"""Versioned, paired proposal-quality audit for TG-3B checkpoints."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import tempfile
import time
from typing import Any, Sequence

from .tensor_game import (
    RankOneAction,
    matrix_multiplication_tensor,
    standard_2x2_decomposition,
    strassen_2x2_decomposition,
)
from .tensor_game_dataset import SyntheticDataset, action_from_dict
from .tensor_game_learning import (
    PolicyValueConfig,
    create_policy_value_model,
    rank_complete_actions_audited,
    valid_action_log_probabilities,
)
from .tensor_game_search import (
    HybridSampledPolicy,
    ResidualInformedPolicy,
    UniformSampledPolicy,
    canonicalize_action,
)

AUDIT_SCHEMA_VERSION = 2
POLICY_VERSIONS = {
    "trained": "tg3b.factorized_mlp.v1",
    "untrained": "tg3b.factorized_mlp.v1",
    "uniform_sampled": "tg2a.uniform_sampled.v1",
    "residual_informed": "tg2b.residual_informed.v1",
    "hybrid": "tg2b.hybrid.v1",
}


def _torch():
    import torch

    return torch


def _flat_action(action: RankOneAction) -> tuple[int, ...]:
    return action.u + action.v + action.w


def _known_actions(sample: dict[str, Any]) -> set[RankOneAction]:
    return {canonicalize_action(action_from_dict(value)) for value in sample["next_actions"]}


def reference_exclusion_audit(dataset: SyntheticDataset) -> dict[str, Any]:
    """Check post hoc that no stored origin is a known 2x2 reference trajectory."""
    target = matrix_multiplication_tensor(2, 2, 2)

    def multiset(actions: Sequence[RankOneAction]):
        return sorted(_flat_action(canonicalize_action(action)) for action in actions)

    standard = multiset(standard_2x2_decomposition())
    strassen = multiset(strassen_2x2_decomposition())
    target_matches = standard_matches = strassen_matches = 0
    for record in dataset.decompositions:
        stored_target = tuple(tuple(tuple(row) for row in plane) for plane in record["target"])
        factors = tuple(action_from_dict(value) for value in record["factors"])
        target_matches += stored_target == target
        standard_matches += multiset(factors) == standard
        strassen_matches += multiset(factors) == strassen
    return {
        "matrix_multiplication_target_matches": target_matches,
        "standard_reference_multiset_matches": standard_matches,
        "strassen_reference_multiset_matches": strassen_matches,
        "passed": target_matches == standard_matches == strassen_matches == 0,
        "timing": "post_hoc_dataset_audit_only; references are not supplied to proposal policies",
    }


def _known_mass(logits, known: set[RankOneAction]) -> float:
    torch = _torch()
    targets = torch.tensor([[_flat_action(action) for action in sorted(known, key=_flat_action)]], dtype=torch.int64, device=logits.device)
    mask = torch.ones((1, len(known)), dtype=torch.bool, device=logits.device)
    return float(torch.exp(valid_action_log_probabilities(logits.unsqueeze(0), targets, mask))[0])


def _model_logits(model, samples: Sequence[dict[str, Any]], device, batch_size: int = 128):
    torch = _torch()
    result = {}
    model.eval()
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch = samples[start : start + batch_size]
            residuals = torch.tensor([sample["residual"] for sample in batch], dtype=torch.float32, device=device)
            logits, values = model(residuals)
            for index, sample in enumerate(batch):
                result[sample["sample_id"]] = (logits[index], float(values[index]))
    return result, time.perf_counter() - started


def _neural_outcome(logits, known: set[RankOneAction], shape, proposal_count: int):
    proposals, effort = rank_complete_actions_audited(logits, shape, proposal_count)
    actions = [action for action, _ in proposals]
    return {
        "hit": any(action in known for action in actions),
        "top1_hit": bool(actions and actions[0] in known),
        "known_action_probability_mass": _known_mass(logits, known),
        "returned": len(actions),
        "valid_unique": len(set(actions)),
        **effort,
    }


def _metrics_snapshot(policy) -> tuple[int, int, int, int, float]:
    metrics = policy.metrics
    return (
        metrics.attempted,
        metrics.valid,
        metrics.accepted,
        metrics.duplicates,
        metrics.generation_seconds,
    )


def _handcrafted_outcome(policy, residual, remaining_moves, rng, proposal_count: int):
    before = _metrics_snapshot(policy)
    started = time.perf_counter()
    proposals = policy.propose_actions(residual, remaining_moves, rng, proposal_count)
    wall = time.perf_counter() - started
    after = _metrics_snapshot(policy)
    actions = [item.action for item in proposals]
    return actions, {
        "requested": proposal_count,
        "attempted": after[0] - before[0],
        "valid_before_deduplication": after[1] - before[1],
        "accepted": after[2] - before[2],
        "duplicates_rejected": after[3] - before[3],
        "returned": len(actions),
        "valid_unique": len(set(actions)),
        "generation_seconds": after[4] - before[4],
        "wall_seconds": wall,
    }


def _aggregate(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    opportunities = len(records)
    hits = sum(record["hit"] for record in records)
    return {
        "unique_residual_states": len({record["sample_id"] for record in records}),
        "decomposition_groups": len({record["decomposition_id"] for record in records}),
        "state_seed_evaluations": opportunities,
        "successful_state_seed_evaluations": hits,
        "aggregate_hit_rate": hits / opportunities if opportunities else 0.0,
        "requested_action_slots": sum(record["requested"] for record in records),
        "returned_actions": sum(record["returned"] for record in records),
        "valid_unique_actions": sum(record["valid_unique"] for record in records),
        "effective_valid_action_fraction": (
            sum(record["valid_unique"] for record in records)
            / sum(record["requested"] for record in records)
            if records else 0.0
        ),
        "proposal_generation_seconds": sum(record.get("generation_seconds", 0.0) for record in records),
        "proposal_attempts": sum(record.get("attempted", record.get("decoded_candidates", 0)) for record in records),
        "internal_candidates_considered": sum(record.get("internal_candidates_considered", 0) for record in records),
        "invalid_rejected": sum(record.get("invalid_rejected", 0) for record in records),
        "duplicates_rejected": sum(
            record.get("duplicates_rejected", record.get("canonical_duplicates_rejected", 0))
            for record in records
        ),
    }


def _per_seed(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for seed in sorted({record["seed"] for record in records if record["seed"] is not None}):
        group = [record for record in records if record["seed"] == seed]
        result[str(seed)] = _aggregate(group)
    return result


def paired_disagreements(
    learned: dict[str, bool], heuristic_records: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    counts = {"both_hit": 0, "learned_only": 0, "heuristic_only": 0, "neither": 0}
    per_seed: dict[str, dict[str, int]] = {}
    rows = []
    for record in heuristic_records:
        learned_hit = learned[record["sample_id"]]
        heuristic_hit = bool(record["hit"])
        key = (
            "both_hit" if learned_hit and heuristic_hit else
            "learned_only" if learned_hit else
            "heuristic_only" if heuristic_hit else "neither"
        )
        counts[key] += 1
        seed_counts = per_seed.setdefault(str(record["seed"]), {name: 0 for name in counts})
        seed_counts[key] += 1
        rows.append(
            {
                "decomposition_id": record["decomposition_id"],
                "learned": int(learned_hit),
                "heuristic": int(heuristic_hit),
            }
        )
    total = len(rows)
    difference = (counts["learned_only"] - counts["heuristic_only"]) / total if total else 0.0
    for seed_counts in per_seed.values():
        seed_total = sum(seed_counts.values())
        seed_counts["state_evaluations"] = seed_total
        seed_counts["learned_minus_heuristic"] = (
            (seed_counts["learned_only"] - seed_counts["heuristic_only"]) / seed_total
            if seed_total else 0.0
        )
    return {**counts, "state_seed_evaluations": total, "learned_minus_heuristic": difference, "per_seed": per_seed, "rows": rows}


def decomposition_cluster_bootstrap(
    paired_rows: Sequence[dict[str, Any]], *, seed: int = 20261013, replicates: int = 2000
) -> dict[str, Any]:
    """Descriptive percentile interval resampling held-out decomposition groups."""
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    groups: dict[str, list[int]] = {}
    for row in paired_rows:
        groups.setdefault(row["decomposition_id"], []).append(row["learned"] - row["heuristic"])
    if not groups:
        raise ValueError("paired rows must not be empty")
    identifiers = sorted(groups)
    rng = random.Random(seed)
    estimates = []
    for _ in range(replicates):
        sampled = [rng.choice(identifiers) for _ in identifiers]
        values = [value for identifier in sampled for value in groups[identifier]]
        estimates.append(sum(values) / len(values))
    estimates.sort()

    def percentile(probability: float) -> float:
        index = round(probability * (len(estimates) - 1))
        return estimates[index]

    return {
        "method": "decomposition_cluster_percentile_bootstrap",
        "grouping_unit": "originating_decomposition",
        "groups": len(groups),
        "replicates": replicates,
        "seed": seed,
        "confidence_level": 0.95,
        "lower": percentile(0.025),
        "upper": percentile(0.975),
        "interpretation": "descriptive interval; residuals and repeated policy seeds are not treated as independent groups",
    }


def run_evaluation_audit(
    trained_model,
    untrained_model,
    dataset: SyntheticDataset,
    model_config: PolicyValueConfig,
    *,
    device,
    proposal_count: int = 16,
    policy_seeds: Sequence[int] = (3, 11, 42),
    untrained_initialization_seed: int = 42,
) -> dict[str, Any]:
    torch = _torch()
    samples = [sample for sample in dataset.samples if sample["split"] == "test"]
    trained_logits, trained_forward_seconds = _model_logits(trained_model, samples, device)
    untrained_logits, untrained_forward_seconds = _model_logits(untrained_model, samples, device)
    records: dict[str, list[dict[str, Any]]] = {
        "trained": [], "untrained": [], "uniform_sampled": [], "residual_informed": [], "hybrid": []
    }
    for sample in samples:
        known = _known_actions(sample)
        common = {
            "sample_id": sample["sample_id"],
            "residual_digest": sample["residual_digest"],
            "decomposition_id": sample["decomposition_id"],
            "known_action_count": len(known),
            "requested": proposal_count,
            "seed": None,
        }
        records["trained"].append({**common, **_neural_outcome(trained_logits[sample["sample_id"]][0], known, model_config.shape, proposal_count)})
        records["untrained"].append({**common, **_neural_outcome(untrained_logits[sample["sample_id"]][0], known, model_config.shape, proposal_count)})
    factories = {
        "uniform_sampled": lambda: UniformSampledPolicy(model_config.shape),
        "residual_informed": lambda: ResidualInformedPolicy(model_config.shape),
        "hybrid": lambda: HybridSampledPolicy(model_config.shape),
    }
    for name, factory in factories.items():
        for seed in policy_seeds:
            policy = factory()
            rng = random.Random(seed)
            for sample in samples:
                residual = tuple(tuple(tuple(row) for row in plane) for plane in sample["residual"])
                actions, effort = _handcrafted_outcome(
                    policy, residual, sample["known_completion_length"], rng, proposal_count
                )
                known = _known_actions(sample)
                records[name].append(
                    {
                        "sample_id": sample["sample_id"],
                        "residual_digest": sample["residual_digest"],
                        "decomposition_id": sample["decomposition_id"],
                        "known_action_count": len(known),
                        "seed": seed,
                        "hit": any(action in known for action in actions),
                        **effort,
                    }
                )
    summaries = {}
    for name, policy_records in records.items():
        summaries[name] = {
            "deterministic": name in {"trained", "untrained"},
            "policy_version": POLICY_VERSIONS[name],
            "aggregate": _aggregate(policy_records),
            "per_seed": _per_seed(policy_records),
        }
        if name in {"trained", "untrained"}:
            forward_seconds = trained_forward_seconds if name == "trained" else untrained_forward_seconds
            summaries[name]["model_forward_seconds"] = forward_seconds
            summaries[name]["total_proposal_wall_seconds"] = (
                summaries[name]["aggregate"]["proposal_generation_seconds"] + forward_seconds
            )
            summaries[name]["top1_known_action_match"] = sum(row["top1_hit"] for row in policy_records) / len(policy_records)
            summaries[name]["average_known_action_probability_mass"] = sum(
                row["known_action_probability_mass"] for row in policy_records
            ) / len(policy_records)
        else:
            summaries[name]["total_proposal_wall_seconds"] = summaries[name]["aggregate"][
                "proposal_generation_seconds"
            ]
    learned_hits = {row["sample_id"]: bool(row["hit"]) for row in records["trained"]}
    paired = paired_disagreements(learned_hits, records["residual_informed"])
    interval = decomposition_cluster_bootstrap(paired.pop("rows"))
    return {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "dataset_id": dataset.manifest["dataset_id"],
        "test_decomposition_groups": sorted({sample["decomposition_id"] for sample in samples}),
        "unique_residual_states": len({sample["residual_digest"] for sample in samples}),
        "test_residual_rows": len(samples),
        "proposal_count": proposal_count,
        "policy_seeds": list(policy_seeds),
        "untrained_initialization_seed": untrained_initialization_seed,
        "summaries": summaries,
        "paired_trained_vs_residual_informed": {**paired, "uncertainty": interval},
        "records": records,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "device": str(device),
        },
        "metric_definitions": {
            "hit": "at least one returned canonical complete action exactly matches a stored known continuation",
            "valid_unique": "returned actions passing coefficient, nonzero-factor, and canonical deduplication checks",
            "known_action_scope": "stored continuations are used only after proposal generation and are not exhaustive",
        },
        "reference_exclusion": reference_exclusion_audit(dataset),
        "limitations": [
            "The trained and untrained decoders are deterministic and are evaluated once per residual.",
            "Handcrafted policies are stochastic and are evaluated once per residual for each listed seed.",
            "Equal requested output counts do not imply equal internal candidates, attempts, or CPU cost.",
            "Known-action hits are proposal-quality evidence, not complete decomposition or optimality evidence.",
            "The cluster bootstrap is descriptive and does not establish statistical superiority.",
        ],
    }


def checkpoint_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_audit(output: Path, audit: dict[str, Any]) -> None:
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(audit, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, output)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def load_audit(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("audit_schema_version") != AUDIT_SCHEMA_VERSION:
        raise ValueError("unsupported TensorGame evaluation audit schema")
    return value
