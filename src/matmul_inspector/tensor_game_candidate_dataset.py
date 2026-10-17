"""Deterministic CPU-only datasets of exact 2x2 bilinear decompositions."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Sequence

from .tensor_game import RankOneAction, matrix_multiplication_tensor, standard_2x2_decomposition, strassen_2x2_decomposition, sum_actions, validate_action
from .tensor_game_hardware_cost import action_cost, decomposition_cost
from .tensor_game_triton_lowering import create_lowering

GENERATOR_VERSION = "tensor_game_candidate_dataset.v1"
TRANSFORMATIONS = ("permutation", "sign_scaling", "index_symmetry")


def _bytes(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _digest(value) -> str:
    return hashlib.sha256(_bytes(value)).hexdigest()


def actions_payload(actions: Sequence[RankOneAction]) -> list[dict[str, list[int]]]:
    return [{"u": list(a.u), "v": list(a.v), "w": list(a.w)} for a in actions]


def validate_decomposition(actions: Sequence[RankOneAction]) -> None:
    if not actions:
        raise ValueError("decomposition is empty")
    for action in actions:
        validate_action(action, (4, 4, 4), allowed_coefficients=None)
        if not all(isinstance(x, int) and not isinstance(x, bool) for x in action.u + action.v + action.w):
            raise ValueError("coefficients must be exact integers")
    if sum_actions(actions, (4, 4, 4)) != matrix_multiplication_tensor(2, 2, 2):
        raise ValueError("decomposition does not exactly reconstruct the target")


def permute_terms(actions: Sequence[RankOneAction], order: Sequence[int]) -> tuple[RankOneAction, ...]:
    if sorted(order) != list(range(len(actions))):
        raise ValueError("term order must be a permutation")
    return tuple(actions[i] for i in order)


def sign_scale_term(action: RankOneAction, factors: tuple[int, int, int]) -> RankOneAction:
    if any(x not in (-1, 1) for x in factors) or factors[0] * factors[1] * factors[2] != 1:
        raise ValueError("exact sign factors must be +/-1 with product +1")
    return RankOneAction(tuple(factors[0] * x for x in action.u), tuple(factors[1] * x for x in action.v), tuple(factors[2] * x for x in action.w))


def index_symmetry(actions: Sequence[RankOneAction], swap_i: bool, swap_j: bool, swap_k: bool) -> tuple[RankOneAction, ...]:
    def vector(values, axes):
        result = [0] * 4
        for first in range(2):
            for second in range(2):
                new_first = 1 - first if axes[0] else first
                new_second = 1 - second if axes[1] else second
                result[2 * new_first + new_second] = values[2 * first + second]
        return tuple(result)
    return tuple(RankOneAction(vector(a.u, (swap_i, swap_k)), vector(a.v, (swap_k, swap_j)), vector(a.w, (swap_i, swap_j))) for a in actions)


def canonical_actions(actions: Sequence[RankOneAction]) -> tuple[RankOneAction, ...]:
    normalized = []
    factors = ((1, 1, 1), (-1, -1, 1), (-1, 1, -1), (1, -1, -1))
    for action in actions:
        normalized.append(min((sign_scale_term(action, f) for f in factors), key=lambda a: a.u + a.v + a.w))
    return tuple(sorted(normalized, key=lambda a: a.u + a.v + a.w))


def structural_signature(actions: Sequence[RankOneAction]) -> dict[str, object]:
    cost = decomposition_cost(actions, matrix_multiplication_tensor(2, 2, 2))
    action_costs = [action_cost(a) for a in actions]
    return {
        "rank": len(actions), "pre_additions": cost.pre_additions, "post_additions": cost.post_additions,
        "coefficient_scalings": cost.coefficient_scalings,
        "u_support_histogram": sorted(Counter(x.u.nonzeros for x in action_costs).items()),
        "v_support_histogram": sorted(Counter(x.v.nonzeros for x in action_costs).items()),
        "w_support_histogram": sorted(Counter(x.w.nonzeros for x in action_costs).items()),
        "negative_coefficient_counts": [sum(x.u.negative_coefficients for x in action_costs), sum(x.v.negative_coefficients for x in action_costs), sum(x.w.negative_coefficients for x in action_costs)],
        "max_abs_coefficient": max(max(x.u.max_abs_coefficient, x.v.max_abs_coefficient, x.w.max_abs_coefficient) for x in action_costs),
    }


def _record(seed_id, family_id, actions, provenance):
    validate_decomposition(actions)
    payload = actions_payload(actions); canonical = actions_payload(canonical_actions(actions)); signature = structural_signature(actions)
    lowering = create_lowering(seed_id, actions)
    return {"candidate_id": _digest(payload), "seed_id": seed_id, "family_id": family_id, "rank": len(actions), "actions": payload, "exact_validation": True, "coefficient_domain": "integers", "transformation_provenance": provenance, "canonical_id": _digest(canonical), "canonicalization": "rank-term order and per-term exact +/-1 factor scaling", "static_cost": asdict(decomposition_cost(actions, matrix_multiplication_tensor(2, 2, 2))), "lowering_metadata": asdict(lowering.metadata), "structural_signature": signature, "structural_signature_id": _digest(signature)}


def generate_candidates(seed: int = 20261016, maximum: int = 128, enabled: Iterable[str] = TRANSFORMATIONS, archive: Path | None = None):
    enabled = tuple(enabled)
    unknown = set(enabled) - set(TRANSFORMATIONS)
    if unknown: raise ValueError(f"unknown transformations: {sorted(unknown)}")
    rng = random.Random(seed)
    seeds = [("conventional", "conventional", standard_2x2_decomposition()), ("strassen", "strassen", strassen_2x2_decomposition())]
    rejections = Counter(); archive_records = 0
    if archive is not None:
        data = json.loads(archive.read_text())
        rows = data if isinstance(data, list) else data.get("decompositions", data.get("solutions", []))
        for index, row in enumerate(rows):
            archive_records += 1
            try:
                values = row.get("actions", row.get("solution"))
                actions = tuple(RankOneAction(tuple(a["u"]), tuple(a["v"]), tuple(a["w"])) for a in values)
                validate_decomposition(actions); seeds.append((f"archive-{index:04d}", f"archive-{index:04d}", actions))
            except Exception as exc: rejections[type(exc).__name__ + ": " + str(exc)] += 1
    generated=[]
    quota = max(1, maximum // len(seeds))
    for seed_id, family_id, base in seeds:
        seed_start = len(generated)
        symmetries=[(False,False,False)] if "index_symmetry" not in enabled else [(bool(i&4),bool(i&2),bool(i&1)) for i in range(8)]
        for symmetry in symmetries:
            transformed=index_symmetry(base,*symmetry)
            variants=[(transformed,[{"type":"index_symmetry","swap_i":symmetry[0],"swap_j":symmetry[1],"swap_k":symmetry[2]}])]
            if "sign_scaling" in enabled:
                for term in range(len(transformed)):
                    factors=(( -1,-1,1),(-1,1,-1),(1,-1,-1))[term%3]
                    changed=list(transformed); changed[term]=sign_scale_term(changed[term],factors)
                    variants.append((tuple(changed),variants[0][1]+[{"type":"sign_scaling","term":term,"factors":list(factors)}]))
            for actions, provenance in variants:
                orders=[tuple(range(len(actions)))]
                if "permutation" in enabled:
                    order=list(range(len(actions))); rng.shuffle(order); orders.append(tuple(order)); orders.append(tuple(reversed(range(len(actions)))))
                for order in orders:
                    candidate=permute_terms(actions,order)
                    prov=provenance+([{"type":"rank_term_permutation","order":list(order)}] if order != tuple(range(len(actions))) else [])
                    try: generated.append(_record(seed_id,family_id,candidate,prov))
                    except Exception as exc: rejections[type(exc).__name__ + ": " + str(exc)] += 1
                    if len(generated) - seed_start >= quota: break
                if len(generated) - seed_start >= quota: break
            if len(generated) - seed_start >= quota: break
        if len(generated)>=maximum: break
    unique={r["candidate_id"]:r for r in generated}; records=sorted(unique.values(),key=lambda r:r["candidate_id"])
    summary={"generated_attempts":len(generated),"valid_attempts":len(generated),"invalid_attempts":sum(rejections.values()),"rejection_reasons":dict(sorted(rejections.items())),"unique_exact_representations":len(records),"duplicate_rate":1-len(records)/len(generated) if generated else 0.0,"canonical_classes":len({r["canonical_id"] for r in records}),"seed_families":len({r["family_id"] for r in records}),"structural_signatures":len({r["structural_signature_id"] for r in records}),"rank_distribution":dict(sorted(Counter(r["rank"] for r in records).items())),"pre_addition_distribution":dict(sorted(Counter(r["static_cost"]["pre_additions"] for r in records).items())),"post_addition_distribution":dict(sorted(Counter(r["static_cost"]["post_additions"] for r in records).items())),"canonical_family_concentration":dict(sorted(Counter(r["canonical_id"] for r in records).items())),"archive_records_considered":archive_records}
    return records, summary


def write_dataset(output: Path, records, manifest):
    if output.exists(): raise FileExistsError(f"output directory already exists: {output}")
    output.mkdir(parents=True)
    candidates=b"".join(_bytes(r) for r in records)
    (output/"candidates.jsonl").write_bytes(candidates)
    manifest=dict(manifest); manifest["files"]={"candidates.jsonl":{"sha256":hashlib.sha256(candidates).hexdigest(),"records":len(records)}}
    (output/"manifest.json").write_bytes(_bytes(manifest))
