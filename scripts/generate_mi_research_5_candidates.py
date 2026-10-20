"""Freeze two certified rank-7 variants and three study-specific Triton sources.

This is an offline, bounded source snapshot generator. It does not run CUDA.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

from matmul_inspector.tensor_game import RankOneAction, strassen_2x2_decomposition
from matmul_inspector.tensor_game_candidate_dataset import (
    _digest, _record, structural_signature, validate_decomposition,
)
from matmul_inspector.tensor_game_change_of_basis import transform_decomposition


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "artifacts/tensor-game/change-of-basis-v1/representatives.json"
CORPUS_SHA256 = "d63ac247d2bb5e8c1e482506bceb6a9cf59752d512be47aaf26fde6c2f2b336b"
BASELINE = ROOT / "examples/kernels/mi_research_4c_fused_rank7.py"
BASELINE_SHA256 = "319c6cf6997807199d7b8f20b4971e6f1755a3bf6c764d1f801117af5139987c"
OUT = ROOT / "artifacts/mi-research-5/selection"
SOURCES = ROOT / "examples/kernels"
CHOICES = {
    "peer_corpus": ("0f0a7a60a1ed8043f46a5cf623a87fea35cb700189ac9a2368fade8c231ca8bc", tuple(range(7))),
    "peer_liveness": ("0f0a7a60a1ed8043f46a5cf623a87fea35cb700189ac9a2368fade8c231ca8bc", (1, 4, 2, 0, 5, 6, 3)),
    "scaled_corpus": ("0b88a39c7ca3237f69af292cf82e7ba7c4d3873d4dd4f032192ff22347da7263", tuple(range(7))),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def actions_of(record: dict) -> tuple[RankOneAction, ...]:
    return tuple(RankOneAction(**action) for action in record["actions"])


def live_score(actions: tuple[RankOneAction, ...], order: tuple[int, ...]) -> tuple[int, int]:
    if sorted(order) != list(range(len(actions))):
        raise ValueError("term order must be a permutation")
    live = []
    for position in range(len(order)):
        active = sum(
            any(actions[term].w[output] for term in order[:position + 1])
            and any(actions[term].w[output] for term in order[position + 1:])
            for output in range(4)
        )
        live.append(active)
    return max(live), sum(live)


def select_records() -> dict[str, dict]:
    if digest(CORPUS) != CORPUS_SHA256 or digest(BASELINE) != BASELINE_SHA256:
        raise ValueError("committed corpus or baseline source identity changed")
    records = json.loads(CORPUS.read_text())
    strassen = [row for row in records if row["parent_decomposition"] == "strassen"]
    peer_first = next(row for row in strassen
                      if (row["static_cost"]["pre_additions"],
                          row["static_cost"]["post_additions"],
                          row["static_cost"]["coefficient_scalings"],
                          row["structural_signature"]["max_abs_coefficient"]) ==
                      (10, 8, 0, 1))
    scaled_first = next(row for row in strassen
                        if row["structural_signature"]["max_abs_coefficient"] == 2)
    if (peer_first["candidate_id"] != CHOICES["peer_corpus"][0] or
            scaled_first["candidate_id"] != CHOICES["scaled_corpus"][0]):
        raise ValueError("predeclared representative-order selection changed")
    by_id = {row["candidate_id"]: row for row in records}
    chosen = {candidate_id: by_id[candidate_id] for candidate_id, _ in CHOICES.values()}
    if len(chosen) != 2:
        raise ValueError("expected exactly two alternate representations")
    for row in chosen.values():
        if row["parent_decomposition"] != "strassen" or row["rank"] != 7:
            raise ValueError("selected record is not a rank-7 Strassen equivalent")
        actions = actions_of(row)
        validate_decomposition(actions)
        basis = row["change_of_basis"]
        matrices = tuple(tuple(tuple(int(v) for v in part) for part in basis[key])
                         for key in ("P", "Q", "R"))
        if actions != transform_decomposition(strassen_2x2_decomposition(), *matrices):
            raise ValueError("change-of-basis provenance does not reproduce actions")
        rebuilt = _record(row["seed_id"], row["family_id"], actions, [])
        for field in ("candidate_id", "canonical_id", "structural_signature_id", "static_cost"):
            if rebuilt[field] != row[field]:
                raise ValueError(f"independent {field} check failed")
    peer, scaled = (chosen[CHOICES[label][0]] for label in ("peer_corpus", "scaled_corpus"))
    if (peer["static_cost"]["pre_additions"], peer["static_cost"]["post_additions"],
            peer["static_cost"]["coefficient_scalings"],
            peer["structural_signature"]["max_abs_coefficient"]) != (10, 8, 0, 1):
        raise ValueError("equal-cost peer selection rule failed")
    if scaled["structural_signature"]["max_abs_coefficient"] != 2:
        raise ValueError("coefficient-two selection rule failed")
    if peer["structural_signature_id"] == scaled["structural_signature_id"]:
        raise ValueError("selected signatures are identical")
    original_signature = _digest(structural_signature(strassen_2x2_decomposition()))
    if any(row["structural_signature_id"] == original_signature for row in chosen.values()):
        raise ValueError("selected signature duplicates literal Strassen")
    peer_actions = actions_of(peer)
    best = min((*live_score(peer_actions, order), order)
               for order in itertools.permutations(range(7)))
    if best[2] != CHOICES["peer_liveness"][1]:
        raise ValueError("liveness order changed")
    return chosen


def _coefficient_product(coefficient: int) -> str:
    if coefficient == 1:
        return "p"
    if coefficient == -1:
        return "-p"
    return f"({coefficient} * p)"


def render(record: dict, order: tuple[int, ...]) -> str:
    actions = actions_of(record)
    if sorted(order) != list(range(7)):
        raise ValueError("invalid rank term order")
    baseline = BASELINE.read_text()
    product = baseline.split("@triton.jit\ndef _product(", 1)[1].split(
        "@triton.jit\ndef _strassen_kernel(", 1)[0]
    product = "@triton.jit\ndef _product(" + product
    lines = [
        f'"""Frozen MI-Research-5 rank-7 variant {record["candidate_id"]}.',
        f"Term order: {order}. Generated before GPU execution.\n\"\"\"",
        "", "import torch", "import triton", "import triton.language as tl", "",
        'PARAMETERS = {"block_m": 16, "block_n": 16, "block_k": 32,',
        '              "num_warps": 4, "num_stages": 2, "group_size_m": 1}', "",
        product.rstrip(), "", "", "@triton.jit", "def _variant_kernel(",
        "    a_ptr, b_ptr, out_ptr, half: tl.constexpr,",
        "    stride_am: tl.constexpr, stride_ak: tl.constexpr,",
        "    stride_bk: tl.constexpr, stride_bn: tl.constexpr,",
        "    stride_om: tl.constexpr, stride_on: tl.constexpr,",
        "    block_m: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr,",
        "):",
        "    rows = tl.program_id(0) * block_m + tl.arange(0, block_m)",
        "    cols = tl.program_id(1) * block_n + tl.arange(0, block_n)",
        "    inner = tl.arange(0, block_k)",
    ]
    initialized: set[int] = set()
    for term in order:
        action = actions[term]
        lines += [f"    # Frozen corpus term {term + 1}.",
                  "    p = _product(a_ptr, b_ptr, rows, cols, inner, half, stride_am, stride_ak,",
                  "                 stride_bk, stride_bn, block_m, block_n, block_k,",
                  "                 " + ", ".join(map(str, action.u + action.v)) + ")"]
        for output, coefficient in enumerate(action.w):
            if coefficient:
                name = ("c00", "c01", "c10", "c11")[output]
                operator = "+=" if output in initialized else "="
                lines.append(f"    {name} {operator} {_coefficient_product(coefficient)}")
                initialized.add(output)
    if initialized != set(range(4)):
        raise ValueError("one or more output quadrants were omitted")
    lines += [
        "    mask = (rows[:, None] < half) & (cols[None, :] < half)",
        "    tl.store(out_ptr + rows[:, None] * stride_om + cols[None, :] * stride_on,",
        "             c00.to(tl.float16), mask)",
        "    tl.store(out_ptr + rows[:, None] * stride_om + (half + cols[None, :]) * stride_on,",
        "             c01.to(tl.float16), mask)",
        "    tl.store(out_ptr + (half + rows[:, None]) * stride_om + cols[None, :] * stride_on,",
        "             c10.to(tl.float16), mask)",
        "    tl.store(out_ptr + (half + rows[:, None]) * stride_om + (half + cols[None, :]) * stride_on,",
        "             c11.to(tl.float16), mask)", "", "",
        "def matmul_into(a, b, out, parameters):",
        "    if parameters != PARAMETERS:",
        '        raise ValueError("MI-Research-5 uses one frozen 16x16x32 configuration")',
        "    if (a.ndim != 2 or b.ndim != 2 or out.ndim != 2 or a.shape != b.shape or",
        "            a.shape != out.shape or a.shape[0] not in (128, 256) or",
        "            a.dtype != torch.float16 or b.dtype != torch.float16 or",
        "            out.dtype != torch.float16 or not a.is_cuda or not b.is_cuda or not out.is_cuda):",
        '        raise ValueError("expected matching 128 or 256 square CUDA FP16 matrices")',
        "    half = a.shape[0] // 2",
        "    grid = (triton.cdiv(half, 16), triton.cdiv(half, 16))",
        "    return _variant_kernel[grid] (",
        "        a, b, out, half,",
        "        a.stride(0), a.stride(1), b.stride(0), b.stride(1),",
        "        out.stride(0), out.stride(1),",
        "        16, 16, 32, num_warps=4, num_stages=2)",
    ]
    return "\n".join(lines) + "\n"


def freeze() -> dict:
    if OUT.exists() or any((SOURCES / f"mi_research_5_{label}.py").exists()
                           for label in CHOICES):
        raise FileExistsError("selection or generated source already exists")
    records = select_records()
    OUT.mkdir(parents=True)
    source_hashes = {}
    for label, (candidate_id, order) in CHOICES.items():
        path = SOURCES / f"mi_research_5_{label}.py"
        path.write_text(render(records[candidate_id], order))
        source_hashes[label] = digest(path)
    selection = {
        "schema_version": 1,
        "corpus_sha256": CORPUS_SHA256,
        "baseline_sha256": BASELINE_SHA256,
        "records": list(records.values()),
        "choices": {label: {"candidate_id": candidate_id, "order": list(order),
                            "liveness": list(live_score(actions_of(records[candidate_id]), order)),
                            "source_sha256": source_hashes[label]}
                    for label, (candidate_id, order) in CHOICES.items()},
        "selection_rule": "first rank-7 same-cost representative and first coefficient-two representative; fixed lexicographic liveness schedule",
    }
    (OUT / "selection.json").write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
    (OUT / "hash_manifest.json").write_text(json.dumps(
        {"selection.json": digest(OUT / "selection.json")}, indent=2, sort_keys=True) + "\n")
    return verify()


def verify() -> dict:
    records = select_records()
    manifest = json.loads((OUT / "hash_manifest.json").read_text())
    if manifest != {"selection.json": digest(OUT / "selection.json")}:
        raise ValueError("selection manifest mismatch")
    selection = json.loads((OUT / "selection.json").read_text())
    if (selection["corpus_sha256"] != CORPUS_SHA256 or
            selection["baseline_sha256"] != BASELINE_SHA256 or
            {r["candidate_id"]: r for r in selection["records"]} != records):
        raise ValueError("selection provenance mismatch")
    for label, (candidate_id, order) in CHOICES.items():
        choice = selection["choices"][label]
        path = SOURCES / f"mi_research_5_{label}.py"
        if (choice["candidate_id"] != candidate_id or choice["order"] != list(order) or
                choice["liveness"] != list(live_score(actions_of(records[candidate_id]), order)) or
                choice["source_sha256"] != digest(path) or
                path.read_text() != render(records[candidate_id], order)):
            raise ValueError(f"{label} source or choice mismatch")
    return {"status": "PASS", "representations": len(records), "sources": len(CHOICES)}


def verify_source_hash(label: str) -> str:
    """Return the frozen digest only after checking the complete selection."""
    verify()
    return json.loads((OUT / "selection.json").read_text())["choices"][label]["source_sha256"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify() if args.verify else freeze(), sort_keys=True))


if __name__ == "__main__":
    main()
