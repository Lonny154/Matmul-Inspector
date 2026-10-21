"""CPU-only, exact 3x3 Laderman feasibility study and artifact replay.

The source snapshot is the University of Lille 3x3x3 Maple tensor triads.
Its parameter l is set to 1. Its third factor is transposed as specified by
the source's Trace(Transpose(U).A) * Trace(Transpose(V).B) * Transpose(W).
No downloaded text is executed.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import re

from matmul_inspector.tensor_game import (
    RankOneAction, matrix_multiplication_tensor, sum_actions,
)
from matmul_inspector.tensor_game_hardware_cost import decomposition_cost

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/mi-research-6"
SOURCE = OUT / "laderman_3x3x3_source.mpl"
SOURCE_URL = "https://fmm.univ-lille.fr/3x3x3_tensor.mpl.bz2"
SHAPE = (9, 9, 9)
BLOCK_SIZE = 2


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_laderman(source: str) -> tuple[RankOneAction, ...]:
    """Extract 23 triads with a restricted integer-token parser (l=1)."""
    if "Tensor:=TriadSet([" not in source:
        raise ValueError("missing tensor triads")
    if ("Transpose(op([1,i,1,1],Tensor)).A" not in source
            or "Transpose(op([1,i,1,2],Tensor)).B" not in source
            or "Transpose(op([1,i,1,3],Tensor))" not in source):
        raise ValueError("missing source coefficient convention")
    body = source.split("Tensor:=TriadSet([", 1)[1].split("]):", 1)[0]
    matrices = []
    for match in re.finditer(r"Matrix\(3,\s*3,\s*", body):
        start = match.end()
        if body[start:start + 2] != "[[":
            raise ValueError("unexpected matrix encoding")
        depth = 0
        end = None
        for pos in range(start, len(body)):
            depth += (body[pos] == "[") - (body[pos] == "]")
            if depth == 0:
                end = pos + 1
                break
        if end is None:
            raise ValueError("unterminated matrix")
        rows = []
        for row in re.findall(r"\[([^\[\]]+)\]", body[start:end]):
            values = []
            for token in row.split(","):
                token = token.strip()
                token = {"l": "1", "-l": "-1", "1/l": "1", "-1/l": "-1"}.get(token, token)
                if not re.fullmatch(r"-?\d+", token):
                    raise ValueError(f"unsupported coefficient: {token!r}")
                values.append(int(token))
            rows.append(values)
        if len(rows) != 3 or any(len(row) != 3 for row in rows):
            raise ValueError("source matrix is not 3x3")
        matrices.append(rows)
    if len(matrices) != 69 or body.count("Triad([") != 23:
        raise ValueError("expected exactly 23 three-matrix triads")
    actions = []
    for index in range(0, 69, 3):
        u, v, source_w = matrices[index:index + 3]
        actions.append(RankOneAction(
            (value for row in u for value in row),
            (value for row in v for value in row),
            (source_w[col][row] for row in range(3) for col in range(3)),
        ))
    return tuple(actions)


def conventional() -> tuple[RankOneAction, ...]:
    actions = []
    for i in range(3):
        for k in range(3):
            for j in range(3):
                u, v, w = [0] * 9, [0] * 9, [0] * 9
                u[3 * i + k] = v[3 * k + j] = w[3 * i + j] = 1
                actions.append(RankOneAction(u, v, w))
    return tuple(actions)


def coefficients(actions: tuple[RankOneAction, ...]) -> list[dict[str, list[int]]]:
    return [{"u": list(a.u), "v": list(a.v), "w": list(a.w)} for a in actions]


def zero_block(block_size: int = BLOCK_SIZE) -> list[list[int]]:
    return [[0] * block_size for _ in range(block_size)]


def block_matmul(a: list[list[int]], b: list[list[int]]) -> list[list[int]]:
    size = len(a)
    return [[sum(a[i][k] * b[k][j] for k in range(size)) for j in range(size)] for i in range(size)]


def direct_blocks(left: list[list[list[int]]], right: list[list[list[int]]]) -> list[list[list[int]]]:
    return [
        [[sum(left[3 * i + k][row][inner] * right[3 * k + j][inner][col]
               for k in range(3) for inner in range(BLOCK_SIZE))
          for col in range(BLOCK_SIZE)] for row in range(BLOCK_SIZE)]
        for i in range(3) for j in range(3)
    ]


def evaluate_blocks(actions: tuple[RankOneAction, ...], order: tuple[int, ...],
                    left: list[list[list[int]]], right: list[list[list[int]]]) -> list[list[list[int]]]:
    """Independent matrix-valued interpreter. Product order is U @ V."""
    outputs = [zero_block() for _ in range(9)]
    for index in order:
        action = actions[index]
        u = [[sum(action.u[b] * left[b][i][j] for b in range(9)) for j in range(BLOCK_SIZE)]
             for i in range(BLOCK_SIZE)]
        v = [[sum(action.v[b] * right[b][i][j] for b in range(9)) for j in range(BLOCK_SIZE)]
             for i in range(BLOCK_SIZE)]
        product = block_matmul(u, v)
        for output, coefficient in enumerate(action.w):
            if coefficient:
                for i in range(BLOCK_SIZE):
                    for j in range(BLOCK_SIZE):
                        outputs[output][i][j] += coefficient * product[i][j]
    return outputs


def block_cases() -> list[tuple[str, list, list]]:
    zero = [zero_block() for _ in range(9)]
    identity = [zero_block() for _ in range(9)]
    for i in range(3):
        identity[3 * i + i] = [[1, 0], [0, 1]]
    sparse_a = [zero_block() for _ in range(9)]
    sparse_b = [zero_block() for _ in range(9)]
    sparse_a[2][0][1] = 3
    sparse_a[7][1][0] = -2
    sparse_b[6][1][1] = 4
    sparse_b[5][0][0] = -5
    cases = [("zero", zero, zero), ("identity_left", identity, sparse_b),
             ("identity_right", sparse_a, identity), ("sparse", sparse_a, sparse_b)]
    for seed in (0, 1, 2, 3, 17):
        rng = random.Random(seed)
        left = [[[rng.randrange(-3, 4) for _ in range(BLOCK_SIZE)] for _ in range(BLOCK_SIZE)] for _ in range(9)]
        right = [[[rng.randrange(-3, 4) for _ in range(BLOCK_SIZE)] for _ in range(BLOCK_SIZE)] for _ in range(9)]
        cases.append((f"random_{seed}", left, right))
    return cases


def liveness_order(actions: tuple[RankOneAction, ...]) -> tuple[int, ...]:
    """Greedy deterministic output-lifetime order, not a general optimizer."""
    remaining = set(range(len(actions)))
    seen: set[int] = set()
    order = []
    while remaining:
        def score(index: int) -> tuple[int, int, int]:
            future = remaining - {index}
            next_seen = seen | {i for i, x in enumerate(actions[index].w) if x}
            active = sum(any(actions[j].w[i] for j in future) for i in next_seen)
            # Prefer finishing output accumulators, then fewer active values.
            completed = sum(i in next_seen and not any(actions[j].w[i] for j in future) for i in range(9))
            return (active, -completed, index)
        chosen = min(remaining, key=score)
        remaining.remove(chosen)
        seen.update(i for i, x in enumerate(actions[chosen].w) if x)
        order.append(chosen)
    return tuple(order)


def schedule_metrics(actions: tuple[RankOneAction, ...], order: tuple[int, ...],
                     materialize_products: bool) -> dict:
    if sorted(order) != list(range(len(actions))):
        raise ValueError("schedule is not a permutation")
    first = [next(pos for pos, index in enumerate(order) if actions[index].w[out]) for out in range(9)]
    last = [max(pos for pos, index in enumerate(order) if actions[index].w[out]) for out in range(9)]
    live_outputs = [sum(first[out] <= pos < last[out] for out in range(9)) for pos in range(len(order))]
    form_uses: dict[tuple[str, tuple[int, ...]], list[int]] = {}
    for pos, index in enumerate(order):
        for side, vector in (("A", actions[index].u), ("B", actions[index].v)):
            if sum(bool(x) for x in vector) == 1 and 1 in vector:
                continue
            form_uses.setdefault((side, vector), []).append(pos)
    live_forms = [sum(uses[0] <= pos < uses[-1] for uses in form_uses.values()) for pos in range(len(order))]
    return {
        "order": list(order), "live_output_accumulators_after_term": live_outputs,
        "peak_live_output_accumulators": max(live_outputs),
        "sum_live_output_accumulator_steps": sum(live_outputs),
        "peak_live_prepared_forms_between_uses": max(live_forms, default=0),
        "materialized_product_count": len(actions) if materialize_products else 0,
        "peak_current_product_count": 0 if materialize_products else 1,
        "output_first_term_position": first, "output_last_term_position": last,
    }


def dag(actions: tuple[RankOneAction, ...]) -> dict:
    nodes = []
    for side in ("A", "B"):
        for index in range(9):
            nodes.append({"id": f"{side}{index}", "kind": "INPUT_BLOCK", "deps": []})
    forms: dict[tuple[str, tuple[int, ...]], str] = {}
    products = []
    for term, action in enumerate(actions):
        inputs = []
        for side, vector in (("A", action.u), ("B", action.v)):
            if sum(bool(x) for x in vector) == 1 and 1 in vector:
                inputs.append(f"{side}{vector.index(1)}")
                continue
            key = (side, vector)
            if key not in forms:
                name = f"{side}_FORM{sum(k[0] == side for k in forms):02d}"
                forms[key] = name
                nodes.append({"id": name, "kind": "LINEAR_FORM", "side": side,
                              "coefficients": list(vector),
                              "deps": [f"{side}{i}" for i, x in enumerate(vector) if x]})
            inputs.append(forms[key])
        name = f"M{term:02d}"
        products.append(name)
        nodes.append({"id": name, "kind": "BLOCK_PRODUCT", "term": term, "deps": inputs})
    for output in range(9):
        source_terms = [i for i, action in enumerate(actions) if action.w[output]]
        name = f"R{output}"
        nodes.append({"id": name, "kind": "OUTPUT_RECONSTRUCTION", "output": output,
                      "coefficients": [actions[i].w[output] for i in source_terms],
                      "deps": [products[i] for i in source_terms]})
        nodes.append({"id": f"O{output}", "kind": "OUTPUT_WRITE", "deps": [name]})
    ids = {node["id"] for node in nodes}
    if len(ids) != len(nodes) or any(set(node["deps"]) - ids for node in nodes):
        raise ValueError("invalid DAG")
    return {"nodes": nodes, "node_count": len(nodes), "edge_count": sum(len(n["deps"]) for n in nodes),
            "form_count": len(forms), "product_count": len(products), "output_count": 9}


def algorithm_record(name: str, actions: tuple[RankOneAction, ...]) -> tuple[dict, dict]:
    target = matrix_multiplication_tensor(3, 3, 3)
    reconstructed = sum_actions(actions, SHAPE)
    mismatches = [[i, j, k, reconstructed[i][j][k], target[i][j][k]]
                  for i in range(9) for j in range(9) for k in range(9)
                  if reconstructed[i][j][k] != target[i][j][k]]
    if mismatches:
        raise ValueError(f"{name} fails exact tensor reconstruction: {mismatches[:3]}")
    natural = tuple(range(len(actions)))
    liv = liveness_order(actions)
    cases = []
    for label, left, right in block_cases():
        expected = direct_blocks(left, right)
        for schedule_name, order in (("natural", natural), ("liveness", liv)):
            result = evaluate_blocks(actions, order, left, right)
            if result != expected:
                raise ValueError(f"{name} {label} {schedule_name} block failure")
            cases.append({"case": label, "schedule": schedule_name,
                          "input_sha256": digest(canonical_bytes([left, right])),
                          "output_sha256": digest(canonical_bytes(result)), "mismatch_count": 0})
    cost = decomposition_cost(actions, target)
    nonzero = Counter(abs(x) for a in actions for x in a.u + a.v + a.w if x)
    output_terms = [sum(a.w[o] != 0 for a in actions) for o in range(9)]
    vector_uses = Counter(("A", a.u) for a in actions) + Counter(("B", a.v) for a in actions)
    record = {
        "algorithm": name, "rank": len(actions), "coefficient_sha256": digest(canonical_bytes(coefficients(actions))),
        "tensor_positions_checked": 729, "tensor_nonzero_count": 27, "exact_mismatch_count": 0,
        "direct_matrix_block_cases": cases, "static_cost": asdict(cost),
        "coefficient_magnitude_histogram": dict(sorted(nonzero.items())),
        "max_coefficient_magnitude": max(nonzero), "output_product_contributions": output_terms,
        "unique_operand_forms": len(vector_uses),
        "reused_operand_forms": sum(count > 1 for count in vector_uses.values()),
        "maximum_operand_form_reuse": max(vector_uses.values()),
        "natural_schedule": schedule_metrics(actions, natural, True),
        "liveness_schedule": schedule_metrics(actions, liv, False),
    }
    return record, dag(actions)


def build() -> dict[str, bytes]:
    source_bytes = SOURCE.read_bytes()
    source = source_bytes.decode("utf-8")
    candidates = {"conventional_27": conventional(), "laderman_23_l_equals_1": parse_laderman(source)}
    records, dags = {}, {}
    for name, actions in candidates.items():
        records[name], dags[name] = algorithm_record(name, actions)
    hardware = {
        "device": "RTX 4060 Ti (target from retained studies; no device execution in this run)",
        "example_matrix_size": 96, "block_size": 32, "tile_mn": 16,
        "configured_output_tile_ctas_per_product": 4,
        "fused_output_tile_ctas": 4,
        "fused_fp32_output_accumulator_elements_per_cta": 9 * 16 * 16,
        "fused_accumulator_floats_per_thread_at_128_threads": 18,
        "fused_current_product_elements_per_cta": 16 * 16,
        "staged_laderman_fp32_product_buffer_bytes": 23 * 32 * 32 * 4,
        "staged_conventional_fp32_product_buffer_bytes": 27 * 32 * 32 * 4,
        "staged_laderman_gpu_launches_if_each_product_is_a_kernel": 24,
        "staged_conventional_gpu_launches_if_each_product_is_a_kernel": 28,
        "output_writes": 9,
        "estimate_limitations": "No compiler register/shared-memory/spill report, occupancy, timing, or observed launch trace. Input fragments, pointers, compiler temporaries and reconstruction pressure are excluded from the accumulator floor.",
        "gpu_probe_gate": "ELIGIBLE_WITH_SEPARATE_CUDA_ACCESS",
        "selected_conditional_design": "23 product kernels with fused input preparation, FP32 product scratch, and one reconstruction kernel; 24 complete-operation launches",
    }
    metadata = {
        "schema_version": 1, "source_url": SOURCE_URL,
        "source_sha256": digest(source_bytes), "source_parameter": "l=1",
        "study_script_sha256": digest(Path(__file__).read_bytes()),
        "repository_base_commit": "5947fa97b35c7b4c840d3aa8118212504ed60327",
        "source_convention": "row-major U,V; transpose source W; U@V order",
        "target": "3x3 block matrix multiplication", "tensor_shape": list(SHAPE),
        "tested_block_size": BLOCK_SIZE, "algorithm_records": records,
        "hardware_estimates": hardware, "decision": "PENDING_GPU_PROBE",
    }
    return {
        "coefficients.json": canonical_bytes({name: coefficients(a) for name, a in candidates.items()}),
        "exact_and_block_results.json": canonical_bytes(metadata),
        "execution_dags.json": canonical_bytes(dags),
    }


def manifest(files: dict[str, bytes]) -> bytes:
    return canonical_bytes({"schema_version": 1, "files": {name: digest(data) for name, data in sorted(files.items())}})


def run(verify: bool) -> None:
    files = build()
    expected_manifest = manifest({SOURCE.name: SOURCE.read_bytes(), **files})
    if verify:
        for name, data in files.items():
            if (OUT / name).read_bytes() != data:
                raise ValueError(f"offline replay mismatch: {name}")
        if (OUT / "hash_manifest.json").read_bytes() != expected_manifest:
            raise ValueError("hash manifest mismatch")
        actual = sorted(path.name for path in OUT.iterdir() if path.is_file())
        expected = sorted([SOURCE.name, *files, "hash_manifest.json"])
        if actual != expected:
            raise ValueError(f"unexpected bundle files: {actual}")
    else:
        if any((OUT / name).exists() for name in (*files, "hash_manifest.json")):
            raise FileExistsError("study artifacts already exist; use --verify")
        for name, data in files.items():
            (OUT / name).write_bytes(data)
        (OUT / "hash_manifest.json").write_bytes(expected_manifest)
    print(json.dumps({"status": "PASS", "mode": "verify" if verify else "generate",
                      "algorithms": list(json.loads(files["coefficients.json"])),
                      "tensor_positions_per_algorithm": 729, "block_cases_per_algorithm": 9,
                      "decision": "PENDING_GPU_PROBE"}, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="recompute and hash-check retained CPU evidence")
    run(parser.parse_args().verify)
