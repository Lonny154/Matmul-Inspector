"""Independent exact and static execution study of the fixed cn122 circuits.

Only JSON data is imported from the published repository. No upstream program
is executed. The GPU fusion designs in the report are estimates, not kernels.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

try:
    from scripts import run_mi_research_6_feasibility as mi6
except ModuleNotFoundError:
    import run_mi_research_6_feasibility as mi6


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/mi-research-6c"
UPSTREAM = OUT / "upstream"
CERT = UPSTREAM / "certificate.json"
SOURCE = UPSTREAM / "perminov_source.json"
EXPECTED = {
    "certificate.json": "91c36b1165a71d4ba78cd8eb9c51ef9666d05e9162fdef5cba28d1d697a17dff",
    "perminov_source.json": "80bb4187ddf0cfc04654ca3e524f4d311c5a62245ee231a1101280b43fd2ae3a",
}
UPSTREAM_COMMIT = "34949f9ce50a89a5ad6b47a17f5834ad5a87a2fb"
PERMUTATION = (0, 3, 6, 1, 4, 7, 2, 5, 8)
BLOCK_BYTES = 32 * 32 * 4  # Whole N=96 FP32 block; not a per-CTA register estimate.
PROBE_SOURCE = ROOT / "examples/kernels/mi_research_6c_fused_perminov55.py"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def reference(ref: tuple[str, int]) -> dict:
    return {"node": ref[0], "sign": ref[1]}


class Circuit:
    def __init__(self) -> None:
        self.nodes: list[dict] = []
        self.by_id: dict[str, dict] = {}
        for side in ("A", "B"):
            for index in range(9):
                self.add(f"{side}{index}", "INPUT", (), (), side)

    def add(self, name: str, kind: str, deps: tuple[str, ...],
            signs: tuple[int, ...], phase: str) -> str:
        check(name not in self.by_id, f"duplicate DAG node {name}")
        check(all(dep in self.by_id for dep in deps), f"forward DAG dependency: {name}")
        check(len(deps) == len(signs) and all(sign in (-1, 1) for sign in signs),
              f"invalid signed dependencies: {name}")
        node = {"id": name, "kind": kind, "deps": list(deps),
                "signs": list(signs), "phase": phase}
        self.by_id[name] = node
        self.nodes.append(node)
        return name

    def source_map(self, prefix: str, input_refs: list[str], fresh: list,
                   outputs: list) -> list[tuple[str, int]]:
        wires = [(ref, 1) for ref in input_refs]
        gate_number = 0

        def form(parts: list[dict]) -> tuple[str, int]:
            nonlocal gate_number
            check(isinstance(parts, list) and bool(parts), "empty source form")
            terms = []
            for part in parts:
                index, sign = part["index"], part["value"]
                check(type(index) is int and 0 <= index < len(wires) and
                      type(sign) is int and sign in (-1, 1), "invalid source term")
                ref, prior_sign = wires[index]
                terms.append((ref, sign * prior_sign))
            value = terms[0]
            for term in terms[1:]:
                name = f"{prefix}g{gate_number:02d}"
                gate_number += 1
                self.add(name, "ADD", (value[0], term[0]),
                         (value[1], term[1]), prefix)
                value = (name, 1)
            return value

        for parts in fresh:
            wires.append(form(parts))
        result = [form(parts) for parts in outputs]
        return result

    def certificate_map(self, prefix: str, input_refs: list[str],
                        circuit: dict) -> list[tuple[str, int]]:
        check(circuit["input_count"] == len(input_refs), "certificate input count")
        wires = [(ref, 1) for ref in input_refs]
        for gate in circuit["gates"]:
            check(gate["slot"] == len(wires), "certificate gate slot")
            left, right = gate["left"], gate["right"]
            check(type(left) is int and type(right) is int and
                  0 <= left < len(wires) and 0 <= right < len(wires),
                  "certificate gate operand")
            first, second = wires[left], wires[right]
            signs = (gate["left_sign"] * first[1],
                     gate["right_sign"] * second[1])
            name = f"{prefix}g{len(wires) - len(input_refs):02d}"
            self.add(name, "ADD", (first[0], second[0]), signs, prefix)
            wires.append((name, 1))
        check(circuit["gate_count"] == len(circuit["gates"]), "certificate gate count")
        result = []
        for output in circuit["outputs"]:
            slot, sign = output["slot"], output["sign"]
            check(type(slot) is int and 0 <= slot < len(wires) and sign in (-1, 1),
                  "certificate output alias")
            result.append((wires[slot][0], sign * wires[slot][1]))
        return result


def build(source: dict, cert: dict, *, optimized: bool) -> tuple[Circuit, dict]:
    graph = Circuit()
    if optimized:
        maps = cert["circuits"]
        u = graph.certificate_map("U", [f"A{i}" for i in range(9)],
                                  maps["U_input_9_to_23"])
        v = graph.certificate_map("V", [f"B{i}" for i in range(9)],
                                  maps["V_input_9_to_23"])
    else:
        u = graph.source_map("U", [f"A{i}" for i in range(9)],
                             source["u_fresh"], source["u"])
        v = graph.source_map("V", [f"B{i}" for i in range(9)],
                             source["v_fresh"], source["v"])
    check(len(u) == len(v) == 23, "not 23 bilinear factors")
    for index, (left, right) in enumerate(zip(u, v)):
        graph.add(f"M{index:02d}", "PRODUCT", (left[0], right[0]),
                  (left[1], right[1]), "M")
    product_refs = [f"M{i:02d}" for i in range(23)]
    if optimized:
        w = graph.certificate_map("W", product_refs,
                                  cert["circuits"]["W_output_task_23_to_9"])
    else:
        raw = graph.source_map("W", product_refs, source["w_fresh"], source["w"])
        w = [raw[i] for i in PERMUTATION]
    check(len(w) == 9, "not nine output blocks")
    for index, (ref, sign) in enumerate(w):
        graph.add(f"C{index}", "OUTPUT", (ref,), (sign,), "C")
    return graph, {"U": [reference(ref) for ref in u],
                   "V": [reference(ref) for ref in v],
                   "W": [reference(ref) for ref in w]}


def build_laderman(actions: tuple) -> tuple[Circuit, dict]:
    """Expand MI-Research-6's independent linear forms into binary gates."""
    graph = Circuit()
    def forms(rows: list, inputs: list[str], prefix: str) -> list[tuple[str, int]]:
        expressions = [[{"index": index, "value": value}
                        for index, value in enumerate(row) if value] for row in rows]
        return graph.source_map(prefix, inputs, [], expressions)
    u = forms([action.u for action in actions], [f"A{i}" for i in range(9)], "U")
    v = forms([action.v for action in actions], [f"B{i}" for i in range(9)], "V")
    for index, (left, right) in enumerate(zip(u, v)):
        graph.add(f"M{index:02d}", "PRODUCT", (left[0], right[0]),
                  (left[1], right[1]), "M")
    w = forms([[action.w[i] for action in actions] for i in range(9)],
              [f"M{i:02d}" for i in range(23)], "W")
    for index, (ref, sign) in enumerate(w):
        graph.add(f"C{index}", "OUTPUT", (ref,), (sign,), "C")
    return graph, {"U": [reference(ref) for ref in u],
                   "V": [reference(ref) for ref in v],
                   "W": [reference(ref) for ref in w]}


def factor_rows(graph: Circuit, refs: dict) -> dict:
    rows = {}
    for side in ("A", "B"):
        for index in range(9):
            rows[f"{side}{index}"] = tuple(int(index == coordinate) for coordinate in range(9))
    for index in range(23):
        rows[f"M{index:02d}"] = tuple(int(index == coordinate) for coordinate in range(23))
    for node in graph.nodes:
        if node["kind"] != "ADD":
            continue
        deps = node["deps"]
        rows[node["id"]] = tuple(
            node["signs"][0] * a + node["signs"][1] * b
            for a, b in zip(rows[deps[0]], rows[deps[1]]))
    return {side: [[sign * value for value in rows[ref["node"]]]
                   for ref in refs[side] for sign in (ref["sign"],)]
            for side in ("U", "V", "W")}


def tensor_check(factors: dict) -> dict:
    u, v, w = factors["U"], factors["V"], factors["W"]
    check((len(u), len(v), len(w)) == (23, 23, 9), "factor dimensions")
    check(all(len(row) == 9 for row in u + v) and all(len(row) == 23 for row in w),
          "factor row dimensions")
    check(all(type(coefficient) is int and coefficient in (-1, 0, 1)
              for row in u + v + w for coefficient in row), "nonternary coefficient")
    units = zeros = 0
    for ai in range(9):
        for bi in range(9):
            for ci in range(9):
                observed = sum(u[r][ai] * v[r][bi] * w[ci][r] for r in range(23))
                target = int(ai // 3 == ci // 3 and ai % 3 == bi // 3 and bi % 3 == ci % 3)
                check(observed == target, f"Brent identity failed at {(ai, bi, ci)}")
                units += target
                zeros += 1 - target
    return {"checked": units + zeros, "unit": units, "zero": zeros,
            "mismatches": 0}


def order_and_metrics(graph: Circuit, *, liveness: bool) -> dict:
    operations = [node for node in graph.nodes if node["kind"] != "INPUT"]
    consumers = Counter(dep for node in operations for dep in node["deps"])
    remaining = dict(consumers)
    created = {node["id"] for node in graph.nodes if node["kind"] == "INPUT"}
    unscheduled = {node["id"]: node for node in operations}
    live: set[str] = set()
    live_with_inputs = set(created)
    order = []
    live_steps = []
    live_tile_elements = [18 * 16 * 32]
    depth = {name: 0 for name in created}
    static_positions = {node["id"]: i for i, node in enumerate(graph.nodes)}
    while unscheduled:
        ready = [node for node in unscheduled.values()
                 if all(dep in created for dep in node["deps"])]
        check(bool(ready), "cyclic dependency DAG")
        def projected(node: dict) -> tuple[int, int]:
            current = live | ({node["id"]} if consumers[node["id"]] else set())
            for dep in set(node["deps"]):
                if remaining[dep] == node["deps"].count(dep):
                    current.discard(dep)
            return len(current), static_positions[node["id"]]
        chosen = min(ready, key=projected if liveness else
                     lambda node: static_positions[node["id"]])
        name = chosen["id"]
        depth[name] = 1 + max((depth[dep] for dep in chosen["deps"]), default=0)
        if consumers[name] and chosen["kind"] != "OUTPUT":
            live.add(name)
            live_with_inputs.add(name)
        for dep in chosen["deps"]:
            remaining[dep] -= 1
            if remaining[dep] == 0:
                live.discard(dep)
                live_with_inputs.discard(dep)
        created.add(name)
        del unscheduled[name]
        order.append(name)
        live_steps.append({"after": name,
                           "gate_values": sum(graph.by_id[x]["kind"] == "ADD" for x in live),
                           "product_values": sum(graph.by_id[x]["kind"] == "PRODUCT" for x in live)})
        live_tile_elements.append(sum(16 * 32 if graph.by_id[x]["phase"] in ("A", "B", "U", "V")
                                      else 16 * 16 for x in live_with_inputs))
    check(not live, "unconsumed intermediate")
    positions = {name: index for index, name in enumerate(order)}
    lifetimes = {}
    for node in graph.nodes:
        if node["kind"] not in ("ADD", "PRODUCT"):
            continue
        uses = [positions[consumer["id"]] for consumer in operations
                for dep in consumer["deps"] if dep == node["id"]]
        check(bool(uses), f"dead intermediate: {node['id']}")
        lifetimes[node["id"]] = {"created_at": positions[node["id"]],
                                 "last_used_at": max(uses),
                                 "consumer_count": len(uses)}
    return {"order": order, "peak_live_intermediates": max(
                step["gate_values"] + step["product_values"] for step in live_steps),
            "peak_live_gate_values": max(step["gate_values"] for step in live_steps),
            "peak_live_products": max(step["product_values"] for step in live_steps),
            "peak_whole_block_fp32_bytes_if_materialized": BLOCK_BYTES * max(
                step["gate_values"] + step["product_values"] for step in live_steps),
            "peak_local_live_fp32_tile_elements_including_inputs": max(live_tile_elements),
            "optimistic_128_thread_fp32_values_each": (max(live_tile_elements) + 127) // 128,
            "peak_dependency_depth_operations": max(depth[f"C{i}"] for i in range(9)),
            "value_lifetimes": lifetimes, "live_steps": live_steps}


def evaluate_blocks(graph: Circuit, order: list[str], left: list, right: list) -> list:
    values = {f"A{i}": left[i] for i in range(9)}
    values.update({f"B{i}": right[i] for i in range(9)})
    for name in order:
        node = graph.by_id[name]
        if node["kind"] == "PRODUCT":
            first, second = (values[dep] for dep in node["deps"])
            a, b = node["signs"]
            values[name] = mi6.block_matmul(
                [[a * x for x in row] for row in first],
                [[b * x for x in row] for row in second])
        elif node["kind"] == "ADD":
            x, y = (values[dep] for dep in node["deps"])
            a, b = node["signs"]
            values[name] = [[a * x[i][j] + b * y[i][j] for j in range(2)]
                            for i in range(2)]
        else:
            sign = node["signs"][0]
            values[name] = [[sign * value for value in row]
                            for row in values[node["deps"][0]]]
    return [values[f"C{i}"] for i in range(9)]


def render_probe(graph: Circuit, order: list[str]) -> str:
    """Freeze one output-owned kernel from the checked 55-gate schedule."""
    check(sum(node["kind"] == "ADD" for node in graph.nodes) == 55 and
          sum(node["kind"] == "PRODUCT" for node in graph.nodes) == 23,
          "probe must be the verified 55-addition, rank-23 circuit")
    lines = [
        '"""MI-Research-6C: one fixed N=96 output-owned Perminov 55-gate probe.',
        '',
        'Generated from the independently checked frozen circuit. Four programs',
        'each own one local 16x16 tile of all nine output blocks. No atomics or',
        'rank-product buffers are used. This is a feasibility kernel, not a',
        'performance-tuned GEMM.',
        '"""',
        '',
        'import torch',
        'import triton',
        'import triton.language as tl',
        '',
        '',
        '@triton.jit',
        'def _fused(a_ptr, b_ptr, c_ptr):',
        '    row = tl.program_id(0) * 16 + tl.arange(0, 16)',
        '    col = tl.program_id(1) * 16 + tl.arange(0, 16)',
        '    inner = tl.arange(0, 32)',
    ]
    for side in ("A", "B"):
        for index in range(9):
            block_row, block_col = divmod(index, 3)
            if side == "A":
                address = (f'a_ptr + ({block_row} * 32 + row[:, None]) * 96 + '
                           f'{block_col} * 32 + inner[None, :]')
            else:
                address = (f'b_ptr + ({block_row} * 32 + inner[:, None]) * 96 + '
                           f'{block_col} * 32 + col[None, :]')
            lines.append(f'    {side}{index} = tl.load({address}).to(tl.float32)')
    for name in order:
        node = graph.by_id[name]
        if node["kind"] == "ADD":
            a, b = node["deps"]
            x, y = node["signs"]
            left = a if x == 1 else f'(-{a})'
            operator = '+' if y == 1 else '-'
            lines.append(f'    {name} = {left} {operator} {b}')
        elif node["kind"] == "PRODUCT":
            a, b = node["deps"]
            x, y = node["signs"]
            left = a if x == 1 else f'(-{a})'
            right = b if y == 1 else f'(-{b})'
            lines.append(f'    {name} = tl.dot({left}, {right}, '
                         'tl.zeros((16, 16), tl.float32), input_precision="tf32x3")')
        elif node["kind"] == "OUTPUT":
            index = int(name[1:])
            block_row, block_col = divmod(index, 3)
            dep = node["deps"][0]
            value = dep if node["signs"][0] == 1 else f'(-{dep})'
            lines.append(f'    tl.store(c_ptr + ({block_row} * 32 + row[:, None]) * 96 + '
                         f'{block_col} * 32 + col[None, :], {value})')
    lines += [
        '',
        '',
        'def call(left, right, output):',
        '    if (left.shape != (96, 96) or right.shape != (96, 96) or',
        '            output.shape != (96, 96) or left.dtype != torch.float16 or',
        '            right.dtype != torch.float16 or output.dtype != torch.float16 or',
        '            not left.is_cuda or not right.is_cuda or not output.is_cuda or',
        '            not left.is_contiguous() or not right.is_contiguous() or',
        '            not output.is_contiguous()):',
        '        raise ValueError("expected contiguous CUDA FP16 96x96 inputs/output")',
        '    return _fused[(2, 2)](left, right, output, num_warps=4, num_stages=2)',
        '',
    ]
    return "\n".join(lines)


def analyze(source: dict, cert: dict) -> dict:
    check(source["n"] == [3, 3, 3] and source["m"] == 23 and source["z2"] is False,
          "source tensor identity")
    check(cert["rank"] == 23 and cert["dimensions"] == [3, 3, 3],
          "certificate identity")
    check(cert["transposition"]["raw_to_task_output_permutation"] == list(PERMUTATION),
          "output-coordinate convention")
    check(cert["source"]["sha256"] == EXPECTED["perminov_source.json"],
          "certificate source hash")
    laderman = mi6.parse_laderman(mi6.SOURCE.read_text())
    configurations = {}
    for name, optimized in (("laderman_98", None), ("perminov_58", False),
                            ("perminov_55", True)):
        graph, refs = (build_laderman(laderman) if optimized is None else
                       build(source, cert, optimized=optimized))
        factors = factor_rows(graph, refs)
        if optimized is None:
            check(factors == {"U": [list(action.u) for action in laderman],
                              "V": [list(action.v) for action in laderman],
                              "W": [[action.w[i] for action in laderman] for i in range(9)]},
                  "Laderman factor mismatch")
        else:
            check(factors["U"] == cert["factor_coefficients"]["alpha_U_23x9"]
                  and factors["V"] == cert["factor_coefficients"]["beta_V_23x9"]
                  and factors["W"] == cert["factor_coefficients"]["gamma_task_9x23"],
                  f"{name} factor mismatch")
        tensor = tensor_check(factors)
        counts = {phase: sum(node["kind"] == "ADD" and node["phase"] == phase
                             for node in graph.nodes) for phase in ("U", "V", "W")}
        expected = ({"U": 13, "V": 14, "W": 28} if optimized is True else
                    {"U": 14, "V": 15, "W": 29} if optimized is False else
                    {"U": 28, "V": 28, "W": 42})
        check(counts == expected, f"{name} addition counts: {counts}")
        schedules = {}
        for order_name, greedy in (("published", False), ("liveness", True)):
            metrics = order_and_metrics(graph, liveness=greedy)
            case_hashes = {}
            for case, left, right in mi6.block_cases():
                observed = evaluate_blocks(graph, metrics["order"], left, right)
                check(observed == mi6.direct_blocks(left, right),
                      f"{name}/{order_name}/{case} noncommutative block mismatch")
                case_hashes[case] = mi6.digest(mi6.canonical_bytes(observed))
            metrics["case_output_sha256"] = case_hashes
            schedules[order_name] = metrics
        consumers = Counter(dep for node in graph.nodes for dep in node["deps"])
        configurations[name] = {
            "tensor": tensor, "addition_counts": counts, "total_additions": sum(counts.values()),
            "products": 23, "input_block_nodes": 18,
            "nodes": graph.nodes, "factor_coefficients": factors,
            "product_output_fanout": [sum(bool(factors["W"][i][r]) for i in range(9))
                                      for r in range(23)],
            "shared_gate_nodes": [node["id"] for node in graph.nodes
                                  if node["kind"] == "ADD" and consumers[node["id"]] > 1],
            "recomputation_eligible_single_use_gates": [node["id"] for node in graph.nodes
                                                        if node["kind"] == "ADD" and consumers[node["id"]] == 1],
            "schedules": schedules}
    check(configurations["perminov_58"]["factor_coefficients"] ==
          configurations["perminov_55"]["factor_coefficients"],
          "Perminov tensor factorization changed")
    probe_graph, _ = build(source, cert, optimized=True)
    probe_text = render_probe(probe_graph,
                              configurations["perminov_55"]["schedules"]["liveness"]["order"])
    return {"schema_version": 1, "study": "MI-Research-6C",
            "upstream_commit": UPSTREAM_COMMIT,
            "upstream_source_sha256": EXPECTED["perminov_source.json"],
            "upstream_certificate_sha256": EXPECTED["certificate.json"],
            "laderman_source_sha256": mi6.digest(mi6.SOURCE.read_bytes()),
            "cost_model": "binary signed addition/subtraction; copies and sign changes free; 23 products counted separately",
            "block_model": "2x2 integer matrix blocks, ordered left-times-right products",
            "block_cases": [name for name, _, _ in mi6.block_cases()],
            "laderman_note": "98 additions are the MI-Research-6 independent-form realization (56 input + 42 output), not a proven shared-circuit minimum",
            "configurations": configurations,
            "probe_source_sha256": hashlib.sha256(probe_text.encode()).hexdigest(),
            "cpu_decision": "B", "gpu_probe_selection": "perminov_55_liveness_single_launch"}


def verify() -> dict:
    for name, expected in EXPECTED.items():
        check(digest(UPSTREAM / name) == expected, f"upstream {name} changed")
    result = analyze(json.loads(SOURCE.read_text()), json.loads(CERT.read_text()))
    check(json.loads((OUT / "analysis.json").read_text()) == result,
          "exact/scheduling replay mismatch")
    graph, _ = build(json.loads(SOURCE.read_text()), json.loads(CERT.read_text()), optimized=True)
    rendered = render_probe(graph, result["configurations"]["perminov_55"]["schedules"]["liveness"]["order"])
    check(PROBE_SOURCE.read_text() == rendered and digest(PROBE_SOURCE) == result["probe_source_sha256"],
          "selected GPU probe differs from verified circuit")
    gate = json.loads((OUT / "pre_probe_gate.json").read_text())
    check(gate.get("stage") == "PRE_GPU_COMPILATION" and
          gate.get("selected") == result["gpu_probe_selection"] and
          gate.get("decision") == "PASS_FOR_ONE_COMPILATION_AND_CORRECTNESS_PROBE" and
          gate.get("analysis_sha256") == digest(OUT / "analysis.json") and
          gate.get("probe_source_sha256") == digest(PROBE_SOURCE),
          "pre-GPU gate provenance mismatch")
    listed = json.loads((OUT / "hash_manifest.json").read_text())
    actual = {str(path.relative_to(OUT)) for path in OUT.rglob("*")
              if path.is_file() and path.name != "hash_manifest.json"}
    check(set(listed) == actual, "artifact file set mismatch")
    for name, expected in listed.items():
        check(not Path(name).is_absolute() and ".." not in Path(name).parts and
              digest(OUT / name) == expected, f"artifact hash mismatch: {name}")
    return {"status": "PASS", "cpu_decision": result["cpu_decision"],
            "brent_identities_per_configuration": 729,
            "block_cases_per_schedule": len(result["block_cases"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        result = verify()
    else:
        check(not (OUT / "analysis.json").exists(), "analysis already exists")
        for name, expected in EXPECTED.items():
            check(digest(UPSTREAM / name) == expected, f"upstream {name} changed")
        source_data, cert_data = json.loads(SOURCE.read_text()), json.loads(CERT.read_text())
        analysis = analyze(source_data, cert_data)
        write_json(OUT / "analysis.json", analysis)
        graph, _ = build(source_data, cert_data, optimized=True)
        PROBE_SOURCE.write_text(render_probe(
            graph, analysis["configurations"]["perminov_55"]["schedules"]["liveness"]["order"]))
        write_json(OUT / "hash_manifest.json", {
            str(path.relative_to(OUT)): digest(path)
            for path in sorted(OUT.rglob("*")) if path.is_file()
            and path.name != "hash_manifest.json"})
        result = verify()
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
