"""CPU-only output-slice cost replay for the two frozen Perminov circuits."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

try:
    from scripts import run_mi_research_6c as exact
    from scripts import generate_mi_research_8_kernel as sliced_source
    from scripts import analyze_mi_research_8 as prior
except ModuleNotFoundError:
    import run_mi_research_6c as exact
    import generate_mi_research_8_kernel as sliced_source
    import analyze_mi_research_8 as prior


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/mi-research-9"
ANALYSIS = OUT / "analysis.json"
MANIFEST = OUT / "hash_manifest.json"
FOUR_SOURCE = exact.PROBE_SOURCE
THIRTY_SIX_SOURCES = {"55": sliced_source.BASELINE, "58": sliced_source.TARGET}
INPUT_ORDER = tuple(f"{side}{i}" for side in ("A", "B") for i in range(9))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, record: dict) -> None:
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def closure(graph: exact.Circuit, outputs: tuple[str, ...]) -> set[str]:
    if not outputs or len(set(outputs)) != len(outputs):
        raise ValueError("output ownership must be nonempty and unique")
    if any(name not in graph.by_id or graph.by_id[name]["kind"] != "OUTPUT"
           for name in outputs):
        raise ValueError("unknown or non-output owner")
    selected = set(outputs)
    pending = list(outputs)
    while pending:
        for dep in graph.by_id[pending.pop()]["deps"]:
            if dep not in graph.by_id:
                raise ValueError(f"missing dependency {dep}")
            if dep not in selected:
                selected.add(dep)
                pending.append(dep)
    return selected


def mapping_roles(mapping: str) -> list[dict]:
    if mapping == "four_cta":
        return [{"role": "all_outputs", "outputs": [f"C{i}" for i in range(9)],
                 "tile_positions": [[r, c] for r in range(2) for c in range(2)],
                 "grid": [2, 2, 1]}]
    if mapping == "thirty_six_cta":
        return [{"role": f"output_{i}", "outputs": [f"C{i}"],
                 "tile_positions": [[r, c] for r in range(2) for c in range(2)],
                 "grid": [2, 2, 9]} for i in range(9)]
    raise ValueError(f"unknown output mapping: {mapping}")


def validate_ownership(roles: list[dict], graph: exact.Circuit) -> None:
    owned = Counter()
    for role in roles:
        for pair in role["tile_positions"]:
            if (not isinstance(pair, list) or len(pair) != 2 or
                any(type(x) is not int or x not in (0, 1) for x in pair)):
                raise ValueError("invalid tile position")
            for name in role["outputs"]:
                if name not in graph.by_id or graph.by_id[name]["kind"] != "OUTPUT":
                    raise ValueError("invalid output node")
                owned[(pair[0], pair[1], name)] += 1
    expected = {(r, c, f"C{i}") for r in range(2) for c in range(2) for i in range(9)}
    if set(owned) != expected or any(count != 1 for count in owned.values()):
        raise ValueError("output ownership must cover each tile exactly once")


def category(node: dict) -> str:
    if node["kind"] == "INPUT":
        return "input_tile"
    if node["kind"] == "PRODUCT":
        return "product_accumulator"
    if node["kind"] == "ADD" and node["phase"] in ("U", "V"):
        return "input_combination"
    if node["kind"] == "ADD" and node["phase"] == "W":
        return "reconstruction_value"
    raise ValueError(f"unmodeled value: {node}")


def elements(node: dict) -> int:
    # These are logical FP32 tensor elements per CTA, not registers or bytes moved.
    return 512 if node["phase"] in ("A", "B", "U", "V") else 256


def role_cost(graph: exact.Circuit, order: list[str], outputs: tuple[str, ...]) -> dict:
    selected = closure(graph, outputs)
    scheduled = [*([name for name in INPUT_ORDER if name in selected]),
                 *[name for name in order if name in selected]]
    if set(scheduled) != selected or len(scheduled) != len(selected):
        raise ValueError("frozen execution order does not cover the slice exactly")
    positions = {name: index for index, name in enumerate(scheduled)}
    consumers: dict[str, list[str]] = defaultdict(list)
    depth: dict[str, int] = {}
    for name in scheduled:
        node = graph.by_id[name]
        if any(positions[dep] >= positions[name] for dep in node["deps"]):
            raise ValueError(f"non-topological order at {name}")
        depth[name] = (0 if node["kind"] == "INPUT" else
                       1 + max((depth[dep] for dep in node["deps"]), default=0))
        for dep in node["deps"]:
            consumers[dep].append(name)
    lifetimes = {}
    for name in scheduled:
        node = graph.by_id[name]
        if node["kind"] == "OUTPUT":
            continue  # Store is a use, not an extra materialized FP32 array.
        uses = consumers[name]
        if not uses:
            raise ValueError(f"dead selected expression: {name}")
        lifetimes[name] = {"category": category(node), "created_at": positions[name],
                           "last_used_at": max(positions[use] for use in uses),
                           "consumers": uses, "logical_fp32_elements": elements(node)}
    peaks = {key: 0 for key in ("input_tile", "input_combination",
                                "product_accumulator", "reconstruction_value")}
    peak_all = peak_intermediate = peak_elements = 0
    peak_intermediate_elements = 0
    peak_step = None
    live_steps = []
    for index, name in enumerate(scheduled):
        # Values with last use at this step are released after the step.
        live = [node_id for node_id, life in lifetimes.items()
                if life["created_at"] <= index < life["last_used_at"]]
        counts = Counter(lifetimes[node_id]["category"] for node_id in live)
        all_elements = sum(lifetimes[node_id]["logical_fp32_elements"] for node_id in live)
        intermediate = sum(counts[key] for key in peaks if key != "input_tile")
        intermediate_elements = sum(lifetimes[node_id]["logical_fp32_elements"] for node_id in live
                                    if lifetimes[node_id]["category"] != "input_tile")
        for key in peaks:
            peaks[key] = max(peaks[key], counts[key])
        if all_elements > peak_elements:
            peak_step = name
        peak_all = max(peak_all, len(live))
        peak_intermediate = max(peak_intermediate, intermediate)
        peak_elements = max(peak_elements, all_elements)
        peak_intermediate_elements = max(peak_intermediate_elements, intermediate_elements)
        live_steps.append({"after": name, "live_by_category": dict(sorted(counts.items())),
                           "live_values": len(live), "logical_fp32_elements": all_elements})
    adds = [name for name in scheduled if graph.by_id[name]["kind"] == "ADD"]
    products = [name for name in scheduled if graph.by_id[name]["kind"] == "PRODUCT"]
    locally_shared = [name for name, life in lifetimes.items()
                      if life["category"] != "input_tile" and len(life["consumers"]) > 1]
    return {"outputs": list(outputs), "dependency_nodes": sorted(selected),
            "execution_order": scheduled, "input_loads": sum(graph.by_id[name]["kind"] == "INPUT" for name in scheduled),
            "gate_instances": len(adds), "dot_instances": len(products),
            "gates_by_phase": {phase: sum(graph.by_id[name]["phase"] == phase for name in adds)
                               for phase in ("U", "V", "W")},
            "products": products, "locally_shared_expressions": sorted(locally_shared),
            "lifetimes": lifetimes, "live_steps": live_steps,
            "peak_live_by_category": peaks, "peak_live_values_including_inputs": peak_all,
            "peak_live_intermediates": peak_intermediate,
            "peak_logical_fp32_elements_including_inputs": peak_elements,
            "peak_estimated_live_bytes_including_inputs": peak_elements * 4,
            "peak_intermediate_elements": peak_intermediate_elements,
            "peak_element_step": peak_step,
            "dependency_path_depth_operations": max(depth[name] for name in outputs)}


def mapping_cost(graph: exact.Circuit, order: list[str], mapping: str) -> dict:
    roles = mapping_roles(mapping)
    validate_ownership(roles, graph)
    costs = {}
    for role in roles:
        cost = role_cost(graph, order, tuple(role["outputs"]))
        costs[role["role"]] = {**role, **cost, "cta_repetitions": len(role["tile_positions"]),
                               "owned_output_elements_per_cta": 256 * len(role["outputs"])}
    repetitions = sum(row["cta_repetitions"] for row in costs.values())
    node_roles: dict[str, list[str]] = defaultdict(list)
    for name, row in costs.items():
        for node in row["dependency_nodes"]:
            if graph.by_id[node]["kind"] in ("ADD", "PRODUCT"):
                node_roles[node].append(name)
    shared = [{"node": name, "kind": graph.by_id[name]["kind"],
               "phase": graph.by_id[name]["phase"], "roles": sorted(owners),
               "role_count": len(owners), "extra_instances_per_tile_position": len(owners)-1,
               "extra_instances_per_grid": 4*(len(owners)-1)}
              for name, owners in node_roles.items() if len(owners) > 1]
    shared.sort(key=lambda row: (-row["extra_instances_per_grid"], row["node"]))
    unique_gates = sum(node["kind"] == "ADD" for node in graph.nodes)
    unique_products = sum(node["kind"] == "PRODUCT" for node in graph.nodes)
    gate_total = sum(row["gate_instances"] * row["cta_repetitions"] for row in costs.values())
    dot_total = sum(row["dot_instances"] * row["cta_repetitions"] for row in costs.values())
    if sum(row["extra_instances_per_grid"] for row in shared if row["kind"] == "ADD") != gate_total-4*unique_gates:
        raise ValueError("gate duplication accounting mismatch")
    if sum(row["extra_instances_per_grid"] for row in shared if row["kind"] == "PRODUCT") != dot_total-4*unique_products:
        raise ValueError("product duplication accounting mismatch")
    return {"mapping": mapping, "grid": roles[0]["grid"], "cta_count": repetitions,
            "roles": costs, "cross_output_shared_expressions": shared,
            "cost_vector": {"mathematical_rank": unique_products,
                            "whole_circuit_additions": unique_gates,
                            "unique_product_labels_per_tile_position": unique_products,
                            "unique_gate_labels_per_tile_position": unique_gates,
                            "per_grid_dot_instances": dot_total,
                            "per_grid_gate_instances": gate_total,
                            "per_grid_extra_dots_due_to_output_slicing": dot_total-4*unique_products,
                            "per_grid_extra_gates_due_to_output_slicing": gate_total-4*unique_gates,
                            "peak_live_values_including_inputs_per_cta": max(row["peak_live_values_including_inputs"] for row in costs.values()),
                            "peak_live_intermediates_per_cta": max(row["peak_live_intermediates"] for row in costs.values()),
                            "max_estimated_live_bytes_per_cta": max(row["peak_estimated_live_bytes_including_inputs"] for row in costs.values()),
                            "max_dependency_path_depth_operations": max(row["dependency_path_depth_operations"] for row in costs.values()),
                            "cta_count": repetitions,
                            "output_blocks_per_cta": len(roles[0]["outputs"]),
                            "output_elements_per_cta": 256*len(roles[0]["outputs"])}}


def pareto(a: dict, b: dict) -> dict:
    fields = ("per_grid_dot_instances", "per_grid_gate_instances",
              "peak_live_intermediates_per_cta", "max_estimated_live_bytes_per_cta",
              "max_dependency_path_depth_operations")
    def dominates(x: dict, y: dict) -> bool:
        return all(x[key] <= y[key] for key in fields) and any(x[key] < y[key] for key in fields)
    return {"minimized_fields": list(fields), "55_dominates_58": dominates(a, b),
            "58_dominates_55": dominates(b, a),
            "field_comparison_55_minus_58": {key: a[key]-b[key] for key in fields}}


def analyze() -> dict:
    if exact.verify()["status"] != "PASS" or prior.verify()["status"] != "PASS":
        raise ValueError("frozen source evidence did not verify")
    source = json.loads(exact.SOURCE.read_text())
    cert = json.loads(exact.CERT.read_text())
    retained = json.loads((exact.OUT / "analysis.json").read_text())["configurations"]
    graphs = {}
    factors = {}
    records = {}
    for label in ("55", "58"):
        graph, refs = exact.build(source, cert, optimized=label == "55")
        factor = exact.factor_rows(graph, refs)
        if factor != retained[f"perminov_{label}"]["factor_coefficients"] or exact.tensor_check(factor)["mismatches"]:
            raise ValueError("exact factor or tensor mismatch")
        order = retained[f"perminov_{label}"]["schedules"]["liveness"]["order"]
        if graph.nodes != retained[f"perminov_{label}"]["nodes"]:
            raise ValueError("retained DAG differs from exact replay")
        if label == "55" and exact.render_probe(graph, order) != FOUR_SOURCE.read_text():
            raise ValueError("four-CTA source differs from DAG and order")
        if sliced_source.render(graph, order, optimized=label == "55")[0] != THIRTY_SIX_SOURCES[label].read_text():
            raise ValueError("36-CTA source differs from DAG and order")
        graphs[label], factors[label] = graph, factor
        records[label] = {mapping: mapping_cost(graph, order, mapping)
                          for mapping in ("four_cta", "thirty_six_cta")}
        for mapping, record in records[label].items():
            zero_net_total = 0
            for role in record["roles"].values():
                support = {f"M{index:02d}" for output in role["outputs"]
                           for index, coefficient in enumerate(factor["W"][int(output[1:])])
                           if coefficient}
                selected_products = set(role["products"])
                if not support <= selected_products:
                    raise ValueError("net output factor is missing from DAG dependency slice")
                role["nonzero_output_factor_products"] = sorted(support)
                role["zero_net_product_dependencies"] = sorted(selected_products - support)
                zero_net_total += len(selected_products-support) * role["cta_repetitions"]
            record["cost_vector"]["per_grid_zero_net_product_dependency_instances"] = zero_net_total
            record["cost_vector"]["per_grid_gates_by_phase"] = {
                phase: sum(role["gates_by_phase"][phase] * role["cta_repetitions"]
                           for role in record["roles"].values())
                for phase in ("U", "V", "W")}
    if factors["55"] != factors["58"]:
        raise ValueError("factor maps differ")
    historical = json.loads(prior.ANALYSIS.read_text())
    compiler = {label: historical["sessions"]["graph-s1"]["compiler"][slot]
                for label, slot in (("55", "A"), ("58", "B"))}
    comparisons = {mapping: pareto(records["55"][mapping]["cost_vector"],
                                   records["58"][mapping]["cost_vector"])
                   for mapping in ("four_cta", "thirty_six_cta")}
    return {"schema_version": 1, "study": "MI-Research-9", "checkpoint": "c2b8de73981662193284f231c8dd67b5fa1cbdae",
            "model": {"execution_order": "retained MI-Research-6C liveness order filtered to backward-reachable nodes; all selected inputs load first",
                      "live_interval": "created_at <= step < last_used_at, after each step; output stores count as final uses",
                      "tile_shapes": {"input_and_input_combination": [16, 32], "product_and_reconstruction": [16, 16]},
                      "bytes": "logical FP32 elements times four, excluding layout replication and compiler temporaries",
                      "duplicate_scope": "same logical DAG expression repeated across output owners at one local tile position, multiplied by four tile positions; distinct positions carry distinct numerical tiles",
                      "four_cta_58_status": "structural projection onto verified four-CTA ownership and retained order; no frozen four-CTA 58 kernel or compiler record"},
            "input_hashes": {str(path.relative_to(ROOT)): digest(path) for path in
                             (exact.SOURCE, exact.CERT, exact.OUT / "analysis.json", FOUR_SOURCE,
                              THIRTY_SIX_SOURCES["55"], THIRTY_SIX_SOURCES["58"], prior.ANALYSIS,
                              prior.MANIFEST)},
            "tensor": retained["perminov_55"]["tensor"],
            "identical_factor_maps": True, "circuits": records,
            "pareto_comparisons": comparisons, "historical_compiler": compiler,
            "historical_timing_decisions": historical["decisions"],
            "classification": "A — Slice-aware cost analysis validated"}


def verify() -> dict:
    manifest = json.loads(MANIFEST.read_text())
    if manifest != {"analysis.json": digest(ANALYSIS),
                    "scripts/analyze_mi_research_9.py": digest(Path(__file__))}:
        raise ValueError("MI-Research-9 hash manifest mismatch")
    actual = analyze()
    if json.loads(ANALYSIS.read_text()) != actual:
        raise ValueError("MI-Research-9 analysis replay mismatch")
    return {"status": "PASS", "classification": actual["classification"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        result = verify()
    else:
        if ANALYSIS.exists() or MANIFEST.exists():
            raise FileExistsError("MI-Research-9 evidence already exists; use --verify")
        OUT.mkdir(parents=True, exist_ok=True)
        write_json(ANALYSIS, analyze())
        write_json(MANIFEST, {"analysis.json": digest(ANALYSIS),
                              "scripts/analyze_mi_research_9.py": digest(Path(__file__))})
        result = verify()
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
