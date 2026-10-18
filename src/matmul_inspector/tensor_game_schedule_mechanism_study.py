"""CPU-only accounting for the bounded HA-Schedule-4 mechanism study."""

from __future__ import annotations

from collections import defaultdict
import difflib
import hashlib
import random
import re
import statistics

from .tensor_game_schedule_gpu_study import timing_distribution


STORED = "rank8_stored"
RECOMPUTED = "rank8_recomputed"
DUMMY = "rank8_recomputed_plus_dummy_DIAGNOSTIC_ONLY"
MECHANISM_LABELS = (
    "RECOMPUTE_COSTS_BUT_MATERIALIZATION_COSTS_MORE",
    "RECOMPUTE_CONSUMER_ITSELF_FASTER", "LAUNCH_REDUCTION_DOMINANT_SIGNAL",
    "MEMORY_TRAFFIC_SIGNAL", "MIXED_MECHANISM", "COMPILER_RESOURCE_EFFECT",
    "SESSION_UNSTABLE", "INCONCLUSIVE",
)
COMMON = tuple(f"common_gemm_{index}" for index in range(2, 8)) + ("reconstruction",)
SPECIAL = ("consumer_0", "consumer_1")


def expected_topology(mode: str) -> list[dict]:
    """Physical source launches, distinct from HA-Schedule-2 logical groups."""
    if mode not in (STORED, RECOMPUTED, DUMMY):
        raise ValueError(f"unknown mechanism mode: {mode}")
    rows = []
    if mode == STORED:
        rows.append({"role": "materialization", "kernel_name": "_materialize_u_form"})
    if mode == DUMMY:
        rows.append({"role": "diagnostic_dummy", "kernel_name": "_diagnostic_tiny_launch"})
    rows.extend({"role": role, "kernel_name": "_shared_form_gemm"} for role in SPECIAL)
    rows.extend({"role": role, "kernel_name": "_fused_input_gemm_kernel"} for role in COMMON[:-1])
    rows.append({"role": "reconstruction", "kernel_name": "_fused_reconstruction_kernel"})
    return rows


def validate_topology(mode: str, actual_kernel_names: list[str]) -> list[dict]:
    expected = expected_topology(mode)
    if actual_kernel_names != [entry["kernel_name"] for entry in expected]:
        raise ValueError(f"kernel topology mismatch for {mode}: {actual_kernel_names}")
    return [{**entry, "launch_index": index} for index, entry in enumerate(expected)]


def classify_kernel_roles(stored: list[dict], recomputed: list[dict]) -> dict:
    left = {entry["role"]: entry for entry in stored}
    right = {entry["role"]: entry for entry in recomputed}
    if set(left) != set(COMMON + SPECIAL + ("materialization",)) or set(right) != set(COMMON + SPECIAL):
        raise ValueError("unexpected stored/recomputed launch roles")
    if any(left[role]["kernel_name"] != right[role]["kernel_name"] for role in COMMON):
        raise ValueError("supposedly common kernel name differs")
    return {"stored_only": ["materialization"], "differing": list(SPECIAL),
            "common": list(COMMON), "stored_launches": len(stored),
            "recomputed_launches": len(recomputed)}


def summarize_raw(rows: list[dict]) -> list[dict]:
    """Use within-round medians as independent descriptive replicates."""
    grouped: dict[tuple, list[float]] = defaultdict(list)
    for row in rows:
        key = (row["session"], row["matrix_size"], row["phase"], row["variant"], row["round"])
        grouped[key].append(float(row["elapsed_ms"]))
    by_variant: dict[tuple, list[tuple[int, float, int]]] = defaultdict(list)
    for (*prefix, round_index), values in grouped.items():
        by_variant[tuple(prefix)].append((round_index, statistics.median(values), len(values)))
    result = []
    for (session, size, phase, variant), rounds in sorted(by_variant.items()):
        ordered = sorted(rounds)
        result.append({"session": session, "matrix_size": size, "phase": phase,
                       "variant": variant, "round_count": len(ordered),
                       "operations_per_round": [count for _, _, count in ordered],
                       "sample_count": sum(count for _, _, count in ordered),
                       **timing_distribution([value for _, value, _ in ordered])})
    return result


def _bootstrap_interval(values: list[float], seed: int, repetitions: int = 5000) -> list[float]:
    rng = random.Random(seed)
    sampled = sorted(statistics.median(rng.choices(values, k=len(values)))
                     for _ in range(repetitions))
    return [sampled[int(0.025 * repetitions)], sampled[int(0.975 * repetitions)]]


def paired_summary(summary: list[dict], phase: str, stored_variant: str = STORED,
                   recomputed_variant: str = RECOMPUTED) -> list[dict]:
    by_key = {(row["session"], row["matrix_size"], row["phase"], row["variant"]): row for row in summary}
    result = []
    for session, size in sorted({(row["session"], row["matrix_size"]) for row in summary
                                 if row["phase"] == phase}):
        stored = by_key.get((session, size, phase, stored_variant))
        recomputed = by_key.get((session, size, phase, recomputed_variant))
        if not stored or not recomputed:
            continue
        first = stored["round_medians_ms"]
        second = recomputed["round_medians_ms"]
        if len(first) != len(second):
            raise ValueError("paired variants have different round counts")
        delta = [b - a for a, b in zip(first, second)]
        ratio = [a / b for a, b in zip(first, second)]
        seed = 31019 + size + sum(map(ord, session + phase))
        result.append({"session": session, "matrix_size": size, "phase": phase,
                       "stored_variant": stored_variant, "recomputed_variant": recomputed_variant,
                       "round_count": len(first), "median_recomputed_minus_stored_ms": statistics.median(delta),
                       "delta_bootstrap_95pct_ms": _bootstrap_interval(delta, seed),
                       "median_stored_over_recomputed_ratio": statistics.median(ratio),
                       "ratio_bootstrap_95pct": _bootstrap_interval(ratio, seed + 1),
                       "round_deltas_ms": delta, "round_ratios": ratio})
    return result


def compare_sessions(pairs: list[dict]) -> list[dict]:
    by_size: dict[int, dict[str, dict]] = defaultdict(dict)
    for row in pairs:
        by_size[row["matrix_size"]][row["session"]] = row
    result = []
    for size, sessions in sorted(by_size.items()):
        if set(sessions) != {"A", "B"}:
            raise ValueError(f"both independent sessions are required at N={size}")
        a = sessions["A"]["median_stored_over_recomputed_ratio"]
        b = sessions["B"]["median_stored_over_recomputed_ratio"]
        a_interval = sessions["A"]["ratio_bootstrap_95pct"]
        b_interval = sessions["B"]["ratio_bootstrap_95pct"]
        intervals_overlap = max(a_interval[0], b_interval[0]) <= min(a_interval[1], b_interval[1])
        result.append({"matrix_size": size, "session_a_ratio": a, "session_b_ratio": b,
                       "direction_agrees": (a > 1 and b > 1) or (a < 1 and b < 1) or (a == b == 1),
                       "ratio_magnitude_difference": abs(a - b),
                       "session_a_95pct": a_interval, "session_b_95pct": b_interval,
                       "intervals_overlap": intervals_overlap,
                       "material_disagreement": not intervals_overlap and
                       ((a - 1) * (b - 1) < 0 or abs(a - b) > 0.25)})
    return result


_ADDRESS = re.compile(r"/\*[0-9a-fA-F]{4,8}\*/")
_MNEMONIC = re.compile(r"\b(LDGSTS|LDG|STG|LDS|STS|FADD|FMUL|FFMA|HMMA|MMA|IMAD|IADD3)\b")


def normalize_sass(text: str) -> list[str]:
    """Drop instruction addresses, not register or operand differences."""
    return [" ".join(_ADDRESS.sub("", line).split()) for line in text.splitlines()
            if _ADDRESS.search(line)]


def compiler_diff_summary(stored: dict, recomputed: dict, stored_sass: str,
                          recomputed_sass: str, max_diff_lines: int = 80) -> dict:
    left = normalize_sass(stored_sass)
    right = normalize_sass(recomputed_sass)
    diff = list(difflib.unified_diff(left, right, fromfile="stored", tofile="recomputed", lineterm=""))
    def counts(lines: list[str]) -> dict[str, int]:
        result: dict[str, int] = {}
        for line in lines:
            match = _MNEMONIC.search(line)
            if match:
                result[match.group(1)] = result.get(match.group(1), 0) + 1
        return result
    return {"stored_cubin_sha256": stored["cubin_sha256"],
            "recomputed_cubin_sha256": recomputed["cubin_sha256"],
            "same_cubin": stored["cubin_sha256"] == recomputed["cubin_sha256"],
            "registers_per_thread": [stored["registers_per_thread"], recomputed["registers_per_thread"]],
            "static_shared_memory_bytes": [stored["static_shared_memory_bytes"],
                                           recomputed["static_shared_memory_bytes"]],
            "static_sass_mnemonic_counts": {"stored": counts(left), "recomputed": counts(right)},
            "normalized_sass_sha256": [hashlib.sha256("\n".join(lines).encode()).hexdigest()
                                       for lines in (left, right)],
            "bounded_normalized_sass_diff": diff[:max_diff_lines],
            "diff_lines_total": len(diff), "diff_truncated": len(diff) > max_diff_lines}


def join_static_measured(selected: list[dict], sizes: list[int], complete: list[dict],
                         stages: list[dict], topology: dict, profiler: dict) -> list[dict]:
    by_name = {entry["name"]: entry for entry in selected}
    complete_by = {(row["session"], row["matrix_size"], row["variant"]): row for row in complete}
    stage_by = {(row["matrix_size"], row["variant"]): row for row in stages if row["session"] == "A"}
    rows = []
    for size in sizes:
        for variant in (STORED, RECOMPUTED):
            metrics = by_name[variant]["static_per_size"][size]["metrics"]
            rows.append({"matrix_size": size, "variant": variant,
                         "schedule_hash": by_name[variant]["static_per_size"][size]["schedule_hash"],
                         "modeled_temporary_hbm_bytes": metrics["temporary_hbm_traffic_bytes"],
                         "modeled_recompute_operations": metrics["recomputation_arithmetic_operations"],
                         "modeled_extra_source_read_bytes": metrics["extra_source_read_bytes"],
                         "modeled_launch_groups": metrics["logical_launch_groups"],
                         "actual_launches": len(topology[variant]),
                         "complete_median_ms_session_a": complete_by.get(("A", size, variant), {}).get("median_ms"),
                         "complete_median_ms_session_b": complete_by.get(("B", size, variant), {}).get("median_ms"),
                         "consumer_stage_median_ms": stage_by.get((size, variant), {}).get("median_ms"),
                         "profiler": profiler.get((size, variant))})
    return rows


def classify_mechanism(session_comparison: list[dict], complete_pairs: list[dict],
                       consumer_pairs: list[dict], materialization: list[dict],
                       graph_pairs: list[dict], dummy_pairs: list[dict],
                       graph_eager_pairs: list[dict] | None = None) -> dict:
    """Conservative, explicit label from matched observations; never infer DRAM causality."""
    if not session_comparison:
        return {"primary": "INCONCLUSIVE", "decision": "INCONCLUSIVE",
                "reason": "Both fresh-process timing sessions are required."}
    if any(row.get("material_disagreement", not row["direction_agrees"])
           for row in session_comparison):
        return {"primary": "SESSION_UNSTABLE", "decision": "SESSION_UNSTABLE",
                "reason": "Fresh-process complete-operation evidence has a material discrepancy."}
    complete_good = {row["matrix_size"] for row in complete_pairs if row["session"] == "A"
                     and row["ratio_bootstrap_95pct"][0] > 1}
    complete_good &= {row["matrix_size"] for row in complete_pairs if row["session"] == "B"
                      and row["ratio_bootstrap_95pct"][0] > 1}
    consumer_slower = {row["matrix_size"] for row in consumer_pairs
                       if row["delta_bootstrap_95pct_ms"][0] > 0}
    consumer_faster = {row["matrix_size"] for row in consumer_pairs
                       if row["delta_bootstrap_95pct_ms"][1] < 0}
    prep_by_size = {row["matrix_size"]: row["median_ms"] for row in materialization}
    supported_slower = {size for size in complete_good & consumer_slower
                        if prep_by_size.get(size, 0) > next(
                            row["median_recomputed_minus_stored_ms"] for row in consumer_pairs
                            if row["matrix_size"] == size)}
    if len(supported_slower) >= 2:
        return {"primary": "RECOMPUTE_COSTS_BUT_MATERIALIZATION_COSTS_MORE",
                "decision": "MECHANISM_IDENTIFIED", "supporting_sizes": sorted(supported_slower),
                "reason": "Consumer recomputation costs more locally, but complete latency favors it and the removed stage is larger."}
    if len(complete_good & consumer_faster) >= 2:
        return {"primary": "RECOMPUTE_CONSUMER_ITSELF_FASTER",
                "decision": "MECHANISM_IDENTIFIED", "supporting_sizes": sorted(complete_good & consumer_faster),
                "reason": "Both consumer and complete timings favor recomputation in two independent complete sessions."}
    if graph_eager_pairs and dummy_pairs and len(complete_good) >= 2:
        graph_by = {row["matrix_size"]: row for row in graph_pairs}
        eager_by = {row["matrix_size"]: row for row in graph_eager_pairs}
        dummy_by = {row["matrix_size"]: row for row in dummy_pairs}
        launch_supported = set()
        for size in complete_good & graph_by.keys() & eager_by.keys() & dummy_by.keys():
            eager_gap = -eager_by[size]["median_recomputed_minus_stored_ms"]
            graph_gap = -graph_by[size]["median_recomputed_minus_stored_ms"]
            dummy_ratio = dummy_by[size]["median_stored_over_recomputed_ratio"]
            if eager_gap > 0 and abs(graph_gap) <= 0.5 * eager_gap and 0.95 <= dummy_ratio <= 1.05:
                launch_supported.add(size)
        if len(launch_supported) >= 2:
            return {"primary": "LAUNCH_REDUCTION_DOMINANT_SIGNAL",
                    "decision": "MECHANISM_IDENTIFIED", "supporting_sizes": sorted(launch_supported),
                    "reason": "Both graph replay and the equal-launch control erase most of the matched eager gap."}
    if complete_good and graph_pairs and dummy_pairs:
        # A diagnostic can strengthen a mixed interpretation but cannot supply
        # a launch-causality label without stable component evidence.
        return {"primary": "MIXED_MECHANISM", "decision": "MIXED_BUT_ACTIONABLE",
                "supporting_sizes": sorted(complete_good),
                "reason": "Complete advantage repeats, but component and launch controls do not isolate one cause."}
    return {"primary": "INCONCLUSIVE", "decision": "INCONCLUSIVE",
            "reason": "Matched component or fresh-process evidence is insufficient for a specific mechanism."}
