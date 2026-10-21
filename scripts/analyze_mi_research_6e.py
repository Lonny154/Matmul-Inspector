"""CPU-only timing-control replay and decision-rule sensitivity analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

try:
    from scripts import run_mi_research_2c_graph as artifacts
    from scripts import run_mi_research_6d_graph as historical
    from scripts import run_mi_research_6e_controls as study
except ModuleNotFoundError:
    import run_mi_research_2c_graph as artifacts
    import run_mi_research_6d_graph as historical
    import run_mi_research_6e_controls as study

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "artifacts/mi-research-6e"


def empirical_quantile(values: list[float], probability: float) -> float:
    """Linear interpolation between sorted observations; descriptive only."""
    ordered = sorted(values)
    location = (len(ordered) - 1) * probability
    low = int(location)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (location - low) * (ordered[high] - ordered[low])


def trace_summary(trace: dict, algorithm: str) -> dict:
    kernels = sorted((event for event in trace["traceEvents"] if event.get("cat") == "kernel"),
                     key=lambda event: event["ts"])
    stride = 24 if algorithm == "staged" else 1
    if len(kernels) != study.OPERATIONS * stride:
        raise ValueError("profiled graph event count changed")
    durations = [float(event["dur"]) for event in kernels]
    gaps = [max(0.0, kernels[i + 1]["ts"] - (event["ts"] + event["dur"]))
            for i, event in enumerate(kernels[:-1])]
    spans = [float(group[-1]["ts"] + group[-1]["dur"] - group[0]["ts"])
             for group in (kernels[i:i + stride] for i in range(0, len(kernels), stride))]
    result = {"kernel_count": len(kernels), "kernel_duration_median_us": statistics.median(durations),
              "kernel_duration_max_us": max(durations),
              "inter_kernel_gap_median_us": statistics.median(gaps) if gaps else None,
              "inter_kernel_gap_max_us": max(gaps) if gaps else None,
              "operation_span_median_us": statistics.median(spans),
              "operation_span_max_us": max(spans),
              "instrumented_trace_not_primary_timing": True}
    if algorithm == "staged":
        result["product_duration_median_us"] = statistics.median(
            event["dur"] for index, event in enumerate(kernels) if index % stride != 23)
        result["reconstruction_duration_median_us"] = statistics.median(
            event["dur"] for index, event in enumerate(kernels) if index % stride == 23)
    return result


def sample_view(sample: dict) -> dict:
    return {"round": sample["round"], "phase": sample["phase"], "position": sample["position"],
            "event_us_per_operation": sample["cuda_event_elapsed_ms_per_operation"] * 1000,
            "host_submission_us_per_operation": sample["host_submission_ms_per_block"] * 1000 / study.OPERATIONS,
            "synchronized_perf_us_per_operation": sample["host_wall_ms_per_block"] * 1000 / study.OPERATIONS,
            "raw_us_per_operation": sample.get("monotonic_raw_ms_per_block", 0) * 1000 / study.OPERATIONS
                                     if "monotonic_raw_ms_per_block" in sample else None,
            "outer_start_perf_ns": sample["outer_start_perf_ns"],
            "outer_end_perf_ns": sample["outer_end_perf_ns"]}


def session_analysis(bundle: Path) -> dict:
    verification = study.verify(bundle)
    if verification["status"] != "PASS":
        raise ValueError(f"6E session verification failed: {bundle}")
    raw = json.loads((bundle / "raw.json").read_text())
    summary = json.loads((bundle / "summary.json").read_text())
    if not raw["precision"]["restored"]:
        raise ValueError("precision flags were not restored")
    rounds = raw["benchmark"]["rounds"]
    values = summary["event_us"]
    absolute = [abs(x) for x in summary["paired_X_minus_Y_us"]]
    first = [row["measurements"][row["order"][0]] for row in rounds]
    second = [row["measurements"][row["order"][1]] for row in rounds]
    slowest = {}
    for slot in ("X", "Y"):
        selected = max((row["measurements"][slot] for row in rounds),
                       key=lambda sample: sample["cuda_event_elapsed_ms_per_operation"])
        slowest[slot] = sample_view(selected)
    return {"condition": raw["condition"], "session": raw["session"],
            "manifest_sha256": artifacts.digest(bundle / "hash_manifest.json"),
            "recovered_summary_after_complete_collection": (bundle / "recovery.json").exists(),
            "capture_order": raw["capture_order"], "environment": raw["environment"],
            "clock_preflight": raw["benchmark"]["host_clock_preflight"]["status"],
            "event_us": {slot: {key: values[slot][key] for key in
                                 ("median_us", "min_us", "max_us", "mad_us", "cv")}
                         for slot in ("X", "Y")},
            "paired_X_minus_Y_us": summary["paired_X_minus_Y_us"],
            "paired_abs_median_us": statistics.median(absolute),
            "paired_abs_q90_us": empirical_quantile(absolute, 0.9),
            "paired_abs_max_us": max(absolute),
            "paired_abs_second_max_us": sorted(absolute)[-2],
            "positive_rounds": summary["positive_rounds"],
            "first_half_median_us": {slot: statistics.median(values[slot]["rounds_us"][:4])
                                     for slot in ("X", "Y")},
            "second_half_median_us": {slot: statistics.median(values[slot]["rounds_us"][4:])
                                      for slot in ("X", "Y")},
            "first_position_event_median_us": statistics.median(
                sample["cuda_event_elapsed_ms_per_operation"] * 1000 for sample in first),
            "second_position_event_median_us": statistics.median(
                sample["cuda_event_elapsed_ms_per_operation"] * 1000 for sample in second),
            "slowest_event_block": slowest,
            "same_instance_aa_us": summary.get("same_instance_aa_us"),
            "existing_rule": {key: summary["existing_rule"][key] for key in
                              ("decision", "reason", "warnings", "ab_exceeds_aa_variability")}
                             if "existing_rule" in summary else None,
            "profiled_replays": {slot: trace_summary(json.loads((bundle / f"{slot}_graph_trace.json").read_text()),
                                                    raw["algorithms"][slot]) for slot in ("X", "Y")}}


def analyze(root: Path = EVIDENCE) -> dict:
    study.check_protocol(json.loads((root / "protocol.json").read_text()))
    sessions = {condition: [session_analysis(root / f"{condition}-s{number}")
                            for number in (1, 2)] for condition in study.CONDITIONS}
    for condition, rows in sessions.items():
        if [row["session"] for row in rows] != [1, 2] or any(row["condition"] != condition for row in rows):
            raise ValueError("missing or mislabeled 6E session")
    historical_decisions = []
    for number in (1, 2):
        path = ROOT / f"artifacts/mi-research-6d/graph-s{number}"
        if historical.verify(path)["status"] != "PASS":
            raise ValueError("historical 6D evidence failed verification")
        old = json.loads((path / "summary.json").read_text())
        old_raw = json.loads((path / "raw.json").read_text())
        aa_blocks = []
        for row in old_raw["benchmark"]["aa_rounds"]:
            aa_blocks.append({"round": row["round"], "order": row["order"],
                              "measurements": {slot: {
                                  "event_us_per_operation": sample["cuda_event_elapsed_ms_per_operation"] * 1000,
                                  "host_submission_us_per_operation": sample["host_submission_ms_per_block"]
                                                                      * 1000 / study.OPERATIONS,
                                  "synchronized_perf_us_per_operation": sample["host_wall_ms_per_block"]
                                                                          * 1000 / study.OPERATIONS,
                                  "raw_us_per_operation": sample["monotonic_raw_ms_per_block"]
                                                          * 1000 / study.OPERATIONS}
                                  for slot, sample in row["measurements"].items()}})
        historical_decisions.append({"session": number, "manifest_sha256": artifacts.digest(path / "hash_manifest.json"),
                                     "decision": old["decision"]["decision"],
                                     "paired_A_minus_B_us": old["paired_A_minus_B_us"],
                                     "aa_differences_us": old["aa_differences_us"],
                                     "aa_max_absolute_gap_us": old["aa_max_absolute_gap_us"],
                                     "aa_blocks": aa_blocks})
    sensitivity = []
    for number in (1, 2):
        cross = sessions["cross"][number - 1]
        staged = sessions["staged_null"][number - 1]
        fused = sessions["fused_null"][number - 1]
        gaps = cross["paired_X_minus_Y_us"]
        same_max = max(abs(x) for x in cross["same_instance_aa_us"])
        sensitivity.append({"session": number,
                            "existing_rule": cross["existing_rule"]["decision"],
                            "positive_cross_pairs": sum(x > 0 for x in gaps),
                            "cross_min_gap_us": min(gaps),
                            "same_instance_staged_aa_max_us": same_max,
                            "independent_staged_null_abs_q90_us": staged["paired_abs_q90_us"],
                            "independent_staged_null_abs_max_us": staged["paired_abs_max_us"],
                            "independent_fused_null_abs_q90_us": fused["paired_abs_q90_us"],
                            "independent_fused_null_abs_max_us": fused["paired_abs_max_us"],
                            "all_cross_gaps_exceed_same_instance_max": all(x > same_max for x in gaps),
                            "all_cross_gaps_exceed_independent_staged_max": all(
                                x > staged["paired_abs_max_us"] for x in gaps),
                            "all_cross_gaps_exceed_independent_staged_q90": all(
                                x > staged["paired_abs_q90_us"] for x in gaps)})
    # No mechanism can be assigned from event/host intervals or normal profiled
    # replays alone. This bounded study does not prospectively validate a new rule.
    classification = "C"
    return {"schema_version": 1, "milestone": "MI-Research-6E",
            "protocol_sha256": artifacts.digest(root / "protocol.json"),
            "sessions": sessions, "sensitivity": sensitivity,
            "historical_6d_unchanged": historical_decisions,
            "classification": classification,
            "decision_rule_changed": False,
            "interpretation": "Controls and cross sessions quantify variability and max-gap sensitivity, but do not isolate a reproducible physical or measurement-window cause of the 6D staged A/A gaps or validate a replacement rule."}


def run(*, verify: bool = False, root: Path = EVIDENCE) -> dict:
    result = analyze(root)
    path = root / "analysis.json"
    manifest = root / "analysis_manifest.json"
    expected = {"analysis.json": artifacts.digest(path) if path.exists() else None,
                "protocol.json": artifacts.digest(root / "protocol.json")}
    expected.update({f"{condition}-s{session}/hash_manifest.json": artifacts.digest(
        root / f"{condition}-s{session}/hash_manifest.json")
        for condition in study.CONDITIONS for session in (1, 2)})
    if verify:
        if result != json.loads(path.read_text()) or expected != json.loads(manifest.read_text()):
            raise ValueError("6E cross-session analysis replay mismatch")
    else:
        if path.exists() or manifest.exists():
            raise FileExistsError("6E analysis already exists")
        artifacts.write_json(path, result)
        expected["analysis.json"] = artifacts.digest(path)
        artifacts.write_json(manifest, expected)
    return {"status": "PASS", "classification": result["classification"],
            "cross_decisions": [row["existing_rule"]["decision"] for row in result["sessions"]["cross"]]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--root", type=Path, default=EVIDENCE)
    args = parser.parse_args()
    print(json.dumps(run(verify=args.verify, root=args.root), sort_keys=True))


if __name__ == "__main__":
    main()
