"""Verify and summarize the bounded MI-Research-6F node-level trace offline."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sqlite3
import statistics

try:
    from scripts import run_mi_research_6f_trace as capture
    from scripts import run_mi_research_2c_graph as evidence
except ModuleNotFoundError:
    import run_mi_research_6f_trace as capture
    import run_mi_research_2c_graph as evidence


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/mi-research-6f"
SQLITE = BASE / "session2_node_trace.sqlite"
REPORT = BASE / "session2_node_trace.nsys-rep"
ANALYSIS = BASE / "trace_analysis.json"
TIMELINE = BASE / "kernel_timeline.csv"
MANIFEST = BASE / "analysis_manifest.json"
MANIFEST_FILES = (
    "protocol.json", "capture_amendment.json", "session1/hash_manifest.json",
    "session2/hash_manifest.json", "session2_node_trace.nsys-rep",
    "session2_node_trace.sqlite", "trace_analysis.json", "kernel_timeline.csv")


def _rows(connection: sqlite3.Connection, query: str) -> list[dict]:
    return [dict(row) for row in connection.execute(query)]


def analyze() -> tuple[dict, list[dict]]:
    for session in (1, 2):
        if capture.verify(BASE / f"session{session}")["status"] != "PASS":
            raise ValueError(f"session {session} failed verification")
    raw1 = json.loads((BASE / "session1/raw.json").read_text())
    raw2 = json.loads((BASE / "session2/raw.json").read_text())
    if raw2.get("capture_trigger") != "cudaProfilerApi":
        raise ValueError("unexpected node trace trigger")
    if not REPORT.exists() or not SQLITE.exists():
        raise ValueError("missing retained node trace")
    con = sqlite3.connect(SQLITE)
    con.row_factory = sqlite3.Row
    try:
        nvtx = _rows(con, "SELECT start,end,text FROM NVTX_EVENTS WHERE text LIKE 'MI6F_%'")
        runtime = _rows(con, """SELECT r.start,r.end,s.value AS name,r.correlationId
            FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId
            ORDER BY r.start""")
        kernels = _rows(con, """SELECT k.start,k.end,s.value AS name,k.graphId,
            k.graphNodeId,k.streamId,k.correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds s ON s.id=k.shortName ORDER BY k.start,k.graphNodeId""")
        diagnostics = _rows(con, "SELECT text FROM DIAGNOSTIC_EVENT ORDER BY timestamp")
        gpu = _rows(con, "SELECT name,smCount FROM TARGET_INFO_GPU")
    finally:
        con.close()
    if len(kernels) != 960 or len(nvtx) != 3:
        raise ValueError("node or NVTX event count changed")
    by_text = {row["text"]: row for row in nvtx}
    outer = by_text.get("MI6F_TRACE")
    if outer is None or outer["end"] is None:
        raise ValueError("missing outer trace range")
    graph_calls = [row for row in runtime if row["name"].startswith("cudaGraphLaunch")]
    if len(graph_calls) != 2:
        raise ValueError("expected exactly two graph launches")
    blocks = []
    timeline = []
    node_sequence = None
    for index, call in enumerate(graph_calls, 1):
        replay_id = f"MI6F_S2_NODE_B{index:02d}"
        label = by_text.get(replay_id)
        if (label is None or label["end"] is None or
                not outer["start"] <= label["start"] < call["start"] < call["end"] < label["end"] <= outer["end"]):
            raise ValueError(f"graph API/NVTX correlation failed: {replay_id}")
        activity = [row for row in kernels if row["correlationId"] == call["correlationId"]]
        if (len(activity) != 480 or any(row["name"] not in ("_block_product", "_reconstruct")
                                    for row in activity) or
                len({(row["graphId"], row["streamId"]) for row in activity}) != 1):
            raise ValueError(f"unexpected node topology: {replay_id}")
        names = [row["name"] for row in activity]
        expected = ["_block_product"] * 23 + ["_reconstruct"]
        if names != expected * 20:
            raise ValueError(f"product/reconstruction order changed: {replay_id}")
        node_ids = [row["graphNodeId"] for row in activity]
        if len(set(node_ids)) != 480 or (node_sequence is not None and node_ids != node_sequence):
            raise ValueError("graph node IDs changed between replays")
        node_sequence = node_ids
        if any(a["end"] > b["start"] for a, b in zip(activity, activity[1:])):
            raise ValueError("overlapping kernels make gap sums invalid")
        gaps = [b["start"] - a["end"] for a, b in zip(activity, activity[1:])]
        kernel_ns = sum(row["end"] - row["start"] for row in activity)
        span_ns = activity[-1]["end"] - activity[0]["start"]
        if kernel_ns + sum(gaps) != span_ns:
            raise ValueError("kernel span does not reconcile")
        sample = raw2["measurements"][index - 1]
        if sample["replay_id"] != replay_id:
            raise ValueError("measurement/replay ID mismatch")
        for ordinal, row in enumerate(activity):
            timeline.append({"replay_id": replay_id, "ordinal": ordinal,
                             "operation": ordinal // 24 + 1,
                             "node_in_operation": ordinal % 24 + 1, **row})
        blocks.append({
            "replay_id": replay_id,
            "cuda_event_us_per_operation": sample["cuda_event_elapsed_ms_per_operation"] * 1000,
            "cuda_event_us_per_block": sample["cuda_event_elapsed_ms_per_block"] * 1000,
            "host_submission_us_per_block": sample["host_submission_ms_per_block"] * 1000,
            "nvtx_range_us": (label["end"] - label["start"]) / 1000,
            "graph_launch_api_us": (call["end"] - call["start"]) / 1000,
            "api_end_to_first_kernel_us": (activity[0]["start"] - call["end"]) / 1000,
            "first_kernel_start_ns": activity[0]["start"],
            "last_kernel_end_ns": activity[-1]["end"],
            "kernel_span_us": span_ns / 1000,
            "kernel_sum_us": kernel_ns / 1000,
            "product_sum_us": sum(row["end"] - row["start"] for row in activity
                                  if row["name"] == "_block_product") / 1000,
            "reconstruction_sum_us": sum(row["end"] - row["start"] for row in activity
                                         if row["name"] == "_reconstruct") / 1000,
            "inter_kernel_gap_sum_us": sum(gaps) / 1000,
            "largest_inter_kernel_gap_us": max(gaps) / 1000,
            "kernel_count": len(activity), "product_count": 460, "reconstruction_count": 20,
            "graph_id": activity[0]["graphId"], "stream_id": activity[0]["streamId"],
            "graph_launch_correlation_id": call["correlationId"],
            "first_graph_node_id": node_ids[0], "last_graph_node_id": node_ids[-1]})
    historical = [sample["cuda_event_elapsed_ms_per_operation"] * 1000
                  for sample in raw1["measurements"]]
    baseline = statistics.median(historical[:4])
    result = {
        "schema_version": 1,
        "source": {"node_report_sha256": evidence.digest(REPORT),
                   "node_sqlite_sha256": evidence.digest(SQLITE),
                   "session1_manifest_sha256": evidence.digest(BASE / "session1/hash_manifest.json"),
                   "session2_manifest_sha256": evidence.digest(BASE / "session2/hash_manifest.json")},
        "gpu": gpu,
        "session1": {"trace_status": "no Nsight report generated; NVTX trigger failed",
                     "blocks": 12, "minimum_event_us_per_operation": min(historical),
                     "median_event_us_per_operation": statistics.median(historical),
                     "maximum_event_us_per_operation": max(historical),
                     "predeclared_profiled_relative_baseline_us_per_operation": baseline,
                     "relative_anomaly_threshold_us_per_operation": baseline * 1.35,
                     "relative_anomaly_count": sum(x >= baseline * 1.35 for x in historical),
                     "historical_slow_band_count": sum(x >= 80 for x in historical)},
        "session2": {"trace_status": "node trace exported and correlated by NVTX range plus graph launch correlation ID",
                     "blocks": blocks,
                     "node_order_identical": True,
                     "all_kernels_single_stream_no_overlap": True,
                     "cuda_event_gpu_timestamps_available_in_sqlite": False,
                     "profiler_overhead_note": "Node instrumentation alters the timing environment; block event-to-kernel difference cannot be partitioned precisely without GPU event timestamps."},
        "trace_diagnostics": [d["text"] for d in diagnostics if
                              "not supported" in d["text"].lower() or
                              "not all" in d["text"].lower() or
                              "legacy" in d["text"].lower()],
        "decision": "C — Still unexplained",
        "decision_basis": "No event interval meeting the historical slow band occurred in the bounded graph session, which generated no trace. Node tracing shows nearly unchanged kernel spans and gaps while host graph-launch API duration differs, but its large overhead prevents attributing the historical unprofiled slow mode."}
    return result, timeline


def write() -> dict:
    if ANALYSIS.exists() or TIMELINE.exists() or MANIFEST.exists():
        raise FileExistsError("6F analysis already retained")
    result, timeline = analyze()
    evidence.write_json(ANALYSIS, result)
    with TIMELINE.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(timeline[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(timeline)
    evidence.write_json(MANIFEST, {name: evidence.digest(BASE / name)
                                   for name in MANIFEST_FILES})
    return verify()


def verify() -> dict:
    manifest = json.loads(MANIFEST.read_text())
    if set(manifest) != set(MANIFEST_FILES) or any(
            evidence.digest(BASE / name) != value for name, value in manifest.items()):
        raise ValueError("6F analysis manifest mismatch")
    actual, timeline = analyze()
    if json.loads(ANALYSIS.read_text()) != actual:
        raise ValueError("6F trace analysis mismatch")
    with TIMELINE.open(newline="") as handle:
        retained = list(csv.DictReader(handle))
    if retained != [{key: str(value) for key, value in row.items()} for row in timeline]:
        raise ValueError("6F kernel timeline mismatch")
    return {"status": "PASS", "decision": actual["decision"],
            "blocks": len(actual["session2"]["blocks"]), "kernel_rows": len(timeline)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify() if args.verify else write(), sort_keys=True))


if __name__ == "__main__":
    main()
