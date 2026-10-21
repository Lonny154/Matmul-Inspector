"""CPU-only checks for the bounded timing probe's offline evidence rules."""

import json

import pytest

from scripts import mi_timing_1_probe as probe


def fixture_rows():
    rows = []
    for workload in probe.WORKLOADS:
        for mode, (count, submissions) in probe.MODES.items():
            for round_id in range(1, probe.ROUNDS + 1):
                for slot in probe.SLOTS:
                    event = float(10 + round_id + (slot == "B"))
                    rows.append({"workload": workload, "mode": mode, "round": round_id,
                                 "slot": slot, "operation_count": count,
                                 "submission_calls": submissions,
                                 "cuda_event_ms": event, "perf_counter_ms": event - 0.2,
                                 "monotonic_raw_ms": event + 0.1,
                                 "event_minus_perf_ms": 0.2,
                                 "event_minus_raw_ms": -0.1,
                                 "raw_over_perf_ratio": (event + 0.1) / (event - 0.2)})
    return rows


def test_summary_uses_complete_null_pairs_and_total_block_durations():
    summary = probe.summarize(fixture_rows())
    assert len(summary) == 6
    entry = summary["fc1/eager_100"]
    assert entry["cuda_event_ms"]["count"] == 16
    assert entry["paired_A_minus_B_ms"]["cuda_event_ms"]["median"] == -1.0
    assert entry["event_minus_raw_ms"]["median"] == -0.1


def test_summary_rejects_missing_and_changed_operation_counts():
    rows = fixture_rows()
    with pytest.raises(ValueError, match="incomplete"):
        probe.summarize(rows[:-1])
    rows[0]["operation_count"] = 2
    with pytest.raises(ValueError, match="count"):
        probe.summarize(rows)


def test_manifest_and_summary_replay_detect_tampering(tmp_path):
    rows = fixture_rows()
    probe.write_json(tmp_path / "summary.json", probe.summarize(rows))
    (tmp_path / "raw_blocks.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows))
    probe.write_json(tmp_path / "hash_manifest.json", {
        path.name: probe.sha256(path) for path in tmp_path.iterdir()
        if path.name != "hash_manifest.json"})
    assert probe.verify(tmp_path) == {"status": "PASS", "blocks": 96, "files": 2}
    (tmp_path / "raw_blocks.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="manifest"):
        probe.verify(tmp_path)
