#!/usr/bin/env python3
"""Summarize global-versus-fused device-resident PCR benchmark results."""

import argparse
import csv
import math
from pathlib import Path


TIE_FRACTION = 0.02
WORK_THRESHOLD = 786_432
BLOCK_THREADS = 256


def load_points(path: Path):
    pairs = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row.get("timing_scope") != "device_resident":
                continue
            execution = row.get("batch_execution")
            if execution not in {"true_batched_device_resident", "true_batched_fused"}:
                continue
            key = (int(row["system_size"]), int(row["batch_size"]))
            pairs.setdefault(key, {})[execution] = float(row["median_ms"])
    points = []
    for (n, batch), timings in sorted(pairs.items()):
        if len(timings) != 2:
            continue
        global_ms = timings["true_batched_device_resident"]
        fused_ms = timings["true_batched_fused"]
        speedup = global_ms / fused_ms
        if 1.0 - TIE_FRACTION <= speedup <= 1.0 + TIE_FRACTION:
            winner = "tie"
        else:
            winner = "fused" if speedup > 1.0 else "global"
        total = n * batch
        points.append({
            "system_size": n,
            "batch_size": batch,
            "total_equations": total,
            "fused_blocks": math.ceil(total / BLOCK_THREADS),
            "global_median_ms": global_ms,
            "fused_median_ms": fused_ms,
            "speedup": speedup,
            "winner": winner,
        })
    if not points:
        raise ValueError("No paired global/fused device-resident rows found")
    return points


def evaluate(points, name, choose_fused):
    decisive = [point for point in points if point["winner"] != "tie"]
    false_fusion = false_global = correct = 0
    runtime = 0.0
    for point in points:
        prediction = "fused" if choose_fused(point) else "global"
        runtime += point[f"{prediction}_median_ms"]
        if point["winner"] == "tie":
            continue
        correct += prediction == point["winner"]
        false_fusion += prediction == "fused" and point["winner"] == "global"
        false_global += prediction == "global" and point["winner"] == "fused"
    return name, correct / len(decisive), false_fusion, false_global, runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sm-count", type=int, help="Adds a one-block-per-SM wave estimate")
    args = parser.parse_args()
    if args.sm_count is not None and args.sm_count <= 0:
        parser.error("--sm-count must be positive")
    source = args.run_directory / "summary.csv"
    output = args.output or args.run_directory / "dispatch_analysis.csv"
    points = load_points(source)

    fields = list(points[0])
    if args.sm_count:
        fields.append("estimated_block_waves")
        for point in points:
            point["estimated_block_waves"] = math.ceil(
                point["fused_blocks"] / args.sm_count)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(points)

    rules = [
        ("A total_work", lambda p: p["total_equations"] >= WORK_THRESHOLD),
        ("B minimum_N_and_B", lambda p: p["system_size"] >= 1536 and p["batch_size"] >= 512),
        ("C combined", lambda p: p["total_equations"] >= WORK_THRESHOLD and p["system_size"] >= 1536),
        ("D fused_blocks", lambda p: p["fused_blocks"] >= WORK_THRESHOLD // BLOCK_THREADS),
    ]
    global_total = sum(point["global_median_ms"] for point in points)
    fused_total = sum(point["fused_median_ms"] for point in points)
    oracle_total = sum(min(point["global_median_ms"], point["fused_median_ms"])
                       for point in points)
    print(f"points={len(points)} output={output}")
    print(f"always_global_ms={global_total:.6f} always_fused_ms={fused_total:.6f} "
          f"oracle_ms={oracle_total:.6f}")
    for rule in rules:
        name, accuracy, false_fusion, false_global, runtime = evaluate(points, *rule)
        print(f"{name}: accuracy={accuracy:.1%} false_fusion={false_fusion} "
              f"false_global={false_global} aggregate_ms={runtime:.6f}")


if __name__ == "__main__":
    main()
