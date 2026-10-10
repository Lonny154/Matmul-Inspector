
import csv
import json
import random
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = 5
STAMP = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
OUTPUT = ROOT / "artifacts" / f"nki-k-blocking-sweep-{STAMP}"

KERNELS = {
    "bf16_baseline": "nki-bf16_1024_n512",
    "bf16_k_blocked": "nki-bf16_1024_kblocked",
}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    rng = random.Random(42)

    for round_num in range(1, RUNS + 1):
        order = list(KERNELS)
        rng.shuffle(order)

        print(f"\nRound {round_num}/{RUNS}: {order}", flush=True)

        for name in order:
            neff = ROOT / "artifacts" / KERNELS[name] / "kernel.neff"
            destination = OUTPUT / f"round_{round_num}" / name

            if not neff.is_file():
                raise FileNotFoundError(neff)

            command = [
                "neuron-bench", "exec",
                "--warmup=20",
                "--work=200",
                "--enable-only-latency",
                "--fixed-nc-count=1",
                f"--output-directory={destination}",
                str(neff),
            ]

            subprocess.run(command, check=True)

            info_files = list(destination.rglob("info.json"))
            if len(info_files) != 1:
                raise RuntimeError(
                    f"Expected one info.json in {destination}, "
                    f"found {len(info_files)}"
                )

            info_path = info_files[0]
            info = json.loads(info_path.read_text())

            row = {
                "round": round_num,
                "configuration": name,
                "ncl_p50_us": float(info["nc_latency"]["50"]),
                "ncl_p99_us": float(info["nc_latency"]["99"]),
                "overall_p50_us": float(info["latency"]["50"]),
                "throughput_inf_s": float(info["throughput"]),
                "neff": str(neff.relative_to(ROOT)),
            }
            rows.append(row)

            print(
                f"{name}: NCL p50={row['ncl_p50_us']} us",
                flush=True,
            )

            with (OUTPUT / "summary.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

    averages = {
        name: statistics.mean(
            r["ncl_p50_us"]
            for r in rows
            if r["configuration"] == name
        )
        for name in KERNELS
    }

    baseline = averages["bf16_baseline"]
    blocked = averages["bf16_k_blocked"]

    print("\n===== RESULTS =====")
    print(f"Baseline mean p50: {baseline:.2f} us")
    print(f"K-blocked mean p50: {blocked:.2f} us")
    print(f"Speedup: {baseline / blocked:.3f}x")
    print(f"Latency reduction: {(baseline - blocked) / baseline * 100:.2f}%")
    print(f"\nResults saved: {OUTPUT / 'summary.csv'}")


if __name__ == "__main__":
    main()
