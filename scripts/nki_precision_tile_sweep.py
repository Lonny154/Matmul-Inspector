import csv
import json
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = 5
STAMP = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
OUTPUT = ROOT / "artifacts" / f"nki-controlled-sweep-{STAMP}"

KERNELS = {
    "fp32_n128": "nki-trn1-512",
    "bf16_n128": "nki-trn1-bf16",
    "fp32_n512": "nki-trn1-fp32-512tile",
    "bf16_n512": "nki-trn1-bf16-512",
}


def main():
    OUTPUT.mkdir(parents=True)
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

            sample_path = info_path.with_name("nc_latency_data.json")
            sample_data = json.loads(sample_path.read_text())
            samples = [
                value
                for values in sample_data["latency_data"].values()
                for value in values
            ]

            if not samples:
                raise RuntimeError(f"No latency samples: {sample_path}")

            row = {
                "round": round_num,
                "configuration": name,
                "ncl_p50_us": info["nc_latency"]["50"],
                "ncl_p99_us": info["nc_latency"]["99"],
                "overall_p50_us": info["latency"]["50"],
                "throughput_inf_s": info["throughput"],
                "sample_count": len(samples),
                "neff": str(neff.relative_to(ROOT)),
                "info_json": str(info_path.relative_to(ROOT)),
            }
            rows.append(row)
            print(
                f"{name}: NCL p50={row['ncl_p50_us']} us, "
                f"p99={row['ncl_p99_us']} us",
                flush=True,
            )

            # Save partial results after every successful run.
            csv_path = OUTPUT / "summary.csv"
            with csv_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

    print(f"\nCompleted {len(rows)} benchmark runs.")
    print(f"Results: {OUTPUT / 'summary.csv'}")


if __name__ == "__main__":
    main()
