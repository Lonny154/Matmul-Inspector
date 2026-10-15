#!/usr/bin/env python3
"""Generate evidence-backed figures for docs/CaseStudy/TensorGame.md.

Primary inputs (all paths are relative to the repository root):

* Policy regression: artifacts/tensor-game/tg4c-policy-regression-v1/analysis.json
* Local 150-target benchmark: artifacts/tensor-game/tg4w-rollout-decision-v1/summary.json
* AWS replay and cross-host checks:
  artifacts/tensor-game/tg4x-cross-machine-preparation-v1/aws-full/summary.json
  artifacts/tensor-game/tg4x-cross-machine-preparation-v1/cross-machine-comparison.json
* Streaming-heap memory: artifacts/tensor-game/tg4t-bounded-memory-v1/summary.json

The artifacts directory is intentionally ignored by Git. Restore the archived
evidence bundle before regenerating these figures from a clean checkout.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "CaseStudy" / "figures"
POLICY_PATH = ROOT / "artifacts/tensor-game/tg4c-policy-regression-v1/analysis.json"
LOCAL_PATH = ROOT / "artifacts/tensor-game/tg4w-rollout-decision-v1/summary.json"
AWS_PATH = ROOT / "artifacts/tensor-game/tg4x-cross-machine-preparation-v1/aws-full/summary.json"
CROSS_HOST_PATH = ROOT / "artifacts/tensor-game/tg4x-cross-machine-preparation-v1/cross-machine-comparison.json"
MEMORY_PATH = ROOT / "artifacts/tensor-game/tg4t-bounded-memory-v1/summary.json"

BLUE = "#2f6f9f"
ORANGE = "#e07a3f"
GREEN = "#3c8d73"
DARK = "#25313c"
GRID = "#d9e0e6"


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required evidence file is missing: {path.relative_to(ROOT)}\n"
            "Restore the ignored Tensor Game evidence bundle before regenerating figures."
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not load evidence file {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"Evidence validation failed: {message}")


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": DARK,
            "axes.labelcolor": DARK,
            "axes.titlecolor": DARK,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "xtick.color": DARK,
            "ytick.color": DARK,
            "legend.frameon": False,
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUTPUT / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def add_bar_labels(ax: plt.Axes, bars: Any, formatter: Any, pad: float = 0.02) -> None:
    ymax = max(float(bar.get_height()) for bar in bars) or 1.0
    for bar in bars:
        value = float(bar.get_height())
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + ymax * pad,
            formatter(value),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color=DARK,
        )


def policy_regression_figure() -> None:
    data = load_json(POLICY_PATH)
    metrics = data["validation_metrics"]
    frozen = metrics["baseline"]
    trained = metrics["trained"]
    successes = data["success"]

    require(data.get("paired_examples") == 180, "policy comparison must contain 180 paired trials")
    require(frozen.get("examples") == trained.get("examples") == 292, "validation metrics must use 292 decisions")
    require(successes == {"baseline": 24, "trained": 0}, "unexpected exact-discovery totals")

    labels = ["Frozen", "Policy-only"]
    colors = [BLUE, ORANGE]
    panels = [
        ("Validation cross-entropy", [frozen["policy_cross_entropy"], trained["policy_cross_entropy"]], "loss", lambda v: f"{v:.3f}"),
        ("Validation top-1 agreement", [100 * frozen["top1_agreement"], 100 * trained["top1_agreement"]], "%", lambda v: f"{v:.1f}%"),
        ("Held-out exact discovery", [successes["baseline"], successes["trained"]], "successful trials (of 180)", lambda v: f"{int(v)}/180"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.7))
    fig.suptitle("Policy training improved the offline objective but hurt exact discovery", fontsize=16, fontweight="bold", color=DARK)
    fig.text(0.5, 0.915, "Separate panels preserve the different metric scales and evaluation sets.", ha="center", color="#56636f")
    for ax, (title, values, ylabel, formatter) in zip(axes, panels, strict=True):
        bars = ax.bar(labels, values, color=colors, width=0.62)
        ax.set_title(title, pad=10)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, max(values) * 1.25 if max(values) else 1)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        add_bar_labels(ax, bars, formatter)
    fig.text(
        0.5,
        0.015,
        "Offline metrics: 292 recorded MCTS decisions from 30 validation target groups.  "
        "Discovery: 30 held-out targets × 6 seeds.",
        ha="center",
        fontsize=9.5,
        color="#56636f",
    )
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.18, top=0.80, wspace=0.34)
    save(fig, "policy_regression_overview.png")


def benchmark_figure() -> None:
    local = load_json(LOCAL_PATH)
    aws = load_json(AWS_PATH)
    comparison = load_json(CROSS_HOST_PATH)
    local_agg = local["aggregates"]
    aws_agg = aws["aggregates"]
    providers = ("exhaustive", "bounded_128_integrated")

    require(local.get("paired_outcomes") == {"both_fail": 16, "both_succeed": 134}, "unexpected local paired outcomes")
    require(aws.get("paired_outcomes") == {"both_fail": 16, "both_succeed": 134}, "unexpected AWS paired outcomes")
    require(comparison.get("discovery_mismatches") == [], "cross-host comparison contains discovery mismatches")
    require(comparison.get("timing_pooled") is False, "cross-host timing must remain unpooled")
    for aggregate in (local_agg, aws_agg):
        for provider in providers:
            require(aggregate[provider]["successes"] == 134, f"unexpected successes for {provider}")
            require(aggregate[provider]["trials"] == 150, f"unexpected trial count for {provider}")

    hosts = ["Local AMD CPU", "AWS Intel host CPU\n(trn1.2xlarge)"]
    exhaustive = [local_agg[providers[0]]["mean_search_seconds"], aws_agg[providers[0]]["mean_search_seconds"]]
    bounded = [local_agg[providers[1]]["mean_search_seconds"], aws_agg[providers[1]]["mean_search_seconds"]]
    speedups = [a / b for a, b in zip(exhaustive, bounded, strict=True)]
    x = [0, 1]
    width = 0.34

    fig, ax = plt.subplots(figsize=(10.8, 6.1))
    bars_a = ax.bar([v - width / 2 for v in x], exhaustive, width, label="Exhaustive", color=BLUE)
    bars_b = ax.bar([v + width / 2 for v in x], bounded, width, label="bounded_128", color=ORANGE)
    ax.set_title("bounded_128 reduced end-to-end search latency\nwith no observed discovery change on the tested corpus", fontsize=16, pad=16)
    ax.set_ylabel("Mean search latency (seconds)")
    ax.set_xticks(x, hosts)
    ax.set_ylim(0, max(exhaustive) * 1.28)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left")
    add_bar_labels(ax, bars_a, lambda v: f"{v:.4f} s", 0.015)
    add_bar_labels(ax, bars_b, lambda v: f"{v:.4f} s", 0.015)
    for index, speedup in enumerate(speedups):
        ax.text(index, max(exhaustive[index], bounded[index]) * 1.13, f"{speedup:.2f}× faster", ha="center", color=GREEN, fontweight="bold", fontsize=12)
        ax.text(index, 0.05, "134/150 successes each", ha="center", va="bottom", fontsize=10, color=DARK, bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": GRID})
    fig.text(
        0.5,
        0.018,
        "Same frozen 150-target corpus and seed on each host. Timings are host-specific and are not pooled.\n"
        "AWS measurements used the Intel host CPU, not Trainium. No provider discovery mismatch was observed.",
        ha="center",
        fontsize=9.5,
        color="#56636f",
    )
    fig.subplots_adjust(left=0.11, right=0.97, bottom=0.20, top=0.80)
    save(fig, "bounded_vs_exhaustive_benchmark.png")


def memory_figure() -> None:
    data = load_json(MEMORY_PATH)
    original = data["aggregates"]["original"]["mean_python_peak_bytes"] / 1_000_000
    optimized = data["aggregates"]["optimized"]["mean_python_peak_bytes"] / 1_000_000
    ratio = original / optimized
    require(data.get("differential_states") == 49, "memory study must contain 49 differential states")
    require(data.get("all_candidate_outputs_exact") is True, "candidate outputs did not match")
    require(data.get("all_search_trajectories_equivalent") is True, "search trajectories did not match")

    fig, ax = plt.subplots(figsize=(8.7, 5.5))
    bars = ax.bar(["Original bounded selector", "Streaming-heap selector"], [original, optimized], color=[BLUE, GREEN], width=0.58)
    ax.set_title("Streaming-heap selection reduced traced peak Python memory", fontsize=16, pad=16)
    ax.set_ylabel("Mean traced peak memory (MB)")
    ax.set_ylim(0, original * 1.23)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    add_bar_labels(ax, bars, lambda v: f"{v:.3f} MB" if v < 0.1 else f"{v:.2f} MB", 0.02)
    ax.text(0.5, original * 0.72, f"{ratio:.2f}× lower", ha="center", fontsize=14, color=GREEN, fontweight="bold")
    fig.text(
        0.5,
        0.025,
        "Python tracemalloc measurement of the selector path, not total process or system memory.\n"
        "Selected pairs and final candidates matched on all 49 differential states.",
        ha="center",
        fontsize=9.5,
        color="#56636f",
    )
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.20, top=0.82)
    save(fig, "streaming_heap_memory.png")


def pipeline_figure() -> None:
    fig, ax = plt.subplots(figsize=(12.4, 5.6))
    ax.set_xlim(0, 12.4)
    ax.set_ylim(0, 5.6)
    ax.axis("off")
    ax.set_title("Candidate-provider pipeline (schematic)", fontsize=16, fontweight="bold", color=DARK, pad=12)

    def box(x: float, y: float, w: float, h: float, text: str, color: str, fontsize: float = 10.5) -> None:
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.08", facecolor=color, edgecolor=DARK, linewidth=1.2)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color=DARK, fontweight="bold")

    def arrow(x1: float, y1: float, x2: float, y2: float) -> None:
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, linewidth=1.5, color="#65727e"))

    box(0.25, 2.15, 1.7, 1.0, "Residual\ntensor", "#eaf1f6")
    box(2.45, 1.95, 2.25, 1.4, "Score all 6,400\n(u, v) contractions", "#d8e8f2")
    arrow(1.95, 2.65, 2.45, 2.65)

    ax.text(5.25, 4.75, "Exhaustive path", color=BLUE, fontweight="bold", fontsize=12)
    box(5.15, 3.55, 2.5, 0.95, "Downstream processing\nfor all 6,400 pairs", "#d8e8f2")
    box(8.25, 3.55, 1.9, 0.95, "Exact ranking", "#eaf1f6")
    box(10.65, 3.55, 1.45, 0.95, "Top 12\nproposals", "#e3f0eb")
    arrow(4.70, 2.90, 5.15, 4.02)
    arrow(7.65, 4.02, 8.25, 4.02)
    arrow(10.15, 4.02, 10.65, 4.02)

    ax.text(5.25, 1.55, "bounded_128 path", color=ORANGE, fontweight="bold", fontsize=12)
    box(5.15, 0.35, 2.0, 0.95, "Keep top 128\ncoarse pairs", "#f8e4d8")
    box(7.65, 0.35, 2.5, 0.95, "Downstream processing\nfor retained 128", "#f8e4d8")
    box(10.65, 0.35, 1.45, 0.95, "Top 12\nproposals", "#e3f0eb")
    arrow(4.70, 2.40, 5.15, 0.82)
    arrow(7.15, 0.82, 7.65, 0.82)
    arrow(10.15, 0.82, 10.65, 0.82)

    fig.text(
        0.5,
        0.02,
        "Schematic only. bounded_128 still computes every contraction; it reduces later discrete w selection,\n"
        "candidate construction, canonicalization, and exact scoring. Fixed-budget retention is heuristic.",
        ha="center",
        fontsize=9.5,
        color="#56636f",
    )
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.14, top=0.88)
    save(fig, "tensor_game_candidate_pipeline.png")


def main() -> None:
    style()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    policy_regression_figure()
    benchmark_figure()
    memory_figure()
    pipeline_figure()
    for path in sorted(OUTPUT.glob("*.png")):
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
