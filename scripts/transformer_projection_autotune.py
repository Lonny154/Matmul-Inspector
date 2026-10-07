#!/usr/bin/env python3
"""Controlled tile search for the QKV and FC2 Transformer GEMMs."""

import argparse
import csv
from dataclasses import asdict, dataclass
import itertools
import json
import math
from pathlib import Path
import statistics

import torch
import torch.nn.functional as F
import triton

from transformer_workload import benchmark_callable_once
from triton_fc1 import TritonGemmConfig, triton_gemm


@dataclass(frozen=True)
class ProjectionSpec:
    name: str
    m: int
    k: int
    n: int


PROJECTIONS = {
    "qkv": ProjectionSpec("qkv", 512, 768, 2304),
    "fc2": ProjectionSpec("fc2", 512, 3072, 768),
}


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def nonnegative_int(value):
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def parse_int_list(value):
    try:
        result = tuple(positive_int(item.strip()) for item in value.split(","))
    except (ValueError, argparse.ArgumentTypeError) as error:
        raise argparse.ArgumentTypeError("expected comma-separated positive integers") from error
    if not result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("values must be nonempty and unique")
    return result


def parse_operations(value):
    operations = tuple(item.strip() for item in value.split(",") if item.strip())
    if not operations or len(set(operations)) != len(operations):
        raise argparse.ArgumentTypeError("operations must be nonempty and unique")
    invalid = set(operations) - set(PROJECTIONS)
    if invalid:
        raise argparse.ArgumentTypeError(
            "supported operations are qkv and fc2; FC1 is intentionally excluded"
        )
    return operations


def parse_args(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operations", type=parse_operations, default=("qkv", "fc2"))
    parser.add_argument("--block-m", type=parse_int_list, default=(32, 64, 128))
    parser.add_argument("--block-n", type=parse_int_list, default=(64, 128, 256))
    parser.add_argument("--block-k", type=parse_int_list, default=(32, 64))
    parser.add_argument("--warps", type=parse_int_list, default=(4, 8))
    parser.add_argument("--stages", type=parse_int_list, default=(2, 3))
    parser.add_argument("--search-warmup", type=nonnegative_int, default=5)
    parser.add_argument("--search-iterations", type=positive_int, default=50)
    parser.add_argument("--search-repeats", type=positive_int, default=3)
    parser.add_argument("--verify-top", type=positive_int, default=3)
    parser.add_argument("--verify-warmup", type=nonnegative_int, default=20)
    parser.add_argument("--verify-iterations", type=positive_int, default=200)
    parser.add_argument("--verify-repeats", type=positive_int, default=7)
    parser.add_argument("--minimum-improvement", type=float, default=0.02)
    parser.add_argument("--seed", type=nonnegative_int, default=42)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--output", type=Path, default=Path("results/transformer-autotune"))
    args = parser.parse_args(arguments)
    if args.minimum_improvement < 0:
        parser.error("--minimum-improvement must be nonnegative")
    return args


def candidate_configs(args):
    return (
        TritonGemmConfig(
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            num_warps=warps,
            num_stages=stages,
            group_size_m=1,
        )
        for block_m, block_n, block_k, warps, stages in itertools.product(
            args.block_m,
            args.block_n,
            args.block_k,
            args.warps,
            args.stages,
        )
    )


def make_inputs(spec, seed):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    a = torch.randn(
        spec.m,
        spec.k,
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    )
    if spec.name == "fc2":
        # FC2 consumes the output of GELU in the real workload.
        a = F.gelu(a)
    weight = torch.empty(
        spec.n,
        spec.k,
        device="cuda",
        dtype=torch.float16,
    )
    # Match torch.nn.Linear.reset_parameters rather than using unit-scale
    # weights, which creates an unrepresentative high-error FC2 fixture.
    bound = 1.0 / math.sqrt(spec.k)
    weight.uniform_(-bound, bound, generator=generator)
    return a, weight, weight.T.contiguous()


def alternating_samples(baseline, candidate, warmup, iterations, repeats):
    samples = {"pytorch": [], "triton": []}
    functions = {"pytorch": baseline, "triton": candidate}
    for repeat in range(repeats):
        order = ("pytorch", "triton") if repeat % 2 == 0 else ("triton", "pytorch")
        for implementation in order:
            samples[implementation].append(
                benchmark_callable_once(functions[implementation], warmup, iterations)
            )
    return samples


def summarize_samples(samples, minimum_improvement):
    ratios = [
        baseline / candidate
        for baseline, candidate in zip(samples["pytorch"], samples["triton"])
    ]
    median_ratio = statistics.median(ratios)
    return {
        "pytorch_median_ms": statistics.median(samples["pytorch"]),
        "triton_median_ms": statistics.median(samples["triton"]),
        "median_ratio": median_ratio,
        "minimum_paired_ratio": min(ratios),
        "maximum_paired_ratio": max(ratios),
        "repeatable_improvement": (
            min(ratios) > 1.0 and median_ratio >= 1.0 + minimum_improvement
        ),
    }


def evaluate_candidate(
    spec,
    config,
    a,
    weight,
    weight_t,
    warmup,
    iterations,
    repeats,
    atol,
    rtol,
    minimum_improvement,
    phase,
):
    row = {
        "operation": spec.name,
        "M": spec.m,
        "K": spec.k,
        "N": spec.n,
        "phase": phase,
        **asdict(config),
        "correct": False,
        "max_abs_error": "",
        "mean_abs_error": "",
        "warmup": warmup,
        "iterations": iterations,
        "repeats": repeats,
        "pytorch_median_ms": "",
        "triton_median_ms": "",
        "median_ratio": "",
        "minimum_paired_ratio": "",
        "maximum_paired_ratio": "",
        "repeatable_improvement": False,
        "error": "",
    }
    try:
        reference = F.linear(a, weight)
        candidate = triton_gemm(a, weight_t, config)
        error = (candidate - reference).abs().float()
        row["max_abs_error"] = error.max().item()
        row["mean_abs_error"] = error.mean().item()
        row["correct"] = bool(torch.allclose(candidate, reference, atol=atol, rtol=rtol))
        if not row["correct"]:
            return row

        samples = alternating_samples(
            lambda: F.linear(a, weight),
            lambda: triton_gemm(a, weight_t, config),
            warmup,
            iterations,
            repeats,
        )
        row.update(summarize_samples(samples, minimum_improvement))
    except (RuntimeError, ValueError) as error:
        row["error"] = str(error).replace("\n", " ")
        torch.cuda.empty_cache()
    return row


def config_from_row(row):
    return TritonGemmConfig(
        block_m=row["block_m"],
        block_n=row["block_n"],
        block_k=row["block_k"],
        num_warps=row["num_warps"],
        num_stages=row["num_stages"],
        group_size_m=row["group_size_m"],
    )


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    all_rows = []
    selections = {}
    for operation_index, operation in enumerate(args.operations):
        spec = PROJECTIONS[operation]
        a, weight, weight_t = make_inputs(spec, args.seed + operation_index)
        search_rows = []
        configs = list(candidate_configs(args))
        print(f"Searching {operation} ({spec.m}x{spec.k}x{spec.n}), {len(configs)} configs")
        for index, config in enumerate(configs, start=1):
            row = evaluate_candidate(
                spec,
                config,
                a,
                weight,
                weight_t,
                args.search_warmup,
                args.search_iterations,
                args.search_repeats,
                args.atol,
                args.rtol,
                args.minimum_improvement,
                "search",
            )
            search_rows.append(row)
            ratio = row["median_ratio"]
            ratio_text = f"{ratio:.4f}x" if isinstance(ratio, float) else "unavailable"
            print(f"  [{index:02}/{len(configs)}] {config} ratio={ratio_text}")
        all_rows.extend(search_rows)

        valid = [row for row in search_rows if row["correct"] and row["median_ratio"] != ""]
        valid.sort(key=lambda row: row["median_ratio"], reverse=True)
        verification_rows = []
        for row in valid[: args.verify_top]:
            config = config_from_row(row)
            verification_rows.append(
                evaluate_candidate(
                    spec,
                    config,
                    a,
                    weight,
                    weight_t,
                    args.verify_warmup,
                    args.verify_iterations,
                    args.verify_repeats,
                    args.atol,
                    args.rtol,
                    args.minimum_improvement,
                    "verification",
                )
            )
        all_rows.extend(verification_rows)

        passing = [row for row in verification_rows if row["repeatable_improvement"]]
        passing.sort(key=lambda row: row["median_ratio"], reverse=True)
        if passing:
            selected = passing[0]
            selections[operation] = {
                "selected": True,
                "config": asdict(config_from_row(selected)),
                "verification": {
                    key: selected[key]
                    for key in (
                        "pytorch_median_ms",
                        "triton_median_ms",
                        "median_ratio",
                        "minimum_paired_ratio",
                        "maximum_paired_ratio",
                    )
                },
            }
            print(f"Selected {operation}: {selections[operation]}")
        else:
            selections[operation] = {
                "selected": False,
                "reason": "no verified candidate met the repeatability rule",
            }
            print(f"No {operation} configuration met the repeatability rule")

    with (args.output / "autotune.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    summary = {
        "seed": args.seed,
        "environment": {
            "gpu": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability()),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "triton_version": triton.__version__,
        },
        "search_space": {
            "block_m": list(args.block_m),
            "block_n": list(args.block_n),
            "block_k": list(args.block_k),
            "warps": list(args.warps),
            "stages": list(args.stages),
            "group_size_m": 1,
        },
        "acceptance_rule": {
            "all_paired_ratios_above": 1.0,
            "minimum_median_ratio": 1.0 + args.minimum_improvement,
        },
        "search": {
            "warmup": args.search_warmup,
            "iterations": args.search_iterations,
            "repeats": args.search_repeats,
        },
        "verification": {
            "top_candidates": args.verify_top,
            "warmup": args.verify_warmup,
            "iterations": args.verify_iterations,
            "repeats": args.verify_repeats,
        },
        "selections": selections,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return selections


def main(arguments=None):
    args = parse_args(arguments)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
