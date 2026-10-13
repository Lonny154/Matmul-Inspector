#!/usr/bin/env python3
"""Evaluate five sampled FC1 candidates on an SM 8.9 NVIDIA GPU."""

from pathlib import Path

from matmul_inspector.evaluation import EvaluationStatus, write_evaluation_jsonl
from matmul_inspector.sampling import (
    SearchContext, TritonGemmSearchSpace, UniformSampler, make_sampling_records,
)
from matmul_inspector.triton_evaluator import TritonEvaluatorConfig, TritonGemmEvaluator


space = TritonGemmSearchSpace(
    {
        "block_m": [32, 64], "block_n": [64, 128], "block_k": [32, 64],
        "num_warps": [4, 8], "num_stages": [2, 3],
    },
    SearchContext(512, 3072, 768, "fp16", hardware_target="sm_89"),
    identifier="fc1-sm89-exploratory-v1",
)
sampler = UniformSampler(space, seed=42)
sampled = make_sampling_records(sampler.sample(5, unique=True), sampler)
evaluator = TritonGemmEvaluator(TritonEvaluatorConfig(
    seed=0, atol=1e-2, rtol=1e-2, warmups=100,
    iterations_per_sample=500, samples=10,
))
evaluations = [evaluator.evaluate(candidate) for candidate in sampled]

output = Path("triton_fc1_evaluations.jsonl")
write_evaluation_jsonl(output, evaluations)
successful = []
for sampled_candidate, result in zip(sampled, evaluations):
    parameters = sampled_candidate.parameters.to_dict()
    if result.status is EvaluationStatus.COMPLETED:
        latency = result.benchmark.latency
        successful.append((latency, result))
        print(f"{parameters}: completed; median CUDA-event latency={latency:.3f} us")
    else:
        print(f"{parameters}: failed; {result.failure.category.value}: "
              f"{result.failure.description}")

if successful:
    latency, fastest = min(successful, key=lambda item: item[0])
    print("Fastest correct candidate in this five-candidate exploratory sample: "
          f"{fastest.extensions['candidate_parameters']} at median CUDA-event "
          f"latency {latency:.3f} us. This is not a global optimum claim.")
else:
    print("No candidate completed correctness validation and benchmarking.")
print(f"Wrote {len(evaluations)} evaluation records to {output}")
