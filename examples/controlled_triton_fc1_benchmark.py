#!/usr/bin/env python3
"""Run a controlled five-candidate FC1 comparison against PyTorch/cuBLAS."""

from pathlib import Path

from matmul_inspector.controlled_benchmark import (
    BASELINE_ID, ControlledBenchmarkConfig, ControlledBenchmarkRunner,
    CudaControlledBenchmarkBackend, write_controlled_benchmark,
)
from matmul_inspector.sampling import (
    SearchContext, TritonGemmSearchSpace, UniformSampler, make_sampling_records,
)


space = TritonGemmSearchSpace(
    {"block_m": [32, 64], "block_n": [64, 128], "block_k": [32, 64],
     "num_warps": [4, 8], "num_stages": [2, 3]},
    SearchContext(512, 3072, 768, "fp16", hardware_target="sm_89"),
    identifier="fc1-controlled-sm89-v1",
)
sampler = UniformSampler(space, seed=42)
sampled = make_sampling_records(sampler.sample(5, unique=True), sampler)
config = ControlledBenchmarkConfig(
    scheduling_seed=17, input_seed=0, rounds=10, samples_per_round=3,
    launches_per_sample=100, warmups=100, atol=1e-2, rtol=1e-2,
)
result = ControlledBenchmarkRunner(CudaControlledBenchmarkBackend()).run(sampled, config)
output = write_controlled_benchmark(Path("controlled_triton_fc1"), result)
manifest = result["manifest"]
baseline = manifest["implementations"][BASELINE_ID]["summary"]["median_us"]
print(f"PyTorch/cuBLAS median CUDA-event latency: {baseline:.3f} us")
successful = []
for candidate in sampled:
    row = manifest["implementations"].get(candidate.candidate_id)
    if row and row["status"] == "completed":
        median = row["summary"]["median_us"]
        speedup = row["comparison_to_baseline"]["speedup_baseline_over_candidate"]
        successful.append((median, candidate))
        print(f"{candidate.parameters.to_dict()}: {median:.3f} us; "
              f"paired-session speedup vs baseline={speedup:.3f}x")
    else:
        failure = next(item for item in manifest["failures"]
                       if item["candidate_id"] == candidate.candidate_id)
        print(f"{candidate.parameters.to_dict()}: failed; {failure['description']}")
if successful:
    median, fastest = min(successful, key=lambda item: item[0])
    print(f"Fastest measured candidate in this sample: {fastest.parameters.to_dict()} "
          f"at {median:.3f} us median CUDA-event latency. No global optimum is claimed.")
print(f"Reports written to {output}")
