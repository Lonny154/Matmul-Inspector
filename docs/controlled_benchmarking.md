# Controlled FC1 baseline comparisons

The controlled benchmark compares sampled Triton FC1 candidates with a matched
PyTorch baseline in one GPU session. It uses the established FP16 workload
`M=512, N=3072, K=768` on an SM 8.9 NVIDIA GPU.

```sh
python examples/controlled_triton_fc1_benchmark.py
```

The default example samples five candidates with seed 42. It uses input seed 0
and scheduling seed 17. Output is written to a new `controlled_triton_fc1/`
directory. Existing directories are never overwritten.

## Execution paths

All implementations receive the same contiguous FP16 input tensors and produce
the same FP16 output shape. Outputs are allocated once and reused. The baseline
operation is `torch.matmul(a, b, out=baseline_output)`, normally dispatched to
cuBLAS. Triton calls the packaged `fc1_kernel` directly with a preallocated
output. This avoids comparing Triton without allocation against allocating
`torch.matmul(a, b)`.

The operations are mathematically equivalent, but their implementation paths
are not identical. PyTorch dispatch and cuBLAS internals remain library-selected.
The Triton launch uses the candidate tile, warp, and stage parameters with
`GROUP_SIZE_M=1`. Results support a matched session comparison, not a claim that
wrapper overhead or library internals are identical.

Candidates compile and run once before correctness validation. Compilation,
input creation, reference creation, and correctness checks are outside timing.
Correctness uses `atol=1e-2`, `rtol=1e-2`, maximum and mean absolute error, and
shared inputs. Failed candidates are excluded from every timing round.

After warmup, each round independently shuffles the baseline and correct
candidates using the scheduling seed. Each implementation's timing block is
synchronized before the next begins, so measurement windows do not overlap.
Defaults are 10 rounds, three samples per implementation per round, and 100
launches per sample. CUDA-event time, host submission time, and synchronized
host-wall time remain separate.

## Statistics and artifacts

For every implementation the manifest records raw per-round observations,
per-round medians, overall median, mean, minimum, maximum, sample standard
deviation, and observation count. Candidate comparisons use measurements from
the same rounds. They include the ratio of baseline median to candidate median,
percentage latency reduction, paired round speedups, and variability of those
paired ratios. No confidence interval or significance test is claimed.

The version-1 experiment directory contains:

- `experiment.json`: manifest, environment, schedule, raw measurements, and summaries.
- `evaluations.jsonl`: candidate lifecycle, correctness, and final benchmark records.
- `report.md`: concise human-readable comparison.

The manifest links evaluation IDs to candidate IDs and records detected device,
candidate configurations, sampling provenance, PyTorch, Triton, and CUDA runtime
versions. Raw timing data lives only in the manifest to avoid duplication.
Evaluation records contain their candidate summary and matched-baseline
comparison.

GPU clocks, thermals, other processes, power state, launch scheduling, and
library algorithm selection remain potential confounders. A small measured
difference is descriptive unless supported by adequate repeated evidence. The
fastest row is only the fastest successfully measured candidate in that run.
This milestone does not perform adaptive sampling or update sampler weights.
