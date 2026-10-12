# CUDA Thomas versus parallel cyclic reduction

## Research question

Under what tridiagonal system sizes and batch counts does an untuned CUDA PCR
implementation outperform a matched CUDA Thomas implementation on an NVIDIA RTX
4060 Ti?

The experiment tests two competing effects. Thomas performs linear work but each
thread has a sequential dependency chain. PCR performs more work and uses one
thread per equation, but reduces the dependency depth to logarithmic stages. A
large batch may give Thomas enough system-level parallelism to offset its serial
work. A small batch may give PCR an advantage by exposing parallelism within a
system. These are hypotheses until measured.

## Implementations

The experiment is separate from the repository's historical FP64 PCR benchmark.
It preserves those implementations and uses FP32 for both new GPU candidates.

- CPU Thomas runs in FP64 and is the numerical reference.
- CUDA Thomas maps one independent system to one CUDA thread. Each thread performs
  forward elimination and back substitution sequentially. Blocks contain 256
  threads, so the grid has `ceil(B / 256)` blocks.
- CUDA PCR flattens `B` systems of size `N`. One CUDA thread owns one equation.
  Each doubling offset launches over `B*N` equations, followed by a pointwise
  solve kernel. Blocks contain 256 threads. Neighbors are bounded to the current
  system.

The candidates are deliberately untuned baselines. There is no hybrid dispatch,
shared-memory specialization, fusion, or launch-configuration search in this
first comparison.

## Correctness methodology

Inputs are generated from a fixed seed. Off-diagonal coefficients are sampled
from a bounded distribution and each diagonal is set above the sum of the
off-diagonal magnitudes. This makes every system strictly diagonally dominant.
The right-hand side is derived from a known solution, rounded to FP32, then CPU
Thomas recomputes the FP64 reference for that rounded system.

Every candidate and configuration records maximum absolute error, maximum
relative error, and the maximum per-system normalized residual

```text
||Ax - d||_2 / (||A||_F ||x||_2 + ||d||_2).
```

The initial elementwise acceptance rule is `rtol=1e-4`, `atol=1e-5`. A failed
configuration is written to `failures.csv` and stops the run. Small systems can
be checked with `--sizes 1,2,3,4` before the required grid.

## Benchmark methodology

The required grid is `N=128,256,512,1024` and `B=1,32,1024`. Each candidate is
warmed up before measurement. Candidate order is shuffled with the case seed for
each paired iteration. CUDA events measure GPU work. A separate host clock measures launch
through stop-event synchronization. Allocation, upload, CPU reference generation,
result download, and correctness checks are excluded from kernel samples.

PCR mutates its working coefficient arrays. Its four device-to-device reset
copies are included in PCR's event interval because a repeated solve requires
restoring that state. Thomas initializes its working diagonal and right-hand side
inside its solve kernel. This is part of the algorithms' tested implementation
cost, not an isolated arithmetic-only comparison.

Raw samples are retained in `raw_samples.csv`. `summary.csv` reports median, p10,
p90, and solved systems per second. `correctness.csv`, `failures.csv`,
`environment.json`, and `methodology.md` preserve the remaining evidence.

## Reproduction

```sh
cmake -S . -B build-tridiagonal -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build-tridiagonal --parallel --target tridiagonal-comparison

# Edge-case validation
./build-tridiagonal/tridiagonal-comparison \
  --sizes 1,2,3,4 --batches 1,32 --warmups 2 --rounds 2 --iterations 5 \
  --output results/thomas-vs-pcr-edge

# Required experiment grid
./build-tridiagonal/tridiagonal-comparison \
  --sizes 128,256,512,1024 --batches 1,32,1024 \
  --warmups 5 --rounds 5 --iterations 20 --seed 20261012 \
  --output results/thomas-vs-pcr-rtx4060ti
```

## Results

The required grid ran on an NVIDIA GeForce RTX 4060 Ti (SM 8.9, 34 SMs) with
CUDA driver 13.4, CUDA runtime 13.3, and nvcc 13.3. The Release build targeted
SM 8.9. Each row contains 100 CUDA-event samples: five rounds with 20 paired,
randomly ordered iterations per round. Speedup is Thomas median divided by PCR
median. Values above one favor PCR.

| N | B | Thomas median (ms) | PCR median (ms) | PCR speedup | Winner |
|---:|---:|---:|---:|---:|---|
| 128 | 1 | 0.024416 | 0.176320 | 0.138x | Thomas |
| 128 | 32 | 0.059392 | 0.181760 | 0.327x | Thomas |
| 128 | 1024 | 0.223888 | 0.150112 | 1.491x | PCR |
| 256 | 1 | 0.045056 | 0.116368 | 0.387x | Thomas |
| 256 | 32 | 0.116736 | 0.202368 | 0.577x | Thomas |
| 256 | 1024 | 0.445024 | 0.214864 | 2.071x | PCR |
| 512 | 1 | 0.087040 | 0.144688 | 0.602x | Thomas |
| 512 | 32 | 0.215040 | 0.235520 | 0.913x | Thomas |
| 512 | 1024 | 0.815712 | 0.293376 | 2.780x | PCR |
| 1024 | 1 | 0.157696 | 0.204768 | 0.770x | Thomas |
| 1024 | 32 | 0.425984 | 0.212768 | 2.002x | PCR |
| 1024 | 1024 | 1.567744 | 0.658432 | 2.381x | PCR |

PCR was faster for every tested `B=1024` case. It also crossed over at
`N=1024, B=32`. Thomas was faster for all `B=1` cases and for `B=32` through
`N=512`. The tested grid does not locate an exact crossover between these
points. PCR's widest distributions occurred in several small and medium cases.
For example, `N=128, B=32` had p10 0.068573 ms and p90 0.222208 ms. The median
winner is clear there, but precise ratio claims should account for that spread.

All 24 required candidate/configuration checks passed. The largest absolute
error was `8.3914e-7`. The largest normalized residual was `9.4263e-9`. Maximum
relative error can look much larger near a reference value of zero, so pass/fail
uses the documented combined absolute and relative elementwise rule. Edge runs
at `N=1,2,3,4` and `B=1,32` also passed.

Saved evidence:

- `results/tridiagonal-thomas-pcr-20261012/raw_samples.csv`
- `results/tridiagonal-thomas-pcr-20261012/summary.csv`
- `results/tridiagonal-thomas-pcr-20261012/correctness.csv`
- `results/tridiagonal-thomas-pcr-20261012/failures.csv`
- `results/tridiagonal-thomas-pcr-20261012/environment.json`
- `artifacts/tridiagonal-thomas-pcr-20261012/edge/`
- `artifacts/tridiagonal-thomas-pcr-20261012/profiles/`

These generated directories are ignored by Git under the repository's artifact
policy. The metadata records base revision `ca4624784206`; the experiment source
was an uncommitted working-tree change on top of that revision.

## Hardware profiling

Nsight Compute `--set basic` captured one representative launch from each path at
`N=1024, B=1024`. These observations apply to those profiled launches only.

| Observation | CUDA Thomas | PCR stage at offset 1 |
|---|---:|---:|
| Grid blocks | 4 | 4,096 |
| Threads per block | 256 | 256 |
| Registers per thread | 40 | 27 |
| Static/dynamic shared memory | 0 / 0 bytes | 0 / 0 bytes |
| Achieved occupancy | 16.62% | 91.20% |
| DRAM throughput | 5.16% | 95.65% |
| Compute throughput | 0.54% | 18.93% |
| Profiled kernel duration | 1.69456 ms | 0.073472 ms |

Nsight reported that the four-block Thomas grid was too small to fill 34 SMs.
This directly supports underutilization as an explanation for the large-batch,
large-`N` Thomas result. The profiled early PCR stage had 20.08 waves per SM and
was DRAM-throughput limited. PCR repeats this kind of full-grid work for every
doubling offset and also pays reset and launch costs. Those costs are consistent
with PCR losing at small batches, but the profile does not prove a single causal
breakdown of the end-to-end event time.

## Limitations and next experiment

This study compares two specific untuned kernels and one input distribution on
one GPU. It does not establish the best possible Thomas or PCR implementation.
The PCR event includes device-to-device reset. The Thomas kernel initializes its
work arrays internally. Host-to-device and device-to-host transfers are excluded.
GPU clock and scheduling variability remain visible in the raw distributions.

The next experiment should densify batch counts between 32 and 1024 at each `N`
to locate crossover intervals. After that, test a smaller Thomas block size to
increase its block count. Preserve the current kernels and timing records as the
immutable baseline.

The next experiment should be chosen from measured evidence. Plausible options
include launch-configuration exploration, shared-memory PCR, or a hybrid method.
None should replace these immutable untuned baselines.
