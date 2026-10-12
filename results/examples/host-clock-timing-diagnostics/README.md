# Host-clock timing diagnostics

This evidence bundle preserves three bounded diagnostics that investigated a
CUDA-event versus synchronized host-wall timing discrepancy on one NVIDIA RTX
4060 Ti under WSL2. The scripts and measurements are examples. They are not a
general GPU performance result.

The tracked CSV files are byte-for-byte copies of the original raw measurement
files. The configurations and statistical summaries are also preserved. The
provenance files retain the relevant software, hardware, clock, and source
revision details while omitting transient utilization, process listings,
worktree state, absolute paths, and host-identifying `/proc/version` text.

## What was measured

| Directory | Diagnostic | Raw blocks |
| --- | --- | ---: |
| [`event-wall-scaling/`](event-wall-scaling/) | CUDA events versus an enclosing `perf_counter` interval at 50, 500, and 5,000 launches | 36 |
| [`cross-clock/`](cross-clock/) | CUDA events, `perf_counter`, and `CLOCK_MONOTONIC_RAW` at 5,000 launches | 12 |
| [`cpu-host-clocks/`](cpu-host-clocks/) | `perf_counter`, `CLOCK_MONOTONIC`, and `CLOCK_MONOTONIC_RAW` around fixed CPU work and sleep | 60 |

The two GPU diagnostics used the same preallocated FC1 workload and launch
configuration: M=512, K=768, N=3072, FP16, 64x128x32 tiles, four warps, three
stages, and `group_size_m=1`. Both checked the Triton output independently
against `torch.matmul` before timing. See each `config.json` for the exact
timestamp boundaries, warmups, scheduling, and tolerances.

## Findings

The scaling run found that the CUDA-event interval grew relative to the
`perf_counter` interval as block duration increased. Its descriptive fit was
`CUDA_ms = -0.1087 + 1.02410 * host_ms`. This is evidence of a predominantly
proportional disagreement in that run. The fit is descriptive only and must not
be used as a correction.

The cross-clock run localized the disagreement. Median block durations were
295.466 ms for `perf_counter`, 302.626 ms for `CLOCK_MONOTONIC_RAW`, and
302.518 ms for CUDA events. CUDA was a median 0.108 ms shorter than RAW across
the paired blocks. That systematic paired shortfall is compatible with the host
bracket including timestamp-read and synchronization-return overhead that CUDA
events do not include. It does not prove that explanation or absolute accuracy.

The CPU-only run reproduced a host-clock disagreement with no GPU work. Python
reported `perf_counter` as `clock_gettime(CLOCK_MONOTONIC)`, and those two
readings stayed aligned. RAW did not keep a fixed relationship to them:

| Phase | Blocks | Median RAW / perf | Ratio MAD | Median RAW - perf |
| --- | ---: | ---: | ---: | ---: |
| Early | 1--26 | 1.015557 | 0.000014 | +1.445 ms |
| Transition | 27 | 1.007572 | n/a | +3.232 ms |
| Late | 28--60 | 0.999383 | 0.000011 | -0.054 ms |

The relationship therefore changed during the run. Whole-run medians hide that
transition. Linux can adjust `CLOCK_MONOTONIC` incrementally while
`CLOCK_MONOTONIC_RAW` is not subject to those adjustments. That distinction is
relevant, but these measurements do not establish a root cause, select an
absolutely accurate clock, or justify a correction factor. Existing timing
results have not been rewritten or corrected.

## Correctness is separate

Both GPU diagnostics passed their configured `rtol=0.01`, `atol=0.01` check on
the fixed seeded input. Their recorded maximum absolute error was zero. That is
evidence for those executions only. It does not prove correctness for every
input. The CPU-only run measured clocks around CPU work and sleep and had no
numerical kernel correctness claim.

None of these timing comparisons demonstrates a kernel speedup. Timer
consistency must be resolved before using the affected host-clock measurements
for performance claims.

## Reproduction

Run from the repository root. The first two commands require a compatible
NVIDIA environment with CUDA-enabled PyTorch and Triton. They launch GPU work:

```console
python scripts/fc1_event_wall_scaling_diagnostic.py
python scripts/fc1_cross_clock_diagnostic.py
```

The host-clock diagnostic does no GPU work, but it checks `nvidia-smi` first so
it can avoid overlapping a benchmark:

```console
python scripts/host_clock_cpu_diagnostic.py
```

Each command creates a new timestamped directory beneath ignored `artifacts/`.
Do not compare absolute timestamps from different clocks. Inspect elapsed
durations, paired differences, timestamp order, and the recorded environment.
