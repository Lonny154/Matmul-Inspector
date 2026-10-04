# Tridiagonal solver baseline benchmark

The `tridiagonal-benchmark` executable compares the existing CPU Thomas, CPU PCR,
and naive CUDA PCR implementations. Its defaults cover system sizes
`32,64,128,256,512,1024,2048,4096` and batch sizes `1,8,32,128,512`.

```sh
cmake --build build -j
./build/tridiagonal-benchmark \
  --warmups 3 --iterations 20 \
  --output results/tridiagonal-baseline
```

For profiling, select one configuration and execution path:

```sh
./build/tridiagonal-benchmark \
  --system-size 4096 \
  --batch-size 512 \
  --mode true_batched_device_resident \
  --warmups 1 --iterations 1 \
  --output results/ncu-4096x512
```

`--system-size` and `--batch-size` accept any positive value, including
non-power-of-two system sizes. When omitted, the existing default grids remain
in effect. The legacy comma-separated `--sizes` and `--batches` options remain
available.

`--mode` defaults to `all`. Its values select these rows:

- `all`: every CPU and CUDA path.
- `cpu`: CPU Thomas and CPU PCR.
- `serial_host_loop`: serial multi-system CUDA kernel-only and end-to-end.
- `true_batched_gpu`: one-shot true-batched kernel-only and end-to-end.
- `true_batched_reuse`: persistent-workspace decomposition and reusable total.
- `true_batched_device_resident`: D2D reset and device-resident solve.
- `true_batched_hybrid`: global device-resident baseline plus the experimental
  shared/global hybrid row.

`summary.csv` has one row for each algorithm, backend, timing scope, system size,
and system count. The legacy-compatible `batch_size` column is that system count;
it does **not** mean simultaneous batched GPU execution. The `batch_execution`
column distinguishes `serial_host_loop`, one-shot `true_batched_gpu`,
`true_batched_reuse`, `true_batched_device_resident`, and
`true_batched_hybrid`. The CSV also records
warmups, iterations, median, mean, minimum, and maximum
latency in milliseconds. These statistics use the same `benchmark::Sample` and
`benchmark::analyze` implementation as the matrix workflows. Each latency is for
the complete requested batch, not one system. `methodology.txt` travels with the
CSV as a compact timing-scope description.

Before timing, the harness creates deterministic diagonally dominant systems from
known solutions. Every benchmark path validates its final solutions. Sizes need
not be powers of two; size 33 is included in automated benchmark integration tests.

## Timing scopes

- CPU `end_to_end` uses `steady_clock` around the full batch. It includes solver
  working/output allocation and computation.
- CUDA `kernel_only` allocates buffers and uploads coefficients before sampling.
  Before every sample it restores the input coefficient buffers outside the CUDA
  event because PCR mutates them. The event spans all existing PCR stage and solve
  kernel launches for the batch. Allocation, H2D resets, D2H result copies,
  validation, and cleanup are excluded.
- CUDA `end_to_end` includes host/device allocation, coefficient preparation, H2D
  copies, kernel execution and synchronization, D2H copies, and cleanup.
- CUDA `reusable_end_to_end` uses `steady_clock` around coefficient reset/H2D,
  kernel launches, synchronization, and result download. Its persistent workspace
  was allocated before the sample, so allocation and cleanup are excluded.

The `serial_host_loop` CUDA path is intentionally a serial host loop over independent
invocations of the current naive PCR kernels. Both `kernel_only` and `end_to_end`
measure all systems in that loop as one sample. This is a serial multi-system
baseline, not a true batched GPU implementation.

The `true_batched_gpu` path flattens the coefficients for `B` systems of size `N`
into arrays of `B*N` equations. A global thread index maps to
`system = global / N` and `equation = global % N`; left and right PCR neighbors
are restricted to that system's `N`-element range. Every stage launches once over
all `B*N` equations. Consequently, the stage-launch count changes from about
`B*ceil(log2(N))` for the serial host loop to `ceil(log2(N))` for the true-batched
path. Both retain the final solve launch.

This change exposes system-level parallelism in addition to equation-level
parallelism. It does not change PCR mathematics or add shared memory, warp
shuffles, fusion, cooperative groups, persistent kernels, mixed precision, or
another algorithm. Keeping both named paths provides a baseline for later PCR
optimization experiments.

## Persistent workspace and timing decomposition

The one-shot `true_batched_gpu` solver intentionally remains the baseline that
allocates and frees nine device buffers on every call. The `true_batched_reuse`
experiment constructs a `CudaPcrBatchedWorkspace` once and separates its reusable
operations into input upload/reset, execution of the unchanged PCR kernels, and
result download. Reusing storage tests host/device and CUDA runtime overhead; it
does not optimize the kernels or alter PCR arithmetic.

The reusable rows have these `timing_scope` values:

- `allocation_setup`: host `steady_clock` time to allocate the workspace's device
  buffers. Buffer destruction is outside this measurement because allocation is
  a one-time cost amortized across solves.
- `h2d`: CUDA-event time for copying the four flattened coefficient arrays into
  already allocated device buffers.
- `kernel_only`: CUDA-event time for all PCR stages and the final solve kernel.
- `d2h`: CUDA-event time for copying the flattened result into preallocated host
  storage.
- `reusable_end_to_end`: host `steady_clock` time for H2D reset, kernel execution,
  synchronization, and D2H copy using an existing workspace.

Conceptually, a one-shot solve contains allocation/setup + H2D + kernel + D2H
(plus host API and cleanup overhead), while a reusable solve contains H2D + kernel
+ D2H (plus host API overhead). Component means should therefore be treated as a
decomposition aid rather than expected to sum exactly to a host-clock total.
CUDA events measure work on the CUDA timeline and require synchronization of the
stop event; `steady_clock` measures host-visible latency. Pageable host memory,
runtime submission costs, synchronization, and measurement noise can make the
two clocks differ.

## Device-resident solve

`true_batched_device_resident` models a multi-kernel ML or HPC pipeline in which
the tridiagonal coefficients are already on the accelerator and the result is
consumed by later GPU work. The workspace keeps four immutable device copies of
the original flattened coefficients alongside the mutable PCR working buffers.
It uploads and snapshots inputs before timing, restores the mutable buffers with
device-to-device copies between solves, and downloads only after timing for
correctness verification.

Two rows keep the required reset work visible without mixing it into solve time:

- `d2d_reset` uses CUDA events around the four device-to-device coefficient
  copies from immutable sources into mutable working buffers.
- `device_resident` restores coefficients before its start event, then measures
  all PCR stage kernels and the final solve kernel through its stop event. The
  output remains resident until after the event has completed. Allocation, H2D,
  D2H, D2D reset, and correctness checks are excluded.

The existing `kernel_only` row also measures PCR kernels, but belongs to the
one-shot or reusable host-driven benchmark setup and retains its established
meaning. `reusable_end_to_end` includes H2D reset and D2H transfer using allocated
buffers. `device_resident` represents only the solve portion of an accelerator
pipeline. All three paths invoke the same kernels and use the same PCR arithmetic;
this experiment changes data residency and measurement boundaries only.

## Stage-aware profiling

Set `MATMUL_INSPECTOR_NVTX=1` to add NVTX push/pop ranges around the existing
true-batched kernel launches. Normal benchmark runs leave these ranges disabled,
so marker construction and NVTX calls do not affect their timing. If the CUDA
toolkit's `nvtx3/nvToolsExt.h` header is unavailable, the same source builds with
no-op ranges.

Stage labels have the deterministic form
`pcr_<global|shared>_stage_<stage-index>_offset_<offset>`. Stage index starts at
zero and offset is `2^stage-index`. The final solve kernel is labeled
`pcr_final_solve`. The global baseline uses `global` at every stage. For
`N=4096`, the logical mapping is:

| Stage | Offset | NVTX label |
|---:|---:|---|
| 0 | 1 | `pcr_global_stage_0_offset_1` |
| 1 | 2 | `pcr_global_stage_1_offset_2` |
| 2 | 4 | `pcr_global_stage_2_offset_4` |
| 3 | 8 | `pcr_global_stage_3_offset_8` |
| 4 | 16 | `pcr_global_stage_4_offset_16` |
| 5 | 32 | `pcr_global_stage_5_offset_32` |
| 6 | 64 | `pcr_global_stage_6_offset_64` |
| 7 | 128 | `pcr_global_stage_7_offset_128` |
| 8 | 256 | `pcr_global_stage_8_offset_256` |
| 9 | 512 | `pcr_global_stage_9_offset_512` |
| 10 | 1024 | `pcr_global_stage_10_offset_1024` |
| 11 | 2048 | `pcr_global_stage_11_offset_2048` |

Capture all ranges and their CUDA launches with Nsight Systems:

```sh
MATMUL_INSPECTOR_NVTX=1 nsys profile \
  --trace=cuda,nvtx \
  --output results/ncu-4096x512/timeline \
  ./build/tridiagonal-benchmark \
    --system-size 4096 --batch-size 512 \
    --mode true_batched_device_resident \
    --warmups 0 --iterations 1 \
    --output results/ncu-4096x512/run
```

Select one logical stage with Nsight Compute, for example the final PCR stage:

```sh
MATMUL_INSPECTOR_NVTX=1 ncu \
  --nvtx --nvtx-include pcr_global_stage_11_offset_2048 \
  --launch-count 1 \
  --set full \
  --export results/ncu-4096x512/stage-11 \
  ./build/tridiagonal-benchmark \
    --system-size 4096 --batch-size 512 \
    --mode true_batched_device_resident \
    --warmups 0 --iterations 1 \
    --output results/ncu-4096x512/ncu-run
```

The NVTX range encloses only the host-side launch and adds no synchronization,
copy, ordering change, or event-boundary change. With profiling enabled, the
profiler necessarily observes the small host marker cost; leave the environment
variable unset for benchmark measurements.

## Experimental shared-memory stages

Nsight Compute measurements on the global kernel at `N=4096`, `B=512` showed
offsets 1 through 256 near 90–92% DRAM utilization with substantial
L1TEX/long-scoreboard stalls. Offset 512 was the observed transition toward
higher compute utilization. The `true_batched_hybrid` experiment therefore uses
shared memory for offsets up to the explicit `pcr_shared_max_offset=256` cutoff
and retains the established global kernel for offsets 512 and larger.

For a 256-thread block and FP64 coefficients, each shared stage loads a contiguous
window of `256 + 2*offset` values for each of `a`, `b`, `c`, and `d`. Its dynamic
shared-memory requirement is `4 * (256 + 2*offset) * 8` bytes: 8,256 bytes at
offset 1, 10,240 at offset 32, 16,384 at offset 128, and 24,576 at offset 256.
All threads cooperatively load the center and halo and reach one barrier before
using the data. Per-equation bounds still decide whether left and right neighbors
belong to the same system, so a flattened CUDA block may cross system boundaries
without reading another system's coefficients. Partial blocks and non-power-of-two
systems use the same guarded loads. The offset-256 halo is safe but relatively
large, so its benefit was treated as an experimental question rather than assumed.

The hybrid path uses labels such as `pcr_shared_stage_5_offset_32` through
`pcr_shared_stage_8_offset_256`, followed by
`pcr_global_stage_9_offset_512`. Its `device_resident` timing boundary matches the
global-only device-resident baseline; D2D reset remains outside both measurements.

The following hardware-specific medians were measured on an NVIDIA GeForce RTX
4060 Ti with CUDA 13.3, using three warmups and 20 measured iterations. Speedup is
global time divided by hybrid time. All configurations passed comparison against
CPU PCR and the global CUDA implementation.

| N | B | Global ms | Hybrid ms | Speedup |
|---:|---:|---:|---:|---:|
| 128 | 32 | 0.120832 | 0.057856 | 2.088x |
| 128 | 128 | 0.065024 | 0.065024 | 1.000x |
| 128 | 512 | 0.100352 | 0.113040 | 0.888x |
| 256 | 32 | 0.061744 | 0.063488 | 0.973x |
| 256 | 128 | 0.086016 | 0.096256 | 0.894x |
| 256 | 512 | 0.191488 | 0.193024 | 0.992x |
| 512 | 32 | 0.077824 | 0.082640 | 0.942x |
| 512 | 128 | 0.128000 | 0.132608 | 0.965x |
| 512 | 512 | 0.380928 | 0.382976 | 0.995x |
| 1024 | 32 | 0.103424 | 0.112128 | 0.922x |
| 1024 | 128 | 0.246784 | 0.247296 | 0.998x |
| 1024 | 512 | 0.840720 | 0.844800 | 0.995x |
| 2048 | 32 | 0.202752 | 0.203776 | 0.995x |
| 2048 | 128 | 0.521216 | 0.521216 | 1.000x |
| 2048 | 512 | 2.956800 | 2.958448 | 0.999x |
| 4096 | 32 | 0.313856 | 0.311296 | 1.008x |
| 4096 | 128 | 1.060864 | 1.064960 | 0.996x |
| 4096 | 512 | 6.559312 | 6.698496 | 0.979x |

These results do not show a consistent benefit. Most larger configurations were
effectively tied or slightly slower, and the small configurations are sensitive
to launch and measurement noise. The shared-memory path remains an experimental
comparison while the global kernel remains the baseline.
