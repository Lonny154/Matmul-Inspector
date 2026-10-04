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

`summary.csv` has one row for each algorithm, backend, timing scope, system size,
and system count. The legacy-compatible `batch_size` column is that system count;
it does **not** mean simultaneous batched GPU execution. The `batch_execution`
column distinguishes `serial_host_loop`, one-shot `true_batched_gpu`, and
`true_batched_reuse`, and `true_batched_device_resident`. The CSV also records
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
`pcr_stage_<stage-index>_offset_<offset>`. Stage index starts at zero and offset
is `2^stage-index`. The final solve kernel is labeled `pcr_final_solve`. For
`N=4096`, the mapping is:

| Stage | Offset | NVTX label |
|---:|---:|---|
| 0 | 1 | `pcr_stage_0_offset_1` |
| 1 | 2 | `pcr_stage_1_offset_2` |
| 2 | 4 | `pcr_stage_2_offset_4` |
| 3 | 8 | `pcr_stage_3_offset_8` |
| 4 | 16 | `pcr_stage_4_offset_16` |
| 5 | 32 | `pcr_stage_5_offset_32` |
| 6 | 64 | `pcr_stage_6_offset_64` |
| 7 | 128 | `pcr_stage_7_offset_128` |
| 8 | 256 | `pcr_stage_8_offset_256` |
| 9 | 512 | `pcr_stage_9_offset_512` |
| 10 | 1024 | `pcr_stage_10_offset_1024` |
| 11 | 2048 | `pcr_stage_11_offset_2048` |

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
  --nvtx --nvtx-include pcr_stage_11_offset_2048 \
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
