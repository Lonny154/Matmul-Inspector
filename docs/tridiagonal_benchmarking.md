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

`summary.csv` has one row for each algorithm, backend, timing scope, system size,
and system count. The legacy-compatible `batch_size` column is that system count;
it does **not** mean simultaneous batched GPU execution. The `batch_execution`
column distinguishes `serial_host_loop` from `true_batched_gpu`. The CSV also records
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
