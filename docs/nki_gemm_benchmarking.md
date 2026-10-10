# NKI GEMM benchmark harness

`scripts/nki_gemm_benchmark.py` compares the existing kernels in:

- `scripts/nki_matmul_full_tiled.py`: loads and transposes each A tile in the kernel.
- `scripts/nki_matmul_pretransposed.py`: accepts a contiguous `A.T` prepared by the caller.

The harness imports these kernels directly; it does not copy or alter their math.
Both currently require `M` and `K` to be divisible by 128 and `N` by 512.

Run explicit shapes in an AWS Neuron/NKI environment:

```bash
python3 scripts/nki_gemm_benchmark.py \
  --shapes 128x128x512,256x512x1024 \
  --warmups 1 --iterations 3 \
  --output results/nki-gemm
```

Or form a Cartesian product:

```bash
python3 scripts/nki_gemm_benchmark.py --cartesian \
  --m 128,256 --k 128,256 --n 512,1024 \
  --output results/nki-gemm-grid
```

Inputs come from an explicit NumPy generator seed. Both variants receive the
same logical A and B for a shape. The pretransposed layout is prepared outside
the timed region, as are input generation, NumPy reference GEMM, and numerical
analysis. Correctness uses `numpy.allclose` with configurable `--atol` and
`--rtol`; the CSV also records maximum and mean absolute error.

The current backend uses `nki.simulate`. Its timing is host wall-clock simulator
latency and is labeled `simulator_wall_clock`; it must not be interpreted as
Trainium kernel performance. `ExecutionBackend` separates orchestration and the
result schema from execution. A future Trainium backend can provide device timing
and profiler integration through that interface while preserving inputs,
correctness checks, CSV rows, and timing scopes.

Each new output directory contains:

- `summary.csv`: variant, M/K/N, seed, backend, correctness, errors, tolerances,
  warmups/iterations, sample count, mean/median/min/max/population-standard-
  deviation timing, timing mode, and timing note.
- `metadata.json`: configuration, backend identity, timing semantics, and kernel
  source paths.

The harness refuses to overwrite an existing output directory. Normal CPU-only
CTest validates parsing, statistics, identical variant inputs, correctness/error
metrics, schema, metadata, and overwrite protection using a synthetic backend;
those tests do not claim NKI kernel execution.
