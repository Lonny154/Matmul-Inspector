# Supported operations

Matmul-Inspector supports `matmul` (the default), `dot`, and `reduction_sum`.
Existing matmul commands, kernels, crossover experiments and input fixtures are
unchanged. Vector operations support `compare` and `benchmark`, positive lengths,
and the existing deterministic `random` generator. `--size` selects one length;
`--sizes` selects several. Empty vectors and matrix dimension flags are rejected.

```bash
# Existing matmul workflow
build/matmul-inspector compare --sizes 4,257

# Inspect scalar CPU vs GPU diagnostics
build/matmul-inspector compare --operation dot --sizes 3,257 --seed 42

# Short smoke captures, not statistically strong performance evidence
build/matmul-inspector benchmark --operation dot --size 1000000 \
  --seed 42 --warmups 2 --iterations 5 --trials 2 --bootstrap-samples 100 \
  --atol 0.01 --rtol 0.001 --save-output --output results/dot-smoke
build/matmul-inspector benchmark --operation reduction_sum --size 1000003 \
  --seed 42 --warmups 2 --iterations 5 --trials 2 --bootstrap-samples 100 \
  --atol 0.01 --rtol 0.001 --save-output --output results/sum-smoke
python3 scripts/report.py results/dot-smoke
python3 scripts/report.py results/sum-smoke
```

The example tolerances are explicit exploratory choices, not error guarantees.
Default tolerances remain `atol=1e-6`, `rtol=1e-5`. Large or cancelling reductions
can fail them. Failures remain visible in artifacts and produce exit code 1;
the tool never widens tolerances automatically. A missing GPU returns 77. For a
CPU-only smoke run use `--reference cpu --candidate cpu`.

## Execution and numerical meaning

`operation::Kind` and the functions in `include/operation.hpp` / `src/operation.cpp`
provide identity, serial execution, nominal FLOPs, compulsory bytes and stage
counts. No plugin system or class hierarchy is involved. The existing `Matrix`
container carries vector input A as 1×length and dot input B as length×1 (strides
are respected). A scalar is a 1×1 result, so comparison, hashing and native binary
comparison remain unchanged. Sum allocates no B input in the CLI.

- **Dot:** `sum_i A[i]*B[i]`. CPU `cpu` visits increasing indices with a single
  FP32 accumulator; contraction follows compiler flags. CUDA `cuda-tree`
  computes one rounded FP32 product per lane, then adds it through a tree.
- **Sum:** `sum_i A[i]`. CPU visits increasing indices from positive zero. CUDA
  loads one input per lane and uses the same tree, without multiplication.
- **CUDA tree:** 256 threads per block; shared-memory offsets 128,64,…,1;
  left partial plus right partial using explicit round-to-nearest FP32 addition.
  All lanes reach each barrier. Out-of-range lanes load positive zero. Each block
  writes one partial. Separate kernel launches repeatedly reduce those partials
  until one remains, ping-ponging between buffers. No atomics, races, mixed
  precision or nondeterministic scheduling-dependent association are used.
  Lengths 1–256 require one stage, 257–65536 two, and 1000003 three.

FP32 addition is not associative. Serial and tree outputs need not match bits;
bitwise divergence and numerical tolerance failure remain separate. CPU dot
contraction can also differ from the explicit separate CUDA multiply/add. This
is **not** an experiment isolating only contraction or only association; existing
controlled matmul variants still serve that purpose. Deterministic association
on one build is not a promise of universal cross-toolchain bitwise equality.

Scalar diagnostics reuse `comparison` and `numeric`: IEEE bits, ULPs, absolute
and relative errors, tolerance, NaN/infinity conventions and aggregate bins.
For one output divergence is either 0% or 100%. Signed zero retains distinct
bits but has zero ULP distance. The legacy relative-error diagnostic is zero for
a zero reference; absolute error and the symmetric tolerance test remain the
meaningful checks there. Nonfinite pairs are separately counted, not assigned
misleading finite aggregate errors.

## Timing and reports

CPU `host_serial` uses `steady_clock` around serial execution, including scalar
output allocation. CUDA `kernel_only` uses an event pair around **all stages**,
with a synchronization for each sample. Inter-stage launches are included;
allocation, input/output transfers, priming and analysis are excluded. Each
trial has one untimed GPU priming call plus configured timed warmups; warmups
are saved but excluded from measured statistics. All reference trials precede
candidate trials. Existing mean/median/min/stddev, percentiles, MAD/IQR/CV,
bootstrap intervals and stability warnings are reused without a second
statistics implementation.

CPU/GPU latency ratios in these reports compare CPU host execution to GPU
kernel-only work; they are not end-to-end offload/crossover conclusions. Report
plots use vector length, latency, nominal GFLOP/s, reference/candidate ratio,
scalar divergence and existing reliability distributions. Implementation and
reduction metadata plus numerical errors appear in the Markdown report.

## Additive artifact and replay extension

Result schema remains 3; vector runs carry `operation_extension: "1"` and
`config.operation`. Every summary, mismatch and timing CSV includes `operation`.
Summary adds `length`, `flop_count`, `model_bytes`, `reduction_block_size` and
`reduction_stages`. Existing contraction/accumulation fields identify
`serial_increasing_index` and `block_tree_256_multistage`. Configuration records
the fixed block size for replay. A run contains one operation and any number
of its supported sizes. For vectors the compatibility shape is **M=N=1,
K=length**: M/N describe the scalar result, not fictitious matrix operands.

`--save-output` preserves the existing `MIFP32LE` binary encoding: 24-byte header
with rows=cols=1, then exactly four little-endian FP32 bytes. The SHA-256 covers
those four payload bytes, retaining signed zero and NaN payloads. The adjacent
JSON includes operation, shape, seeds and implementation semantics. Hash equality
alone is meaningful only after checking compatible inputs/operation/shape.

```bash
python3 scripts/reproduce.py results/dot-smoke/metadata.json \
  --executable build/matmul-inspector --output results/dot-replay
python3 scripts/compare_hardware.py results/dot-smoke results/dot-replay \
  --comparator build/matmul-compare-outputs --output results/dot-aggregate
```

Replay retains operation, sizes, seeds, kernels, tolerances, trial controls and
saved-output choice; unsupported block sizes are refused. Cross-hardware
aggregation validates scalar output context and operation identity before using
the same native C++ detailed comparator. Source/toolchain comparability rules
remain in force; dirty-tree captures are not controlled performance evidence.
Legacy artifacts without operation are treated as matmul only for known legacy
matmul kernel names (including `cpu`). Unknown/vector kernel names without an
operation are rejected rather than guessed. Old CPU-only artifacts therefore
retain their original matmul interpretation.

## Compulsory roofline models

| Operation | Nominal FLOPs | Compulsory bytes |
|---|---:|---:|
| matmul | 2 M N K | 4 (M K + K N + M N) |
| dot, length L | 2 L | 4 (2 L + 1) |
| reduction_sum, length L | L − 1 | 4 (L + 1) |

These are mathematical conventions, not executed instruction counts. Tree
padding, partial-buffer reads/writes, synchronization and launch overhead are
not in the compulsory byte model. Bytes are not measured DRAM traffic. Sum at
length 1 has zero nominal FLOPs: its timing remains valid, but roofline analysis
excludes it with a reason because logarithmic intensity/efficiency is undefined.

```bash
# Illustrative limits only: substitute documented/measured ceilings for your GPU.
python3 scripts/roofline.py --input results/dot-smoke \
  --peak-fp32-tflops 10 --memory-bandwidth-gbps 100 \
  --source-note 'Illustrative limits, not GPU specifications' --report
```

GPU kernel-only rows use the operation-specific model; CPU rows are excluded.
Reports/CSV retain all numerical and reliability fields. No inference about an
actual hardware bottleneck follows from this simple model alone.

## Current boundaries

The CUDA resource helpers and vector launch code remain in `cuda_matmul.cu` to
reuse checked allocations/events without refactoring the existing backend.
The legacy output transport and M/N/K compatibility columns retain matrix names.
Vector inputs currently support only the random fixture. Block size is fixed,
not tuned. Crossover mode and its separate cross-hardware restriction remain
matmul-specific; there is no vector end-to-end workflow in this milestone.
Normal CPU tests require no GPU; CUDA integration tests are separately labeled
and skip when the runtime/device is unavailable.
