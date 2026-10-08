# Trainium1 GEMM: NKI Kernel Development and Hardware Profiling

**Project:** Matmul Inspector  
**Date:** October 8, 2026  
**Hardware:** AWS EC2 `trn1.2xlarge`, one Trainium accelerator with two NeuronCores and 32 GB device memory  
**Software:** NKI `0.6.0+31049202112.g85070674` (as reported by Neuron Explorer); Neuron Explorer `2.32.0.498-b1f3998`; Ubuntu 24.04 Neuron DLAMI  
**Source branch:** [`nki-tiled-gemm`](https://github.com/Lonny154/Matmul-Inspector/tree/nki-tiled-gemm)

## Objective

Implement a tiled matrix-multiplication (GEMM) kernel using AWS Neuron Kernel Interface (NKI), validate its numerical correctness on physical Trainium1 hardware, measure NeuronCore latency separately from Python invocation overhead, inspect the compiler and hardware profile, and test an operand-reuse optimization.

## Implementation

The original proof of concept was a 16×16 NKI GEMM using `nisa.nc_matmul`. It was extended to a tiled kernel for 256×256 and 512×512 FP32 matrices. The baseline uses tiles of **M=128, N=128, K=128**. Input matrix A is supplied as a contiguous transposed array (`[K, M]`), while B uses `[K, N]` layout. Each output tile accumulates partial products in PSUM and is copied through SBUF to HBM.

The 512×512 baseline comprises 16 output tiles and 64 source-level `nc_matmul` invocations. The compiler reported 128 hardware `MATMUL` opcodes; source-level calls and emitted instructions should not be conflated.

A second implementation (`nki_tiled_matmul_v2.py`) explicitly preloads A tiles into SBUF for reuse across output-column tiles. Matrix dimensions, numerical inputs, tolerances, and tile geometry were held constant.

## Correctness

All comparisons used NumPy's matrix multiplication as a reference. The tiled tests applied `rtol=1e-3`, `atol=1e-3`.

| Test | Max absolute error | Mean absolute error | Result |
|---|---:|---:|---|
| 256×256 tiled baseline | 5.340576e-05 | 3.5582843e-06 | PASS |
| 512×512 tiled baseline | 5.340576e-05 | 5.299911e-06 | PASS |
| 512×512 explicit A-tile reuse | 5.340576e-05 | 5.299911e-06 | PASS |

Passing these tolerances does not establish bitwise equivalence or accuracy over all possible inputs.

## Measurement methodology

The initial Python-to-NKI timing included compilation/setup and repeated load/unload of temporary NEFF executables. Runtime logging showed new temporary `kernel.neff` paths and model load/unload for successive calls. Those Python call timings therefore **cannot** be used as isolated kernel execution latency.

Setting `NKI_ARTIFACTS_DIR` before launching the Python script preserved the generated `kernel.neff` and associated compiler artifacts. The compiled NEFF was then benchmarked using `neuron-bench exec` with **20 warmups, 200 work iterations**, and latency-only mode. The comparison below uses the **`NC USED = 1`** rows, to avoid conflating one-core and two-core configurations. Device profiling used `neuron-explorer capture` and its `summary-text` output.

A single displayed benchmark result per configuration was collected; repeated, interleaved runs and confidence intervals have not yet been performed.

## Performance results

| Measurement | 256×256 baseline | 512×512 baseline | 512×512 A-reuse v2 |
|---|---:|---:|---:|
| NeuronCore latency p50 | 16 µs | 40 µs | 40 µs |
| NeuronCore latency p99 | 16 µs | 41 µs | 41 µs |
| Runtime latency p50 (`L(50)`) | 59 µs | 92 µs | 94 µs |
| Runtime latency p99 (`L(99)`) | 75 µs | 109 µs | 114 µs |
| Correctness | PASS | PASS | PASS |

**Calculated throughput from reported NeuronCore p50:** 256×256: ~2.10 TFLOP/s; 512×512: ~6.71 TFLOP/s, using `2 × N³ / latency`. These are derived effective arithmetic rates, **not** peak hardware efficiency metrics.

### Standalone-call overhead

A diagnostic comparison of repeated standalone NKI invocations found median Python call times of **1.3710 seconds (16×16)** and **1.3405 seconds (256×256)**. A separate five-call test showed **8.0340 seconds** for its first call and approximately **1.34–1.39 seconds** thereafter. Runtime logs showed repeated NEFF loading/unloading. These measurements are not comparable to the microsecond-level NeuronCore latencies.

## Hardware profiling observations

| Profile metric | 256×256 baseline | 512×512 baseline |
|---|---:|---:|
| `total_exec_time` | 9.00 µs | 33.85 µs |
| DMA active time | 6.14 µs | 28.80 µs |
| Tensor Engine active time | 3.27 µs | 17.07 µs |
| HBM read bytes | 524,288 (512 KiB) | 2,097,152 (2 MiB) |
| HBM write bytes | 262,144 (256 KiB) | 1,048,576 (1 MiB) |
| Reported arithmetic intensity | 42.67 FLOPs/byte | 85.33 FLOPs/byte |
| Hardware `MATMUL` count | 16 | 128 |

Engine-active durations may overlap and **must not be summed** to estimate execution time. The report's `total_time`, `total_exec_time`, and the benchmark's `NCL` latency are distinct metrics using different accounting windows.

The HBM read/write totals match one full read of each FP32 input and one full write of the output at the respective matrix sizes. This **does not by itself prove** that all redundant transfers were eliminated at every level of the memory hierarchy, nor that DMA is on the critical execution path. Perfetto inspection identified overlapping DMA-engine events and matrix operations, including one selected 360 ns DMA event overlapping a 294 ns MATMUL event for approximately 93 ns; the direct data dependency between those two selected events was not established.

## Optimization experiment: explicit A-tile reuse

**Hypothesis:** Keeping A tiles resident in SBUF across neighboring output columns might reduce repeated data movement and/or improve scheduling.

**Change:** In v2, A tiles were preloaded once per M block and reused across N blocks, while retaining the original 128×128×128 tile geometry.

**Outcome:** v2 passed the same numerical test but recorded **40 µs NeuronCore p50**, unchanged from v1 at the reported precision. Its runtime p50 was 94 µs compared with 92 µs for v1, an inconclusive small difference given the limited number of runs.

The compiler's `dma_stats.txt` and `instruction_stats.txt` outputs were identical between versions (including 128 `MATMUL`, 128 `LDWEIGHTS`, 16 `COPY`, and 22 `EVENT_SEMAPHORE` opcodes). However, the NEFF SHA-256 hashes and the MLIR source differed, so the compiled artifacts were **not byte-identical**. Matching instruction counts do not demonstrate identical instruction scheduling or machine code.

**Conclusion:** Explicit A-tile reuse did **not demonstrate a speedup** in this experiment. A stronger claim about compiler optimization would require instruction-level and repeated-run analysis.

## Next investigations

1. Run matched v1/v2 benchmarks repeatedly and interleave their order; save raw measurements and report variability.
2. Test tile geometry **M=128, N=256, K=128**, subject to NKI layout/resource constraints, while preserving all other benchmark settings.
3. Compare compiler-generated instruction timing, SBUF allocation, DMA traffic, and the critical path in Perfetto rather than relying on aggregate active times alone.
4. Preserve source code and trace artifacts with SDK versions, instance type, build options, and benchmark commands for reproducibility.

## Reproduction commands

Run in the activated NKI-compatible virtual environment on a Trn1 instance:

```bash
cd ~/Matmul-Inspector
unset NKI_SIMULATOR
mkdir -p artifacts/nki-trn1-512
NKI_ARTIFACTS_DIR="$PWD/artifacts/nki-trn1-512" python scripts/nki_matmul_512.py

neuron-bench exec \
  --warmup=20 --work=200 --enable-only-latency --fixed-nc-count=1 \
  --output-directory=artifacts/nki-trn1-512-benchmark \
  artifacts/nki-trn1-512/kernel.neff

neuron-explorer capture \
  --neff=artifacts/nki-trn1-512/kernel.neff \
  --session-file=artifacts/nki-trn1-512/profile.ntff --num-exec=3

neuron-explorer view \
  --neff-path=artifacts/nki-trn1-512/kernel.neff \
  --session-file=artifacts/nki-trn1-512/profile.ntff \
  --output-format=summary-text
```

*The original 512×512 baseline measurement used the one-core row of a benchmark that tested both one-core and two-core modes. The reproduction command pins one core explicitly for consistency with v2; numbers may vary between runs.*

## Artifacts to retain

- `artifacts/nki-trn1-baseline/` — 256×256 compiled NEFF, compiler artifacts, and profile
- `artifacts/nki-trn1-512/` — 512×512 baseline NEFF, reports, and Perfetto trace
- `artifacts/nki-trn1-512-v2/` — A-reuse compiled kernel and compiler artifacts
- Corresponding `*-benchmark/` directories — raw benchmark records

Generated artifacts are excluded from Git; maintain a separate archive of the original files. Never include credentials in the archive or repository.
