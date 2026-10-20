# Paired FC1 A/B comparison

FP16 M=512, K=512, N=512; eager preallocated mode.
Independent reference: PyTorch/cuBLAS `torch.matmul`.

- A: `{"block_k": 32, "block_m": 16, "block_n": 16, "group_size_m": 1, "num_stages": 2, "num_warps": 4}`; correctness PASS; max absolute error 0.03125.
- B: `{"block_k": 32, "block_m": 16, "block_n": 16, "group_size_m": 1, "num_stages": 2, "num_warps": 4}`; correctness PASS; max absolute error 0.03125.
- Additional seed 1: A PASS, B PASS.
- Additional seed 2: A PASS, B PASS.

**Decision: INCONCLUSIVE** — an A/B paired gap does not exceed the largest observed A/A gap

- A: CUDA-event median 0.211046 ms/op (range 0.186973–0.232648, CV 8.46%); host-wall median 0.218407 ms/op.
- B: CUDA-event median 0.200900 ms/op (range 0.167834–0.216011, CV 9.12%); host-wall median 0.204027 ms/op.
- Paired A−B: median 0.015237 ms/op; range -0.028926–0.049725; MAD 0.026393 ms/op.
- Paired signs: {'A_faster': 3, 'B_faster': 5, 'ties': 0}. Timer discrepancy: False.

| Round | CUDA-event A−B (ms/op) | Host-wall A−B (ms/op) |
|---:|---:|---:|
| 1 | +0.049725 | +0.048759 |
| 2 | +0.038912 | +0.060179 |
| 3 | -0.013874 | -0.014013 |
| 4 | +0.013517 | +0.014383 |
| 5 | +0.016957 | +0.016290 |
| 6 | -0.013920 | -0.013447 |
| 7 | -0.028926 | -0.030663 |
| 8 | +0.023042 | +0.026867 |

- A event−wall control: -0.117705 ms/block; round range -0.485217 to -0.044773 ms/block.
- B event−wall control: -0.051207 ms/block; round range -0.202781 to -0.048360 ms/block.

## Clock-aware diagnostics (v3)

The event interval brackets the configured calls on one CUDA stream. The perf and RAW host intervals enclose end-event synchronization. Perf is sampled just outside RAW, so the host windows differ by timestamp-read overhead. No clock is rescaled or treated as a universal absolute reference.

- Host-clock preflight: NO_RESOLVED_DIFFERENCE; median RAW/perf rate ratio 0.9999733829150751; five 20 ms windows when RAW is available.
- Same-kernel A/A: 8 balanced rounds; event X−Y median -0.006286 ms/op; maximum absolute paired gap 0.050848 ms/op.
- A median ms/op: CUDA event 0.211046; perf_counter wall 0.218407; MONOTONIC_RAW 0.218370.
- B median ms/op: CUDA event 0.200900; perf_counter wall 0.204027; MONOTONIC_RAW 0.203982.
- RAW enclosing-boundary discrepancy: False; A/B exceeds observed A/A gap: False.

| Round | Event A−B | perf A−B | RAW A−B | A/A event X−Y |
|---:|---:|---:|---:|---:|
| 1 | +0.049725 | +0.048759 | +0.048775 | -0.017488 |
| 2 | +0.038912 | +0.060179 | +0.060176 | +0.031795 |
| 3 | -0.013874 | -0.014013 | -0.014013 | +0.009336 |
| 4 | +0.013517 | +0.014383 | +0.014388 | -0.036757 |
| 5 | +0.016957 | +0.016290 | +0.016295 | -0.021864 |
| 6 | -0.013920 | -0.013447 | -0.013425 | +0.050848 |
| 7 | -0.028926 | -0.030663 | -0.030569 | +0.004917 |
| 8 | +0.023042 | +0.026867 | +0.026859 | -0.017664 |

All table differences are ms per operation; raw block totals and operation counts are retained in `raw_result.json`.

- Warnings: AB_WITHIN_AA_VARIABILITY.
- A/A is a local variability control, not a confidence interval. A supported ranking requires correctness, complete controls, event range separation, both host paired directions, and each A/B gap beyond the largest A/A gap. A RAW rate difference can explain a perf boundary reversal but cannot prove RAW absolute accuracy.

- A compiler: shared=2048 bytes; registers/thread=146; spills=0; warps=4; stages=2; PTX=CAPTURED; decoded SASS=UNAVAILABLE.
- B compiler: shared=4096 bytes; registers/thread=157; spills=0; warps=4; stages=2; PTX=CAPTURED; decoded SASS=UNAVAILABLE.
- Compiler shared-memory B−A: +2048 bytes per program.
- Compiler registers/thread B−A: +11.
- Normalized PTX equal: False; changed lines: 6295.
- A PTX SHA-256: `44ee762e87641251c3cd0c64ce229960f3a7125b5b2c6a9ec5d3c0dccdeb0b88`.
- B PTX SHA-256: `b2633905b799459640981dd7fa6ac326ae1b7cc9a080e33ca9530bfa0e163d47`.
- PTX comparison: `ptx_comparison/comparison.json`.
- PTX is not decoded SASS. Resource metadata alone does not prove a latency gain.

## Provenance and reproduction

GPU: NVIDIA GeForce RTX 4060 Ti; driver: 616.92.
Software: {"cuda_runtime": "12.8", "python": "3.13.11", "torch": "2.11.0+cu128", "triton": "3.6.0"}.
Source SHA-256: `a88226c46f335a9768f15b52eb275abf6066bfb57b1f7e012faa937a51768552`; Git: `f4a5860bea4bb1b929f1db50f81a5466c64d37c5` (dirty=True).
Run: `matmul-inspector-python experiment pair --spec examples/mi_research_4c_n512_pair.json --output artifacts/mi-research-4c/n512-eager-s2 --allow-trusted-local-source`.
A source: `sources/A.py`; entrypoint `matmul_into`; SHA-256 `f1f355254182f73195282921d437d6baa0a565b2994cc0e99d7e87266e2ae468`.
B source: `sources/B.py`; entrypoint `matmul_into`; SHA-256 `319c6cf6997807199d7b8f20b4971e6f1755a3bf6c764d1f801117af5139987c`.
Verify: `matmul-inspector-python experiment pair --verify artifacts/mi-research-4c/n512-eager-s2`.
Raw paired blocks, compiler files, source snapshot, and SHA-256 hashes are in this bundle.
