# Paired FC1 A/B comparison

FP16 M=256, K=256, N=256; eager preallocated mode.
Independent reference: PyTorch/cuBLAS `torch.matmul`.

- A: `{"block_k": 32, "block_m": 16, "block_n": 16, "group_size_m": 1, "num_stages": 2, "num_warps": 4}`; correctness PASS; max absolute error 0.03125.
- B: `{"block_k": 32, "block_m": 16, "block_n": 16, "group_size_m": 1, "num_stages": 2, "num_warps": 4}`; correctness PASS; max absolute error 0.03125.
- Additional seed 1: A PASS, B PASS.
- Additional seed 2: A PASS, B PASS.

**Decision: INCONCLUSIVE** — an A/B paired gap does not exceed the largest observed A/A gap

- A: CUDA-event median 0.029251 ms/op (range 0.028968–0.039168, CV 11.60%); host-wall median 0.030932 ms/op.
- B: CUDA-event median 0.026949 ms/op (range 0.026010–0.028774, CV 2.84%); host-wall median 0.028542 ms/op.
- Paired A−B: median 0.002306 ms/op; range 0.001979–0.010394; MAD 0.000234 ms/op.
- Paired signs: {'A_faster': 0, 'B_faster': 8, 'ties': 0}. Timer discrepancy: False.

| Round | CUDA-event A−B (ms/op) | Host-wall A−B (ms/op) |
|---:|---:|---:|
| 1 | +0.002150 | +0.002244 |
| 2 | +0.010394 | +0.010693 |
| 3 | +0.002619 | +0.002476 |
| 4 | +0.002958 | -0.002694 |
| 5 | +0.002245 | +0.002686 |
| 6 | +0.002318 | +0.002340 |
| 7 | +0.001979 | +0.002448 |
| 8 | +0.002294 | +0.002461 |

- A event−wall control: -0.353516 ms/block; round range -0.038477 to -0.032382 ms/block.
- B event−wall control: -0.041230 ms/block; round range -0.146605 to -0.028488 ms/block.

## Clock-aware diagnostics (v3)

The event interval brackets the configured calls on one CUDA stream. The perf and RAW host intervals enclose end-event synchronization. Perf is sampled just outside RAW, so the host windows differ by timestamp-read overhead. No clock is rescaled or treated as a universal absolute reference.

- Host-clock preflight: NO_RESOLVED_DIFFERENCE; median RAW/perf rate ratio 0.9999791095879679; five 20 ms windows when RAW is available.
- Same-kernel A/A: 8 balanced rounds; event X−Y median +0.000041 ms/op; maximum absolute paired gap 0.017275 ms/op.
- A median ms/op: CUDA event 0.029251; perf_counter wall 0.030932; MONOTONIC_RAW 0.030922.
- B median ms/op: CUDA event 0.026949; perf_counter wall 0.028542; MONOTONIC_RAW 0.028531.
- RAW enclosing-boundary discrepancy: False; A/B exceeds observed A/A gap: False.

| Round | Event A−B | perf A−B | RAW A−B | A/A event X−Y |
|---:|---:|---:|---:|---:|
| 1 | +0.002150 | +0.002244 | +0.002247 | +0.001074 |
| 2 | +0.010394 | +0.010693 | +0.010690 | -0.000410 |
| 3 | +0.002619 | +0.002476 | +0.002475 | -0.001058 |
| 4 | +0.002958 | -0.002694 | -0.002700 | +0.012750 |
| 5 | +0.002245 | +0.002686 | +0.002678 | +0.017275 |
| 6 | +0.002318 | +0.002340 | +0.002341 | +0.000051 |
| 7 | +0.001979 | +0.002448 | +0.002447 | -0.007269 |
| 8 | +0.002294 | +0.002461 | +0.002461 | +0.000030 |

All table differences are ms per operation; raw block totals and operation counts are retained in `raw_result.json`.

- Warnings: AB_WITHIN_AA_VARIABILITY.
- A/A is a local variability control, not a confidence interval. A supported ranking requires correctness, complete controls, event range separation, both host paired directions, and each A/B gap beyond the largest A/A gap. A RAW rate difference can explain a perf boundary reversal but cannot prove RAW absolute accuracy.

- A compiler: shared=2048 bytes; registers/thread=146; spills=0; warps=4; stages=2; PTX=CAPTURED; decoded SASS=UNAVAILABLE.
- B compiler: shared=4096 bytes; registers/thread=157; spills=0; warps=4; stages=2; PTX=CAPTURED; decoded SASS=UNAVAILABLE.
- Compiler shared-memory B−A: +2048 bytes per program.
- Compiler registers/thread B−A: +11.
- Normalized PTX equal: False; changed lines: 6295.
- A PTX SHA-256: `e1e77b8b6546c615d3f6ae9821e0f99663ebc0d53789c3bc63e4dd26c24babaf`.
- B PTX SHA-256: `cb4e13c0f258c869d5f02137f201f104c5475c2a4c4ee7b6cb43bf084a90d061`.
- PTX comparison: `ptx_comparison/comparison.json`.
- PTX is not decoded SASS. Resource metadata alone does not prove a latency gain.

## Provenance and reproduction

GPU: NVIDIA GeForce RTX 4060 Ti; driver: 616.92.
Software: {"cuda_runtime": "12.8", "python": "3.13.11", "torch": "2.11.0+cu128", "triton": "3.6.0"}.
Source SHA-256: `a88226c46f335a9768f15b52eb275abf6066bfb57b1f7e012faa937a51768552`; Git: `f4a5860bea4bb1b929f1db50f81a5466c64d37c5` (dirty=True).
Run: `matmul-inspector-python experiment pair --spec examples/mi_research_4c_n256_pair.json --output artifacts/mi-research-4c/n256-eager-s1 --allow-trusted-local-source`.
A source: `sources/A.py`; entrypoint `matmul_into`; SHA-256 `f1f355254182f73195282921d437d6baa0a565b2994cc0e99d7e87266e2ae468`.
B source: `sources/B.py`; entrypoint `matmul_into`; SHA-256 `319c6cf6997807199d7b8f20b4971e6f1755a3bf6c764d1f801117af5139987c`.
Verify: `matmul-inspector-python experiment pair --verify artifacts/mi-research-4c/n256-eager-s1`.
Raw paired blocks, compiler files, source snapshot, and SHA-256 hashes are in this bundle.
