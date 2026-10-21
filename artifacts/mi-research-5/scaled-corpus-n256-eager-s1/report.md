# Paired FC1 A/B comparison

FP16 M=256, K=256, N=256; eager preallocated mode.
Independent reference: PyTorch/cuBLAS `torch.matmul`.

- A: `{"block_k": 32, "block_m": 16, "block_n": 16, "group_size_m": 1, "num_stages": 2, "num_warps": 4}`; correctness PASS; max absolute error 0.03125.
- B: `{"block_k": 32, "block_m": 16, "block_n": 16, "group_size_m": 1, "num_stages": 2, "num_warps": 4}`; correctness PASS; max absolute error 0.03125.
- Additional seed 1: A PASS, B PASS.
- Additional seed 2: A PASS, B PASS.

**Decision: INCONCLUSIVE** — an A/B paired gap does not exceed the largest observed A/A gap

- A: CUDA-event median 0.027520 ms/op (range 0.026264–0.040792, CV 16.61%); host-wall median 0.029075 ms/op.
- B: CUDA-event median 0.027869 ms/op (range 0.025958–0.039219, CV 16.92%); host-wall median 0.031433 ms/op.
- Paired A−B: median -0.000524 ms/op; range -0.011125–0.014834; MAD 0.003549 ms/op.
- Paired signs: {'A_faster': 5, 'B_faster': 3, 'ties': 0}. Timer discrepancy: False.

| Round | CUDA-event A−B (ms/op) | Host-wall A−B (ms/op) |
|---:|---:|---:|
| 1 | -0.010856 | -0.010754 |
| 2 | -0.001016 | -0.001728 |
| 3 | +0.014834 | +0.010073 |
| 4 | -0.000032 | -0.000270 |
| 5 | -0.011125 | -0.011175 |
| 6 | +0.000102 | -0.000179 |
| 7 | -0.006995 | -0.006992 |
| 8 | +0.000051 | -0.000192 |

- A event−wall control: -0.069639 ms/block; round range -0.040324 to -0.030439 ms/block.
- B event−wall control: -0.034350 ms/block; round range -0.130987 to -0.031693 ms/block.

## Clock-aware diagnostics (v3)

The event interval brackets the configured calls on one CUDA stream. The perf and RAW host intervals enclose end-event synchronization. Perf is sampled just outside RAW, so the host windows differ by timestamp-read overhead. No clock is rescaled or treated as a universal absolute reference.

- Host-clock preflight: NO_RESOLVED_DIFFERENCE; median RAW/perf rate ratio 0.9999826984639069; five 20 ms windows when RAW is available.
- Same-kernel A/A: 8 balanced rounds; event X−Y median +0.000136 ms/op; maximum absolute paired gap 0.013466 ms/op.
- A median ms/op: CUDA event 0.027520; perf_counter wall 0.029075; MONOTONIC_RAW 0.029064.
- B median ms/op: CUDA event 0.027869; perf_counter wall 0.031433; MONOTONIC_RAW 0.031415.
- RAW enclosing-boundary discrepancy: False; A/B exceeds observed A/A gap: False.

| Round | Event A−B | perf A−B | RAW A−B | A/A event X−Y |
|---:|---:|---:|---:|---:|
| 1 | -0.010856 | -0.010754 | -0.010754 | +0.002099 |
| 2 | -0.001016 | -0.001728 | -0.001714 | +0.006246 |
| 3 | +0.014834 | +0.010073 | +0.010063 | -0.006349 |
| 4 | -0.000032 | -0.000270 | -0.000271 | +0.009278 |
| 5 | -0.011125 | -0.011175 | -0.011174 | -0.000035 |
| 6 | +0.000102 | -0.000179 | -0.000180 | -0.013466 |
| 7 | -0.006995 | -0.006992 | -0.006989 | +0.000307 |
| 8 | +0.000051 | -0.000192 | -0.000192 | -0.003072 |

All table differences are ms per operation; raw block totals and operation counts are retained in `raw_result.json`.

- Warnings: AB_WITHIN_AA_VARIABILITY.
- A/A is a local variability control, not a confidence interval. A supported ranking requires correctness, complete controls, event range separation, both host paired directions, and each A/B gap beyond the largest A/A gap. A RAW rate difference can explain a perf boundary reversal but cannot prove RAW absolute accuracy.

- A compiler: shared=4096 bytes; registers/thread=157; spills=0; warps=4; stages=2; PTX=CAPTURED; decoded SASS=UNAVAILABLE.
- B compiler: shared=5120 bytes; registers/thread=157; spills=0; warps=4; stages=2; PTX=CAPTURED; decoded SASS=UNAVAILABLE.
- Compiler shared-memory B−A: +1024 bytes per program.
- Compiler registers/thread B−A: +0.
- Normalized PTX equal: False; changed lines: 6782.
- A PTX SHA-256: `cb4e13c0f258c869d5f02137f201f104c5475c2a4c4ee7b6cb43bf084a90d061`.
- B PTX SHA-256: `512dbed448097753ad0c7a653ad3a73f825a42c58ffb1d34acad3d8fdf1d26a2`.
- PTX comparison: `ptx_comparison/comparison.json`.
- PTX is not decoded SASS. Resource metadata alone does not prove a latency gain.

## Provenance and reproduction

GPU: NVIDIA GeForce RTX 4060 Ti; driver: 616.92.
Software: {"cuda_runtime": "12.8", "python": "3.13.11", "torch": "2.11.0+cu128", "triton": "3.6.0"}.
Source SHA-256: `a88226c46f335a9768f15b52eb275abf6066bfb57b1f7e012faa937a51768552`; Git: `2fbc3c5c6d31419bc3454f604e934664030bc10a` (dirty=True).
Run: `matmul-inspector-python experiment pair --spec examples/mi_research_5_scaled_corpus_n256_pair.json --output artifacts/mi-research-5/scaled-corpus-n256-eager-s1 --allow-trusted-local-source`.
A source: `sources/A.py`; entrypoint `matmul_into`; SHA-256 `319c6cf6997807199d7b8f20b4971e6f1755a3bf6c764d1f801117af5139987c`.
B source: `sources/B.py`; entrypoint `matmul_into`; SHA-256 `4a56243a55c92745d2bc8e08263666448036469f85b16013c465e0fe7df9da41`.
Verify: `matmul-inspector-python experiment pair --verify artifacts/mi-research-5/scaled-corpus-n256-eager-s1`.
Raw paired blocks, compiler files, source snapshot, and SHA-256 hashes are in this bundle.
