# Paired FC1 A/B comparison

FP16 M=256, K=256, N=256; eager preallocated mode.
Independent reference: PyTorch/cuBLAS `torch.matmul`.

- A: `{"block_k": 32, "block_m": 16, "block_n": 16, "group_size_m": 1, "num_stages": 2, "num_warps": 4}`; correctness PASS; max absolute error 0.03125.
- B: `{"block_k": 32, "block_m": 16, "block_n": 16, "group_size_m": 1, "num_stages": 2, "num_warps": 4}`; correctness PASS; max absolute error 0.015625.
- Additional seed 1: A PASS, B PASS.
- Additional seed 2: A PASS, B PASS.

**Decision: INCONCLUSIVE** — an A/B paired gap does not exceed the largest observed A/A gap

- A: CUDA-event median 0.051308 ms/op (range 0.027798–0.086003, CV 36.86%); host-wall median 0.054286 ms/op.
- B: CUDA-event median 0.054221 ms/op (range 0.027245–0.090106, CV 37.12%); host-wall median 0.056956 ms/op.
- Paired A−B: median 0.000123 ms/op; range -0.038701–0.032294; MAD 0.005673 ms/op.
- Paired signs: {'A_faster': 4, 'B_faster': 4, 'ties': 0}. Timer discrepancy: False.

| Round | CUDA-event A−B (ms/op) | Host-wall A−B (ms/op) |
|---:|---:|---:|
| 1 | +0.000554 | +0.000601 |
| 2 | -0.000307 | -0.000068 |
| 3 | -0.038701 | -0.037388 |
| 4 | -0.003870 | -0.005248 |
| 5 | -0.007394 | -0.007651 |
| 6 | +0.001587 | +0.001335 |
| 7 | +0.032294 | +0.033411 |
| 8 | +0.007475 | +0.007849 |

- A event−wall control: -0.067231 ms/block; round range -0.081852 to -0.036868 ms/block.
- B event−wall control: -0.036588 ms/block; round range -0.074034 to -0.035917 ms/block.

## Clock-aware diagnostics (v3)

The event interval brackets the configured calls on one CUDA stream. The perf and RAW host intervals enclose end-event synchronization. Perf is sampled just outside RAW, so the host windows differ by timestamp-read overhead. No clock is rescaled or treated as a universal absolute reference.

- Host-clock preflight: NO_RESOLVED_DIFFERENCE; median RAW/perf rate ratio 0.9999773230553065; five 20 ms windows when RAW is available.
- Same-kernel A/A: 8 balanced rounds; event X−Y median +0.001568 ms/op; maximum absolute paired gap 0.034301 ms/op.
- A median ms/op: CUDA event 0.051308; perf_counter wall 0.054286; MONOTONIC_RAW 0.054272.
- B median ms/op: CUDA event 0.054221; perf_counter wall 0.056956; MONOTONIC_RAW 0.056943.
- RAW enclosing-boundary discrepancy: False; A/B exceeds observed A/A gap: False.

| Round | Event A−B | perf A−B | RAW A−B | A/A event X−Y |
|---:|---:|---:|---:|---:|
| 1 | +0.000554 | +0.000601 | +0.000600 | -0.000051 |
| 2 | -0.000307 | -0.000068 | -0.000038 | +0.028374 |
| 3 | -0.038701 | -0.037388 | -0.037377 | -0.000402 |
| 4 | -0.003870 | -0.005248 | -0.005246 | -0.001818 |
| 5 | -0.007394 | -0.007651 | -0.007646 | +0.034301 |
| 6 | +0.001587 | +0.001335 | +0.001336 | -0.001862 |
| 7 | +0.032294 | +0.033411 | +0.033404 | +0.004722 |
| 8 | +0.007475 | +0.007849 | +0.007847 | +0.003187 |

All table differences are ms per operation; raw block totals and operation counts are retained in `raw_result.json`.

- Warnings: AB_WITHIN_AA_VARIABILITY.
- A/A is a local variability control, not a confidence interval. A supported ranking requires correctness, complete controls, event range separation, both host paired directions, and each A/B gap beyond the largest A/A gap. A RAW rate difference can explain a perf boundary reversal but cannot prove RAW absolute accuracy.

- A compiler: shared=4096 bytes; registers/thread=157; spills=0; warps=4; stages=2; PTX=CAPTURED; decoded SASS=UNAVAILABLE.
- B compiler: shared=4096 bytes; registers/thread=154; spills=0; warps=4; stages=2; PTX=CAPTURED; decoded SASS=UNAVAILABLE.
- Compiler shared-memory B−A: +0 bytes per program.
- Compiler registers/thread B−A: -3.
- Normalized PTX equal: False; changed lines: 6290.
- A PTX SHA-256: `cb4e13c0f258c869d5f02137f201f104c5475c2a4c4ee7b6cb43bf084a90d061`.
- B PTX SHA-256: `c0f3093ed8ee8580560d4507bc5d7624daa66cd161b3ccf4544aabdb775f2455`.
- PTX comparison: `ptx_comparison/comparison.json`.
- PTX is not decoded SASS. Resource metadata alone does not prove a latency gain.

## Provenance and reproduction

GPU: NVIDIA GeForce RTX 4060 Ti; driver: 616.92.
Software: {"cuda_runtime": "12.8", "python": "3.13.11", "torch": "2.11.0+cu128", "triton": "3.6.0"}.
Source SHA-256: `a88226c46f335a9768f15b52eb275abf6066bfb57b1f7e012faa937a51768552`; Git: `2fbc3c5c6d31419bc3454f604e934664030bc10a` (dirty=True).
Run: `matmul-inspector-python experiment pair --spec examples/mi_research_5_peer_liveness_n256_pair.json --output artifacts/mi-research-5/peer-liveness-n256-eager-s1 --allow-trusted-local-source`.
A source: `sources/A.py`; entrypoint `matmul_into`; SHA-256 `319c6cf6997807199d7b8f20b4971e6f1755a3bf6c764d1f801117af5139987c`.
B source: `sources/B.py`; entrypoint `matmul_into`; SHA-256 `f580b6d635b288dc8ab9998cd92f9c7e01efe0212cdc03ff8cd52e26f49b436d`.
Verify: `matmul-inspector-python experiment pair --verify artifacts/mi-research-5/peer-liveness-n256-eager-s1`.
Raw paired blocks, compiler files, source snapshot, and SHA-256 hashes are in this bundle.
