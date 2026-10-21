# Paired FC1 A/B comparison

FP16 M=256, K=256, N=256; eager preallocated mode.
Independent reference: PyTorch/cuBLAS `torch.matmul`.

- A: `{"block_k": 32, "block_m": 16, "block_n": 16, "group_size_m": 1, "num_stages": 2, "num_warps": 4}`; correctness PASS; max absolute error 0.03125.
- B: `{"block_k": 32, "block_m": 16, "block_n": 16, "group_size_m": 1, "num_stages": 2, "num_warps": 4}`; correctness PASS; max absolute error 0.015625.
- Additional seed 1: A PASS, B PASS.
- Additional seed 2: A PASS, B PASS.

**Decision: INCONCLUSIVE** — an A/B paired gap does not exceed the largest observed A/A gap

- A: CUDA-event median 0.028054 ms/op (range 0.026573–0.051302, CV 30.86%); host-wall median 0.029906 ms/op.
- B: CUDA-event median 0.027673 ms/op (range 0.026573–0.031949, CV 6.25%); host-wall median 0.030018 ms/op.
- Paired A−B: median 0.000750 ms/op; range -0.004416–0.023451; MAD 0.002361 ms/op.
- Paired signs: {'A_faster': 3, 'B_faster': 5, 'ties': 0}. Timer discrepancy: False.

| Round | CUDA-event A−B (ms/op) | Host-wall A−B (ms/op) |
|---:|---:|---:|
| 1 | +0.023451 | +0.027829 |
| 2 | +0.002406 | +0.002518 |
| 3 | +0.020067 | +0.020657 |
| 4 | -0.000138 | +0.000046 |
| 5 | -0.004416 | -0.004125 |
| 6 | +0.000710 | +0.000051 |
| 7 | -0.002315 | -0.002181 |
| 8 | +0.000789 | +0.000217 |

- A event−wall control: -0.092527 ms/block; round range -0.131291 to -0.032889 ms/block.
- B event−wall control: -0.047719 ms/block; round range -0.050092 to -0.030208 ms/block.

## Clock-aware diagnostics (v3)

The event interval brackets the configured calls on one CUDA stream. The perf and RAW host intervals enclose end-event synchronization. Perf is sampled just outside RAW, so the host windows differ by timestamp-read overhead. No clock is rescaled or treated as a universal absolute reference.

- Host-clock preflight: NO_RESOLVED_DIFFERENCE; median RAW/perf rate ratio 0.9999419833376345; five 20 ms windows when RAW is available.
- Same-kernel A/A: 8 balanced rounds; event X−Y median -0.000034 ms/op; maximum absolute paired gap 0.020960 ms/op.
- A median ms/op: CUDA event 0.028054; perf_counter wall 0.029906; MONOTONIC_RAW 0.029893.
- B median ms/op: CUDA event 0.027673; perf_counter wall 0.030018; MONOTONIC_RAW 0.029998.
- RAW enclosing-boundary discrepancy: False; A/B exceeds observed A/A gap: False.

| Round | Event A−B | perf A−B | RAW A−B | A/A event X−Y |
|---:|---:|---:|---:|---:|
| 1 | +0.023451 | +0.027829 | +0.027823 | +0.000154 |
| 2 | +0.002406 | +0.002518 | +0.002516 | -0.000419 |
| 3 | +0.020067 | +0.020657 | +0.020647 | -0.006472 |
| 4 | -0.000138 | +0.000046 | +0.000047 | -0.000222 |
| 5 | -0.004416 | -0.004125 | -0.004075 | +0.000717 |
| 6 | +0.000710 | +0.000051 | +0.000064 | -0.018278 |
| 7 | -0.002315 | -0.002181 | -0.002176 | +0.000597 |
| 8 | +0.000789 | +0.000217 | +0.000224 | +0.020960 |

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
Run: `matmul-inspector-python experiment pair --spec examples/mi_research_5_peer_liveness_n256_pair.json --output artifacts/mi-research-5/peer-liveness-n256-eager-s2 --allow-trusted-local-source`.
A source: `sources/A.py`; entrypoint `matmul_into`; SHA-256 `319c6cf6997807199d7b8f20b4971e6f1755a3bf6c764d1f801117af5139987c`.
B source: `sources/B.py`; entrypoint `matmul_into`; SHA-256 `f580b6d635b288dc8ab9998cd92f9c7e01efe0212cdc03ff8cd52e26f49b436d`.
Verify: `matmul-inspector-python experiment pair --verify artifacts/mi-research-5/peer-liveness-n256-eager-s2`.
Raw paired blocks, compiler files, source snapshot, and SHA-256 hashes are in this bundle.
