# Paired FC1 A/B comparison

FP16 M=256, K=256, N=256; eager preallocated mode.
Independent reference: PyTorch/cuBLAS `torch.matmul`.

- A: `{"block_k": 32, "block_m": 16, "block_n": 16, "group_size_m": 1, "num_stages": 2, "num_warps": 4}`; correctness PASS; max absolute error 0.03125.
- B: `{"block_k": 32, "block_m": 16, "block_n": 16, "group_size_m": 1, "num_stages": 2, "num_warps": 4}`; correctness PASS; max absolute error 0.03125.
- Additional seed 1: A PASS, B PASS.
- Additional seed 2: A PASS, B PASS.

**Decision: INCONCLUSIVE** — an A/B paired gap does not exceed the largest observed A/A gap

- A: CUDA-event median 0.027005 ms/op (range 0.026675–0.047400, CV 25.10%); host-wall median 0.028638 ms/op.
- B: CUDA-event median 0.027182 ms/op (range 0.026061–0.035531, CV 10.79%); host-wall median 0.029121 ms/op.
- Paired A−B: median 0.000023 ms/op; range -0.008549–0.019394; MAD 0.001638 ms/op.
- Paired signs: {'A_faster': 4, 'B_faster': 4, 'ties': 0}. Timer discrepancy: False.

| Round | CUDA-event A−B (ms/op) | Host-wall A−B (ms/op) |
|---:|---:|---:|
| 1 | +0.002067 | +0.003451 |
| 2 | +0.019394 | +0.019278 |
| 3 | -0.000005 | -0.000065 |
| 4 | +0.000051 | -0.000004 |
| 5 | -0.008549 | -0.008526 |
| 6 | -0.001208 | -0.001191 |
| 7 | +0.013614 | +0.008042 |
| 8 | -0.000118 | -0.000199 |

- A event−wall control: -0.091928 ms/block; round range -0.057615 to -0.031149 ms/block.
- B event−wall control: -0.046745 ms/block; round range -0.154868 to -0.029947 ms/block.

## Clock-aware diagnostics (v3)

The event interval brackets the configured calls on one CUDA stream. The perf and RAW host intervals enclose end-event synchronization. Perf is sampled just outside RAW, so the host windows differ by timestamp-read overhead. No clock is rescaled or treated as a universal absolute reference.

- Host-clock preflight: NO_RESOLVED_DIFFERENCE; median RAW/perf rate ratio 0.9999842661866458; five 20 ms windows when RAW is available.
- Same-kernel A/A: 8 balanced rounds; event X−Y median -0.000121 ms/op; maximum absolute paired gap 0.004762 ms/op.
- A median ms/op: CUDA event 0.027005; perf_counter wall 0.028638; MONOTONIC_RAW 0.028627.
- B median ms/op: CUDA event 0.027182; perf_counter wall 0.029121; MONOTONIC_RAW 0.029110.
- RAW enclosing-boundary discrepancy: False; A/B exceeds observed A/A gap: False.

| Round | Event A−B | perf A−B | RAW A−B | A/A event X−Y |
|---:|---:|---:|---:|---:|
| 1 | +0.002067 | +0.003451 | +0.003443 | -0.000242 |
| 2 | +0.019394 | +0.019278 | +0.019278 | +0.000970 |
| 3 | -0.000005 | -0.000065 | -0.000057 | -0.000570 |
| 4 | +0.000051 | -0.000004 | -0.000003 | -0.001286 |
| 5 | -0.008549 | -0.008526 | -0.008525 | +0.004762 |
| 6 | -0.001208 | -0.001191 | -0.001191 | +0.000000 |
| 7 | +0.013614 | +0.008042 | +0.008042 | -0.000398 |
| 8 | -0.000118 | -0.000199 | -0.000200 | +0.001221 |

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
Run: `matmul-inspector-python experiment pair --spec examples/mi_research_5_scaled_corpus_n256_pair.json --output artifacts/mi-research-5/scaled-corpus-n256-eager-s2 --allow-trusted-local-source`.
A source: `sources/A.py`; entrypoint `matmul_into`; SHA-256 `319c6cf6997807199d7b8f20b4971e6f1755a3bf6c764d1f801117af5139987c`.
B source: `sources/B.py`; entrypoint `matmul_into`; SHA-256 `4a56243a55c92745d2bc8e08263666448036469f85b16013c465e0fe7df9da41`.
Verify: `matmul-inspector-python experiment pair --verify artifacts/mi-research-5/scaled-corpus-n256-eager-s2`.
Raw paired blocks, compiler files, source snapshot, and SHA-256 hashes are in this bundle.
