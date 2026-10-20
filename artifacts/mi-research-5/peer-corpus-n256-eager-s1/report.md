# Paired FC1 A/B comparison

FP16 M=256, K=256, N=256; eager preallocated mode.
Independent reference: PyTorch/cuBLAS `torch.matmul`.

- A: `{"block_k": 32, "block_m": 16, "block_n": 16, "group_size_m": 1, "num_stages": 2, "num_warps": 4}`; correctness PASS; max absolute error 0.03125.
- B: `{"block_k": 32, "block_m": 16, "block_n": 16, "group_size_m": 1, "num_stages": 2, "num_warps": 4}`; correctness PASS; max absolute error 0.03125.
- Additional seed 1: A PASS, B PASS.
- Additional seed 2: A PASS, B PASS.

**Decision: INCONCLUSIVE** — an A/B paired gap does not exceed the largest observed A/A gap

- A: CUDA-event median 0.027639 ms/op (range 0.026266–0.042536, CV 19.10%); host-wall median 0.030339 ms/op.
- B: CUDA-event median 0.030196 ms/op (range 0.025837–0.042066, CV 18.86%); host-wall median 0.032372 ms/op.
- Paired A−B: median -0.000442 ms/op; range -0.015390–0.008130; MAD 0.002944 ms/op.
- Paired signs: {'A_faster': 5, 'B_faster': 3, 'ties': 0}. Timer discrepancy: False.

| Round | CUDA-event A−B (ms/op) | Host-wall A−B (ms/op) |
|---:|---:|---:|
| 1 | -0.004998 | +0.004082 |
| 2 | -0.000845 | -0.001288 |
| 3 | -0.005171 | -0.005288 |
| 4 | -0.000040 | -0.000374 |
| 5 | +0.000307 | -0.000182 |
| 6 | +0.008130 | -0.009966 |
| 7 | +0.000890 | -0.000834 |
| 8 | -0.015390 | -0.015492 |

- A event−wall control: -0.200922 ms/block; round range -0.226997 to -0.028698 ms/block.
- B event−wall control: -0.072391 ms/block; round range -0.410509 to -0.038489 ms/block.

## Clock-aware diagnostics (v3)

The event interval brackets the configured calls on one CUDA stream. The perf and RAW host intervals enclose end-event synchronization. Perf is sampled just outside RAW, so the host windows differ by timestamp-read overhead. No clock is rescaled or treated as a universal absolute reference.

- Host-clock preflight: NO_RESOLVED_DIFFERENCE; median RAW/perf rate ratio 0.9999733502738085; five 20 ms windows when RAW is available.
- Same-kernel A/A: 8 balanced rounds; event X−Y median +0.001580 ms/op; maximum absolute paired gap 0.012910 ms/op.
- A median ms/op: CUDA event 0.027639; perf_counter wall 0.030339; MONOTONIC_RAW 0.030324.
- B median ms/op: CUDA event 0.030196; perf_counter wall 0.032372; MONOTONIC_RAW 0.032358.
- RAW enclosing-boundary discrepancy: False; A/B exceeds observed A/A gap: False.

| Round | Event A−B | perf A−B | RAW A−B | A/A event X−Y |
|---:|---:|---:|---:|---:|
| 1 | -0.004998 | +0.004082 | +0.004073 | +0.002504 |
| 2 | -0.000845 | -0.001288 | -0.001291 | +0.001178 |
| 3 | -0.005171 | -0.005288 | -0.005285 | +0.001982 |
| 4 | -0.000040 | -0.000374 | -0.000374 | +0.002509 |
| 5 | +0.000307 | -0.000182 | -0.000181 | -0.000005 |
| 6 | +0.008130 | -0.009966 | -0.009963 | +0.012910 |
| 7 | +0.000890 | -0.000834 | -0.000836 | -0.000040 |
| 8 | -0.015390 | -0.015492 | -0.015498 | -0.002302 |

All table differences are ms per operation; raw block totals and operation counts are retained in `raw_result.json`.

- Warnings: AB_WITHIN_AA_VARIABILITY.
- A/A is a local variability control, not a confidence interval. A supported ranking requires correctness, complete controls, event range separation, both host paired directions, and each A/B gap beyond the largest A/A gap. A RAW rate difference can explain a perf boundary reversal but cannot prove RAW absolute accuracy.

- A compiler: shared=4096 bytes; registers/thread=157; spills=0; warps=4; stages=2; PTX=CAPTURED; decoded SASS=UNAVAILABLE.
- B compiler: shared=4096 bytes; registers/thread=148; spills=0; warps=4; stages=2; PTX=CAPTURED; decoded SASS=UNAVAILABLE.
- Compiler shared-memory B−A: +0 bytes per program.
- Compiler registers/thread B−A: -9.
- Normalized PTX equal: False; changed lines: 6364.
- A PTX SHA-256: `cb4e13c0f258c869d5f02137f201f104c5475c2a4c4ee7b6cb43bf084a90d061`.
- B PTX SHA-256: `8500b9f19407e9fe6b34143e46ef0122ea7cf35be38e751fb396532da8a1a976`.
- PTX comparison: `ptx_comparison/comparison.json`.
- PTX is not decoded SASS. Resource metadata alone does not prove a latency gain.

## Provenance and reproduction

GPU: NVIDIA GeForce RTX 4060 Ti; driver: 616.92.
Software: {"cuda_runtime": "12.8", "python": "3.13.11", "torch": "2.11.0+cu128", "triton": "3.6.0"}.
Source SHA-256: `a88226c46f335a9768f15b52eb275abf6066bfb57b1f7e012faa937a51768552`; Git: `2fbc3c5c6d31419bc3454f604e934664030bc10a` (dirty=True).
Run: `matmul-inspector-python experiment pair --spec examples/mi_research_5_peer_corpus_n256_pair.json --output artifacts/mi-research-5/peer-corpus-n256-eager-s1 --allow-trusted-local-source`.
A source: `sources/A.py`; entrypoint `matmul_into`; SHA-256 `319c6cf6997807199d7b8f20b4971e6f1755a3bf6c764d1f801117af5139987c`.
B source: `sources/B.py`; entrypoint `matmul_into`; SHA-256 `3378cfa1d9703ab9d504a4357e3407763ecf9a270e17ba198be83b4f098762a8`.
Verify: `matmul-inspector-python experiment pair --verify artifacts/mi-research-5/peer-corpus-n256-eager-s1`.
Raw paired blocks, compiler files, source snapshot, and SHA-256 hashes are in this bundle.
