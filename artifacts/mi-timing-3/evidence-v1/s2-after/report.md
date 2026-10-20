# Paired FC1 A/B comparison

FP16 M=512, K=768, N=3072; eager preallocated mode.
Independent reference: PyTorch/cuBLAS `torch.matmul`.

- A: `{"block_k": 32, "block_m": 64, "block_n": 128, "group_size_m": 1, "num_stages": 3, "num_warps": 4}`; correctness PASS; max absolute error 0.0.
- B: `{"block_k": 32, "block_m": 64, "block_n": 128, "group_size_m": 1, "num_stages": 3, "num_warps": 8}`; correctness PASS; max absolute error 0.0.
- Additional seed 1: A PASS, B PASS.
- Additional seed 2: A PASS, B PASS.

**Decision: INCONCLUSIVE** — an A/B paired gap does not exceed the largest observed A/A gap

- A: CUDA-event median 0.069559 ms/op (range 0.059742–0.084623, CV 12.25%); host-wall median 0.071078 ms/op.
- B: CUDA-event median 0.074296 ms/op (range 0.060047–0.084334, CV 11.19%); host-wall median 0.075010 ms/op.
- Paired A−B: median -0.002542 ms/op; range -0.016804–0.005622; MAD 0.004953 ms/op.
- Paired signs: {'A_faster': 6, 'B_faster': 2, 'ties': 0}. Timer discrepancy: False.

| Round | CUDA-event A−B (ms/op) | Host-wall A−B (ms/op) |
|---:|---:|---:|
| 1 | +0.004858 | +0.004985 |
| 2 | -0.016804 | -0.015920 |
| 3 | -0.001946 | -0.003028 |
| 4 | +0.005622 | +0.005403 |
| 5 | -0.003138 | -0.002925 |
| 6 | -0.006604 | -0.006452 |
| 7 | -0.000305 | -0.000302 |
| 8 | -0.008387 | -0.008365 |

- A event−wall control: -0.121582 ms/block; round range -0.211488 to -0.075057 ms/block.
- B event−wall control: -0.099879 ms/block; round range -0.213834 to -0.065815 ms/block.

## Clock-aware diagnostics (v3)

The event interval brackets the configured calls on one CUDA stream. The perf and RAW host intervals enclose end-event synchronization. Perf is sampled just outside RAW, so the host windows differ by timestamp-read overhead. No clock is rescaled or treated as a universal absolute reference.

- Diagnostic A/A placement: after each A/B pair; exact phase order is in `raw_result.json`.
- Host-clock preflight: NO_RESOLVED_DIFFERENCE; median RAW/perf rate ratio 0.9999784565002907; five 20 ms windows when RAW is available.
- Same-kernel A/A: 8 balanced rounds; event X−Y median -0.002148 ms/op; maximum absolute paired gap 0.004209 ms/op.
- A median ms/op: CUDA event 0.069559; perf_counter wall 0.071078; MONOTONIC_RAW 0.071067.
- B median ms/op: CUDA event 0.074296; perf_counter wall 0.075010; MONOTONIC_RAW 0.075001.
- RAW enclosing-boundary discrepancy: False; A/B exceeds observed A/A gap: False.

| Round | Event A−B | perf A−B | RAW A−B | A/A event X−Y |
|---:|---:|---:|---:|---:|
| 1 | +0.004858 | +0.004985 | +0.004979 | -0.001815 |
| 2 | -0.016804 | -0.015920 | -0.015914 | -0.000298 |
| 3 | -0.001946 | -0.003028 | -0.003017 | -0.002772 |
| 4 | +0.005622 | +0.005403 | +0.005397 | -0.001721 |
| 5 | -0.003138 | -0.002925 | -0.002916 | -0.003236 |
| 6 | -0.006604 | -0.006452 | -0.006454 | -0.004209 |
| 7 | -0.000305 | -0.000302 | -0.000304 | -0.002482 |
| 8 | -0.008387 | -0.008365 | -0.008361 | +0.001495 |

All table differences are ms per operation; raw block totals and operation counts are retained in `raw_result.json`.

- Warnings: AB_WITHIN_AA_VARIABILITY.
- A/A is a local variability control, not a confidence interval. A supported ranking requires correctness, complete controls, event range separation, both host paired directions, and each A/B gap beyond the largest A/A gap. A RAW rate difference can explain a perf boundary reversal but cannot prove RAW absolute accuracy.

- A compiler: shared=24576 bytes; registers/thread=121; spills=0; warps=4; stages=3; PTX=CAPTURED; decoded SASS=UNAVAILABLE.
- B compiler: shared=24576 bytes; registers/thread=78; spills=0; warps=8; stages=3; PTX=CAPTURED; decoded SASS=UNAVAILABLE.
- Compiler shared-memory B−A: +0 bytes per program.
- Compiler registers/thread B−A: -43.
- Normalized PTX equal: False; changed lines: 978.
- A PTX SHA-256: `16be26977e1fc6b029cdaac1df5870f307b4e1e9f63f0240ec223f04f4fda668`.
- B PTX SHA-256: `e836a1ae66c33f19c35c5e00df3399a5a36744637bac8d61b77e1c3ad6f574c5`.
- PTX comparison: `ptx_comparison/comparison.json`.
- PTX is not decoded SASS. Resource metadata alone does not prove a latency gain.

## Provenance and reproduction

GPU: NVIDIA GeForce RTX 4060 Ti; driver: 616.92.
Software: {"cuda_runtime": "12.8", "python": "3.13.11", "torch": "2.11.0+cu128", "triton": "3.6.0"}.
Source SHA-256: `a88226c46f335a9768f15b52eb275abf6066bfb57b1f7e012faa937a51768552`; Git: `e3bef96e1efc549f7b52eceb73535faae3c1c2cc` (dirty=True).
Run: `matmul-inspector-python experiment pair --spec examples/mi_research_1_warps8.json --output artifacts/mi-timing-3/evidence-v1/s2-after --allow-trusted-local-source --diagnostic-aa-placement after`.
A source: `candidate_source.py`; entrypoint `fc1_kernel`; SHA-256 `a88226c46f335a9768f15b52eb275abf6066bfb57b1f7e012faa937a51768552`.
B source: `sources/B.py`; entrypoint `matmul_into`; SHA-256 `bccab8ab65314fb6dcb5b521146ff4d99dec88f112697f5b12e463385e9d52a9`.
Verify: `matmul-inspector-python experiment pair --verify artifacts/mi-timing-3/evidence-v1/s2-after`.
Raw paired blocks, compiler files, source snapshot, and SHA-256 hashes are in this bundle.
