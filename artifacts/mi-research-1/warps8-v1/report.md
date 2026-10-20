# Paired FC1 A/B comparison

FP16 M=512, K=768, N=3072; eager preallocated mode.
Independent reference: PyTorch/cuBLAS `torch.matmul`.

- A: `{"block_k": 32, "block_m": 64, "block_n": 128, "group_size_m": 1, "num_stages": 3, "num_warps": 4}`; correctness PASS; max absolute error 0.0.
- B: `{"block_k": 32, "block_m": 64, "block_n": 128, "group_size_m": 1, "num_stages": 3, "num_warps": 8}`; correctness PASS; max absolute error 0.0.
- Additional seed 1: A PASS, B PASS.
- Additional seed 2: A PASS, B PASS.

**Decision: INCONCLUSIVE** — an A/B paired gap does not exceed the largest observed A/A gap

- A: CUDA-event median 0.067354 ms/op (range 0.063693–0.073714, CV 6.42%); host-wall median 0.068130 ms/op.
- B: CUDA-event median 0.064650 ms/op (range 0.064154–0.069704, CV 2.78%); host-wall median 0.065518 ms/op.
- Paired A−B: median 0.002767 ms/op; range -0.005673–0.008151; MAD 0.003682 ms/op.
- Paired signs: {'A_faster': 4, 'B_faster': 4, 'ties': 0}. Timer discrepancy: False.

| Round | CUDA-event A−B (ms/op) | Host-wall A−B (ms/op) |
|---:|---:|---:|
| 1 | +0.008151 | +0.008377 |
| 2 | -0.000276 | -0.000481 |
| 3 | +0.005810 | +0.005767 |
| 4 | -0.000751 | -0.000894 |
| 5 | +0.006614 | +0.007089 |
| 6 | +0.007750 | +0.008055 |
| 7 | -0.005673 | -0.005630 |
| 8 | -0.000461 | -0.000429 |

- A event−wall control: -0.147391 ms/block; round range -0.132788 to -0.060222 ms/block.
- B event−wall control: -0.117306 ms/block; round range -0.110184 to -0.055885 ms/block.

## Clock-aware diagnostics (v3)

The event interval brackets the configured calls on one CUDA stream. The perf and RAW host intervals enclose end-event synchronization. Perf is sampled just outside RAW, so the host windows differ by timestamp-read overhead. No clock is rescaled or treated as a universal absolute reference.

- Host-clock preflight: NO_RESOLVED_DIFFERENCE; median RAW/perf rate ratio 0.9999473232740442; five 20 ms windows when RAW is available.
- Same-kernel A/A: 8 balanced rounds; event X−Y median +0.000144 ms/op; maximum absolute paired gap 0.007301 ms/op.
- A median ms/op: CUDA event 0.067354; perf_counter wall 0.068130; MONOTONIC_RAW 0.068116.
- B median ms/op: CUDA event 0.064650; perf_counter wall 0.065518; MONOTONIC_RAW 0.065510.
- RAW enclosing-boundary discrepancy: False; A/B exceeds observed A/A gap: False.

| Round | Event A−B | perf A−B | RAW A−B | A/A event X−Y |
|---:|---:|---:|---:|---:|
| 1 | +0.008151 | +0.008377 | +0.008369 | +0.003195 |
| 2 | -0.000276 | -0.000481 | -0.000483 | +0.002755 |
| 3 | +0.005810 | +0.005767 | +0.005766 | +0.004956 |
| 4 | -0.000751 | -0.000894 | -0.000897 | -0.000430 |
| 5 | +0.006614 | +0.007089 | +0.007093 | +0.000029 |
| 6 | +0.007750 | +0.008055 | +0.008063 | -0.000021 |
| 7 | -0.005673 | -0.005630 | -0.005614 | -0.007301 |
| 8 | -0.000461 | -0.000429 | -0.000427 | +0.000258 |

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
Source SHA-256: `a88226c46f335a9768f15b52eb275abf6066bfb57b1f7e012faa937a51768552`; Git: `30930ab0f5c20e603cb5fc9cd0711dfa6dc6c00d` (dirty=True).
Run: `matmul-inspector-python experiment pair --spec examples/mi_research_1_warps8.json --output artifacts/mi-research-1/warps8-v1 --allow-trusted-local-source`.
A source: `candidate_source.py`; entrypoint `fc1_kernel`; SHA-256 `a88226c46f335a9768f15b52eb275abf6066bfb57b1f7e012faa937a51768552`.
B source: `sources/B.py`; entrypoint `matmul_into`; SHA-256 `bccab8ab65314fb6dcb5b521146ff4d99dec88f112697f5b12e463385e9d52a9`.
Verify: `matmul-inspector-python experiment pair --verify artifacts/mi-research-1/warps8-v1`.
Raw paired blocks, compiler files, source snapshot, and SHA-256 hashes are in this bundle.
