# Paired FC1 A/B comparison

FP16 M=512, K=768, N=3072; eager preallocated mode.
Independent reference: PyTorch/cuBLAS `torch.matmul`.

- A: `{"block_k": 32, "block_m": 64, "block_n": 128, "group_size_m": 1, "num_stages": 3, "num_warps": 4}`; correctness PASS; max absolute error 0.0.
- B: `{"block_k": 32, "block_m": 64, "block_n": 128, "group_size_m": 1, "num_stages": 3, "num_warps": 8}`; correctness PASS; max absolute error 0.0.
- Additional seed 1: A PASS, B PASS.
- Additional seed 2: A PASS, B PASS.

**Decision: INCONCLUSIVE** — an A/B paired gap does not exceed the largest observed A/A gap

- A: CUDA-event median 0.068644 ms/op (range 0.063601–0.075478, CV 6.14%); host-wall median 0.069633 ms/op.
- B: CUDA-event median 0.068842 ms/op (range 0.064111–0.080333, CV 7.72%); host-wall median 0.069759 ms/op.
- Paired A−B: median 0.000051 ms/op; range -0.011305–0.010612; MAD 0.001273 ms/op.
- Paired signs: {'A_faster': 4, 'B_faster': 4, 'ties': 0}. Timer discrepancy: False.

| Round | CUDA-event A−B (ms/op) | Host-wall A−B (ms/op) |
|---:|---:|---:|
| 1 | +0.000782 | +0.001778 |
| 2 | -0.011305 | -0.011028 |
| 3 | -0.010525 | -0.010614 |
| 4 | +0.010612 | +0.010388 |
| 5 | -0.000510 | -0.000462 |
| 6 | -0.000358 | -0.000533 |
| 7 | +0.000461 | +0.000431 |
| 8 | +0.001867 | +0.002040 |

- A event−wall control: -0.131357 ms/block; round range -0.185371 to -0.058999 ms/block.
- B event−wall control: -0.106932 ms/block; round range -0.097753 to -0.064705 ms/block.

## Clock-aware diagnostics (v3)

The event interval brackets the configured calls on one CUDA stream. The perf and RAW host intervals enclose end-event synchronization. Perf is sampled just outside RAW, so the host windows differ by timestamp-read overhead. No clock is rescaled or treated as a universal absolute reference.

- Diagnostic A/A placement: before each A/B pair; exact phase order is in `raw_result.json`.
- Host-clock preflight: NO_RESOLVED_DIFFERENCE; median RAW/perf rate ratio 0.9999749295103888; five 20 ms windows when RAW is available.
- Same-kernel A/A: 8 balanced rounds; event X−Y median -0.001413 ms/op; maximum absolute paired gap 0.014177 ms/op.
- A median ms/op: CUDA event 0.068644; perf_counter wall 0.069633; MONOTONIC_RAW 0.069623.
- B median ms/op: CUDA event 0.068842; perf_counter wall 0.069759; MONOTONIC_RAW 0.069752.
- RAW enclosing-boundary discrepancy: False; A/B exceeds observed A/A gap: False.

| Round | Event A−B | perf A−B | RAW A−B | A/A event X−Y |
|---:|---:|---:|---:|---:|
| 1 | +0.000782 | +0.001778 | +0.001769 | -0.007137 |
| 2 | -0.011305 | -0.011028 | -0.011022 | -0.005427 |
| 3 | -0.010525 | -0.010614 | -0.010607 | +0.002849 |
| 4 | +0.010612 | +0.010388 | +0.010384 | -0.000686 |
| 5 | -0.000510 | -0.000462 | -0.000463 | +0.000054 |
| 6 | -0.000358 | -0.000533 | -0.000536 | -0.002140 |
| 7 | +0.000461 | +0.000431 | +0.000429 | -0.014177 |
| 8 | +0.001867 | +0.002040 | +0.002036 | +0.000197 |

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
Run: `matmul-inspector-python experiment pair --spec examples/mi_research_1_warps8.json --output artifacts/mi-timing-3/evidence-v1/s1-before --allow-trusted-local-source --diagnostic-aa-placement before`.
A source: `candidate_source.py`; entrypoint `fc1_kernel`; SHA-256 `a88226c46f335a9768f15b52eb275abf6066bfb57b1f7e012faa937a51768552`.
B source: `sources/B.py`; entrypoint `matmul_into`; SHA-256 `bccab8ab65314fb6dcb5b521146ff4d99dec88f112697f5b12e463385e9d52a9`.
Verify: `matmul-inspector-python experiment pair --verify artifacts/mi-timing-3/evidence-v1/s1-before`.
Raw paired blocks, compiler files, source snapshot, and SHA-256 hashes are in this bundle.
