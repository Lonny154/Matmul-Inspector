# Paired FC1 A/B comparison

FP16 M=512, K=768, N=3072; eager preallocated mode.
Independent reference: PyTorch/cuBLAS `torch.matmul`.

- A: `{"block_k": 32, "block_m": 64, "block_n": 128, "group_size_m": 1, "num_stages": 3, "num_warps": 4}`; correctness PASS; max absolute error 0.0.
- B: `{"block_k": 32, "block_m": 64, "block_n": 128, "group_size_m": 8, "num_stages": 3, "num_warps": 4}`; correctness PASS; max absolute error 0.0.
- Additional seed 1: A PASS, B PASS.
- Additional seed 2: A PASS, B PASS.

**Decision: INCONCLUSIVE** — an A/B paired gap does not exceed the largest observed A/A gap

- A: CUDA-event median 0.060911 ms/op (range 0.058921–0.065534, CV 4.11%); host-wall median 0.061475 ms/op.
- B: CUDA-event median 0.063204 ms/op (range 0.059142–0.066601, CV 3.95%); host-wall median 0.063770 ms/op.
- Paired A−B: median -0.001346 ms/op; range -0.005045–0.001577; MAD 0.001040 ms/op.
- Paired signs: {'A_faster': 6, 'B_faster': 2, 'ties': 0}. Timer discrepancy: False.

| Round | CUDA-event A−B (ms/op) | Host-wall A−B (ms/op) |
|---:|---:|---:|
| 1 | -0.002149 | -0.002119 |
| 2 | +0.000175 | +0.000281 |
| 3 | -0.001067 | -0.001102 |
| 4 | -0.002621 | -0.002672 |
| 5 | +0.001577 | +0.001427 |
| 6 | -0.001318 | -0.001327 |
| 7 | -0.005045 | -0.004898 |
| 8 | -0.001373 | -0.001586 |

- A event−wall control: -0.076958 ms/block; round range -0.070409 to -0.044013 ms/block.
- B event−wall control: -0.073210 ms/block; round range -0.068352 to -0.046954 ms/block.

## Clock-aware diagnostics (v3)

The event interval brackets the configured calls on one CUDA stream. The perf and RAW host intervals enclose end-event synchronization. Perf is sampled just outside RAW, so the host windows differ by timestamp-read overhead. No clock is rescaled or treated as a universal absolute reference.

- Host-clock preflight: RATE_DIFFERENCE_OBSERVED; median RAW/perf rate ratio 1.00113178736899; five 20 ms windows when RAW is available.
- Same-kernel A/A: 8 balanced rounds; event X−Y median -0.000956 ms/op; maximum absolute paired gap 0.008579 ms/op.
- A median ms/op: CUDA event 0.060911; perf_counter wall 0.061475; MONOTONIC_RAW 0.061532.
- B median ms/op: CUDA event 0.063204; perf_counter wall 0.063770; MONOTONIC_RAW 0.063833.
- RAW enclosing-boundary discrepancy: False; A/B exceeds observed A/A gap: False.

| Round | Event A−B | perf A−B | RAW A−B | A/A event X−Y |
|---:|---:|---:|---:|---:|
| 1 | -0.002149 | -0.002119 | -0.002122 | -0.000021 |
| 2 | +0.000175 | +0.000281 | +0.000282 | -0.005791 |
| 3 | -0.001067 | -0.001102 | -0.001103 | +0.005372 |
| 4 | -0.002621 | -0.002672 | -0.002671 | -0.002140 |
| 5 | +0.001577 | +0.001427 | +0.001422 | -0.008579 |
| 6 | -0.001318 | -0.001327 | -0.001334 | -0.001892 |
| 7 | -0.005045 | -0.004898 | -0.004903 | +0.004166 |
| 8 | -0.001373 | -0.001586 | -0.001586 | +0.004054 |

All table differences are ms per operation; raw block totals and operation counts are retained in `raw_result.json`.

- Warnings: HOST_CLOCK_RATE_DIFFERENCE, AB_WITHIN_AA_VARIABILITY.
- A/A is a local variability control, not a confidence interval. A supported ranking requires correctness, complete controls, event range separation, both host paired directions, and each A/B gap beyond the largest A/A gap. A RAW rate difference can explain a perf boundary reversal but cannot prove RAW absolute accuracy.

- A compiler: shared=24576 bytes; registers/thread=121; spills=0; warps=4; stages=3; PTX=CAPTURED; decoded SASS=UNAVAILABLE.
- B compiler: shared=24576 bytes; registers/thread=121; spills=0; warps=4; stages=3; PTX=CAPTURED; decoded SASS=UNAVAILABLE.
- Compiler shared-memory B−A: +0 bytes per program.
- Compiler registers/thread B−A: +0.
- Normalized PTX equal: False; changed lines: 807.
- A PTX SHA-256: `16be26977e1fc6b029cdaac1df5870f307b4e1e9f63f0240ec223f04f4fda668`.
- B PTX SHA-256: `2404ab6c6894928aef896014ffb3ec73f435a2f3e7e36a8bb117bd8055b5d75e`.
- PTX comparison: `ptx_comparison/comparison.json`.
- PTX is not decoded SASS. Resource metadata alone does not prove a latency gain.

## Provenance and reproduction

GPU: NVIDIA GeForce RTX 4060 Ti; driver: 616.92.
Software: {"cuda_runtime": "12.8", "python": "3.13.11", "torch": "2.11.0+cu128", "triton": "3.6.0"}.
Source SHA-256: `a88226c46f335a9768f15b52eb275abf6066bfb57b1f7e012faa937a51768552`; Git: `30930ab0f5c20e603cb5fc9cd0711dfa6dc6c00d` (dirty=True).
Run: `matmul-inspector-python experiment pair --spec examples/mi_research_1_group8.json --output artifacts/mi-research-1/group8-v1 --allow-trusted-local-source`.
A source: `candidate_source.py`; entrypoint `fc1_kernel`; SHA-256 `a88226c46f335a9768f15b52eb275abf6066bfb57b1f7e012faa937a51768552`.
B source: `sources/B.py`; entrypoint `matmul_into`; SHA-256 `bccab8ab65314fb6dcb5b521146ff4d99dec88f112697f5b12e463385e9d52a9`.
Verify: `matmul-inspector-python experiment pair --verify artifacts/mi-research-1/group8-v1`.
Raw paired blocks, compiler files, source snapshot, and SHA-256 hashes are in this bundle.
