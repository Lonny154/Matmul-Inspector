# Paired FC1 A/B comparison

FP16 M=32, K=4096, N=2048; eager preallocated mode.
Independent reference: PyTorch/cuBLAS `torch.matmul`.

- A: `{"block_k": 32, "block_m": 64, "block_n": 128, "group_size_m": 1, "num_stages": 3, "num_warps": 4}`; correctness PASS; max absolute error 0.0.
- B: `{"block_k": 32, "block_m": 64, "block_n": 128, "group_size_m": 1, "num_stages": 3, "num_warps": 4}`; correctness PASS; max absolute error 0.125.
- Additional seed 1: A PASS, B PASS.
- Additional seed 2: A PASS, B PASS.

**Decision: INCONCLUSIVE** — an A/B paired gap does not exceed the largest observed A/A gap

- A: CUDA-event median 0.063015 ms/op (range 0.061901–0.064756, CV 1.46%); host-wall median 0.063400 ms/op.
- B: CUDA-event median 0.036141 ms/op (range 0.034212–0.069079, CV 35.09%); host-wall median 0.036527 ms/op.
- Paired A−B: median 0.026593 ms/op; range -0.007178–0.030236; MAD 0.002129 ms/op.
- Paired signs: {'A_faster': 2, 'B_faster': 6, 'ties': 0}. Timer discrepancy: False.

| Round | CUDA-event A−B (ms/op) | Host-wall A−B (ms/op) |
|---:|---:|---:|
| 1 | -0.005118 | -0.005094 |
| 2 | +0.029317 | +0.029195 |
| 3 | +0.025375 | +0.025739 |
| 4 | +0.025496 | +0.025449 |
| 5 | +0.027689 | +0.027746 |
| 6 | +0.030236 | +0.030205 |
| 7 | -0.007178 | -0.007278 |
| 8 | +0.028125 | +0.028076 |

- A event−wall control: -0.055431 ms/block; round range -0.069633 to -0.028133 ms/block.
- B event−wall control: -0.049841 ms/block; round range -0.047136 to -0.033176 ms/block.

## Clock-aware diagnostics (v3)

The event interval brackets the configured calls on one CUDA stream. The perf and RAW host intervals enclose end-event synchronization. Perf is sampled just outside RAW, so the host windows differ by timestamp-read overhead. No clock is rescaled or treated as a universal absolute reference.

- Host-clock preflight: NO_RESOLVED_DIFFERENCE; median RAW/perf rate ratio 0.9999860526026093; five 20 ms windows when RAW is available.
- Same-kernel A/A: 8 balanced rounds; event X−Y median -0.000360 ms/op; maximum absolute paired gap 0.005261 ms/op.
- A median ms/op: CUDA event 0.063015; perf_counter wall 0.063400; MONOTONIC_RAW 0.063396.
- B median ms/op: CUDA event 0.036141; perf_counter wall 0.036527; MONOTONIC_RAW 0.036524.
- RAW enclosing-boundary discrepancy: False; A/B exceeds observed A/A gap: False.

| Round | Event A−B | perf A−B | RAW A−B | A/A event X−Y |
|---:|---:|---:|---:|---:|
| 1 | -0.005118 | -0.005094 | -0.005095 | +0.002111 |
| 2 | +0.029317 | +0.029195 | +0.029195 | -0.003430 |
| 3 | +0.025375 | +0.025739 | +0.025737 | +0.000105 |
| 4 | +0.025496 | +0.025449 | +0.025445 | -0.001988 |
| 5 | +0.027689 | +0.027746 | +0.027742 | +0.005261 |
| 6 | +0.030236 | +0.030205 | +0.030205 | -0.000696 |
| 7 | -0.007178 | -0.007278 | -0.007281 | -0.000025 |
| 8 | +0.028125 | +0.028076 | +0.028076 | -0.000768 |

All table differences are ms per operation; raw block totals and operation counts are retained in `raw_result.json`.

- Warnings: AB_WITHIN_AA_VARIABILITY.
- A/A is a local variability control, not a confidence interval. A supported ranking requires correctness, complete controls, event range separation, both host paired directions, and each A/B gap beyond the largest A/A gap. A RAW rate difference can explain a perf boundary reversal but cannot prove RAW absolute accuracy.

- A compiler: shared=24576 bytes; registers/thread=119; spills=0; warps=4; stages=3; PTX=CAPTURED; decoded SASS=UNAVAILABLE.
- B compiler: shared=0 bytes; registers/thread=22; spills=0; warps=4; stages=3; PTX=CAPTURED; decoded SASS=UNAVAILABLE.
- Compiler shared-memory B−A: -24576 bytes per program.
- Compiler registers/thread B−A: -97.
- Normalized PTX equal: False; changed lines: 693.
- A PTX SHA-256: `2b0633490e7c09795b16b8ac18cf4f50c597c389578dfd5fa8b254cb1a071980`.
- B PTX SHA-256: `b43a6541cac2aa8a54ccfe276d2f02fb4ca1b6e5c7caebcc9b48f08d1d10fe5e`.
- PTX comparison: `ptx_comparison/comparison.json`.
- PTX is not decoded SASS. Resource metadata alone does not prove a latency gain.

## Provenance and reproduction

GPU: NVIDIA GeForce RTX 4060 Ti; driver: 616.92.
Software: {"cuda_runtime": "12.8", "python": "3.13.11", "torch": "2.11.0+cu128", "triton": "3.6.0"}.
Source SHA-256: `a88226c46f335a9768f15b52eb275abf6066bfb57b1f7e012faa937a51768552`; Git: `f86b77aa7d119ab9e4f458f9d81e856e7529b5b7` (dirty=True).
Run: `matmul-inspector-python experiment pair --spec examples/mi_research_2_low_output_split2.json --output artifacts/mi-research-2/evidence-v1/low-output-split2-repeat --allow-trusted-local-source`.
A source: `candidate_source.py`; entrypoint `fc1_kernel`; SHA-256 `a88226c46f335a9768f15b52eb275abf6066bfb57b1f7e012faa937a51768552`.
B source: `sources/B.py`; entrypoint `matmul_into`; SHA-256 `0b9144b81eed225c671518a1d7067f115865c32437ee66fe2db1b51309aa59e9`.
Verify: `matmul-inspector-python experiment pair --verify artifacts/mi-research-2/evidence-v1/low-output-split2-repeat`.
Raw paired blocks, compiler files, source snapshot, and SHA-256 hashes are in this bundle.
