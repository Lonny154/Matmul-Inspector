# Paired FC1 A/B comparison

FP16 M=32, K=4096, N=2048; eager preallocated mode.
Independent reference: PyTorch/cuBLAS `torch.matmul`.

- A: `{"block_k": 32, "block_m": 64, "block_n": 128, "group_size_m": 1, "num_stages": 3, "num_warps": 4}`; correctness PASS; max absolute error 0.0.
- B: `{"block_k": 32, "block_m": 64, "block_n": 128, "group_size_m": 1, "num_stages": 3, "num_warps": 4}`; correctness PASS; max absolute error 0.125.
- Additional seed 1: A PASS, B PASS.
- Additional seed 2: A PASS, B PASS.

**Decision: INCONCLUSIVE** — event ranges overlap or paired host directions disagree

- A: CUDA-event median 0.062909 ms/op (range 0.057252–0.065420, CV 4.73%); host-wall median 0.063525 ms/op.
- B: CUDA-event median 0.037252 ms/op (range 0.032543–0.098564, CV 55.61%); host-wall median 0.037657 ms/op.
- Paired A−B: median 0.023039 ms/op; range -0.035210–0.030407; MAD 0.003716 ms/op.
- Paired signs: {'A_faster': 2, 'B_faster': 6, 'ties': 0}. Timer discrepancy: False.

| Round | CUDA-event A−B (ms/op) | Host-wall A−B (ms/op) |
|---:|---:|---:|
| 1 | +0.026563 | +0.026529 |
| 2 | +0.026948 | +0.026801 |
| 3 | +0.020632 | +0.020610 |
| 4 | +0.030407 | +0.030211 |
| 5 | -0.035210 | -0.034977 |
| 6 | -0.034488 | -0.034731 |
| 7 | +0.025446 | +0.025500 |
| 8 | +0.020236 | +0.020188 |

- A event−wall control: -0.094475 ms/block; round range -0.071466 to -0.030763 ms/block.
- B event−wall control: -0.078740 ms/block; round range -0.062851 to -0.035560 ms/block.

## Clock-aware diagnostics (v3)

The event interval brackets the configured calls on one CUDA stream. The perf and RAW host intervals enclose end-event synchronization. Perf is sampled just outside RAW, so the host windows differ by timestamp-read overhead. No clock is rescaled or treated as a universal absolute reference.

- Host-clock preflight: NO_RESOLVED_DIFFERENCE; median RAW/perf rate ratio 0.999981767609351; five 20 ms windows when RAW is available.
- Same-kernel A/A: 8 balanced rounds; event X−Y median -0.000492 ms/op; maximum absolute paired gap 0.002877 ms/op.
- A median ms/op: CUDA event 0.062909; perf_counter wall 0.063525; MONOTONIC_RAW 0.063519.
- B median ms/op: CUDA event 0.037252; perf_counter wall 0.037657; MONOTONIC_RAW 0.037654.
- RAW enclosing-boundary discrepancy: False; A/B exceeds observed A/A gap: True.

| Round | Event A−B | perf A−B | RAW A−B | A/A event X−Y |
|---:|---:|---:|---:|---:|
| 1 | +0.026563 | +0.026529 | +0.026528 | +0.000207 |
| 2 | +0.026948 | +0.026801 | +0.026800 | +0.000000 |
| 3 | +0.020632 | +0.020610 | +0.020606 | -0.001373 |
| 4 | +0.030407 | +0.030211 | +0.030210 | -0.000983 |
| 5 | -0.035210 | -0.034977 | -0.034980 | -0.001087 |
| 6 | -0.034488 | -0.034731 | -0.034731 | +0.000012 |
| 7 | +0.025446 | +0.025500 | +0.025500 | +0.000693 |
| 8 | +0.020236 | +0.020188 | +0.020186 | -0.002877 |

All table differences are ms per operation; raw block totals and operation counts are retained in `raw_result.json`.

- Warnings: none.
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
Run: `matmul-inspector-python experiment pair --spec examples/mi_research_2_low_output_split2.json --output artifacts/mi-research-2/evidence-v1/low-output-split2 --allow-trusted-local-source`.
A source: `candidate_source.py`; entrypoint `fc1_kernel`; SHA-256 `a88226c46f335a9768f15b52eb275abf6066bfb57b1f7e012faa937a51768552`.
B source: `sources/B.py`; entrypoint `matmul_into`; SHA-256 `0b9144b81eed225c671518a1d7067f115865c32437ee66fe2db1b51309aa59e9`.
Verify: `matmul-inspector-python experiment pair --verify artifacts/mi-research-2/evidence-v1/low-output-split2`.
Raw paired blocks, compiler files, source snapshot, and SHA-256 hashes are in this bundle.
