# Paired FC1 A/B comparison

FP16 M=512, K=768, N=3072; eager preallocated mode.
Independent reference: PyTorch/cuBLAS `torch.matmul`.

- A: `{"block_k": 32, "block_m": 64, "block_n": 128, "group_size_m": 1, "num_stages": 3, "num_warps": 4}`; correctness PASS; max absolute error 0.0.
- B: `{"block_k": 32, "block_m": 64, "block_n": 128, "group_size_m": 1, "num_stages": 3, "num_warps": 4}`; correctness PASS; max absolute error 0.0625.
- Additional seed 1: A PASS, B PASS.
- Additional seed 2: A PASS, B PASS.

**Decision: A_FASTER** — event ranges separate, both host paired directions agree, and every A/B gap exceeds A/A

- A: CUDA-event median 0.063544 ms/op (range 0.059361–0.068606, CV 5.11%); host-wall median 0.063971 ms/op.
- B: CUDA-event median 0.074767 ms/op (range 0.069806–0.081252, CV 4.20%); host-wall median 0.075200 ms/op.
- Paired A−B: median -0.011933 ms/op; range -0.017665–-0.006003; MAD 0.002085 ms/op.
- Paired signs: {'A_faster': 8, 'B_faster': 0, 'ties': 0}. Timer discrepancy: False.

| Round | CUDA-event A−B (ms/op) | Host-wall A−B (ms/op) |
|---:|---:|---:|
| 1 | -0.017665 | -0.017609 |
| 2 | -0.011120 | -0.011097 |
| 3 | -0.010332 | -0.012786 |
| 4 | -0.006003 | -0.005989 |
| 5 | -0.012746 | -0.012600 |
| 6 | -0.014623 | -0.014631 |
| 7 | -0.010445 | -0.010462 |
| 8 | -0.014503 | -0.014491 |

- A event−wall control: -0.066348 ms/block; round range -0.050814 to -0.040654 ms/block.
- B event−wall control: -0.050659 ms/block; round range -0.294634 to -0.035021 ms/block.

## Clock-aware diagnostics (v3)

The event interval brackets the configured calls on one CUDA stream. The perf and RAW host intervals enclose end-event synchronization. Perf is sampled just outside RAW, so the host windows differ by timestamp-read overhead. No clock is rescaled or treated as a universal absolute reference.

- Host-clock preflight: NO_RESOLVED_DIFFERENCE; median RAW/perf rate ratio 0.9999884939152736; five 20 ms windows when RAW is available.
- Same-kernel A/A: 8 balanced rounds; event X−Y median +0.000183 ms/op; maximum absolute paired gap 0.003735 ms/op.
- A median ms/op: CUDA event 0.063544; perf_counter wall 0.063971; MONOTONIC_RAW 0.063966.
- B median ms/op: CUDA event 0.074767; perf_counter wall 0.075200; MONOTONIC_RAW 0.075197.
- RAW enclosing-boundary discrepancy: False; A/B exceeds observed A/A gap: True.

| Round | Event A−B | perf A−B | RAW A−B | A/A event X−Y |
|---:|---:|---:|---:|---:|
| 1 | -0.017665 | -0.017609 | -0.017605 | -0.001327 |
| 2 | -0.011120 | -0.011097 | -0.011101 | +0.001796 |
| 3 | -0.010332 | -0.012786 | -0.012785 | +0.003370 |
| 4 | -0.006003 | -0.005989 | -0.005986 | +0.003083 |
| 5 | -0.012746 | -0.012600 | -0.012594 | +0.000028 |
| 6 | -0.014623 | -0.014631 | -0.014635 | -0.003246 |
| 7 | -0.010445 | -0.010462 | -0.010464 | -0.003735 |
| 8 | -0.014503 | -0.014491 | -0.014489 | +0.000339 |

All table differences are ms per operation; raw block totals and operation counts are retained in `raw_result.json`.

- Warnings: none.
- A/A is a local variability control, not a confidence interval. A supported ranking requires correctness, complete controls, event range separation, both host paired directions, and each A/B gap beyond the largest A/A gap. A RAW rate difference can explain a perf boundary reversal but cannot prove RAW absolute accuracy.

- A compiler: shared=24576 bytes; registers/thread=121; spills=0; warps=4; stages=3; PTX=CAPTURED; decoded SASS=UNAVAILABLE.
- B compiler: shared=0 bytes; registers/thread=22; spills=0; warps=4; stages=3; PTX=CAPTURED; decoded SASS=UNAVAILABLE.
- Compiler shared-memory B−A: -24576 bytes per program.
- Compiler registers/thread B−A: -99.
- Normalized PTX equal: False; changed lines: 754.
- A PTX SHA-256: `16be26977e1fc6b029cdaac1df5870f307b4e1e9f63f0240ec223f04f4fda668`.
- B PTX SHA-256: `c55e88127b68d46b35e81d79d7418baaaf52317068a19ee69f29f30df46b36a2`.
- PTX comparison: `ptx_comparison/comparison.json`.
- PTX is not decoded SASS. Resource metadata alone does not prove a latency gain.

## Provenance and reproduction

GPU: NVIDIA GeForce RTX 4060 Ti; driver: 616.92.
Software: {"cuda_runtime": "12.8", "python": "3.13.11", "torch": "2.11.0+cu128", "triton": "3.6.0"}.
Source SHA-256: `a88226c46f335a9768f15b52eb275abf6066bfb57b1f7e012faa937a51768552`; Git: `f86b77aa7d119ab9e4f458f9d81e856e7529b5b7` (dirty=True).
Run: `matmul-inspector-python experiment pair --spec examples/mi_research_2_fc1_split2.json --output artifacts/mi-research-2/evidence-v1/fc1-split2 --allow-trusted-local-source`.
A source: `candidate_source.py`; entrypoint `fc1_kernel`; SHA-256 `a88226c46f335a9768f15b52eb275abf6066bfb57b1f7e012faa937a51768552`.
B source: `sources/B.py`; entrypoint `matmul_into`; SHA-256 `0b9144b81eed225c671518a1d7067f115865c32437ee66fe2db1b51309aa59e9`.
Verify: `matmul-inspector-python experiment pair --verify artifacts/mi-research-2/evidence-v1/fc1-split2`.
Raw paired blocks, compiler files, source snapshot, and SHA-256 hashes are in this bundle.
