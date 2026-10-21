# Paired FC1 A/B comparison

FP16 M=64, K=64, N=64; eager preallocated mode.
Independent reference: PyTorch/cuBLAS `torch.matmul`.

- A: `{"block_k": 32, "block_m": 32, "block_n": 32, "group_size_m": 1, "num_stages": 2, "num_warps": 4}`; correctness PASS; max absolute error 0.0.
- B: `{"block_k": 32, "block_m": 16, "block_n": 16, "group_size_m": 1, "num_stages": 2, "num_warps": 4}`; correctness PASS; max absolute error 0.0078125.
- Additional seed 1: A PASS, B PASS.
- Additional seed 2: A PASS, B PASS.

**Decision: INCONCLUSIVE** — an A/B paired gap does not exceed the largest observed A/A gap

- A: CUDA-event median 0.012037 ms/op (range 0.011059–0.013656, CV 7.67%); host-wall median 0.013376 ms/op.
- B: CUDA-event median 0.014551 ms/op (range 0.013414–0.016936, CV 8.83%); host-wall median 0.015794 ms/op.
- Paired A−B: median -0.002699 ms/op; range -0.004213–0.000110; MAD 0.000687 ms/op.
- Paired signs: {'A_faster': 7, 'B_faster': 1, 'ties': 0}. Timer discrepancy: False.

| Round | CUDA-event A−B (ms/op) | Host-wall A−B (ms/op) |
|---:|---:|---:|
| 1 | -0.002867 | -0.002871 |
| 2 | -0.002531 | -0.002553 |
| 3 | -0.003280 | -0.003319 |
| 4 | -0.004213 | -0.004060 |
| 5 | -0.003493 | -0.003279 |
| 6 | +0.000110 | +0.000225 |
| 7 | -0.001637 | -0.001563 |
| 8 | -0.002366 | -0.002336 |

- A event−wall control: -0.097825 ms/block; round range -0.029403 to -0.024182 ms/block.
- B event−wall control: -0.140792 ms/block; round range -0.026542 to -0.023805 ms/block.

## Clock-aware diagnostics (v3)

The event interval brackets the configured calls on one CUDA stream. The perf and RAW host intervals enclose end-event synchronization. Perf is sampled just outside RAW, so the host windows differ by timestamp-read overhead. No clock is rescaled or treated as a universal absolute reference.

- Host-clock preflight: NO_RESOLVED_DIFFERENCE; median RAW/perf rate ratio 0.9999804860526051; five 20 ms windows when RAW is available.
- Same-kernel A/A: 8 balanced rounds; event X−Y median +0.000601 ms/op; maximum absolute paired gap 0.002059 ms/op.
- A median ms/op: CUDA event 0.012037; perf_counter wall 0.013376; MONOTONIC_RAW 0.013364.
- B median ms/op: CUDA event 0.014551; perf_counter wall 0.015794; MONOTONIC_RAW 0.015785.
- RAW enclosing-boundary discrepancy: False; A/B exceeds observed A/A gap: False.

| Round | Event A−B | perf A−B | RAW A−B | A/A event X−Y |
|---:|---:|---:|---:|---:|
| 1 | -0.002867 | -0.002871 | -0.002871 | -0.000317 |
| 2 | -0.002531 | -0.002553 | -0.002552 | +0.001843 |
| 3 | -0.003280 | -0.003319 | -0.003317 | +0.001731 |
| 4 | -0.004213 | -0.004060 | -0.004061 | +0.000720 |
| 5 | -0.003493 | -0.003279 | -0.003278 | +0.000926 |
| 6 | +0.000110 | +0.000225 | +0.000226 | -0.000325 |
| 7 | -0.001637 | -0.001563 | -0.001564 | +0.000482 |
| 8 | -0.002366 | -0.002336 | -0.002340 | -0.002059 |

All table differences are ms per operation; raw block totals and operation counts are retained in `raw_result.json`.

- Warnings: AB_WITHIN_AA_VARIABILITY.
- A/A is a local variability control, not a confidence interval. A supported ranking requires correctness, complete controls, event range separation, both host paired directions, and each A/B gap beyond the largest A/A gap. A RAW rate difference can explain a perf boundary reversal but cannot prove RAW absolute accuracy.

- A compiler: shared=4096 bytes; registers/thread=40; spills=0; warps=4; stages=2; PTX=CAPTURED; decoded SASS=UNAVAILABLE.
- B compiler: shared=1024 bytes; registers/thread=255; spills=0; warps=4; stages=2; PTX=CAPTURED; decoded SASS=UNAVAILABLE.
- Compiler shared-memory B−A: -3072 bytes per program.
- Compiler registers/thread B−A: +215.
- Normalized PTX equal: False; changed lines: 2833.
- A PTX SHA-256: `8532803361d0ec6ed59a2ae3f87ce4dabb164401431fe0cc1aded4e863be01e2`.
- B PTX SHA-256: `427c99536c2f3a8cbe846dd9cee95920b2b0a1bd2bde58433926a79d71d8fb74`.
- PTX comparison: `ptx_comparison/comparison.json`.
- PTX is not decoded SASS. Resource metadata alone does not prove a latency gain.

## Provenance and reproduction

GPU: NVIDIA GeForce RTX 4060 Ti; driver: 616.92.
Software: {"cuda_runtime": "12.8", "python": "3.13.11", "torch": "2.11.0+cu128", "triton": "3.6.0"}.
Source SHA-256: `a88226c46f335a9768f15b52eb275abf6066bfb57b1f7e012faa937a51768552`; Git: `f2d6607653d04972bbc04757bd0ac5bf1864af6d` (dirty=True).
Run: `matmul-inspector-python experiment pair --spec examples/mi_research_4_n64_pair.json --output artifacts/mi-research-4/n64-eager-v3 --allow-trusted-local-source`.
A source: `candidate_source.py`; entrypoint `fc1_kernel`; SHA-256 `a88226c46f335a9768f15b52eb275abf6066bfb57b1f7e012faa937a51768552`.
B source: `sources/B.py`; entrypoint `matmul_into`; SHA-256 `0257dd59374f6abb4a008bd136db1dc5633c2fb993ea0706d044f12517a2398f`.
Verify: `matmul-inspector-python experiment pair --verify artifacts/mi-research-4/n64-eager-v3`.
Raw paired blocks, compiler files, source snapshot, and SHA-256 hashes are in this bundle.
