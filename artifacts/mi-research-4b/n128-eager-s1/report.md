# Paired FC1 A/B comparison

FP16 M=128, K=128, N=128; eager preallocated mode.
Independent reference: PyTorch/cuBLAS `torch.matmul`.

- A: `{"block_k": 32, "block_m": 16, "block_n": 16, "group_size_m": 1, "num_stages": 2, "num_warps": 4}`; correctness PASS; max absolute error 0.015625.
- B: `{"block_k": 32, "block_m": 16, "block_n": 16, "group_size_m": 1, "num_stages": 2, "num_warps": 4}`; correctness PASS; max absolute error 0.015625.
- Additional seed 1: A PASS, B PASS.
- Additional seed 2: A PASS, B PASS.

**Decision: INCONCLUSIVE** — an A/B paired gap does not exceed the largest observed A/A gap

- A: CUDA-event median 0.016333 ms/op (range 0.013312–0.017869, CV 8.00%); host-wall median 0.017754 ms/op.
- B: CUDA-event median 0.016759 ms/op (range 0.015974–0.021299, CV 9.97%); host-wall median 0.018389 ms/op.
- Paired A−B: median -0.000614 ms/op; range -0.003482–0.000512; MAD 0.000610 ms/op.
- Paired signs: {'A_faster': 7, 'B_faster': 1, 'ties': 0}. Timer discrepancy: False.

| Round | CUDA-event A−B (ms/op) | Host-wall A−B (ms/op) |
|---:|---:|---:|
| 1 | -0.000238 | -0.000488 |
| 2 | -0.001272 | -0.001415 |
| 3 | -0.003482 | -0.003618 |
| 4 | -0.000256 | -0.000630 |
| 5 | -0.000051 | -0.000087 |
| 6 | -0.003430 | +0.007501 |
| 7 | +0.000512 | +0.000524 |
| 8 | -0.000973 | -0.000868 |

- A event−wall control: -0.171088 ms/block; round range -0.256591 to -0.026536 ms/block.
- B event−wall control: -0.035898 ms/block; round range -0.037967 to -0.026451 ms/block.

## Clock-aware diagnostics (v3)

The event interval brackets the configured calls on one CUDA stream. The perf and RAW host intervals enclose end-event synchronization. Perf is sampled just outside RAW, so the host windows differ by timestamp-read overhead. No clock is rescaled or treated as a universal absolute reference.

- Host-clock preflight: NO_RESOLVED_DIFFERENCE; median RAW/perf rate ratio 0.9999845184624553; five 20 ms windows when RAW is available.
- Same-kernel A/A: 8 balanced rounds; event X−Y median +0.001216 ms/op; maximum absolute paired gap 0.003021 ms/op.
- A median ms/op: CUDA event 0.016333; perf_counter wall 0.017754; MONOTONIC_RAW 0.017747.
- B median ms/op: CUDA event 0.016759; perf_counter wall 0.018389; MONOTONIC_RAW 0.018378.
- RAW enclosing-boundary discrepancy: False; A/B exceeds observed A/A gap: False.

| Round | Event A−B | perf A−B | RAW A−B | A/A event X−Y |
|---:|---:|---:|---:|---:|
| 1 | -0.000238 | -0.000488 | -0.000486 | +0.002107 |
| 2 | -0.001272 | -0.001415 | -0.001416 | +0.003021 |
| 3 | -0.003482 | -0.003618 | -0.003616 | +0.001997 |
| 4 | -0.000256 | -0.000630 | -0.000627 | +0.001525 |
| 5 | -0.000051 | -0.000087 | -0.000086 | +0.000307 |
| 6 | -0.003430 | +0.007501 | +0.007502 | +0.000907 |
| 7 | +0.000512 | +0.000524 | +0.000520 | -0.001019 |
| 8 | -0.000973 | -0.000868 | -0.000866 | -0.001880 |

All table differences are ms per operation; raw block totals and operation counts are retained in `raw_result.json`.

- Warnings: AB_WITHIN_AA_VARIABILITY.
- A/A is a local variability control, not a confidence interval. A supported ranking requires correctness, complete controls, event range separation, both host paired directions, and each A/B gap beyond the largest A/A gap. A RAW rate difference can explain a perf boundary reversal but cannot prove RAW absolute accuracy.

- A compiler: shared=2048 bytes; registers/thread=145; spills=0; warps=4; stages=2; PTX=CAPTURED; decoded SASS=UNAVAILABLE.
- B compiler: shared=4096 bytes; registers/thread=149; spills=0; warps=4; stages=2; PTX=CAPTURED; decoded SASS=UNAVAILABLE.
- Compiler shared-memory B−A: +2048 bytes per program.
- Compiler registers/thread B−A: +4.
- Normalized PTX equal: False; changed lines: 6322.
- A PTX SHA-256: `36a477f8a057f03ae22679ab161a7325e04f962d1eb04869240446cdfb929dd6`.
- B PTX SHA-256: `c34f49f049d3dc623581b905900f24f5ee67fe823237bf507aedd0934a5e0131`.
- PTX comparison: `ptx_comparison/comparison.json`.
- PTX is not decoded SASS. Resource metadata alone does not prove a latency gain.

## Provenance and reproduction

GPU: NVIDIA GeForce RTX 4060 Ti; driver: 616.92.
Software: {"cuda_runtime": "12.8", "python": "3.13.11", "torch": "2.11.0+cu128", "triton": "3.6.0"}.
Source SHA-256: `a88226c46f335a9768f15b52eb275abf6066bfb57b1f7e012faa937a51768552`; Git: `c55011bef73d6d42878481674bc75764fbfd6c50` (dirty=True).
Run: `matmul-inspector-python experiment pair --spec examples/mi_research_4b_n128_pair.json --output artifacts/mi-research-4b/n128-eager-s1 --allow-trusted-local-source`.
A source: `sources/A.py`; entrypoint `matmul_into`; SHA-256 `39e60868c5aae26af26d0470e49e6b221443ab5fa8fc1560f046bfe1842a4bd8`.
B source: `sources/B.py`; entrypoint `matmul_into`; SHA-256 `0257dd59374f6abb4a008bd136db1dc5633c2fb993ea0706d044f12517a2398f`.
Verify: `matmul-inspector-python experiment pair --verify artifacts/mi-research-4b/n128-eager-s1`.
Raw paired blocks, compiler files, source snapshot, and SHA-256 hashes are in this bundle.
