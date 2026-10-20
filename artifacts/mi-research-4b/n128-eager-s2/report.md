# Paired FC1 A/B comparison

FP16 M=128, K=128, N=128; eager preallocated mode.
Independent reference: PyTorch/cuBLAS `torch.matmul`.

- A: `{"block_k": 32, "block_m": 16, "block_n": 16, "group_size_m": 1, "num_stages": 2, "num_warps": 4}`; correctness PASS; max absolute error 0.015625.
- B: `{"block_k": 32, "block_m": 16, "block_n": 16, "group_size_m": 1, "num_stages": 2, "num_warps": 4}`; correctness PASS; max absolute error 0.015625.
- Additional seed 1: A PASS, B PASS.
- Additional seed 2: A PASS, B PASS.

**Decision: INCONCLUSIVE** — an A/B paired gap does not exceed the largest observed A/A gap

- A: CUDA-event median 0.017433 ms/op (range 0.011418–0.045194, CV 52.09%); host-wall median 0.019055 ms/op.
- B: CUDA-event median 0.017944 ms/op (range 0.010138–0.041626, CV 49.57%); host-wall median 0.023050 ms/op.
- Paired A−B: median -0.000491 ms/op; range -0.017552–0.003584; MAD 0.002623 ms/op.
- Paired signs: {'A_faster': 5, 'B_faster': 3, 'ties': 0}. Timer discrepancy: False.

| Round | CUDA-event A−B (ms/op) | Host-wall A−B (ms/op) |
|---:|---:|---:|
| 1 | +0.003568 | +0.003329 |
| 2 | -0.016733 | -0.016568 |
| 3 | -0.000707 | -0.000541 |
| 4 | -0.017552 | -0.011765 |
| 5 | -0.000275 | -0.000351 |
| 6 | -0.001226 | -0.001305 |
| 7 | +0.000696 | +0.000276 |
| 8 | +0.003584 | -0.004776 |

- A event−wall control: -0.154647 ms/block; round range -0.161486 to -0.029475 ms/block.
- B event−wall control: -0.040683 ms/block; round range -0.328678 to -0.028268 ms/block.

## Clock-aware diagnostics (v3)

The event interval brackets the configured calls on one CUDA stream. The perf and RAW host intervals enclose end-event synchronization. Perf is sampled just outside RAW, so the host windows differ by timestamp-read overhead. No clock is rescaled or treated as a universal absolute reference.

- Host-clock preflight: NO_RESOLVED_DIFFERENCE; median RAW/perf rate ratio 0.999977889248941; five 20 ms windows when RAW is available.
- Same-kernel A/A: 8 balanced rounds; event X−Y median -0.001200 ms/op; maximum absolute paired gap 0.010488 ms/op.
- A median ms/op: CUDA event 0.017433; perf_counter wall 0.019055; MONOTONIC_RAW 0.019043.
- B median ms/op: CUDA event 0.017944; perf_counter wall 0.023050; MONOTONIC_RAW 0.023039.
- RAW enclosing-boundary discrepancy: False; A/B exceeds observed A/A gap: False.

| Round | Event A−B | perf A−B | RAW A−B | A/A event X−Y |
|---:|---:|---:|---:|---:|
| 1 | +0.003568 | +0.003329 | +0.003327 | -0.001632 |
| 2 | -0.016733 | -0.016568 | -0.016571 | -0.002717 |
| 3 | -0.000707 | -0.000541 | -0.000545 | -0.005080 |
| 4 | -0.017552 | -0.011765 | -0.011764 | +0.010488 |
| 5 | -0.000275 | -0.000351 | -0.000352 | -0.000768 |
| 6 | -0.001226 | -0.001305 | -0.001305 | +0.000154 |
| 7 | +0.000696 | +0.000276 | +0.000275 | -0.002296 |
| 8 | +0.003584 | -0.004776 | -0.004775 | +0.004880 |

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
Run: `matmul-inspector-python experiment pair --spec examples/mi_research_4b_n128_pair.json --output artifacts/mi-research-4b/n128-eager-s2 --allow-trusted-local-source`.
A source: `sources/A.py`; entrypoint `matmul_into`; SHA-256 `39e60868c5aae26af26d0470e49e6b221443ab5fa8fc1560f046bfe1842a4bd8`.
B source: `sources/B.py`; entrypoint `matmul_into`; SHA-256 `0257dd59374f6abb4a008bd136db1dc5633c2fb993ea0706d044f12517a2398f`.
Verify: `matmul-inspector-python experiment pair --verify artifacts/mi-research-4b/n128-eager-s2`.
Raw paired blocks, compiler files, source snapshot, and SHA-256 hashes are in this bundle.
