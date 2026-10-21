# Paired FC1 A/B comparison

FP16 M=128, K=128, N=128; eager preallocated mode.
Independent reference: PyTorch/cuBLAS `torch.matmul`.

- A: `{"block_k": 32, "block_m": 32, "block_n": 32, "group_size_m": 1, "num_stages": 2, "num_warps": 4}`; correctness PASS; max absolute error 0.0.
- B: `{"block_k": 32, "block_m": 16, "block_n": 16, "group_size_m": 1, "num_stages": 2, "num_warps": 4}`; correctness PASS; max absolute error 0.015625.
- Additional seed 1: A PASS, B PASS.
- Additional seed 2: A PASS, B PASS.

**Decision: INCONCLUSIVE** — an A/B paired gap does not exceed the largest observed A/A gap

- A: CUDA-event median 0.017843 ms/op (range 0.011675–0.055971, CV 63.58%); host-wall median 0.019188 ms/op.
- B: CUDA-event median 0.014917 ms/op (range 0.013718–0.052685, CV 64.72%); host-wall median 0.016345 ms/op.
- Paired A−B: median -0.001560 ms/op; range -0.019440–0.007690; MAD 0.003740 ms/op.
- Paired signs: {'A_faster': 5, 'B_faster': 3, 'ties': 0}. Timer discrepancy: False.

| Round | CUDA-event A−B (ms/op) | Host-wall A−B (ms/op) |
|---:|---:|---:|
| 1 | -0.019440 | -0.018967 |
| 2 | +0.007690 | +0.010959 |
| 3 | -0.004550 | -0.004123 |
| 4 | -0.002590 | -0.002767 |
| 5 | +0.002930 | +0.002910 |
| 6 | +0.003395 | +0.003398 |
| 7 | -0.001077 | -0.001063 |
| 8 | -0.002043 | -0.002032 |

- A event−wall control: -0.069956 ms/block; round range -0.107195 to -0.026195 ms/block.
- B event−wall control: -0.043407 ms/block; round range -0.044920 to -0.026145 ms/block.

## Clock-aware diagnostics (v3)

The event interval brackets the configured calls on one CUDA stream. The perf and RAW host intervals enclose end-event synchronization. Perf is sampled just outside RAW, so the host windows differ by timestamp-read overhead. No clock is rescaled or treated as a universal absolute reference.

- Host-clock preflight: NO_RESOLVED_DIFFERENCE; median RAW/perf rate ratio 0.999977066445125; five 20 ms windows when RAW is available.
- Same-kernel A/A: 8 balanced rounds; event X−Y median -0.001239 ms/op; maximum absolute paired gap 0.007270 ms/op.
- A median ms/op: CUDA event 0.017843; perf_counter wall 0.019188; MONOTONIC_RAW 0.019178.
- B median ms/op: CUDA event 0.014917; perf_counter wall 0.016345; MONOTONIC_RAW 0.016336.
- RAW enclosing-boundary discrepancy: False; A/B exceeds observed A/A gap: False.

| Round | Event A−B | perf A−B | RAW A−B | A/A event X−Y |
|---:|---:|---:|---:|---:|
| 1 | -0.019440 | -0.018967 | -0.018963 | -0.005186 |
| 2 | +0.007690 | +0.010959 | +0.010937 | -0.001453 |
| 3 | -0.004550 | -0.004123 | -0.004122 | -0.002509 |
| 4 | -0.002590 | -0.002767 | -0.002765 | +0.000973 |
| 5 | +0.002930 | +0.002910 | +0.002909 | +0.007270 |
| 6 | +0.003395 | +0.003398 | +0.003395 | -0.001026 |
| 7 | -0.001077 | -0.001063 | -0.001065 | -0.001643 |
| 8 | -0.002043 | -0.002032 | -0.002031 | +0.001178 |

All table differences are ms per operation; raw block totals and operation counts are retained in `raw_result.json`.

- Warnings: AB_WITHIN_AA_VARIABILITY.
- A/A is a local variability control, not a confidence interval. A supported ranking requires correctness, complete controls, event range separation, both host paired directions, and each A/B gap beyond the largest A/A gap. A RAW rate difference can explain a perf boundary reversal but cannot prove RAW absolute accuracy.

- A compiler: shared=4096 bytes; registers/thread=44; spills=0; warps=4; stages=2; PTX=CAPTURED; decoded SASS=UNAVAILABLE.
- B compiler: shared=4096 bytes; registers/thread=149; spills=0; warps=4; stages=2; PTX=CAPTURED; decoded SASS=UNAVAILABLE.
- Compiler shared-memory B−A: +0 bytes per program.
- Compiler registers/thread B−A: +105.
- Normalized PTX equal: False; changed lines: 3706.
- A PTX SHA-256: `483d8f557b9a2bde3875d07b661636494caa678b3edbdcd67f7ab504a481a932`.
- B PTX SHA-256: `c34f49f049d3dc623581b905900f24f5ee67fe823237bf507aedd0934a5e0131`.
- PTX comparison: `ptx_comparison/comparison.json`.
- PTX is not decoded SASS. Resource metadata alone does not prove a latency gain.

## Provenance and reproduction

GPU: NVIDIA GeForce RTX 4060 Ti; driver: 616.92.
Software: {"cuda_runtime": "12.8", "python": "3.13.11", "torch": "2.11.0+cu128", "triton": "3.6.0"}.
Source SHA-256: `a88226c46f335a9768f15b52eb275abf6066bfb57b1f7e012faa937a51768552`; Git: `f2d6607653d04972bbc04757bd0ac5bf1864af6d` (dirty=True).
Run: `matmul-inspector-python experiment pair --spec examples/mi_research_4_n128_pair.json --output artifacts/mi-research-4/n128-eager-v1 --allow-trusted-local-source`.
A source: `candidate_source.py`; entrypoint `fc1_kernel`; SHA-256 `a88226c46f335a9768f15b52eb275abf6066bfb57b1f7e012faa937a51768552`.
B source: `sources/B.py`; entrypoint `matmul_into`; SHA-256 `0257dd59374f6abb4a008bd136db1dc5633c2fb993ea0706d044f12517a2398f`.
Verify: `matmul-inspector-python experiment pair --verify artifacts/mi-research-4/n128-eager-v1`.
Raw paired blocks, compiler files, source snapshot, and SHA-256 hashes are in this bundle.
