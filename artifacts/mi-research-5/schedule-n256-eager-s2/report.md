# Paired FC1 A/B comparison

FP16 M=256, K=256, N=256; eager preallocated mode.
Independent reference: PyTorch/cuBLAS `torch.matmul`.

- A: `{"block_k": 32, "block_m": 16, "block_n": 16, "group_size_m": 1, "num_stages": 2, "num_warps": 4}`; correctness PASS; max absolute error 0.03125.
- B: `{"block_k": 32, "block_m": 16, "block_n": 16, "group_size_m": 1, "num_stages": 2, "num_warps": 4}`; correctness PASS; max absolute error 0.015625.
- Additional seed 1: A PASS, B PASS.
- Additional seed 2: A PASS, B PASS.

**Decision: INCONCLUSIVE** — an A/B paired gap does not exceed the largest observed A/A gap

- A: CUDA-event median 0.048922 ms/op (range 0.036208–0.069376, CV 18.76%); host-wall median 0.053538 ms/op.
- B: CUDA-event median 0.050480 ms/op (range 0.042752–0.062238, CV 13.35%); host-wall median 0.055527 ms/op.
- Paired A−B: median -0.002608 ms/op; range -0.014776–0.008309; MAD 0.003994 ms/op.
- Paired signs: {'A_faster': 6, 'B_faster': 2, 'ties': 0}. Timer discrepancy: False.

| Round | CUDA-event A−B (ms/op) | Host-wall A−B (ms/op) |
|---:|---:|---:|
| 1 | -0.006544 | -0.006644 |
| 2 | +0.001443 | +0.002742 |
| 3 | -0.004454 | +0.004361 |
| 4 | +0.008309 | +0.010336 |
| 5 | -0.008899 | -0.009260 |
| 6 | -0.000256 | -0.005077 |
| 7 | -0.000762 | +0.002571 |
| 8 | -0.014776 | -0.015562 |

- A event−wall control: -0.430304 ms/block; round range -0.235788 to -0.038368 ms/block.
- B event−wall control: -0.045572 ms/block; round range -0.159127 to -0.038227 ms/block.

## Clock-aware diagnostics (v3)

The event interval brackets the configured calls on one CUDA stream. The perf and RAW host intervals enclose end-event synchronization. Perf is sampled just outside RAW, so the host windows differ by timestamp-read overhead. No clock is rescaled or treated as a universal absolute reference.

- Host-clock preflight: NO_RESOLVED_DIFFERENCE; median RAW/perf rate ratio 0.9999782971884457; five 20 ms windows when RAW is available.
- Same-kernel A/A: 8 balanced rounds; event X−Y median -0.000419 ms/op; maximum absolute paired gap 0.029499 ms/op.
- A median ms/op: CUDA event 0.048922; perf_counter wall 0.053538; MONOTONIC_RAW 0.052915.
- B median ms/op: CUDA event 0.050480; perf_counter wall 0.055527; MONOTONIC_RAW 0.055508.
- RAW enclosing-boundary discrepancy: False; A/B exceeds observed A/A gap: False.

| Round | Event A−B | perf A−B | RAW A−B | A/A event X−Y |
|---:|---:|---:|---:|---:|
| 1 | -0.006544 | -0.006644 | -0.006629 | +0.000456 |
| 2 | +0.001443 | +0.002742 | +0.002740 | -0.009590 |
| 3 | -0.004454 | +0.004361 | +0.004362 | -0.001294 |
| 4 | +0.008309 | +0.010336 | +0.010328 | +0.001280 |
| 5 | -0.008899 | -0.009260 | -0.010458 | +0.003758 |
| 6 | -0.000256 | -0.005077 | -0.005072 | -0.011234 |
| 7 | -0.000762 | +0.002571 | +0.002575 | -0.029499 |
| 8 | -0.014776 | -0.015562 | -0.015561 | +0.005786 |

All table differences are ms per operation; raw block totals and operation counts are retained in `raw_result.json`.

- Warnings: AB_WITHIN_AA_VARIABILITY.
- A/A is a local variability control, not a confidence interval. A supported ranking requires correctness, complete controls, event range separation, both host paired directions, and each A/B gap beyond the largest A/A gap. A RAW rate difference can explain a perf boundary reversal but cannot prove RAW absolute accuracy.

- A compiler: shared=4096 bytes; registers/thread=148; spills=0; warps=4; stages=2; PTX=CAPTURED; decoded SASS=UNAVAILABLE.
- B compiler: shared=4096 bytes; registers/thread=154; spills=0; warps=4; stages=2; PTX=CAPTURED; decoded SASS=UNAVAILABLE.
- Compiler shared-memory B−A: +0 bytes per program.
- Compiler registers/thread B−A: +6.
- Normalized PTX equal: False; changed lines: 4576.
- A PTX SHA-256: `8500b9f19407e9fe6b34143e46ef0122ea7cf35be38e751fb396532da8a1a976`.
- B PTX SHA-256: `c0f3093ed8ee8580560d4507bc5d7624daa66cd161b3ccf4544aabdb775f2455`.
- PTX comparison: `ptx_comparison/comparison.json`.
- PTX is not decoded SASS. Resource metadata alone does not prove a latency gain.

## Provenance and reproduction

GPU: NVIDIA GeForce RTX 4060 Ti; driver: 616.92.
Software: {"cuda_runtime": "12.8", "python": "3.13.11", "torch": "2.11.0+cu128", "triton": "3.6.0"}.
Source SHA-256: `a88226c46f335a9768f15b52eb275abf6066bfb57b1f7e012faa937a51768552`; Git: `2fbc3c5c6d31419bc3454f604e934664030bc10a` (dirty=True).
Run: `matmul-inspector-python experiment pair --spec examples/mi_research_5_schedule_n256_pair.json --output artifacts/mi-research-5/schedule-n256-eager-s2 --allow-trusted-local-source`.
A source: `sources/A.py`; entrypoint `matmul_into`; SHA-256 `3378cfa1d9703ab9d504a4357e3407763ecf9a270e17ba198be83b4f098762a8`.
B source: `sources/B.py`; entrypoint `matmul_into`; SHA-256 `f580b6d635b288dc8ab9998cd92f9c7e01efe0212cdc03ff8cd52e26f49b436d`.
Verify: `matmul-inspector-python experiment pair --verify artifacts/mi-research-5/schedule-n256-eager-s2`.
Raw paired blocks, compiler files, source snapshot, and SHA-256 hashes are in this bundle.
