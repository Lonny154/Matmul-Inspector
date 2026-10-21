# Paired FC1 A/B comparison

FP16 M=256, K=256, N=256; eager preallocated mode.
Independent reference: PyTorch/cuBLAS `torch.matmul`.

- A: `{"block_k": 32, "block_m": 16, "block_n": 16, "group_size_m": 1, "num_stages": 2, "num_warps": 4}`; correctness PASS; max absolute error 0.03125.
- B: `{"block_k": 32, "block_m": 16, "block_n": 16, "group_size_m": 1, "num_stages": 2, "num_warps": 4}`; correctness PASS; max absolute error 0.015625.
- Additional seed 1: A PASS, B PASS.
- Additional seed 2: A PASS, B PASS.

**Decision: INCONCLUSIVE** — an A/B paired gap does not exceed the largest observed A/A gap

- A: CUDA-event median 0.027864 ms/op (range 0.027238–0.037427, CV 12.10%); host-wall median 0.029550 ms/op.
- B: CUDA-event median 0.027286 ms/op (range 0.025810–0.046387, CV 23.14%); host-wall median 0.031115 ms/op.
- Paired A−B: median 0.000510 ms/op; range -0.014408–0.011571; MAD 0.001235 ms/op.
- Paired signs: {'A_faster': 3, 'B_faster': 5, 'ties': 0}. Timer discrepancy: False.

| Round | CUDA-event A−B (ms/op) | Host-wall A−B (ms/op) |
|---:|---:|---:|
| 1 | -0.000734 | -0.000538 |
| 2 | +0.000502 | +0.000479 |
| 3 | +0.011571 | +0.014957 |
| 4 | +0.000517 | +0.000580 |
| 5 | -0.014408 | -0.017998 |
| 6 | +0.001472 | +0.002254 |
| 7 | -0.003328 | -0.003366 |
| 8 | +0.001736 | -0.004857 |

- A event−wall control: -0.061325 ms/block; round range -0.235225 to -0.031226 ms/block.
- B event−wall control: -0.030614 ms/block; round range -0.167501 to -0.029933 ms/block.

## Clock-aware diagnostics (v3)

The event interval brackets the configured calls on one CUDA stream. The perf and RAW host intervals enclose end-event synchronization. Perf is sampled just outside RAW, so the host windows differ by timestamp-read overhead. No clock is rescaled or treated as a universal absolute reference.

- Host-clock preflight: NO_RESOLVED_DIFFERENCE; median RAW/perf rate ratio 0.9999753960588362; five 20 ms windows when RAW is available.
- Same-kernel A/A: 8 balanced rounds; event X−Y median -0.000184 ms/op; maximum absolute paired gap 0.018432 ms/op.
- A median ms/op: CUDA event 0.027864; perf_counter wall 0.029550; MONOTONIC_RAW 0.029539.
- B median ms/op: CUDA event 0.027286; perf_counter wall 0.031115; MONOTONIC_RAW 0.031100.
- RAW enclosing-boundary discrepancy: False; A/B exceeds observed A/A gap: False.

| Round | Event A−B | perf A−B | RAW A−B | A/A event X−Y |
|---:|---:|---:|---:|---:|
| 1 | -0.000734 | -0.000538 | -0.000537 | +0.018432 |
| 2 | +0.000502 | +0.000479 | +0.000479 | -0.000491 |
| 3 | +0.011571 | +0.014957 | +0.014955 | +0.002085 |
| 4 | +0.000517 | +0.000580 | +0.000582 | -0.000112 |
| 5 | -0.014408 | -0.017998 | -0.017994 | +0.000408 |
| 6 | +0.001472 | +0.002254 | +0.002274 | -0.001126 |
| 7 | -0.003328 | -0.003366 | -0.003358 | -0.000256 |
| 8 | +0.001736 | -0.004857 | -0.004857 | -0.001245 |

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
Run: `matmul-inspector-python experiment pair --spec examples/mi_research_5_schedule_n256_pair.json --output artifacts/mi-research-5/schedule-n256-eager-s1 --allow-trusted-local-source`.
A source: `sources/A.py`; entrypoint `matmul_into`; SHA-256 `3378cfa1d9703ab9d504a4357e3407763ecf9a270e17ba198be83b4f098762a8`.
B source: `sources/B.py`; entrypoint `matmul_into`; SHA-256 `f580b6d635b288dc8ab9998cd92f9c7e01efe0212cdc03ff8cd52e26f49b436d`.
Verify: `matmul-inspector-python experiment pair --verify artifacts/mi-research-5/schedule-n256-eager-s1`.
Raw paired blocks, compiler files, source snapshot, and SHA-256 hashes are in this bundle.
