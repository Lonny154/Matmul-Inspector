# Paired FC1 A/B comparison

FP16 M=512, K=512, N=512; eager preallocated mode.
Independent reference: PyTorch/cuBLAS `torch.matmul`.

- A: `{"block_k": 32, "block_m": 16, "block_n": 16, "group_size_m": 1, "num_stages": 2, "num_warps": 4}`; correctness PASS; max absolute error 0.03125.
- B: `{"block_k": 32, "block_m": 16, "block_n": 16, "group_size_m": 1, "num_stages": 2, "num_warps": 4}`; correctness PASS; max absolute error 0.03125.
- Additional seed 1: A PASS, B PASS.
- Additional seed 2: A PASS, B PASS.

**Decision: B_FASTER** — event ranges separate, both host paired directions agree, and every A/B gap exceeds A/A

- A: CUDA-event median 0.184346 ms/op (range 0.184269–0.185907, CV 0.30%); host-wall median 0.186796 ms/op.
- B: CUDA-event median 0.167232 ms/op (range 0.166898–0.168550, CV 0.30%); host-wall median 0.169598 ms/op.
- Paired A−B: median 0.017227 ms/op; range 0.016914–0.017422; MAD 0.000118 ms/op.
- Paired signs: {'A_faster': 0, 'B_faster': 8, 'ties': 0}. Timer discrepancy: False.

| Round | CUDA-event A−B (ms/op) | Host-wall A−B (ms/op) |
|---:|---:|---:|
| 1 | +0.017357 | +0.016894 |
| 2 | +0.017054 | +0.018089 |
| 3 | +0.017309 | +0.017479 |
| 4 | +0.017122 | +0.017659 |
| 5 | +0.017422 | +0.017396 |
| 6 | +0.017149 | +0.017088 |
| 7 | +0.016914 | +0.016584 |
| 8 | +0.017306 | +0.016971 |

- A event−wall control: -0.104472 ms/block; round range -0.058216 to -0.039662 ms/block.
- B event−wall control: -0.066416 ms/block; round range -0.061823 to -0.036783 ms/block.

## Clock-aware diagnostics (v3)

The event interval brackets the configured calls on one CUDA stream. The perf and RAW host intervals enclose end-event synchronization. Perf is sampled just outside RAW, so the host windows differ by timestamp-read overhead. No clock is rescaled or treated as a universal absolute reference.

- Host-clock preflight: NO_RESOLVED_DIFFERENCE; median RAW/perf rate ratio 0.9999728370574176; five 20 ms windows when RAW is available.
- Same-kernel A/A: 8 balanced rounds; event X−Y median +0.000110 ms/op; maximum absolute paired gap 0.002099 ms/op.
- A median ms/op: CUDA event 0.184346; perf_counter wall 0.186796; MONOTONIC_RAW 0.186771.
- B median ms/op: CUDA event 0.167232; perf_counter wall 0.169598; MONOTONIC_RAW 0.169563.
- RAW enclosing-boundary discrepancy: False; A/B exceeds observed A/A gap: True.

| Round | Event A−B | perf A−B | RAW A−B | A/A event X−Y |
|---:|---:|---:|---:|---:|
| 1 | +0.017357 | +0.016894 | +0.016857 | +0.000474 |
| 2 | +0.017054 | +0.018089 | +0.018066 | +0.000088 |
| 3 | +0.017309 | +0.017479 | +0.017503 | +0.001752 |
| 4 | +0.017122 | +0.017659 | +0.017637 | -0.000926 |
| 5 | +0.017422 | +0.017396 | +0.017419 | +0.001229 |
| 6 | +0.017149 | +0.017088 | +0.017090 | -0.000307 |
| 7 | +0.016914 | +0.016584 | +0.016581 | -0.002099 |
| 8 | +0.017306 | +0.016971 | +0.016988 | +0.000133 |

All table differences are ms per operation; raw block totals and operation counts are retained in `raw_result.json`.

- Warnings: none.
- A/A is a local variability control, not a confidence interval. A supported ranking requires correctness, complete controls, event range separation, both host paired directions, and each A/B gap beyond the largest A/A gap. A RAW rate difference can explain a perf boundary reversal but cannot prove RAW absolute accuracy.

- A compiler: shared=2048 bytes; registers/thread=146; spills=0; warps=4; stages=2; PTX=CAPTURED; decoded SASS=UNAVAILABLE.
- B compiler: shared=4096 bytes; registers/thread=157; spills=0; warps=4; stages=2; PTX=CAPTURED; decoded SASS=UNAVAILABLE.
- Compiler shared-memory B−A: +2048 bytes per program.
- Compiler registers/thread B−A: +11.
- Normalized PTX equal: False; changed lines: 6295.
- A PTX SHA-256: `44ee762e87641251c3cd0c64ce229960f3a7125b5b2c6a9ec5d3c0dccdeb0b88`.
- B PTX SHA-256: `b2633905b799459640981dd7fa6ac326ae1b7cc9a080e33ca9530bfa0e163d47`.
- PTX comparison: `ptx_comparison/comparison.json`.
- PTX is not decoded SASS. Resource metadata alone does not prove a latency gain.

## Provenance and reproduction

GPU: NVIDIA GeForce RTX 4060 Ti; driver: 616.92.
Software: {"cuda_runtime": "12.8", "python": "3.13.11", "torch": "2.11.0+cu128", "triton": "3.6.0"}.
Source SHA-256: `a88226c46f335a9768f15b52eb275abf6066bfb57b1f7e012faa937a51768552`; Git: `f4a5860bea4bb1b929f1db50f81a5466c64d37c5` (dirty=True).
Run: `matmul-inspector-python experiment pair --spec examples/mi_research_4c_n512_pair.json --output artifacts/mi-research-4c/n512-eager-s1 --allow-trusted-local-source`.
A source: `sources/A.py`; entrypoint `matmul_into`; SHA-256 `f1f355254182f73195282921d437d6baa0a565b2994cc0e99d7e87266e2ae468`.
B source: `sources/B.py`; entrypoint `matmul_into`; SHA-256 `319c6cf6997807199d7b8f20b4971e6f1755a3bf6c764d1f801117af5139987c`.
Verify: `matmul-inspector-python experiment pair --verify artifacts/mi-research-4c/n512-eager-s1`.
Raw paired blocks, compiler files, source snapshot, and SHA-256 hashes are in this bundle.
