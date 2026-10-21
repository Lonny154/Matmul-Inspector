# Paired FC1 A/B comparison

FP16 M=512, K=768, N=3072; eager preallocated mode.
Independent reference: PyTorch/cuBLAS `torch.matmul`.

- A: `{"block_k": 32, "block_m": 64, "block_n": 128, "group_size_m": 1, "num_stages": 3, "num_warps": 4}`; correctness PASS; max absolute error 0.0.
- B: `{"block_k": 32, "block_m": 64, "block_n": 128, "group_size_m": 1, "num_stages": 2, "num_warps": 4}`; correctness PASS; max absolute error 0.0.
- Additional seed 1: A PASS, B PASS.
- Additional seed 2: A PASS, B PASS.

**Decision: INCONCLUSIVE** — an A/B paired gap does not exceed the largest observed A/A gap

- A: CUDA-event median 0.063466 ms/op (range 0.059556–0.066998, CV 4.45%); host-wall median 0.060586 ms/op.
- B: CUDA-event median 0.061648 ms/op (range 0.059474–0.070583, CV 6.12%); host-wall median 0.058878 ms/op.
- Paired A−B: median 0.000067 ms/op; range -0.007108–0.003799; MAD 0.000737 ms/op.
- Paired signs: {'A_faster': 4, 'B_faster': 4, 'ties': 0}. Timer discrepancy: True.

| Round | CUDA-event A−B (ms/op) | Host-wall A−B (ms/op) |
|---:|---:|---:|
| 1 | -0.000194 | -0.000163 |
| 2 | +0.003422 | +0.003260 |
| 3 | -0.007108 | -0.006785 |
| 4 | +0.001281 | +0.001248 |
| 5 | -0.000082 | -0.000117 |
| 6 | -0.000061 | -0.000078 |
| 7 | +0.003799 | +0.003545 |
| 8 | +0.000195 | +0.000201 |

- A event−wall control: +0.264763 ms/block; round range +0.257282 to +0.311242 ms/block.
- B event−wall control: +0.278591 ms/block; round range +0.253794 to +0.320609 ms/block.

## Clock-aware diagnostics (v3)

The event interval brackets the configured calls on one CUDA stream. The perf and RAW host intervals enclose end-event synchronization. Perf is sampled just outside RAW, so the host windows differ by timestamp-read overhead. No clock is rescaled or treated as a universal absolute reference.

- Host-clock preflight: RATE_DIFFERENCE_OBSERVED; median RAW/perf rate ratio 1.0545223757615674; five 20 ms windows when RAW is available.
- Same-kernel A/A: 8 balanced rounds; event X−Y median +0.000019 ms/op; maximum absolute paired gap 0.006728 ms/op.
- A median ms/op: CUDA event 0.063466; perf_counter wall 0.060586; MONOTONIC_RAW 0.063888.
- B median ms/op: CUDA event 0.061648; perf_counter wall 0.058878; MONOTONIC_RAW 0.062085.
- RAW enclosing-boundary discrepancy: False; A/B exceeds observed A/A gap: False.

| Round | Event A−B | perf A−B | RAW A−B | A/A event X−Y |
|---:|---:|---:|---:|---:|
| 1 | -0.000194 | -0.000163 | -0.000171 | -0.001294 |
| 2 | +0.003422 | +0.003260 | +0.003438 | -0.000000 |
| 3 | -0.007108 | -0.006785 | -0.007154 | +0.000195 |
| 4 | +0.001281 | +0.001248 | +0.001312 | -0.000082 |
| 5 | -0.000082 | -0.000117 | -0.000126 | +0.000009 |
| 6 | -0.000061 | -0.000078 | -0.000042 | +0.006728 |
| 7 | +0.003799 | +0.003545 | +0.003735 | +0.000030 |
| 8 | +0.000195 | +0.000201 | +0.000208 | +0.000072 |

All table differences are ms per operation; raw block totals and operation counts are retained in `raw_result.json`.

- Warnings: HOST_CLOCK_RATE_DIFFERENCE, CUDA_EVENT_EXCEEDS_PERF_COUNTER_WALL, AB_WITHIN_AA_VARIABILITY.
- A/A is a local variability control, not a confidence interval. A supported ranking requires correctness, complete controls, event range separation, both host paired directions, and each A/B gap beyond the largest A/A gap. A RAW rate difference can explain a perf boundary reversal but cannot prove RAW absolute accuracy.

- A compiler: shared=24576 bytes; registers/thread=121; spills=0; warps=4; stages=3; PTX=CAPTURED; decoded SASS=UNAVAILABLE.
- B compiler: shared=12288 bytes; registers/thread=122; spills=0; warps=4; stages=2; PTX=CAPTURED; decoded SASS=UNAVAILABLE.
- Compiler shared-memory B−A: -12288 bytes per program.
- Compiler registers/thread B−A: +1.
- Normalized PTX equal: False; changed lines: 1037.
- A PTX SHA-256: `16be26977e1fc6b029cdaac1df5870f307b4e1e9f63f0240ec223f04f4fda668`.
- B PTX SHA-256: `bd78cbe7c66d18dc39550144102775bd80deb973a732b26d5678b8b3c927e9cd`.
- PTX comparison: `ptx_comparison/comparison.json`.
- PTX is not decoded SASS. Resource metadata alone does not prove a latency gain.

## Provenance and reproduction

GPU: NVIDIA GeForce RTX 4060 Ti; driver: 616.92.
Software: {"cuda_runtime": "12.8", "python": "3.13.11", "torch": "2.11.0+cu128", "triton": "3.6.0"}.
Source SHA-256: `a88226c46f335a9768f15b52eb275abf6066bfb57b1f7e012faa937a51768552`; Git: `68b66e501d2d5f304ac2ac85eb2fd24f95123230` (dirty=True).
Run: `matmul-inspector-python experiment pair --spec examples/mi_timing_2_fc1_pair.json --output artifacts/mi-timing-2/fc1-pair-v1`.
Verify: `matmul-inspector-python experiment pair --verify artifacts/mi-timing-2/fc1-pair-v1`.
Raw paired blocks, compiler files, source snapshot, and SHA-256 hashes are in this bundle.
