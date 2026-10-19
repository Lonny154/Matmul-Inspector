# Paired FC1 A/B comparison

FP16 M=512, K=768, N=3072; eager preallocated mode.
Independent reference: PyTorch/cuBLAS `torch.matmul`.

- A: `{"block_k": 32, "block_m": 64, "block_n": 128, "group_size_m": 1, "num_stages": 3, "num_warps": 4}`; correctness PASS; max absolute error 0.0.
- B: `{"block_k": 64, "block_m": 64, "block_n": 128, "group_size_m": 1, "num_stages": 3, "num_warps": 4}`; correctness PASS; max absolute error 0.0.
- Additional seed 1: A PASS, B PASS.
- Additional seed 2: A PASS, B PASS.

**Decision: INCONCLUSIVE** — CUDA-event time exceeded synchronized host wall in a round or control

- A: CUDA-event median 0.064204 ms/op (range 0.063508–0.068628, CV 3.11%); host-wall median 0.058686 ms/op.
- B: CUDA-event median 0.067932 ms/op (range 0.065495–0.070810, CV 2.94%); host-wall median 0.061964 ms/op.
- Paired A−B: median -0.001535 ms/op; range -0.007302–0.000666; MAD 0.002007 ms/op.
- Paired signs: {'A_faster': 6, 'B_faster': 2, 'ties': 0}. Timer discrepancy: True.

| Round | CUDA-event A−B (ms/op) | Host-wall A−B (ms/op) |
|---:|---:|---:|
| 1 | -0.001505 | -0.001347 |
| 2 | +0.000666 | +0.000688 |
| 3 | -0.005753 | -0.005248 |
| 4 | -0.001426 | -0.001267 |
| 5 | +0.000307 | -0.000043 |
| 6 | -0.003707 | -0.003717 |
| 7 | -0.007302 | -0.007232 |
| 8 | -0.001564 | -0.001391 |

- A event−wall control: +0.558632 ms/block; round range +0.528919 to +0.618650 ms/block.
- B event−wall control: +0.643733 ms/block; round range +0.546268 to +0.622729 ms/block.

- A compiler: shared=24576 bytes; registers/thread=121; spills=0; warps=4; stages=3; PTX=CAPTURED; decoded SASS=UNAVAILABLE.
- B compiler: shared=49152 bytes; registers/thread=160; spills=0; warps=4; stages=3; PTX=CAPTURED; decoded SASS=UNAVAILABLE.
- Compiler shared-memory B−A: +24576 bytes per program.
- Compiler registers/thread B−A: +39.
- Normalized PTX equal: False; changed lines: 1343.
- A PTX SHA-256: `16be26977e1fc6b029cdaac1df5870f307b4e1e9f63f0240ec223f04f4fda668`.
- B PTX SHA-256: `2a0555ec27e99968f61ca3c019408864ae936bee7aaf27fc3f0a6f5de6a136f2`.
- PTX comparison: `ptx_comparison/comparison.json`.
- PTX is not decoded SASS. Resource metadata alone does not prove a latency gain.

## Provenance and reproduction

GPU: NVIDIA GeForce RTX 4060 Ti; driver: 616.92.
Software: {"cuda_runtime": "12.8", "python": "3.13.11", "torch": "2.11.0+cu128", "triton": "3.6.0"}.
Source SHA-256: `a88226c46f335a9768f15b52eb275abf6066bfb57b1f7e012faa937a51768552`; Git: `04aa1e571e7e23c5d9ed11a4f47362af8e8c24c5` (dirty=True).
Run: `matmul-inspector-python experiment pair --spec examples/mi_dogfood_2_k64.json --output artifacts/mi-dogfood-2/k64-v1 --allow-trusted-local-source`.
A source: `candidate_source.py`; entrypoint `fc1_kernel`; SHA-256 `a88226c46f335a9768f15b52eb275abf6066bfb57b1f7e012faa937a51768552`.
B source: `sources/B.py`; entrypoint `matmul_into`; SHA-256 `c72b1bab91a071d32f33b002bb88a9c22a041a1309138c0b9eb53bf2574c61ec`.
Verify: `matmul-inspector-python experiment pair --verify artifacts/mi-dogfood-2/k64-v1`.
Raw paired blocks, compiler files, source snapshot, and SHA-256 hashes are in this bundle.
