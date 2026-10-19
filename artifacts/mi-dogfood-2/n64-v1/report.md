# Paired FC1 A/B comparison

FP16 M=512, K=768, N=3072; eager preallocated mode.
Independent reference: PyTorch/cuBLAS `torch.matmul`.

- A: `{"block_k": 32, "block_m": 64, "block_n": 128, "group_size_m": 1, "num_stages": 3, "num_warps": 4}`; correctness PASS; max absolute error 0.0.
- B: `{"block_k": 32, "block_m": 64, "block_n": 64, "group_size_m": 1, "num_stages": 3, "num_warps": 4}`; correctness PASS; max absolute error 0.0.
- Additional seed 1: A PASS, B PASS.
- Additional seed 2: A PASS, B PASS.

**Decision: INCONCLUSIVE** — CUDA-event time exceeded synchronized host wall in a round or control

- A: CUDA-event median 0.063891 ms/op (range 0.063345–0.073878, CV 5.52%); host-wall median 0.062223 ms/op.
- B: CUDA-event median 0.069746 ms/op (range 0.064521–0.073615, CV 4.62%); host-wall median 0.067664 ms/op.
- Paired A−B: median -0.004295 ms/op; range -0.009728–0.004055; MAD 0.003845 ms/op.
- Paired signs: {'A_faster': 7, 'B_faster': 1, 'ties': 0}. Timer discrepancy: True.

| Round | CUDA-event A−B (ms/op) | Host-wall A−B (ms/op) |
|---:|---:|---:|
| 1 | -0.000626 | -0.000559 |
| 2 | -0.008316 | -0.008000 |
| 3 | -0.002925 | -0.002755 |
| 4 | -0.009134 | -0.007193 |
| 5 | -0.009728 | -0.009332 |
| 6 | -0.002260 | -0.001477 |
| 7 | +0.004055 | +0.003814 |
| 8 | -0.005664 | -0.005459 |

- A event−wall control: +0.168966 ms/block; round range +0.016210 to +0.232279 ms/block.
- B event−wall control: +0.208359 ms/block; round range +0.184453 to +0.223479 ms/block.

- A compiler: shared=24576 bytes; registers/thread=121; spills=0; warps=4; stages=3; PTX=CAPTURED; decoded SASS=UNAVAILABLE.
- B compiler: shared=16384 bytes; registers/thread=79; spills=0; warps=4; stages=3; PTX=CAPTURED; decoded SASS=UNAVAILABLE.
- Compiler shared-memory B−A: -8192 bytes per program.
- Compiler registers/thread B−A: -42.
- Normalized PTX equal: False; changed lines: 946.
- A PTX SHA-256: `16be26977e1fc6b029cdaac1df5870f307b4e1e9f63f0240ec223f04f4fda668`.
- B PTX SHA-256: `157178e5f556723a4134f9562174ddacded2bfa5842258c33ce32ff2d3c3039b`.
- PTX comparison: `ptx_comparison/comparison.json`.
- PTX is not decoded SASS. Resource metadata alone does not prove a latency gain.

## Provenance and reproduction

GPU: NVIDIA GeForce RTX 4060 Ti; driver: 616.92.
Software: {"cuda_runtime": "12.8", "python": "3.13.11", "torch": "2.11.0+cu128", "triton": "3.6.0"}.
Source SHA-256: `a88226c46f335a9768f15b52eb275abf6066bfb57b1f7e012faa937a51768552`; Git: `04aa1e571e7e23c5d9ed11a4f47362af8e8c24c5` (dirty=True).
Run: `matmul-inspector-python experiment pair --spec examples/mi_dogfood_2_n64.json --output artifacts/mi-dogfood-2/n64-v1 --allow-trusted-local-source`.
A source: `candidate_source.py`; entrypoint `fc1_kernel`; SHA-256 `a88226c46f335a9768f15b52eb275abf6066bfb57b1f7e012faa937a51768552`.
B source: `sources/B.py`; entrypoint `matmul_into`; SHA-256 `c72b1bab91a071d32f33b002bb88a9c22a041a1309138c0b9eb53bf2574c61ec`.
Verify: `matmul-inspector-python experiment pair --verify artifacts/mi-dogfood-2/n64-v1`.
Raw paired blocks, compiler files, source snapshot, and SHA-256 hashes are in this bundle.
