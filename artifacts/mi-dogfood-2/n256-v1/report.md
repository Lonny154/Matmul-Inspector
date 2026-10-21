# Paired FC1 A/B comparison

FP16 M=512, K=768, N=3072; eager preallocated mode.
Independent reference: PyTorch/cuBLAS `torch.matmul`.

- A: `{"block_k": 32, "block_m": 64, "block_n": 128, "group_size_m": 1, "num_stages": 3, "num_warps": 4}`; correctness PASS; max absolute error 0.0.
- B: `{"block_k": 32, "block_m": 64, "block_n": 256, "group_size_m": 1, "num_stages": 3, "num_warps": 4}`; correctness PASS; max absolute error 0.0.
- Additional seed 1: A PASS, B PASS.
- Additional seed 2: A PASS, B PASS.

**Decision: INCONCLUSIVE** — CUDA-event time exceeded synchronized host wall in a round or control

- A: CUDA-event median 0.072374 ms/op (range 0.064635–0.075520, CV 5.10%); host-wall median 0.065767 ms/op.
- B: CUDA-event median 0.069056 ms/op (range 0.065495–0.077668, CV 6.71%); host-wall median 0.062958 ms/op.
- Paired A−B: median 0.001147 ms/op; range -0.007556–0.007502; MAD 0.004652 ms/op.
- Paired signs: {'A_faster': 3, 'B_faster': 5, 'ties': 0}. Timer discrepancy: True.

| Round | CUDA-event A−B (ms/op) | Host-wall A−B (ms/op) |
|---:|---:|---:|
| 1 | +0.004752 | +0.007336 |
| 2 | +0.006844 | +0.006216 |
| 3 | +0.000070 | -0.000094 |
| 4 | +0.002223 | +0.002103 |
| 5 | -0.007556 | -0.006914 |
| 6 | -0.007548 | -0.007523 |
| 7 | +0.007502 | +0.006712 |
| 8 | -0.000860 | +0.000875 |

- A event−wall control: +0.646370 ms/block; round range +0.347805 to +0.676797 ms/block.
- B event−wall control: +0.634386 ms/block; round range +0.586732 to +0.683270 ms/block.

- A compiler: shared=24576 bytes; registers/thread=121; spills=0; warps=4; stages=3; PTX=CAPTURED; decoded SASS=UNAVAILABLE.
- B compiler: shared=40960 bytes; registers/thread=214; spills=0; warps=4; stages=3; PTX=CAPTURED; decoded SASS=UNAVAILABLE.
- Compiler shared-memory B−A: +16384 bytes per program.
- Compiler registers/thread B−A: +93.
- Normalized PTX equal: False; changed lines: 1485.
- A PTX SHA-256: `16be26977e1fc6b029cdaac1df5870f307b4e1e9f63f0240ec223f04f4fda668`.
- B PTX SHA-256: `e5340d1c43dad12b0acdcc0f0a31238ec3b3f06f66c77ae9ae9830050b10897d`.
- PTX comparison: `ptx_comparison/comparison.json`.
- PTX is not decoded SASS. Resource metadata alone does not prove a latency gain.

## Provenance and reproduction

GPU: NVIDIA GeForce RTX 4060 Ti; driver: 616.92.
Software: {"cuda_runtime": "12.8", "python": "3.13.11", "torch": "2.11.0+cu128", "triton": "3.6.0"}.
Source SHA-256: `a88226c46f335a9768f15b52eb275abf6066bfb57b1f7e012faa937a51768552`; Git: `04aa1e571e7e23c5d9ed11a4f47362af8e8c24c5` (dirty=True).
Run: `matmul-inspector-python experiment pair --spec examples/mi_dogfood_2_n256.json --output artifacts/mi-dogfood-2/n256-v1 --allow-trusted-local-source`.
A source: `candidate_source.py`; entrypoint `fc1_kernel`; SHA-256 `a88226c46f335a9768f15b52eb275abf6066bfb57b1f7e012faa937a51768552`.
B source: `sources/B.py`; entrypoint `matmul_into`; SHA-256 `c72b1bab91a071d32f33b002bb88a9c22a041a1309138c0b9eb53bf2574c61ec`.
Verify: `matmul-inspector-python experiment pair --verify artifacts/mi-dogfood-2/n256-v1`.
Raw paired blocks, compiler files, source snapshot, and SHA-256 hashes are in this bundle.
