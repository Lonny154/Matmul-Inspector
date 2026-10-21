# Paired FC1 A/B comparison

FP16 M=512, K=768, N=3072; eager preallocated mode.
Independent reference: PyTorch/cuBLAS `torch.matmul`.

- A: `{"block_k": 32, "block_m": 64, "block_n": 128, "group_size_m": 1, "num_stages": 3, "num_warps": 4}`; correctness PASS; max absolute error 0.0.
- B: `{"block_k": 32, "block_m": 64, "block_n": 128, "group_size_m": 1, "num_stages": 3, "num_warps": 4}`; correctness PASS; max absolute error 0.0.
- Additional seed 1: A PASS, B PASS.
- Additional seed 2: A PASS, B PASS.

**Decision: INCONCLUSIVE** — CUDA-event time exceeded synchronized host wall in a round or control

- A: CUDA-event median 0.063646 ms/op (range 0.063334–0.068525, CV 2.80%); host-wall median 0.062231 ms/op.
- B: CUDA-event median 0.064020 ms/op (range 0.063459–0.068523, CV 2.79%); host-wall median 0.062644 ms/op.
- Paired A−B: median -0.000155 ms/op; range -0.005188–0.004504; MAD 0.001945 ms/op.
- Paired signs: {'A_faster': 4, 'B_faster': 4, 'ties': 0}. Timer discrepancy: True.

| Round | CUDA-event A−B (ms/op) | Host-wall A−B (ms/op) |
|---:|---:|---:|
| 1 | +0.004504 | +0.004318 |
| 2 | -0.000318 | -0.000261 |
| 3 | +0.002048 | +0.001964 |
| 4 | -0.003085 | -0.003034 |
| 5 | +0.000009 | -0.000022 |
| 6 | -0.005188 | -0.004056 |
| 7 | +0.000111 | +0.000082 |
| 8 | -0.001842 | -0.001673 |

- A event−wall control: +0.135485 ms/block; round range +0.057730 to +0.160739 ms/block.
- B event−wall control: +0.129513 ms/block; round range +0.122676 to +0.171005 ms/block.

- A compiler: shared=24576 bytes; registers/thread=121; spills=0; warps=4; stages=3; PTX=CAPTURED; decoded SASS=UNAVAILABLE.
- B compiler: shared=24576 bytes; registers/thread=121; spills=0; warps=4; stages=3; PTX=CAPTURED; decoded SASS=UNAVAILABLE.
- Compiler shared-memory B−A: +0 bytes per program.
- Compiler registers/thread B−A: +0.
- Normalized PTX equal: False; changed lines: 133.
- A PTX SHA-256: `16be26977e1fc6b029cdaac1df5870f307b4e1e9f63f0240ec223f04f4fda668`.
- B PTX SHA-256: `7fe82567dbfdce96297864f627e212c3811ec53c1c5f87618a270500b52e59e5`.
- PTX comparison: `ptx_comparison/comparison.json`.
- PTX is not decoded SASS. Resource metadata alone does not prove a latency gain.

## Provenance and reproduction

GPU: NVIDIA GeForce RTX 4060 Ti; driver: 616.92.
Software: {"cuda_runtime": "12.8", "python": "3.13.11", "torch": "2.11.0+cu128", "triton": "3.6.0"}.
Source SHA-256: `a88226c46f335a9768f15b52eb275abf6066bfb57b1f7e012faa937a51768552`; Git: `e85a536e32fa2301c0407c400995631fec81d1dc` (dirty=True).
Run: `matmul-inspector-python experiment pair --spec examples/mi_workflow_3_local_pair.json --output artifacts/mi-workflow-3/local-fc1-v1 --allow-trusted-local-source`.
A source: `candidate_source.py`; entrypoint `fc1_kernel`; SHA-256 `a88226c46f335a9768f15b52eb275abf6066bfb57b1f7e012faa937a51768552`.
B source: `sources/B.py`; entrypoint `matmul_into`; SHA-256 `9f6b6ecab30fcc7c75e683685896cc2e2d3b56ea6c5ac71408017da2b935aa73`.
Verify: `matmul-inspector-python experiment pair --verify artifacts/mi-workflow-3/local-fc1-v1`.
Raw paired blocks, compiler files, source snapshot, and SHA-256 hashes are in this bundle.
