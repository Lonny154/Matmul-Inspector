# Paired FC1 A/B comparison

FP16 M=512, K=768, N=3072; eager allocating mode.
Independent reference: PyTorch/cuBLAS `torch.matmul`.

- A: `{"block_k": 32, "block_m": 64, "block_n": 128, "group_size_m": 1, "num_stages": 3, "num_warps": 4}`; correctness PASS; max absolute error 0.0.
- B: `{"block_k": 32, "block_m": 64, "block_n": 128, "group_size_m": 1, "num_stages": 2, "num_warps": 4}`; correctness PASS; max absolute error 0.0.

**Decision: INCONCLUSIVE** — CUDA-event time exceeded synchronized host wall in a round or control

- A: CUDA-event median 0.063937 ms/op (range 0.063556–0.066813, CV 1.87%); host-wall median 0.060754 ms/op.
- B: CUDA-event median 0.065961 ms/op (range 0.063631–0.070144, CV 3.38%); host-wall median 0.062622 ms/op.
- Paired A−B: median -0.001168 ms/op; range -0.006588–0.003176; MAD 0.001501 ms/op.
- Paired signs: {'A_faster': 6, 'B_faster': 2, 'ties': 0}. Timer discrepancy: True.

| Round | CUDA-event A−B (ms/op) | Host-wall A−B (ms/op) |
|---:|---:|---:|
| 1 | -0.003205 | -0.002946 |
| 2 | -0.000133 | -0.000053 |
| 3 | +0.003176 | +0.003111 |
| 4 | -0.000074 | -0.000186 |
| 5 | -0.006588 | -0.006214 |
| 6 | +0.000233 | +0.000333 |
| 7 | -0.002202 | -0.002097 |
| 8 | -0.002770 | -0.002645 |

- A event−wall control: +0.332297 ms/block; round range +0.309125 to +0.335113 ms/block.
- B event−wall control: +0.230378 ms/block; round range +0.319209 to +0.368814 ms/block.

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
Software: {'cuda_runtime': '12.8', 'python': '3.13.11', 'torch': '2.11.0+cu128', 'triton': '3.6.0'}.
Source SHA-256: `a88226c46f335a9768f15b52eb275abf6066bfb57b1f7e012faa937a51768552`; Git: `5544b198e1a08e93cd8cc13d2f028b2a7a53a8be` (dirty=True).
Run: `matmul-inspector-python experiment pair --spec examples/mi_workflow_2_fc1_pair.json --output artifacts/mi-workflow-2/fc1-pair-v1`.
Verify: `matmul-inspector-python experiment pair --verify artifacts/mi-workflow-2/fc1-pair-v1`.
Raw paired blocks, compiler files, source snapshot, and SHA-256 hashes are in this bundle.
