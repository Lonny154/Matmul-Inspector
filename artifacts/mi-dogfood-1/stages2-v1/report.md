# GEMM comparison

Shape: M=512, N=3072, K=768; float16; eager CUDA.
Baseline: PyTorch/cuBLAS `torch.matmul`.
Candidate: packaged Triton `triton_fc1.fc1_kernel`.

Correctness: PASS
Decision: **INCONCLUSIVE** — unranked timing controls found a CUDA-event/host-wall discrepancy

Baseline median: 0.064538 ms/op.
Candidate median: 0.072435 ms/op.
Baseline CV: 3.37%; candidate CV: 3.65%.

CUDA-event intervals include dispatch and possible device idle gaps.
First synchronized invocation is reported separately in the preserved raw run.
Compiler: PTX CAPTURED; SASS UNAVAILABLE; cuBLAS machine code unavailable.
Raw round timings and source hashes are in this bundle.
