# Triton FC1 evaluator

Milestone 2B.1 connects sampled candidates to the packaged Triton GEMM kernel
for the FP16 Transformer FC1 workload `M=512, N=3072, K=768`. It targets the
project's NVIDIA RTX 4060 Ti with compute capability 8.9. It does not implement
adaptive sampling and does not claim that a five-candidate sample finds a global
optimum.

## Prerequisites and execution

Use an NVIDIA system with compute capability 8.9, a compatible driver,
CUDA-enabled PyTorch, and Triton. GPU packages remain optional; importing the
rest of Matmul Inspector requires none of them.

From an installed checkout, run:

```sh
python examples/evaluate_triton_fc1_candidates.py
```

The example uses sampling seed 42 and input seed 0. It evaluates five unique
configurations and writes `triton_fc1_evaluations.jsonl`. Each line follows the
version-1 evaluation schema. Candidate parameters, numerical evidence, device
metadata, software versions, setup duration, raw timing samples, and host clocks
are stored in `extensions`.

The optional hardware test is explicitly gated:

```sh
MATMUL_INSPECTOR_RUN_CUDA_TESTS=1 \
  python -m pytest -q tests/triton_evaluator_test.py
```

## Correctness and timing

Inputs are deterministic contiguous FP16 CUDA tensors. The reference is
`torch.matmul` with a preallocated FP16 output, normally dispatched through
cuBLAS. The Triton kernel accumulates `tl.dot` into FP32 and stores FP16.
Correctness requires matching shape and dtype, finite values, and
`torch.allclose(atol=1e-2, rtol=1e-2)`. Maximum and mean absolute errors are
recorded. A failed correctness check is never benchmarked.

The first synchronized Triton invocation triggers compilation and first-use
setup outside the benchmark. Candidate output storage is reused. The default
benchmark performs 100 warmup launches, then collects 10 timing samples with
500 launches per sample. CUDA events bracket each launch block. The end event is
synchronized before querying elapsed time. The reported result is the median
CUDA-event latency per launch in microseconds. Raw CUDA-event samples remain in
the record. Host submission time and synchronized host-wall time are recorded
separately and are not mixed into the GPU-event statistic.

CUDA-event timing can still be affected by clock state, thermals, background
work, software versions, and launch scheduling. Repeated samples describe this
run only. The example does not measure a matched PyTorch baseline, so its output
must not be used to claim a speedup. It extends the prior manual FC1 autotuning
work by putting candidate identity, correctness gating, environment metadata,
and distributions into the shared evaluation contract. It does not replace the
controlled matched-baseline experiment runner.

Invalid workload, precision, backend, target, or K-blocking configurations fail
before hardware execution. Compilation, correctness, benchmark, and
infrastructure failures retain separate categories. Traceback text is stored in
extensions when an exception is available; no latency is assigned to failure
records.
