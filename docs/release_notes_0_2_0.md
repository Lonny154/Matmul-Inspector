# Matmul Inspector 0.2.0 release notes

Matmul Inspector 0.2.0 adds the experiment infrastructure exercised by the
AI-assisted FC1 case study to the installable Python distribution. It remains an
alpha release: measurements are hardware- and software-specific, and small timing
differences are not optimization evidence by themselves.

## Added

- Reproducible single PyTorch/cuBLAS-versus-Triton experiments with independent
  correctness validation and persisted raw measurements.
- Sequential batch experiments with reproducible scheduling and descriptive
  aggregation.
- Persistent experiment sessions with immutable settings, duplicate detection,
  explicit repeat reasons, recovery, and enforced budgets.
- Separate allocating and preallocated benchmark modes. Results record CUDA-event,
  host-submission, and synchronized host-wall boundaries rather than labeling all
  measurements as pure kernel execution time.
- Explicitly opted-in trusted local Triton source candidates. Candidate bytes are
  hashed and preserved before isolated execution.
- The established registered FC1 Triton kernel as an installed package module, so
  GPU experiment execution no longer depends on `scripts/triton_fc1.py` in a
  repository checkout.

## Installation boundaries

CPU artifact analysis remains standard-library-only and requires no accelerator:

```sh
python -m pip install matmul-inspector==0.2.0
```

GPU experiments additionally require a compatible NVIDIA GPU and driver,
CUDA-enabled PyTorch, and Triton. After following PyTorch's platform-specific
installation instructions, install the optional dependency set with:

```sh
python -m pip install "matmul-inspector[gpu]==0.2.0"
```

The optional extra does not install or configure an NVIDIA driver and does not
guarantee compatibility among the host GPU, driver, CUDA runtime, PyTorch, and
Triton. Trusted source candidates execute local Python code and should not be used
with unreviewed files.

## Not included

The wheel does not contain the native C++/CUDA laboratory, repository examples,
saved experiment artifacts, compiler captures, or profiling results. Those remain
source-checkout resources. Version 0.2.0 does not claim a meaningful FC1 kernel
speedup; it releases the workflow used to produce and preserve the evidence.
