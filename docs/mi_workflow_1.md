# MI-Workflow-1 — one-command GEMM comparison

## Result

**Workflow decision A:** the MVP completed a real, bounded PyTorch/cuBLAS versus
Triton FC1 comparison on an RTX 4060 Ti. It saved numerical, raw timing,
compiler, source, and environment evidence in one verified bundle. The
**performance decision is INCONCLUSIVE**. The candidate passed correctness,
but the candidate timing varied and the existing timing-boundary controls
found a cross-timer discrepancy. No speedup or kernel optimization is claimed.

This is a tool for a human engineer or an external coding agent. It contains
no model call or autonomous candidate generation.

## Getting started

From an editable checkout with the project `.venv`, CUDA-enabled PyTorch,
Triton, and an NVIDIA GPU, copy
[`mi_workflow_1_fc1.json`](../examples/mi_workflow_1_fc1.json) and edit the
shape, seed, tolerances, or bounded benchmark counts. The v1 spec allows only
the registered `triton_fc1` candidate and the `pytorch_matmul` baseline. It
uses FP16 and eager allocating execution. The exact format is in
[`mi_workflow_1_experiment_v1.schema.json`](../schemas/mi_workflow_1_experiment_v1.schema.json).
The command rejects unknown fields, untrusted source paths, unsupported modes,
nonfinite tolerances, oversized shapes, and out-of-budget measurement counts.
It also requires K to be divisible by the candidate's `block_k`.

Run from the repository root with a new output path:

```bash
.venv/bin/matmul-inspector-python experiment compare \
  --spec examples/mi_workflow_1_fc1.json \
  --output results/my-fc1-comparison
```

Read `report.md` first. Read `correctness.json`, `comparison.json`, and
`timings.jsonl` for the machine-readable facts. Verify the complete bundle
without CUDA or a model API:

```bash
.venv/bin/matmul-inspector-python experiment compare \
  --verify results/my-fc1-comparison
```

The output path must not exist. The command reuses the established experiment
worker, including its timeout and stage failures. It seeds identical inputs,
checks the Triton output against the PyTorch reference, and only times a
correct candidate. Thirty warmups precede eight alternating-order rounds of
100 operations each in the example. The underlying worker records one
unranked boundary-control block per implementation. Setup and first
synchronized invocation are outside ranked timing.

The v1 bundle has these stable files:

| File | Meaning |
|---|---|
| `experiment.json` | Validated v1 settings and fixed implementation identities |
| `correctness.json` | Shape, dtype, finiteness, error, tolerances, and pass/fail |
| `timings.jsonl` | One raw CUDA-event and host-clock record per implementation per round |
| `comparison.json` | Recomputed decision, medians, ranges, CVs, and paired differences |
| `report.md` | Short human-readable result |
| `provenance.json` | Command, hardware/software, Git, driver, source hash, raw-run path |
| `compiler.json` | Candidate compilation metadata and PTX/SASS capture statuses |
| `candidate_source.py` | Byte-exact packaged Triton kernel source used for the comparison |
| `candidate.ptx`, `candidate.sass` | Present only when that compiler text is available |
| `raw/<run-id>/` | Unmodified established runner result, stage log, and timing files |
| `hash_manifest.json` | SHA-256 of every other bundle file |

The decision fields have a versioned
[`comparison v1 schema`](../schemas/mi_workflow_1_comparison_v1.schema.json).
`schema_version` is 1 in the spec and comparison. The verifier checks all
file hashes, the spec, correctness, raw timing rows, source identity, and a
recomputed decision. An external coding agent should verify first, reject a
failed correctness record, then inspect the decision and raw rounds. For an
inconclusive result, it should investigate measurement stability or compiler
evidence before proposing a performance claim.

## Retained RTX 4060 Ti comparison

The exact run used M=512, K=768, N=3072, FP16, seed 0, `atol=rtol=0.01`,
30 warmups, 100 iterations per block, and eight alternating rounds. The
versioned [`fc1-rtx4060ti-v1`](../artifacts/mi-workflow-1/fc1-rtx4060ti-v1/)
bundle retains all eight pairs and the original worker output. The hardware
was an NVIDIA GeForce RTX 4060 Ti with CUDA runtime 12.8 and driver 616.92;
PyTorch was 2.11.0+cu128 and Triton was 3.6.0. These are the recorded run
versions, not requirements for other machines.

Correctness passed: matching FP16 shape and dtype, no nonfinite values, and
0.0 maximum and mean absolute error on the seeded input. This is a finite
test, not a proof for every input.

| CUDA-event result | Baseline | Triton candidate |
|---|---:|---:|
| Median ms/operation | 0.064718 | 0.071336 |
| Range ms/operation | 0.064563–0.069427 | 0.067633–0.085778 |
| Coefficient of variation | 2.77% | 10.72% |

The candidate was slower in all eight paired rounds. The ranges overlap and
the candidate had visibly higher dispersion. More critically, the unranked
boundary controls recorded CUDA-event block times 0.363 ms above synchronized
host-wall time for the baseline and 0.443 ms above it for the candidate. The
existing runner records these as cross-timer discrepancies without assigning
a cause. The audited comparison therefore withholds a speedup ratio and any
direction claim. It is `INCONCLUSIVE`, not a demonstrated optimization or a
reliable regression estimate. CUDA-event intervals can include host-induced
idle gaps and the allocating wrapper includes output allocation per call.

The first synchronized invocation took 216.39 ms for the baseline and
365.32 ms for the candidate in host wall time. These include setup and
possible compilation. They are separate from steady-state event timing.
The compiler capture saved Triton PTX 8.7 for `sm_89`, a `fc1_kernel` entry,
and Triton metadata reporting four warps, three stages, and 24,576 bytes of
shared memory. This is PTX and compiler metadata. Decoded SASS and cuBLAS
machine code were unavailable; no CUBIN or Nsight profile was required.

The initial derived report identified overlapping timing ranges. The final
audit added the boundary-discrepancy gate to the decision calculation. The
original worker files and raw measurements remain byte-identical to the first
run in `/tmp/mi-workflow-1-fc1-v1`. The retained provenance records the
original bundle manifest and original comparison hashes. The final manifest
hashes the revised derived report and every preserved raw file.

## Limits and next investigation

The MVP compares one registered FP16 GEMM pair in eager allocating mode. It
does not accept arbitrary Python kernels, cover other dtypes or CUDA Graphs,
decode SASS without available tooling, or isolate pure device execution from
host submission. One bounded run cannot establish a durable speedup or
slowdown. A later study could isolate the timing discrepancy and compare a
preallocated mode under matched controls. That work is outside this milestone.
