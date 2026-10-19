# MI-Dogfood-1 — using Matmul Inspector for one FC1 tuning attempt

## Decision and scope

**Decision B: useful evidence capture, no demonstrated optimization.** I used
Matmul Inspector as an external kernel-performance engineer would: one fresh
baseline run, one parameter change, one candidate run, and a saved PTX
comparison. Both GEMMs passed numerical correctness. The intended resource
change appeared in compiler metadata. Timing remained inconclusive, so the
optimization hypothesis is **not supported by performance evidence**.

This was a developer-experience evaluation, not a new runner or agent. No
kernel source, benchmark loop, correctness tolerance, or workflow code was
changed. No follow-up GPU run was made because both primary runs triggered
the same timing-boundary limitation; another run without isolating that
limitation would not resolve the comparison.

## Hypothesis and controlled change

The workload is FP16 `(512×768) @ (768×3072)` on an RTX 4060 Ti. The
registered Triton kernel uses `64×128×32` tiles, four warps, and three
pipeline stages. Its fresh baseline compiler capture reported 24,576 bytes
of shared memory per program. **Hypothesis, stated before changing the
candidate:** reducing `num_stages` from 3 to 2 might lower shared-memory use
and allow more concurrent programs. It might also reduce latency hiding and
hurt performance.

The checked-in [candidate spec](../examples/mi_dogfood_1_stages2.json) changes
only `candidate.parameters.num_stages` from 3 to 2 relative to the unchanged
[baseline spec](../examples/mi_workflow_1_fc1.json). A CPU check confirmed
that every other field matches: shape, seed, tolerances, warmups, iterations,
rounds, timeout, implementation, and tile geometry. The kernel source hash
is identical in both retained bundles.

## Commands and retained evidence

The exact GPU commands were run once each from the repository root:

```bash
.venv/bin/matmul-inspector-python experiment compare \
  --spec examples/mi_workflow_1_fc1.json \
  --output /tmp/mi-dogfood-1-baseline-v1

.venv/bin/matmul-inspector-python experiment compare \
  --spec examples/mi_dogfood_1_stages2.json \
  --output /tmp/mi-dogfood-1-stages2-v1
```

The complete bundles were copied byte-for-byte to the versioned
[`baseline-v1`](../artifacts/mi-dogfood-1/baseline-v1/) and
[`stages2-v1`](../artifacts/mi-dogfood-1/stages2-v1/) directories. Each has
its own SHA-256 manifest, validated spec, correctness result, 16 raw timing
records, comparison and report, provenance, PTX, source snapshot, and
unmodified underlying experiment-worker files. The offline verifier passes
for both:

```bash
.venv/bin/matmul-inspector-python experiment compare \
  --verify artifacts/mi-dogfood-1/baseline-v1
.venv/bin/matmul-inspector-python experiment compare \
  --verify artifacts/mi-dogfood-1/stages2-v1
```

I also ran the existing static PTX comparison on the two retained PTX files:

```bash
.venv/bin/matmul-inspector-python ptx compare \
  --baseline artifacts/mi-dogfood-1/baseline-v1/candidate.ptx \
  --candidate artifacts/mi-dogfood-1/stages2-v1/candidate.ptx \
  --kernel fc1_kernel \
  --output artifacts/mi-dogfood-1/ptx-comparison-v1
```

Its [report](../artifacts/mi-dogfood-1/ptx-comparison-v1/report.md), JSON,
normalized PTX files, and diff have a separate five-file hash manifest. This
command is offline and did not benchmark or execute a kernel.

Manifest SHA-256 values are `6b8b2d5596e9b0e3a3e1a775f6665dce2a1b84b3690e9b5e448f9b1151ce6bb3`
for the baseline, `9c3215365340d9b49c86a9d8fd1c0a0d96cd2f22c20b15adcb1c3db1db2d5545`
for the candidate, and `673007af90e13b280dbfebb72bf65b20e96c8ac7c28a9db1522aa86625f716ce`
for the PTX comparison. Both GPU bundles record RTX 4060 Ti, driver 616.92,
and the same packaged source SHA-256
`a88226c46f335a9768f15b52eb275abf6066bfb57b1f7e012faa937a51768552`.

## Correctness, timing, and uncertainty

Both runs used the same seeded FP16 input and the unchanged `atol=rtol=0.01`
test against PyTorch/cuBLAS. Both passed shape, dtype, finiteness, and
`torch.allclose`. Maximum and mean absolute error were 0.0 in each saved
correctness record. This is a finite-input check, not universal correctness.

Each run warmed both implementations 30 times, then timed 100 operations per
block across eight alternating-order rounds. CUDA-event time, host submission
time, synchronized host-wall time, and paired order remain in the raw files.

| Run and implementation | Median ms/op | Range ms/op | CV |
|---|---:|---:|---:|
| Baseline run: PyTorch/cuBLAS | 0.066404 | 0.064512–0.070114 | 2.98% |
| Baseline run: Triton, 3 stages | 0.071121 | 0.069949–0.075069 | 2.29% |
| Candidate run: PyTorch/cuBLAS | 0.064538 | 0.064307–0.070748 | 3.37% |
| Candidate run: Triton, 2 stages | 0.072435 | 0.067141–0.074803 | 3.65% |

The 2-stage Triton median was 1.85% higher than the 3-stage Triton median
across *separate* runs. The cuBLAS median shifted by 2.81% between those
runs. Neither number is a paired Triton-vs-Triton effect estimate. Each
workflow run pairs Triton with cuBLAS, not the two Triton settings with each
other. The stage-2 and stage-3 timing ranges overlap.

More importantly, the unranked control blocks in both runs recorded CUDA
event time above synchronized host-wall time: +0.362 and +0.381 ms for the
baseline run's cuBLAS and Triton blocks; +0.306 and +0.342 ms for the
candidate run's blocks. Matmul Inspector classified both runs as
`INCONCLUSIVE` and withheld speedup ratios. The discrepancy is observed; its
cause was not isolated. I did not override that gate or infer a winner from
the medians.

## Compiler evidence

Both captures used PTX 8.7 targeting `sm_89` and the same packaged Triton
source. Decoded SASS was unavailable. Triton reported shared memory falling
from 24,576 to 12,288 bytes per program with the stage change. The PTX
inspector found normalized PTX differences and 408 versus 373 static
instructions in the selected entry. `cp.async.cg.shared.global` went from
18 to 12; tensor instructions stayed at 32. PTX-declared `b32` registers
went from 318 to 303. These are source/IR and compiler-metadata observations.
They do not prove physical occupancy, final SASS changes, dynamic instruction
counts, or a runtime benefit.

## Developer friction log

| What I had to do | Type | Effect |
|---|---|---|
| Copy the full JSON spec to vary one candidate parameter | Workflow overhead | No extra code was needed, but the command has no small parameter override or named variant. |
| Run each Triton setting in a separate cuBLAS comparison | Missing functionality | The saved rounds cannot answer a directly paired 3-stage-versus-2-stage question. |
| Open `comparison.json`, `compiler.json`, and the nested raw `result.json`, then use Python one-liners to extract the boundary deltas, PTX counts, and cross-run medians | Missing presentation | The files contain the facts, but no single side-by-side study view assembles them. |
| Invoke `--verify` once per bundle and `ptx compare` separately, then add a hash manifest to its output | Workflow overhead | Evidence checks work, but the PTX comparison is not linked automatically to the benchmark bundles. |
| Run `ptx compare` again after copying the bundles so its report pointed to versioned paths | Workflow overhead | The first report embedded temporary paths. The second run did no GPU work, but was needed for a self-contained study. |
| Read the workflow guide and command help to find the PTX comparison invocation; its help displays `matmul-inspector-python compare` rather than the full `ptx compare` path | Help-text defect | The command worked; the displayed usage was confusing. |
| Use elevated GPU access in this Codex sandbox | Environment constraint | The default sandbox could not access CUDA. This was not a Matmul Inspector defect. |

The tool successfully automated spec validation, deterministic inputs,
correctness gating, warmups, alternating CUDA-event measurement, host timing,
raw artifact retention, compiler PTX capture, provenance, reporting, and hash
verification. I wrote no new benchmark loop or kernel wrapper.

## Three developer-experience improvements worth prioritizing

1. Add a direct, paired configuration A/B comparison that alternates the two
   Triton settings in one controlled run and keeps the cuBLAS reference.
2. Allow an explicitly trusted local candidate source through this simple
   comparison path, using the existing source snapshot and validation gates.
   The current v1 command is limited to the registered kernel and its parameters.
3. Produce one concise cross-run view with raw timing stability, boundary
   discrepancies, compiler resource changes, and linked PTX evidence. Keep
   the underlying JSON and hashes for auditing.

These are recommendations only. No new framework feature was implemented in
this dogfood study.
