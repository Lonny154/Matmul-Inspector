# MI-Research-3B — Large-M GEMM correctness investigation

## Plan recorded before new GPU work

MI-Research-3 stopped before timing M=256 and M=512 because both conventional
Triton and Split-2 failed the unchanged FP16 PyTorch/cuBLAS gate at seed 0.
M=32 and M=128 passed. The retained MI-Research-3 and 2C checksums pass.
This study tests whether the failure is an implementation defect, a graph
capture defect, or a numerical difference from the selected FP16 reference.
No performance measurements are planned.

Use the exact committed conventional and Split-2 kernels, `64×128×32` tiles,
K=4096, N=2048, and M=32,128,256,512. For each M, use the historical
shape-seeded input construction with seeds 0,1,2. Run eager A and B and
compare them to `torch.matmul` at the unchanged `atol=rtol=0.01`. At seed 0,
also capture one complete A and B operation into separate CUDA Graphs and
compare replay output to eager output and the same reference. Eager B uses
its original per-call FP32 scratch allocation; graph B uses stable scratch
and both original GPU kernels.

For every candidate/reference pair retain full error counts, fractions,
maximum and mean absolute error, finite checks, maximum relative error for
reference magnitudes at least one, and selected coordinates with values.
Record A/B failure-coordinate overlap and their numerical agreement.
For selected failures compute an FP64 dot product of the FP16 operands and
round to FP16. This is a local numerical oracle, not a replacement gate.
Also compare the original FP16 reference with a FP32-accumulation matmul
rounded to FP16 and, where supported, a diagnostic FP16 cuBLAS call with
reduced-precision reduction disabled. Keep these controls separate from
the established gate.

One controlled input family will hold B and the first 512 rows of A fixed.
Evaluate prefixes of that same A at each M, and compare overlapping output
coordinates across shapes. This avoids attributing differences from
independently drawn random tensors to M itself. Check row stride, output
coverage, masked tails, and split scratch initialization. No kernel or
tolerance will be edited unless a reproducible defect is demonstrated.

## Results

**Cause established: the default FP16 PyTorch/cuBLAS reference changes its
numerical result for the larger shapes under this environment's enabled
reduced-precision reduction setting.** The evidence does not show a
conventional or Split-2 kernel indexing defect. The original gate remains
unchanged and the larger shapes remain rejected under it.

The RTX 4060 Ti diagnostic used PyTorch `2.11.0+cu128`, Triton `3.6.0`,
CUDA runtime `12.8`, FP16 inputs and outputs, and seeds 0, 1, 2. PyTorch
reported `allow_fp16_reduced_precision_reduction=True` and
`allow_fp16_accumulation=False`; the FP32 diagnostic disabled TF32. Eager
Split-2 used the original allocating entrypoint. The graph diagnostic used
one complete partial-plus-reduction operation with fixed scratch. Both
Triton implementations passed at M=32/128 and failed at M=256/512 in eager
mode on all three seeds. The single-operation graph replay at seed 0 gave
the same pass/fail result and **bitwise identical** outputs to eager.

| M | Elements | A failing elements, seeds 0/1/2 | B failing elements, seeds 0/1/2 | Maximum absolute error |
| ---: | ---: | --- | --- | ---: |
| 32 | 65,536 | 0 / 0 / 0 | 0 / 0 / 0 | A 0; B 0.125 |
| 128 | 262,144 | 0 / 0 / 0 | 0 / 0 / 0 | A 0; B 0.125 |
| 256 | 524,288 | 1,229 / 1,222 / 1,248 | 1,228 / 1,219 / 1,254 | 0.25 |
| 512 | 1,048,576 | 810 / 829 / 813 | 806 / 815 / 808 | 0.25 |

At M=256, roughly 0.233–0.239% of outputs failed; at M=512,
0.077–0.079% failed. Both A and B had zero nonfinite outputs on every seed.
The maximum relative error for reference magnitudes at least one was
2.99–5.42% across larger-shape seeds. A and B passed the existing
`allclose` tolerance against each other at every shape and seed. Their
failure-coordinate Jaccard overlap was 0.969–0.980 at M=256 and
0.918–0.964 at M=512. They are not bitwise equal to each other; Split-2's
different accumulation order causes small differences. The high overlap
shows that the gate failures are mostly at the same coordinates.

At M=256 seed 0, the FP16 cuBLAS result differed from an FP32 matmul
rounded to FP16 at 1,225 coordinates under the **same unchanged** tolerance.
At M=512 it differed at 809. Conventional and Split-2 each had **zero**
failing coordinates against that diagnostic FP32-rounded result at both
shapes. When the diagnostic temporarily set
`allow_fp16_reduced_precision_reduction=False`, its FP16 cuBLAS output
also had zero failing coordinates against the FP32-rounded result and
passed the unchanged tolerance against A and B. At M=32/128, toggling
that setting did not change the cuBLAS output. This setting is the
controlled factor that explains the observed large-M reference mismatch
on this software and GPU configuration. It was restored after each check.

Selected FP64 dot products of the **FP16 operand values** independently
support the same interpretation. At M=256 seed 0, output `(0, 654)` had
FP64 dot 0.00831454, conventional 0.00861359, Split-2 0.00829315, and
default cuBLAS −0.00475311. At M=512 seed 0, `(0, 1204)` had FP64 dot
0.33132526, conventional 0.33105469, Split-2 0.33129883, and default
cuBLAS 0.31689453. Other representative coordinates and values remain in
`raw.json`. These selected dots do not certify every element, but they
rule out calling the default cuBLAS value the more accurate one at these
examples.

### Same inputs at every M

The controlled family used one 512-row A and one B matrix. Each smaller A
was an exact prefix view with the same 4,096-element row stride. A's
overlapping outputs were bitwise identical to its M=512 outputs at **all**
four M values; B's were also bitwise identical. The default FP16 cuBLAS
references disagreed with their own M=512 prefix under the unchanged
gate at 43, 166, and 1,366 coordinates for M=32, 128, and 256. This is
strong evidence for a shape-dependent reference path rather than output
indexing or masking in either Triton kernel. The graph Split-2 scratch
contained no NaNs or infinities after replay at any shape, and the stored
output exactly equaled `(partial[0] + partial[1]).half()` elementwise.
These checks support complete partial writes and deterministic reduction;
they do not prove every possible input or compiler configuration correct.

## Correction, limitations, and next step

No executable kernel or established tolerance was changed. The existing
FP16 PyTorch/cuBLAS gate is still the authority for the MI-Research-3
bundles, so M=256 and M=512 remain invalid for performance claims in that
study. Changing the reference policy in an experiment would create a new
correctness contract and require fresh correctness checks and fresh paired
performance measurements. Historical M=32 graph performance applies only
to its original kernel and setup; this study ran no performance sweep.

The first diagnostic attempt failed during graph setup because the
diagnostic wrapper captured `scratch=None`. It produced no study result.
The corrected wrapper binds allocated scratch before capture; both the
failed attempt and the first successful diagnostic are retained. A second
correctness-only diagnostic added explicit candidate-prefix comparisons.
The two successful bundles replay offline with their own source snapshots.

This investigation establishes the local failure mechanism, not an exact
cuBLAS implementation algorithm or a universal shape threshold. The
original Split-K crossover question **can be resumed only after an
explicit numerical-reference policy is chosen and the larger shapes pass
that policy**. It was not resumed here. The next evidence-backed step is
to define and review that reference policy, then perform a new bounded
shape comparison if desired.

## Reproduce and verify

```bash
.venv/bin/python scripts/run_mi_research_3b_correctness.py --output /tmp/mi-research-3b-correctness
.venv/bin/python scripts/run_mi_research_3b_correctness.py --verify --output artifacts/mi-research-3b/evidence-v1/diagnostic
.venv/bin/python scripts/run_mi_research_3b_correctness.py --verify --output artifacts/mi-research-3b/evidence-v1/diagnostic-v2
sha256sum -c artifacts/mi-research-3b/evidence-v1/SHA256SUMS
```

The diagnostic never measures latency. The versioned bundle contains
`raw.json`, deterministic `summary.json`, exact source and diagnostic
snapshots, per-run hash manifests, the first failed attempt, provenance,
and the top-level SHA-256 manifest. The MI-Research-3 evidence remains
unchanged.
