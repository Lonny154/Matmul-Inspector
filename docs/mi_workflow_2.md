# MI-Workflow-2 — paired FC1 kernel comparison

## Result

**Workflow decision A: the paired A/B path works and retains one auditable
report. Performance decision: INCONCLUSIVE.** One fresh RTX 4060 Ti run
compared the packaged FP16 FC1 Triton kernel at three stages (A) and two
stages (B). Both passed the PyTorch/cuBLAS numerical reference. The compiler
reported half as much shared memory for B. The eight paired timing rounds
disagreed on direction, and every round plus both unranked controls had
CUDA-event time above synchronized host-wall time. The timing evidence does
not support a winner.

The earlier [MI-Dogfood-1](mi_dogfood_1.md) runs were separate Triton-versus-
cuBLAS comparisons. Their medians are historical context, not a paired
three-stage-versus-two-stage result. This milestone leaves those bundles
unchanged.

## Shortest workflow

From the repository root, with the project `.venv` and a CUDA-enabled GPU:

```bash
.venv/bin/matmul-inspector-python experiment pair \
  --spec examples/mi_workflow_2_fc1_pair.json \
  --output results/my-fc1-pair
```

Use a new output path. Read `results/my-fc1-pair/report.md` for correctness,
paired timing, compiler metadata, PTX identities, and the decision. Verify
the complete bundle offline:

```bash
.venv/bin/matmul-inspector-python experiment pair \
  --verify results/my-fc1-pair
```

The [example spec](../examples/mi_workflow_2_fc1_pair.json) names both
configurations explicitly as `A` and `B`. Its [v1 schema](../schemas/mi_workflow_2_pair_v1.schema.json)
records the accepted fields. The Python validator also enforces finite
tolerances, K divisibility, distinct configurations, size limits, and the
same warmup, iteration, round, and timeout limits as MI-Workflow-1. This
initial path accepts the registered `triton_fc1` kernel in eager allocating
mode. The existing `experiment compare` command remains available.

## Measurement and decision rule

The runner creates one seeded FP16 input pair. It invokes PyTorch/cuBLAS as an
independent numerical reference, compiles each Triton configuration with a
first synchronized call, then checks shape, dtype, finiteness, and
`torch.allclose` at the unchanged `atol=rtol=0.01`. A failed candidate is
`REJECTED_INCORRECT` and is never benchmarked or called faster. Passing this
one finite input does not prove correctness for every input.

After separate warmups, the runner records one unranked timer-control block
per kernel. It then measures one A and one B block in each round. Order
alternates AB, BA; with an odd round count the order differs by one round.
Each block uses the existing CUDA-event and host-clock measurement helper.
The raw rows retain round, order, configuration, iterations, CUDA-event time,
host submission time, synchronized host-wall time, and event-minus-wall
difference. Setup and compilation are separate from ranked timing. Both
timers include the allocating `triton_gemm` wrapper work.

The report gives medians, ranges, standard deviations, CVs, paired A-minus-B
differences, their median absolute deviation, and every round's event and
host-wall difference. These are descriptive uncertainty measures. It does
not assign a probability or confidence interval to correlated timing rounds.

A direction claim needs **all** of these checks: both candidates pass
correctness; at least five complete paired rounds exist; no CUDA-event block
exceeds its synchronized host-wall block in either unranked controls or
ranked rounds; every CUDA-event round for the proposed winner is below every
round for the other kernel; and each paired host-wall difference agrees with
that direction. Otherwise the decision is `INCONCLUSIVE`. The rule uses
observed range separation and directional consistency. It has no tuned
percentage or p-value cutoff. A compiler resource reduction is reported
separately and never satisfies the timing rule by itself.

## Retained RTX 4060 Ti experiment

The run used M=512, K=768, N=3072, FP16, seed 0, 30 warmups per kernel, 100
operations per block, and eight paired rounds. A and B have the same
`64×128×32` tile, four warps, and `group_size_m=1`. Only `num_stages` differs:
A=3, B=2. The exact command was:

```bash
.venv/bin/matmul-inspector-python experiment pair \
  --spec examples/mi_workflow_2_fc1_pair.json \
  --output artifacts/mi-workflow-2/fc1-pair-v1
```

The [versioned bundle](../artifacts/mi-workflow-2/fc1-pair-v1/) contains the
raw result, 16 individual timing records, both PTX files, normalized PTX
comparison and diff, compiler metadata, source snapshot, report, provenance,
and a SHA-256 manifest. The offline verifier returns `PASS`, `INCONCLUSIVE`,
eight rounds, and 16 hashed files. The manifest file's SHA-256 is
`a0986abb2804be6de16069e8661219ab586497faa2a9ba50c7629bc10a2cb6c3`.
The source snapshot SHA-256 is
`a88226c46f335a9768f15b52eb275abf6066bfb57b1f7e012faa937a51768552`.

Both candidates matched the reference output shape and FP16 dtype. Neither
output or reference had a nonfinite value. Both had maximum and mean absolute
error 0.0 on the saved seeded input.

| CUDA-event ms/operation | A: 3 stages | B: 2 stages |
|---|---:|---:|
| Median | 0.063937 | 0.065961 |
| Range | 0.063556–0.066813 | 0.063631–0.070144 |
| CV | 1.87% | 3.38% |
| Host-wall median | 0.060754 | 0.062622 |

The paired A-minus-B CUDA-event median was **−0.001168 ms/operation**. Its
range was **−0.006588 to +0.003176 ms/operation**, with median absolute
deviation **0.001501 ms/operation**. Six rounds favored A and two favored B.
The event ranges overlap. This median is descriptive, not a supported speedup.

The unranked CUDA-event minus synchronized host-wall control was **+0.332297
ms/block for A** and **+0.230378 ms/block for B**. All 16 ranked blocks also
had positive discrepancies (A: +0.309125 to +0.335113; B: +0.319209 to
+0.368814 ms/block). This repeats the timing-boundary concern seen in
MI-Workflow-1 and MI-Dogfood-1. The saved evidence establishes the
disagreement, not its cause. The decision therefore remains `INCONCLUSIVE`
even though the event medians differ.

Triton metadata reported shared memory of 24,576 bytes/program for A and
12,288 bytes/program for B. It reported 121 versus 122 registers/thread and
zero spills for both. That is a **compiler resource reduction in shared
memory**, not a measured latency improvement. Captured PTX 8.7 for `sm_89`
has 408 versus 373 static instructions and 18 versus 12
`cp.async.cg.shared.global` instructions. The normalized PTX differs. Its
text comparison reports 1,037 changed lines; this is not a dynamic
instruction count. Decoded SASS was unavailable. PTX differences alone do
not establish final machine-code changes, occupancy, or runtime behavior.

The bundle records RTX 4060 Ti, driver 616.92, CUDA runtime 12.8, PyTorch
2.11.0+cu128, Triton 3.6.0, Python 3.13.11, the pre-commit Git HEAD, and
the exact command. After the GPU run, the derived JSON comparison and Markdown
report were regenerated offline to show per-round differences, timer-control
values, and PTX identities. The raw result, timing rows, correctness record,
PTX files, and source snapshot remained byte-for-byte unchanged. The manifest
was updated and verified afterward; no second GPU run was made.

## What improved and what remains tedious

An engineer can now answer the three initial questions from one report:
both kernels are correct on the tested input; neither is demonstrably faster
under the captured timing conditions; and the compiler reduced reported
shared memory while changing PTX. The command removed the need to copy two
single-candidate specs, run two separate benchmarks, manually join their
rounds, invoke PTX comparison, or assemble hashes by hand.

The remaining friction is concrete. The current path accepts only the
packaged FC1 kernel, so comparing trusted edited source still needs another
workflow. Its conservative timing gate exposes the recurring CUDA-event
versus host-wall discrepancy but does not diagnose it. Decoded SASS is absent,
so the report cannot answer whether final machine code changed. These are
future work items. They do not justify relabeling this experiment as a win.

## Validation

The 63 relevant CPU-only paired, experiment, and PTX tests passed. The versioned
bundle passed offline hash and decision replay. The existing MI-Workflow-1
command and its tests remain functional. No kernel source, numerical
tolerance, or established benchmark method was changed.
