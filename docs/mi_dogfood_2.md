# MI-Dogfood-2 — agent-driven FC1 GEMM experiments

## Decision

**Workflow decision A: an external coding agent can run a bounded series of
trusted local kernels through the existing paired comparison without changing
Matmul Inspector internals. Performance decision: INCONCLUSIVE.** Three distinct
Triton configurations passed numerical checks against PyTorch/cuBLAS on three
seeds each. All three comparisons retained eight balanced paired rounds,
compiler output, source snapshots, raw measurements, and verified hashes. Each
had the existing CUDA-event versus synchronized host-wall discrepancy. There
is no evidence-supported faster kernel and no repeat was justified.

This study ran on branch `feat/mi-dogfood-2` from `04aa1e5` with an NVIDIA
GeForce RTX 4060 Ti. Existing unrelated working-tree changes were preserved.
The purpose was to evaluate the developer workflow. No registry, benchmark,
correctness, or framework code changed.

## Timing preflight and controlled baseline

The controlled A kernel is the packaged FC1 Triton GEMM: `64×128×32`, four
warps, three stages, and `group_size_m=1`. Workload: FP16 `(512×768) @
(768×3072)`, seed 0, with independent PyTorch/cuBLAS `torch.matmul` reference.
The trusted local B file uses the same JIT body and launch interface as the
MI-Workflow-3 example. Only the six explicit launch parameters in each spec
select a candidate. The example tolerances remain `atol=rtol=0.01`.

Before testing, [MI-Workflow-1](mi_workflow_1.md),
[MI-Workflow-2](mi_workflow_2.md), and [MI-Workflow-3](mi_workflow_3.md)
already documented positive CUDA-event-minus-host-wall controls and
inconclusive rankings. The current runner first invokes and synchronizes the
cuBLAS reference and both Triton kernels, checks numerical results, then does
30 warmup calls per kernel followed by a synchronization. Each unranked timer
control and ranked block contains **100 eager calls** to the same preallocated
output. Eight alternating AB/BA rounds yield 800 ranked calls per kernel,
plus 100 in each unranked control. There is one Triton GEMM launch per call
in these sources. The start event is recorded before block submissions; the
end event after them. The host submission timer stops before end-event
synchronization, while synchronized host-wall stops after it. Raw records
retain both windows per block. Compilation, warmup, and correctness are outside
the ranked windows. No CUDA Graph is involved. The existing decision gate
rejects a ranking if a CUDA-event interval exceeds its synchronized host-wall
interval, among other conservative checks. I kept that gate unchanged and
treated all timing findings as exploratory.

## Candidate sequence

Each candidate changes one parameter from A. The hypotheses were chosen
before their respective comparison commands. The third rationale was not
separately timestamped in a durable note before execution; the saved spec and
source still identify exactly what ran.

| Order | B configuration change | Pre-run hypothesis | Observation and next choice |
|---|---|---|---|
| 1 | `block_n: 128 → 64` | Lower per-program resource demand may improve occupancy enough to offset twice as many N programs. | Resource use fell, but seven of eight event pairs favored A and the timer gate failed. Test the opposite N direction to see whether fewer programs help. |
| 2 | `block_n: 128 → 256` | Fewer N programs may reduce launch and scheduling overhead, at a resource cost. | Resource use rose sharply; event pairs split 3 A / 5 B and the timer gate failed. Test a different axis once, then stop if timing remains unreliable. |
| 3 | `block_k: 32 → 64` | Halving K-loop trips may offset larger per-program resource demand. | Resource use rose, event pairs split 6 A / 2 B, and the timer gate failed again. Stop at three candidates. |

No candidate changed indexing, arithmetic, numerical tolerance, workload,
warps, stages, or measurement settings. The shared trusted-local source is
[mi_dogfood_2_fc1.py](../examples/kernels/mi_dogfood_2_fc1.py), SHA-256
`c72b1bab91a071d32f33b002bb88a9c22a041a1309138c0b9eb53bf2574c61ec`.
Every bundle also contains the exact source snapshot run by the GPU worker.
The packaged A source snapshot hash is
`a88226c46f335a9768f15b52eb275abf6066bfb57b1f7e012faa937a51768552`.

## Correctness and paired timing

All A and B checks passed on seeds 0, 1, and 2, including output shape,
FP16 dtype, finiteness, and the unchanged cuBLAS comparison. The saved
correctness records show maximum absolute error 0.0 on each seed. This is a
finite-input check, not a proof for every input.

Times below are eager preallocated **ms per GEMM**. A and B medians belong to
their own paired run. The A baseline itself varies across runs, so medians
from different rows must not be compared as if paired.

| Candidate bundle | A event median | B event median | Paired A−B median; range | Event pair signs A/B | Event CV A/B | Event−wall control A/B, ms/block | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| [N64](../artifacts/mi-dogfood-2/n64-v1/) | 0.063891 | 0.069746 | −0.004295; −0.009728 to +0.004055 | 7 / 1 | 5.52% / 4.62% | +0.168966 / +0.208359 | INCONCLUSIVE |
| [N256](../artifacts/mi-dogfood-2/n256-v1/) | 0.072374 | 0.069056 | +0.001147; −0.007556 to +0.007502 | 3 / 5 | 5.10% / 6.71% | +0.646370 / +0.634386 | INCONCLUSIVE |
| [K64](../artifacts/mi-dogfood-2/k64-v1/) | 0.064204 | 0.067932 | −0.001535; −0.007302 to +0.000666 | 6 / 2 | 3.11% / 2.94% | +0.558632 / +0.643733 | INCONCLUSIVE |

The paired median absolute deviations were 0.003845, 0.004652, and
0.002007 ms/op respectively. Every unranked control and every ranked block
had positive event-minus-wall time. The N64 round discrepancy ranges were
+0.016210 to +0.232279 ms/block for A and +0.184453 to +0.223479 for B.
For N256 they were +0.347805 to +0.676797 for A and +0.586732 to +0.683270
for B. For K64 they were +0.528919 to +0.618650 for A and +0.546268 to
+0.622729 for B. Host-wall paired measurements and every individual event
sample are retained in `timings.jsonl` and `comparison.json`. The discrepancy
is observed, not diagnosed as a faulty clock or a faulty kernel. The evidence
supports no performance ranking, including for the lower-median N256 run.
No matched repeat was performed.

## Compiler evidence

Triton metadata and PTX were captured for every comparison. A consistently
reported 24,576 shared-memory bytes/program, 121 registers/thread, zero
spills, and 408 static PTX instructions. The B values were:

| B | Shared bytes/program | Registers/thread | Static PTX instructions | Normalized PTX changed lines | B PTX SHA-256 |
|---|---:|---:|---:|---:|---|
| N64 | 16,384 | 79 | 298 | 946 | `157178e5f556723a4134f9562174ddacded2bfa5842258c33ce32ff2d3c3039b` |
| N256 | 40,960 | 214 | 652 | 1,485 | `e5340d1c43dad12b0acdcc0f0a31238ec3b3f06f66c77ae9ae9830050b10897d` |
| K64 | 49,152 | 160 | 531 | 1,343 | `2a0555ec27e99968f61ca3c019408864ae936bee7aaf27fc3f0a6f5de6a136f2` |

The B values show real compiler-resource changes, but not measured improvement.
These are Triton metadata and static PTX counts. PTX text differences do not
establish final machine-code behavior; decoded SASS was unavailable in every
bundle.

## Reproduce and verify

Run one spec at a time from the repository root with CUDA access. Use a fresh
output path for each invocation:

```bash
.venv/bin/matmul-inspector-python experiment pair \
  --spec examples/mi_dogfood_2_n64.json \
  --allow-trusted-local-source \
  --output artifacts/mi-dogfood-2/n64-v1
```

Replace `n64` with `n256` or `k64` for the other two specs and bundles. The
specs retain all benchmark settings. To verify a bundle offline, without GPU
or importing its source:

```bash
.venv/bin/matmul-inspector-python experiment pair \
  --verify artifacts/mi-dogfood-2/n64-v1
```

The verifier returned `PASS`, eight rounds, 17 hashed files, and
`INCONCLUSIVE` for each bundle. Manifest SHA-256 values are N64
`b75df66178f5502062f4cfbdd1e1fca46adeeb39f443461d6658c49040930fd1`,
N256 `13afd6c07085dbad95e6cabf09f9c4d449fed919bc22e293d27d861f8d859a52`,
and K64 `ee4b7e02efe7b911f12defa5b80f98fe480f3ca7cec85e1ef2035b6b778139f5`.
Each bundle retains its validated spec, source snapshots, raw result, 16
ranked timing rows, correctness checks, PTX files and normalized comparison,
compiler metadata, environment and source provenance, report, and hash
manifest. These files are versioned study evidence.

The relevant CPU suite passed: `45 passed` from
`experiment_pair_source_test.py`, `experiment_pair_test.py`,
`experiment_compare_test.py`, `experiment_source_test.py`, and
`ptx_inspector_test.py`, using `.venv/bin/python -m pytest -q`.

## Developer experience

Matmul Inspector did the important work: it accepted a trusted local kernel
without a registry edit, checked three seeds against cuBLAS, compiled and
warmed both sides, alternated paired blocks, retained both clocks and all
samples, captured compiler and PTX evidence, reported uncertainty, withheld
unsupported wins, snapshotted source, and verified the full bundles offline.
There was no custom benchmark harness or framework change.

Friction recorded during this study:

1. I copied the sample local kernel and hand-edited a full JSON spec for each
   single-parameter candidate. This is workflow overhead, not a defect.
   All three specs repeat the same workload, tolerance, baseline, and budget.
2. To make a study-wide table, I read three individual reports and used a
   short one-off Python command to extract manifest hashes and compiler
   counts. Each report answers its own comparison, but the CLI has no small
   multi-attempt index. This is missing functionality, not a benchmark defect.
3. The timing gate flagged the recurring event/host-wall discrepancy in all
   three runs, but the workflow cannot explain its cause. This is a distinct
   measurement-investigation need. I did not rewrite timing or bypass the
   gate during this dogfood study.
4. The artifact directory is ignored by Git by default. Retaining these
   intended study bundles requires explicit force-staging of their files.
   This is repository workflow overhead, not a measurement defect.

The three most valuable improvements would be a safe spec-variant command
for one-parameter trials, a read-only index of verified attempt bundles, and
an independent timing-boundary diagnostic. Those are recommendations, not
implemented features. The immediate next engineering step is to investigate
the timer discrepancy in a separate controlled study before resuming
performance search. This study does not justify more candidate trials now.
