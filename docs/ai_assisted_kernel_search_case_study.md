# AI-assisted FC1 kernel parameter search

This case study documents an AI-assisted experiment workflow for one existing
Triton matrix-multiplication kernel. Codex proposed controlled parameter changes,
read the resulting evidence, and chose follow-up experiments. Matmul Inspector
validated every proposal, rejected duplicate identities unless a repeat reason
was supplied, executed registered implementations, persisted raw measurements,
and enforced an immutable 24-unit session budget.

This was a Codex-operated parameter search, not autonomous model-API execution.
Codex did not generate or modify kernel source. Every candidate used the existing
`scripts/triton_fc1.py` computation with a different allowed parameter tuple.
The primary result is validation of an evidence-preserving experimentation
framework; it is not a successful kernel optimization.

**No meaningful speedup was demonstrated.**

## Objective and experimental boundary

The objective was to build and exercise a workflow in which an AI assistant could:

1. inspect completed measurements and session constraints;
2. state a concrete hypothesis for each proposed configuration;
3. submit only registered, structured parameter changes;
4. use prior-result references and explicit reasons for repeated identities;
5. stop when the enforced experiment budget was exhausted; and
6. distinguish measured observations from possible explanations.

Matmul Inspector, rather than Codex, owned configuration validation, canonical
fingerprints, duplicate detection, scheduling, correctness gating, execution,
artifact persistence, crash-aware budget accounting, and descriptive ranking.
Session proposals could not change the workload, baseline, tolerances, benchmark
mode, repetition settings, or allowed parameter space.

The completed preallocated session is preserved at
[`artifacts/experiment-sessions/20261012T143844.311515Z-618642c5/`](../artifacts/experiment-sessions/20261012T143844.311515Z-618642c5/).
Its three proposals were `initial-resource-sweep`,
`controlled-interaction-followup`, and `contemporaneous-control-repeat`.

## Hardware, software, and numerical setup

The session and child experiment records report:

| Item | Recorded value |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Ti, compute capability 8.9 |
| GPU memory | 17,175,150,592 bytes |
| Platform | Linux 6.18.40.1 Microsoft WSL2, x86-64, glibc 2.39 |
| Python | 3.13.11 |
| PyTorch | 2.11.0+cu128 |
| Triton | 3.6.0 |
| CUDA runtime reported by PyTorch | 12.8 |
| Session Git commit | `97132b5b820b1d52d957d7641af63f03a40e5079` |
| Session worktree at creation | clean |

The earlier allocating session and the diagnostic were recorded at Git commit
`6196e4d1df127607502deb3d864d9cc0b5080631`. The diagnostic provenance marks the
worktree dirty because the new diagnostic script, tests, and documentation were
still uncommitted. The primary preallocated session was subsequently created
from the clean commit shown above.

The fixed workload was `(512 x 768) @ (768 x 3072)` with contiguous FP16
inputs generated from seed 0. The reference was `torch.matmul`, dispatched by
PyTorch to its CUDA/cuBLAS implementation. The result schema deliberately labels
the reference accumulation behavior as implementation-selected. The Triton
candidate used an FP32 `tl.dot` accumulator and stored FP16 output.

Correctness required matching shape, finite output, and `torch.allclose` with
`atol=0.01` and `rtol=0.01`. In preallocated mode, separate baseline and
candidate output tensors were created before measurement, initialized with NaNs,
and checked after execution to verify that the selected paths overwrote them.

All 24 session experiments passed these checks. Every saved child result reports
maximum and mean absolute error of `0.0` for its seeded input. That observation
does not prove correctness for every possible input, shape, device, library
version, or floating-point execution path.

## Why measurement mode became part of experiment identity

An earlier session used the original allocating wrappers. Its omitted mode is
interpreted compatibly as `allocating`; its artifacts remain unchanged at
[`artifacts/experiment-sessions/20261012T135042.830072Z-0f3501a8/`](../artifacts/experiment-sessions/20261012T135042.830072Z-0f3501a8/).
For stages2, its three run-level speedups were `0.94846x`, `1.00272x`, and
`0.90463x`, producing a median of `0.94846x` and MAD of `0.04383`. The wide
spread made the parameter result difficult to interpret.

A separate measurement-method diagnostic then compared three modes for the same
stages2 candidate and shared seeded inputs:

- allocating PyTorch and Triton wrappers;
- ordinary host submission into preallocated outputs; and
- CUDA Graph replay into preallocated outputs.

The diagnostic used 100 warmups, 500 operations per measured block, 20 blocks
per implementation/mode, alternating implementation order, and reproducibly
shuffled mode order. Each graph captured 500 operations, and one complete replay
was measured per block; it did not divide a single-operation replay by 500. The
diagnostic was recorded separately at
[`artifacts/fc1-timing-diagnostics/20261012T140520.850926Z-24a32de6/`](../artifacts/fc1-timing-diagnostics/20261012T140520.850926Z-24a32de6/)
and consumed no session budget.

| Mode | PyTorch CUDA-event CV | Triton CUDA-event CV | Triton relative range/median | Median Triton host submission/block |
|---|---:|---:|---:|---:|
| Allocating | 1.14% | 3.63% | 14.15% | 8.615 ms |
| Preallocated | 0.98% | 0.77% | 3.11% | 6.424 ms |
| CUDA Graph | 1.00% | 0.77% | 3.00% | 0.090 ms |

The allocating Triton path contained two large timing excursions associated with
long host submission intervals. Preallocation removed those excursions in this
diagnostic. Graph replay sharply reduced host submission duration but did not
appreciably reduce Triton CUDA-event variability beyond ordinary preallocation.
These are associations, not proof that allocation alone caused the earlier
outliers. Preallocation also removed per-call input validation and grid
construction from the Triton wrapper path. The evidence supported making
preallocation an explicit measurement mode while leaving graph replay in the
standalone diagnostic.

Allocating and preallocated results are therefore different measurement
populations. They are not combined in the primary table or ranking.

## Timing boundaries and estimator

The preallocated session fixed 100 warmup operations per implementation, 500
operations per measured block, and 10 alternating baseline/candidate rounds per
experiment. Compilation, reusable output allocation, and Triton grid preparation
occurred before steady-state measurement.

Each round stored three timing domains separately:

- **CUDA-event elapsed time per operation:** an event before block submission to
  an event after all 500 operations, divided by 500;
- **host submission duration per block:** host wall time through submission of
  the ending event, stopping before synchronization; and
- **host wall duration per block:** host wall time through completion of the
  ending event.

The compatibility latency fields are aliases for CUDA-event milliseconds per
operation. A CUDA-event interval around many ordinary host submissions may
include device idle gaps induced by the host failing to keep the stream supplied.
It should not automatically be described as pure kernel execution time.
Preallocation removes output allocation and some wrapper work, but it does not
remove Python submission, Triton launch-wrapper overhead, synchronization
effects, clock or thermal movement, or unrelated system activity.

For each run, the reported speedup is:

```text
median baseline CUDA-event ms/operation
---------------------------------------
median candidate CUDA-event ms/operation
```

The session ranking then takes the median of these run-level ratios. It is a
descriptive estimator, not a significance test or confidence interval.

## Completed preallocated search

The table groups all successful results by the complete resolved Triton parameter
tuple. Explicit repeats of default and stages2 are combined with their original
runs, giving six repetitions each. For each group, the reported value is the
median of all underlying run-level speedups; MAD is the median absolute deviation
of those same ratios. It is not an average or median of proposal-level summaries.

All configurations used `BLOCK_N=128`, four warps, and `GROUP_SIZE_M=1`.

| Configuration | BLOCK_M | BLOCK_K | Stages | Runs | Median run-level speedup | MAD |
|---|---:|---:|---:|---:|---:|---:|
| Default control | 64 | 32 | 3 | 6 | 1.00091x | 0.00065 |
| Stages2 | 64 | 32 | 2 | 6 | 1.00440x | 0.00225 |
| K64 | 64 | 64 | 3 | 3 | 0.99314x | 0.00055 |
| M128 + stages2 | 128 | 32 | 2 | 3 | 0.99937x | 0.00266 |
| Stages2 + K64 | 64 | 64 | 2 | 3 | 0.93439x | 0.00130 |
| M128 + K64 | 128 | 64 | 3 | 3 | 0.92435x | 0.00111 |

The session executed 24 experiments: six in the initial proposal, twelve in the
controlled follow-up, and six explicit repeats. All 24 completed successfully.
Budget accounting ended at 24 consumed, zero reserved, and zero remaining.
Canonical fingerprints detected the repeated default and stages2 identities;
the final proposal was accepted only because each included an explicit repeat
reason.

## Baseline movement and interpretation

The original three stages2 preallocated runs reported `1.00182x`, `1.00206x`,
and `1.03953x`. The largest ratio did not come from a comparably large reduction
in candidate latency: its baseline median rose to `0.06192 ms`, while its
candidate median remained near `0.05957 ms`. That run demonstrates why a matched
baseline and repeated measurements matter.

The final contemporaneous repeats reported:

| Identity | Three run-level speedups | Median | MAD |
|---|---|---:|---:|
| Default control | 1.00362x, 1.00081x, 1.00211x | 1.00211x | 0.00130 |
| Stages2 | 1.00484x, 1.00395x, 1.00656x | 1.00484x | 0.00088 |

Their median difference was about 0.27 percentage points. One final stages2 run
also showed shared raw-round movement: baseline and candidate CVs were both about
4.5%, and their maximum round latencies reached approximately `0.06955 ms` and
`0.06896 ms`. A matched ratio can remain close while both implementations move.

The two K64 combinations were clearly slower within this session: stages2+K64
had about 7% higher candidate latency than its matched baseline, while M128+K64
had about 8% higher latency. K64 alone was only slightly slower. This pattern is
consistent with configuration sensitivity, but the experiment did not collect
register counts, occupancy, instruction schedules, memory transactions, or
profiler counters. It therefore does not prove a parameter interaction in a
formal statistical sense or establish a hardware mechanism such as resource
pressure.

The default and stages2 results are too close relative to observed baseline and
round-level movement to support an optimization claim. **No meaningful speedup
was demonstrated.**

## Reproduction and artifact inspection

The following commands match the current CLI. They inspect the completed session
without launching experiments:

```bash
SESSION=artifacts/experiment-sessions/20261012T143844.311515Z-618642c5

PYTHONPATH=src python -m matmul_inspector.cli session status \
  --session "$SESSION"

PYTHONPATH=src python -m matmul_inspector.cli session context \
  --session "$SESSION" --output /tmp/fc1-preallocated-session-context.json
```

To create a fresh preallocated session and submit the checked-in initial proposal:

```bash
PYTHONPATH=src python -m matmul_inspector.cli session create \
  --config examples/fc1_preallocated_experiment_session.json \
  --output-root artifacts/experiment-sessions

# Set SESSION to the directory printed by the create command.
PYTHONPATH=src python -m matmul_inspector.cli session submit \
  --session "$SESSION" --proposal examples/fc1_session_proposal.json
```

Creating a session does not reproduce the complete adaptive search by itself.
The second and third proposals depended on observations and valid result
references from earlier proposals. Their exact accepted records, resolved
parameters, hypotheses, repeat reasons, schedules, child results, raw
`timings.json`/`timings.csv`, and summaries are preserved beneath:

```text
artifacts/experiment-sessions/20261012T143844.311515Z-618642c5/proposals/
```

The diagnostic can be rerun independently with:

```bash
.venv/bin/python scripts/fc1_timing_diagnostic.py
```

That command launches GPU work and creates a new ignored diagnostic artifact; it
does not submit to or consume a persistent session budget.

## Limitations and next steps

- The study covers one FP16 FC1 shape, one seeded input pair, one RTX 4060 Ti,
  one software environment, and a small allowed parameter space.
- Correctness passed for the recorded inputs and tolerances only. Zero recorded
  error is not a universal numerical guarantee.
- CUDA events around host-submitted blocks are not guaranteed to isolate kernel
  execution from host-induced stream idle time.
- No profiler counters, clock/temperature telemetry, register/occupancy data, or
  compiler-output comparison was available for the session. Optional diagnostic
  telemetry was unavailable because `nvidia-ml-py` was not installed.
- Median/MAD summaries are descriptive. No formal uncertainty model or
  significance threshold was used.
- The search changed only launch parameters for one existing kernel. It did not
  test generated source or alternative kernel computations.

A useful next framework milestone is support for reviewed candidate kernel source
variants with immutable source hashes, explicit compilation provenance, static
compiler-artifact comparison, the same correctness gate, and separate budgets.
That would broaden the search boundary while preserving the distinction between
AI-generated hypotheses, validated executable inputs, numerical evidence,
compiler evidence, and measured performance.
