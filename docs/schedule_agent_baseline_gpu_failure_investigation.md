# Baseline GPU profiler launch-count investigation

## Decision for the Schedule Agent commit

The 28 HBM and mixed-lowering launch-count failures reproduce at the clean
`f6c5113` baseline, without any Schedule Agent files. A fresh Triton cache
and the ordered general-IR, HBM, and mixed test suites trigger the same
28 cases, with the same planned and observed kernel-name sequences, in both
the baseline and active checkout. The failing source and test files have
identical SHA-256 hashes in both trees. This is an independent existing
validation issue. It is not evidence of a Schedule Agent regression.

The strongest supported explanation is a cold-compilation and profiler
collection interaction after earlier profiler-heavy tests. The exact
internal cause is unresolved. The evidence does **not** establish that a
GPU kernel was physically skipped. A representative profiled call produced
the correct numerical output even though PyTorch reported one fewer CUDA
kernel event. The runtime iterates the complete launch sequence. An
independent launch trace would be needed to settle whether the profiler
missed an event.

## Controlled comparison

The baseline was a detached Git worktree at `f6c51135e3e0afed9515837e5fc4502c2ce2f1f6`.
Both checkouts used the active repository's `.venv`: Python 3.13.11,
PyTorch 2.11.0+cu128, Triton 3.6.0, CUDA 12.8, driver 616.92, and the
same RTX 4060 Ti. `PYTHONPATH` was set explicitly for each tree. Import
checks resolved `matmul_inspector` and `scripts` modules to the baseline
worktree during baseline tests, not to the active editable checkout.

| Condition | Baseline | Active checkout |
|---|---:|---:|
| HBM and mixed GPU cases alone, warm shared cache | 28 passed | 28 passed |
| Five shared schedule suites, warm shared cache | 97 passed | 97 passed |
| General IR, HBM, mixed; separate fresh Triton caches | 28 failed, 50 passed | 28 failed, 50 passed |
| Fresh-cache HBM failures | 16 | 16 |
| Fresh-cache mixed failures | 12 | 12 |

The broader five-suite cold runs also each failed one older conventional
materialized-template test: 19 profiler events versus 20 planned launches.
The original active six-suite run failed the same 28 HBM/mixed cases. A
later warm run of those six suites passed 116 tests, including the new
Schedule Agent tests. These different outcomes make cache and test order
part of the reproduction conditions. A single isolated HBM test with a
fresh cache passed. Running the general-IR suite first and then that HBM
test with a fresh cache reproduced 8 observed versus 9 planned launches
on the baseline.

All 28 reproduced failures reach the same assertion: the number of CUDA
kernel events returned by `torch.profiler` is exactly one less than the
lowering plan's expected launch count. Expected versus observed counts are:

| Family and algorithm | Cases, one per size unless noted | Planned | Profiled |
|---|---:|---:|---:|
| HBM conventional | 4 | 9 | 8 |
| HBM project rank-8, two schedules | 8 | 8 | 7 |
| HBM Strassen | 4 | 23 | 22 |
| Mixed conventional | 4 | 8 | 7 |
| Mixed Strassen | 4 | 17 | 16 |
| Mixed project rank-8 | 4 | 16 | 15 |

The four sizes are N=128, 256, 512, and 1024. The
[case records](../artifacts/schedule-agent/baseline-topology-v1/cold_case_counts.jsonl)
retain every exact pytest ID, abstract group count, planned kernel-name
sequence, and profiler-observed sequence for each checkout. The sequences
agree between baseline and active. The missing event kind is an HBM GEMM
group kernel in all HBM cases, a fused GEMM/reconstruction kernel in four
mixed cases, and a form-materialization kernel in eight mixed cases.

An abstract schedule group describes schedule semantics. The lowering
plan's launch sequence is a separate intended runtime operation list.
Profiler events are observations of that execution, subject to the
profiler's collection behavior. These 28 selected plans happen to have
equal abstract-group and planned-launch counts. That equality does not
make an abstract `ON_CHIP` placement a measured physical fact.

The failing test blocks call `execute_hbm` or `execute_mixed` in eager
mode inside `torch.profiler`; they do not use CUDA Graph replay. The
runtime iterates its planned launch sequence, and all cases reached the
topology assertion after their numerical checks. In an additional cold
baseline trace for HBM conventional N=256, the profiled call's output
passed `torch.allclose` with unchanged `rtol=0.005`, `atol=0.05`, while
the plan had nine launches and the profiler listed eight. Warm
[launch diagnostics](../artifacts/schedule-agent/baseline-topology-v1/)
recorded 28/28 planned counts and numerical gates in each tree. The
Schedule Agent's own general-IR GPU bundle has separate eager and CUDA
Graph correctness checks and 300 raw timing samples per mode. Its audit
passed; the older eager profiler failures do not revise that evidence.

## Reproduce and evidence

The [retained investigation bundle](../artifacts/schedule-agent/baseline-topology-v1/)
contains raw cold pytest logs for both trees, paired case counts and kernel
names, warm diagnostic rows, environment and source hashes, and a SHA-256
manifest. It also retains the small profiler probe and warm diagnostic
scripts used to record those rows. To reproduce, create a detached worktree at `f6c5113`, set
`PYTHONPATH` to that worktree's `src` and root, and run the three test
files with the repository `.venv` and a new empty `TRITON_CACHE_DIR`:

```bash
TRITON_CACHE_DIR=/tmp/new-baseline-triton-cache \
PYTHONPATH=/tmp/matmul-inspector-agent-baseline-f6c5113/src:/tmp/matmul-inspector-agent-baseline-f6c5113 \
/home/daniel/projects/matmul-inspector/.venv/bin/python -m pytest -q \
  tests/tensor_game_schedule_ir_test.py \
  tests/tensor_game_schedule_hbm_lowering_test.py \
  tests/tensor_game_schedule_mixed_lowering_test.py
```

The follow-up is to isolate the PyTorch profiler collection behavior
from actual launch execution, for example with an independent CUDA trace
and a controlled compilation/cache sequence. It should be handled as a
separate GPU validation issue. No kernel, numerical tolerance, assertion,
or expected launch count was changed for this investigation.
