# Bounded Schedule Agent Prototype

## Decision

**B — deterministic orchestration demonstrated.** One RTX 4060 Ti run completed
the fixed two-batch controller with saved structured proposals. Batch 1
measurements were validated, summarized, and supplied to the Batch 2 proposal
step and the existing mode-specific selector. Batch 2 passed fresh CPU and GPU
gates and produced separate eager and CUDA Graph measurements. A CPU replay of
the saved run reproduced the exact Batch 2 selection. No live model was
configured or invoked. This is a hardware-executed *controller*, not a
hardware-validated live agent or a measured optimization win.

The earlier HA-10 result remains intact. Its anomalous eager CV of 0.439,
unmeasured frozen Batch 2 performance control, and untested graph-only
counterfactual are not reinterpreted here. This prototype ran a new bounded
session with a graph-priority saved proposal. It did not run a matched
frozen-control performance comparison.

## Hypothesis and controls

Hypothesis: a planner can propose a bounded research priority while a
deterministic controller retains authority over candidate identity, legality,
correctness, measurements, resource limits, and selection evidence. The
frozen control selects Batch 2 from the same 156-record candidate universe
with no Batch 1 observations. The calibrated control uses the new Batch 1
observations and the HA-10 selector. The guided decision uses the same
universe and observations, then applies the validated proposal's mode and
optional candidate scope. The saved proposal chose CUDA Graph priority and
left the candidate scope empty. Thus the calibrated and guided lists are
identical *within each mode*. The proposal affected which mode's four
candidates were physically run. It did not change the selector's vote rule.

| Decision | Four Batch 2 hash prefixes |
|---|---|
| Frozen, either mode | `140404eb`, `296f8a36`, `2ddd8dd3`, `3174902e` |
| New calibrated eager | `61e6aef2`, `ab0cd7a4`, `3174902e`, `64ebbdfb` |
| New calibrated graph, physically run | `995cfaf7`, `56c48071`, `58df39d0`, `5ebc4315` |

Both calibrated modes changed relative to frozen. The graph list replaced all
four frozen choices in this session. The physical graph list and default
eager list have no shared hashes. This shows a proposal-controlled research
priority and measurement-driven selection. It does not show that guidance
improved performance. The proposal was saved input, not live AI output.

## Controller and trust boundary

The CPU-safe [agent module](../src/matmul_inspector/tensor_game_schedule_agent.py)
defines a typed planner protocol, deterministic fallback, saved-proposal
replay, optional OpenAI Responses planner, fixed limits, proposal validation,
feedback summary, and three-way selection. The
[runner](../scripts/run_schedule_agent_prototype.py) uses the retained HA-10
corpus and selector, HA-10 CPU gate, HA-9 lowering runtime, and existing
CUDA-event benchmark helper. It does not alter kernels, compiler paths,
decompositions, tolerances, or benchmark implementation. The optional live
adapter uses the [Responses Structured Outputs format](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=responses)
and still passes its result through independent validation. No model SDK is
needed for replay or tests. Provider response ID and usage can be retained;
credentials and prompt context are not written to the artifact bundle.

Proposals have an ID, hypothesis, approved family, descriptor names,
execution mode, optional candidate hashes, comparison preferences, a fixed
configuration request, and a reason. The first eight schedules are frozen.
Batch 2 scope may contain four to sixteen full hashes, each an eligible,
retained-correct N=256 schedule outside Batch 1. The selector ranks within
that scope. Empty scope means the full eligible universe. Comparison
preferences are recorded as research intent; they cannot allocate extra
measurements. Unknown fields, identities, modes, descriptors, families, and
budgets are rejected with an explicit validation record. GPU failures and
provider errors are retained. A provider timeout falls back once to a
marked deterministic proposal. There is no recursive call or retry loop.

Hard limits are two batches, eight plus four N=256 configurations, one
default CUDA device, two execution modes, three correctness seeds, three
warmups, five rounds of five samples per mode, 600 timing samples, 12
profiler invocations, two proposal calls, 15 seconds per provider call,
120 seconds per GPU batch, 300 seconds total, and 2 MB of retained bundle
files. The GPU batch runs in a child process with a deadline. The parent
checkpoints completed batches, proposal validation, controls, feedback,
failures, and terminal state. A failure during a batch may leave that
batch's worker output incomplete; prior completed stages remain retained.
Terminal states include `COMPLETED`, `COMPLETED_WITH_LIMITATIONS`,
`FAILED_VALIDATION`, `BUDGET_EXHAUSTED`, and `BLOCKED`.

## Fresh GPU execution

The run used the repository `.venv` on an NVIDIA GeForce RTX 4060 Ti with
16,380 MiB, PyTorch 2.11.0+cu128, Triton 3.6.0, and CUDA runtime 12.8.
The artifact provenance records the branch, source commit, source hashes,
HA-5/HA-9 manifest hashes, software, and hardware identity. The source
commit at the run was `f6c5113` on
`feat/bounded-schedule-agent-prototype`; the working branch contained the
new uncommitted controller. Replay and audit refinements changed the
controller files after GPU execution. The
[source-at-run snapshots](../artifacts/schedule-agent/gpu-source-at-run-v1/)
match both hashes recorded in the GPU provenance exactly. All eight frozen Batch 1 hashes match HA-10.
The four frozen Batch 2 control hashes were written before measurement.

All twelve executed configurations passed legality, IR validity, exact
structural lowerability, three fresh eager numerical checks at seeds 11,
23, and 37, a graph output check, logical workspace audit, and profiler
launch-topology comparison. The tolerances stayed `rtol=0.005` and
`atol=0.05`. The run retained 36 seeded eager checks, 12 graph checks,
12 topology checks, and 600 raw CUDA-event samples. Each candidate and
mode has five round medians from five samples per round. The Batch 1
feedback contains 16 matched within-family pair observations: 12
directional and four indeterminate. All observation IDs and timing
summaries were reconstructed from raw samples by the
[audit](../scripts/audit_schedule_agent_bundle.py).

The physical graph-priority Batch 2 medians below summarize device-event
time for complete scheduled executions with preallocated workspaces.
CV is the coefficient of variation across five round medians.

| Hash prefix | Eager median ms (CV) | Graph median ms (CV) |
|---|---:|---:|
| `995cfaf7` | 1.286752 (0.017) | 0.056480 (0.262) |
| `56c48071` | 1.030592 (0.007) | 0.075552 (0.572) |
| `58df39d0` | 0.815840 (0.271) | 0.074304 (0.028) |
| `5ebc4315` | 1.009440 (0.275) | 0.072224 (0.040) |

The graph CV of `56c48071` was 0.572; two eager CVs exceeded 0.27.
Some Batch 1 graph CVs were also high, up to 0.671. The calibrated lists
in this session differed from HA-10's earlier lists. That session
variation and five-round sample size limit confidence in the exact
ranking. Pair directions require all five matched round differences to
agree. This screen is descriptive, not a statistical confidence interval.
We retain indeterminate and conflicting evidence instead of forcing a
winner. No frozen Batch 2 configuration was freshly timed in a matched
comparison, so no speedup or Batch 2 performance superiority is claimed.

## Replay, failures, and limits

The saved graph-priority proposals passed both validation steps. The
controller called the second proposal step with candidate identities,
correctness, timing distributions, pair preferences, indeterminate counts,
remaining budget, and eligible hashes. Measurements and interpretation are
separate fields. The selector used only validated Batch 1 observations.
The controller then stopped after Batch 2 and retained its observations
for future replay without selecting a Batch 3. The run used 12 GPU
configurations, 12 profiler invocations, 600 timing samples, two proposal
calls, and about 9.4 seconds of controller wall time. The bundle is under
the 2 MB artifact cap.

The first direct replay attempt is retained as `gpu-replay-v1`: it failed
validation because the newly added replay check compared JSON lists with
Python tuples in the benchmark configuration. The comparison was fixed.
`gpu-replay-v2` completed before the derived audit files were appended.
`gpu-replay-v3` replays the final audited bundle and confirms the full
Batch 2 selection exactly. CPU tests also cover invalid proposals,
budget limits, failed-candidate exclusion, mode isolation, contradictory
and missing evidence, provider timeout, malformed saved output, mid-run
failure preservation, and manifest hashes. An initial CPU suite used the
system Python environment. A later `.venv` run with CUDA available passed
all 116 Schedule Agent and related schedule tests with a warm cache. The
fresh GPU run and audit also used `.venv`.

An additional `.venv` review found cold-cache profiler launch-count
failures in older HBM and mixed-lowering tests. The same 28 failures,
counts, and observed kernel sequences reproduce at the clean HA-10
baseline. Warm runs pass in both trees. The separate
[baseline GPU failure investigation](schedule_agent_baseline_gpu_failure_investigation.md)
retains the comparison and its limits. This does not change Decision B or
make a performance claim.

## Reproduce and retained evidence

From the repository root:

```bash
PYTHONPATH=src:. .venv/bin/python scripts/run_schedule_agent_prototype.py \
  --gpu --proposals artifacts/schedule-agent/replay_proposals_v1.json \
  --output /tmp/schedule-agent-gpu
PYTHONPATH=src:. .venv/bin/python scripts/audit_schedule_agent_bundle.py \
  artifacts/schedule-agent/gpu-v1
PYTHONPATH=src:. .venv/bin/python scripts/run_schedule_agent_prototype.py \
  --replay-bundle artifacts/schedule-agent/gpu-v1 \
  --proposals artifacts/schedule-agent/replay_proposals_v1.json \
  --output /tmp/schedule-agent-replay
PYTHONPATH=src:. python -m pytest -q tests/tensor_game_schedule_agent_test.py \
  tests/tensor_game_schedule_reselection_test.py
```

The [GPU bundle](../artifacts/schedule-agent/gpu-v1/) contains the universe,
proposal and validation records, frozen and calibrated controls, batch
gates, correctness, topology, raw samples, timing summaries, feedback,
pair observations, calibration after each batch, failure-independent
decision, resource use, audit, and SHA-256 manifest. The
[final CPU replay](../artifacts/schedule-agent/gpu-replay-v3/) references
the audited source manifest and records exact selection agreement. The
[saved proposals](../artifacts/schedule-agent/replay_proposals_v1.json) are
replay inputs. They must not be described as live model decisions. The
[exact source snapshots](../artifacts/schedule-agent/gpu-source-at-run-v1/)
identify the code that generated `gpu-v1`; the committed code also supports
direct replay of that bundle.

## Next step

This is a credible foundation for a later live-model trial because the
model's role is narrow and the deterministic controller has been exercised
on real GPU results. **A remains unproven** until an actual configured
model proposes a valid Batch 1 experiment, receives measured feedback,
proposes a valid Batch 2 decision, and both batches complete under the
same gates. A performance claim would also need an independently measured
matched control and better timing stability. No further milestone was
started. HA-Compiler-1 remains deferred.
