# Schedule-Agent-Live-1 — live provider integration and GPU audit

## Status

The fourth user-run attempt completed two live OpenAI proposals and two fresh
GPU batches on an RTX 4060 Ti. The controller recorded `COMPLETED`, Decision
A, in 21.31 seconds. The first three attempts remain separate negative
evidence: two failed Batch 1 proposal validation and one failed Batch 2
proposal validation after a passing GPU Batch 1. A standalone Batch 1
preflight passed between the first and second full attempts. The prior
[Schedule Agent case study](schedule_agent_prototype.md) remains Decision B
for its saved-proposal run. Decision A here establishes live orchestration,
not an optimization or a causal effect of evidence on the model's choice.

## Hypothesis and control

Hypothesis: a model can choose a bounded research priority after reading
Batch 1 evidence, while the controller remains the authority for candidate
identity, legality, numerical correctness, measurement, selection, and
resources. The frozen and calibrated deterministic Batch 2 selections are
controls for schedule choice. The earlier saved-proposal run and deterministic
CPU replay control the orchestration path. There is no comparable live-model
proposal obtained without Batch 1 measurements, so these controls cannot
isolate whether those measurements changed the model's decision.

## Integration and limits

The optional `openai` provider uses the [Responses API Structured Outputs
format](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=responses)
through Python's standard library. It returns the existing proposal fields.
The same independent validator checks field names, descriptor and family
allowlists, mode, candidate hashes, and fixed batch sizes. The controller
freezes Batch 1 and the Batch 2 control before planning. It runs the existing
CPU/GPU gates and selector. A model cannot change tolerances, kernels,
benchmark settings, or budgets.

The full controller accepts at most two calls in Batch 1 then Batch 2 order. A
separate proposal-only Batch 2 preflight accepts one call from a verified
Batch 1 checkpoint. Every
attempt counts, including a timeout or failed request. There are no retries.
Each call has a 15-second wall deadline, a 32,000-byte serialized request cap,
a 16,000-byte response cap, and an API `max_output_tokens` value of 650.
Before a call, the adapter reserves its full request-byte cap plus the output
token cap against a 66,000-token session budget. After each response it requires
valid `input_tokens`, `output_tokens`, and `total_tokens` usage and checks the
reported totals. Request bytes are a conservative local input guard; exact
server-side input token use is known only from the response. A failed request
may consume provider tokens that the server does not report. Missing usage or
budget exhaustion ends the run. A provider transport error or timeout produces
one marked deterministic fallback, so it cannot qualify for Decision A.

The proposal schema now uses explicit schedule-hash enums. In Batch 1,
`comparison_preferences` can use only the eight frozen hashes and
`candidate_hashes` must be empty. In Batch 2, candidate scope uses the
eligible N=256 hashes outside Batch 1. Comparison preferences can use those
hashes plus the measured Batch 1 hashes. These choices come from controller
feedback. The independent deterministic validator still rejects every
invalid hash. The eligible Batch 2 list appears in the schema, so the adapter
omits its duplicate from the prompt text while preserving the full feedback
artifact. The v3 Batch 1 feedback produces a 30,578-byte mocked Batch 2
request under the unchanged 32,000-byte cap.

A full schema audit after the third live attempt also added enums for
`experiment_family`, `execution_mode`, and descriptor items. It bounded the
three free-text fields with a supported string pattern. Batch 2 candidate
scope now permits either an empty list or 4–16 eligible hashes. The eligible
hash enum is defined once and referenced by both length branches so the
request stays within the existing byte budget. Array uniqueness is not a
supported Structured Outputs constraint. The prompt asks for distinct values;
the independent validator rejects duplicates. The string pattern requires no
leading whitespace, which is narrower than the validator's trim check. It
prevents empty and over-400-character model output without rewriting it.

`requested_configurations` remains a required field in the existing proposal
contract, but it is a controller-owned resource value. Its structured-output
schema now has a one-value integer enum: eight for Batch 1 and four for
Batch 2. The model receives the fixed value; it does not choose a batch
size. The independent validator still checks the unmodified returned value.
An invalid provider output is retained with its failed validation record.
Batch 1 `candidate_hashes` is also fixed to an empty array because its eight
candidates are frozen. All other hard limits are supplied by the controller,
not requested in proposal fields.

The existing experiment limits remain two batches of eight and four N=256
configurations, 12 GPU candidates, 12 profiler invocations, 600 timing samples,
120 seconds per GPU batch, 300 seconds total, and 2 MB of retained files.
The provider is available with `--gpu` or the one-call `--proposal-preflight`.
CPU replay always uses saved proposals and makes no API call.

## Configuration and commands

Use the repository `.venv`. Supply `OPENAI_API_KEY` through your shell or
secret manager. Do not put it in a repository file, command argument, or
artifact path. Set `SCHEDULE_AGENT_MODEL` in your shell to the exact ID of a
model that supports Responses Structured Outputs. No package installation is
needed.

Run this one-call proposal preflight before any GPU run. It uses the same
frozen eight Batch 1 candidates and the existing deterministic validator. It
prints JSON with validation status, comparison hashes, model, API-call count,
and reported token usage. It exits nonzero if validation fails. It creates no
artifact and does not run CPU gates, GPU kernels, Batch 2, or calibration.
The model below is the one recorded by the first failed attempt. **This
command was not executed during implementation.**

```bash
PYTHONPATH=src:. .venv/bin/python scripts/run_schedule_agent_prototype.py \
  --proposal-preflight --provider openai --model gpt-4.1-mini
```

For a separate **Batch 2 proposal-only** preflight, reuse the complete hashed
Batch 1 checkpoint from the third attempt. The command below makes at most one
new model request. It verifies the saved manifest, frozen candidate universe,
Batch 1 validation and gates, correctness and timing record counts, and
reconstructs the model feedback before calling the provider. It does not
resume the GPU experiment, write an artifact, or run a kernel. This live
command has **not** been executed here:

```bash
PYTHONPATH=src:. .venv/bin/python scripts/run_schedule_agent_prototype.py \
  --proposal-preflight --batch2-evidence /tmp/schedule-agent-live-gpu-v3 \
  --provider openai --model "$SCHEDULE_AGENT_MODEL"
```

Example for a new, separately authorized run. This audit made no live request
and ran no GPU benchmark:

```bash
PYTHONPATH=src:. .venv/bin/python scripts/run_schedule_agent_prototype.py \
  --gpu --provider openai --model "$SCHEDULE_AGENT_MODEL" \
  --output /tmp/schedule-agent-live-gpu
```

Offline replay of a completed bundle, with no API credential or GPU execution:

```bash
PYTHONPATH=src:. .venv/bin/python scripts/run_schedule_agent_prototype.py \
  --replay-bundle artifacts/schedule-agent/live-gpu-v4 \
  --output /tmp/schedule-agent-live-replay
```

Replay verifies the source hash manifest, reads the two saved proposals, and
requires an exact Batch 2 selection match. It records replay as saved input,
not as a new live decision. The bundle stores proposal source, validated
proposal, response ID and token usage, controller evidence, and hashes. It
does not store the API key or the model prompt. Live-provider exceptions are
recorded by type without raw provider error text. If the model echoes the key
in a proposal or metadata, the controller rejects that output before writing
it to the bundle.

CPU validation:

```bash
PYTHONPATH=src:. .venv/bin/python -m pytest -q \
  tests/tensor_game_schedule_agent_test.py \
  tests/tensor_game_schedule_reselection_test.py \
  tests/tensor_game_schedule_calibration_test.py \
  tests/tensor_game_schedule_model_test.py
```

## Limits of the integration tests

Mocked API responses test schema parsing, call order, budgets, timeout,
credential exclusion, and fallback. They did not themselves establish that a
live proposal would be valid or GPU execution would complete. The fourth
attempt supplies that separate evidence. Offline replay checks the saved
decision without a new API call or GPU run.

## First live proposal failure

The user-run attempt saved at `/tmp/schedule-agent-live-gpu/` contains one
completed OpenAI response. Its Batch 1 `comparison_preferences` were
`runtime` and `memory_usage`, not schedule hashes. The deterministic validator
rejected `runtime` and marked the attempted run `FAILED_VALIDATION` (Decision
D). The bundle hash manifest verifies. No CPU gate, GPU batch, or Batch 2
proposal record was created.

The adapter had supplied the eight frozen hashes in feedback but described
`comparison_preferences` only as an array of strings. The instruction to use
listed hashes did not define that field. The model's apparent interpretation
was metric preferences. The new schema and prompt specify schedule IDs for
both hash fields. The regression test feeds the exact invalid values through
a mocked response and confirms that the controller still rejects them. A
standalone live preflight subsequently passed with the revised hash schema.

## Second live proposal failure

The user-run attempt saved at `/tmp/schedule-agent-live-gpu-v2/` returned
`requested_configurations: 5` in Batch 1. The fixed Batch 1 budget is eight.
The deterministic validator rejected the raw proposal with
`requested_configurations exceeds or changes the fixed batch budget`. The
bundle's ten-file hash manifest verifies. The run ended at `FAILED_VALIDATION`
(Decision D) before a CPU gate, GPU batch, or Batch 2 model request.

The prior schema declared this field only as `integer`. The prompt included
the limits object but did not explicitly say that the model must echo the
fixed Batch 1 count. Structured Outputs therefore permitted five. The schema
now constrains the existing field to the controller-approved count for each
batch. Mocked responses with five remain rejected and saved unchanged. CPU
tests exercise standalone preflight, both batch-specific schemas, and the
two-batch controller path with mocked provider and GPU functions. At that
stage, no new live API request or GPU benchmark had tested the correction.
Decision B for the earlier saved-proposal prototype remains unchanged.

## Third live proposal failure and full schema audit

The user-run attempt saved at `/tmp/schedule-agent-live-gpu-v3/` returned a
**445-character, nonempty** Batch 2 `hypothesis`. The validator allows at most
400 characters after stripping surrounding whitespace, so it rejected that
raw proposal. A separate read-only field audit also found that its Batch 2
`candidate_hashes` list contained just one eligible schedule. That would fail
the controller's minimum of four for a nonempty scope, but the saved
validation stopped at the earlier hypothesis error. The other saved fields
met their corresponding validator rules. The raw proposal has not been
edited or replayed as valid. Batch 1 had passed validation. It left eight
passing CPU gate records, 32 passing GPU numerical checks, eight agreeing
topology records, 400 raw timing samples, 16 timing summaries, and 16
observations. Its saved feedback reconstructs exactly from those records.
The bundle's 21-file hash manifest verifies. The run ended at
`FAILED_VALIDATION` (Decision D). There
was no Batch 2 GPU batch or completed live two-batch decision. These records
are evidence of Batch 1 execution, not a performance improvement claim.

The model-facing schema and deterministic validator were compared field by
field. All nine fields were already required with additional fields forbidden.
The remaining gaps and their resolution are:

| Proposal field | Prior schema gap | Current model-facing constraint |
|---|---|---|
| `proposal_id`, `hypothesis`, `reason` | Any string, including empty or over 400 characters | Supported pattern requires a non-whitespace first character and at most 400 characters; validator still checks all three |
| `experiment_family` | Any string | Enum of the three approved families |
| `descriptors` | Any strings and no length bound | Approved descriptor enum and at most eight items |
| `execution_mode` | Any string | Enum of eager and CUDA Graph modes |
| Batch 1 `candidate_hashes` | Already max zero | Still fixed empty; Batch 1 selection remains frozen |
| Batch 2 `candidate_hashes` | Eligible hash enum and max 16, but 1–3 entries were possible | Empty or 4–16 entries from the controller's eligible hash list |
| `comparison_preferences` | Already used eligible hash enums and max four | Batch 1 allows frozen hashes; Batch 2 allows measured Batch 1 and eligible Batch 2 hashes |
| `requested_configurations` | Already fixed after the second attempt | Exact integer enum of eight or four by batch |

Types already matched the proposal contract. Full hash identity and
eligibility are derived from controller feedback and independently checked
against the retained candidate corpus. The schema cannot express unique
array items in the supported subset, so duplicate descriptors and hashes
remain a strict validator check. It also cannot prove the research meaning of
free text. Neither limitation leads to automatic repair or relaxed
validation. The schema uses documented Responses Structured Outputs features
for ordinary supported models; the stricter pattern and array cardinality
keywords are not supported for fine-tuned models.

CPU tests use mocked Responses data for both stages, including the observed
445-character failure, valid 400-character boundaries, malformed values,
invalid enums and hashes, budget limits, and the full two-proposal path with
retained Batch 1 measurements. A separate mocked Batch 2 preflight using the
v3 bundle completed with one request of 30,578 bytes, below the unchanged
32,000-byte cap. The fourth user-run attempt later tested the corrected schema
with two valid live proposals and two GPU batches.

## Fourth live attempt: completed two-batch GPU run

The original bundle is `/tmp/schedule-agent-live-gpu-v4/`. Its byte-for-byte
retained copy is [`artifacts/schedule-agent/live-gpu-v4/`](../artifacts/schedule-agent/live-gpu-v4/).
The 31-entry SHA-256 manifest covers every other bundle file. The recorded
source commit is `7a62c4c4d46573130674c47a4beba86776a0af78` on
`feat/schedule-agent-live-1`. The recorded hashes for the runner, agent,
selector, and HA-10 runner match this checkout. The retained HA-5 and HA-9
inputs also match their recorded hashes. The bundle records `FRESH_GPU` with
Python 3.13.11, Torch 2.11.0+cu128, Triton 3.6.0, CUDA runtime 12.8, and an
RTX 4060 Ti with 16 GB and compute capability 8.9.

Both proposal records say `LIVE_OPENAI_RESPONSES`, identify `gpt-4.1-mini`,
and contain different completed `resp_...` IDs. The first reports 1,199 input
and 301 output tokens. The second reports 11,634 input and 455 output tokens.
The cumulative total is 13,589 tokens from two API calls. No fallback record
is present. These local records support the live-provider path. No separate
provider-side receipt was queried during this offline audit. The raw proposal
and response metadata, including full response IDs, remain in the retained
files. The credential scan found no matching local API key or common credential
patterns in the retained bundles.

The controller froze eight Batch 1 candidates before planning. Both proposals
passed the independent validator unchanged. The CPU gates passed for all 12
measured candidates. There are 48 passing GPU numerical checks: three seeded
eager checks and one CUDA Graph check for each candidate, using the established
tolerances. All 12 topology and workspace audits passed. The resource record
shows two batches, 12 GPU candidates, 12 profiler invocations, 600 timing
samples, two planner/API calls, 13,589 tokens, 517,988 controller-accounted
artifact bytes, and 21.31 seconds. These are within the unchanged two-batch,
8+4 candidate, 12 profiler, 600 sample, 66,000 token, 2 MB artifact, and
300-second run limits. The saved proposal usage is below the 650 output-token
cap for each request.

Raw timing consists of five rounds of five samples for each candidate and
mode. All 24 saved timing summaries reproduce exactly from their 600 raw
samples by taking per-round medians and the median of those five medians.
The 16 Batch 1 summaries span 0.802592–2.278560 ms in eager mode and
0.062592–0.091584 ms in CUDA Graph mode. Eager round-median coefficients of
variation reach 0.3396. Batch 1 feedback records nine directional and seven
indeterminate paired observations. It flags three conflicted eager descriptor
tokens and states that its same-sign round screen is neither a confidence
interval nor a speedup claim. The eight Batch 2 summaries span
1.015584–1.428256 ms eager and 0.019808–0.037216 ms CUDA Graph. These
measurements cover different candidate sets and cannot establish an improved
selection against an unmeasured alternative.

### Model proposal and deterministic selection

Batch 1 proposed `MATERIALIZATION_TOPOLOGY` in `CUDA_GRAPH` mode. Its
hypothesis was that topology choices might reduce `launch_groups` and
`max_group_operations` relative to eager execution. Its reason emphasized
launch behavior and four distinct comparison hashes. The Batch 1 candidate
set remained the frozen eight; those comparison hashes were advisory. The
controller measured every frozen candidate in both modes regardless of the
proposal's preferred mode.

Batch 2 proposed `STRUCTURAL_DIVERSITY` in `EAGER` mode. Its hypothesis said
that stored/recomputed forms and launch groups among project-rank8 schedules
would affect eager runtime significantly. Its reason proposed varied
structural schedules and descriptor counts to clarify execution time. It
supplied this four-hash scope, which the controller validated against the
eligible universe:

| Hash prefix | Candidate family | Batch 2 eager median (ms) | Batch 2 CUDA Graph median (ms) |
|---|---|---:|---:|
| `3174902e` | project-rank8 | 1.015584 | 0.019808 |
| `140404eb` | Strassen | 1.099584 | 0.021344 |
| `296f8a36` | Strassen | 1.428256 | 0.037216 |
| `2ddd8dd3` | Strassen | 1.057984 | 0.020352 |

The model supplied the mode and candidate scope. The deterministic selector
ranked eligible schedules within that scope and chose the four physical
Batch 2 candidates in the order shown. The model's `comparison_preferences`
recorded intent but did not allocate measurements or directly change rank.
The physical mode is eager, which equals the controller's default mode; the
model did change its own stated preference from Batch 1 CUDA Graph to Batch 2
eager. The saved `agent_changed_scope` flag is true and
`agent_changed_mode` is false relative to that controller default.

| Selection control | Batch 2 hashes, in selected order |
|---|---|
| Frozen deterministic, eager | `140404eb`, `296f8a36`, `2ddd8dd3`, `3174902e` |
| Calibrated deterministic, eager | `61e6aef2`, `3174902e`, `566c98e3`, `64ebbdfb` |
| Guided by model scope, eager | `3174902e`, `140404eb`, `296f8a36`, `2ddd8dd3` |
| Calibrated deterministic, CUDA Graph | `995cfaf7`, `140404eb`, `296f8a36`, `2ddd8dd3` |

The guided eager set equals the frozen set, although its order changed. It
differs from the unrestricted calibrated eager set in three of four hashes.
That difference is attributable to the model's chosen scope; the ranking
within the scope came from the deterministic selector. The calibrated eager
alternatives were not physically benchmarked in Batch 2, so the saved run
does not measure which set performs better. The model's text does not cite a
Batch 1 timing, coefficient of variation, indeterminate pair, or conflicted
descriptor. It does not visibly recognize the unstable eager measurements.
Its description of all four scoped schedules as project-rank8 is inaccurate
for the three Strassen schedules. Its claim that structure affects runtime
"significantly" is not established by the retained uncertainty screen.
Feedback was supplied to the model before Batch 2, and it made a substantive
scope choice, but the record does not show that measured evidence caused or
justified that choice. There is no comparable no-evidence live-model control.

### Offline verification and decision

Read-only audit of the retained copy verifies the source manifest, frozen
selection, proposal validation, numerical checks, topology, raw timing
statistics, all 22 paired observations, and the controller's Batch 2
selection. Deterministic replay produced an exact Batch 2 selection match
with zero API calls and no GPU execution. Replay is classified Decision B
because it consumes saved proposals; it does not change the original live
run's Decision A classification. The earlier cold-Triton discrepancy in
[`schedule_agent_baseline_gpu_failure_investigation.md`](schedule_agent_baseline_gpu_failure_investigation.md)
remains separate: 28 profiler launch-count assertions can fail on the
unchanged baseline with a cold cache. That finding does not weaken those
assertions and does not invalidate this run's 12 passing topology checks.

```bash
PYTHONPATH=src:. .venv/bin/python scripts/audit_schedule_agent_bundle.py \
  artifacts/schedule-agent/live-gpu-v4
PYTHONPATH=src:. .venv/bin/python scripts/run_schedule_agent_prototype.py \
  --replay-bundle artifacts/schedule-agent/live-gpu-v4 \
  --output /tmp/schedule-agent-live-1-audit-replay
```

Decision A: two live-model proposals and two bounded GPU batches completed.
The model made a substantive research-scope choice after receiving Batch 1
evidence, but a measurement-informed reason for that choice remains unproven.
Whether it would choose differently without evidence is untested. No matched
measurement shows an optimization. A no-evidence live-model control and
matched performance measurements would be needed for those stronger claims.

## Retained failed attempts

The original failed bundles remain untouched under `/tmp/`. Byte-for-byte
copies with their own verifying hash manifests are retained separately at
[`live-gpu-v1-failed`](../artifacts/schedule-agent/live-gpu-v1-failed/),
[`live-gpu-v2-failed`](../artifacts/schedule-agent/live-gpu-v2-failed/), and
[`live-gpu-v3-failed`](../artifacts/schedule-agent/live-gpu-v3-failed/). The
sections above describe their distinct validation failures. None is folded
into the completed run or reported as a passing proposal.
