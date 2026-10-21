# Schedule-Agent-Live-2 — evidence-grounded decision ablation

## Status and question

The bounded model-only run completed on October 19, 2026. It made six live
OpenAI Responses calls and no GPU benchmark calls. Five proposals failed the
unchanged deterministic validator. There were **zero complete valid matched
pairs**. The controller status was `COMPLETED_WITH_INVALID_PROPOSALS`, and
the research decision is **D: no demonstrated evidence-driven decision
benefit**. The committed [Live-1 study](schedule_agent_live_1.md)
demonstrated two live proposals and two GPU batches, but its Batch 2 rationale
did not cite measurements or acknowledge unstable eager timing. It also
described three Strassen schedules as project-rank8. This ablation asks
whether supplying the retained Batch 1 evidence changes a model's Batch 2
research decision in a reproducible, factually grounded way.

## Matched conditions

The source is the committed, hash-verified
[`live-gpu-v4`](../artifacts/schedule-agent/live-gpu-v4/) bundle. Before any
model request, the runner audits its numerical, topology, timing, observation,
and selection records. It reconstructs Batch 1 feedback from the raw saved
files. The 39 correctness-eligible N=256 schedule identities, algorithm-family
labels, and eight descriptors come from the retained candidate corpus. The
eight frozen Batch 1 hashes and remaining Batch 2 allowlist are checked
against that corpus. The model receives the family label for every eligible
schedule, not a label inferred from its own explanation.

Each pair uses the same research task, catalog, eligible hashes, proposal
schema, four-candidate Batch 2 budget, and deterministic validator. The only
prompt difference is `batch1_evidence`:

| Condition | Evidence slot |
|---|---|
| Control | `{"status":"WITHHELD"}`; no Batch 1 timing, pair outcome, or uncertainty result |
| Treatment | Actual Batch 1 eager and CUDA Graph medians, five round medians per series, coefficients of variation, paired outcomes, correctness status, and uncertainty interpretation |

The treatment's pair records use indices into the shared frozen Batch 1 list
to keep the request within its byte cap. They preserve the original pair
directions, confidence labels, and delta ranges. The original observations
and all raw GPU samples remain in the source bundle. No Batch 2 measurement
is sent to either condition. The task instruction tells both conditions to
use catalog identities, cite a hash and mode for a timing claim, and avoid
inventing measurements. It does not supply a preferred choice.

The existing `OpenAIPlanner` and strict Batch 2 validator handle both
conditions. The model may choose a mode, family, descriptors, candidate scope,
and comparison preferences. The deterministic selector still decides ranking
within an eligible scope. Neither condition can change schedule legality,
correctness, kernel behavior, benchmarking, or resource limits. The runner
does not repair invalid proposals or fall back to a deterministic proposal.

## Bounded model-only run

The default run has three matched pairs and alternates order: control then
treatment, treatment then control, control then treatment. `--pairs` accepts
one, two, or three; at most six API requests occur in one run. There is no
retry. Each call retains the existing 15-second deadline, 16 KB response cap,
and 650 output-token cap. The ablation narrows the request cap to 24 KB;
both prepared requests are below it. A global 66,000 reported-token cap is
checked with a conservative 24,650-token reservation before each call.
Therefore a high-usage run may stop before six requests. No GPU candidate,
profiler, or timing-sample budget is spent. The output path must not exist.

Supply `OPENAI_API_KEY` through the shell or secret manager. Never put it in
the command, repository, or output path. The runner stores the credential-free
request body, unmodified proposal, response ID/status/usage, source manifest
hash, source provenance, validation outcome, and deterministic assessment.
Provider errors are retained by type without raw exception text. A response
echoing the credential is withheld and stops the run. The output has a
SHA-256 manifest and remains usable for offline review.

The recorded one-time live command was:

```bash
PYTHONPATH=src:. .venv/bin/python scripts/run_schedule_agent_decision_ablation.py \
  --model gpt-4.1-mini --pairs 3 \
  --output /tmp/schedule-agent-live-2-ablation-v1
```

The complete original run was copied byte-for-byte from
`/tmp/schedule-agent-live-2-ablation-v1` into the versioned
[`live-2-ablation-v1`](../artifacts/schedule-agent/live-2-ablation-v1/)
bundle. Offline replay makes no API request and runs no GPU kernel:

```bash
PYTHONPATH=src:. .venv/bin/python scripts/run_schedule_agent_decision_ablation.py \
  --replay-bundle artifacts/schedule-agent/live-2-ablation-v1
```

The replay verifies the output manifest, source bundle, condition prompts,
identical schedule schema and budget, proposal validation, selected schedule
identities, and paired comparison. The mock tests run this path with the API
credential removed.

## Observed run and evidence audit

The saved design records `gpt-4.1-mini`, three alternating-order pairs, a
six-call maximum, a 650 output-token cap per call, a 24 KB request cap, and
a 66,000 reported-token cap. The saved progress reports six calls, six
records, and 48,696 reported tokens. Every record has a credential-free
request body, a distinct response ID, `completed` provider status, the
reported input/output token counts, an unmodified proposal, and a validation
result. The response model was `gpt-4.1-mini-2025-04-14`. The recorded source
is `LIVE_OPENAI_RESPONSES`; the runner has no replay or fallback path during
the live run. This provenance is saved by the local controller, not an
independent provider receipt.

The hash manifest covers all ten files and matches their bytes. The design's
runner, adapter, and ablation-module hashes match the reviewed sources. Its
Live-1 source manifest hash matches the committed `live-gpu-v4` bundle, whose
read-only audit still passes. The control requests have a `WITHHELD` evidence
slot; treatment requests have the retained Batch 1 timings and uncertainty.
All six requests use the same candidate allowlist, strict schema, task,
four-configuration budget, and controller validator. Offline replay checks
these facts and reproduces all six validation outcomes and the saved summary.
No credential pattern was found in the retained bundle.

| Trial | Condition | Result | Distinct candidate hashes in four entries | Exact validator finding |
|---:|---|---|---:|---|
| 1 | Control | Failed | 2 | `candidate_hashes must be a unique bounded list` |
| 1 | Treatment | Passed | 4 | All approved fields and fixed budget validated |
| 2 | Treatment | Failed | 1 | `candidate_hashes must be a unique bounded list` |
| 2 | Control | Failed | 2 | `candidate_hashes must be a unique bounded list` |
| 3 | Control | Failed | 3 | `candidate_hashes must be a unique bounded list` |
| 3 | Treatment | Failed | 2 | `candidate_hashes must be a unique bounded list` |

In every failed proposal, each individual hash was in the 31-schedule Batch 2
allowlist. The model repeated one or more of those hashes. The saved schema
allows an empty list or 4–16 eligible entries and the prompt asks for
distinct choices, but the schema has no uniqueness constraint. These were
model-generated outputs that satisfied the saved schema's item and length
rules and failed the controller's stricter uniqueness rule. The API accepted
the requests and returned completed structured responses; there is no saved
schema/API rejection or controller error. The controller did not repair or
reinterpret any proposal. The one valid treatment cannot form a matched pair
because trial 1's control failed.

The valid treatment chose CUDA Graph and a four-hash scope. Its assessed
selection spans Strassen, conventional, and project-rank8 families, yet its
reason calls all four schedules project-rank8. It mentions prior timing
uncertainty but cites no specific measured hash, mode, or millisecond value.
The family claim is contradicted by the retained candidate corpus. Even this
valid proposal therefore does not establish a well-grounded measurement
decision. Neither it nor any other Live-2 proposal was benchmarked on GPU.

With zero valid control/treatment pairs, the ablation cannot compare valid
decisions, show that hardware evidence changed the model's decision, or
evaluate whether any choice improved performance. The summary's zero
within-condition variability flags mean that no *valid-pair* variability
could be computed; they do not establish stable model behavior. The trial
records still preserve the invalid choices as negative evidence. Further
internal-agent development should be deferred. The next planned milestone
is MI-Workflow-1, an end-to-end kernel experiment experience, outside this
study.

## Evaluation and decision rules

For each proposal, the assessment records the model's substantive fields,
the resulting deterministic selection, family labels checked against the
corpus, frozen and calibrated deterministic selections for the proposed mode,
hashes mentioned in its explanation, numeric millisecond values matching a
retained Batch 1 median, checkable hash/mode/millisecond citations, and whether
its text mentions uncertainty. These checks are conservative. A matching
number alone does not prove that the model attributed it to the right hash or
mode. The stricter citation check requires one identifiable hash, mode, and
millisecond value in a clause. A mixed-family scope plus a
single family in the explanation is flagged for review, not silently marked
correct. Free-text scientific claims require manual inspection against raw
evidence. The paired summary reports field and selected-set differences and
within-condition selection variability; it never labels a difference causal.

- **Reproducible evidence-driven change:** At least two complete pairs show
  the same substantive treatment-versus-control decision difference. The
  treatment explanation correctly links specific Batch 1 hashes, modes,
  timings, and uncertainty to that choice. Control prompts and outputs contain
  no retained measurement results. The decision must be useful under the
  stated research objective; selection alone is not an optimization claim.
- **Model variability:** Control decisions or treatment decisions vary across
  repeated identical-condition trials, or pair differences change direction.
  A single changed proposal is insufficient to attribute a choice to evidence.
- **Unsupported explanation:** A wrong algorithm-family label, invented or
  misattributed timing, confident speedup claim from unstable data, or an
  ungrounded causal claim remains unsupported even when the proposal passes
  structural validation. Numeric and wording flags guide manual review.
- **No demonstrated benefit from evidence:** No consistent substantive
  difference, or treatment differences that are no better grounded than the
  controls. The result may still demonstrate successful bounded model calls;
  it does not establish improved performance without matched GPU timings.

This experiment compares model decisions, not kernel speed. The frozen and
calibrated deterministic selections are reference decisions, not measured
performance controls for the proposed Batch 2 schedules. Three pairs are a
small diagnostic, not a population-level estimate of model reliability.

## CPU validation

```bash
PYTHONPATH=src:. .venv/bin/python -m pytest -q \
  tests/tensor_game_schedule_agent_ablation_test.py \
  tests/tensor_game_schedule_agent_test.py \
  tests/tensor_game_schedule_reselection_test.py \
  tests/tensor_game_schedule_calibration_test.py \
  tests/tensor_game_schedule_model_test.py
```
