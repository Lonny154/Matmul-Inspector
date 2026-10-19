# HA-Schedule-10: IR-wide calibration and result-driven reselection

## Decision

**A — empirically demonstrated for the bounded eager Batch 2 experiment.**
Fresh RTX 4060 Ti measurements from the eight N=256 Batch 1 configurations
changed the four-config eager Batch 2 selection relative to the frozen
control. Separate CUDA Graph measurements also changed its mode-specific
selection, but the graph-only fourth choice was a counterfactual and was not
run in Batch 2. All twelve executed configurations passed fresh correctness,
workspace, and launch-topology gates. A saved-observation replay reproduced
both selections exactly. This establishes result-driven *selection*, not a
faster measured outcome. The bounded Schedule Agent Prototype is ready as a
small deterministic workflow with the existing gates and budget; it is not
an autonomous agent or a general performance optimizer.

## Hypothesis and control

Hypothesis: matched device measurements from a small, structurally varied
Batch 1 can change a deterministic Batch 2 choice. The frozen control uses
the same 156-candidate universe, configuration, ordering, and four-candidate
budget, with an empty observation list. The updated replay differs only in
the supplied observations. Eager and CUDA Graph evidence are isolated. A
change of selection is distinct from a measured improvement, and no speedup
is claimed here.

## Corpus and selection

The source is the hash-verified HA-5 `candidate_schedules.jsonl` joined by
full schedule hash to HA-9 IR, classification, and seed-level correctness.
The join checks algorithm, size, IR hash, exact mapping, and three correctness
seeds for each record. It reproduces **156/156 IR-valid**, **156/156
structurally exact-lowerable**, and **132/156 numerically passing** records.
The 24 failures are all Strassen at N=512 or 1024. They remain in the
universe and cannot enter a timing preference or selected Batch 2. No new
GPU correctness claim follows from this join.

The bounded experiment uses N=256 only. This holds matrix size fixed while
testing three algorithm families and varied schedule structure. Batch 1 has
four frozen rank-8, two conventional, and two Strassen schedules. Within each
family a greedy rule adds the candidate with the most unseen structural
tokens, breaking ties by full schedule hash. Tokens describe stored or
recomputed forms, HBM products and partials, reconstruction in GEMM groups,
launch-group count, and maximum group width. Tokens are schedule descriptors,
not proven independent physical mechanisms. Batch 2 uses four additional
configurations. The eager updated choice determines which four would be run
on GPU. The graph choice is recorded separately as a mode-specific
counterfactual if its four choices differ.

| Family | Batch 1 schedule hash prefixes | Structural reason |
|---|---|---|
| Frozen rank-8 | `1406976b`, `749d19cf`, `552e97f0`, `fcd175d5` | Recomputation, materialized products, and group topology |
| Conventional | `3e5b6637`, `47181a39` | Fused versus materialized product/reconstruction topology |
| Strassen | `11ae1fc3`, `154a036a` | Product/partial placement and reconstruction topology |

Full hashes, IR hashes, lowering plan hashes, and descriptors are in the
[Batch 1 manifest](../artifacts/ha-schedule-10/cpu-v1/batch1_selection.json).
The earlier CPU bundle checked the selected eight and the four frozen Batch 2
schedules for legality, IR validity, and exact structural lowerability. The
new GPU run checked the eight plus four eager-updated Batch 2 choices again.
All have retained HA-9 `PASS` status and fresh GPU correctness passes. No
failed large-size Strassen record received a latency ranking.

## Calibration and replay rule

The new CPU module keeps HA-5's ordinal calibration intact. HA-5's older
rank-8 profile has fixed physical topology and is unsuitable as a general
IR-path profile, so this study stores separate content-hashed pair
observations. Each observation carries two full schedule hashes, algorithm,
size, dtype, execution mode, matched round medians, median and delta range,
benchmark settings, correctness gate, provenance, confidence, and reason.
All round differences must have the same sign to form a directional
preference. Overlapping or reversed rounds produce `INDETERMINATE` with no
vote. Opposing observations are retained and their structural votes cancel.
This conservative rule is a repeatability screen. It is not a confidence
interval or proof of a population-level winner.

Only descriptors that distinguish the pair receive votes. Transfer is
restricted to the same algorithm, size, and execution mode. Batch 2 sorts
first by net ordinal support, then by new descriptor coverage, then full
schedule hash. The retained `supporting_tokens` list names each observation
ID and the signed vote behind a changed selection. This is an inspectable
exploration rule, not a latency predictor or a causal claim about a token.

The synthetic fixture deliberately makes `1406976b` win the eager pair and
`749d19cf` win the graph pair. Its `1.0` and `2.0` values are illustrative
numbers, **not hardware milliseconds**. The frozen control selects the same
four prefixes in both modes: `140404eb`, `296f8a36`, `2ddd8dd3`,
`3174902e`.

| Mode | Synthetic updated Batch 2 | Difference from frozen |
|---|---|---|
| Eager | `61e6aef2`, `3174902e`, `566c98e3`, `64ebbdfb` | Three replacements from rank-8 structural votes |
| CUDA Graph | `995cfaf7`, `140404eb`, `296f8a36`, `2ddd8dd3` | One replacement from the opposite graph preference |

This ablation shows the mechanics: remove only the observations and both
selections return to the frozen list. It does not show that any changed
schedule is faster. The CPU replay also covers missing evidence, ties,
contradictory evidence, tampered observation IDs, failed-candidate exclusion,
mode isolation, quotas, and deterministic reruns.

## Fresh GPU result: October 19, 2026

The run used the repository's `.venv` on an NVIDIA GeForce RTX 4060 Ti
(16,380 MiB, driver 616.92). Its saved observation provenance records
Python 3.13.11, PyTorch 2.11.0+cu128, Triton 3.6.0, CUDA runtime 12.8,
compute capability, source hashes, and the fixed benchmark configuration.
The actual commit was `4def6e3` on
`feat/ha-schedule-10-calibration-reselection`. The eight Batch 1 hashes and
the frozen four-hash Batch 2 control were verified against the retained
CPU manifest and written to a separate
[preflight control](../artifacts/ha-schedule-10/gpu-preflight-control-v1.json)
before GPU execution. The GPU run then used a new output directory.

Every executed candidate passed three fresh eager numerical checks at seeds
11, 23, and 37, with unchanged `rtol=0.005` and `atol=0.05`. Each also
passed a CUDA Graph output check, logical workspace audit, and profiler
launch-topology comparison. That is 36 seeded eager checks, 12 graph checks,
and 12 topology checks, with no failures. The run retained 600 raw CUDA-event
samples: 400 in Batch 1 and 200 in the four executed eager-selected Batch 2
configurations. Each candidate/mode has five rounds of five samples.

Batch 1 median device-event times below are medians of five round medians.
The CV is computed across those round medians. Times are complete scheduled
executions with preallocated workspaces, not host-visible operation latency.

| Family | Hash prefix | Eager median ms (CV) | Graph median ms (CV) |
|---|---|---:|---:|
| Frozen rank-8 | `1406976b` | 1.554752 (0.009) | 0.151680 (0.111) |
| Frozen rank-8 | `749d19cf` | 1.793504 (0.036) | 0.214912 (0.074) |
| Frozen rank-8 | `552e97f0` | 1.587936 (0.021) | 0.161248 (0.113) |
| Frozen rank-8 | `fcd175d5` | 1.595680 (0.018) | 0.153440 (0.109) |
| Conventional | `3e5b6637` | 1.266880 (0.019) | 0.093056 (0.177) |
| Conventional | `47181a39` | 1.478112 (0.041) | 0.137600 (0.053) |
| Strassen | `11ae1fc3` | 1.373888 (0.181) | 0.209152 (0.101) |
| Strassen | `154a036a` | 1.359200 (0.156) | 0.158368 (0.103) |

Sixteen within-family pair observations were constructed from matched round
medians. Twelve met the rule that all five round differences have the same
sign. Four were `INDETERMINATE` and contributed no vote. For example, the
rank-8 `1406976b` versus `749d19cf` pair favored the former in all five
eager rounds (median 1.554752 versus 1.793504 ms; left-minus-right round
range −0.330880 to −0.161824 ms) and all five graph rounds (0.151680 versus
0.214912 ms; range −0.073280 to −0.035968 ms). The rank-8 `552e97f0`
versus `fcd175d5` comparison was indeterminate in both modes. The Strassen
eager CVs were 0.156–0.181, so their narrow median difference is not treated
as a reliable speed ranking. The all-same-sign screen is descriptive, not a
95% confidence interval; five rounds and one session cannot establish a
general winner.

| Mode | Frozen Batch 2 | Updated Batch 2 from real Batch 1 evidence |
|---|---|---|
| Eager | `140404eb`, `296f8a36`, `2ddd8dd3`, `3174902e` | `61e6aef2`, `3174902e`, `566c98e3`, `64ebbdfb` |
| CUDA Graph | `140404eb`, `296f8a36`, `2ddd8dd3`, `3174902e` | `61e6aef2`, `ab0cd7a4`, `3174902e`, `64ebbdfb` |

The eager update replaces three of four frozen choices. The graph update
also replaces three, but `ab0cd7a4` is graph-only and was **not** freshly
validated or timed in Batch 2. Its older HA-9 correctness status remains
`PASS`. The physical Batch 2 budget was the four eager choices; those four
were checked and timed in both modes. Their median device-event times were:

| Eager-selected Batch 2 hash | Eager median ms (CV) | Graph median ms (CV) |
|---|---:|---:|
| `61e6aef2` | 1.564032 (0.439) | 0.148608 (0.044) |
| `3174902e` | 1.549344 (0.040) | 0.135104 (0.051) |
| `566c98e3` | 1.594880 (0.023) | 0.152160 (0.054) |
| `64ebbdfb` | 1.535648 (0.046) | 0.148640 (0.047) |

`61e6aef2` had one unusually low eager round median of 0.293664 ms while
its other eager rounds were near 1.5–1.7 ms. That yields CV 0.439. We
retain the outlier and do **not** claim a Batch 2 latency winner or a better
outcome than the frozen control. The frozen Batch 2 candidates were a
selection control, not measured in this new GPU batch.

The observed selection change is traceable to measured rank-8 pair votes
on distinguishing schedule descriptors such as `hbm_products=False`,
`gemm_recon=True`, and recomputation/group topology. These are correlated
descriptors, not isolated hardware mechanisms. An offline ablation using
only rank-8 observations reproduces the full updated selections; removing
rank-8 observations produces different choices. Four rank-8 candidates
generated six pairs per mode, while the other families generated one each,
so pair-count asymmetry contributes to their larger ordinal support. No
cross-family latency preference is inferred.

The saved real observations replayed exactly: both updated and frozen
selection files match the GPU run byte-for-byte after parsing, and every
observation was recomputed from the retained raw round medians. Removing
each of the five rounds in turn still changes both selections relative to
the frozen control. The eager four-candidate *set* is identical in all five
such replays. The graph set is identical in four of five; one fold replaces
its fourth choice. This supports a reproducible feedback effect while
limiting claims about the exact graph ranking. The
[validation audit](../artifacts/ha-schedule-10/gpu-validation-audit-v1.json)
retains these checks and ablations.

## GPU method and limits

The `--gpu` path reuses the HA-9 IR runtime and the existing
interleaved CUDA-event benchmark helper. Every selected candidate is checked
again for legality, IR validity, and exact plan identity. The GPU path then
audits workspace buffers, checks FP16 output against PyTorch at seeds 11, 23,
and 37 with unchanged `rtol=0.005`, `atol=0.05`, and checks profiler launch
topology. Only passing paths are timed. Graph replay receives a separate
correctness check. Each algorithm and mode uses three warmups and five
rounds of five iterations with deterministic order. Raw CUDA-event samples
and round summaries are retained. These are device-event observations;
host-visible latency, allocation time, compiler resources, and physical DRAM
traffic are not measured by this command. The separate eager and graph
choices are not assumed to rank the same way. The command has a fixed
8 plus 4 configuration budget and stops after Batch 2.

The GPU topology record also keeps per-launch cubin hashes and a logical
workspace audit. These hashes identify compiled binaries but do not measure
their resource use or dynamic instruction counts.

The work first had a CPU-only checkpoint because CUDA access was blocked in
that earlier session. CUDA later became available in this VS Code Codex
session, so the bounded GPU study above was run. Eager and graph samples
come from one session on one device. The graph-only updated choice was not
part of the physical four-configuration Batch 2 budget. No new PTX/SASS
analysis, host timing, allocation timing, or measured DRAM traffic is claimed.

## Reproduce and artifacts

From the repository root:

```bash
PYTHONPATH=src:. .venv/bin/python scripts/run_ha_schedule_10.py --output /tmp/ha10-cpu
PYTHONPATH=src:. .venv/bin/python scripts/run_ha_schedule_10.py \
  --observations artifacts/ha-schedule-10/synthetic-observations-v1.json \
  --output /tmp/ha10-synthetic
PYTHONPATH=src:. .venv/bin/python scripts/run_ha_schedule_10.py --gpu --output /tmp/ha10-gpu
PYTHONPATH=src:. .venv/bin/python - <<'PY'
import json
from pathlib import Path
source = Path('artifacts/ha-schedule-10/gpu-v1/calibration_after.json')
Path('/tmp/ha10-real-observations.json').write_text(
    json.dumps(json.loads(source.read_text())['observations']) + '\n')
PY
PYTHONPATH=src:. .venv/bin/python scripts/run_ha_schedule_10.py \
  --observations /tmp/ha10-real-observations.json --output /tmp/ha10-real-replay
PYTHONPATH=src:. python -m pytest -q tests/tensor_game_schedule_reselection_test.py
```

The [CPU bundle](../artifacts/ha-schedule-10/cpu-v1) retains the exact
candidate universe, selected manifests, CPU gates, empty timing files,
calibration states, frozen and updated selections, provenance, and SHA-256
manifest. Provenance includes the uncommitted runner and selector source
hashes. Its decision is `D_HARDWARE_CALIBRATION_PENDING`. The
[synthetic replay bundle](../artifacts/ha-schedule-10/synthetic-replay-v1)
contains the same universe and configuration plus the separately labeled
[fixture](../artifacts/ha-schedule-10/synthetic-observations-v1.json).
Its decision is `B_MECHANICALLY_DEMONSTRATED_ONLY`. Neither CPU bundle
contains GPU measurements. The [real GPU bundle](../artifacts/ha-schedule-10/gpu-v1)
retains correctness, topology and cubin hashes, 600 raw event samples,
round summaries, observations, frozen and updated selections, and a verified
file-hash manifest. The separately exported
[real observations](../artifacts/ha-schedule-10/gpu-observations-v1.json)
fed the [offline replay](../artifacts/ha-schedule-10/gpu-replay-v1).
That replay has a `B` label because it did not execute GPU work itself; its
selection files match the `A` GPU run. Output directories are never
overwritten. The preflight control and validation audit are separate small
JSON files; the audit records their hashes and source-bundle manifest hashes.

The earlier CPU checkpoint was created from
`feat/hardware-aware-general-lowering-ir` at `5793012`. The GPU run used
`feat/ha-schedule-10-calibration-reselection` at `4def6e3`. Unrelated
existing modifications and untracked files were left untouched. This GPU
study made no commit, push, or PR.

## Next step and deferred roadmap

The empirical reselection prerequisite is met for a bounded eager workflow.
The next milestone can build a small deterministic Schedule Agent Prototype
using the current legality, IR, correctness, benchmark, and evidence-replay
components. Keep the 8 plus 4 budget, exact mode separation, hard gates,
raw evidence, and frozen-control ablation. Treat graph-only choices as
counterfactual until fresh graph-specific Batch 2 validation is budgeted.
Do not promote a candidate on a single latency result; the eager Batch 2
outlier and one-session limits call for a separate stability check before
any performance recommendation.

After the agent foundation, **HA-Compiler-1 — Alternate Open Compiler
Backend** may investigate a narrow optional MLIR/LLVM-NVPTX lowering for
`PREPARE_FORM` or `RECONSTRUCT` against the Triton backend. Preserve the
Matmul Inspector schedule IR as the source of schedule semantics. Compare
correctness, PTX and SASS, latency, and compiler provenance. The sequence is
HA-Schedule-10 → Bounded Schedule Agent Prototype → HA-Compiler-1. No alternate
compiler work is part of this milestone.
