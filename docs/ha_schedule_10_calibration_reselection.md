# HA-Schedule-10: IR-wide calibration and result-driven reselection

## Decision

**B — mechanically demonstrated only.** A deterministic two-batch replay changes
Batch 2 when mode-specific pair evidence is supplied. The replay uses fixed
synthetic timings. CUDA was unavailable in this workspace, so no new IR-path
hardware calibration or empirical reselection was demonstrated. The retained
HA-9 five-sample execution smoke is not comparative ranking evidence. The
bounded Schedule Agent Prototype is **not ready**. The smallest next step is to
run the existing bounded GPU command on a CUDA host, retain raw timings and
correctness gates, then compare updated and frozen selections.

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
The selected 8 plus the 4 frozen Batch 2 schedules passed fresh CPU legality,
IR validity, and exact structural lowerability checks. Their retained HA-9
GPU correctness statuses are `PASS`. Fresh CUDA correctness and benchmark
results are pending. No failed Strassen record received a latency ranking.

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

## Hardware method when CUDA is available

The optional `--gpu` path reuses the HA-9 IR runtime and the existing
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

This GPU path was **not run** here: `torch.cuda.is_available()` returned
`False`, and `nvidia-smi` reported that GPU access was blocked. Its timing
and graph-capture behavior therefore still need CUDA-host validation.

## Reproduce and artifacts

From the repository root:

```bash
PYTHONPATH=src:. python scripts/run_ha_schedule_10.py --output /tmp/ha10-cpu
PYTHONPATH=src:. python scripts/run_ha_schedule_10.py \
  --observations artifacts/ha-schedule-10/synthetic-observations-v1.json \
  --output /tmp/ha10-synthetic
PYTHONPATH=src:. python scripts/run_ha_schedule_10.py --gpu --output /tmp/ha10-gpu
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
Its decision is `B_MECHANICALLY_DEMONSTRATED_ONLY`. Neither bundle contains
new GPU measurements. Output directories are never overwritten.

The checkout was `feat/hardware-aware-general-lowering-ir` at `5793012`.
An attempted dedicated branch could not be created because `.git` is
read-only in this workspace. Unrelated existing modifications and untracked
files were left untouched. No commit, push, or PR was made.

## Next step and deferred roadmap

Run the fixed GPU batch on an accessible CUDA host. Require the fresh
correctness and topology gates, inspect raw timing dispersion, and compare
the real updated Batch 2 against the frozen control. If real evidence
changes selection reproducibly, a bounded Schedule Agent Prototype can use
the current deterministic two-batch components. If not, retain the null
result and adjust only the smallest supported part of the calibration or
candidate diversity rule before building an agent.

After the agent foundation, **HA-Compiler-1 — Alternate Open Compiler
Backend** may investigate a narrow optional MLIR/LLVM-NVPTX lowering for
`PREPARE_FORM` or `RECONSTRUCT` against the Triton backend. Preserve the
Matmul Inspector schedule IR as the source of schedule semantics. Compare
correctness, PTX and SASS, latency, and compiler provenance. The sequence is
HA-Schedule-10 → Bounded Schedule Agent Prototype → HA-Compiler-1. No alternate
compiler work is part of this milestone.
