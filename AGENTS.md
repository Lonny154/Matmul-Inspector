# Matmul Inspector — Codex Instructions

You are working on Matmul Inspector, an open-source kernel experimentation and
compiler-analysis toolkit.

## Mission

Matmul Inspector gives external coding agents a deterministic way to evaluate kernel
candidates. It validates correctness and runs controlled paired benchmarks. It analyzes PTX,
SASS, compiler output, and profiler evidence. It records failures, experiment provenance,
artifact hashes, and machine-readable evidence for reproducible candidate comparisons.
Codex, Claude Code, or other agents may propose kernels and experiments. The toolkit remains
the source of truth for correctness, timing, hashes, provenance, and hard resource limits.

Hardware-aware schedule search and alternative matrix multiplication algorithms are research
applications of the toolkit. They do not define its primary product direction. Do not claim
performance improvements without measurements or mathematical discovery without proof.

## Project context

Before significant development work, read `project_context.md` if present, `README.md`, and
the relevant retained research reports. Read `docs/hardware_aware_general_lowering_ir.md`
and `TENSOR_GAME_RESEARCH.md` for schedule or Tensor Game work. Read
`docs/trainium1_nki_gemm_case_study.md` for NKI or Trainium work.

Use `project_context.md`, retained research reports, and verified repository state for
current progress. If they disagree, inspect the actual checkout and explicitly report the
discrepancy. Do not keep time-sensitive milestone status in this file.

## Development and validation

- Preserve existing benchmarks, experimental results, failures, and negative findings.
  Prefer incremental changes with relevant tests.
- Keep analysis utilities CPU-compatible. CUDA, Triton, NKI, Trainium hardware, and OR-Tools
  are optional unless a specific feature requires them. Ordinary CPU tests must not require
  accelerator hardware or constraint solving.
- Do not modify CUDA, Triton, or NKI kernels unless the task explicitly requires it. Follow
  repository structure and document new commands and analysis features.
- Run relevant tests and report failed or blocked checks. Keep changes scoped to the task.
- Do not commit generated NEFF, CUBIN, or profiler artifacts unless a study explicitly calls
  for a small retained artifact.

## Experimental standards

- Correctness gates performance. Never loosen tolerances merely to rescue a candidate.
  Record numerical precision and reference policies.
- Use controlled paired comparisons, repeat measurements, and appropriate A/A controls.
  Never claim a speedup from a single timing result or causality without controls that
  isolate it.
- Distinguish CUDA-event device timing from host timing. Treat eager execution and CUDA
  Graph replay as separate modes; record the mode for each result.
- Distinguish source code, compiler IR, PTX, and final SASS or other machine code. Static
  instruction counts are not dynamic counts. Compiler resource reports are not measured
  runtime behavior.
- Static modeled HBM traffic is not measured DRAM traffic. `ON_CHIP` is an abstract schedule
  location until hardware evidence shows physical placement.
- IR validity and structural lowerability do not establish numerical correctness. Numerical
  correctness does not establish performance.
- Bound experiments with explicit time, candidate, and resource limits. Separate
  observations from interpretations and state uncertainty.

## Evidence and artifacts

- Preserve raw measurements, relevant compiler and profiler outputs, experiment
  configurations, source identity, environment metadata, software versions, hardware
  configuration, and hashes.
- Prefer deterministic experiment manifests. Keep timestamps and other volatile metadata
  separate from stable configuration identifiers when practical.
- Keep large generated artifacts outside Git when appropriate, but retain stable references,
  hashes, and reproduction instructions. Preserve retained artifacts required by a study.
- Never alter historical artifacts to force hash matches. Label regenerated evidence
  separately from original evidence.
- Report missing historical evidence as unavailable or unverified, never as passed. Do not
  fabricate benchmark measurements or profiler results.

## Repository and checkout safety

Before modifying anything, read this file and run:

```bash
git branch --show-current
git rev-parse --short HEAD
git status --short
```

Inspect the relevant diff before editing.
Preserve unrelated work and untracked files. Never reset, stash, delete, or modify them.
Stage only task files; never use `git add -A` when unrelated files are present. Never push,
merge, or open a pull request without explicit authorization.

Keep valuable research branches and worktrees in a persistent location such as
`~/projects/`. Never leave the only copy of committed research in `/tmp` or another
disposable directory. Temporary checkouts are for disposable experiments, not sole storage
of completed milestones. Before removing a research checkout, verify that required commits
and artifacts have durable copies. A local commit is not a backup. Report when a milestone
commit exists only locally, including its branch, full commit SHA, checkout location, and
publication status.

## Research milestones

For each milestone, state the hypothesis and baseline, make the smallest useful change,
validate correctness, run bounded experiments where applicable, preserve raw evidence,
separate observation from interpretation, document limits, assign an explicit decision
classification, and recommend the next step. Do not repeat a completed track without a
specific new hypothesis.

Before calling a milestone complete, run relevant checks, validate retained evidence, and
review the final diff for correctness and unrelated changes. If required checks pass, stage
only milestone files and make a descriptive local commit. If checks fail or changes cannot
be isolated, report the blocker and do not classify the work as a validated milestone. An
explicitly requested recovery or preservation checkpoint may be committed with incomplete
validation if its commit and documentation clearly identify every failed or blocked check.
Never push automatically.

Each milestone report must include:

- Branch, full commit SHA, checkout location, and local-only or published status.
- Hypothesis, control or baseline, and implemented changes.
- Correctness results and benchmark results when applicable.
- Evidence and artifact paths, including failed or blocked tests.
- Limitations, uncertainty, decision classification, and next recommended step.
- Remaining uncommitted changes that affect the milestone.

## Lowering architecture

The bounded schedule-semantics IR uses `PREPARE_FORM`, `RECOMPUTE_FORM`, `GEMM`,
`RECONSTRUCT`, and `FINALIZE_OUTPUT`. Its value classes are `INPUT`, `PREPARED_FORM`,
`GEMM_PRODUCT`, `PARTIAL_OUTPUT`, and `FINAL_OUTPUT`. Do not return to whole-schedule
template proliferation without evidence that the general IR cannot represent the required
semantics.

Any bounded search or agent workflow must keep legality, lowering, correctness,
benchmarking, evidence capture, and result-driven selection under deterministic gates and an
explicit budget.

## Collaboration and writing

Explain significant architectural decisions and limitations. Favor readable, modular
implementations and reproducible technical reports. Write in a friendly, straightforward,
conversational style in explanations, documentation, case studies, reports, and drafted
messages. Use short sentences and few commas. Avoid corporate language and unnecessary
transitions. Fix spelling and grammar without losing the user's natural voice. Keep technical
details precise and claims tied to evidence.
