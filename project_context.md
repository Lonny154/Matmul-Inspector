# Matmul Inspector — Project Context

**Context updated:** October 23, 2026
**Repository:** https://github.com/Lonny154/Matmul-Inspector
**Verified boundary:** GitHub PR and repository documents were checked October 23. PRs #26 (MI-Research-4–8), #27 (MI-Research-9), #29 (durable Codex instructions), and #28 (MI-Research-10 source recovery) are merged into `main`. The MI-Research-10 recovery was locally validated, but its lost original GPU evidence and frozen-plan bytes are still unavailable. The checked GitHub state does not establish that any separate package release contains these research changes.

## Mission and product direction

Matmul Inspector is an open-source, evidence-first low-level kernel engineering toolkit and research instrument. Its primary product purpose is to automate tedious workflows: kernel correctness validation, controlled paired benchmarking, PTX/SASS/compiler inspection, profiler evidence capture, experiment provenance, failure recording, and reproducible comparisons.

The intended AI integration is **external-agent-compatible tooling**. Codex, Claude Code, or other agents can propose and implement kernels; Matmul Inspector provides deterministic experiment contracts, resource and correctness gates, machine-readable evidence, and replay. The internal bounded Schedule Agent was a useful research track but is not the primary product or a substitute for a capable coding agent.

Research is a second use of the tool: evaluate hardware-aware execution strategies and exact alternative matrix-multiplication decompositions. Mathematical rank, addition counts, source dots, compiler IR, static PTX, final machine instructions, numerical error, device latency, and physical memory traffic are separate evidence layers. A lower arithmetic count or compiler resource count does not by itself imply a faster kernel. No independent novel matrix-multiplication algorithm has yet been claimed.

## Baseline platforms and release

- Primary GPU: NVIDIA RTX 4060 Ti (SM 8.9), 34 SMs, under Ubuntu/WSL2. CUDA/Triton hardware paths remain optional where possible; CPU-only analysis must remain functional.
- AWS Trainium1 NKI (`trn1.2xlarge`) work established tiled GEMM, hardware profiling, BF16/tile-width sensitivity, and negative logical K-blocking results. See the [NKI case study](docs/trainium1_nki_gemm_case_study.md).
- Tagged release `matmul-inspector==0.3.0` provides saved PTX, SASS, NKI, and unified artifact inspection on Python 3.10–3.13. Research branch workflows are not automatically part of that released package. See the [release notes](docs/release_notes_0_3_0.md).
- Existing Transformer FC1 workload: FP16, M=512, K=768, N=3072. Earlier Triton/cublas differences were small and should not be treated as stable universal speedups.

## Completed tracks: schedule search, agent feasibility, and product workflows

### Tensor Game / hardware-aware schedule foundation

- Exact conventional rank-8 and Strassen rank-7 references, MCTS/learning, bounded residual search, algebraic pools, exact reconstruction, feasibility pruning, and MITM diagnostics exist. Earlier learning/search approaches did not independently discover a new rank-7 algorithm. See the [Tensor Game research history](TENSOR_GAME_RESEARCH.md).
- [HA-Schedule-9](docs/hardware_aware_general_lowering_ir.md) supplied bounded general schedule-lowering IR and structural lowerability for 156 retained schedule-size records; 132/156 passed the established GPU numerical correctness gate, while 24 large-size Strassen cases did not.
- [HA-Schedule-10](docs/ha_schedule_10_calibration_reselection.md) demonstrated real GPU evidence changing deterministic Batch 2 selection relative to a frozen no-update control. This is evidence-driven reselection, not a demonstrated faster Batch 2.
- The [bounded Schedule Agent](docs/schedule_agent_prototype.md) completed two GPU batches using replayed proposals (Decision B). [Live-1](docs/schedule_agent_live_1.md) completed two model-driven GPU batches, but did not establish evidence-grounded reasoning or a performance win. [Live-2](docs/schedule_agent_live_2.md) had 0/3 valid matched model decision pairs because of duplicate hash proposals; its negative evidence is retained. Further internal model-agent development is deferred in favor of external-agent tooling.

### Matmul Inspector developer workflow

- [MI-Workflow-1](docs/mi_workflow_1.md): baseline/candidate correctness, repeated CUDA-event benchmarking, compiler artifacts, self-contained report and hashed evidence bundle. The initial FC1 comparison was inconclusive.
- [MI-Workflow-2](docs/mi_workflow_2.md): paired Triton configuration A/B comparisons with raw paired differences, uncertainties, and integrated evidence. Stage-count comparison remained inconclusive.
- [MI-Workflow-3](docs/mi_workflow_3.md): explicit opt-in trusted local Python/Triton source via `--allow-trusted-local-source`; no registry edit. Callable contract: `matmul_into(a, b, out, parameters)`. This is **not a sandbox**: local source executes with process privileges, so secrets should be removed from the environment before running unfamiliar kernels.
- [MI-Dogfood-1/2](docs/mi_dogfood_2.md): Codex used the tool to evaluate local kernel variants without modifying benchmark internals. Correctness and evidence handling were useful; none of the small FC1 variants established a repeatable advantage.
- [MI-Timing-1/2](docs/mi_timing_2.md): WSL2 host clock discrepancy: `CLOCK_MONOTONIC_RAW` approximately 1.0545× Python `perf_counter`, while matched CUDA-event and RAW intervals agreed closely. Three-clock diagnostics and A/A controls were integrated. This does **not** invalidate CUDA-event paired measurements automatically; A/A and session variability still limit small-performance claims.
- [MI-Timing-3](docs/mi_timing_3.md): A/A-before vs. A/A-after warm-state investigation was inconclusive; the default A/A-before policy was unchanged.

## Completed hardware research

### Split-K and numerical reference policy — MI-Research-1–3C

- FC1 tiling, warps, and grouped-output investigations did not establish repeatable performance improvements beyond A/A variability.
- Split-2 on low-output-parallelism GEMM (M=32, K=4096, N=2048, FP16) exposed 32 partial GEMM CTAs instead of 16 conventional output CTAs, with a deterministic reduction. Eager comparisons suffered recurring slow blocks.
- Under fixed-address CUDA Graph replay, Split-2 achieved supported advantages over conventional Triton at M=32 in repeated sessions; comparisons at M=128 were inconclusive, and higher M lacked supported crossover evidence.
- [MI-Research-3B](docs/mi_research_3b.md) found that default PyTorch/cuBLAS FP16 reduced-precision reduction behavior caused a large-M reference mismatch. Explicit FP32-rounded reference policy enabled the previously blocked numerical checks without relaxing the original tolerances. Historical failures remain intact.
- [MI-Research-3C](docs/mi_research_3c.md) established no reproducible Split-K crossover or automatic shape selector. Execution mode and precision policy must always be recorded.

### Single-kernel Strassen and algebraic representation — MI-Research-4–5

- [Fused one-level Strassen](docs/mi_research_4.md) (rank 7) passed correctness and one-launch topology validation at N=64 and N=128 with no explicit product scratch. Neither size established a speedup.
- [Matched fused rank-8 versus Strassen rank-7](docs/mi_research_4b.md) at N=128: 8 vs. 7 TTIR dot operations; 192 vs. 168 static PTX MMA instructions. Two paired sessions were inconclusive.
- [Scaling to N=256 and N=512](docs/mi_research_4c.md) was compiler-feasible, numerically correct, and reported no spills. A rank-7 lead at N=512 met the decision rule in one session but not the second; no reproducible advantage over rank-8 or optimized GEMM was established.
- [Strassen-equivalent algebraic variants](docs/mi_research_5.md) and schedule reorders altered compiler register use (e.g. 157 to 148 registers/thread) without supported timing improvements. Rank-7 representation search was not expanded.

### Larger 3×3 tensors, Laderman, and Perminov — MI-Research-6–8

- [MI-Research-6](docs/mi_research_6.md): conventional 3×3 (27 products) and known Laderman (23 products) exactly reconstructed all 729 tensor coefficients and passed matrix-valued block tests. At N=96, GPU-verified staged operations used 28 and 24 physical kernels, respectively, with FP32 product scratch; eager timing was inconclusive.
- [MI-Research-6B](docs/mi_research_6b.md): fixed-address CUDA Graph replay passed correctness for both staged algorithms. Laderman's small paired leads were inconclusive due to A/A variability.
- The [published paper](https://arxiv.org/abs/2607.28676) and [certificate](https://github.com/trylogical/cn122_add55) give an optimized 55-addition straight-line circuit for Perminov's existing rank-23 `cr58_cn122` factorization, **not** classical Laderman. The 55- and 58-addition circuits have identical rank-23 factor maps. Whole-circuit DAG depth was 15 vs. 12. Both were independently verified exactly.
- [MI-Research-6C](docs/mi_research_6c.md): fused 55-addition Perminov compiled as one physical GPU launch at N=96, passed six correctness cases twice, and used no explicit HBM product scratch. Original output-owned mapping used 4 CTAs, 23 source dots/CTA, 255 compiler-reported registers/thread; subsequent inspection reported `n_spills=422`. This established hardware feasibility, not speedup.
- [MI-Research-6D–6F](docs/mi_research_6f.md): fused Perminov was much faster than **staged Laderman** in observed graph medians. 6D had inconclusive paired decisions due to A/A variability; two fresh 6E sessions returned `B_FASTER` under the unchanged rule. The source of unusually slow staged blocks remained unresolved after focused tracing. This result does not isolate addition-circuit effects or imply conventional GEMM competitiveness.
- [MI-Research-7](docs/mi_research_7.md): matched CUDA Graph comparisons at N=96 established that fixed conventional Triton (~1.4 µs) and cuBLAS (~1.85 µs) beat the original fused Perminov (~15.7 µs) in two sessions. Eager comparisons were inconclusive. The Triton reference was one fixed, correctly implemented tile configuration, not a globally autotuned baseline.
- [MI-Research-7B](docs/mi_research_7b.md): remapping Perminov from 4 to 36 CTAs reduced graph median latency from ~15.7–16.3 to ~9.6–10.0 µs, passing the paired rule twice, despite 2.78× more source-level dots. Compiler `n_spills` fell from 422 to 166. cuBLAS remained faster (~2.53 vs. ~10.05 µs) in a triggered matched comparison.
- [MI-Research-7C](docs/mi_research_7c.md): Nsight Compute observed 4/34 vs. 34/34 SM usage; executed local loads 9,632 vs. 4,000 and tensor-pipe instructions 8,832 vs. 24,576. Achieved occupancy on active SMs remained ~8–9% and measured DRAM reads increased. Wider SM coverage and reduced local-load activity are plausible contributors, but their causal latency contributions are not isolated.
- [MI-Research-8](docs/mi_research_8.md): matched 55-addition and 58-addition Perminov circuits on the same 36-CTA output-owned mapping passed exact and GPU checks. Whole-circuit addition count was **not** predictive of output-sliced work: 55-addition induced 704 gate instances and 256 dot instances across the grid, versus 568 and 232 for 58-addition. The 58-addition kernel also emitted fewer static arithmetic/MMA instructions and fewer compiler-reported spills (100 vs. 166). Both paired timing decisions were inconclusive. No GPU speedup is claimed.

### Prospective source-cost screening and recovery — MI-Research-9–10

- [MI-Research-9](docs/mi_research_9.md) is **merged** ([PR #27](https://github.com/Lonny154/Matmul-Inspector/pull/27), October 21). This CPU-only output-slice-aware analyzer replayed both exact Perminov circuits under four- and 36-CTA ownership. For four CTAs, 55/58 yielded 220/232 gate instances and 92/92 dot instances per grid; modeled peak live bytes/CTA were 59,392/63,488. For 36 CTAs, 55/58 yielded 704/568 gates and 256/232 dots; modeled live bytes were 38,912/36,864. The ranking of source-work/storage tradeoffs changes with the mapping. These are *structural modeled costs*, not GPU latency, actual allocation, or measured traffic.
- [MI-Research-10](docs/mi_research_10.md) historically screened the previously uncompiled **Perminov-58, four-CTA** combination against Perminov-55, four-CTA. The original study text reports equal static TTIR dot and PTX MMA counts, higher compiler-reported spills for 58, and one inconclusive matched CUDA Graph timing session. The historical plan prose mistakenly said **4,096 fewer** modeled live bytes for 58, while both frozen descriptor vectors show **4,096 more** (63,488 vs 59,392). The contradiction invalidates a clean preregistered liveness prediction. Its raw GPU validation, paired timing, analysis, and final manifest did **not survive** the temporary checkout's deletion. Treat original GPU observations as historical report claims, not independently replayed evidence in the recovery.
- [MI-Research-10 source recovery](docs/mi_research_10_recovery.md) is **merged** ([PR #28](https://github.com/Lonny154/Matmul-Inspector/pull/28), October 23; merge commit `487ff0e08a435d802975c29051778192bdd989cd`). Seventeen recorded Codex patches reproduced the source. The reconstructed four-CTA 58 kernel has the original SHA-256 `fa0715e83b773816966ec70ea6f536d16d0f9ca72be0753ed16875bb504d0cd0`. Both candidate descriptor vectors, source hashes, and five upstream evidence hashes matched saved session outputs.
- The regenerated frozen plan SHA-256 is `0063d03a5b7a5f9727a968a709a164072d060b62231c67b73c6d5fc37d4482cd`; the original recorded plan SHA-256 was `89952d7ecf4a08764f768cadfd827940d64352f0921dd5b7c3b0db6e1490b55d`. The old `frozen_at_utc` value and plan bytes are missing. Repeated regenerations agree after removing the new timestamp, but **byte equivalence to the lost plan is unverified**. Neither artifact was altered to force a hash match.
- After the recovery test adjustment (commit `6ceee6cbe6145330078374fb5149f7a85257bfd6`), the relevant MI-Research-9/10 CPU test selection returned **28 passed, 1 skipped**. The skip applies only when *all* historical MI-Research-10 analysis, manifest, GPU validation, and paired timing evidence are absent. Partial, corrupt, or mismatched evidence is still verified strictly and must fail. The historical offline analysis replay remains unavailable. The preregistration checker rejects the erroneous `4,096 fewer` descriptor claim and blocks new compilation from the legacy frozen plan without structured claims. No new GPU compilation or timing ran during recovery.


## Integration boundary and next work

- **GitHub integration verified October 23:** [PR #26](https://github.com/Lonny154/Matmul-Inspector/pull/26) merged research through MI-Research-8 and [PR #27](https://github.com/Lonny154/Matmul-Inspector/pull/27) merged MI-Research-9 on October 21. [PR #29](https://github.com/Lonny154/Matmul-Inspector/pull/29) merged durable Codex workflow instructions on October 23 (commit `b722a84d3bcef96a9733c86527fc7dca4a4199c6`). [PR #28](https://github.com/Lonny154/Matmul-Inspector/pull/28) merged MI-Research-10 **source recovery** on October 23. These merges do not imply the original MI-Research-10 GPU artifacts were recovered or that a new PyPI release shipped.
- **Primary next step:** Design a *new*, correctly preregistered four-CTA Perminov screening run with deterministic plan identity and its own GPU validation, compiler artifacts, raw paired timings, A/A controls, and hashed evidence. Do not use the historical plan to bypass the preregistration gate. Do not label any fresh measurements as recovered historical evidence. Keep correctness and topology as hard gates and separate static source descriptors from measured compiler and GPU behavior. Run only with explicit resource limits and a stable hardware environment.
- **Product direction:** Continue developing Matmul Inspector as deterministic, external-agent-compatible correctness, benchmarking, compiler/profiler-inspection, and provenance tooling. Further internal autonomous-agent work requires a new evidence-backed reason; do not confuse agent text with validated measurements.
- **Checkout safety:** The older `~/projects/matmul-inspector` checkout was last reported on `feat/mi-research-8-integration` at `1637aa4` with an uncommitted `AGENTS.md` edit and unrelated files. A malformed `refs/codex/turn-diffs/...` reference previously blocked its fetch. The new `AGENTS.md` is merged remotely, but do not assume that older local checkout is clean or updated. Preserve its uncommitted work; use a clean clone/worktree or deliberately repair the ref before manipulating it. Check current branch, HEAD, status, and upstream refs first.
- **Documentation sync:** Keep repository `project_context.md` and the ChatGPT Matmul-Inspector Project's uploaded `project_context.md` aligned after verified milestone updates. Project attachments are snapshots and do not synchronize automatically with GitHub. `AGENTS.md` holds durable operating rules; this file holds current facts; retained study reports hold experimental detail.

## Maintenance rules

- Inspect `git status`, current branch/HEAD, remote refs, `AGENTS.md`, and CI before mutations.
- Preserve mathematical correctness, source identity, numerical precision contracts, GPU mode, raw benchmarks, hashes, controls, negative findings, and unrelated files.
- Label facts as user-reported, locally verified, or remote-verified when boundaries matter. Do not equate compilation, static cost, profiler counts, and runtime latency.
- The primary product is an accessible, agent-compatible deterministic kernel experiment tool; keep the Python toolkit CPU-compatible and GPU features optional.
- Never claim a mathematical discovery, kernel speedup, PR merge, push, or release without corresponding verified evidence.
