# Matmul Inspector — Project Context

**Context updated:** October 21, 2026
**Repository:** https://github.com/Lonny154/Matmul-Inspector
**Verified boundary:** Local Git ancestry, committed case studies, selected offline evidence replays, and GitHub branch state were checked on October 21. This integration branch includes research through MI-Research-8 at `c2b8de73981662193284f231c8dd67b5fa1cbdae`. It excludes the separately committed MI-Research-9 study.

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

## Integration boundary and next work

- The GitHub-published feature checkpoint is `feat/mi-research-3c` at `f2d6607653d04972bbc04757bd0ac5bf1864af6d`. The MI-Research-4–8 commits descend from it. This branch is the through-8 integration candidate; branch publication or a PR does not mean these commits have merged into `main` or shipped in a package release.
- `origin/main` was `5df8f5ba3d76b9acc2e6af2b5c2e584ed21f5543` when checked on October 21. GitHub reported no open PR before this integration work. Recheck remote state and CI before a merge.
- At the MI-Research-8 checkpoint, **MI-Research-9 — Output-Slice-Aware Algebraic Cost Analysis** was the next planned CPU-only study. It was subsequently completed and committed locally as `f90fdbaf52377d7b634cb14692b234c5268196a7` on a separate branch. It is deliberately outside this through-8 integration candidate. Do not describe it as unfinished or already published.
- Review the full research PR, retained artifacts, CI, and release boundaries before any merge. Further internal model-agent or backend work needs a specific evidence-backed reason.
- The pre-existing uncommitted `AGENTS.md` edit and unrelated untracked files are outside this integration. Preserve them.

## Maintenance rules

- Inspect `git status`, current branch/HEAD, remote refs, `AGENTS.md`, and CI before mutations.
- Preserve mathematical correctness, source identity, numerical precision contracts, GPU mode, raw benchmarks, hashes, controls, negative findings, and unrelated files.
- Label facts as user-reported, locally verified, or remote-verified when boundaries matter. Do not equate compilation, static cost, profiler counts, and runtime latency.
- The primary product is an accessible, agent-compatible deterministic kernel experiment tool; keep the Python toolkit CPU-compatible and GPU features optional.
- Never claim a mathematical discovery, kernel speedup, PR merge, push, or release without corresponding verified evidence.
