# Tensor Game Research History

**Last verified:** October 18, 2026

**Verified lineage:** `c78f1ba` through `be4bd85`

**Status:** No independent rank-7 algorithm has been discovered. Known conventional rank-8 and Strassen rank-7 trajectories are correctness references.

## Scope and evidence

The project studies exact decomposition of the 2x2 matrix-multiplication tensor. Factors are usually nonzero vectors in `{-1,0,1}^4`. The raw rank-one domain has `80^3 = 512,000` triples and 128,000 canonical tensors after sign equivalence. Oracle-assisted results are labeled and are not discoveries.

The overview is `docs/tensor_game.md`; chronology is in `docs/tensor_game_optimization_history.md`; raw outputs are under `artifacts/tensor-game/`.

## Search and learning baseline

Commits `c78f1ba`–`e633567` established the exact environment, MCTS, synthetic data, learning, replay, held-out protocols, and hardened providers. The 1,800-trial held-out study found 24/180 successes for frozen-policy configurations, 0/180 for learned policy and joint variants, and 18/180 for residual variants. Learned values did not change residual outcomes. See `docs/tensor_game_offline_learning_case_study.md` and `artifacts/tensor-game/tg4b2c-heldout-v1-analysis/`.

Diagnostics then located the main failure in proposal coverage and ranking. Frozen top-12 and a hybrid recovered 24/24 historical discoveries while trained top-12 recovered 0/24. All policies solved 0/48 trials on fresh targets. A later held-out evaluation found learned providers at 0/24 and residual search at 24/24. Evidence spans `docs/tensor_game_policy_regression.md` through `docs/tensor_game_adaptive_routing.md`.

## `bounded_128` and search benchmarks

Commits `873a6da`, `f52e200`, and `e633567` developed and integrated the bounded residual provider.

- TG-4Q: `bounded_128` matched exhaustive coverage at 28/30 states and 31/44 positives. Mean proposal latency was 29.03 ms versus 47.03 ms.
- TG-4R: both providers succeeded in 36/36 trials on 12 fresh targets. Bounded mean search was 1.23x faster.
- TG-4S: both succeeded in 33/48 trials on 16 new targets. Bounded was 1.24x faster but used 1.93 MB mean traced peak memory.
- TG-4T replaced materialized ranking with a bounded heap and preserved frozen replay exactly. See `docs/tensor_game_bounded_memory.md`.
- TG-4V: both succeeded in 63/72 trials on 24 structurally targeted targets. Bounded was 2.16x faster on mean and faster in 71/72 pairs. Exhaustive remained the default. See `docs/tensor_game_default_readiness.md`.

These are CPU search results, not a new multiplication algorithm.

## Hardware cost and Pareto analysis

Commit `726599b` added exact descriptors for rank, preparation additions, reconstruction additions, coefficient scaling, and support plus deterministic Pareto analysis. Recorded CPU overhead was small: 210.66 ms mean with analysis versus 207.81 ms baseline. Static cost is not GPU time. See `docs/tensor_game_hardware_cost_study.md` and `artifacts/tensor-game/hardware-cost-v1/`.

Commit `6f15fa5` tested hardware-aware reranking across 30 trials. It changed no selection because reached candidates all had zero local preparation additions and residual reduction one. Rank-8 remained conventional and rank-7 unsolved. GPU evaluation was skipped because no new representation existed. See `docs/tensor_game_hardware_aware_search_study.md`.

## Triton lowering, TF32x3, and fusion

Commits `e7b72a4` and `670e6ea` added and validated conventional and Strassen Triton lowerings on an RTX 4060 Ti. The materialized path prepared forms, ran batched rank GEMMs, and reconstructed outputs. Across sizes 128–1024, conventional lowering was 5.62x–18.63x slower than cuBLAS; Strassen was 9.5%–14.9% slower than conventional. See `docs/tensor_game_triton_lowering_study.md`.

Commit `f7e0f01` compared TF32 and TF32x3. TF32x3 improved accuracy but increased aggregate GEMM-only phase time by about 1.92x conventional and 2.03x Strassen. Complete-operation ratios varied with size and noise.

Commit `ec9c8b0` fused input preparation. Complete latency improved 1.408x–1.869x conventional and 1.459x–2.093x Strassen, but remained 2.19x–8.63x slower than cuBLAS. See `docs/tensor_game_fused_lowering_study.md`.

Commit `c11b5cd` fused output reconstruction. It improved complete latency 1.538x–1.647x conventional and 1.489x–1.889x Strassen. The best paths still remained 1.95x–7.71x slower than cuBLAS. See `docs/tensor_game_output_fusion_study.md`.

## Timing stability and GPU cost calibration

Commit `4e1e118` documented warm-state and timing dispersion. At size 512, allocated-path CVs were 45.4% conventional and 43.0% Strassen; at 1024 they were 15.6% and 11.4%. Device, host, order, allocation, synchronization, and telemetry evidence must remain separate. See `docs/tensor_game_timing_diagnostics.md`.

Commit `835b4ce` calibrated 12 exact change-of-basis candidates. Median device latency ranged from 0.4013 to 0.5350 ms. Conventional anchor `30b13f831111` was fastest in that set. Static descriptors helped within-rank ordering but did not establish a universal cost model. See `docs/tensor_game_gpu_cost_calibration_study.md` and `artifacts/tensor-game/gpu-cost-calibration-v1/`.

## Change-of-basis generation

Commit `79409d5` generated equivalent forms with bounded unimodular integer changes of basis. All 1,024 attempts reconstructed exactly. Two duplicates collapsed, leaving 1,022 unique records: 511 conventional and 511 Strassen. They span 598 structural signatures and 35 primary cost points with maximum coefficient magnitude two. They are equivalent representations, not discoveries. See `docs/tensor_game_change_of_basis_study.md` and `artifacts/tensor-game/change-of-basis-v1/`.

## Multi-`w`, quotas, and provenance admission

Commit `faa8eb7` separated pair representability, output selection, filtering, exact scoring, and downstream search.

Commit `680c76d` improved known-action recall from 3/10 for `bounded_128` to 5/10 pair and 4/10 complete-action recall. Rank-8 search fell from 3/3 to 0/3 under the tested node budget, while evaluations rose from 97,920 to 7,864,320. See `docs/tensor_game_multi_w_provider_study.md`.

Commit `87585d0` added quota-preserving exact separable top-k output generation. One alternative slot retained 5/10 pair and 4/10 action recall while protecting baseline proposals, but rank-8 remained 0/3. See `docs/tensor_game_quota_provider_study.md`.

Commit `583b653` added provenance-aware beam quotas. Alternatives survived at every depth as intended. All tested settings remained 0/3 at rank 7 and 3/3 at rank 8 in the smaller matched setup. See `docs/tensor_game_provenance_beam_study.md`.

## Strassen action coverage

Commit `fc32aaa` evaluated all 5,040 Strassen term orderings. No provider admitted a complete first action. Baseline excluded 3,600 orderings at pair selection, 720 at output choice, and 720 at final admission. Quotas moved exclusions but remained 0/5,040. See `docs/tensor_game_w_coverage_study.md`.

Commit `286474c` tested sign-invariant structural pair diversity. It did not improve exact coverage or produce rank-7 success. See `docs/tensor_game_structural_pairs_study.md`.

## Two-step residual and subspace experiments

Commit `f9bc187` tested exact residual-subspace features and bounded two-step lookahead. Every Strassen two-term total remained negative. At the initial tensor, immediate, two-step, and structural top-12 each recalled 4/7 only in an oracle pool containing known actions; the ordinary provider recalled 0/7. The extra horizon did not rescue initial Strassen actions. See `docs/tensor_game_residual_lookahead_study.md` and its artifact directory.

## Exact rank-feasibility pruning

Commit `bba5718` added rational flattening-rank bounds. All 51 known-trajectory states passed. In a failed rank-7 replay, 12/21 sampled states and 34/89 provider children were provably infeasible. Checks took 13.34 ms total, about 2.3% of provider time in that diagnostic. The bound is useful late but cannot distinguish rank 7 from rank 8 at the root. See `docs/tensor_game_rank_feasibility_study.md`.

Commits `7e4d16f` and `05a4866` tested opt-in late pruning and combined it with quota proposals and provenance admission. They did not establish rank-7 discovery or justify a default change. See `docs/tensor_game_feasibility_gate_study.md` and `docs/tensor_game_feasible_exploration_study.md`.

## Meet-in-the-middle exploration

Commit `ad1c8cb` implemented bounded exact 3+4 complement matching. The isolated seven-term Strassen oracle produced 35 complementary partitions and one deduplicated decomposition, validating mechanics. Ordinary residual, structural, and mixed pools produced no complement or decomposition. Complete size-20 prefixes and specific size-64 samples are ruled out, not the full domain. See `docs/tensor_game_meet_in_middle_study.md` and `artifacts/tensor-game/meet-in-middle-v1/`.

## Algebraic pools and pair-to-output completion

Commit `fc0533f` used exact slices and rational subspaces. No root ternary slice combination had matrix rank one, so the simplest hypothesis failed. A rank-two fallback generated 192 pairs, exact-scored 15,360 actions, and produced 662 unique actions. At pool size 128, algebraic-only recalled six of seven held-out Strassen mode-pairs and one sign-equivalent complete term. Residual and structural pools recalled five and three pairs with no complete terms. MITM found no decomposition. See `docs/tensor_game_algebraic_pools_study.md` and its artifact directory.

Commit `be4bd85` froze those 192 pairs and compared output completion rules. At pool size 128 with two outputs per pair, slice projection recalled 2/7 complete Strassen actions versus 1/7 for residual. Larger local quotas caused non-monotonic recall because global residual ranking displaced cancellation-heavy actions. Matched MITM runs again found no complement. See `docs/tensor_game_algebraic_w_completion_study.md` and `artifacts/tensor-game/algebraic-w-completion-v1/`.

## Negative findings and limitations

- Learned policies did not generalize and often removed critical actions.
- Residual scoring favors conventional sparse actions; Strassen initially increases residual norm.
- Two-step lookahead still rejects Strassen pairs. Flattening rank helps only late.
- Multi-`w`, quotas, and provenance preserve alternatives but add work and interference without a joint objective.
- Static costs do not replace stable GPU measurement.
- Fused lowerings improved substantially but stayed slower than cuBLAS in all reported sizes.
- MITM oracle recovery validates code, not discovery. Sampled ordinary pools do not rule out the full domain.
- Change-of-basis and held-out Strassen overlap are not independent algorithms.

## Proposed next milestone (planned work)

Build a bounded joint algebraic solver with SAT/SMT or constraint programming. Start with frozen finite pair/action pools. Select exactly seven canonical terms and enforce exact reconstruction of all 64 tensor entries. Model cross-term cancellation directly. Add symmetry breaking, exact arithmetic, strict time and memory limits, deterministic settings, explicit SAT/UNSAT/UNKNOWN results, and independent reconstruction of any model.

First validate a CPU-only offline prototype on isolated conventional and Strassen oracle pools, then frozen ordinary and algebraic pools without oracle leakage. Do not integrate it into production search or claim discovery until an independently generated decomposition passes exact reconstruction, provenance, and reproducibility checks.
