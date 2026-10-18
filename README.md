# Matmul Inspector

**See what your GPU or AI-accelerator compiler actually produced.**

Matmul Inspector turns saved NVIDIA PTX, NVIDIA SASS, and AWS Neuron NKI compiler
artifacts into readable JSON and Markdown reports. This matters because a small
source change can disappear in an intermediate representation, produce different
machine instructions, change numerical results, or affect performance—and none
of those outcomes can safely be inferred from another.

Install the Python toolkit from [PyPI](https://pypi.org/project/matmul-inspector/):

```sh
python -m pip install matmul-inspector==0.3.0
matmul-inspector-python --version
```

Version 0.3.0 supports Python 3.10–3.13. Saved-artifact analysis has no mandatory
CUDA, AWS Neuron, GPU, or native-library dependency. The optional experiment
commands require an NVIDIA GPU, a compatible driver, CUDA-enabled PyTorch, and
Triton; installing Matmul Inspector does not install or configure a driver and
does not guarantee that a particular GPU/software combination is compatible.

## Why Matmul Inspector?

Kernel work crosses several evidence layers:

```text
source -- compilation --> compiler IR / PTX -- lowering --> SASS / NEFF
SASS / NEFF -- execution -------------------------------> numerical outputs
            `-- timed/profiled execution ---------------> performance measurements
```

A source-level optimization may be canonicalized away. A PTX instruction need
not map one-to-one to a hardware instruction. Fewer static instructions do not
guarantee fewer dynamically executed instructions or lower latency. A faster run
is not successful if it violates the required numerical tolerance.

Matmul Inspector keeps these layers separate while making them easy to compare.

## Quick start: compare two PTX modules

This CPU-only example compares one fused multiply-add instruction with separate
multiply and add instructions. It demonstrates a **static PTX difference**; it
does not show the final machine code or predict performance.

The inline PTX fixtures are synthetic inputs for static analysis, not executable
numerical kernels. They contain no initialized inputs or observable stored output.

Create the fused module:

```sh
cat > fused.ptx <<'PTX'
.version 8.0
.target sm_80
.address_size 64

.visible .entry update_value()
{
    .reg .f32 %f<4>;
    fma.rn.f32 %f3, %f1, %f2, %f0;
    ret;
}
PTX
```

Create the separate-operation module:

```sh
cat > separate.ptx <<'PTX'
.version 8.0
.target sm_80
.address_size 64

.visible .entry update_value()
{
    .reg .f32 %f<5>;
    mul.rn.f32 %f3, %f1, %f2;
    add.rn.f32 %f4, %f3, %f0;
    ret;
}
PTX
```

Compare them with the explicit Python entry point:

```sh
matmul-inspector-python ptx compare \
  --baseline fused.ptx \
  --candidate separate.ptx \
  --kernel update_value \
  --output comparison
```

The command writes `comparison/comparison.json`, `comparison/report.md`, two
normalized PTX files, and a bounded unified diff. With the published 0.1.0 wheel,
the generated Markdown and JSON report:

| Selected-entry observation | Fused baseline | Separate candidate | Delta |
|---|---:|---:|---:|
| `fma.rn.f32` | 1 | 0 | -1 |
| `mul.rn.f32` | 0 | 1 | +1 |
| `add.rn.f32` | 0 | 1 | +1 |
| Arithmetic instructions | 1 | 2 | +1 |
| PTX-declared `f32` registers | 4 | 5 | +1 |

PTX registers are virtual declarations, not final physical register allocation,
hardware resource use, or occupancy. The output directory must not already exist;
this protects previous reports from accidental overwrite.

The package also installs `matmul-inspector`. The native C++ source build uses
the same name, so `matmul-inspector-python` is the unambiguous Python command when
both are on `PATH`.

## See it in action: FMA versus No-FMA

A controlled CUDA 13.3 experiment on an NVIDIA GeForce RTX 4060 Ti compared an
explicit fused multiply-add kernel with one using separate round-to-nearest
multiply and add operations. Inspection of the captured `sm_89` SASS found:

| Static selected-function observation | FMA | No-FMA |
|---|---:|---:|
| `FFMA` | 5 | 0 |
| `FMUL` | 0 | 5 |
| `FADD` | 0 | 5 |
| Compiler-recorded registers/thread | 28 | 28 |

This is hardware-specific final-machine-code evidence for that build. These are
static instruction counts, not dynamic execution counts. The inspector also kept
embedded division helpers separate from the selected kernel.

The same study recorded CUDA-event kernel timings and numerical comparisons:

| Square size | FMA median | No-FMA median | No-FMA vs FMA | No-FMA tolerance failures |
|---:|---:|---:|---:|---:|
| 256 | 0.034816 ms | 0.034752 ms | -0.18% | 0 |
| 512 | 0.260848 ms | 0.246784 ms | -5.39% | 25 |
| 1024 | 2.299328 ms | 2.364848 ms | +2.85% | 1,332 |

The timing direction changes with size, measurements used fixed
reference-before-candidate order, and the candidate fails the configured tolerance
at larger sizes. The static instruction difference therefore does **not** establish
the cause of any timing difference or demonstrate a successful optimization.
See the [PTX/SASS FMA case study](docs/cuda_ptx_sass_fma_case_study.md) and
[floating-point experiment report](docs/floating_point_experiments.md) for the
capture conditions, methodology, numerical evidence, and limitations.

## Capabilities

| Area | What it does | Availability |
|---|---|---|
| NVIDIA PTX | Inspect entries, instructions, virtual registers, shared-memory declarations, and conservative text/structure diffs | Python package |
| NVIDIA SASS | Inspect saved `nvdisasm` text, physical-register metadata, instruction addresses, and kernel/helper boundaries | Python package |
| AWS Neuron NKI | Compare MLIR, opcode/DMA summaries, NEFF hashes and sizes, provenance, and optional benchmark summaries | Python package |
| Unified reporting | Combine PTX, SASS, and benchmark evidence while separating identity mapping, provenance, completion, and correctness | Python package |
| Numerical laboratory | Compare FP32 results, ULPs, tolerances, accumulation order, FMA contraction, reductions, and saved experiments | Native source build |
| Performance laboratory | Run CPU/CUDA benchmarks, reliability analysis, crossover studies, roofline models, and reproducibility captures | Native source build and scripts |
| GPU experiment runner | Single experiments, batches, persistent sessions, allocating/preallocated modes, and explicitly trusted local Triton candidates | Python package; optional PyTorch/Triton and compatible NVIDIA environment |
| Research experiments | Additional Triton Transformer projections and Trainium NKI GEMM studies | Source checkout; additional toolchains/hardware |

The source tree also includes a CPU-only [TensorGame research environment](docs/tensor_game.md)
for exact matrix multiplication tensor decomposition. Its standard and Strassen
trajectories are known correctness references, not newly discovered algorithms.
A bounded [sampled MCTS planner](docs/tensor_game_mcts.md) supports deterministic
reference-guided validation and small unguided search experiments without neural
networks or hardware dependencies.
The [independent discovery study](docs/tensor_game_discovery.md) compares uniform,
residual-informed, and hybrid proposals under matched CPU search budgets.
The [synthetic demonstration engine](docs/tensor_game_synthetic_data.md) creates
exact, leakage-checked policy/value training examples without requiring an ML
framework.
The optional [policy/value learning baseline](docs/tensor_game_learning.md) trains
a small supervised PyTorch MLP and compares held-out proposal quality without
changing MCTS or adding reinforcement learning.
Its versioned evaluation audit reports explicit deterministic and seeded
denominators, paired outcomes, grouped uncertainty, and proposal-generation cost.
The optional [neural-guided MCTS experiment](docs/tensor_game_neural_mcts.md)
separates learned proposals from learned leaf values and reports five matched
ablations without updating the trained model.
The [neural value study](docs/tensor_game_value_study.md) compares trained,
untrained, constant, and handcrafted leaf values under fixed residual proposals.
The [value-scale study](docs/tensor_game_value_scale.md) traces PUCT mechanics and
separates optimistic value magnitude from learned state ranking.
The [experience replay guide](docs/tensor_game_experience_replay.md) records
actual MCTS-driven environment decisions and validates exact episode replay.
The optional [offline learning guide](docs/tensor_game_offline_learning.md)
trains policy and direct-return value targets on validated sampled MCTS supports.
Its TG-4A run is a pipeline smoke test, not evidence of improved discovery.
The [diverse experience guide](docs/tensor_game_diverse_experience.md) defines a
leakage-safe 200-target curriculum and staged MCTS collection audit for future
offline learning.
The [controlled offline ablation](docs/tensor_game_offline_ablation.md) compares
strict policy-only, value-only, and joint updates using validation-only selection
and a frozen held-out search plan.
The [held-out protocol addendum](docs/tensor_game_heldout_protocol.md) resolves
pre-execution path, comparison-matrix, bootstrap, and wall-time ambiguities while
leaving the original frozen plan unchanged.
The [offline-learning held-out case study](docs/tensor_game_offline_learning_case_study.md)
reports the completed 1,800-trial discovery evaluation and its exact evidence audit.
The [policy-regression investigation](docs/tensor_game_policy_regression.md) traces
the policy-only failure to critical one-step actions removed by bounded decoding.
The [candidate-coverage investigation](docs/tensor_game_candidate_coverage.md)
tests bounded recall, experiment-only proposal unions, and training support.
The [out-of-sample policy evaluation](docs/tensor_game_out_of_sample_policy.md)
compares frozen, trained, and equal-budget hybrid proposals on fresh targets.
The [search calibration study](docs/tensor_game_search_calibration.md) validates
reference trajectories and separates candidate coverage from search budget.
The [held-out policy and search evaluation](docs/tensor_game_heldout_evaluation.md)
compares all frozen candidate providers under the preregistered TG-4G protocol.
The [residual-supervision feasibility study](docs/tensor_game_residual_supervision.md)
audits teacher compatibility, supervision overlap, multi-step limits, and cost.
The [coverage-aware policy study](docs/tensor_game_coverage_training.md) tests
verified-positive supervision against a reproduced MCTS-only control.
The [coverage-retention investigation](docs/tensor_game_coverage_retention.md)
compares individual-positive supervision and selective frozen-policy retention.
The [prefix-coverage investigation](docs/tensor_game_prefix_coverage.md) traces
the exact beam stages and tests an experimental prefix-survival objective.
The [final-ranking investigation](docs/tensor_game_final_ranking.md) tests a
decoder-cutoff hinge under matched tiny-set training and validation.
The [proposal-coverage study](docs/tensor_game_proposal_coverage_study.md)
separates ternary action representability, deterministic output-factor choice,
heuristic pair filtering, exact scoring, and downstream search.
The [multi-w provider study](docs/tensor_game_multi_w_provider_study.md) tests
bounded output-factor alternatives and measures their recall, search impact, and
CPU evaluation cost without changing the default provider.
The [quota-provider study](docs/tensor_game_quota_provider_study.md) adds exact
separable top-k output generation and protects baseline proposals while tracing
remaining interference in global beam scheduling.
The [provenance-aware beam study](docs/tensor_game_provenance_beam_study.md)
adds optional per-edge admission quotas and depth-level interference diagnostics
without changing the default global beam policy.
The [Strassen w-coverage study](docs/tensor_game_w_coverage_study.md) exhaustively
separates pair, output-factor, complete-action, and admission cutoffs across all
5,040 known term orderings without changing the solver.
The [structural-pairs study](docs/tensor_game_structural_pairs_study.md) tests a
fixed-budget, sign-invariant pair-diversity quota and records its negative exact
coverage result alongside matched rank-7 and rank-8 searches.
The [residual-subspace and two-step lookahead study](docs/tensor_game_residual_lookahead_study.md)
tests exact short-horizon and algebraic residual diagnostics offline.
The [exact tensor-rank feasibility study](docs/tensor_game_rank_feasibility_study.md)
tests rational flattening-rank lower bounds as offline impossibility checks.
The [experimental feasibility-gate study](docs/tensor_game_feasibility_gate_study.md)
measures opt-in late-depth exact pruning under matched CPU search limits.
The [feasibility-guided exploration study](docs/tensor_game_feasible_exploration_study.md)
combines exact pruning with quota multi-`w` proposals and provenance admission.
The [meet-in-the-middle study](docs/tensor_game_meet_in_middle_study.md) tests
bounded exact 3+4 complement matching with separate ordinary and oracle pools.
The [algebraic-pools study](docs/tensor_game_algebraic_pools_study.md) derives
bounded action proposals from exact low-rank residual-slice combinations.
The [algebraic output-completion study](docs/tensor_game_algebraic_w_completion_study.md)
compares residual, structural, and exact slice-projection `w` selection on a
frozen algebraic pair manifest.
The [TG-Global-1 design](docs/tensor_game_tempered_search_design.md) documents a
separate CPU-only parallel-tempering prototype over complete rank-7 candidates
with bounded exact two-term repair and strict oracle/discovery separation.
The [TG-Global-2 study](docs/tensor_game_tempered_search_study.md) compares
matched independent and exchanged replicas with and without exact repair.
The [TG-Global-3 study](docs/tensor_game_completion_depth_study.md) audits
bounded two-, three-, and four-term completion and one exact recombination step.
The [prefix-generalization investigation](docs/tensor_game_prefix_generalization.md)
tests broader verified-positive coverage and audits beam-aware supervision cost.
The [policy-interference investigation](docs/tensor_game_policy_interference.md)
separates local gradient effects from independent-token combinatorial dilution.
The [conditional-policy study](docs/tensor_game_conditional_policy.md) implements
and evaluates an isolated `u -> v -> w` policy and conditional beam decoder.
The [adaptive-routing study](docs/tensor_game_adaptive_routing.md) compares
frozen, exhaustive residual, fixed-mix, and train-selected proposal routing cost.
The [bounded-residual study](docs/tensor_game_bounded_residual.md) evaluates a
continuous-bound coarse-to-fine approximation to exhaustive residual ranking.
The [held-out bounded-search study](docs/tensor_game_heldout_bounded_search.md)
compares bounded and exhaustive residual providers in paired end-to-end search.
The [provider-integration study](docs/tensor_game_provider_integration.md)
documents opt-in residual-provider selection and the expanded regression corpus.
The [bounded-memory study](docs/tensor_game_bounded_memory.md) validates an
exact-output-equivalent streaming heap selector for the opt-in bounded provider.
The [streaming integration report](docs/tensor_game_streaming_integration.md)
records its integration and permanent differential/search regressions.
The [default-readiness study](docs/tensor_game_default_readiness.md) evaluates
the integrated provider on an independent structurally targeted corpus. The
[optimization history](docs/tensor_game_optimization_history.md) records the
TG-4C through TG-4V decision trail, including negative results.
The [rollout decision](docs/tensor_game_rollout_decision.md) applies the final
release gates on a 150-target corpus while preserving exhaustive as the default.
The [cross-machine verification guide](docs/tensor_game_cross_machine_verification.md)
records the completed integrity-pinned, resumable TG-4X run on a separate AWS
CPU host and the current clean-checkout reproduction procedure.
The [default-rollout gate report](docs/tensor_game_default_rollout.md) records
the repaired TG-4Y release gates and the rollout of `bounded_128` as the public
default. Explicit `exhaustive` selection remains available as a fallback.
The [release-preparation summary](docs/tensor_game_release_preparation.md)
collects usage, evidence, limitations, and the remaining publication checklist.

Direct script entry points remain available from a source checkout, but the
installed dispatcher is normally simpler:

```sh
matmul-inspector-python ptx --help
matmul-inspector-python sass --help
matmul-inspector-python nki --help
matmul-inspector-python report --help
```

The installed package includes the end-to-end CUDA/Triton experiment runner. It
validates the established packaged FC1 candidate, compares it with
PyTorch/cuBLAS, alternates CUDA-event measurement order, and preserves successful
and failed records under the ignored `artifacts/experiments/` directory:

```sh
matmul-inspector-python experiment
```

The runner supports the compatibility-default `allocating` mode and an explicit
`preallocated` mode with reusable PyTorch and Triton outputs. Results keep CUDA
event, host submission, and synchronized host-wall timings separate; modes are
not combined in rankings. It also accepts explicitly opted-in, trusted local
Triton source candidates in preallocated mode. Their bytes are hashed, preserved
with the run, and loaded only in the worker process.

See the [experiment runner guide](docs/experiment_runner.md) for configuration,
result fields, failure behavior, and the callable Python interface.

Run a reproducibly shuffled, sequential batch of predefined FC1 candidates with
fresh matched baselines. From a source checkout, the checked-in example is:

```sh
matmul-inspector-python experiment-batch \
  --config examples/fc1_experiment_batch.json \
  --output-root artifacts/experiment-batches
```

Persistent search sessions add immutable experiment settings, structured
hypotheses and history, canonical duplicate detection, locking, crash recovery,
and an enforced experiment budget without adding a model API:

```sh
matmul-inspector-python session create \
  --config examples/fc1_experiment_session.json \
  --output-root artifacts/experiment-sessions
```

See the experiment runner guide for proposal, status, context, and recovery
commands.

## Technical Case Studies

- [Tensor Game search optimization](docs/CaseStudy/TensorGame.md) — follows a
  failed offline policy update through candidate-coverage diagnosis, bounded
  residual ranking, streaming-heap memory reduction, and matched CPU benchmarks.
  The case study includes evidence-backed figures and their reproducible
  generation requirements.
- [Tensor Game hardware-cost study](docs/tensor_game_hardware_cost_study.md) —
  adds optional static algebraic descriptors and deterministic Pareto analysis
  without changing the default bounded search ranking.
- [Hardware-aware rank-8 tensor study](docs/hardware_aware_tensor_search.md) —
  generates exact CPU-only basis-equivalent candidates and a static Pareto
  frontier. It does not benchmark or predict GPU latency.
- [Direct exact rank-8/9 structural search](docs/hardware_aware_rank89_constraint_search.md)
  — uses an optional bounded CP-SAT solver to study exact finite action pools
  and operand reuse without GPU execution.
- [Hardware-aware execution schedule model](docs/hardware_aware_schedule_model.md)
  — builds exact block DAGs and compares abstract CPU-only memory, fusion, and
  recomputation schedules without predicting GPU latency.
- [Constraint-based execution schedule search](docs/hardware_aware_schedule_search.md)
  — searches fixed-order launch groups, materialization, and bounded
  recomputation under an abstract capacity using optional CPU-only CP-SAT.
- [Selected GPU schedule study](docs/hardware_aware_schedule_gpu_study.md) —
  tests a small static-frontier subset with RTX 4060 Ti Triton lowerings,
  correctness gates, CUDA-event timing, and compiler evidence.
- [Recompute-vs-store mechanism study](docs/hardware_aware_schedule_mechanism_study.md)
  — isolates the selected rank-8 schedule's preparation, consumer, launch,
  graph-replay, and memory-hierarchy effects on the RTX 4060 Ti.
- [Tensor Game Triton lowering study](docs/tensor_game_triton_lowering_study.md)
  — validates conventional and Strassen decompositions through one experimental
  FP32 block-lowering interface with skip-safe GPU tests and benchmark capture.
- [AI-assisted FC1 kernel search](docs/ai_assisted_kernel_search_case_study.md) —
  separates candidate correctness, CUDA-event timing, and compiler evidence in a
  controlled Triton optimization workflow.
- [CUDA PTX/SASS FMA analysis](docs/cuda_ptx_sass_fma_case_study.md) — connects
  source changes to PTX, final SASS, numerical behavior, and measured latency.
- [Trainium1 NKI GEMM](docs/trainium1_nki_gemm_case_study.md) — compares source,
  compiler artifacts, numerical results, and hardware measurements without
  conflating those evidence layers.

## How artifact comparison works

1. Read existing artifacts without modifying them.
2. Parse identities, instructions, declarations, metadata, and available resource
   records without guessing across ambiguous boundaries.
3. Preserve raw hashes and apply only documented conservative normalization.
4. Compare full-file text separately from selected-kernel structure.
5. Write deterministic structured JSON and a concise Markdown evidence report.
6. Keep compiler observations, numerical correctness, benchmark completion, and
   performance measurements independently labeled.

Output directories are never overwritten. Unknown or unavailable information is
reported rather than silently treated as equal, empty, or successful.

## Native C++/CUDA laboratory

The native executable is not bundled in the Python wheel. Build it from a source
checkout with CMake 3.17+ and a C++17 compiler:

```sh
cmake -S . -B build-cpu \
  -DCMAKE_BUILD_TYPE=Release \
  -DMATMUL_INSPECTOR_ENABLE_CUDA=OFF
cmake --build build-cpu --parallel
ctest --test-dir build-cpu -L cpu --output-on-failure
./build-cpu/matmul-inspector compare \
  --reference cpu --candidate cpu --sizes 4,17 --seed 42 \
  --output results/cpu-check
```

The CUDA source build also provides an FP32 comparison between a one-thread-per-
system batched Thomas baseline and a one-thread-per-equation PCR baseline. It
retains raw CUDA-event samples and correctness evidence. See the
[CUDA Thomas versus PCR case study](docs/cuda_thomas_pcr_case_study.md).

Enable CUDA with `-DMATMUL_INSPECTOR_ENABLE_CUDA=ON` when a supported toolkit is
available. The laboratory includes naive and tiled matmul, controlled FMA and
accumulation-order variants, correctness comparisons, benchmark distributions,
CPU/GPU crossover analysis, reductions, and reproducible artifact capture. It is
an experimental laboratory, not a replacement for a tuned BLAS.

## Documentation and reproducible experiments

- [Experimental findings](docs/findings.md) — concise results and qualifications
- [Benchmark methodology](docs/benchmarking.md) — timing boundaries, trials,
  bootstrap intervals, and reliability flags
- [Floating-point experiments](docs/floating_point_experiments.md) — contraction,
  reordering, correctness, and replay
- [PTX compiler analysis](docs/ptx_compiler_analysis.md)
- [CUDA PTX/SASS FMA case study](docs/cuda_ptx_sass_fma_case_study.md)
- [NKI compiler-artifact comparison](docs/nki_compiler_artifact_comparison.md)
- [Compiler-to-performance reporting](docs/compiler_performance_reporting.md)
- [Trainium1 NKI GEMM case study](docs/trainium1_nki_gemm_case_study.md)
- [NKI GEMM benchmarking](docs/nki_gemm_benchmarking.md)
- [Roofline methodology](docs/roofline.md)
- [Cross-hardware comparison](docs/cross_hardware.md) and
  [reproducibility](docs/reproducibility.md)
- [Summation methods](docs/summation.md), [operations](docs/operations.md), and
  [tridiagonal benchmarking](docs/tridiagonal_benchmarking.md)
- [Transformer optimization study](docs/transformer_optimization.md)
- [CUDA/Triton experiment runner](docs/experiment_runner.md)
- [AI-assisted FC1 kernel search case study](docs/ai_assisted_kernel_search_case_study.md)
- [Python toolkit release guide](docs/python_toolkit_release.md),
  [0.3.0 release notes](docs/release_notes_0_3_0.md), and
  [0.2.0 release notes](docs/release_notes_0_2_0.md)
- [Checked-in RTX 4060 Ti floating-point captures](results/examples/rtx4060ti-controlled-fp/README.md)

## Requirements and limitations

- Saved-artifact analysis supports Linux and Python 3.10–3.13 and uses only the
  Python standard library.
- GPU experiments are optional. Follow PyTorch's platform-specific installation
  instructions for a CUDA-enabled build, then install
  `matmul-inspector[gpu]==0.3.0`. The extra requests PyTorch and Triton but does
  not configure NVIDIA drivers, select a system CUDA driver, or guarantee GPU
  compatibility. Verify `torch.cuda.is_available()` before running experiments.
- Creating PTX/CUBIN/SASS requires NVIDIA compiler tools; running CUDA benchmarks
  additionally requires compatible hardware and drivers.
- Creating NKI/NEFF/profiler captures requires an AWS Neuron environment and, for
  hardware measurements, Trainium. Comparing saved captures does not.
- PTX is an intermediate representation; SASS is hardware-specific machine code.
- Static counts do not establish dynamic execution, scheduling, stalls, traffic,
  occupancy, or performance causality.
- Conservative normalized equality is not proof of binary or semantic equality.
- Benchmark conclusions apply only to their recorded hardware, software, inputs,
  timing boundaries, and numerical acceptance criteria.

## Contributing

Bug reports and focused contributions are welcome through
[GitHub Issues](https://github.com/Lonny154/Matmul-Inspector/issues). Preserve
reproducible inputs and outputs, report correctness independently from timing,
and avoid optimization claims without controlled measurements.

## License

Matmul Inspector is licensed under the [Apache License 2.0](LICENSE). The project
copyright notice is recorded in [NOTICE](NOTICE).
