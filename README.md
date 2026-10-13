# Matmul Inspector

**See what your GPU or AI-accelerator compiler actually produced.**

Matmul Inspector turns saved NVIDIA PTX, NVIDIA SASS, and AWS Neuron NKI compiler
artifacts into readable JSON and Markdown reports. This matters because a small
source change can disappear in an intermediate representation, produce different
machine instructions, change numerical results, or affect performance—and none
of those outcomes can safely be inferred from another.

Install the Python toolkit from [PyPI](https://pypi.org/project/matmul-inspector/):

```sh
python -m pip install matmul-inspector==0.2.0
matmul-inspector-python --version
```

Version 0.2.0 supports Python 3.10–3.13. Saved-artifact analysis has no mandatory
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
- [Python toolkit release guide](docs/python_toolkit_release.md) and
  [0.2.0 release notes](docs/release_notes_0_2_0.md)
- [Checked-in RTX 4060 Ti floating-point captures](results/examples/rtx4060ti-controlled-fp/README.md)

## Requirements and limitations

- Saved-artifact analysis supports Linux and Python 3.10–3.13 and uses only the
  Python standard library.
- GPU experiments are optional. Follow PyTorch's platform-specific installation
  instructions for a CUDA-enabled build, then install
  `matmul-inspector[gpu]==0.2.0`. The extra requests PyTorch and Triton but does
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
