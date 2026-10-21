# MI-Research-7 — fused Perminov versus conventional GEMM

## Question and frozen comparison

**Hypothesis.** The existing single-kernel Perminov-55 implementation might
offer a practical N=96 latency advantage over conventional GPU GEMM. The
controls are one unchanged packaged Triton GEMM and PyTorch/cuBLAS. This
study asks that question separately for eager calls and fixed-address CUDA
Graph replay. It does not change the Perminov circuit, kernel, compiler
settings, or Matmul Inspector's schema-v3 paired decision rule.

The [protocol](../artifacts/mi-research-7/protocol.json) was saved before
GPU execution. It fixes M=N=K=96, contiguous row-major FP16 inputs and
outputs, the RTX 4060 Ti, six established input cases, and
`atol=rtol=0.01`. The independent reference is FP32 multiplication of the
exact FP16-valued inputs, rounded once to FP16. During reference creation,
correctness checks, and timing, `float32_matmul_precision=highest`,
`allow_tf32=false`, and
`allow_fp16_reduced_precision_reduction=false`. The original backend flags
were restored afterward. The cuBLAS path is `torch.mm(left, right,
out=preallocated_output)`, so it receives the original FP16 inputs and has
the same FP16 output contract. The reference is a separate FP32 computation;
it is never used as cuBLAS's measured operation.

| Label | Complete operation | Fixed implementation/configuration |
| --- | --- | --- |
| P | Fused Perminov-55 | Original MI-Research-6C `_fused` kernel, 2×2 CTAs, 16×16×32 `tf32x3` dots, four warps, two stages; 23 source-level dots per CTA and no explicit product buffer. |
| T | Conventional Triton | Unchanged packaged `triton_fc1.fc1_kernel`, 32×32×32 tiles, four warps, three stages, `group_size_m=1`, FP16 `tl.dot` with FP32 accumulator and FP16 store. This one predeclared shape-compatible configuration gives nine CTAs. No autotuning sweep was run. |
| C | PyTorch/cuBLAS | `torch.mm(..., out=...)` into a fixed FP16 output, with reduced FP16 reduction disabled. The observed library kernel is a CUTLASS tensor-core GEMM with 12 CTAs. |

The P source SHA-256 is
`c9b2d61fc6a1b6a658f1198d1a3938fae154efce2029c463e439b7fabbfaf8db`.
The packaged T source SHA-256 is
`a88226c46f335a9768f15b52eb275abf6066bfb57b1f7e012faa937a51768552`.
The fixed 32×32 T geometry covers N=96 without a large masked N tile and
exposes nine CTAs rather than the packaged FC1 default's two at this
shape. It is a realistic single configuration, not a demonstrated N=96
optimum. The cuBLAS anchor provides a separately optimized library path.
The MI-Research-6F commit
`d095aa9939316414c8122ec5fa09a40839e10c60`, its evidence, and the
earlier fused-kernel probe were verified before execution. No historical
MI-Research-6 files were edited.

## Numerical and execution gates

Four fresh Python processes ran: two eager and two CUDA Graph. Session 1
initialized or captured P, T, C and measured P/T before P/C. Session 2
reversed both orders. Each implementation had ten eager warmups. Graph mode
also had ten graph warmups after capture. Each timed eager block calls its
preallocated operation 20 times. Each timed graph block submits one graph
containing 20 complete operations. Compilation, graph construction,
correctness, and profiler traces were outside primary timed blocks.

For seed 0, seed 1, seed 2, zero, identity, and cancellation inputs, all
three paths ran twice after poisoning the full output with NaNs. Graph
mode also replayed each captured graph twice on the same fixed addresses
after updating input contents and poisoning output. Every result passed
the unchanged reference gate with zero failing or nonfinite output elements.
The maximum absolute error across all implementations and cases was
0.015625. All graph results matched a fresh eager call of their own
implementation bitwise. Input and output addresses remained fixed. These
checks cover the tested inputs and sequence, not all possible inputs.

Separate one-operation eager traces show one physical kernel launch each.
Separate graph traces show one `cudaGraphLaunch` and 20 kernel events each.
The recorded grids are P 2×2×1, T 9×1×1, and C 12×1×1, with 128 threads
per CTA in these traces. There was no extra output-finalization kernel.
The profiler runs are topology evidence and are not used as timing samples.

## Eager results

One eager block measures 20 preallocated calls with one CUDA-event bracket.
The host submission interval includes Python and library dispatch for those
20 calls. `perf_counter` and MONOTONIC_RAW windows also include end-event
synchronization. No allocation is requested by the three wrappers inside
the measured operation; the library may still perform internal runtime
work. The two P comparisons each use eight balanced AB/BA rounds. Before
each A/B pair, an identical P/P control runs in balanced XY/YX order.
The table gives untrimmed CUDA-event microseconds per complete operation.
P−baseline below zero favors P. A/A is the largest absolute P/P gap.

| Session / pair | P median | Baseline median | P−baseline median | P-faster rounds | A/A max | Existing rule |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 / T | 27.634 | 37.815 | −10.298 | 8/8 | 7.318 | `INCONCLUSIVE` |
| 2 / T | 21.354 | 13.014 | +7.956 | 1/8 | 8.710 | `INCONCLUSIVE` |
| 1 / C | 26.981 | 32.528 | −5.312 | 7/8 | 3.768 | `INCONCLUSIVE` |
| 2 / C | 27.634 | 32.701 | −4.697 | 8/8 | 6.349 | `INCONCLUSIVE` |

The paired P−baseline values, in round order, are:

| Session / pair | Eight raw differences, µs/op |
| --- | --- |
| 1 / T | −12.427, −2.750, −5.602, −11.437, −11.453, −10.259, −10.338, −9.157 |
| 2 / T | −10.346, +2.077, +9.555, +8.138, +7.774, +8.717, +8.203, +6.989 |
| 1 / C | −6.856, −6.334, −4.923, −5.701, −2.534, −3.744, +4.611, −9.776 |
| 2 / C | −5.282, −4.112, −6.547, −3.570, −5.821, −5.894, −3.965, −2.995 |

The T pair reverses its median direction between sessions. Session 2's
first T round favors P; the next seven favor T. The C pair's medians favor
P in both sessions, but an A/B gap fails to clear the session's P/P A/A
maximum in each. All four eager decisions are therefore `INCONCLUSIVE`
under the unchanged rule. The host `perf_counter` and RAW paired medians
track the broad event directions; their full samples and A/A sequences
are retained. Both eager host-clock preflights detected a RAW versus
`perf_counter` rate difference. The rule records that discrepancy and
does not rescale event times. The eager medians do not establish a P win
over either external baseline.

## CUDA Graph results

The graph timing window is one fixed-address replay of 20 complete GEMMs,
normalized by 20. It has one host graph submission per block. Capture
setup was recorded separately. The same P/P control and eight balanced
rounds are used for each pair. The table again gives untrimmed CUDA-event
microseconds per complete operation. Positive P−baseline favors the
conventional baseline.

| Session / pair | P median | Baseline median | P−baseline median | Baseline-faster rounds | A/A max | Existing rule |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 / T | 15.781 | 1.407 | +14.336 | 8/8 | 1.120 | `B_FASTER` |
| 2 / T | 15.663 | 1.394 | +14.245 | 8/8 | 1.424 | `B_FASTER` |
| 1 / C | 15.767 | 1.851 | +13.914 | 8/8 | 0.205 | `B_FASTER` |
| 2 / C | 15.667 | 1.845 | +13.806 | 8/8 | 0.226 | `B_FASTER` |

The raw P−baseline differences are:

| Session / pair | Eight raw differences, µs/op |
| --- | --- |
| 1 / T | +13.354, +14.344, +14.323, +14.382, +14.328, +14.160, +14.365, +14.408 |
| 2 / T | +13.430, +14.307, +14.237, +14.283, +14.253, +15.379, +12.595, +13.856 |
| 1 / C | +13.901, +13.958, +13.890, +13.917, +13.910, +13.936, +13.899, +13.922 |
| 2 / C | +13.822, +13.330, +13.786, +13.832, +13.787, +13.848, +13.790, +13.824 |

Every graph A/B gap exceeds the local A/A maximum. The CUDA-event ranges
separate, and all paired `perf_counter` and RAW directions agree. The
schema-v3 rule therefore supports T faster than P and C faster than P in
**both** fresh sessions. This is a specific N=96 fixed-address graph
result. It does not establish the same ordering in eager mode or at other
shapes. By the per-session medians, P took about 11.2× T's graph time
and about 8.5× C's graph time. Those ratios describe the measured
implementation at this shape; they are not isolated algorithmic costs.
Graph capture setup varied by session: P 16.257/0.303 ms,
T 0.443/0.432 ms, and C 0.689/13.199 ms. These are setup observations
with reversed capture order. They are excluded from steady-state replay;
the study does not infer an application break-even point from them.

## Compiler and hardware evidence

The saved TTIR, TTGIR, LLIR, and PTX identify the P and T builds. The
compiler metadata was stable across all four processes:

| Static or configured property | P | T |
| --- | ---: | ---: |
| Configured CTAs | 4 | 9 |
| Registers/thread reported | 255 | 44 |
| Shared memory/program reported | 1,024 B | 8,192 B |
| `n_spills` reported | 422 | 0 |
| Global scratch size reported | 0 B | 0 B |
| Static PTX `mma.sync` occurrences | 552 | 12 |
| PTX `ld.local` / `st.local` text | 0 / 0 | 0 / 0 |

The P `n_spills` value is a compiler field. It coexists here with zero
reported global scratch and no PTX local-load/store text. We cannot infer
actual local-memory traffic or a machine-code spill count from these
fields. PTX is not decoded SASS, and static `mma.sync` counts are not
dynamic instruction counts. Four configured P CTAs are not achieved
occupancy. Each P CTA owns output tiles and repeats 23 source-level dots;
the four-CTA grid, 255 registers, reported spills, and 55-addition
dependency network are plausible performance constraints. This comparison
does not isolate any of them as the cause of the graph gap. Achieved
occupancy, per-SM utilization, measured DRAM traffic, and decoded SASS
were not collected. The cuBLAS library kernel's internal compiler
resources were not captured by Triton.

## Decision and one next action

**B — valid comparison without a supported Perminov advantage.** The
numerical, graph, topology, and evidence gates passed. Both
conventional paths are faster than P in both CUDA Graph sessions under
the established rule. Eager performance cannot be ranked with this
study's controls. The previously supported Perminov advantage over
staged Laderman therefore does not establish a practical advantage over
conventional GEMM or cuBLAS at N=96. This is an assessment of the frozen
GPU implementation, not a claim about the best possible rank-23 circuit
mapping or a novel mathematical algorithm.

**Next research action:** test one higher-parallelism Perminov
output-ownership mapping at N=96 under the same CUDA Graph correctness
and paired cuBLAS protocol. Treat reduced register pressure or higher
occupancy as hypotheses to measure, not assumed effects. This study does
not start that redesign.

## Retained evidence and reproduction

The [evidence root](../artifacts/mi-research-7/) contains the frozen
protocol, four raw session bundles, all correctness checks, exact paired
blocks and A/A controls, source snapshots, compiler IR and PTX, separate
profiler traces, capture costs, environment and precision settings,
per-session manifests, and [cross-session analysis](../artifacts/mi-research-7/analysis.json).
The protocol SHA-256 is
`c722f61b851faa31a1983f79057018b3e654bcc092fe35508da4dbc4b3874e98`.
No generated CUBIN or NEFF is retained. The profiler traces are small
study-specific topology evidence.

All retained checks replay without a GPU:

```sh
for mode in eager cuda_graph; do
  for session in 1 2; do
    .venv/bin/python scripts/run_mi_research_7.py --verify --output "artifacts/mi-research-7/${mode}-s${session}"
  done
done
.venv/bin/python scripts/analyze_mi_research_7.py --verify
.venv/bin/python -m pytest -q tests/mi_research_7_test.py tests/mi_research_6f_test.py tests/mi_research_6e_test.py
```

The four original GPU invocations used separate fresh processes and
nonexistent output directories:

```sh
.venv/bin/python scripts/run_mi_research_7.py --mode eager --session 1 --output artifacts/mi-research-7/eager-s1
.venv/bin/python scripts/run_mi_research_7.py --mode eager --session 2 --output artifacts/mi-research-7/eager-s2
.venv/bin/python scripts/run_mi_research_7.py --mode cuda_graph --session 1 --output artifacts/mi-research-7/cuda_graph-s1
.venv/bin/python scripts/run_mi_research_7.py --mode cuda_graph --session 2 --output artifacts/mi-research-7/cuda_graph-s2
.venv/bin/python scripts/analyze_mi_research_7.py
```

The runner refuses to overwrite a retained session. Use fresh paths for a
separately authorized experiment. The four session manifests and top-level
analysis manifest bind the reported decisions to raw measurements and
source identities.
