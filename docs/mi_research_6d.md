# MI-Research-6D — fused Perminov versus staged Laderman

## Question and frozen comparison

**Hypothesis.** The single-launch Perminov-55 implementation may have a
repeatable complete-operation CUDA Graph latency advantage over the staged
Laderman implementation at N=96. The control is staged Laderman (A). The
candidate is fused Perminov (B). The comparison uses the committed
MI-Research-6C checkpoint `1b3ae45f5fb00f49f1901e3ea3e1fc8b17607c24`.

Both algorithms compute 96×96 GEMM from FP16 inputs to FP16 output. The
reference is `torch.matmul(left.float(), right.float()).half()` with TF32
and reduced FP16 reduction disabled while the reference and comparison run.
The prior `atol=rtol=0.01` gate is unchanged. A uses the known Laderman
rank-23 3×3 factorization, 23 product kernels, an output reconstruction
kernel, and 94,208 bytes of explicit FP32 product scratch. B uses the
known Perminov `cr58_cn122` rank-23 factorization and its published
55-addition circuit. Its four output-owning CTAs each execute 23 source-level
`tl.dot` expressions and store all nine 16×16 output tiles they own. It uses
one kernel and no explicit product buffer. The two candidates differ in
algebraic representation **and** GPU execution design.

The frozen source SHA-256 values are
`8b328950567868b2d119e021a0dd3bcc312d6e1c8138a1577b0e0491949fbca1`
for A and
`c9b2d61fc6a1b6a658f1198d1a3938fae154efce2029c463e439b7fabbfaf8db`
for B. The staged coefficient JSON SHA-256 is
`4abdcb3db4094b17dba24e8c23b17206fae2e82d769b61fa35952831480a68ef`.
The prior 6B and 6C evidence manifests are linked by hash from each new
raw record. Both retained implementations and the new diagnostic script
are snapshotted inside each session bundle. No kernel, coefficient, tile,
warp, stage, precision, or schedule change was made for this comparison.

## Capture and correctness gates

Two independent Python processes ran on an NVIDIA GeForce RTX 4060 Ti with
34 SMs, compute capability 8.9, driver 616.92, PyTorch 2.11.0+cu128, and
Triton 3.6.0. Session 1 captured A then B. Session 2 captured B then A.
Compilation and ten eager warmups per implementation preceded capture.
Each graph contains 20 complete operations. Input, coefficient, scratch,
and output addresses remained stable. Graph construction and first compile
calls were timed separately and excluded from replay measurements.

For seeds 0, 1, and 2 plus zero, identity, and cancellation inputs, the
script changed input contents at the same addresses. It poisoned A's entire
product scratch and each candidate's output with NaNs before **each** of
two replays per case. All 48 graph checks passed the common reference with
zero failing or nonfinite output elements. The largest absolute errors were
0.0078125 for A and 0.015625 for B in each session. The retained raw checks
include per-case maximum and mean errors, failing fractions, and representative
elements; they do not retain every elementwise error. All graph outputs matched a fresh
eager call of their own frozen implementation bitwise. These tests support
full recomputation for the tested inputs and replay sequence. They do not
prove numerical stability for all inputs.

Profiler traces record one `cudaGraphLaunch` and 480 A kernel events per
replay: 20 repetitions of 23 products followed by reconstruction. B has
20 `_fused` kernel events per replay, one per complete operation. Its
observed grid is 2×2×1 with 128 threads per CTA. The verifier checks the
ordered kernel sequence and the B grid in every operation. Four configured
CTAs are not achieved occupancy.

## Compiler and physical resource evidence

| Selected compiled kernel | Registers/thread | Shared memory | Global compiler scratch | PTX `mma.sync` occurrences | PTX local load/store |
| --- | ---: | ---: | ---: | ---: | ---: |
| A product | 56 | 8,192 B | 0 B | 24 | 0 / 0 |
| A reconstruction | 40 | 0 B | 0 B | 0 | 0 / 0 |
| B fused | 255 | 1,024 B | 0 B | 552 | 0 / 0 |

The compiled PTX hashes matched across both fresh processes. TTIR, TTGIR,
LLIR, PTX, and metadata are retained in each bundle. B's 552 static PTX
`mma.sync` occurrences are in one compiled kernel body. They are not a
dynamic instruction count or a SASS count. The source has 23 dots per CTA;
four output owners repeat that source-level product work. There is no
inter-CTA product sharing in this implementation. The longer 55-gate DAG
dependency path documented in 6C is a plausible cost, but this experiment
does not measure its isolated contribution. Triton's `n_spills` field was
unavailable. Zero reported global scratch and no PTX local-memory text do
not establish zero SASS spills. Build warnings were not captured as a
structured compiler artifact. Decoded SASS, achieved occupancy, DRAM
traffic, and per-SM utilization were not measured. The 255-register report
and four-CTA grid therefore cannot support a causal latency attribution.

## Bounded paired CUDA Graph timing

The existing Matmul Inspector schema-v3 paired decision rule was used
unchanged. One CUDA-event timing block submits exactly one graph replay of
20 complete operations. Ten graph warmups per candidate preceded timing.
Each session has eight balanced A/B rounds and eight A/A rounds immediately
before their A/B counterparts. `perf_counter` and
`CLOCK_MONOTONIC_RAW` host diagnostics are retained along with every raw
CUDA-event sample. No outlier was discarded. Both host-clock preflights
observed a resolved clock-rate difference, which the decision rule records.

| Session | A event median (range), µs/op | B event median (range), µs/op | Paired A−B median, µs/op | Largest absolute A/A gap, µs/op | Existing decision |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | 68.255 (61.110–113.309) | 15.775 (15.194–17.515) | +52.463 | 56.400 | Inconclusive |
| 2 | 87.529 (53.133–121.707) | 15.191 (15.050–16.019) | +72.471 | 66.546 | Inconclusive |

The paired differences, in round order, are in microseconds per complete
operation. Positive values favor B. All 16 rounds favored B, but the
existing rule requires **every** paired absolute gap to exceed the largest
observed A/A gap in its session. That requirement failed in both sessions.

| Session | Eight paired A−B differences, µs/op | Eight A/A X−Y differences, µs/op |
| --- | --- | --- |
| 1 | +46.701, +79.853, +54.043, +97.400, +48.784, +50.883, +45.917, +77.006 | −52.264, −56.400, −31.774, −0.762, +17.944, −1.966, −31.896, +1.512 |
| 2 | +45.368, +100.570, +38.709, +100.381, +77.302, +67.640, +106.414, +38.043 | −50.344, −52.488, −60.058, −57.461, +5.371, −18.976, +21.926, +66.546 |

The A/A spread came from the staged control. B's event times were much
tighter. This is an observation, not proof that launch count, scratch,
register use, CTA count, or circuit depth caused the gap or its variation.
Historical conventional, Strassen, and cuBLAS timings are unmatched
context. They are not direct controls here. A B win in this comparison
would not establish superiority over optimized conventional GEMM or
cuBLAS. This experiment cannot compare Perminov's 55- and 58-addition
circuits in isolation; that requires a matched same-decomposition design.

## Setup and amortization

| Session | A first compile call / graph setup | B first compile call / graph setup |
| --- | ---: | ---: |
| 1 | 491.103 / 48.333 ms | 18.205 / 0.622 ms |
| 2 | 306.681 / 6.117 ms | 12.593 / 16.264 ms |

The first compile call includes a synchronized execution and possible
runtime setup, so it is not pure compiler time. For 100 graph replays, the
observed graph setup alone would average 24.17/3.06 µs per A operation
and 0.31/8.13 µs per B operation in sessions 1/2. This is arithmetic
over the observed setup values, not a measured application break-even
point. Capture order and runtime effects are confounded in two sessions.
Graph lifetime, buffers, and setup frequency matter for an application.

## Decision and next step

**C — timing variability prevents interpretation under the existing
rule.** Both frozen implementations pass numerical, capture, and physical
execution gates. The observed B timings show a consistent directional
opportunity, but the A/A control prevents a supported performance ranking
in both sessions. There is no claimed optimization improvement or isolated
causal mechanism. This exhausts the two-session, eight-round budget.

The evidence-backed next step is to investigate the staged A/A variability
with a separately planned measurement-control study before interpreting
this gap as a stable advantage. The current study does not run more
sessions or tune either kernel.

## Offline replay and reproduction

All retained checks replay without a GPU:

```sh
.venv/bin/python scripts/run_mi_research_6d_graph.py --verify --output artifacts/mi-research-6d/graph-s1
.venv/bin/python scripts/run_mi_research_6d_graph.py --verify --output artifacts/mi-research-6d/graph-s2
.venv/bin/python scripts/analyze_mi_research_6d.py --verify
.venv/bin/python -m pytest -q tests/mi_research_6d_test.py tests/mi_research_6c_test.py tests/mi_research_6b_test.py tests/mi_research_6_test.py tests/mi_research_2c_test.py
```

The original GPU collection commands used separate fresh processes and
nonexistent output directories:

```sh
.venv/bin/python scripts/run_mi_research_6d_graph.py --session 1 --output artifacts/mi-research-6d/graph-s1
.venv/bin/python scripts/run_mi_research_6d_graph.py --session 2 --output artifacts/mi-research-6d/graph-s2
.venv/bin/python scripts/analyze_mi_research_6d.py
```

Use fresh paths for a new experiment. These paths contain the retained
original and cannot be overwritten by the runner. Both bundles have
SHA-256 manifests; `analysis_manifest.json` binds the cross-session
analysis to their manifests. No CUBIN or NEFF binary is retained.
