# MI-Research-3 — Split-K graph performance boundary

## Plan recorded before GPU measurement

The prior [MI-Research-2C](mi_research_2c.md) study found a reproducible
fixed-address CUDA Graph advantage for Split-2 at FP16 M=32, K=4096,
N=2048 on the RTX 4060 Ti. [MI-Research-2](mi_research_2.md) and
[2B](mi_research_2b.md) found unstable eager Split-2 timing. This study asks
whether increasing conventional output-grid parallelism erodes the graph
advantage. It does not assume that output CTA count equals achieved
occupancy or that a crossover exists.

The control is the exact registered conventional `64×128×32`, four-warp,
three-stage GEMM. The candidate is the exact committed Split-2 source with
the same GEMM tile. Its complete operation is the partial GEMM plus a
deterministic reduction. The graph retains fixed input and output addresses
and one reusable FP32 scratch tensor. One replay contains 100 complete
operations for each implementation. Compilation, graph construction, and
profiling are outside the measured replay interval. Graph setup time is
saved separately. A matched fixed-address `torch.mm(..., out=...)` graph is
an external cuBLAS anchor. Its comparisons are separate from A versus B.

All shapes keep K=4096 and N=2048. The planned M values and **configured**
grid sizes are:

| M | Conventional output CTAs | Split-2 partial CTAs | Reduction CTAs | FP32 scratch |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 16 | 32 | 64 | 0.5 MiB |
| 128 | 32 | 64 | 256 | 2 MiB |
| 256 | 64 | 128 | 512 | 4 MiB |
| 512 | 128 | 256 | 1024 | 8 MiB |

The device reports 34 SMs. These grids cross that count while keeping
the kernel geometry and K work fixed. Scratch is bounded to 8 MiB.
Each shape is valid for the 64×128 tile and K is divisible by two partitions
and the K tile.

Use two fresh-process sessions per shape, in order M=32,128,256,512 and
then M=512,256,128,32. Each session uses seeds 0,1,2 and the unchanged
`atol=rtol=0.01` correctness gate against PyTorch/cuBLAS. Poison output
and scratch before each correctness replay. Require the profiler trace to
show one graph launch and 100 complete A, B, and cuBLAS operations. Save
PTX and compiler metadata for conventional, partial, and reduction kernels.

Each session has 30 replay warmups, 100 operations per timed block, eight
balanced paired rounds, and eight preceding identical-kernel A/A rounds
for each of A/B, A/cuBLAS, and Split-2/cuBLAS. The phase order is A/B,
A/cuBLAS, Split-2/cuBLAS in session 1 and the reverse in session 2.
This adds anchor measurements but may also change warm state. Save every
CUDA-event, synchronized `perf_counter`, and MONOTONIC_RAW block and the
host-clock preflight. Use the existing clock-aware paired decision rule
within each phase. Do not pool unlike shapes or execution modes.

The setup measurements are order-sensitive and include graph construction
only, not an application-wide setup model. Any break-even estimate must
use the observed setup difference and replay latency difference for the
same shape/session and state those assumptions. A lower median alone will
not establish a winner. A supported shape-specific result requires the
existing paired decision in both independent sessions, with correctness
and graph completeness verified. Disagreement or instability stays
inconclusive. The study stops after these eight planned sessions.

## Observed correctness and captured operations

M=32 and M=128 passed seeds 0, 1, and 2 for A, B, and C in both independent sessions. M=256 and M=512 stopped **before timing** at seed 0. Both Triton outputs failed the unchanged FP16 cuBLAS reference check at each larger shape; C matched the reference. The M=256 check was repeated once solely to save the per-kernel failure record after the first attempt had saved only the stop reason. The maximum absolute error for both A and B was 0.25 at each larger shape. Their mean absolute errors were about 0.01155 at M=256 and 0.00635 at M=512. Similar aggregate errors suggest a common numerical difference from the selected FP16 cuBLAS result, but this study did not isolate its cause or establish elementwise A/B agreement. No tolerance was changed. Neither larger shape has a timing rank.

Every successful session's separate profiler replay showed one `cudaGraphLaunch`, 100 conventional `fc1_kernel` events, 100 ordered `split_k_partial` and `reduce_partials` pairs, and 100 cuBLAS kernel events. The cuBLAS trace is checked for a repeated per-operation kernel sequence. These checks verify captured execution, not achieved occupancy. Each correctness seed replaced the fixed-address inputs, poisoned output and scratch, and replayed each graph before comparison. M=32 and M=128 Split-2 maximum absolute error was 0.125 against the reference. Mean Split-2 error was about 0.0002. Conventional and cuBLAS outputs matched the reference exactly in the saved checks for these two shapes.

## Graph-mode paired results

Values are untrimmed CUDA-event microseconds per **complete GEMM**. The paired difference is A−B, so positive favors Split-2. A/A is the largest absolute identical-kernel paired gap in that phase. Each row has eight balanced pairs of 100-operation graph replay blocks plus eight preceding A/A pairs. The existing schema-v3 decision also checks host-clock paired directions, event-range separation, and clock boundaries.

| Shape / session | A median | B median | A−B median; range | B wins | A/A max | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| M=32 / 1 | 61.76 | 30.27 | +30.95; +24.31 to +35.07 | 8/8 | 3.87 | `B_FASTER` |
| M=32 / 2 | 60.13 | 30.57 | +27.23; +22.84 to +31.81 | 8/8 | 4.01 | `B_FASTER` |
| M=128 / 1 | 59.48 | 60.06 | −0.95; −3.79 to +5.03 | 3/8 | 6.38 | `INCONCLUSIVE` |
| M=128 / 2 | 57.59 | 57.89 | −0.36; −5.05 to +4.02 | 3/8 | 4.86 | `INCONCLUSIVE` |

The M=32 graph-mode Split-2 advantage reproduced in two fresh processes. The M=128 differences crossed zero and stayed within observed identical-kernel variation. This is a **shape-dependent pattern** under the tested fixed-address graph setup: an advantage at 16 conventional output CTAs, then no established rank at 32 CTAs. It is not a demonstrated crossover to conventional GEMM. The planned 64- and 128-CTA shapes failed correctness, so the study cannot locate where conventional becomes faster. Configured CTA counts give a plausible parallelism hypothesis but do not measure occupancy. No phase-only graph measurement isolates the cost of growing FP32 scratch or reduction. Complete B timing includes both kernels on every operation.

### Matched cuBLAS anchor

The cuBLAS C graph had the same fixed-address, 100-operation capture and replay semantics. It was measured in separate balanced A/C and B/C phases with their own identical-kernel controls. No eager result was compared to these graph latencies.

| Shape / session | A/C medians; decision | B/C medians; decision |
| --- | --- | --- |
| M=32 / 1 | 61.79 / 20.89; C faster | 31.49 / 21.19; inconclusive |
| M=32 / 2 | 61.57 / 23.32; C faster | 32.26 / 20.73; inconclusive |
| M=128 / 1 | 61.65 / 54.70; inconclusive | 59.13 / 57.68; inconclusive |
| M=128 / 2 | 57.20 / 50.97; inconclusive | 56.14 / 56.72; inconclusive |

At M=32, C beat conventional A under the retained paired rule in both sessions. B/C medians suggest C may be faster, but at least one paired gap in each session fell within the B/B control's observed variation, so the strict decision stayed inconclusive. At M=128, neither comparison supports a rank. These anchor results do not change the A/B finding.

### Setup, clocks, and compiler evidence

Graph construction times for A/B/C were 30.90/5.05/1.85 ms at M=32 session 1 and 1.03/2.43/13.53 ms in session 2. M=128 construction took 17.18/2.57/1.20 ms and 1.63/2.70/15.42 ms. The first captured graph in each fresh process was much more expensive, so setup differences are confounded by order. In M=32 session 2, B cost 1.398 ms more to construct than A while their median replay difference was 29.56 µs per complete operation. Arithmetic break-even is about 48 operations, less than one 100-operation replay. This estimate applies only to the measured capture sequence and fixed-address setup; it is not an application-level promise. M=128 has no supported replay advantage to amortize.

All four successful host-clock preflights returned `NO_RESOLVED_DIFFERENCE`. Saved A/B decisions have no clock-boundary warnings. Every block retains CUDA-event, synchronized `perf_counter`, MONOTONIC_RAW, host-submission, and paired-order data. The three phases within each session can influence warm state; their order was reversed between sessions. One M=128 B/C round reached 84.70 µs for B. It remains in the raw distribution. Pre/post `nvidia-smi` snapshots are retained where collected, but there is no in-round clock or temperature attribution.

Retained PTX and compiler metadata cover conventional, partial, and reduction kernels. At M=32 the compiler reported 119/124/22 registers per thread and 24,576/24,576/0 bytes shared memory respectively, with no reported spills. At M=128 the conventional count was 121; partial and reduction counts were unchanged. PTX hashes differ across shapes for the GEMM kernels. Decoded SASS is unavailable. Static resources or PTX cannot establish final machine-code behavior or a utilization mechanism.

## Decision and reproduction

**Decision: limited shape-dependent evidence, no supported crossover.** The M=32 complete-operation graph advantage is reproducible. At M=128 the ranking is inconclusive, and at M=256/512 the numerical gate blocks a performance claim. An evidence-based automatic strategy selector is **not justified** by this small, partly invalid sample. A future study would first need to understand the larger-shape numerical mismatch while preserving correctness, then collect valid larger-grid comparisons. No eager-mode or changing-address selection rule follows from this work.

The narrow study script reuses the existing kernels, graph normalization, clock measurements, paired ordering, and schema-v3 decision code. It does not modify production benchmarking. From the repository root:

```bash
.venv/bin/python scripts/run_mi_research_3_graph.py --m 32 --session 1 --output /tmp/mi-research-3-m32-s1
.venv/bin/python scripts/run_mi_research_3_graph.py --m 32 --session 2 --output /tmp/mi-research-3-m32-s2
for name in m32-s1 m32-s2 m128-s1 m128-s2; do .venv/bin/python scripts/run_mi_research_3_graph.py --verify --output "artifacts/mi-research-3/evidence-v1/$name"; done
sha256sum -c artifacts/mi-research-3/evidence-v1/SHA256SUMS
```

The full four-shape plan and exact session/failure files are under `artifacts/mi-research-3/evidence-v1/`. `analysis.json` is a compact derivation. Each successful session's `raw.json`, `summary.json`, traces, PTX, source snapshot, diagnostic script snapshot, and hash manifest remain the evidence of record. The M=256 first stop and correctness-only retry, and the M=512 stop, are retained as negative evidence. The top-level SHA-256 manifest covers the full bundle, including failures and telemetry.
