# MI-Research-6B — Laderman CUDA Graph evaluation

## Plan frozen before GPU execution

**Hypothesis.** Fixed-address CUDA Graph replay may reduce the large host
submission variation seen in the MI-Research-6 eager staged comparison enough
to reveal a repeatable complete-operation difference between conventional
3×3 multiplication and Laderman's 23-product decomposition. No winner is
assumed.

**Control and identity.** A is the committed 27-product conventional 3×3
algorithm and B is the committed 23-product Laderman algorithm at parameter
`l=1`. Both use the exact coefficient table, N=96, FP16 input/output,
FP32 preparation/product scratch/accumulation, `tf32x3`, 16×16×32 product
tiles, four warps, two stages, and deterministic reconstruction from
MI-Research-6 commit `c52b21e4f87cfdb3bb254b0964099ffa13148c5e`.
The kernel source SHA-256 is
`8b328950567868b2d119e021a0dd3bcc312d6e1c8138a1577b0e0491949fbca1`;
the coefficient JSON SHA-256 is
`4abdcb3db4094b17dba24e8c23b17206fae2e82d769b61fa35952831480a68ef`.
The source's mathematical operation and execution order will not change.

**Capture design.** Each graph contains 20 complete operations. One A
operation has 27 ordered product kernels followed by its reconstruction.
One B operation has 23 ordered product kernels followed by its
reconstruction. Thus one replay should expose 560 A or 480 B GPU kernel
nodes. A single replay is one host graph submission, not one GPU kernel.
Each algorithm owns fixed coefficient, FP32 product-scratch, and FP16 output
buffers; both share fixed-address A/B inputs. All kernels are compiled and
warmed before capture. Capture, compilation, and profiling are excluded from
replay timing and their setup costs are recorded separately. The product
kernels overwrite all 32×32 partial tiles on every operation, and
reconstruction overwrites all nine output tiles. Replaying after poisoning
scratch/output and after changing input values at the same addresses will
test those claims.

**Correctness gates.** Use seeds 0/1/2 and zero, identity, and cancellation
inputs. Compare each graph output independently to the MI-Research-3C
FP32-rounded FP16 reference with unchanged `atol=rtol=0.01`. Record maximum
and mean errors, failing and nonfinite counts, repeated replays, eager
agreement, and buffer-address stability. Stop before timing on any failure.
Profile one complete replay per algorithm and require exact product then
reconstruction ordering for all 20 operations plus one graph submission.

**Bounded timing.** Use two fresh processes, capture A then B in session 1
and B then A in session 2. After graph warmup, each session measures eight
balanced paired A/B rounds with 20 complete operations per block and one
graph replay per block. Use the existing A/A controls before each A/B round,
CUDA events, synchronized `perf_counter`, optional MONOTONIC_RAW, and host
clock preflight. Retain every raw block, including slow blocks. Apply the
existing Matmul Inspector schema-v3 paired decision rule independently to
each session. Compare historical eager measurements only as unmatched
context. A fresh eager run is not planned. No optimized GEMM/cuBLAS timing
anchor is planned within this two-session budget.

**Preflight.** The active branch started at the verified MI-Research-6
checkpoint. Its exact CPU evidence and N=96 eager GPU evidence replayed
successfully. The RTX 4060 Ti is available when the command has GPU device
access. The pre-existing `AGENTS.md` edit and unrelated untracked material
remain outside this study.

## Results

### Capture, correctness, and physical execution

Both complete operations captured without a kernel change, allocation, or
synchronization inside the capture region. Compilation and 10 eager warmups
per algorithm preceded capture. Inputs, coefficients, product scratch, and
outputs kept stable device addresses. Session 1 captured A then B; session
2 captured B then A. The graph setup time was measured around graph
construction, capture, and final synchronization. It was excluded from
timed replay. The first synchronized compilation/warmup call was recorded
separately.

| Fresh process | First compile call A / B | Graph capture setup A / B | Capture order |
| --- | ---: | ---: | --- |
| 1 | 384.732 / 2.860 ms | 31.446 / 6.144 ms | A, B |
| 2 | 257.221 / 1.825 ms | 8.865 / 28.316 ms | B, A |

The first graph captured in each process cost more. Capture order, runtime
initialization, and other setup effects are confounded; these two sessions
do not identify a mechanism. The compiler call costs are not graph replay
costs. Reusing each graph for 100 replays would spread the **observed**
6.144–31.446 ms setup cost over 2,000 complete operations, or 3.1–15.7 µs
per operation in simple accounting. That arithmetic is not a measured
eager-versus-graph break-even point. Capture costs and buffer lifetime
must be included for a real application.

Each profiler trace shows one `cudaGraphLaunch` submission and the full
ordered GPU sequence for **one replay**. A contains 20 repetitions of 27
product kernels followed by reconstruction: **560 physical kernel events**.
B contains 20 repetitions of 23 product kernels followed by reconstruction:
**480 events**. The verifier checks every product/reconstruction position,
not just event totals. The same ordered-sequence hashes reproduced in both
sessions. One graph submission is not one GPU kernel. There are still 28
or 24 physical kernels per complete operation.

Before **each** correctness replay, the script poisoned the entire FP32
product scratch and FP16 output buffers with NaNs. It then replayed the
20-operation graph twice with unchanged inputs. It changed the input
values at the same fixed addresses across seeds 0, 1, and 2, zero,
identity, and cancellation cases. Every A and B replay passed the
MI-Research-3C FP32-rounded FP16 reference at unchanged
`atol=rtol=0.01`. There were no failing or nonfinite output elements.
The largest maximum absolute errors across both sessions were 0.015625
for A and 0.0078125 for B. Graph outputs matched the same staged eager
kernel outputs **bitwise** for all six cases in each session. The source,
profiler topology, poisoning, and repeated checks jointly support full
scratch overwrite and output recomputation for these inputs. They do not
prove stability for every possible input.

The original MI-Research-6 source and coefficient hashes were unchanged.
The compiled product and reconstruction PTX hashes in both graph sessions
match the original eager bundle. The product kernel reports 56 registers
per thread and 8,192 bytes shared memory. Reconstruction reports 40
registers and zero shared memory. Both report zero global scratch;
`n_spills` is unavailable. Retained TTIR, TTGIR, LLIR, and PTX are compiler
evidence. Decoded SASS, measured DRAM traffic, and achieved occupancy are
unavailable. Compiler identity does not establish identical runtime state.

The stable buffers use 110,592 bytes of FP32 product scratch and 2,916 bytes
of coefficients for A, versus 94,208 and 2,484 bytes for B. Each output
buffer uses 18,432 bytes. The graph captures all product writes and the
complete reconstruction. It does not capture allocation or compilation.

### Paired graph replay timing

Both fresh processes used 10 graph warmups per algorithm, 20 complete
operations per replay, one replay per timing block, eight balanced A/B
rounds, and eight identical-A A/A rounds before A/B. The existing
schema-v3 rule was applied without modification. All values below are
untrimmed **CUDA-event microseconds per complete operation**. A−B positive
favors Laderman B.

The A/A block always preceded A/B, so its placement may affect warm state.
This study did not counterbalance A/A placement. The A/A gaps describe the
observed control variability under this specific sequence.

| Session | Median A / B | A range / B range | Paired A−B median (MAD); signs favoring B | A/A maximum gap | Existing decision |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | 59.922 / 59.053 | 58.250–60.843 / 58.458–60.587 | +0.770 (0.378); 6/8 | 7.203 | `INCONCLUSIVE` |
| 2 | 60.233 / 59.525 | 58.218–68.330 / 59.034–68.206 | +0.708 (1.398); 5/8 | 19.406 | `INCONCLUSIVE` |

Session 1 paired differences ranged from −0.208 to +1.258 µs. Session 2
ranged from −8.858 to +7.312 µs. The median observed B lead was less than
one microsecond in both sessions. A/A gaps were much larger than that lead,
and signs were mixed. Both decisions are `INCONCLUSIVE` under the existing
rule. No slow round was removed. Session 2 had A and B event blocks near
68 µs and an A/A gap of 19.406 µs. Eight rounds per process cannot define a
tail-latency distribution or prove that a slow mode has disappeared.

The median synchronized `perf_counter` times were 61.742/61.013 µs in
session 1 and 62.904/62.019 µs in session 2 (A/B). The corresponding RAW
medians were 61.729/60.999 and 62.586/61.714 µs. Both host clocks include
end-event synchronization, so their intervals differ from CUDA-event
device time. No event interval exceeded its enclosing host interval.
Session 1's clock preflight found no resolved RAW versus `perf_counter`
rate difference. Session 2 flagged a host-clock rate difference (median
RAW/`perf_counter` ratio 0.995245). Measurements were not rescaled. The
warning further limits interpretation of small host-clock differences.

The historical **eager** session remains `INCONCLUSIVE`: event medians
380.878/300.542 µs, with a 677.155 µs A/A gap. Its A and B event CVs were
about 54.7% and 54.4%. In the new graph sessions, A/B CVs were 1.3%/1.4%
and 5.1%/6.3%. This is a narrower observed distribution in separate
processes under different execution semantics. It is **not** a controlled
eager-to-graph speedup or proof that host orchestration caused the eager
variation. The graph comparison still has unresolved A/A and paired
variation at the scale of the algorithm difference.

No matched optimized Triton or cuBLAS graph timing anchor was collected.
These results cannot establish that either staged design is practical
against a production GEMM. Four fewer mathematical products, four fewer
physical launches per operation, and less explicit scratch coexist with
extra Laderman input/output additions. The measurements do not isolate the
cost of any one factor.

### Decision and one next step

**C — timing variability prevents a reliable graph-mode ranking.** Both
graphs are executable and numerically correct at N=96. Their complete
23/27-product sequences survive capture. The two paired sessions do not
support a reproducible Laderman advantage. The historical eager result
also remains inconclusive. This does not establish a hardware benefit from
the rank-23 decomposition, nor does it rule out different execution
organizations. The conclusion is limited to fixed-address replay, 20
operations per graph, this GPU, size, precision policy, and two sessions.

**One evidence-backed next step:** conduct a *resource-only* compiler
feasibility assessment of one bounded output-fused Laderman design before
any further latency benchmarking. The staged graph pair remains near parity
despite fewer products, while it still materializes every FP32 product.
Removing that storage would change the cost balance, but the nine-output
accumulator pressure identified in MI-Research-6 is a clear compile risk.
No fusion work is started in this milestone.

### Reproduction and retained evidence

The [evidence root](../artifacts/mi-research-6b/) contains independent
[`graph-s1`](../artifacts/mi-research-6b/graph-s1/) and
[`graph-s2`](../artifacts/mi-research-6b/graph-s2/) bundles. Each retains
the exact source and coefficients, a diagnostic-script snapshot, compiler
text and metadata, full profiler traces, capture preflight, repeated
correctness records, raw timing blocks, a summary, and a SHA-256 manifest.
[`analysis.json`](../artifacts/mi-research-6b/analysis.json) replays the
cross-session classification and references the unchanged historical eager
manifest. No CUBIN or SASS binary is retained.

To repeat the bounded experiment, use two **fresh** output paths and a
CUDA-enabled project `.venv`:

```sh
.venv/bin/python scripts/run_mi_research_6b_graph.py --session 1 --output /tmp/mi-research-6b-graph-s1
.venv/bin/python scripts/run_mi_research_6b_graph.py --session 2 --output /tmp/mi-research-6b-graph-s2
```

To verify retained evidence without GPU execution:

```sh
.venv/bin/python scripts/run_mi_research_6b_graph.py --verify --output artifacts/mi-research-6b/graph-s1
.venv/bin/python scripts/run_mi_research_6b_graph.py --verify --output artifacts/mi-research-6b/graph-s2
.venv/bin/python scripts/analyze_mi_research_6b.py --verify
.venv/bin/python -m pytest -q tests/mi_research_6b_test.py tests/mi_research_6_test.py tests/mi_research_2c_test.py
```
