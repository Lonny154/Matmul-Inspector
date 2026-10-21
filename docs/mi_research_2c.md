# MI-Research-2C — Split-K CUDA Graph evaluation

## Plan recorded before GPU measurement

The hypothesis is that reducing host submission of Split-2's partial GEMM
and reduction may change its latency distribution on FP16 M=32, K=4096,
N=2048. MI-Research-2B linked slow complete-operation blocks to elevated
host submission, but did not establish the initiating cause. CUDA Graph
replay may change this behavior. A faster or more stable result is not
assumed.

Use the committed conventional `64×128×32`, four-warp, three-stage kernel
as A and the exact committed Split-2 source as B. Keep the established
PyTorch/cuBLAS reference, seeds 0/1/2, and `atol=rtol=0.01`. Eager mode
uses the existing schema-v3 paired CLI and its original per-call FP32
scratch allocation. Graph mode uses a narrow diagnostic because the paired
CLI does not expose graph replay. It captures 100 complete operations in
each graph: 100 conventional launches for A, and 100 ordered partial-GEMM
plus reduction pairs for B. One replay measures a 100-operation block.
Graph B uses one preallocated 0.5 MiB FP32 scratch tensor that stays live
with the graph. The partial kernel overwrites every scratch element before
each reduction. Inputs and output addresses stay fixed. Compilation is
completed before capture. Graph construction cost is recorded separately
and excluded from replay timings. Graph and eager modes are different
execution environments; their absolute latencies will not be pooled into
one ranking.

Run two fresh processes per mode in order eager-1, graph-1, graph-2,
eager-2. Each session has 30 warmups, eight balanced A/B paired rounds,
eight identical-A A/A rounds before A/B, and 100 complete operations per
block. Use the existing CUDA-event, synchronized `perf_counter`, RAW, host
submission, host-clock preflight, and paired decision rule. Keep unranked
A and B controls. Retain every raw block and any failure. Try graph debug
dump to inspect capture nodes and names; if it cannot establish a complete
sequence, treat that as an evidence limitation or stop the graph comparison.

The primary graph question is whether B slow blocks recur and whether a
graph-mode B advantage is reproducible across both sessions under the
existing paired rule. Compare event spread, MAD, paired signs, A/A spread,
and host-submission timing. A/A does not necessarily bound B-specific
variability. Do not infer that a changed distribution proves host
orchestration caused the original slow state. No kernel, tile, tolerance,
normal benchmark implementation, or historical evidence will be changed.

## Results

**Classification A: a reproducible graph-mode Split-2 advantage was
supported by complete-operation paired measurements.** The eager-mode
comparison remained inconclusive in both new sessions. This classification
is specific to the fixed-address graph workload and the two retained fresh
sessions. It does not establish that graph replay caused the historical
slow mode or that graph replay always avoids it.

The planned process order was eager-1, graph-1, graph-2, eager-2. Both eager
sessions used the existing paired CLI unchanged. Both graph sessions used
the narrow diagnostic script. All four used the same shape, source, tile,
input seeds, tolerance, eight paired rounds, eight A/A rounds, and 100
complete operations per block. The graph capture was profiled separately
before measured rounds and followed by 30 replay warmups. No sample was
removed. The original MI-Research-2 and 2B evidence remains unchanged.

### Correctness, operation identity, and setup

All A and B checks passed seeds 0, 1, and 2 against independent
PyTorch/cuBLAS `torch.matmul` at `atol=rtol=0.01`. A's maximum absolute
error was zero. B's was 0.125 on each seed in both modes. For each graph
seed, the fixed-address inputs were replaced, both outputs and the FP32
scratch tensor were poisoned with NaNs, and one graph replay recomputed
the output before comparison. This checks each full replay against fresh
inputs; it does not prove correctness for every possible input.

The graph profiler traces report **one `cudaGraphLaunch` per profiled
replay**, with 100 ordered `fc1_kernel` executions for A and 100 ordered
`split_k_partial` then `reduce_partials` pairs for B. The saved trace
sequence is checked during offline replay. These are profiler-observed
kernel events from one separate, pre-benchmark graph replay. The eager
source enqueues one A or two B kernels per complete call; these eager
sessions did not independently profile launch counts. B graph capture
keeps a 524,288-byte FP32 scratch allocation alive and reuses it across
the 100 operations and later replays. Each partial kernel writes all
logical scratch elements before its reduction. Graph capture has fixed
input and output addresses, unlike eager B's per-call logical allocation.
Capture cost is excluded from the timing below. A/B graph capture took
16.616/2.680 ms in session 1 and 1.751/17.156 ms in session 2. The
first-captured graph in each process cost more; the observations do not
isolate why. Compilation and profiler tracing were also outside timed
replays.

The committed Split-2 source hash is
`0b9144b81eed225c671518a1d7067f115865c32437ee66fe2db1b51309aa59e9`
in all four sessions. Graph compiler PTX hashes match MI-Research-2B for
conventional GEMM, partial GEMM, and reduction. This establishes source
and PTX identity, not identical runtime state or final SASS.

### Paired timing

All values are **microseconds per complete GEMM**, from untrimmed
100-operation blocks. A−B positive favors Split-2 B. MAD is the median
absolute deviation over eight rounds. The decision column uses the
unchanged Matmul Inspector schema-v3 rule independently within each mode.

| Fresh session | Event median A / B | B range; MAD; CV | Paired A−B median; range | B-faster rounds | A/A maximum gap | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Eager 1 | 64.94 / 114.06 | 34.79–120.71; 3.75; 39.4% | −50.61; −53.91 to +33.96 | 2/8 | 4.93 | `INCONCLUSIVE` |
| Graph 1 | 60.58 / 31.70 | 29.36–35.33; 2.11; 7.5% | +29.62; +24.83 to +33.84 | 8/8 | 1.88 | `B_FASTER` |
| Graph 2 | 60.46 / 31.14 | 28.79–35.78; 1.83; 7.8% | +29.44; +23.62 to +32.00 | 8/8 | 7.23 | `B_FASTER` |
| Eager 2 | 62.91 / 99.54 | 33.94–107.48; 2.84; 25.9% | −35.32; −46.02 to +27.36 | 1/8 | 3.46 | `INCONCLUSIVE` |

Graph A and B event ranges are separate in each session. Every graph
paired event gap exceeds that session's observed A/A gap. Synchronized
`perf_counter` and RAW paired directions agree in all 16 graph rounds.
Both graph sessions have `B_FASTER` under the established rule. The A/A
control was run before A/B, matching the existing workflow; its own
warm-state effect remains a limitation. The two eager sessions have
overlapping ranges and mixed paired signs, so their saved decisions remain
`INCONCLUSIVE` even though conventional A won most rounds. The earlier
MI-Research-2 eager sessions had lower Split-2 medians but the same
inconclusive decision. Eager behavior is clearly session dependent.

No graph B block in these 16 rounds resembled the eager slow blocks: graph
B ranged 28.79–35.78 µs, while eager B reached 120.71 and 107.48 µs.
The fast eager B blocks, about 34–35 µs, were near the upper end of the
graph B range. The graph B dispersion was narrower in these sessions,
especially by range and CV. Eight blocks per session do not establish a
tail-latency percentile or prove the slow state cannot recur later.

Eager B host submission medians were 11.441 and 9.993 ms per 100-call
block. Their event/submission correlations across rounds were near 1.0;
the slow B event blocks tracked slow submission. Graph B submitted one
100-operation replay per block. Its host submission medians were 0.093 and
0.143 ms, while the CUDA-event blocks took roughly 3 ms. Graph B's
event/submission correlations were 0.312 and −0.495. This correspondence
changed along with mode, scratch lifetime, and submission count. It is
consistent with reduced host pacing in graph mode, but does **not** isolate
host orchestration as the cause of the eager slow state.

Both host-clock preflights in each mode reported
`NO_RESOLVED_DIFFERENCE`; neither CUDA-event interval exceeded its
synchronized `perf_counter` or RAW enclosing interval. The full
`perf_counter`, RAW, CUDA-event, host-submission, A/A, and unranked-control
totals are retained per block. Pre/post `nvidia-smi` snapshots show changing
idle or transitional clocks and 48–49 °C. They are not in-round telemetry
and cannot explain a particular block.

### Interpretation and recommendation

The graph-mode ranking supports pursuing Split-2 **for fixed-shape,
fixed-address workloads that can amortize capture setup and replay enough
times**. It does not support an eager-mode optimization claim. The graph
comparison includes a preallocated reusable scratch tensor, so a
graph-versus-eager difference cannot be assigned solely to CUDA Graph
submission. MI-Research-2B found that scratch reuse alone did not remove
slow blocks in its diagnostic, but that is not a matched causal control for
this exact run. The captures did not measure achieved occupancy or in-round
GPU clocks. No rare-tail guarantee follows from 16 graph rounds.

The specific Split-K follow-up, if this execution mode matters to an
application, is to test its full workload with capture cost and stable
buffer lifetimes included in the application-level accounting. The eager
Split-2 line should remain inconclusive. No broader timing infrastructure
or additional partition search follows from this study.

### Reproduce and verify

From the repository root with the project `.venv` and RTX 4060 Ti:

```bash
.venv/bin/matmul-inspector-python experiment pair --spec examples/mi_research_2_low_output_split2.json --allow-trusted-local-source --output artifacts/mi-research-2c/evidence-v1/eager-1
.venv/bin/python scripts/run_mi_research_2c_graph.py --session 1 --output artifacts/mi-research-2c/evidence-v1/graph-1
.venv/bin/python scripts/run_mi_research_2c_graph.py --session 2 --output artifacts/mi-research-2c/evidence-v1/graph-2
.venv/bin/matmul-inspector-python experiment pair --spec examples/mi_research_2_low_output_split2.json --allow-trusted-local-source --output artifacts/mi-research-2c/evidence-v1/eager-2
```

Each output path must be fresh. To verify offline:

```bash
for s in eager-1 eager-2; do .venv/bin/matmul-inspector-python experiment pair --verify "artifacts/mi-research-2c/evidence-v1/$s"; done
for s in graph-1 graph-2; do .venv/bin/python scripts/run_mi_research_2c_graph.py --verify --output "artifacts/mi-research-2c/evidence-v1/$s"; done
sha256sum -c artifacts/mi-research-2c/evidence-v1/SHA256SUMS
```

The eager verifier replays its manifest, correctness and schema-v3 paired
decision. The graph verifier checks exact source/spec/script identities,
all saved file hashes, both profiler traces and kernel order, every
100-operation normalization, balanced paired order, correctness records,
and the unchanged schema-v3 decision. `analysis.json` holds a compact
cross-session derivation without replacing the raw records. The initial
graph capture preflight is also retained: PyTorch's debug-dump call returned
without a DOT file, while the one-operation profiler trace showed the
partial and reduction in order. Full-session profiler traces supplied the
required capture verification.

Validation passed in the project `.venv`: 47 relevant CPU tests; both eager
offline bundle replays (`INCONCLUSIVE`); both graph bundle replays
(`B_FASTER`, with 100/200 observed kernel nodes); and the top-level SHA-256
check for all 71 listed evidence files. The prior MI-Research-2 and 2B
checksums also passed at preflight. The graph diagnostic did not modify
the production paired benchmark or either kernel.
