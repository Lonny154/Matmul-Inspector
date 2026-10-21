# MI-Research-6F — staged CUDA Graph slow-block attribution

## Question and frozen control

**Hypothesis.** A long staged-Laderman CUDA-event block might coincide with
longer product or reconstruction kernels, inter-node idle time, or host
submission delay. The control is the retained MI-Research-6D and 6E timing
evidence. This diagnostic leaves both kernels, their algebra, the graph
topology, and the schema-v3 decision rule unchanged. The checkpoint is
`5c914467985028a38e297cdb0c5d93186d75e35f`.

The [protocol](../artifacts/mi-research-6f/protocol.json) was written before
GPU collection. It fixes one RTX 4060 Ti N=96 FP16 workload, 20 complete
staged operations per graph, ten eager and ten graph warmups, 12 graph-level
blocks in session 1, and two node-level blocks in session 2. Each block is
one graph replay. The six established input cases, FP32-rounded reference,
`atol=rtol=0.01`, scratch poisoning, and fixed addresses are unchanged.
The staged source SHA-256 is
`8b328950567868b2d119e021a0dd3bcc312d6e1c8138a1577b0e0491949fbca1`.
The fused Perminov source is unchanged but is a historical reference only;
it was not run in this capture. Both 6D and 6E retained evidence passed
offline verification before execution.

The independent 6D/6E data place ordinary staged CUDA-event blocks near
58–65 µs per complete operation. Session 6E staged-null 1 had an isolated
80.717 µs block; 6D had several 97–128 µs blocks and much wider A/A gaps.
We therefore predeclared ≤65 µs/op as the descriptive historical normal
band and ≥80 µs/op as the historical slow band. Values between them remain
intermediate. For the lower-overhead graph diagnostic, a separate profiled
relative anomaly required at least 1.35 times the median of its first four
blocks. These thresholds describe evidence; they do not alter any
benchmark decision rule. Absolute profiled and unprofiled times are not
exchangeable.

## Capture and numerical gate

Nsight Systems 2026.1.3 advertises CUDA Graph `graph` and `node` granularity,
NVTX capture, and CUDA API tracing. Session 1 requested graph-level tracing
with one NVTX capture range. It saved all 12 CUDA-event blocks but Nsight
generated **no report**. A short CPU-only smoke test reproduced that failure.
A full-process smoke trace contained the NVTX range. A separate smoke test
showed that `cudaProfilerStart/Stop` did start and stop a ranged capture.
The [amendment](../artifacts/mi-research-6f/capture_amendment.json) was saved
before session 2. Session 2 used the profiler API trigger around the same
`MI6F_TRACE` NVTX range and exactly two measured blocks. This changed only
how Nsight started collection. It did not change the workload or limits.

Both fresh processes passed 12 graph correctness checks each: two poisoned
replays for every seed-0/1/2, zero, identity, and cancellation case. All
checks had zero failing and nonfinite output elements. The largest absolute
error was 0.0078125. Every saved graph output matched its own eager call
bitwise. Input, coefficient, scratch, and output addresses remained fixed.
The retained records include the individual errors, source snapshots,
software versions, GPU properties, host clock checks, and SHA-256 manifests.
This verifies the tested cases and replay sequence, not every possible input.

## Bounded timing and node timeline

The CUDA-event interval brackets a graph replay of 20 complete operations.
Host submission ends before event synchronization. Each session-2 NVTX
replay ID encloses one `cudaGraphLaunch` API call, and its correlation ID
selects exactly 480 kernels. The trace shows 20 repetitions of 23
`_block_product` kernels followed by one `_reconstruct` kernel. Graph ID 2,
stream 7, and all 480 graph-node IDs occur in the same order in both
replays. The retained [kernel timeline](../artifacts/mi-research-6f/kernel_timeline.csv)
contains every node timestamp and launch correlation ID.

| Observation | Session 1 graph capture | Session 2 block 1 | Session 2 block 2 |
| --- | ---: | ---: | ---: |
| CUDA event, µs/op | median 59.646; range 58.315–68.848 | 98.635 | 84.381 |
| Host submission, µs/block | retained per block | 1,232.742 | 587.640 |
| `cudaGraphLaunch` API, µs/block | no trace | 883.298 | 467.942 |
| First-to-last kernel span, µs/block | no trace | 1,181.592 | 1,182.102 |
| Product kernel sum, µs/block | no trace | 1,131.340 | 1,132.532 |
| Reconstruction kernel sum, µs/block | no trace | 33.657 | 33.626 |
| Inter-kernel gaps, µs/block | no trace | 16.595 | 15.944 |
| Largest inter-kernel gap, µs | no trace | 0.064 | 0.064 |

The first four session-1 blocks have median 60.011 µs/op. The predeclared
1.35× relative threshold is 81.015 µs/op. None of the 12 blocks crossed it
or the 80 µs/op historical slow band. Block 4 reached 68.848 µs/op, an
intermediate observation, but there is no corresponding trace. Session 2's
98.635 and 84.381 µs/op are **node-profiled** values. They must not be
called recurrences of the unprofiled slow mode.

The two node traces have nearly identical product, reconstruction, and
gap sums. Their CUDA-event blocks differ by 285.088 µs, while the host
`cudaGraphLaunch` calls differ by 415.356 µs and the retained host
submission intervals differ by 645.102 µs. The graph launch API ends
47.147 and 41.756 µs before the first kernel. This supports a correlation
between the *profiled* block difference and host launch behavior, rather
than a change in the traced kernel sequence or inter-kernel gaps. These
windows overlap in time and are measured by different clocks. Their
differences cannot be added or subtracted as a causal decomposition.

The SQLite export provides event IDs but no usable GPU completion
timestamps for the CUDA timing events. It cannot locate the exact start
and end of each CUDA-event interval on the Nsight GPU timeline. Nsight also
reported that not all NVTX and CUDA events may have been collected, that
driver 13.4 was traced with 13.3 libraries, and that hardware tracing was
unavailable, so it used a software-instrumented trace. The 480 expected
kernels and two replay NVTX labels are present, but those diagnostics limit
claims about complete auxiliary event capture. Before/after `nvidia-smi`
clock, temperature, power, and utilization snapshots are retained. They
are not per-block telemetry and cannot establish external contention or a
clock cause. The large session-2 host/API times show that instrumentation
can affect the timing environment. Session 1 was launched under Nsight
even though its collection never started; it is not a clean unprofiled
reference.

## Decision and one next action

**C — still unexplained.** This bounded diagnostic identifies a host launch
correlation in the two *node-profiled* blocks. It does not isolate the
historical staged slow mode: session 1 did not capture a slow block or a
trace, and session 2 used materially heavier tracing. Longer kernels,
inter-node gaps, GPU scheduling or contention, and host queueing remain
unresolved as causes of the historical 6D/6E observations. This does not
relabel 6D's `INCONCLUSIVE` decisions or 6E's two new `B_FASTER` cross
decisions. No optimization improvement or causal mechanism is claimed.

**Next research action:** run one prospectively matched comparison of fused
Perminov against optimized conventional GEMM/cuBLAS at N=96, with the same
correctness gate and explicit eager and CUDA Graph modes, to establish
practical performance relevance. This study does not start that experiment.

## Evidence and offline reproduction

The evidence root is [artifacts/mi-research-6f](../artifacts/mi-research-6f/).
It contains the original protocol, capture amendment, both raw session
bundles and source snapshots, the session-2 `.nsys-rep` and SQLite export,
the complete node timeline, [trace analysis](../artifacts/mi-research-6f/trace_analysis.json),
and a binding [analysis manifest](../artifacts/mi-research-6f/analysis_manifest.json).
The protocol SHA-256 is
`3f1f1a58167f2cc9b5a705f51991446e813f82dd8b938f1ddaea94ccb5504979`.
No generated CUBIN, NEFF, or unrelated profiler artifact is committed.

Offline verification needs no GPU:

```sh
.venv/bin/python scripts/run_mi_research_6f_trace.py --verify --output artifacts/mi-research-6f/session1
.venv/bin/python scripts/run_mi_research_6f_trace.py --verify --output artifacts/mi-research-6f/session2
.venv/bin/python scripts/analyze_mi_research_6f.py --verify
.venv/bin/python -m pytest -q tests/mi_research_6f_test.py tests/mi_research_6e_test.py
```

The two original GPU commands were:

```sh
nsys profile --trace=cuda,nvtx --cuda-graph-trace=graph:host-only --capture-range=nvtx --nvtx-capture=MI6F_TRACE --capture-range-end=stop --sample=none --cpuctxsw=none --output=artifacts/mi-research-6f/session1_graph_trace .venv/bin/python scripts/run_mi_research_6f_trace.py --session 1 --output artifacts/mi-research-6f/session1
nsys profile --trace=cuda,nvtx --cuda-graph-trace=node:host-only --capture-range=cudaProfilerApi --capture-range-end=stop --sample=none --cpuctxsw=none --output=artifacts/mi-research-6f/session2_node_trace .venv/bin/python scripts/run_mi_research_6f_trace.py --session 2 --output artifacts/mi-research-6f/session2
nsys export --type sqlite --output artifacts/mi-research-6f/session2_node_trace.sqlite artifacts/mi-research-6f/session2_node_trace.nsys-rep
.venv/bin/python scripts/analyze_mi_research_6f.py
```

These retained session paths cannot be overwritten by the runner. Use
fresh paths for any separately authorized experiment. The session-1 script
snapshot records the original NVTX-trigger attempt; the session-2 snapshot
records the corrected capture trigger.
