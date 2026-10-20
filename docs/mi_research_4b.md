# MI-Research-4B — Matched fused rank-7 versus rank-8 GEMM

## Plan recorded before new GPU execution

**Hypothesis.** With launch count, output grouping, intermediate storage, and
precision matched, seven Strassen block products may cost less than eight
conventional block products. The extra Strassen input combinations and output
reconstruction may erase that benefit. A lower mathematical rank is not a
prediction of fewer PTX instructions, lower machine work, or lower latency.

**Primary control.** A new fused conventional rank-8 Triton kernel computes
all eight ordinary block products inside one launch. B is the unmodified
MI-Research-4 fused rank-7 source at SHA-256
`0257dd59374f6abb4a008bd136db1dc5633c2fb993ea0706d044f12517a2398f`.
Both use N=128 FP16 inputs/output, FP32 operand values and accumulators,
`tl.dot(input_precision="tf32x3")`, 16×16×32 tiles, four warps, two stages,
and caller-owned output. One program produces four corresponding 16×16
quadrant tiles, so each launch has a 4×4 program grid. Both store only final
FP16 output. No atomics or rank-product buffers are planned. The rank-8
source will follow the same sequential product helper and output stores as
the rank-7 source. Its one-hot input factors require no artificial additions.

The eight products are A00B00 and A01B10 for C00; A00B01 and A01B11 for
C01; A10B00 and A11B10 for C10; A10B01 and A11B11 for C11. This matches
`standard_2x2_decomposition()` in row-major action order. Rank-8 has four
source-level output additions. Rank-7 has seven products, input combinations,
and more reconstruction additions. Each kernel retains four FP32 output
tiles and one current product tile. A simple common live-value floor is
1024 + 256 FP32 elements over 128 threads, before operands, pointers, and
compiler temporaries. It is not a register-allocation or occupancy estimate.

**Correctness gate.** On seeds 0/1/2 plus zero, identity, and cancellation
cases, compare both outputs independently with
`torch.matmul(a.float(), b.float()).half()` while TF32 and reduced-precision
FP16 reduction are disabled. Use the unchanged `atol=rtol=0.01` gate. Record
individual error distributions, nonfinite counts, and rank-7 versus rank-8
agreement. Compare no-reduced-precision FP16 cuBLAS as a diagnostic. The
numerical target is an operational FP32-rounded reference, not a bitwise
specified exact reduction.

**Execution gate.** Capture PTX, compiler resource metadata, and profiler
traces before benchmarking. Require one observed GPU kernel event and one
launch call per complete operation, all four quadrants correct, and no
compiler-reported spill. PTX and reported registers are compiler evidence,
not decoded SASS or measured occupancy. Preserve any failed gate.

**Performance budget.** If the gates pass, run the existing trusted-local
paired CLI twice in fresh processes at N=128. Keep the MI-Research-4 benchmark
settings: preallocated output, 10 warmups per kernel, 20 complete operations
per measured block, eight balanced paired rounds, A/A, and all three clocks.
Use the existing decision rule. The historical registered conventional
GEMM measurement is an external reference only; it is not a matched pair in
this study. Stop if either gate fails. No graph-mode or tile sweep is planned.

## Results

### Correctness and execution structure

The bounded feasibility run used the NVIDIA RTX 4060 Ti (34 SMs, compute
capability 8.9), driver 616.92, and the repository `.venv`. Both sources
passed all six cases independently against the FP32-rounded target. Each
case had zero failing elements and no nonfinite outputs. The largest absolute
error across cases was 0.0078125 for each kernel. On the three random seeds,
the rank-8/rank-7 maximum output difference was 0.0078125. Their outputs
were close under the established tolerance but not bitwise equal; 16,363,
16,365, and 16,357 of 16,384 elements were exactly equal on seeds 0, 1,
and 2. Zero, identity, and the structured cancellation case were exactly
equal. The no-reduced-precision FP16 cuBLAS diagnostic passed all cases.
The paired CLI independently passed seeds 0/1/2 against its usual FP16
cuBLAS reference in both sessions.

The profiler trace for each candidate contains one kernel event and one
`cuLaunchKernelEx` driver call. Both wrappers make one Triton launch with a
4×4 program grid, each program writing four disjoint output tiles. Neither
source allocates a rank-product tensor or invokes a reduction kernel. Each
compiled kernel has three input/output pointers, four static PTX global
stores, no reported global scratch, and zero reported spills. That supports
the absence of **explicit** HBM rank-product materialization. It does not
measure actual DRAM transactions or every compiler/hardware temporary.

| Candidate | Mathematical block products | TTIR `tt.dot` operations | PTX static `mma.sync` instructions | Registers/thread reported | Shared memory | Spills | Global scratch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fused conventional rank-8 A | 8 | 8 | 192 | 145 | 2048 B | 0 | 0 B |
| Fused Strassen rank-7 B | 7 | 7 | 168 | 149 | 4096 B | 0 | 0 B |

The generated TTIR retains the eight versus seven dots. Their `tf32x3`
lowering expands those to 192 versus 168 static PTX MMA instructions. This
supports a compiler-level tensor-instruction reduction for this exact
specialization. It does not imply a measured execution-work or latency
reduction. PTX is not decoded SASS; decoded SASS was unavailable. The
rank-7 kernel used four more reported registers per thread and twice the
reported shared memory. Occupancy and actual data movement were not
measured. Both kernels have the same product helper, block masks, output
layout, launch topology, warps, stages, and precision. The intended
algorithmic differences are one extra rank-8 dot versus Strassen's input
combinations and larger reconstruction.

### Two paired eager sessions

Each session used a separate process and the existing v3 trusted-local pair
workflow. Compilation and correctness preceded 10 warmups per candidate.
Each of eight rounds measured 20 complete operations for A and B in
alternating A/B then B/A order. An identical-A A/A control preceded the
paired rounds. CUDA events, `perf_counter`, and `CLOCK_MONOTONIC_RAW` raw
blocks are retained separately. The table uses microseconds per complete
operation. A negative paired A−B difference means fused rank-8 was faster
in that round.

| Session | CUDA-event medians A / B | Host medians A / B | RAW medians A / B | Paired A−B median (MAD) | A/A maximum absolute gap | Paired signs A / B | Decision |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 16.333 / 16.759 | 17.754 / 18.389 | 17.747 / 18.378 | −0.614 (0.610) | 3.021 | 7 / 1 | Inconclusive |
| 2 | 17.433 / 17.944 | 19.055 / 23.050 | 19.043 / 23.039 | −0.491 (2.623) | 10.488 | 5 / 3 | Inconclusive |

The first session's paired differences ranged from −3.482 to +0.512
microseconds. The second ranged from −17.552 to +3.584 microseconds. The
second session had much wider A/A and A/B dispersion. The existing paired
decision rule marked both sessions `INCONCLUSIVE` because the A/B gaps did
not exceed the observed A/A variability. Raw and `perf_counter` did not
resolve a host-clock rate difference in either preflight. Host clocks
enclose end-event synchronization, so their absolute durations do not have
the same boundaries as CUDA-event device time. Session 2 had especially
large host-visible B samples. We retain them and do not infer their cause.
The A/A control's placement before A/B could itself affect warm state; the
earlier MI-Timing-3 study limits causal interpretation of small differences.

### External anchor and decision

The original registered conventional Triton GEMM had a historical N=128
CUDA-event median of 17.843 microseconds in MI-Research-4. That run paired
the registered GEMM against rank-7, not the new fused rank-8, and occurred
in a different session. Its latency is context, not a fresh three-way
ranking. Neither fused candidate has a matched new comparison to the
registered kernel or to cuBLAS performance. The numerical cuBLAS reference
was not a timed performance anchor in these sessions.

**B — feasible comparison, no supported advantage.** Matching launch count,
output grouping, and explicit intermediate storage exposed a real compiler
reduction from eight to seven dots and 192 to 168 static PTX MMA instructions.
The two fresh paired sessions did not establish a Strassen latency advantage.
The observed medians slightly favored fused rank-8, and measurement variation
was large enough to prevent a supported ranking under the existing rule.
Fewer mathematical products alone are therefore insufficient evidence for
this N=128 eager-mode configuration. A different workload or precision
strategy would require a new study; these results do not justify changing
the production GEMM path.

### Retained evidence and reproduction

The [rank-8 source](../examples/kernels/mi_research_4b_fused_rank8.py),
[frozen rank-7 source](../examples/kernels/mi_research_4_single_kernel_strassen.py),
and [pair specification](../examples/mi_research_4b_n128_pair.json) identify
the exact comparison. The [artifact root](../artifacts/mi-research-4b/)
contains `feasibility-v1`, `n128-eager-s1`, and `n128-eager-s2`. Each bundle
has a SHA-256 manifest. The feasibility bundle includes raw numerical
errors, cross-output comparisons, PTX, TTIR, TTGIR, LLVM IR, compiler
metadata, source snapshots, and profiler traces. Pair bundles include
all raw timings, A/A controls, compiler comparisons, source snapshots,
environment, commands, and offline reports.

```sh
.venv/bin/python scripts/run_mi_research_4b_feasibility.py --verify --output artifacts/mi-research-4b/feasibility-v1
for session in s1 s2; do
  .venv/bin/matmul-inspector-python experiment pair --verify "artifacts/mi-research-4b/n128-eager-$session"
done
.venv/bin/python -m pytest -q tests/mi_research_4b_test.py tests/mi_research_4_test.py tests/experiment_pair_test.py tests/experiment_pair_source_test.py tests/mi_timing_2_test.py
```

To collect a **new** GPU session, run the feasibility script without
`--verify` into a fresh output directory first. After it passes, run:

```sh
.venv/bin/matmul-inspector-python experiment pair \
  --spec examples/mi_research_4b_n128_pair.json \
  --allow-trusted-local-source \
  --output artifacts/mi-research-4b/new-session-name
```

Do not combine the new session with the retained ones without checking
source hashes, precision policy, and timing conditions.
