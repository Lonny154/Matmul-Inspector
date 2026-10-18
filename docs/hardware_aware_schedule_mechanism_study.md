# HA-Schedule-4: recompute-vs-store mechanism isolation

HA-Schedule-3 found a rank-8 recomputation timing signal on the RTX 4060 Ti.
It did not identify the cause. The stored path has 10 launches and the
recomputed path has 9. Their consumer machine code and memory behavior also
differ. This milestone keeps the decomposition and production Triton kernels
fixed, then measures these pieces separately. It does not extend CP-SAT search
or implement HA-Schedule-5.

## Exact production topology

Both paths use the exact frozen rank-8 decomposition and FP16 inputs and
output with FP32 product accumulation. The common code writes eight FP32
rank-product buffers and reconstructs the FP16 output.

| Order | Stored path | Recomputed path |
|---:|---|---|
| 0 | `_materialize_u_form` for `A_FORM00 = -A00 - A10` | — |
| 1–2 / 0–1 | two `_shared_form_gemm` consumers loading prepared FP16 form | two `_shared_form_gemm` consumers forming FP16-rounded tiles internally |
| next six | six `_fused_input_gemm_kernel` launches | same six, with identical factor coefficients |
| last | one `_fused_reconstruction_kernel` | same reconstruction |

Only the preparation launch is present in the stored path. The first two
consumer GEMMs use the same source grid, tile shape, warps, stages, factors,
FP32 dot, and FP32 product layout. The `use_prepared` compile-time branch
changes the input-form computation and memory loads. Compiler scheduling and
shared-memory allocation can therefore differ. The remaining seven launches
are checked by name and cubin hash for both paths at every tested size.

The HA-Schedule-3 artifact retains TTIR, TTGIR, PTX, SASS, cubin hashes,
registers, shared-memory metadata, and two first-execution Nsight captures at
N=512. The earlier selective metrics were `dram__bytes_read.sum`,
`dram__bytes_write.sum`, and `lts__t_sectors.sum`. They did not measure a
complete-operation mechanism. CUDA Graph support already exists in the
repository's FC1 timing diagnostic; this study checks capture again for these
specific rank-8 paths.

## Stage and session protocol

The standalone command launches two fresh Python subprocesses. Session A
collects complete-operation timings, stage probes, graph replay, and launch
diagnostics. Session B independently repeats the complete-operation pair.
Each session has its own process ID and UTC start/end timestamps. The runner
does not pool sessions before comparing their paired round ratios.

For every size, five fixed input seeds gate both production paths against
PyTorch matmul with `rtol=0.005`, `atol=0.05`. The staged path and each of its
first two product buffers must match the unchanged production path. The
stored/recomputed outputs are compared directly. Graph replay and the dummy
diagnostic are separately correctness-gated. Failed diagnostics cannot enter
the mechanism conclusion.

Stage probes call the same compiled Triton kernels with the same arguments as
the production path. They measure preparation, each consumer GEMM, the two
consumers together, the six common GEMMs together, and reconstruction. The
stored form is prepared before the consumer-only paired timing block. It is
not prepared inside that block. All buffers are preallocated. CUDA events
bracket the actual stages. The stage pair alternates order by round and uses
the same original matrices. Raw operations and round medians are retained.
The complete-operation pair is measured separately with the original
`execute_selected` entry point. Stage medians are not naively added to infer
complete latency. Event intervals can contain host-enqueue gaps between
kernels, and device state can vary across blocks.

The default retained protocol uses 20 warmups, 100 operations per round, and
15 rounds for each complete and stage benchmark. The graph and dummy controls
use the same counts. The stage summary reports median, mean, standard
deviation, min, max, CV, round count, and operations per round. Paired
bootstrap intervals resample round medians, not individual operations.

The CUDA Graph control captures each complete production path separately
into preallocated buffers and checks replay output. Within each graph
diagnostic round, eager and graph pair blocks are interleaved. Graph replay
still executes the device kernels; it does not eliminate device work or all
launch overhead. It is a secondary diagnostic, not a replacement primary
latency number.

The optional equal-launch control uses a separate one-word scratch kernel
before the recomputed production path. It never touches matrix inputs,
products, or output. This `DIAGNOSTIC_ONLY` path is checked numerically and
must show 10 launches. Its tiny standalone duration gives launch-scale
context. It is not a production candidate or a constant to subtract from
the complete-operation result. Kernel launch overhead depends on context.

## Compiler and profiler evidence

The study captures both consumer specializations, the preparation kernel,
and one copy of each common kernel at the largest size. It retains bounded
TTIR, TTGIR, PTX, and SASS text, compiler register and static shared-memory
metadata, and cubin hashes. It does not commit cubin binaries. The comparison
normalizes only SASS instruction addresses; register and operand differences
remain visible. It records a bounded diff and selected static mnemonic
counts. PTX is not SASS. Static counts are not dynamic execution counts.

When Nsight Compute is available, the runner queries metrics, then captures
only the preparation and two consumer kernels at N=512 in two separate
processes per variant. It requests kernel duration, DRAM reads/writes, L2
sectors, and active-warp percentage if available. Nsight replays kernels and
its durations are not CUDA-event benchmark times. DRAM and L2 counters are
different memory hierarchy observations. First-execution and cache effects
limit causal interpretation. The profiler is not run over the candidate set.

HA-Schedule-2 temporary HBM bytes remain an abstract model. They are not
measured DRAM bytes. Caches, compiler fusion, tiled source reads, extra
launches, register allocation, shared memory, and execution order can all
change hardware behavior. A specific mechanism is claimed only if the
controls isolate it. One GPU and two process sessions do not establish a
universal architecture result.

## Reproduce

Use an RTX 4060 Ti environment with CUDA-enabled PyTorch and Triton. The
command refuses an existing output directory. A smoke run is a functionality
check, not retained performance evidence.

```bash
PYTHONPATH=src:. /home/daniel/IsaacLab/.venv/bin/python \
  scripts/run_hardware_aware_schedule_mechanism_study.py \
  --output /tmp/ha-schedule4-smoke --smoke

PYTHONPATH=src:. /home/daniel/IsaacLab/.venv/bin/python \
  scripts/run_hardware_aware_schedule_mechanism_study.py \
  --output artifacts/tensor-game/hardware-aware-schedule-mechanism-v1
```

The retained directory contains config, provenance, hardware/software,
topology, correctness, raw session A/B and stage timings, graph and launch
diagnostics, compiler and profiler comparisons, static/measurement joins,
aggregate results, and a SHA-256 manifest. The GPU runtime remains optional
for CPU-only package tests.

## Results and decision

The retained study ran from implementation commit `0081ffd` on an RTX 4060 Ti
with CUDA 12.8, PyTorch 2.11.0+cu128, and Triton 3.6.0. All 372 retained
correctness checks passed. The production paths had 10 stored and 9
recomputed launches in both sessions. The six common GEMMs and reconstruction
had matching cubin hashes and launch order between paths at every size.

### Complete operation and independent session

Times below are medians of 15 round medians, in microseconds. Ratios are the
median of *paired round* stored/recomputed ratios, not ratios of the two
displayed marginal medians. Parentheses give paired-bootstrap 95% intervals.
Each round contains 100 complete operations per path after 20 warmups.

| N | A stored / recomputed (µs) | A paired ratio (95% interval) | B stored / recomputed (µs) | B paired ratio (95% interval) |
|---:|---:|---:|---:|---:|
| 128 | 167.70 / 150.88 | 1.096 (1.090–1.107) | 207.36 / 193.07 | 1.085 (1.074–1.105) |
| 256 | 195.39 / 184.35 | 1.053 (1.046–1.081) | 163.01 / 148.48 | 1.092 (1.078–1.098) |
| 512 | 187.28 / 172.83 | 1.072 (1.059–1.091) | 551.31 / 491.50 | 1.131 (1.113–1.139) |
| 1024 | 444.51 / 440.61 | 1.007 (0.990–1.048) | 434.19 / 437.20 | 1.005 (0.991–1.019) |

Both fresh processes agree on recomputation's direction at 128–512 and on
parity at 1024. They do **not** agree tightly on magnitude or absolute timing:
the N=512 paired-ratio intervals do not overlap. Several round-median series
have large CVs and transient slow blocks. We therefore report the sessions
separately and do not pool them. This evidence establishes a reproducible
direction in this environment, not a stable absolute latency prediction.

### Direct stage measurements

Session A stage medians are in microseconds. Consumer numbers are the two
relevant GEMMs together. The final column is the median *paired* recomputed
minus stored consumer delta with its bootstrap interval. A positive delta
means local recomputation costs more.

| N | Materialization | Stored consumers | Recomputed consumers | Paired consumer delta (95% interval) |
|---:|---:|---:|---:|---:|
| 128 | 34.66 | 42.26 | 44.51 | −0.58 (−1.68 to +0.35) |
| 256 | 21.81 | 36.11 | 38.18 | +2.11 (+1.90 to +2.27) |
| 512 | 36.88 | 49.49 | 54.91 | +1.92 (−1.06 to +6.26) |
| 1024 | 35.74 | 123.30 | 134.43 | +11.76 (+11.04 to +12.48) |

The common six-GEMM stage medians were 126.34, 107.89, 147.98, and
301.71 µs at ascending sizes. Reconstruction was 38.27, 30.45, 44.22, and
66.24 µs. Individual consumer timings and raw event records are retained.
These CUDA-event stages include their stage's host enqueue interval. They
must not be summed or treated as isolated kernel-device durations. At 256,
local recomputation is measurably slower while the complete path is faster.
At 128 and 512 the paired consumer interval includes parity. At 1024 the
consumer penalty is clear and the eager complete-operation advantage vanishes.

### Launch and graph controls

Graph capture and correctness passed at all four sizes. The matched eager
and replay paired ratios were:

| N | Matched eager stored/recomputed (95% interval) | Graph replay stored/recomputed (95% interval) |
|---:|---:|---:|
| 128 | 1.108 (1.073–1.131) | 1.020 (0.999–1.228) |
| 256 | 1.115 (1.003–1.188) | 1.001 (0.994–1.041) |
| 512 | 1.125 (1.096–1.160) | 0.978 (0.972–0.985) |
| 1024 | 1.006 (0.984–1.040) | 0.970 (0.951–0.982) |

Thus the eager recomputation advantage largely collapses under replay;
at 512 and 1024 the stored path is faster. Replay still runs every GPU
kernel. This points to eager launch/host orchestration as a substantial
factor, not to removal of device execution cost. Graph replay absolute
medians (stored/recomputed) were 25.98/21.25, 57.26/60.90,
90.26/91.01, and 388.08/401.04 µs. As above, marginal-median ratios
are not the paired ratios; high variation is especially visible at 128.

The separate `DIAGNOSTIC_ONLY` scratch launch gives a second scale check.
Its standalone CUDA-event median was 18.06, 13.79, and 37.92 µs at
128, 256, and 512. Adding it to recomputation increased diagnostic complete
medians from 154.64 to 165.76, 154.34 to 165.70, and 205.54 to
218.40 µs respectively. Matched stored medians were 168.10, 169.89, and
223.82 µs. Within these *same* diagnostic blocks, paired stored/recomputed
ratios without the dummy were 1.095, 1.096, and 1.104. Paired
stored/(recomputed+dummy) ratios narrowed to 1.022, 1.024, and 1.045.
This removes a substantial share of the matched eager gap at all three
sizes, but leaves a measurable residual. The one-word load/store kernel
itself is tiny; these event times
mostly reflect its launch context and are **not** a transferable per-launch
constant. At 1024 the retained profiler topology capture omitted the dummy
event once. Although its numerical output passed, that diagnostic failed its
topology gate and was excluded from conclusions.

### Compiler, profile, and modeled traffic

At N=1024 the two stored and two recomputed consumer specializations each
used 40 registers/thread. Stored consumers used 8,192 bytes static shared
memory each; recomputed consumers used 10,240. Their normalized SASS and
cubin hashes differed. Each consumer's saved SASS has 128 static `FFMA`
mnemonics, but recomputation has 12 `FADD` rather than 4 and 9 `LDGSTS`
rather than 6. The bounded diff also shows scheduling and operand changes.
These are static counts, not dynamic instruction or memory-transaction
counts. The corresponding HA-Schedule-3 materialization and consumer TTIR,
TTGIR, PTX, and disassembled SASS are byte-identical to this study, despite
different cubin-file hashes. The cross-study hash difference alone is not
evidence of changed machine instructions.

Selective Nsight Compute 2026.2.1.0 captured only the changed kernels at
N=512, twice per path in fresh profiler processes. Sums below are across the
stored preparation plus two consumers, or the two recomputed consumers.
They are **not** complete-operation counters or benchmark latency.

| Path / capture | Kernel duration sum (µs) | DRAM reads (B) | DRAM writes (B) | L2 sectors |
|---|---:|---:|---:|---:|
| Stored 1 | 20.032 | 807,296 | 0 | 163,165 |
| Recomputed 1 | 20.448 | 853,504 | 0 | 217,249 |
| Stored 2 | 20.032 | 813,952 | 0 | 163,313 |
| Recomputed 2 | 20.288 | 804,352 | 0 | 215,711 |

The recomputed subset is slightly *slower* on device in both profiler
captures. Its measured DRAM-read direction changes between captures, while
L2 sectors increase by roughly one third in both. The requested DRAM-write
counter returned zero for both paths, so it cannot establish whether the
prepared form was written to external memory. Consumer active-warp
percentages were about 15.75–15.80% for both variants; materialization was
about 51%. The extra shared memory and changed SASS are real compiler
differences, but these measurements do not establish an occupancy-driven win.
Nsight kernel replay, cache state, and the selected changed-kernel scope limit
causal interpretation.

| N | HA-2 modeled temporary HBM stored/recomputed (B) | Actual launches stored/recomputed | A complete stored/recomputed (µs) |
|---:|---:|---:|---:|
| 128 | 393,216 / 368,640 | 10 / 9 | 167.70 / 150.88 |
| 256 | 1,572,864 / 1,474,560 | 10 / 9 | 195.39 / 184.35 |
| 512 | 6,291,456 / 5,898,240 | 10 / 9 | 187.28 / 172.83 |
| 1024 | 25,165,824 / 23,592,960 | 10 / 9 | 444.51 / 440.61 |

HA-Schedule-2 also assigns recomputation 4,096, 16,384, 65,536, and
262,144 extra form operations and 16,384, 65,536, 262,144, and
1,048,576 extra source-read bytes at ascending sizes. It assigns both
paths eight abstract launch groups. Those groups are not the actual
10-versus-9 launches. Abstract temporary HBM bytes cannot be compared
numerically with the selective measured DRAM reads, and the measured DRAM
direction is inconsistent. The traffic-only explanation is weakened.

### Classification and next milestone

Primary classification: `LAUNCH_REDUCTION_DOMINANT_SIGNAL`. Decision gate:
`MECHANISM_IDENTIFIED`, qualified as an **eager launch/host-orchestration
signal**, strongest at 128 and 256. At 512 the graph control removes the
eager win but the dummy control leaves a residual; changed-kernel device
time and L2 traffic also differ. We cannot apportion that residual or claim
the missing device launch alone causes a fixed number of microseconds.
At 1024 the eager paths are at parity and the stored graph path wins.
The matched consumer result at 256 and 1024 specifically rejects the idea
that local recomputation is simply faster. The profile and graph controls
do not support measured DRAM reduction or compiler resource changes as the
primary explanation on this GPU.

This is actionable for HA-Schedule-5: represent **actual** launch topology,
separate materialization and consumer costs, and distinguish eager execution
from graph replay when calibrating a schedule decision. Keep modeled traffic
as a hypothesis to test, not a GPU byte prediction. Do not fit a black-box
latency formula from this one architecture and these two sessions. No
search-model change is part of HA-Schedule-4.
