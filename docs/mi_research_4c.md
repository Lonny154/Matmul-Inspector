# MI-Research-4C — Fused Strassen scaling feasibility

## Plan recorded before new GPU execution

**Hypothesis.** As N increases, the saved fused Strassen rank-7 algorithm may
gain enough from seven rather than eight block products to offset operand
preparation and reconstruction costs. MI-Research-4B found no supported
advantage at N=128. The hypothesis remains unproven.

**Control and lineage.** A is the committed fused conventional rank-8 source
from MI-Research-4B (`39e60868c5aae26af26d0470e49e6b221443ab5fa8fc1560f046bfe1842a4bd8`).
B is the committed single-kernel Strassen rank-7 source
(`0257dd59374f6abb4a008bd136db1dc5633c2fb993ea0706d044f12517a2398f`).
The historical N=128 pair and optimized conventional Triton result are
context only. They will not be rerun or pooled with new sessions.

Both kernels already compute one 16×16 output tile in each of four output
quadrants per program. N=256 needs an 8×8 grid (64 programs). N=512 needs
a 16×16 grid (256 programs). These are configured CTAs, not measured
occupancy. Each program still owns four 16×16 FP32 accumulators and one
current 16×16 product, regardless of N. The simple live-value floor from
MI-Research-4 is unchanged: roughly 18 scalar FP32 values per thread before
compiler temporaries and pointers. The K loop grows from two 32-wide chunks
at N=128 to four at N=256 and eight at N=512. This may change generated
code or resource allocation even though the per-program tile does not grow.

The original wrappers restrict N to their tested sizes. This study will
snapshot each source and extend only the wrapper shape guard to N=256 and
N=512. The Triton kernel bodies, 16×16×32 tile, four warps, two stages,
FP32 operand formation, `tf32x3` dot policy, quadrant mapping, and output
stores will remain identical to their committed versions. No resource-driven
configuration adjustment is planned. Only one narrowly justified adjustment
is permitted if compilation reports a genuine resource blocker.

**Numerical gate.** Test N=256 before N=512. For each size use seeds 0/1/2,
zero, identity, and cancellation cases. Compare A and B separately with
`torch.matmul(a.float(), b.float()).half()` under scoped highest FP32
precision, TF32 disabled, and FP16 reduced-precision reduction disabled.
The unchanged gate is `atol=rtol=0.01`. Record error distributions,
nonfinite counts, failures, A/B differences, and no-reduced-precision FP16
cuBLAS as a production-library diagnostic. Default cuBLAS FP16 is not the
mathematical oracle. Stop at N=256 if correctness or the single-launch gate
fails, and retain the failure.

**Execution and performance gates.** Before timing, retain source snapshots,
TTIR/PTX, compiler resource metadata, and profiler traces. Require one
physical kernel launch, complete quadrant output, no explicit rank-product
buffers, no reported spill or global scratch. The benchmark will use the
existing trusted-local v3 paired CLI: A rank-8, B rank-7, preallocated
output, 10 warmups, 20 complete operations per measurement, eight balanced
paired rounds, A/A, CUDA events, `perf_counter`, RAW where supported, and
the existing decision rule. Two fresh processes per eligible size, at most
four paired sessions total. No graph-mode comparison, tile sweep, or
benchmark-rule change is planned.

## Results

### Correctness and resource feasibility

Both sizes passed the explicit FP32-rounded FP16 numerical gate on seeds
0/1/2, zero, identity, and cancellation inputs. Every A and B case had zero
failing elements and no NaN or infinity. The no-reduced-precision FP16
cuBLAS diagnostic passed all cases as well. The paired CLI separately passed
seeds 0/1/2 against its ordinary FP16 cuBLAS reference in every session.
The explicit reference is the numerical target; the cuBLAS observations do
not redefine it.

| N | A maximum absolute error | B maximum absolute error | Largest case mean absolute error A / B | Maximum A/B difference | Failing elements A / B |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 0.015625 | 0.015625 | 2.98e−6 / 2.96e−6 | 0.015625 | 0 / 0 |
| 512 | 0.0625 | 0.0625 | 6.45e−6 / 6.70e−6 | 0.03125 | 0 / 0 |

The larger absolute errors occur at larger output values and pass the
unchanged mixed `atol=rtol=0.01` gate. Maximum relative error where the
reference magnitude is at least one stayed below 0.001 for both kernels
at both sizes. Rank-7 and rank-8 are not bitwise identical. The raw records
retain individual errors, representative coordinates, FP64 diagnostic dots,
error fractions, and nonfinite counts. The precision settings were restored
after each feasibility run.

Profiler traces show one `_rank8_kernel` or `_strassen_kernel` GPU event and
one `cuLaunchKernelEx` driver call per complete operation at both sizes.
Each wrapper supplies A, B, and a preallocated output to one Triton launch.
The source writes four quadrant tiles and allocates no rank-product buffer.
The compiler reports zero spills and zero global scratch. This supports the
absence of **explicit** HBM intermediates. It does not measure DRAM traffic
or prove that hardware has no transient storage. No resource-driven
configuration adjustment was needed.

| N | Grid, configured CTAs | Rank-8 registers/thread, shared | Rank-7 registers/thread, shared | TTIR dots A / B | Static PTX MMA A / B | Static PTX async global loads A / B |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128, historical | 4×4, 16 | 145, 2048 B | 149, 4096 B | 8 / 7 | 192 / 168 | 32 / 48 |
| 256 | 8×8, 64 | 146, 2048 B | 157, 4096 B | 8 / 7 | 192 / 168 | 32 / 48 |
| 512 | 16×16, 256 | 146, 2048 B | 157, 4096 B | 8 / 7 | 192 / 168 | 32 / 48 |

The static TTIR and PTX operation counts remain the same as the K loop grows
from two to four to eight chunks. Rank-7 saves one mathematical block
product and 24 static PTX `mma.sync` instructions, while its captured PTX
has more static async global-load instructions and reported resource use.
Static instructions are not dynamically executed counts or SASS. Decoded
SASS and achieved occupancy were unavailable. Configured CTA count is not
measured utilization, and the 34-SM device does not guarantee every CTA
runs concurrently. The bundles do not contain a structured compiler-warning
channel, so absence of warnings is not established.

The mathematical product work grows with the cube of N, and seven versus
eight half-size products retains the 12.5% block-product reduction. The
extra output reconstruction is performed once for each output tile. Input
combinations are formed inside every program and every K chunk, so this
specific fused implementation repeats preparation across output programs.
Its preparation cost is not merely one O(N²) whole-matrix pass. The saved
PTX load and arithmetic differences are consistent with extra work, but
they do not isolate its device-time cost or actual DRAM traffic.

### Four paired eager sessions

Each size used two independent CLI processes. A and B shared the same
inputs, FP16 workload, frozen 16×16×32 tile, four warps, two stages, and
preallocated output semantics. Each process compiled and checked both
kernels before 10 warmups per candidate. It then measured eight rounds of
20 complete operations per block in alternating A/B and B/A order, with
the existing A/A control before A/B. CUDA events, synchronized
`perf_counter`, and `CLOCK_MONOTONIC_RAW` were retained separately.
All four host-clock preflights reported no resolved RAW versus
`perf_counter` rate difference. Host durations enclose synchronization;
they are not the same absolute timing window as CUDA events.

The table uses microseconds per complete operation. Paired A−B is positive
when fused rank-7 was faster. MAD is the paired median absolute deviation.

| N, session | CUDA-event median A / B | Host median A / B | RAW median A / B | Paired A−B median (MAD) | A/A maximum gap | Paired signs A / B | Existing decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 256, 1 | 29.251 / 26.949 | 30.932 / 28.542 | 30.922 / 28.531 | +2.306 (0.234) | 17.275 | 0 / 8 | Inconclusive |
| 256, 2 | 30.150 / 27.868 | 31.894 / 29.758 | 31.883 / 29.750 | +2.259 (0.822) | 14.866 | 0 / 8 | Inconclusive |
| 512, 1 | 184.346 / 167.232 | 186.796 / 169.598 | 186.771 / 169.563 | +17.227 (0.118) | 2.099 | 0 / 8 | B faster |
| 512, 2 | 211.046 / 200.900 | 218.407 / 204.027 | 218.370 / 203.982 | +15.237 (26.393) | 50.848 | 3 / 5 | Inconclusive |

At N=256 all 16 paired rounds favored B, but the two A/A controls observed
gaps much larger than the approximately 2.3 microsecond paired medians.
The existing rule therefore did not support a ranking in either session.
At N=512, session 1 had eight consistent B-favoring paired differences
(16.914–17.422 microseconds), beyond its 2.099 microsecond A/A gap. The
existing rule ranked B faster. Session 2 had differences from −28.926 to
+49.725 microseconds and a 50.848 microsecond A/A gap. Three rounds favored
A. It was inconclusive. We retained all slow and reversed rounds. The
session-to-session change prevents a reproducible B advantage claim.

The N=128 MI-Research-4B sessions both remained inconclusive. The larger
sizes show a more frequent B-favoring observed direction, but only one of
four new sessions produced a supported decision. This pattern is not enough
to attribute a latency change to the seven-product reduction or to infer
that fixed overhead became negligible. Warm state, cache, clocks, and the
preceding A/A control remain possible influences. Neither hardware
occupancy nor memory traffic was measured.

### External anchor, decision, and next direction

The four-session budget was used for the matched fused A/B question. No new
paired measurement against the optimized registered conventional Triton
kernel or cuBLAS was made. The earlier N=128 registered-kernel result in
MI-Research-4 is historical context under a different comparison. It is
not pooled with these new timings. No practical advantage over optimized
GEMM or cuBLAS is established.

**B — scaling feasible, no demonstrated advantage.** Both unchanged kernel
bodies scale to N=256 and N=512 with one launch, correct output, and no
reported spills or scratch. The compiler retains eight versus seven dots.
Strassen's N=512 advantage appeared in one session but did not reproduce
as a supported decision in the second. The fixed-tile eager Strassen track
has now been tested at N=128, 256, and 512 without a repeatable supported
advantage. I recommend closing this fixed-implementation track and moving
to a separately scoped hardware-aware algebraic exploration that varies
mathematical representation and execution schedule together. That is a
recommendation, not work performed in this milestone.

### Reproduction and evidence

The scaling [rank-8 source](../examples/kernels/mi_research_4c_fused_rank8.py)
and [rank-7 source](../examples/kernels/mi_research_4c_fused_rank7.py)
extend only the wrapper shape guards. The [N=256 pair spec](../examples/mi_research_4c_n256_pair.json)
and [N=512 pair spec](../examples/mi_research_4c_n512_pair.json) freeze
the matched benchmark. The [versioned evidence](../artifacts/mi-research-4c/)
contains two feasibility bundles and four complete pair bundles. Each has
its own SHA-256 manifest. Raw numerical outputs, TTIR, TTGIR, LLVM IR,
PTX, compiler metadata, profiler traces, source snapshots, timing blocks,
A/A controls, and offline reports are retained where available.

```sh
for n in 256 512; do
  .venv/bin/python scripts/run_mi_research_4c_feasibility.py --verify --output "artifacts/mi-research-4c/n$n-feasibility-v1"
  for s in 1 2; do
    .venv/bin/matmul-inspector-python experiment pair --verify "artifacts/mi-research-4c/n$n-eager-s$s"
  done
done
.venv/bin/python -m pytest -q tests/mi_research_4c_test.py tests/mi_research_4b_test.py tests/mi_research_4_test.py tests/experiment_pair_test.py tests/experiment_pair_source_test.py tests/mi_timing_2_test.py
```

For a **new** GPU run, pass `--size 256` or `--size 512` to the feasibility
script with a fresh output directory. After it passes, use the matching
pair spec with `matmul-inspector-python experiment pair`,
`--allow-trusted-local-source`, and a fresh output path. New measurements
must be analyzed as new sessions. Historical bundles remain unchanged.
