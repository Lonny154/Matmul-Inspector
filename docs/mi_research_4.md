# MI-Research-4 — Single-kernel Strassen feasibility

## Plan recorded before compilation

**Hypothesis.** One-level Strassen can execute as one Triton launch without
HBM rank-product buffers. Avoiding product materialization and separate
reconstruction launches may compensate for extra operand combinations and
resource pressure. This is an empirical question, not an expected speedup.

**Controls.** The primary comparable control is the registered conventional
FP16 Triton GEMM with a 32×32×32 output tile, four warps, and two stages.
The new kernel groups four 16×16 quadrant tiles into the same 32×32 output
region per CTA. Thus both grids have four CTAs at N=64 and 16 at N=128.
The older materialized Strassen lowering has FP32 inputs and output and
seven separate product GEMMs plus reconstruction. It is useful as a
launch/materialization architecture reference but not a matched FP16
performance control. We will not compare its historical latency directly
with these new FP16 timings.

**Algebra and mapping.** With row-major blocks A00, A01, A10, A11 and the
same layout for B, use the exact seven coefficient triplets from
`strassen_2x2_decomposition()`:

1. P1=(A00+A11)(B00+B11)
2. P2=(A10+A11)B00
3. P3=A00(B01−B11)
4. P4=A11(B10−B00)
5. P5=(A00+A01)B11
6. P6=(A10−A00)(B00+B01)
7. P7=(A01−A11)(B10+B11)

Reconstruct C00=P1+P4−P5+P7, C01=P3+P5,
C10=P2+P4, and C11=P1−P2+P3+P6. A program owns one 16×16 tile in
each output quadrant. It computes products sequentially and updates four
FP32 accumulator tiles, then writes the four disjoint FP16 output tiles.
No program consumes another program's output. One launch contains four or
16 CTAs, not one CTA for the entire matrix.

**Precision.** Inputs and output are FP16. Input combinations are formed in
FP32 from the exact FP16 values. The seven products use FP32 `tl.dot` with
`input_precision="tf32x3"` and FP32 accumulation, followed by one FP16
rounding on output. The reference is FP32 PyTorch matmul of the original
FP16 values, rounded once to FP16, with TF32 disabled. This is an
operational approximation to the intended mathematical sum, not a bitwise
specified FP32 reduction order. The established trusted-local tolerance is
`atol=rtol=0.01`. No numerical tolerance will be changed.

**Precompile resource estimate.** Four persistent 16×16 FP32 output tiles
contain 1024 scalar elements. At four warps, that is a lower bound of eight
FP32 values per thread before dot fragments, pointers, operands, and compiler
temporaries. One product accumulator adds 256 FP32 elements. An A operand
tile and B operand tile each hold 16×32 FP32 elements, although the compiler
may stage or duplicate these values. The simple live-value floor is
(1024+256+512+512)/128 = 18 scalar registers per thread. This is **not** a
register allocation prediction. Seven sequential products aim to avoid
seven simultaneously live product tiles. The input combinations require
24 logical source-block loads per reduction chunk across the seven terms,
versus two for a conventional dot tile. The Strassen reduction dimension is
half as long. Cache reuse, register allocation, shared-memory staging,
spilling, and occupancy are compiler/hardware questions.

The shape pair requires seven versus eight mathematical block products, a
12.5% product-count reduction. It does not imply a 12.5% reduction in
physical MMA instructions: TF32x3 expands each FP32 dot. The small grids
also expose at most 4 or 16 CTAs on an RTX 4060 Ti with 34 SMs. This is a
potential utilization limit, not measured occupancy. Compile the 16×16×32,
four-warp, two-stage design first. Permit only one resource-driven
configuration adjustment if compilation fails or spills materially.

**Gates and budget.** Test N=64 and N=128 only, with seeds 0, 1, and 2 plus
zero, identity, and cancellation-heavy structured cases. Record errors
against the FP32-rounded target and cuBLAS with reduced-precision reduction
disabled. Require profiler evidence of one GPU kernel launch per complete
operation and compiler evidence for the single generated kernel. Only then
run the existing preallocated paired benchmark, with A/A and three clocks.
Prioritize eager execution. A graph comparison is optional and bounded to
one if correctness and resources allow. Keep all failures and raw samples.

## Feasibility results

The RTX 4060 Ti (SM 8.9, 34 SMs) compiled the one source configuration at
N=64 and N=128. The first diagnostic bundle, `feasibility-v1`, reported a
failed topology gate even though each trace contained exactly one GPU kernel
event and one `cuLaunchKernelEx` driver call. The diagnostic counted only CUDA
runtime launch calls. We preserved that bundle unchanged. The schema-2
diagnostic counts driver calls when present and runtime calls otherwise so a
trace that contains both does not double-count one launch. The new
`feasibility-v2` bundle passes offline replay. This was a diagnostic parser
error, not a second kernel launch.

Across seeds 0/1/2 and zero, identity, and cancellation-heavy inputs, both
implementations passed all 12 cases against the FP32-rounded FP16 target at
the unchanged `atol=rtol=0.01`. The largest observed absolute error for the
single-kernel Strassen output was 0.0078125 at N=64 and 0.0078125 at N=128.
The no-reduced-precision cuBLAS diagnostic also passed all 12 cases. The
paired CLI independently passed seeds 0/1/2 against its normal FP16 cuBLAS
reference at each size. These are small-shape results, not a general Strassen
numerical guarantee.

The profiler observed one `_strassen_kernel` GPU event and one driver launch
call per complete operation at both sizes. Each launch contains four CTAs at
N=64 or 16 CTAs at N=128. The source takes only A, B, and output pointers;
its four global output stores write the quadrant tiles. The compiler reports
zero spills and zero global scratch bytes in the paired captures. Together
these support no **explicit** HBM product buffers or reduction kernel. They
do not measure physical DRAM traffic or rule out every compiler/hardware
temporary.

| N | Kernel | Registers/thread reported | Shared memory | Spills | PTX static instructions | PTX tensor instructions |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 64 | Conventional A | 40 | 4096 B | 0 | 150 | 8 |
| 64 | Strassen B | 255 | 1024 B | 0 | 1603 | 168 |
| 128 | Conventional A | 44 | 4096 B | 0 | 184 | 16 |
| 128 | Strassen B | 149 | 4096 B | 0 | 2385 | 168 |

The compiled Strassen kernel is much larger and uses more reported registers
than A. The seven mathematical products are visible in the source and a
single generated PTX entry. The PTX tensor-instruction counts show that
seven FP32 `tf32x3` dots are not equivalent to seven ordinary FP16 MMA
instructions. Static PTX counts are not dynamic counts or final SASS.
Decoded SASS was unavailable. Shared-memory bytes alone do not describe
resource pressure or occupancy; occupancy was not measured.

## Paired eager timing

The first N=64 pair used the standard 30 warmups and 100 operations per
block, but its worker timed out at 180 seconds with no timing samples. A
second attempt used 10 warmups and 20 operations per block and also timed out.
Both failed bundles are retained. A CPU reproduction isolated a workflow
defect: the paired runner joined its child before reading the result queue.
The 111–150 KB Strassen PTX capture exceeds the queue pipe buffer, leaving
the child alive while the parent waits for exit. A 150 KB CPU-only payload
reproduced the block and exited as soon as the queue was read. The runner now
drains the result before joining and retains its timeout and trusted-source
failure redaction. This small infrastructure correction was necessary to use
the established paired tool; it did not change any timing calculation or
historical evidence. A 200 KB regression fixture and an exited-child fixture
cover the fix.

The completed runs used preallocated outputs, 10 warmups per kernel, 20
complete operations per measured block, and eight balanced paired rounds.
They retained A/A controls, CUDA events, `perf_counter`, and
`CLOCK_MONOTONIC_RAW`. Units below are microseconds per complete operation.
The paired difference is A minus B, so a negative value favors conventional
GEMM. Each comparison has one fresh process.

| N | CUDA event median A / B | Host median A / B | RAW median A / B | Paired A−B median (MAD) | A/A maximum absolute gap | Paired signs A / B | Decision |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 64 | 12.037 / 14.551 | 13.376 / 15.794 | 13.364 / 15.785 | −2.699 (0.687) | 2.059 | 7 / 1 | Inconclusive |
| 128 | 17.843 / 14.917 | 19.188 / 16.345 | 19.178 / 16.336 | −1.560 (3.740) | 7.270 | 5 / 3 | Inconclusive |

At N=64 the conventional kernel had the lower observed median. One paired
round reversed the sign, and the workflow's existing decision rule did not
support a ranking beyond the A/A variation. At N=128, the separate medians
appear to favor B, but the paired median favors A and the paired differences
range from −19.440 to +7.690 microseconds. The A/A gap reached 7.270
microseconds. The large variation makes a median-only speedup claim
especially misleading. Host and RAW paired signs tracked CUDA-event signs;
neither clock changed the inconclusive decision. The host clocks enclose
synchronized event completion, so their absolute durations are not the same
measurement window as device events. The host-clock preflight found no
resolved RAW versus `perf_counter` rate difference in these sessions.

The paired reports retain normalized PTX differences (2833 changed lines at
N=64 and 3706 at N=128), both PTX hashes, compiler metadata, raw per-round
timings, correctness, environment, commands, and manifests. These static PTX
differences do not establish machine-code behavior. No graph run was made:
the primary eager hypothesis had no supported advantage, and a graph run
would answer a different execution-mode question. The older materialized
Strassen implementation is FP32 with multiple GEMM/reconstruction launches.
It is not a matched FP16 latency control and its historical timings were not
reinterpreted. The numerical reference cuBLAS call was not separately timed
under matched paired conditions, so there is no cuBLAS performance ranking.

## Decision and limits

**B — feasible without a demonstrated speedup.** One-level Strassen executes
as one launch and passes the tested numerical gate without explicit product
buffers or reported spills. It did not demonstrate a repeatable complete
operation advantage over the stated conventional FP16 Triton control.
The main limitation is the tiny two-size sample with one completed timing
session each, especially the unstable N=128 session. The high reported
register count and expanded PTX are plausible performance costs, not a
proven causal explanation. A larger or differently mapped kernel would be a
new research hypothesis, not a conclusion from this study.

## Reproduction and evidence

The frozen source is [the trusted-local kernel](../examples/kernels/mi_research_4_single_kernel_strassen.py).
The two [pair specs](../examples/mi_research_4_n64_pair.json) and
[N=128 spec](../examples/mi_research_4_n128_pair.json) record the completed
settings. The [artifact root](../artifacts/mi-research-4/) contains both
feasibility bundles, both failed timeout attempts, and the two completed
pair bundles. Each bundle has its own SHA-256 manifest. The failed bundles
have no performance samples and must not be included in a latency analysis.

```sh
.venv/bin/python scripts/run_mi_research_4_feasibility.py --verify --output artifacts/mi-research-4/feasibility-v1
.venv/bin/python scripts/run_mi_research_4_feasibility.py --verify --output artifacts/mi-research-4/feasibility-v2
for bundle in n64-eager-v1 n64-eager-v2 n64-eager-v3 n128-eager-v1; do
  .venv/bin/matmul-inspector-python experiment pair --verify "artifacts/mi-research-4/$bundle"
done
.venv/bin/python -m pytest -q tests/mi_research_4_test.py tests/experiment_pair_test.py tests/experiment_pair_source_test.py tests/mi_timing_2_test.py tests/experiment_compare_test.py tests/mi_research_3c_test.py
```

To repeat GPU work into new directories, use the same feasibility command
without `--verify`, then run `matmul-inspector-python experiment pair` with
the saved N=64 or N=128 spec, `--allow-trusted-local-source`, and a fresh
`--output` path. These are new measurements; the retained results remain
unchanged.
