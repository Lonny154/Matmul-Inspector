# MI-Research-1 — hardware-aware FC1 GEMM exploration

## Pre-run research record

This record was started before candidate GPU execution on 2026-10-20. The
controlled baseline is the original packaged FP16 FC1 Triton kernel:
M=512, K=768, N=3072, `64×128×32`, four warps, three stages,
`group_size_m=1`. Every candidate uses the existing trusted-local interface,
the same inputs and PyTorch/cuBLAS correctness reference, and the MI-Timing-2
v3 paired benchmark settings. No framework benchmark or numerical tolerance
is changed. The RTX 4060 Ti is available in the CUDA-enabled session.

The prior [MI-Dogfood-2](mi_dogfood_2.md) study tested N64, N256, and K64
tiles without a supported performance win. Earlier Transformer work tried
several M/N tiles and reported low occupancy and pipeline stalls. MI-Timing-2
explained a host-clock rate disagreement and showed an A/A gap of up to
0.006728 ms/operation in one session. Those studies prevent interpreting a
new lower median or compiler-resource reduction as an optimization by itself.

### Candidate 1: full-M CTA grouping

**Hypothesis recorded before execution:** setting `group_size_m=8` with the
unchanged `64×128×32` tile makes the eight M tiles for one N tile adjacent
in program order. This may increase reuse of the much larger B operand in
cache across CTAs. It could also reduce A reuse or change scheduling in a
harmful way. The old Transformer study mentions CTA grouping but retains no
matched clock-aware A/B result for this exact candidate. The v3 A/A and RAW
controls provide a specific new reason to revisit it. The source body is a
standalone copy of the validated FC1 kernel; the spec changes only the
grouping parameter. The candidate is
[`mi_research_1_group8.json`](../examples/mi_research_1_group8.json).

### Contingent candidate 2: eight-warps per CTA

**Hypothesis recorded before execution:** retaining the baseline tile and
grouping but using eight warps may spread the same GEMM tile over more
threads, lower per-thread register pressure, and expose more ready warps.
It may instead reduce resident CTAs or add cross-warp work. This tests a
different execution organization from CTA grouping. It will be run only if
Candidate 1 leaves an unresolved performance question and the first evidence
suggests further investigation is informative. Its spec is
[`mi_research_1_warps8.json`](../examples/mi_research_1_warps8.json).

The shared trusted-local source is
[`mi_research_1_fc1.py`](../examples/kernels/mi_research_1_fc1.py).
Each successful run will preserve its exact source snapshot in its own
verified bundle. The candidate budget is at most five distinct strategies,
with at most one separate-session repeat of a promising comparison. This
study currently plans two candidates and will stop sooner if the evidence
stops being informative.

## Sequential observation after Candidate 1, before Candidate 2

The first clock-aware paired run is retained at
[`group8-v1`](../artifacts/mi-research-1/group8-v1/). It passed the reference
on seeds 0, 1, and 2 with maximum absolute error 0.0. The B compiler
reported the same 24,576 shared-memory bytes/program and 121 registers/thread
as A. Six of eight paired CUDA-event rounds favored A. A's event median was
0.060911 ms/operation and B's was 0.063204. The paired A−B median was
−0.001346 ms/operation. The A/A maximum absolute paired gap was 0.008579
ms/operation, larger than every A/B gap. The decision is `INCONCLUSIVE`.
The five-window RAW/perf ratio was about 1.00113 in this session, much closer
than in MI-Timing-2; neither host interval had an event-boundary reversal.
This does not establish a clock mechanism change or a performance ranking.

Grouping did not reduce compiler resource use and gave no supported benefit.
The low-occupancy motivation in the earlier Transformer study makes the
independent eight-warp arrangement worth one bounded test. Candidate 2 remains
a distinct hypothesis, not a parameter sweep around Candidate 1. No repeat
of Candidate 1 is planned.

## Final decision

**Decision B: the two architecture-motivated local candidates were correct,
measured, and replay-verified, but neither demonstrated a reproducible
improvement beyond observed A/A variation.** The strongest positive compiler
finding is that eight warps lowered Triton's reported registers per thread
from 121 to 78 at unchanged shared-memory allocation. The strongest timing
finding is negative: its favorable B median did not produce consistent
paired signs, and A/A variation was of comparable size. Full-M grouping
also did not show a supported benefit. Neither candidate was repeated in a
fresh session because its first result lacked a consistent advantage.

No further candidates were run. The study used **two of five** allowed new
strategies and **zero of one** allowed repeat. One lower median, one compiler
resource reduction, and a supported performance improvement are distinct
claims; only the first two were observed here.

## Exact experiment and evidence

Both specs use FP16 `(512×768) @ (768×3072)`, seed 0, the unchanged
`atol=rtol=0.01` correctness gate, 30 warmups per kernel, 100 eager
preallocated GEMM calls per timed block, and eight balanced A/B rounds.
Each v3 run also measures eight interleaved identical-A X/Y pairs and two
unranked timing controls. Every block retains CUDA-event, `perf_counter`,
and MONOTONIC_RAW total durations and exact operation count. The existing
worker compiled and warmed kernels before timing. No custom benchmark,
registry edit, or change to Matmul Inspector internals was needed.

The exact commands were:

```bash
.venv/bin/matmul-inspector-python experiment pair \
  --spec examples/mi_research_1_group8.json \
  --allow-trusted-local-source \
  --output artifacts/mi-research-1/group8-v1
.venv/bin/matmul-inspector-python experiment pair \
  --spec examples/mi_research_1_warps8.json \
  --allow-trusted-local-source \
  --output artifacts/mi-research-1/warps8-v1
```

The common local source SHA-256 is
`bccab8ab65314fb6dcb5b521146ff4d99dec88f112697f5b12e463385e9d52a9`.
Both bundles contain an exact hash-matched snapshot of that source. Candidate
A is the original packaged three-stage kernel in both runs. Candidate B
changes only the parameter shown in the table. All A and B checks passed
shape, FP16 dtype, finiteness, and PyTorch/cuBLAS `torch.matmul` comparison
on seeds 0, 1, and 2. Saved maximum absolute error was **0.0** in every
candidate/seed check. This is finite-input validation, not a proof
for every possible input.

| Candidate B | Execution change and architectural question | CUDA-event median A / B, ms/op | Paired A−B median and range, ms/op | Paired signs A / B | A/A max absolute gap, ms/op | Decision |
|---|---|---:|---:|---:|---:|---|
| [Group 8](../artifacts/mi-research-1/group8-v1/) | Visit all eight M tiles per N tile; may reuse B across CTAs | 0.060911 / 0.063204 | −0.001346; −0.005045 to +0.001577 | 6 / 2 | 0.008579 | INCONCLUSIVE |
| [Eight warps](../artifacts/mi-research-1/warps8-v1/) | Spread the same tile over twice as many warps; may lower per-thread register demand | 0.067354 / 0.064650 | +0.002767; −0.005673 to +0.008151 | 4 / 4 | 0.007301 | INCONCLUSIVE |

These are separate paired runs. Their A medians moved from 0.060911 to
0.067354 ms/op, so cross-row medians cannot be used as a matched comparison.
Group 8's paired median absolute deviation was 0.001040 ms/op; eight warps'
was 0.003682. CUDA-event CVs for A/B were 4.11%/3.95% and 6.42%/2.78%,
respectively. The per-round values, raw totals, and A/A differences are in
each `timings.jsonl`, `raw_result.json`, and `comparison.json`. Both reports
make the A/A overlap and lack of a ranking explicit.

| Run | `perf_counter` median A / B, ms/op | RAW median A / B, ms/op | Preflight RAW/perf ratio and status | Event > perf / event > RAW, out of 34 blocks |
|---|---:|---:|---|---:|
| Group 8 | 0.061475 / 0.063770 | 0.061532 / 0.063833 | 1.001132, `RATE_DIFFERENCE_OBSERVED` | 0 / 0 |
| Eight warps | 0.068130 / 0.065518 | 0.068116 / 0.065510 | 0.999947, `NO_RESOLVED_DIFFERENCE` | 0 / 0 |

The host clocks were much closer in these sessions than in MI-Timing-2's
1.054522 ratio. The first run's preflight still detected a small consistent
difference above its timestamp-read span; the second did not resolve one.
All CUDA-event blocks were shorter than both enclosing host windows. This
removes the earlier positive-boundary warning for these runs but does not
remove warm-state, order, or background variability. The A/A control itself
executes A twice before each A/B pair. It may warm A differently from B.
That limitation, plus the mixed paired signs, prevents a causal claim for
either candidate.

## Compiler observations and external anchor

| Candidate B | Shared memory A / B, bytes/program | Registers A / B, per thread | PTX static instructions A / B | Normalized PTX changed lines | Decoded SASS |
|---|---:|---:|---:|---:|---|
| Group 8 | 24,576 / 24,576 | 121 / 121 | 408 / 413 | 807 | unavailable |
| Eight warps | 24,576 / 24,576 | 121 / 78 | 408 / 280 | 978 | unavailable |

The eight-warp PTX has 16 static MMA instructions in its selected entry
versus 32 for A. Warp organization changes which instructions are emitted
per thread/program. Static PTX count is not a dynamic instruction count or
decoded machine code. The compiler metadata does not show achieved
occupancy, cache hit rate, or DRAM traffic. Group 8's source-level CTA
order plausibly changes B reuse, but the retained evidence does not measure
cache reuse directly. These are compiler observations, not explanations
proved by a profiler.

PyTorch/cuBLAS was the independent numerical reference in both new runs.
The v3 A/B workflow does not time cuBLAS in the same paired rounds, so this
study makes **no new performance comparison against cuBLAS**. The historical
[MI-Dogfood-1](mi_dogfood_1.md) cuBLAS medians were 0.066404 and
0.064538 ms/op in separate runs and shifted by 2.81%. They are an external
scale reference only. They cannot be paired with these new candidate
measurements or used to claim a cuBLAS speedup.

## Developer friction and recommendation

Matmul Inspector handled source snapshots, independent correctness gates,
compilation, warmup, paired A/B and A/A measurements, three clocks, PTX and
resource capture, manifests, and offline replay. I did not write a benchmark
harness or modify internal kernels. Three workflow costs remained:

1. I copied a full JSON spec for each candidate. The first draft silently
   inherited `num_stages=2` from the timing example. An explicit parameter
   diff before GPU execution caught it. This is a usability risk, not a
   benchmark defect.
2. The CLI reports one pair at a time. I manually extracted both verified
   outputs to build the study-wide table. This is missing aggregation, not
   missing raw evidence.
3. The v3 A/A control deliberately adds A-only work before each A/B pair.
   Its 0.007–0.009 ms/op maximum gaps are large enough to mask the observed
   candidate effects, and its A-only warm-state effect remains unisolated.
   The reports expose this limitation; the study does not alter the runner.

**Recommendation:** before another small FC1 tile or warp variation, run a
separate controlled measurement study that tests whether the A-only A/A
sequence changes subsequent A/B timing. Then evaluate a strategy expected
to change work or data movement by more than the observed null variation.
The two correct strategies here did not justify another parameter sweep.

## Verification

Each bundle retains the validated experiment spec, source snapshots,
three-seed correctness record, raw A/A and A/B timings, clock preflight,
compiler metadata, PTX comparison, report, provenance, and SHA-256 manifest.
Both runs recorded an RTX 4060 Ti (`sm_89`), NVIDIA driver 616.92,
Python 3.13.11, PyTorch 2.11.0+cu128, Triton 3.6.0, and CUDA runtime 12.8.
The Git provenance points to `30930ab` with a dirty tree because the new
research files and pre-existing unrelated changes were present during GPU
execution. The source snapshots and file hashes identify the code actually
run.

The [Group 8 manifest](../artifacts/mi-research-1/group8-v1/hash_manifest.json)
has SHA-256
`5ff30b9418a9807b97d203ccf0817f63f5cc3af7b3c862ac66684d777c539dae`.
The [eight-warp manifest](../artifacts/mi-research-1/warps8-v1/hash_manifest.json)
has SHA-256
`84193ad6209d719c35e202b6b8ce830768d2819bc63e18455339a0b8ca107460`.
Both offline verifiers returned `PASS`, eight rounds, 17 hashed files, and
`INCONCLUSIVE`.

```bash
.venv/bin/matmul-inspector-python experiment pair --verify artifacts/mi-research-1/group8-v1
.venv/bin/matmul-inspector-python experiment pair --verify artifacts/mi-research-1/warps8-v1
```

The relevant CPU suite passed **53 tests** with:

```bash
.venv/bin/python -m pytest -q \
  tests/mi_timing_2_test.py tests/experiment_pair_test.py \
  tests/experiment_pair_source_test.py tests/experiment_source_test.py \
  tests/experiment_compare_test.py tests/ptx_inspector_test.py
```

Both bundles replayed offline with no GPU execution or source import.
Historical evidence and unrelated working-tree files were left unchanged.
