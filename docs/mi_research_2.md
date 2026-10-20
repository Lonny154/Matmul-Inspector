# MI-Research-2 — Split-K versus conventional GEMM

## Pre-run plan

Recorded on 2026-10-20 before candidate GPU execution. The hypothesis is
that two independent K partitions can raise the number of GEMM CTAs for a
shape with few output tiles. Any benefit must pay for FP32 partial storage,
a deterministic reduction kernel, a second launch, and the scratch allocation
performed by the trusted-local callable. No speedup is presumed.

The control is the established registered three-stage FP16 GEMM with
`64×128×32` tiles, four warps, and `group_size_m=1`. Split-2 uses the same
GEMM tile and warp/stage configuration. It computes separate FP32 partials
for two contiguous K ranges, then sums the partials in fixed order and
casts once to FP16. No atomics are used. The trusted `matmul_into` entrypoint
allocates scratch, launches both kernels, and writes the preallocated output
on every call. This makes the complete operation, including scratch
allocation and both launches, the measured B wrapper. The conventional A
wrapper writes its own preallocated output in one launch. The source returns
the reduction launch result for the existing compiler capture. That PTX and
resource metadata describe the reduction kernel only; the partial GEMM
compiler output is unavailable through this one-result interface.

Two shapes will use identical benchmark settings and independent
PyTorch/cuBLAS numerical checks on three seeds at the established
`atol=rtol=0.01`. The FC1 control is M=512, K=768, N=3072. It has
`ceil(512/64) × ceil(3072/128) = 192` conventional output CTAs. Split-2
will have 384 partial GEMM CTAs. The low-output-parallelism shape is
M=32, K=4096, N=2048. It has `1 × 16 = 16` conventional output CTAs and
32 Split-2 partial GEMM CTAs. The RTX 4060 Ti reports 34 SMs, so the low
shape has fewer conventional output CTAs than SMs. This is a scheduling
opportunity, not a measured occupancy value. The reduction grid and exact
scratch bytes will be recorded with the source and results.

Use schema-v3 eager preallocated paired comparisons: 30 warmups per kernel,
100 complete calls per timed block, eight balanced A/B rounds, eight
interleaved A/A rounds, and the existing CUDA-event, `perf_counter`, and
MONOTONIC_RAW diagnostics. The same input tensors feed A and B. Each
comparison gets a new output directory and retains source, PTX where
available, raw blocks, correctness, hashes, and offline replay. The prior
MI-Timing-3 study found that A/A placement may be associated with a small
paired-direction shift, but did not establish a causal effect. Keep the
default A/A-before method for both shapes. Interpret small differences with
that limitation in view.

The fixed initial budget is **two** comparisons: FC1 against Split-2 and
the low-output shape against Split-2. Split-4 is contingent on the first
low-output result providing a correct and directionally promising case for
more parallelism. At most one fresh-session repeat of the most promising
comparison is allowed. Incorrect candidates stop before timing. A source
or interface failure is retained and diagnosed. No benchmark engine or
registered kernel will be changed to make a candidate pass.

The decision will distinguish a supported reproducible speedup, a supported
slowdown, an inconclusive result, and inability to evaluate the complete
operation fairly. The normal paired decision rule stays authoritative.
Across-shape comparisons are explanatory, not a ranking of different
matrix multiplications. Compiler resources and static PTX do not establish
GPU occupancy or machine-code behavior. No hardware mechanism will be
claimed without direct evidence.

## Sequential observation after the two initial comparisons

Recorded after both initial GPU sessions and before any repeat. Split-2
passed the three-seed correctness gate on both shapes. At M=32, K=4096,
N=2048, its maximum absolute error against PyTorch/cuBLAS was 0.125 and
mean absolute error was about 0.0002. It passed the unchanged tolerance.
Six of eight paired rounds favored Split-2, but two B blocks took about
98 µs/operation while the other six took about 33–42 µs. The paired
decision was `INCONCLUSIVE`; the event A−B median was +23.039 µs/operation
with a −35.210 to +30.407 µs range. The B unranked control was also about
97 µs/operation. This bimodality is retained rather than removed as an
outlier. The three clocks show the same broad pattern and no event-over-host
boundary reversal. No clock, cache, or allocation cause has been isolated.

At FC1, all eight paired rounds favored conventional A. Its event median
was 63.544 µs/operation versus 74.767 for Split-2, with paired A−B median
−11.933 µs. The existing decision rule returned `A_FASTER` for this one
session. It is a supported within-session slowdown of Split-2; there is no
fresh-session FC1 repeat to call it reproducible.

The one permitted fresh-session repeat will go to the low-output Split-2
comparison. It is the only shape with a plausible gain and the two slow B
blocks make its first-session result unstable. Split-4 is deferred: another
partition count would introduce a new configuration before the basic
two-partition result is repeatable. The repeat will use the exact same spec,
source, and benchmark settings in a fresh CLI process. This decision is
based on the retained first-session evidence, not a change to the source.

## Final result

**Overall classification: inconclusive.** Split-2 demonstrated a repeated
lower *median* on the low-output shape, but did not demonstrate a supported
complete-operation speedup. Six of eight rounds favored it in each of two
fresh sessions. Two rounds in each session favored conventional GEMM. The
Split-2 timing distribution was bimodal, the event ranges overlapped, and
both existing paired decisions were `INCONCLUSIVE`. On the FC1 control,
conventional GEMM was faster in all eight rounds and the saved decision was
`A_FASTER`. That is a supported **one-session** Split-2 slowdown, not a
fresh-session reproducibility claim. No Split-4 candidate was run because
the two-way result did not become stable enough to motivate another
partition count. The study used three paired GPU sessions: two planned
comparisons and the single allowed repeat.

The complete-operation comparison **was feasible** through the existing
trusted-local interface. No benchmark infrastructure, registered kernel,
numerical tolerance, or decision rule was changed. The interface's one
returned compiler object limited B's compiler capture to the reduction
kernel. That is a compiler-evidence limitation, not a missing GPU phase in
the timed operation.

## Actual grids, storage, and launch boundaries

The GPU reported 34 SMs. These are the exact configured Triton program
counts from each launch grid, not measured active occupancy or profiler
observed launches:

| Shape | Conventional output CTAs | Split-2 partial CTAs | Reduction CTAs | FP32 partial tensor |
|---|---:|---:|---:|---:|
| FC1, 512×768×3072 | 192 | 384 | 1,536 | 12,582,912 bytes (12 MiB) |
| Low-output, 32×4096×2048 | 16 | 32 | 64 | 524,288 bytes (0.5 MiB) |

The low-output conventional grid contains only 16 CTAs for 34 SMs. Split-2
increases the partial GEMM grid to 32 CTAs, still fewer than the SM count.
That makes improved distribution of the K work plausible, but actual
occupancy and utilization were not measured. FC1 starts with 192 output
CTAs, so the same underfilled-grid argument does not apply there.

Conventional A invokes one registered GEMM launch per complete call.
Split-2 B allocates a fresh logical FP32 scratch tensor using `torch.empty`
inside `matmul_into`, enqueues one two-dimensional grid of partitioned GEMM
CTAs, then enqueues one reduction grid on the same stream. Each output
element has one writer in each partition and one writer in the reduction.
The reduction adds partition 0 and partition 1 in fixed FP32 order before
one FP16 cast. There are two explicit Triton launches per complete B call.
The pairing harness invokes exactly 100 complete calls per timed block;
allocation and both launches are inside each event and synchronized host
window. Physical allocator reuse is possible and was not separately
measured. At minimum, the partial path writes and reads 24 MiB per FC1
operation or 1 MiB per low-output operation in logical FP32 scratch traffic.
Those are source-level bytes, not measured DRAM bytes. The separate
allocation, launch, GEMM, and reduction costs cannot be disentangled from
the saved complete-operation timing.

## Correctness and paired timing

The independent reference was PyTorch/cuBLAS `torch.matmul` on seeded FP16
inputs. Both A and B passed the unchanged `atol=rtol=0.01` gate on seeds
0, 1, and 2 in all three sessions. The output was fully overwritten,
shape and dtype matched, and no nonfinite values were present. Split-2's
maximum absolute error was 0.125 in all low-output seed checks and 0.0625
in all FC1 seed checks. Its low-output mean absolute error was about
0.00019–0.00021; FC1 was about 0.000016. The different accumulation order
explains why bit equality with the reference is not required. These finite
checks do not prove correctness for every input.

All timings below are **µs per complete GEMM call**, including both Split-2
launches and its scratch allocation. A−B below zero favors conventional A.
MAD is median absolute deviation over eight paired rounds. The saved raw
files retain block totals and exact 100-call normalization before these
summaries.

| Session | CUDA event median A / B | Paired A−B median; MAD; range | A-faster / B-faster rounds | A/A maximum absolute gap | Decision |
|---|---:|---:|---:|---:|---|
| Low-output Split-2 | 62.909 / 37.252 | +23.039; 3.716; −35.210 to +30.407 | 2 / 6 | 2.877 | `INCONCLUSIVE` |
| FC1 Split-2 | 63.544 / 74.767 | −11.933; 2.085; −17.665 to −6.003 | 8 / 0 | 3.735 | `A_FASTER` |
| Low-output fresh repeat | 63.015 / 36.141 | +26.593; 2.129; −7.178 to +30.236 | 2 / 6 | 5.261 | `INCONCLUSIVE` |

| Session | `perf_counter` median A / B | RAW median A / B | Paired `perf_counter` A−B median | Paired RAW A−B median |
|---|---:|---:|---:|---:|
| Low-output Split-2 | 63.525 / 37.657 | 63.519 / 37.654 | +23.055 | +23.053 |
| FC1 Split-2 | 63.971 / 75.200 | 63.966 / 75.197 | −12.693 | −12.689 |
| Low-output fresh repeat | 63.400 / 36.527 | 63.396 / 36.524 | +26.743 | +26.740 |

The low-output B event samples were 32.5–41.8 µs in six first-session
rounds, but 98.6 and 98.2 µs in the other two. The B unranked control was
also about 97.3 µs. In the repeat, six B samples were 34.2–38.1 µs and
two were 68.2 and 69.1 µs. The slow rounds moved from rounds 5–6 to
rounds 1 and 7. A remained around 57.3–65.4 µs in the first session and
61.9–64.8 µs in the repeat. The A/A identical-kernel controls had maximum
paired gaps of only 2.9 and 5.3 µs in those sessions. They therefore did
not reproduce the much larger B-only timing state. This limits the A/A
control's usefulness for detecting candidate-specific variation.

All three host-clock preflights returned `NO_RESOLVED_DIFFERENCE`, with
RAW/`perf_counter` ratios between 0.999982 and 0.999988. No saved
CUDA-event block exceeded its enclosing host windows. The three clocks
agree on the broad timing pattern, so the earlier event-versus-host
boundary discrepancy does not explain these results. The `nvidia-smi`
snapshots around the three sessions recorded idle or transitional graphics
clocks of 210 MHz and GPU temperatures of 40–43 °C. They do not reveal
clocks or temperature during the timed blocks. No cache, allocator, or
background-work mechanism can be assigned from these snapshots. The
MI-Timing-3 A/A-placement uncertainty also remains applicable because
this study kept the default A/A-before schedule.

## Compiler evidence and its boundary

For the low-output shape, Triton reported 119 registers/thread and 24,576
shared-memory bytes/program for A. For FC1 it reported 121 registers/thread
and the same shared-memory amount. The returned B launch is
`reduce_partials`: 22 registers/thread, zero reported shared memory, and
zero spills. These numbers describe **different kernels**. They cannot be
used to claim a resource reduction for the complete Split-2 operation.
The paired report captured A's GEMM PTX and B's reduction PTX, plus their
SHA-256 identities and normalized text comparison. Comparing those PTX
files is useful for provenance, not a like-for-like GEMM compiler comparison.
The partial GEMM PTX and its register/shared-memory metadata are not exposed
by the single-return local interface. Decoded SASS was unavailable for both
sides. No machine-code or occupancy improvement is inferred.

The exact trusted source SHA-256 was
`0b9144b81eed225c671518a1d7067f115865c32437ee66fe2db1b51309aa59e9`.
The two low-output sessions retained identical A PTX hash
`2b0633490e7c09795b16b8ac18cf4f50c597c389578dfd5fa8b254cb1a071980`
and reduction PTX hash
`b43a6541cac2aa8a54ccfe276d2f02fb4ca1b6e5c7caebcc9b48f08d1d10fe5e`.
FC1 retained A PTX hash
`16be26977e1fc6b029cdaac1df5870f307b4e1e9f63f0240ec223f04f4fda668`
and reduction PTX hash
`c55e88127b68d46b35e81d79d7418baaaf52317068a19ee69f29f30df46b36a2`.

## Reproduction and retained evidence

Run from the repository root with the project `.venv`, a compatible NVIDIA
GPU, and a fresh output path. The exact first-session commands were:

```bash
.venv/bin/matmul-inspector-python experiment pair \
  --spec examples/mi_research_2_low_output_split2.json \
  --allow-trusted-local-source \
  --output artifacts/mi-research-2/evidence-v1/low-output-split2
.venv/bin/matmul-inspector-python experiment pair \
  --spec examples/mi_research_2_fc1_split2.json \
  --allow-trusted-local-source \
  --output artifacts/mi-research-2/evidence-v1/fc1-split2
```

The repeat used the first command's spec and the fresh output path
`artifacts/mi-research-2/evidence-v1/low-output-split2-repeat`. The CLI
copies the trusted source into each bundle before its worker executes it.
The recorded spec, output and source snapshot hashes, raw timing blocks,
compiler captures, reports, software/hardware provenance, and per-session
hash manifests are in the
[versioned evidence bundle](../artifacts/mi-research-2/evidence-v1/).
`preflight.json` records the original CTA, memory, and budget plan;
`analysis.json` contains descriptive offline summaries and classifications.
The top-level `SHA256SUMS` covers 62 files, including all three session
manifests and six pre/post GPU telemetry snapshots. Its SHA-256 is
`4ed1c12d4bdca9a9d443cae9afde89895dc758362d8288347ae48ed8490fd2bb`.

```bash
(cd artifacts/mi-research-2/evidence-v1 && sha256sum -c SHA256SUMS)
for session in low-output-split2 fc1-split2 low-output-split2-repeat; do
  .venv/bin/matmul-inspector-python experiment pair \
    --verify "artifacts/mi-research-2/evidence-v1/$session"
done
```

All three bundles verified offline: `PASS` with eight paired rounds and 17
hashed files each. The relevant CPU paired-workflow suite passed 69 tests.
The original MI-Timing-3 and MI-Research-1 evidence was not edited.

## Interpretation and next step

Split-K's configured grid is closer to the 34-SM count on the low-output
shape. Its repeated lower median is consistent with the proposed benefit,
but the two slow complete-operation states in each session defeat a
supported speedup claim. The FC1 control supports a one-session slowdown,
which is consistent with paying extra storage, reduction, and launch costs
where conventional tiling already offers many CTAs. Neither consistency is
proof of the underlying hardware mechanism. More partitions were not
tested; Split-4 could have different behavior.

The evidence-backed recommendation is to use complete-operation paired
timing and inspect candidate-specific raw distributions before treating
Split-K as an optimization. The low-output bimodality is the next concrete
question if this strategy is revisited. It would require a separate,
explicitly scoped diagnostic that can distinguish allocation, launches,
GPU clocks, and other effects. This milestone stops here without changing
the normal benchmark or starting another research track.
