# MI-Research-5 — Hardware-aware algebraic exploration

## Plan frozen before GPU execution

**Hypothesis.** Exact change-of-basis representations of the rank-7
matrix-multiplication tensor can have different GPU resource and latency
costs even when their mathematical rank is unchanged. Reordering seven
product evaluations may reduce live output accumulators. Static descriptors
are selection tools, not latency predictions.

**Control.** Use the committed MI-Research-4C fused Strassen kernel as A.
It owns one tile in each of four output quadrants per CTA and runs one
physical GPU kernel. The new implementations retain its 16×16×32 tile,
four warps, two stages, FP32 operand preparation, `tf32x3` dots, FP16 output,
and preallocated output contract. The historical optimized conventional
Triton result is context only unless a matched external comparison fits
the budget.

**Frozen selection.** From the verified `change-of-basis-v1` representatives,
choose the first Strassen-family record with the literal Strassen cost point
`(pre=10, post=8, scaling=0, max coefficient=1)` and the first with
coefficient magnitude two. These are `0f0a7a60a1ed` and `0b88a39c7ca3`.
They test equal aggregate cost with different structure, and higher-cost
coefficient scaling with a clear numerical risk. The original Strassen
decomposition is the reference, so there are two alternative representations.
All identities, coefficients, basis transformations, and source hashes are
frozen in `artifacts/mi-research-5/selection/` before any new GPU run.

For `0f0a…`, also compare its corpus term order with the lexicographically
first order minimizing peak live output accumulators, then summed live
accumulator count. The fixed zero-based order is `(1,4,2,0,5,6,3)`. The
model counts an output tile live after its first contributing product until
its last. It estimates two rather than three simultaneously live output
tiles at peak. It does not predict compiler register allocation. The same
seven factor triples and numerical contract are used in both orders.

**Gates and budget.** Independently reconstruct all 64 integer tensor
entries before code generation. Validate N=128 on seeds 0/1/2, zero,
identity, and cancellation inputs against FP32 matmul of exact FP16 values,
rounded to FP16. Keep `atol=rtol=0.01`. For passing variants, retain TTIR,
PTX, register/shared-memory/spill metadata, source identity, and a one-launch
profiler trace. Advance feasible variants to N=256. Benchmark only variants
that pass N=256 correctness and resource gates, with the existing paired
CLI, two fresh processes per selected comparison at most, eight rounds,
balanced order, A/A, and three clocks. A direct corpus-order versus
liveness-order pair isolates the schedule change from the algebraic change.
The maximum is eight paired sessions: two for each of three variants against
the fixed Strassen reference, and two for the direct schedule pair. N=256 is
the sole new timing size;
N=128 is implementation verification. No broad tile tuning, graph mode,
new algebraic search, or general compiler work is planned. If every variant
fails at N=128, stop rather than timing an invalid comparison.

**Interpretation.** Treat representation and schedule comparisons separately.
A single session or a lower median is not a speedup. Report all failures and
uncertainty. Retain exact source and raw bundles. Historical multi-kernel
change-of-basis timing is architectural context, not a matched latency
comparison with these one-launch kernels.

## Results

### Selection, exactness, and numerical behavior

The retained `change-of-basis-v1/representatives.json` SHA-256 is
`d63ac247d2bb5e8c1e482506bceb6a9cf59752d512be47aaf26fde6c2f2b336b`.
The selected records reproduce their stored coefficients from the original
Strassen factor triples and recorded integer basis matrices. Independent
integer summation reconstructs all 64 tensor entries exactly. Their full
64-character content identities, canonical identities, transforms, and
coefficients are in the selection bundle. Both are known equivalents of
Strassen. Neither is a new multiplication algorithm.

| Variant | Corpus identity prefix | Preparation / reconstruction additions | Coefficient scalings; maximum magnitude | Support | Evaluation order | Modeled peak / sum live output tiles |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| Original Strassen reference | original factor triples | 10 / 8 | 0; 1 | 36 | committed source | not modeled here |
| Equal-cost peer | `0f0a7a60a1ed` | 10 / 8 | 0; 1 | 36 | corpus `(0,1,2,3,4,5,6)` | 3 / 14 |
| Same peer, liveness order | `0f0a7a60a1ed` | 10 / 8 | 0; 1 | 36 | `(1,4,2,0,5,6,3)` | 2 / 10 |
| Coefficient-two peer | `0b88a39c7ca3` | 14 / 12 | 4; 2 | 44 | corpus `(0,1,2,3,4,5,6)` | 4 / 19 |

The live-tile model counts output accumulators between their first and last
contribution. It excludes input fragments, product fragments, pointers,
compiler temporaries, and register-allocation effects. Its lower score for
the reordered peer is a source-level scheduling hypothesis only.

At N=128 and N=256, every variant passed six cases: seeds 0/1/2, zero,
identity, and cancellation inputs. Each candidate and the original
Strassen reference was checked independently against FP32 PyTorch matmul
of the exact FP16-valued inputs, rounded to FP16. The unchanged gate was
`atol=rtol=0.01`; TF32 and reduced-precision FP16 reduction were disabled
for the scoped diagnostic and their prior settings restored. The separately
configured FP16 cuBLAS diagnostic also passed. All candidate cases had
zero failing or nonfinite elements. The normal paired CLI separately
passed its ordinary FP16 cuBLAS correctness gate in all eight sessions.

| Variant | N=128 max / largest case mean absolute error | N=256 max / largest case mean absolute error | Maximum A/B absolute difference at N=128 / 256 |
| --- | ---: | ---: | ---: |
| Equal-cost peer | 0.015625 / 3.35e−6 | 0.015625 / 3.11e−6 | 0.015625 / 0.015625 |
| Same peer, liveness order | 0.0078125 / 2.40e−6 | 0.03125 / 3.24e−6 | 0.0078125 / 0.03125 |
| Coefficient-two peer | 0.015625 / 3.75e−6 | 0.03125 / 4.46e−6 | 0.015625 / 0.03125 |

Maximum relative error over outputs whose reference magnitude is at least
one was below 0.001 for each variant and size. The coefficient-two peer
did not fail this bounded numerical test. The different error maxima and
rounding behavior show that exact tensor identity does not mean bitwise
floating-point equality or a general stability guarantee. Raw records
retain per-case distributions, representative coordinates, and selected
FP64 diagnostic dots.

### Compiler and physical execution

All six variant/size feasibility bundles passed the one-launch topology
gate. Each profiler trace contains one `_variant_kernel` GPU event and one
driver launch call. The source writes four output quadrants with no explicit
rank-product buffer or reduction launch. The grid is 4×4 (16 configured
CTAs) at N=128 and 8×8 (64 configured CTAs) at N=256. These counts are not
measured occupancy. All variants retain seven source-level TTIR dots and
168 static PTX `mma.sync` instructions. No generated kernel reports spills
or global scratch.

| N | Kernel | Registers / thread | Shared memory | Static PTX async global loads | Spills / global scratch |
| ---: | --- | ---: | ---: | ---: | ---: |
| 128 | Original Strassen | 149 | 4096 B | 48 | 0 / 0 |
| 128 | Equal-cost peer | 149 | 4096 B | 48 | 0 / 0 |
| 128 | Same peer, liveness order | 149 | 4096 B | 48 | 0 / 0 |
| 128 | Coefficient-two peer | 149 | 5120 B | 56 | 0 / 0 |
| 256 | Original Strassen | 157 | 4096 B | 48 | 0 / 0 |
| 256 | Equal-cost peer | 148 | 4096 B | 48 | 0 / 0 |
| 256 | Same peer, liveness order | 154 | 4096 B | 48 | 0 / 0 |
| 256 | Coefficient-two peer | 157 | 5120 B | 56 | 0 / 0 |

At N=256 the equal-cost representation uses nine fewer reported registers
than the original Strassen source. Reordering that representation to reduce
modeled output-tile liveness *raises* the reported count from 148 to 154.
The coefficient-two representation uses more reported shared memory and
has more static PTX async global loads. These are compiler observations,
not measurements of occupancy, DRAM traffic, or dynamic instructions.
Decoded SASS was unavailable. PTX is not final machine code. The bundles
do not contain a structured compiler-warning channel, so absence of
warnings is not established.

### Paired eager timing at N=256

Each comparison used two fresh CLI processes, one fixed shape and input
seed, 10 warmups, 20 complete operations per timing block, eight balanced
paired rounds, and the existing A/A control before A/B. A is the committed
Strassen kernel in the first three comparisons. The direct schedule pair
uses corpus order as A and liveness order as B. CUDA events, synchronized
`perf_counter`, and `CLOCK_MONOTONIC_RAW` are retained separately. Every
host-clock preflight found no resolved RAW-versus-`perf_counter` rate
difference. The host clocks enclose synchronization, so their absolute
windows are not identical to CUDA-event windows.

The table gives microseconds per complete operation. Positive paired A−B
favors B. MAD is the paired median absolute deviation. The A/A maximum gap
is the existing workflow's variability diagnostic, not a statistical
confidence interval. Individual rounds and all three clocks remain in the
raw bundles.

| B versus A | Session | CUDA-event median A / B | Paired A−B median (MAD) | A/A max gap | Paired signs A / B | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Equal-cost peer vs Strassen | 1 | 27.639 / 30.196 | −0.442 (2.944) | 12.910 | 5 / 3 | Inconclusive |
| Equal-cost peer vs Strassen | 2 | 29.924 / 29.824 | +0.411 (2.849) | 21.917 | 4 / 4 | Inconclusive |
| Liveness order vs Strassen | 1 | 51.308 / 54.221 | +0.123 (5.673) | 34.301 | 4 / 4 | Inconclusive |
| Liveness order vs Strassen | 2 | 28.054 / 27.673 | +0.750 (2.361) | 20.960 | 3 / 5 | Inconclusive |
| Coefficient-two peer vs Strassen | 1 | 27.520 / 27.869 | −0.524 (3.549) | 13.466 | 5 / 3 | Inconclusive |
| Coefficient-two peer vs Strassen | 2 | 27.005 / 27.182 | +0.023 (1.638) | 4.762 | 4 / 4 | Inconclusive |
| Liveness vs corpus order, same algebra | 1 | 27.864 / 27.286 | +0.510 (1.235) | 18.432 | 3 / 5 | Inconclusive |
| Liveness vs corpus order, same algebra | 2 | 48.922 / 50.480 | −2.608 (3.994) | 29.499 | 6 / 2 | Inconclusive |

The largest A/A gaps exceed every paired median difference. Signs are
mixed in every session and the existing controller returned `INCONCLUSIVE`
for all eight. The roughly 28-to-51 microsecond session changes in some
comparisons show that environment or warm-state variation can dominate
small within-session differences. The available evidence does not isolate
its cause. The direct schedule pair did not establish a latency benefit from
the modeled liveness reduction. Nor does the nine-register reduction for
the equal-cost peer establish a performance improvement.

Synchronized `perf_counter` medians for A/B were 30.339/32.372 and
33.649/33.879 microseconds for the equal-cost peer sessions; 54.286/56.956
and 29.906/30.018 for liveness order; 29.075/31.433 and 28.638/29.121
for coefficient two; and 29.550/31.115 and 53.538/55.527 for the direct
schedule pair. RAW medians follow the same broad session shifts. Individual
RAW and host rounds are retained because the three clocks have different
measurement boundaries and cannot be converted into one absolute latency.

There was no new matched paired comparison against optimized conventional
Triton or cuBLAS. Earlier multi-kernel change-of-basis calibrations and
MI-Research-4C's rank-7/rank-8 results use different executions. They are
context, not a pooled external performance ranking.

### Decision and limits

**B — exact variants are feasible and exhibit compiler-resource diversity,
but no supported latency advantage was demonstrated.** Two distinct
change-of-basis representations and one evaluation-order change passed the
bounded numerical and one-launch gates. Static cost and source-level
liveness did not consistently predict compiler registers, and no paired
comparison ranked a variant faster under the existing rule. These results
do not justify launching a broad joint algebraic-and-schedule performance
search yet. A future study would need a narrower hardware hypothesis and
measurement conditions that resolve differences of this size. That is a
recommendation only; no larger search was started.

This study covers one GPU, FP16 I/O, one 16×16×32 fused tile, eager mode,
N=128 correctness and N=256 timing, two basis-equivalent representations,
and one reordered schedule. It says nothing about recursive Strassen,
different tiles, CUDA Graph replay, changing addresses, or other devices.
Coefficient magnitude two passed the sampled numerical cases, but broader
stability remains untested. No mechanism is inferred from PTX counts or
clock correlations alone.

### Reproduction and retained evidence

The [selection bundle](../artifacts/mi-research-5/selection/) contains the
full exact records, frozen choices, source hashes, and its SHA-256 manifest.
The [candidate generator](../scripts/generate_mi_research_5_candidates.py)
replays exact reconstruction, basis provenance, selection, liveness order,
and generated source bytes on a CPU. The [feasibility diagnostic](../scripts/run_mi_research_5_feasibility.py)
retains compiler IR, PTX, profiler traces, numerical records, and hashes.
The [versioned artifacts](../artifacts/mi-research-5/) also include all
eight paired CLI bundles with raw timing blocks and A/A controls. There
are no generated CUBIN or decoded SASS binaries in the versioned bundle.

```sh
.venv/bin/python scripts/generate_mi_research_5_candidates.py --verify
for label in peer-corpus peer-liveness scaled-corpus; do
  for n in 128 256; do
    .venv/bin/python scripts/run_mi_research_5_feasibility.py --verify \
      --output "artifacts/mi-research-5/$label-n$n-feasibility-v1"
  done
done
for label in peer-corpus peer-liveness scaled-corpus schedule; do
  for s in 1 2; do
    .venv/bin/matmul-inspector-python experiment pair --verify \
      "artifacts/mi-research-5/$label-n256-eager-s$s"
  done
done
.venv/bin/python -m pytest -q tests/mi_research_5_test.py tests/mi_research_4c_test.py tests/experiment_pair_source_test.py
```

For a new GPU session, run the feasibility diagnostic with `--label`,
`--size`, and a fresh `--output`. After PASS, run a fresh pair, for example:

```sh
.venv/bin/matmul-inspector-python experiment pair \
  --spec examples/mi_research_5_peer_corpus_n256_pair.json \
  --output /tmp/mi-research-5-peer-corpus-new \
  --allow-trusted-local-source
```

Use `schedule` for the direct-order pair.
Do not overwrite or combine the retained sessions. The diagnostic sources
and all paired specs are frozen to source SHA-256 identities.
