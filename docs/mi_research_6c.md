# MI-Research-6C — 55-addition circuit and output-fusion feasibility

## Question and frozen control

**Hypothesis.** The 55-addition realization of Perminov's rank-23 tensor
may make local reuse and output fusion more practical than the previously
tested staged Laderman implementation. Fewer additions alone do not imply
lower GPU latency.

The control is the committed MI-Research-6/6B Laderman decomposition at
`3909752c0d2fc768a2a105cd7448f375e2b3b8ac`. Its N=96 staged path
uses 23 product kernels, one reconstruction kernel, and 94,208 bytes of
FP32 product scratch. Both historical CUDA Graph sessions were correct but
timing-inconclusive. Their evidence was verified before this branch was
created; no historical files were changed.

This study evaluates exactly three fixed circuits: MI-Research-6's
independent-form Laderman realization; Perminov's original `cr58_cn122`
58-addition realization; and the published 55-addition circuit over the
**same** Perminov tensor. The latter changes the linear circuit, not the
23-product tensor or its rank. It is not a new algebraic discovery or a
claim of global optimality among rank-23 tensors.

## Sources, license, and exact contract

The [paper](https://arxiv.org/html/2607.28676v1) describes a binary
signed-addition circuit with 13 left, 14 right, and 28 output gates. Copies
and sign changes are free in that model. Its fixed oriented tensor comes
from Perminov's pre-existing `cr58_cn122`; the paper's contribution is the
shorter circuit and a fixed-tensor optimality argument. We checked the
circuit and tensor identities. We did **not** independently rerun its
exhaustive lower-bound search.

The [published certificate repository](https://github.com/trylogical/cn122_add55)
was cloned read-only at commit
`34949f9ce50a89a5ad6b47a17f5834ad5a87a2fb`. Before executing any
upstream program, we inspected its files and licenses. This study never
executed upstream code. The retained
[`certificate.json`](../artifacts/mi-research-6c/upstream/certificate.json)
has SHA-256 `91c36b1165a71d4ba78cd8eb9c51ef9666d05e9162fdef5cba28d1d697a17dff`.
The retained [Perminov source JSON](../artifacts/mi-research-6c/upstream/perminov_source.json)
has SHA-256 `80bb4187ddf0cfc04654ca3e524f4d311c5a62245ee231a1101280b43fd2ae3a`.
The companion repository's original content and the vendored Perminov
source carry separate [MIT](../artifacts/mi-research-6c/upstream/LICENSE)
and [Perminov MIT](../artifacts/mi-research-6c/upstream/Perminov-MIT.txt)
notices. The paper's arXiv HTML states CC BY 4.0. The MI-Research-6
Laderman Maple source remains at its previously verified SHA-256
`0ee5814978ec6b35d2041d00b976b8c3a6b2a0f391836184d803b89c91b97c08`.

The independent local [checker](../scripts/run_mi_research_6c.py) parses
only JSON data. It expands both source and optimized straight-line
programs into signed integer coefficient maps. It checks gate references,
counts, output-coordinate permutation, the maps against the frozen
certificate, and every one of the 729 ordered tensor identities. Exactly
27 target coefficients are one and 702 are zero. The two Perminov factor
maps match entry for entry; their canonical factor JSON SHA-256 is
`771918bf1bd2ad2170bcb55401ea5f6184ee2b5ba1849012053b2cc07ccb31b8`.
The distinct Laderman factor JSON SHA-256 is
`b900ee6e8a5d7d796a22c78521f3cff23849299a74d846e34786319de1fb4dea`.

All three circuits were also interpreted using exact integer 2×2 **matrix**
blocks. Direct 6×6 multiplication was the independent target. Zero, left
identity, right identity, sparse, and five seeded cases passed in both
published and reordered schedules. Matrix blocks test the noncommutative
left-times-right product order; scalar-only tests would not.

## Addition DAG and scheduling results

The [analysis artifact](../artifacts/mi-research-6c/analysis.json) retains
every input, signed-addition, product, and output node; all dependency
edges; coefficient maps; each schedule order; consumer counts; creation and
last-use indices; per-step live values; and exact block-output hashes.
“Published” follows each source's gate order with the bilinear layer
between input and output networks. The second order is a deterministic
ready-node choice minimizing the next live intermediate count. It changes
no gate, sign, factor, or output. These are the only two schedules tested
per circuit.

| Circuit | Input U/V + output additions | Shared gates with >1 consumer | Published peak live intermediates | Reordered peak | Longest dependency path |
| --- | ---: | ---: | ---: | ---: | ---: |
| Laderman independent forms | 28/28 + 42 = 98 | 0 | 28 | 17 | 14 |
| Perminov source | 14/15 + 29 = 58 | 21 | 26 | 18 | 12 |
| Perminov optimized | 13/14 + 28 = 55 | 25 | 26 | 17 | 15 |

Thus 55 reduces the represented DAG by three addition nodes relative to
58 and exposes four more multiply-consumed gate results. Eight shared
input gates on the left, nine on the right, and eight in output
reconstruction are candidates for within-kernel reuse. The optimized
circuit does **not** reduce peak published-order liveness; it lowers the
reordered peak by one value and makes the critical path three operations
deeper. Scheduling cannot shorten a fixed DAG's dependency path. The
reordered paths reproduce every exact block output hash.

The live counts exclude base inputs and final output storage. If every
live intermediate were materialized as a 32×32 FP32 block, their modeled
peak storage would be 114,688/69,632 bytes for Laderman,
106,496/73,728 for Perminov-58, and 106,496/69,632 for Perminov-55
(published/reordered). These are **abstract storage estimates**, not
measured HBM traffic or compiler register allocations. The selected
Perminov-55 order has 14,848 simultaneously live FP32 tile elements when
base input fragments are included. Even with ideal even distribution over
128 threads, that is 116 values per thread before layout duplication and
dot temporaries. Output finalization can reuse storage after the last
consumer. Across separate kernels, shared gates and products would need
explicit materialization or recomputation; the model does not assign them
to physical register, shared-memory, or HBM locations.

The optimized Perminov output map has 58 direct product-to-output
dependencies, with one output depending on 11 products. Laderman has 51
direct dependencies, with at most seven for one output. The 28-gate
Perminov output circuit shares expressions to realize that denser map.
Naively giving each output block a separate owner with no product scratch
would duplicate products or require communication; the direct dependency
count is not itself an actual GPU dot count.
The optimized circuit has 30 single-consumer gates that could be recomputed
inside a consumer kernel instead of crossing a launch boundary; the source
58-gate circuit has 37. Recomputation trades additions and input loads for
storage. No such recomputation was added to the frozen circuit here.

## Output-fusion gate, recorded before GPU execution

The [pre-probe decision](../artifacts/mi-research-6c/pre_probe_gate.json)
was saved before compilation. Three execution organizations were assessed:

| Organization | Ownership and materialization | Feasibility conclusion |
| --- | --- | --- |
| Existing staged Laderman | One product buffer per term, then reconstruction; 24 physical kernels and 94,208 B explicit product scratch at N=96 | Already verified in MI-Research-6/6B. It does not execute the new 55-gate sharing pattern. |
| Product-owned partial output updates | A product could update FP32 output tiles through serialized launches, but the published shared output circuit has cross-product dependencies | A 55-gate implementation would need cross-launch intermediate storage or recomputation. Independent output owners would duplicate products. No bounded ownership design was chosen. |
| One output-owned kernel | Each of four 16×16 tile CTAs owns its corresponding tile in **all nine** output blocks, executes all 23 dots and 55 gates locally, then stores nine FP16 tiles | Exact DAG, disjoint ownership, deterministic FP32 accumulation, and bounded explicit memory passed the static gate. One implementation was selected for compilation and correctness only. |

For **Laderman**, serial product-owned partial updates could replace the
23-product scratch with nine FP32 output accumulators. Each launch would
need disjoint tile owners, and the output would still require FP16
finalization. That is a different 51-contribution accumulation circuit,
not the selected probe. An all-output fused Laderman CTA would also have
only four configured CTAs and 23 dots, while retaining 98 independent-form
additions. Its reordered DAG's optimistic local live footprint is 128
FP32 values per thread. The existing staged implementation is a stronger
verified control for this study.

For **Perminov-55**, a faithful staged implementation would have to retain
or recompute the 17 multiply-consumed shared U/V gate results when
preparation and products are separate kernels. The eight shared W gates
could remain inside a reconstruction kernel. Naively forming each product's operands
from flat coefficients inside its own kernel would execute the Perminov
tensor but would *not* execute the 55-addition circuit. Product-owned
partial updates likewise break the shared 28-gate output network unless
its intermediates are explicitly stored or recomputed. No physical launch
count is assigned to those unimplemented options. The selected
all-output owner keeps every shared expression within one CTA, so it
preserves the exact circuit without cross-kernel data dependencies.

The selected design has a 2×2 configured grid, four CTAs, no explicit
rank-product buffer, and one expected launch. It trades 24 kernel boundaries
and 94,208 B of staged product scratch for a long 23-dot program with low
configured parallelism. Four CTAs are **not** a measured occupancy figure.
The 116-value/thread estimate is optimistic and not a compiler register
prediction. The possibility of register exhaustion, spills, expensive
recomputation, and limited utilization remained a real gate risk. The
selected [source](../examples/kernels/mi_research_6c_fused_perminov55.py)
was generated deterministically from the checked liveness order and frozen
before the GPU run. The verifier checks its exact bytes and hash against
the circuit.

## Narrow RTX 4060 Ti GPU probe

The one selected source compiled at N=96 with 16×16×32 `tf32x3` dots, four
warps, and two stages. A CTA computes the nine output tiles it owns.
Inputs and output are FP16. Input combinations, dot accumulation, and
output reconstruction are FP32; each final store rounds to FP16. The
independent numerical reference is
`torch.matmul(left.float(), right.float()).half()` with TF32 and reduced
FP16 reduction disabled in a scoped block. Original PyTorch backend
settings were restored. The established `atol=rtol=0.01` gate was
unchanged.

| GPU case | Maximum absolute error | Mean absolute error | Failing / nonfinite output elements |
| --- | ---: | ---: | ---: |
| Seed 0 | 0.015625 | 3.55e−6 | 0 / 0 |
| Seed 1 | 0.001953125 | 0.53e−6 | 0 / 0 |
| Seed 2 | 0.0078125 | 2.33e−6 | 0 / 0 |
| Zero | 0 | 0 | 0 / 0 |
| Identity | 0 | 0 | 0 / 0 |
| Cancellation | 0.0078125 | 1.06e−6 | 0 / 0 |

Each case executed twice after poisoning the entire output with NaNs.
Every check passed. The [profiler trace](../artifacts/mi-research-6c/gpu-probe-n96/trace.json)
records **one** physical `_fused` kernel and one driver launch with a
2×2×1 grid and 128 threads per CTA. The complete operation stores all
nine outputs; there is no explicit product scratch. The first synchronized
compile-and-call took 1,946.8 ms. It is setup evidence, not latency.

The compiler reported **255 registers/thread**, 1,024 B shared memory,
and zero global scratch bytes. Spill metadata was unavailable. Retained
[PTX](../artifacts/mi-research-6c/gpu-probe-n96/fused.ptx) has 552 static
`mma.sync` occurrences (23 product expressions × 24 in the generated
body), 18 static global-load occurrences, nine static global stores, and
no `ld.local`/`st.local` text. TTIR, TTGIR, and LLIR are also retained.
These are compiler artifacts, not a decoded SASS or dynamic instruction
count. The 255-register report and four-CTA grid limit the usefulness of
the static “no scratch” result as a performance predictor. The profiler's
occupancy field is an estimate; achieved occupancy was not measured.

The [GPU bundle](../artifacts/mi-research-6c/gpu-probe-n96/) retains the
source and diagnostic-script snapshots, pre-probe gate, two checks per
case, raw error distributions, compiler text and metadata, profiler trace,
raw summary, and SHA-256 manifest. No CUBIN or SASS binary is committed.
The top-level [manifest](../artifacts/mi-research-6c/hash_manifest.json)
covers the original certificate, source, licenses, exact analysis, gate,
and nested GPU bundle. No GPU latency comparison was run. The scope
allowed one, but correctness and compilation alone answer the bounded
feasibility question, while the 255-register/four-CTA result makes a
performance prediction especially uncertain. There is no supported
speedup claim relative to staged Laderman, conventional GEMM, or cuBLAS.

## Decision and one follow-up

**A — GPU execution feasibility demonstrated.** The alternative rank-23
55-addition circuit is exact, its output-owned single-kernel mapping
compiled, one complete launch was observed, and all established N=96
numerical cases passed. The result does not show that 55 additions
improve GPU performance. It does not prove general numerical stability,
better occupancy, or a practical advantage over a production GEMM.
The paper's fixed-tensor lower-bound proof was not independently repeated.

**One evidence-backed follow-up:** run one matched, bounded CUDA Graph
paired comparison of this frozen fused-55 kernel against the existing
staged Laderman path. Use the same N=96 inputs, complete-operation graph
semantics, A/A control, and resource diagnostics. The result should be
reported as an execution-organization comparison, not an isolated
addition-count effect. No such timing experiment is part of this study.

## Reproduction

The exact study and GPU artifacts replay offline without CUDA:

```sh
.venv/bin/python scripts/run_mi_research_6c.py --verify
.venv/bin/python scripts/run_mi_research_6c_gpu_probe.py --verify
.venv/bin/python -m pytest -q tests/mi_research_6c_test.py tests/mi_research_6b_test.py tests/mi_research_6_test.py
```

The selected GPU probe can be repeated only in a fresh output directory
with CUDA access. It performs one compilation and correctness/profiler
pass, with no benchmark:

```sh
.venv/bin/python scripts/run_mi_research_6c_gpu_probe.py --output /tmp/mi-research-6c-probe-fresh
```
