# MI-Research-6 — Larger-tensor algebraic feasibility

## Question and control

**Hypothesis.** A known 3×3 rank-23 decomposition may expose useful GPU
execution choices outside the 2×2 Strassen family. Four fewer block products
could matter only if operand preparation, output reconstruction, storage,
parallelism, and numerical behavior permit a viable complete operation.

**Control.** Ordinary 3×3 block multiplication has 27 ordered block products
and nine output blocks. This study compares exact algebra and bounded source
level schedules. It does not compare measured GPU latency.

The starting checkpoint is `5947fa97b35c7b4c840d3aa8118212504ed60327`
(MI-Research-5). Its retained selection generator passed offline verification
before this study. The existing 2×2 fused rank-7/rank-8 results and the
MI-Research-5 negative performance result remain unchanged.

## Imported decomposition and exactness

The [University of Lille 3×3×3 collection](https://fmm.univ-lille.fr/3x3x3.html)
provides Laderman's 23-product decomposition. The [Maple tensor source](https://fmm.univ-lille.fr/3x3x3_tensor.mpl.bz2)
was downloaded and retained as the decompressed
[`laderman_3x3x3_source.mpl`](../artifacts/mi-research-6/laderman_3x3x3_source.mpl).
Its SHA-256 is `0ee5814978ec6b35d2041d00b976b8c3a6b2a0f391836184d803b89c91b97c08`.
The downloaded compressed file had SHA-256
`c145cff7708b8472de23e0112c0722cae05f08bed2d9883b8dbf98cbb843169e`.
The source has a free parameter `l`; this study fixes `l=1`, yielding integer
coefficients in `{−1, 0, 1}`. The parser accepts only a restricted coefficient
grammar. It never executes Maple text.

The source expresses each term with traces against its first two 3×3 factors
and **the transpose of its third factor**. We flatten the first two factors
row major into `u` and `v`, transpose the third factor, then flatten it row
major into `w`. Each term evaluates `U_i(A) @ V_i(B)` in that order and adds
`w[i,output]` times the matrix product to the output block. This is valid
for noncommuting matrix blocks. Omitting the output transpose fails the
independent tensor check.

The checked-in [`coefficients.json`](../artifacts/mi-research-6/coefficients.json)
contains all 27 conventional and 23 Laderman factor triples. Both reconstruct
**all 729 integer coefficients** of the 9×9×9 multiplication tensor. The
target has 27 nonzero entries. Each representation has zero mismatches.
An independent block interpreter checked five seeded random cases and four
structured cases (zero, left identity, right identity, sparse) with 2×2
*matrix* blocks. Both source order and liveness order matched direct 6×6
matrix multiplication exactly in every case. The tests also check a pair
of noncommuting blocks so a scalar-only verification cannot mask reversed
operand order. These are exact integer tests, separate from floating-point
correctness.

## Structural and scheduling evidence

The repository's `matrix_multiplication_tensor`, `RankOneAction`,
`sum_actions`, and `decomposition_cost` already accept nine-entry factors.
The established candidate validator and execution-schedule DAG are specific
to 2×2 blocks: they require four A/B/output blocks and a half-size
`BlockConfig`. They cannot represent this 3×3 corpus without a broader IR
change. This study adds only a CPU-only, study-specific DAG and interpreter.
It does not change the production schedule IR or GPU lowering.

| Exact source-level count | Conventional 27 | Laderman 23 |
| --- | ---: | ---: |
| Block multiplications | 27 | 23 |
| Shared-form preparation additions | 0 | 56 |
| Output reconstruction additions | 18 | 42 |
| Nonunit coefficient scalings | 0 | 0 |
| Unique nontrivial A/B forms | 0 | 30 |
| Reused A/B forms | 18 one-hot inputs | 0 nontrivial forms |
| Output product contributions | 3 per block | 3–7 per block |
| DAG nodes / edges | 63 / 90 | 89 / 192 |
| Required output-block writes | 9 | 9 |

The cost convention counts a linear combination of *s* nonzero inputs as
*s−1* additions, sharing identical forms when present. An output with *t*
product contributions costs *t−1* accumulation additions. These are exact
counts for the represented block formulas, not GPU instruction, HBM traffic,
or latency counts. Both candidates use only coefficients of magnitude one.
The conventional one-hot factors are reusable input blocks and require no
preparation additions. Laderman has 30 unique nontrivial operand forms with
no cross-term reuse in this coefficient choice.

The [offline DAGs](../artifacts/mi-research-6/execution_dags.json) contain
18 input accesses, unique operand forms, every ordered block product, nine
reconstruction nodes, and nine final output writes. Edges record the required
inputs and contributing products. No term can run before its operand forms;
no output can finalize before its contributing products. The study evaluates
only two legal deterministic orders for each algorithm:

1. **Natural:** prepare forms, evaluate products in source order, retain the
   products, then reconstruct outputs.
2. **Liveness-conscious:** choose the next term by the smallest projected
   number of live output accumulators, with source index as a deterministic
   tie break. Prepare its forms, multiply, and stream its contribution to
   owned output accumulators. This is an offline conceptual fused schedule,
   not the staged GPU design below.

Both orders reproduce every exact block case. The modeled peak number of
live output accumulators between their first and last contribution is 3
versus 1 for conventional and 8 versus 7 for Laderman. Summed over term
steps, the corresponding counts are 54 versus 18 and 118 versus 87.
Natural materialization retains 27 or 23 complete product blocks before
reconstruction. The streaming model retains one current product. These
lifetime numbers describe a source-level schedule. They exclude input
fragments, compiler temporaries, register allocation, and any output that
must remain live after its last contribution because a particular kernel
cannot write it yet. The two algorithms are not compared using identical
product order because each has its own factors; the schedule result is
within-algorithm only.

## GPU feasibility gate, recorded before any GPU implementation

The target is the retained RTX 4060 Ti with 34 SMs. A possible first GPU
shape would be N=96, divided into 3×3 blocks of size 32. A 16×16 product
tile yields four configured CTAs per product. That is sparse relative to
34 SMs; it is **not** a measurement of occupancy. The block dimensions
are compatible in principle with the project's FP32 `tf32x3` Tensor Core
dot path, subject to actual compilation.

| Conceptual design | Expected benefit | Main resource and semantic risk |
| --- | --- | --- |
| One aggressively fused launch | No product-buffer HBM traffic and one host launch | A program owning nine 16×16 output tiles needs a floor of 2,304 FP32 accumulator values, or 18 floats per thread at 128 threads. It must also hold a current 256-value product, input fragments, pointers, and preparation temporaries. Twenty-three dots and nine output reconstructions may cause large code, register pressure, spills, or compilation failure. |
| Bounded staged execution | Each product can compile separately; explicit complete-operation semantics | Fused input preparation in 23 product launches, 23 FP32 product buffers, and one reconstruction launch: **24 launches** and 94,208 bytes of product scratch at N=96. It must include all those launches and the reduction in correctness and timing. A similarly staged conventional control uses 28 launches and 110,592 bytes. Launch overhead and HBM traffic may dominate. |

Both designs require all nine final FP16 output blocks. The staged scratch
estimate is `rank × 32 × 32 × 4` bytes and excludes output, inputs, allocator
metadata, compiler scratch, and any additional temporary. A streamed fused
implementation might avoid this product storage but must prove its compiler
resource and numerical behavior. No PTX, SASS, register count, shared-memory
report, spill count, profiler launch count, achieved occupancy, or latency
exists for either 3×3 design. Previous 2×2 compiler results cannot be
extrapolated into such a report.

The exactness, matrix-block, dependency, and bounded-storage gates passed.
The staged design is the sole selected GPU probe. The ordinary workspace
sandbox reports `torch.cuda.is_available() == False`, although `nvidia-smi`
lists the RTX 4060 Ti. An isolated preflight with GPU device access reports
`CUDA: True` and identifies the RTX 4060 Ti. This gate was recorded before
writing the experimental Triton source. The bounded probe may therefore
proceed in the GPU-enabled environment, subject to the unchanged
`atol=rtol=0.01` gate and the MI-Research-3C FP32-rounded FP16 reference.

## Bounded N=96 GPU probe

The sole new experimental source is
[`mi_research_6_staged_3x3.py`](../examples/kernels/mi_research_6_staged_3x3.py).
The same dynamic-coefficient product kernel executes the 27-product control
and the 23-product Laderman variant. Each product launch forms its two
FP32 operands from the original FP16 input, uses one `tf32x3` dot, and
writes an FP32 partial block. One final kernel reconstructs all nine FP16
output blocks. Products retain their original A-form then B-form order.
The probe owns fixed coefficient, scratch, and output buffers before each
measured complete operation. No allocation or compilation is included in
the measured call. A first development compile attempt failed because
Triton required literal or `tl.constexpr` tile constants in JIT code; the
source was corrected before the retained feasibility and timing session.

The [retained GPU bundle](../artifacts/mi-research-6/gpu-probe-n96-eager-s1/)
has its own source and diagnostic-script snapshots, coefficient snapshot,
raw numerical records, per-kernel compiler IR and PTX, two profiler traces,
raw A/A and paired timing blocks, environment metadata, and SHA-256 manifest.
The RTX 4060 Ti reports 34 SMs and compute capability 8.9. PyTorch
2.11.0+cu128 and Triton 3.6.0 were used. The scoped FP32 reference was
`torch.matmul(A.float(), B.float()).half()`. TF32 for the PyTorch reference
and reduced-precision FP16 reduction were disabled, then their original
settings were restored. Both algorithms used FP16 inputs and outputs,
FP32 operand preparation and product accumulation, and Triton `tf32x3`.

| Case | Conventional maximum / mean absolute error | Laderman maximum / mean absolute error | Failing elements A / B |
| --- | ---: | ---: | ---: |
| Seed 0 | 0.0078125 / 1.45e−6 | 0.0078125 / 3.42e−6 | 0 / 0 |
| Seed 1 | 0.015625 / 2.02e−6 | 0.0078125 / 1.98e−6 | 0 / 0 |
| Seed 2 | 0.00390625 / 0.76e−6 | 0.0078125 / 2.32e−6 | 0 / 0 |
| Zero | 0 / 0 | 0 / 0 | 0 / 0 |
| Identity | 0 / 0 | 0 / 0 | 0 / 0 |
| Cancellation | 0 / 0 | 0.0078125 / 0.94e−6 | 0 / 0 |

All twelve candidate/case checks passed the existing tolerance. Numerical
agreement is case-specific and does not prove broad stability. Each output
was compared independently against the reference. The original product
scratch is fully overwritten on each call; there is no stale partial sum.

The profiler recorded **27 product events plus one reconstruction event**
for A and **23 plus one** for B. Driver launch-call counts match at 28 and
24. Each product event has a 2×2 configured CTA grid; the reconstruction
has a 9×2×2 grid. Those grids are not measured occupancy. A uses 110,592
bytes of FP32 product scratch; B uses 94,208. Their coefficient buffers
use 2,916 and 2,484 bytes. Both use an 18,432-byte FP16 output buffer.
These are explicit device allocations outside measured calls. The complete
operation includes every product launch and reconstruction launch.

The product kernel reports 56 registers per thread and 8,192 bytes of
shared memory for both algorithms. Its PTX is byte-identical for A and B,
with 24 static `mma.sync` instructions in that *one kernel body*. The
different ranks change the number of kernel launches, not this static PTX
count. The rank-specific reconstruction kernels each report 40 registers
per thread and zero shared memory, with distinct PTX hashes. The compiler
reports zero global scratch. Its `n_spills` field is unavailable; no
`ld.local` or `st.local` appears in retained PTX. This supports compiler
feasibility but does not prove the final SASS has no spills. Decoded SASS
and achieved occupancy were unavailable.

After 10 warmups per algorithm, one exploratory eager paired session used
20 complete operations per timing block and eight balanced rounds. It reused
the existing CUDA-event, synchronized `perf_counter`, RAW, host-clock
preflight, and A/A measurement functions. CUDA events bracket the submitted
operation block; host clocks include end-event synchronization. These are
not identical absolute windows. The host preflight found no resolved RAW
versus `perf_counter` rate difference.

| Timing, µs per complete operation | Conventional A | Laderman B |
| --- | ---: | ---: |
| CUDA-event median (range) | 380.878 (322.918–1048.227) | 300.542 (275.720–892.093) |
| `perf_counter` median | 382.298 | 302.149 |
| MONOTONIC_RAW median | 382.309 | 302.155 |

The paired CUDA-event A−B median was **+48.410 µs**, with a **91.569 µs**
median absolute deviation. Six rounds favored B and two favored A. Paired
differences ranged from **−299.021 to +651.722 µs**. The A/A maximum gap
was **677.155 µs**, far larger than the paired median difference. Raw
samples and ordering are retained. The lower B median is an observation,
**not an evidence-supported performance improvement**. One noisy session
cannot establish reproducibility or isolate launch count, HBM traffic,
clock state, or preparation as the cause of any difference. No matched
optimized Triton or cuBLAS timing anchor was collected.

## Decision and next direction

**A — larger-tensor hardware feasibility demonstrated.** Laderman's known
23-product algorithm is exactly valid for noncommuting 3×3 matrix blocks.
The bounded staged implementation compiled, executed all required GPU
phases, and passed the established FP16 correctness gate at N=96. The
result demonstrates *execution feasibility*, not a faster GEMM. It also
shows a substantial tradeoff: four fewer products and launches but 56
input-preparation and 42 reconstruction additions, explicit FP32 scratch,
and low configured CTA parallelism at this size.

**One next research direction:** test fixed-address CUDA Graph replay of this
same 23/27 staged pair in a separately bounded study. The current eager
pair has large A/A and paired variation, and both implementations launch
many kernels. Graph replay could reduce host submission variability while
preserving complete-operation semantics. Its effect is unproven; this is a
proposal, not a result or a new milestone started here.

## Replay and retained evidence

[`exact_and_block_results.json`](../artifacts/mi-research-6/exact_and_block_results.json)
contains exact counts, case identities, output hashes, static costs,
schedule orders/lifetimes, source/script hashes, the base commit, and the
pre-GPU hardware estimates. The [SHA-256 manifest](../artifacts/mi-research-6/hash_manifest.json)
covers the source snapshot and generated evidence. The replay recomputes the
coefficients, all 729 positions, all block cases, DAGs, static counts, and
artifact bytes. It rejects extra or changed files. The GPU bundle's own
offline verifier checks its manifest, source identities, profiler topology,
PTX identities, and summary statistics without CUDA.

```sh
.venv/bin/python scripts/run_mi_research_6_feasibility.py --verify
.venv/bin/python scripts/run_mi_research_6_gpu_probe.py --verify \
  --output artifacts/mi-research-6/gpu-probe-n96-eager-s1
.venv/bin/python -m pytest -q tests/mi_research_6_test.py tests/tensor_game_test.py tests/tensor_game_hardware_cost_test.py
```

The bounded GPU diagnostic can be repeated only in a fresh directory with
CUDA access. It performs correctness and topology gates before one paired
timing session:

```sh
.venv/bin/python scripts/run_mi_research_6_gpu_probe.py \
  --output /tmp/mi-research-6-gpu-new-session
```

The source has parameter `l=1` only. The small integer block tests are an
exact algebraic check, not a numerical stability study. The hardware table
in the pre-GPU section records estimates; the later compiler and profiler
claims come from the retained N=96 execution. No other shape, precision
mode, eager session, or graph mode was tested.
