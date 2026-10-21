# MI-Research-8 — Matched Perminov 55 vs. 58 addition circuits

## Decision

**C — Feasible comparison without a supported latency ranking.** Both circuits passed the exact, numerical, compiler, and single-launch gates. At the same 36-CTA output mapping, the original 58-addition circuit produced fewer emitted arithmetic and MMA instructions and fewer compiler-reported spills. Both fresh CUDA Graph sessions had lower median latency for 58, but both were `INCONCLUSIVE` under the unchanged paired rule. There is no demonstrated latency advantage for either circuit.

This experiment cannot isolate the effect of removing three mathematical additions. Independent output ownership duplicates different parts of the two circuit DAGs. The 55-addition whole circuit actually executes more source-level gate and dot expressions across this grid. That is a consequence of its dependency structure under the frozen mapping, not an intentional extra workload.

## Frozen circuits and exact contract

The control A is the unchanged MI-Research-7B 55-addition kernel at source SHA-256 `10b94103a2dbf3d1649d1e3d79560d45830eeb5d960161905d69a6d5947db418`. B uses the original Perminov `cr58_cn122` circuit retained and verified by MI-Research-6C. Its generated 36-CTA source SHA-256 is `1ac97e326467fac40fef9f5eca2b15e3e13a9f736d723cdfa0a746f5d0ae731f`. The base checkpoint is MI-Research-7C commit `e08ca67bca9d8af8ef1d10868f69dfb111eebbdf`.

The independent MI-Research-6C checker replayed both circuits from the saved source and certificate JSON. Each reconstructed all 729 ordered tensor coefficients: 27 ones, 702 zeros, and zero mismatches. Their 23 left, right, and output factor maps matched entry for entry. The same checker also validated noncommutative 2×2 block products on its retained cases. [`mapping_plan.json`](../artifacts/mi-research-8/mapping_plan.json) and [`protocol.json`](../artifacts/mi-research-8/protocol.json) were recorded before GPU compilation. The 6C, 7B, and 7C manifests were verified and their identities frozen there.

The new generator takes the exact 58-gate DAG in its established liveness order and emits one branch for each output block's transitive dependency slice. It uses the same source template for both circuits. That template reproduces the existing 55-addition kernel **byte for byte** before emitting B. Each CTA owns one 16×16 tile in one of nine 32×32 output blocks, for 36 CTAs in a `(2,2,9)` grid. Both use 128 threads, four warps, two stages, 16×16×32 `tf32x3` dots, FP16 inputs/output, FP32 form/product/reconstruction values, one launch, no atomics, and no explicit product scratch. Every output element has one writer.

| Exact whole-circuit property | A: 55 | B: 58 |
| --- | ---: | ---: |
| U / V / W addition gates | 13 / 14 / 28 | 14 / 15 / 29 |
| Longest dependency path | 15 | 12 |
| Reordered peak live intermediates in 6C model | 17 | 18 |

The whole-circuit peak is an abstract DAG count. A separate model of names assigned and used later within each generated output branch gives a maximum of 19 for A and 18 for B. Neither count is a physical register allocation or measured lifetime. The two models have different scopes and need not rank the circuits alike.

## Work actually present under 36-CTA ownership

| Output block, row major | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A dots per CTA | 8 | 6 | 6 | 12 | 3 | 10 | 9 | 3 | 7 |
| B dots per CTA | 8 | 6 | 6 | 11 | 3 | 10 | 5 | 3 | 6 |
| A gate additions per CTA | 20 | 16 | 20 | 33 | 2 | 31 | 30 | 2 | 22 |
| B gate additions per CTA | 21 | 19 | 14 | 29 | 2 | 27 | 15 | 2 | 13 |

Summed over all 36 CTAs, A has **256 dot expressions and 704 gate-addition instances**. B has **232 dots and 568 gate-addition instances**. Thus a whole-circuit reduction from 58 to 55 additions does not reduce work under this output slicing. The product factor maps still match exactly. The extra expressions arise because a branch must reconstruct its output using the selected circuit's internal dependencies and cannot share intermediates with another CTA. This difference limits attribution: the GPU comparison is matched on mapping and precision, but it is a comparison of the **complete circuit implementations**, including their unavoidable recomputation and product dependencies.

## GPU correctness and topology

Both sessions ran on an RTX 4060 Ti, compute capability 8.9, driver 616.92, CUDA runtime 12.8, PyTorch 2.11.0+cu128, Triton 3.6.0, and Python 3.13.11. The FP32 matmul of exact FP16 inputs, rounded once to FP16, was the independent reference. `float32_matmul_precision=highest`, TF32 disabled for the reference, and reduced FP16 reduction disabled were scoped and restored. The unchanged gate was `atol=rtol=0.01`.

Seed 0, seed 1, seed 2, zero, identity, and cancellation inputs each passed two eager and two fixed-address graph checks per implementation in both processes. The output was poisoned before every check. There were zero failing or nonfinite output elements, and graph results matched fresh eager results bitwise for each implementation. The graph used updated input contents at the same addresses. A direct pointer comparison also confirmed fixed addresses.

| Case | A maximum absolute error | B maximum absolute error |
| --- | ---: | ---: |
| Seed 0 | 0.015625 | 0.015625 |
| Seed 1 | 0.001953125 | 0.0009765625 |
| Seed 2 | 0.0078125 | 0.0078125 |
| Zero | 0 | 0 |
| Identity | 0 | 0 |
| Cancellation | 0.0078125 | 0.0078125 |

These are maxima from the retained graph checks. The raw records also retain mean errors, repeat checks, and representative elements. Separate profiler traces observed one `cudaGraphLaunch` and 20 physical `_fused` kernel events per replay for each circuit, with `[2,2,9]` grids and `[128,1,1]` blocks. Each replay therefore contains 20 complete single-kernel operations. No separate finalization launch appeared.

## Compiler comparison

The two sessions produced the same resource metadata and static text counts. TTIR and PTX are saved with SHA-256 hashes in each session. The table counts occurrences in one compiled kernel's text, not dynamic instructions or SASS.

| Compiler field or text count | A: 55 | B: 58 |
| --- | ---: | ---: |
| Registers per thread | 255 | 255 |
| Compiler-reported `n_spills` | 166 | 100 |
| Shared memory per CTA | 1,024 B | 1,024 B |
| Global scratch | 0 B | 0 B |
| `tt.dot` in TTIR | 64 | 58 |
| `arith.addf` / `arith.subf` in TTIR | 76 / 115 | 63 / 68 |
| `mma.sync` in PTX | 1,536 | 1,392 |
| `add.f32` / `sub.f32` in PTX | 1,600 / 3,496 | 1,560 / 2,552 |
| `ld.local` / `st.local` PTX text | 0 / 0 | 0 / 0 |

The 58 implementation emits fewer static FP32 arithmetic instructions and MMA expressions despite having more mathematical additions in the unsliced circuit. This agrees with its lower duplicated source-level work. The reported spill count is lower for B, but it does not quantify physical spill traffic. The 7C counters showed that the earlier 55 mapping had measured local-memory activity; this study did **not** collect matched 58 hardware counters. Neither dependency depth nor modeled source liveness can be singled out as the cause of the compiler resource difference. PTX is not final machine code.

## Paired CUDA Graph latency

Two fresh Python processes ran eight balanced paired rounds each. Session 1 initialized and captured A then B; session 2 reversed that order. A/A controls preceded each A/B round. Every timing sample is one fixed-address graph replay with 20 complete operations, normalized by 20. CUDA-event, `perf_counter`, RAW clock, preflight, all raw samples, setup times, and A/A measurements are retained. Compilation, capture, correctness, and profiler traces were outside the timed rounds. No outlier was removed and the existing decision rule was unchanged.

| Session | A median µs/op | B median µs/op | Median paired A−B µs/op | Max A/A gap µs | Existing rule |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | 10.208 | 9.462 | +0.729 | 0.554 | `INCONCLUSIVE` |
| 2 | 10.186 | 9.629 | +0.286 | 2.307 | `INCONCLUSIVE` |

Positive A−B favors B. Session 1 had one paired round favor A and another A/B gap that did not clear its A/A bound. Session 2 had large A/A variation and paired gaps in both directions. Both host-clock preflights recorded a RAW versus `perf_counter` rate difference; the decision rule retained that diagnostic without rescaling CUDA-event times. The lower B medians are observations, **not** a supported B advantage. No new cuBLAS comparison was made. MI-Research-7B's cuBLAS result remains historical context.

## Interpretation, evidence, and next action

The exact 55-addition circuit is mathematically smaller but has a deeper DAG. At this fixed output-owned mapping, it leads to more duplicated gates and dots, more static emitted arithmetic and MMA text, and higher compiler-reported spills than the original 58-addition circuit. Numerical correctness and launch topology are matched. The bounded timing evidence cannot rank the two. This result supports **mapping-aware evaluation of addition circuits** as a research need; it does not show that reducing additions improves GPU performance or that the deeper path caused a slowdown.

The frozen plan, protocol, raw correctness distributions, source snapshots, TTIR/PTX and other compiler artifacts, graph traces, clock diagnostics, and SHA-256 manifests are under [`artifacts/mi-research-8/`](../artifacts/mi-research-8/). [`analysis.json`](../artifacts/mi-research-8/analysis.json) replays the exact and paired conclusions. The offline verifier needs no GPU:

```bash
PYTHONPATH=src .venv/bin/python scripts/generate_mi_research_8_kernel.py
PYTHONPATH=src .venv/bin/python scripts/analyze_mi_research_8.py --verify
PYTHONPATH=src .venv/bin/python -m pytest tests/mi_research_8_test.py
```

For a new GPU measurement, run `scripts/run_mi_research_8.py --output NEW_DIRECTORY --session 1` and then `--session 2` in a separate process. The runner refuses existing output paths. Historical timing results should not be appended to this bounded two-session bundle.

**One next research direction:** add a static per-output-slice cost descriptor to future circuit selection. Count recomputed gates, required product expressions, and branch liveness before compiling an addition circuit for this ownership mapping. The reversal between 55 versus 58 whole-circuit additions and 704 versus 568 executed gate instances is the reason for that recommendation.
