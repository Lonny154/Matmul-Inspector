# Tensor Game hardware-cost study

## Research question

Can Tensor Game expose useful algebraic implementation-cost differences among
residual-ranked candidates without changing `bounded_128` search behavior?

This first milestone adds static descriptors and a two-objective Pareto view. It
does not generate kernels or predict device latency.

## Baseline and integration

`bounded_128` streams all 6,400 nonzero ternary `(u, v)` pairs. It keeps the best
128 under the existing continuous-relaxation bound. It then quantizes `w`,
canonicalizes candidates, and exact-scores the resulting integer actions.

Optional cost analysis runs after that exact ranking. It analyzes only the unique
legal candidates derived from the retained 128 pairs. The returned candidates,
ordering, scores, default provider, and residual updates are unchanged. Call:

```python
rows, audit = bounded_residual_candidates_memory_efficient(
    residual, 128, 12, analyze_costs=True
)
analysis = audit["cost_analysis"]
```

`analysis["candidates"]` is machine-readable. Each row includes its action ID,
exact residual score, residual reduction, descriptors, and Pareto membership.

## Descriptor definitions

For coefficient vector `x`, support is the number of nonzero coefficients. An
unshared scalar linear combination uses `max(support - 1, 0)` additions. Sign,
unit/nonunit coefficients, L1 magnitude, maximum magnitude, and a magnitude
histogram are reported separately. A nonunit coefficient is counted as a
coefficient scaling. This does not assign the scaling a hardware latency.

For one `(u, v, w)` action, preparation additions are the additions for `u` and
`v`. Output support is reported separately. The temporary estimate comes from an
explicit sequential schedule: form two input scalars, multiply them, then update
the nonzero `w` outputs. Three scalar temporaries are live at the multiplication
point. This is not register allocation or a memory-traffic estimate.

For a complete decomposition, each distinct input combination is formed once.
Products are evaluated sequentially. Post-additions are counted independently
for every output coordinate as `max(contributing products - 1, 0)`. The output
accumulators plus three action temporaries define the reported peak scalar
storage estimate. No sharing among partial sums is assumed.

## Pareto method

The current objectives are exact residual reduction, maximized, and preparation
additions, minimized. Candidate A dominates B only when A is no worse on both and
strictly better on at least one. Equal objective points all remain on the
frontier, and input order is preserved. Zero preparation cost is valid. Pareto
selection is analysis only and never replaces the search policy.

## Controlled experiment

The benchmark used the exact 2×2 multiplication tensor and six residuals after
successive prefixes of the known Strassen decomposition. Both modes used the
same seven states, pair budget 128, return limit 12, deterministic candidate
implementation, and `perf_counter_ns`. Twenty repetitions alternated mode order.
Candidate generation consumes no random numbers, so the recorded seed is null.

Run it with:

```bash
PYTHONPATH=src python scripts/benchmark_tensor_game_hardware_cost.py \
  --repetitions 20 \
  --output artifacts/tensor-game/hardware-cost-v1/benchmark_final.json
```

The output path must not already exist. The raw JSON records configuration, host
metadata, every timing, candidate counts, exact scores, action IDs, and complete
Pareto rows.

## Correctness results

The focused CPU suite passed 24 tests. It covers empty and singleton supports,
negative and nonunit coefficients, Pareto dominance and ties, stable output,
unchanged default `bounded_128` rows, exact decomposition reconstruction, and
the existing Tensor Game/provider tests.

Reference decomposition counts match the implemented convention:

| Algorithm | Products | Pre-additions | Post-additions | Total additions |
|---|---:|---:|---:|---:|
| Conventional 2×2 | 8 | 0 | 4 | 4 |
| Strassen 2×2 | 7 | 10 | 8 | 18 |

The conventional count accumulates two products into each of four outputs.
Strassen matches the common 10 input plus 8 output addition schedule. No separate
Winograd decomposition exists in the engine, so this milestone does not invent or
silently substitute one.

## Benchmark results

These are observed wall times on the development CPU for all seven states per
timed sample:

| Mode | Mean | Median | Min–max |
|---|---:|---:|---:|
| Baseline | 207.81 ms | 212.13 ms | 194.10–236.07 ms |
| Cost analysis | 210.66 ms | 212.52 ms | 197.48–238.97 ms |

The observed mean overhead was 2.85 ms (1.0137× ratio), smaller than the roughly
10 ms within-mode standard deviations. This sample does not resolve a reliable
nonzero overhead, though its point estimate is about 1.4%.
The modes returned identical baseline candidate sequences. Each state enumerated
6,400 pairs and fully evaluated 128. Unique analyzed candidates ranged from 5 to
26. Pareto frontier sizes ranged from 2 to 10, except the initial state where all
eight tied candidates were nondominated.

## Candidate tradeoffs

| Residual state | Reduction | Preparation additions | Input supports `(u,v)` | Interpretation |
|---|---:|---:|---:|---|
| Initial | 1 | 0 | (1, 1) | All eight legal candidates tie |
| After Strassen step 1 | 4 | 2 | (2, 2) | More reduction, more preparation |
| After Strassen step 1 | 2 | 1 | (2, 1) or (1, 2) | Lower-cost frontier point |
| After Strassen step 4 | 2 | 0 | (1, 1) | Zero-preparation frontier point |
| After Strassen step 4 | 4 | 2 | (2, 2) | Higher-reduction frontier point |

These are algebraic candidate observations. They are not measured kernel costs.

## Limitations and next experiment

The descriptors omit instruction selection, common-subexpression scheduling
beyond identical whole input combinations, vectorization, memory placement,
register pressure, spills, occupancy, and target-specific coefficient handling.
Single-action `w` support is not confused with complete output reconstruction.
The benchmark uses seven related 2×2 states and does not measure solve rates or
hardware execution. Since search behavior is unchanged, solve-rate comparisons
are not informative for this instrumentation-only mode.

The next concrete experiment should lower a small frozen set of Pareto-diverse
actions to a minimal Triton prototype with an explicit common-subexpression
schedule. Capture generated TTIR/TTGIR/PTX and benchmark identical matrix shapes.
Compare measured latency and compiler resource evidence against each descriptor
separately before introducing any hardware-weighted search score.
