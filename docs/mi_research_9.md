# MI-Research-9 — Output-slice-aware algebraic cost analysis

## Decision

**A — Slice-aware cost analysis validated.** A CPU-only replay of the two exact Perminov DAGs and the frozen output ownership reproduces the MI-Research-8 source totals: 55 additions give **704 gate and 256 dot instances** over 36 CTAs; 58 additions give **568 gates and 232 dots**. The model also records every selected dependency, per-role lifetime, and duplicated expression. It is a structural screen. It does not predict GPU latency.

## Frozen inputs and method

The checkpoint is `c2b8de73981662193284f231c8dd67b5fa1cbdae`. The analyzer rebuilds both circuits from the retained [Perminov source and certificate](../artifacts/mi-research-6c/upstream/), checks them against the retained [6C exact records](../artifacts/mi-research-6c/analysis.json), and reruns the 6C and 8 offline verifiers. Both factor maps match entry for entry. Each rank-23 circuit reconstructs all 729 tensor coefficients: 27 ones, 702 zeros, and zero mismatches. Input hashes are in [the new analysis record](../artifacts/mi-research-9/analysis.json).

The existing `Circuit` DAG is the only algebraic representation. For each owned output, the analyzer walks dependencies backward, then filters the retained 6C liveness order to those nodes. It loads all needed A/B tiles first, as the frozen sources do. The 55 four-CTA rendering and both 36-CTA renderings must reproduce their retained kernel sources byte for byte. The resulting cost counts come from the analyzer's own graph walk, not from the source generator's saved totals. No GPU code, timing, or compiler invocation is part of this milestone.

The four-CTA mapping has grid `(2,2,1)`. One CTA owns the same local 16×16 tile in all nine 32×32 output blocks, runs the whole circuit once, and stores nine tiles. The 36-CTA mapping has grid `(2,2,9)`. One CTA owns one local tile of one output block and runs only that output's dependency slice. Each tile position occurs four times. Every one of the 9,216 output elements has one owner in either mapping. The retained four-CTA **55** source checks this ownership. Four-CTA **58** is a structural projection onto the same verified ownership and frozen circuit order; no four-CTA 58 GPU kernel or compiler result exists.

## Source work and duplication

| Circuit and mapping | CTAs | Whole-circuit gates | Dots/grid | Gates/grid | Extra dots from output slicing | Extra gates from output slicing |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 55, four CTA | 4 | 55 | 92 | 220 | 0 | 0 |
| 58, four CTA projection | 4 | 58 | 92 | 232 | 0 | 0 |
| 55, 36 CTA | 36 | 55 | 256 | 704 | 164 | 484 |
| 58, 36 CTA | 36 | 58 | 232 | 568 | 140 | 336 |

At one local tile position, the four-CTA owner computes each of 23 product labels once. Nine separate output owners require 64 product instances for 55 or 58 for 58. The 55 circuit also needs 176 gate instances across those nine branches; the 58 circuit needs 142. Multiplying each by four tile positions yields the grid totals. “Extra” means occurrences beyond one occurrence of each of the 23 products and each whole-circuit gate **at the same local tile position**. Product tiles at different positions have different numerical data; those are distinct work, not common subexpressions that could be shared without changing the mapping.

The per-output product counts for 55 are `[8,6,6,12,3,10,9,3,7]`; for 58 they are `[8,6,6,11,3,10,5,3,6]`. The corresponding gate counts are `[20,16,20,33,2,31,30,2,22]` and `[21,19,14,29,2,27,15,2,13]`. The 58 counts match the nonzero product support of the final output factor rows exactly. The 55 DAG reaches six **additional product dependencies per tile position with zero net output coefficient**: one in output 3, four in output 6, and one in output 8. In output 6 they are `M05`, `M10`, `M11`, and `M18`. Their contributions cancel algebraically inside reconstruction. Across four tile positions, those six source dependencies account for **all 24 extra dot instances** in 55 versus 58. They are not extra mathematical rank terms.

The 34 extra gates per tile position in the 55 sliced source break down as six U, 18 V, and ten W gate instances. Thus the gate difference comes from both repeated operand preparation and reconstruction, rather than the whole-circuit addition count alone.

The [dependency and duplication records](../artifacts/mi-research-9/analysis.json) identify each repeated expression and its output owners. In the 55 circuit, `M10`, `Ug00`, `Ug01`, `Ug02`, `Vg01`, `Vg04`, `Vg05`, `Vg06`, `Vg08`, and `Vg09` each occur in seven output roles. Each contributes six extra executions per local tile position, or 24 across the grid. In the 58 circuit, only `Ug00` reaches seven roles. `M07`, `M10`, `Vg00`, `Vg01`, and `Vg03` reach six. Local sharing *within* a CTA is retained: the role records identify nodes with more than one consumer, and each selected node is emitted once per CTA. Cross-CTA sharing would require a different mapping or scratch policy and is outside this study.

## Modeled lifetime and dependency cost

For each selected input, U/V gate, product, and W gate, the evidence records its creation step, consumer IDs, last-use step, category, and logical FP32 elements. The output store is the last use of its source; it does not create another array. A value is counted live after step `i` when `created_at <= i < last_used_at`. Inputs and input combinations each represent 16×32 logical FP32 elements; products and reconstruction values each represent 16×16. Estimated bytes are logical elements times four. The peak category counts can occur at different steps and must not be added together.

| Circuit and mapping | Max live intermediates/CTA | Max live values with inputs/CTA | Max logical live bytes/CTA | Max dependency depth |
| --- | ---: | ---: | ---: | ---: |
| 55, four CTA | 17 | 31 | 59,392 | 15 |
| 58, four CTA projection | 18 | 33 | 63,488 | 12 |
| 55, 36 CTA | 11 | 19 | 38,912 | 15 |
| 58, 36 CTA | 11 | 18 | 36,864 | 12 |

The 55 circuit has the longer dependency path in both mappings. That does **not** imply more modeled live storage: in the full four-CTA order it has one fewer peak intermediate and 4,096 fewer peak logical bytes. Under 36-CTA slicing, it has the same peak intermediate count but 2,048 more peak logical bytes. The mapping changes which circuit has the smaller storage estimate. These are abstract tensor lifetimes under a fixed source order, not register allocation, occupancy, or physical local-memory traffic. Compiler layout replication, index arrays, negative operands, dot temporaries, and instruction scheduling are excluded.

With minimized source dots, gates, peak intermediates, modeled bytes, and dependency depth as separate coordinates, neither circuit dominates in the four-CTA projection: 55 has fewer gates and less modeled storage, while 58 has a shorter path. At 36 CTAs, 58 is no worse on any of those five coordinates and strictly better on four. This is a structural Pareto statement for these two frozen circuits and mappings, not a latency winner or a scalar performance score. CTA count differs between mappings, so their vectors should not be ranked as if they represented one identical execution configuration.

## Compiler and measurement boundary

The historical [MI-Research-8 compiler record](../docs/mi_research_8.md) reports 64 versus 58 static `tt.dot` occurrences in TTIR. Those match the sum of one copy of each of the nine source branches; four tile-position CTAs execute each branch at runtime. The PTX text contains 1,536 versus 1,392 `mma.sync` occurrences. The same record reports TTIR `arith.addf`/`arith.subf` counts of 76/115 versus 63/68 and compiler `n_spills` of 166 versus 100, with 255 registers per thread and 1,024 B shared memory for both. Source binary gates do not map one to one to TTIR arithmetic: signed operands, simplification, and lowering change the representation. We cannot infer a number of eliminated algebraic gates from those text counts. PTX is not SASS, and static text counts are not dynamic instruction counts.

The lower static arithmetic, MMA text, and reported spill count for 58 align in direction with its lower 36-CTA source work and modeled bytes. They do not prove that duplication or liveness caused the spill difference. MI-Research-8's two paired CUDA Graph sessions were both `INCONCLUSIVE`, despite lower 58 median times. There is no supported GPU latency ranking from this analysis or the historical timing evidence.

## Reproduction and scope

From the repository root:

```bash
PYTHONPATH=src:. .venv/bin/python scripts/analyze_mi_research_9.py --verify
PYTHONPATH=src:. .venv/bin/python -m pytest -q tests/mi_research_9_test.py
```

The first command verifies the [SHA-256 manifest](../artifacts/mi-research-9/hash_manifest.json), replays exact sources and all structural records, and checks the retained source renderings. The tests cover dependency closure, exact ownership, sharing, duplication totals, topological order, lifetime and last use, stable records, invalid mappings, and artifact tampering. The evidence contains complete per-role dependency graphs and stepwise live counts. Prior studies are unchanged.

The analysis would have flagged the whole-circuit 55 gate advantage as a **36-CTA source-work disadvantage before compilation**. It could not have predicted the GPU timing result, compiler allocation, or whether a different mapping would win. The bounded sample contains only two circuits and two mappings; it does not calibrate a general cost model.

**One next step:** use this verified slice-aware cost vector as a pre-compilation screen in a future bounded circuit comparison, retaining its Pareto tradeoffs and validating any proposed ranking against independent GPU measurements.
