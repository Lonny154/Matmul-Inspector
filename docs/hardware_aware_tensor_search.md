# HA-Tensor-1: exact rank-8 hardware-aware candidate study

## Motivation and evidence boundary

Rank is the number of multiplicative terms, not a complete hardware objective.
Previous rank-7 Tensor Game studies did not discover an independent rank-7
decomposition. Their measured RTX 4060 Ti results also showed that fewer terms
do not guarantee lower latency. HA-Tensor-1 asks a narrower CPU-only question:
does the conventional exact rank-8 algorithm have enough structurally diverse
basis-equivalent representations to justify a later GPU study?

Every generated candidate here is a representation of the *known conventional*
algorithm under an exact `GL(2,Z)^3` change of basis. These are not new
multiplication algorithms. Rank 9 is not studied.

## Generation and exactness

The conventional eight-term decomposition is always admitted first as the
anchor. The generator then samples unique triples of bounded 2×2 unimodular
integer matrices with the repository's seeded change-of-basis machinery.
`--selection-mode enumerate` uses lexicographic triples instead. Matrices,
determinants, inverses, sample indices, seed, and repeated generation paths are
retained. `--max-attempts` includes the anchor and `--max-accepted` limits unique
canonical candidates. Entry bound and coefficient, support, preparation, and
reconstruction limits are explicit. The anchor is always retained even if a
configured limit would otherwise exclude it. The output directory must not exist.

For every attempt, the generated eight terms are independently summed with
exact integers and compared against every entry of the canonical 2×2
multiplication tensor. A failed reconstruction is rejected. The existing
canonicalizer normalizes rank-term order and per-term sign redistribution.
Its SHA-256 canonical ID deduplicates records. It does not identify arbitrary
change-of-basis equivalence from coefficients alone. That equivalence is known
from the generation provenance. The first path determines the retained
canonical record; all equivalent generation paths are also recorded.

## Static descriptors and Pareto method

The existing unshared scalar-schedule model supplies rank, preparation
additions, output reconstruction additions, coefficient scalings, peak
temporary scalars, and shared input combinations. The candidate record also
retains existing lowering metadata. Transparent supplementary descriptors are
the total count of nonzero coefficients across all `u/v/w` vectors (also called
total support), coefficient L1 sum, and maximum absolute coefficient.
No weighted latency formula is introduced.

The Pareto comparison minimizes exactly four dimensions:
`pre_additions`, `post_additions`, `coefficient_scalings`, and `total_support`.
A candidate dominates another if it is no worse in all four and strictly
better in at least one. Equal vectors remain nondominated. Every record stores
its full descriptor vector, dominated flag, and dominator count. Rank is fixed
at eight and is not a frontier dimension. A known rank-7 Strassen reference
has separate descriptors and is excluded from every rank-8 count and frontier.

The structural signature uses the existing deterministic rank/addition/scaling
fields, sorted per-factor support histograms, negative-coefficient counts,
and maximum coefficient magnitude. It is independent of term order.
Signature equality does not imply hardware equivalence. Signature groups retain
member IDs and within-group descriptor ranges.

Representatives come only from the nondominated frontier. Selection starts
with the literal conventional anchor, then descriptor extrema, then uncovered
structural signatures, then further frontier candidates in canonical-hash
order. Selection is deterministic and contains no duplicate canonical IDs.
The target is at most 16 by default, not a promise that 16 nondominated
candidates exist.

## Commands and retained evidence

Small CPU-only smoke run:

```sh
PYTHONPATH=src python scripts/run_hardware_aware_tensor_rank8.py \
  --mode smoke --output /tmp/ha-tensor-rank8-smoke
```

Bounded full study:

```sh
PYTHONPATH=src python scripts/run_hardware_aware_tensor_rank8.py \
  --mode full --seed 20261018 --max-attempts 512 --max-accepted 512 \
  --entry-bound 1 --representatives 16 \
  --output artifacts/tensor-game/hardware-aware-rank8-v1
```

Each run writes configuration, source-revision provenance, baseline, all unique
candidate rows, signature groups, frontier, representatives, aggregate summary,
run notes, optional rank-7 reference, and a SHA-256 file manifest. Candidate
records include complete `u/v/w` coefficient arrays. The retained result and
its interpretation are recorded below after the implementation commit.

## Decision gate and limitations

Fewer than 128 attempted forms means `INCONCLUSIVE` under this bounded gate.
Otherwise `PROCEED_GPU_STUDY` requires at least ten nondominated candidates,
three nondominated structural signatures, and two distinct nondominated cost
vectors. All other adequate runs are `LOW_DIVERSITY`. These thresholds are a
study-design choice, not a hardware law.

Static descriptors are not GPU measurements. Pareto dominance does not prove
lower latency. The naive scalar schedule omits compiler simplification,
common subexpressions across distinct forms, register allocation, traffic,
occupancy, and launch overhead. Exact reconstruction is mandatory. This study
does not claim any performance improvement.

If the gate supports it, HA-Tensor-2 would benchmark frozen representatives on
an RTX 4060 Ti using fused Triton lowering. It would validate numerical
correctness, compare conventional Triton and cuBLAS, use interleaved timing,
retain relevant PTX/SASS or compiler evidence, and compare measured latency
against static descriptors. HA-Tensor-2 is not implemented here.

## Retained results and decision

The retained full run used implementation commit `cf53252`, seed `20261018`,
512 attempts including the anchor, entry bound one, and an accepted limit of
512. All 512 attempts reconstructed the exact tensor. No practical-limit
rejection occurred. Canonicalization removed 389 duplicate generation paths,
leaving 123 unique rank-8 decompositions and 60 structural signatures. This
is structural variation within one known algorithm, not 123 discoveries.

| Static descriptor | Full-set minimum | Median | Maximum | Frontier |
|---|---:|---:|---:|---:|
| Preparation additions | 0 | 10 | 10 | 0 |
| Reconstruction additions | 4 | 14 | 14 | 4 |
| Coefficient scalings | 0 | 0 | 0 | 0 |
| Total coefficient support | 24 | 54 | 54 | 24 |
| Coefficient L1 | 24 | 54 | 54 | 24 |
| Maximum absolute coefficient | 1 | 1 | 1 | 1 |

The literal conventional anchor has canonical hash
`307440126be71fcb9ea285397dc42ef84f9a0615da9e636f95aa12dbd70336b3`.
It uses eight products, zero preparation additions, four reconstruction
additions, zero scalings, and 24 nonzero unit coefficients. Under the four
declared cost objectives, it dominates every other sampled canonical
representation. The frontier has one candidate, one signature, and one cost
point. The representative file therefore contains just the anchor. The target
of 10–30 representatives cannot be met without adding dominated points or
changing the objective; neither is done silently.

The decision is **LOW_DIVERSITY** for this basis-equivalent, bounded rank-8
population under the present static scalar descriptors. Sixty signatures show
algebraic variation, but not a cost-diverse Pareto frontier. This does not rule
out hardware-relevant differences invisible to these descriptors or exact
rank-8 decompositions outside the sampled change-of-basis family. It does not
justify claiming a faster representation. A separate GPU study would need a
new, explicitly motivated representative policy or candidate family.

Raw records and SHA-256 hashes are in
`artifacts/tensor-game/hardware-aware-rank8-v1/`. A 32-attempt smoke run also
passed exact validation; the gate labels such a small run `INCONCLUSIVE`.
