# Tensor Game candidate decomposition dataset study

## Research objective

This is Stage 1 of hardware-cost calibration. It builds a deterministic,
CPU-only collection of exact 2x2 bilinear matrix multiplication decompositions.
It does not benchmark GPUs, change search, or treat transformed algorithms as
new discoveries.

## Existing decomposition inventory

The repository contains two reusable complete decompositions:

| Seed | Rank | Source | Validation |
|---|---:|---|---|
| Conventional | 8 | `standard_2x2_decomposition()` | Exact integer reconstruction |
| Strassen | 7 | `strassen_2x2_decomposition()` | Exact integer reconstruction |

Search traces, residual targets, and synthetic training datasets were inspected
but are not additional complete verified algorithms. No Winograd or independently
discovered alternative is represented in the current engine. The seed inventory
therefore has two unique valid decompositions before transformation.

## Exact transformations

The generator supports three deterministic transformation types:

- Rank-term permutation reorders complete `(u, v, w)` terms.
- Exact sign scaling multiplies one term's three factors by signs whose product
  is +1. This is the complete nontrivial scaling available while retaining the
  current integer, unit-magnitude coefficient domain. No floating reciprocals
  are introduced.
- Index symmetry independently relabels the two output-row, output-column, and
  contracted indices. The A, B, and C coefficient axes are permuted together.
  All eight binary relabelings are considered.

Every transformed result is validated against the original 4x4x4 multiplication
tensor. Transformations are generic and do not encode Strassen formulas.

## Validation and canonicalization

Every action must have shape `(4, 4, 4)` and exact Python integer coefficients.
The generator sums rank-one outer products using integer arithmetic and requires
exact equality with `matrix_multiplication_tensor(2, 2, 2)`. Malformed archive
records and failed transformations are counted by reason and never exported.
Each valid coefficient representation receives a SHA-256 content ID.

Canonicalization normalizes the four supported per-term sign-equivalent forms,
then sorts rank terms. It therefore proves equivalence under term ordering and
exact sign scaling. It does not canonicalize index symmetries, arbitrary basis
changes, or every mathematical tensor-decomposition equivalence. Seed family,
canonical class, exact representation, and full transformation provenance remain
separate fields. Execution variants are retained because equivalent coefficient
orders can still compile or execute differently.

## Reference dataset

The bounded reference dataset was generated with:

```bash
PYTHONPATH=src /home/daniel/IsaacLab/.venv/bin/python \
  scripts/generate_tensor_game_candidate_dataset.py \
  --output artifacts/tensor-game/candidate-dataset-v1 \
  --seed 20261016 --maximum-candidates 128 \
  --transformations permutation sign_scaling index_symmetry
```

The generator divides its bounded attempt budget between seed families. Output
ordering, compact JSONL serialization, IDs, and file hashes are deterministic.
It refuses to overwrite an existing directory. The manifest records the base
Git commit, environment, configuration, seed inventory, counts, and limitations.
An optional archive may add seeds only after the same exact validation.

## Dataset counts and quality controls

| Measure | Count |
|---|---:|
| Generated attempts | 128 |
| Valid attempts | 128 |
| Invalid attempts | 0 |
| Unique exact representations | 128 |
| Canonical classes | 4 |
| Seed families | 2 |
| Structural signatures | 11 |

There are 64 conventional and 64 Strassen records. Exact-representation
deduplication removed none in this bounded sequence. Canonical class sizes are
16, 24, 24, and 64. One class contains all conventional variants. Three classes
cover the retained Strassen index-symmetry variants under the deliberately
limited canonicalizer. These labels must not be interpreted as four independent
algorithm discoveries.

The structural signature records rank, input/output support histograms,
preparation and reconstruction additions, coefficient scaling count, maximum
magnitude, and negative coefficient counts. Its 11 values comprise four
conventional and seven Strassen sign distributions. This distinguishes compiler-
relevant sign layouts without pretending term order changes algebraic cost.

## Structural diversity

The static cost distributions are narrow:

| Family | Rank | Pre-additions | Post-additions | U/V/W support pattern |
|---|---:|---:|---:|---|
| Conventional | 8 | 0 | 4 | eight support-1 terms on every factor |
| Strassen | 7 | 10 | 8 | two support-1 and five support-2 terms per factor |

All coefficients have magnitude one and no nonunit coefficient scalings. Thus
the dataset has meaningful representation and sign-layout diversity, plus the
known rank/addition contrast between two families. It does not have broad
independent cost variation suitable for fitting a hardware-cost predictor.
Generating many more permutations would increase row count without improving
that limitation.

## Representative comparison

A conventional candidate uses one source block per input factor, eight products,
and eight single-output terms. A Strassen candidate uses seven products but has
five two-block combinations on each factor and five two-output reconstruction
terms. Sign-scaled and permuted versions preserve exact outer products and the
same core cost counts. Index relabeling changes coefficient placement and may
change an execution mapping even when the aggregate descriptors match.

Every record includes `lowering_metadata` produced through `create_lowering()`.
This proves compatibility with the existing generic Triton lowering contract
without importing PyTorch, Triton, or CUDA.

## Limitations and next experiment

Only conventional and Strassen are independent seed families. The 11 structural
signatures mostly reflect sign placement, while rank and addition counts occupy
only two points. General scaling by rational values is not introduced because
the current `RankOneAction` and exact lowering contract are integer-based.
Canonicalization is intentionally incomplete, and archive schema support is
limited to explicit `actions` or `solution` lists.

Additional independently validated decompositions are needed before GPU
benchmarking can support a calibrated multifeature cost model. The next step
should import a small reviewed archive of exact integer or explicitly rational
2x2 solutions, validate it independently, and confirm that it adds new rank,
support, addition, or magnitude combinations. If it does not, Stage 2 should be
framed as a controlled two-family execution study rather than model training.
