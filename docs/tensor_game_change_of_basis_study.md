# Exact change-of-basis decomposition generation

## Research question

This CPU-only experiment tests whether bounded integer changes of basis can turn the known conventional and Strassen 2×2 decompositions into exact implementations with broader preparation, reconstruction, support, and coefficient costs. These are equivalent representations of known algorithms. They are not independent tensor-rank discoveries and no performance claim is made.

## Derivation for the repository convention

Matmul Inspector flattens 2×2 block matrices in row-major order. Tensor coordinates are `A[i,k]`, `B[k,j]`, and `C[i,j]`. For unimodular integer matrices `P`, `Q`, and `R`, define

```text
A' = P A Q^-1
B' = Q B R^-1
C' = A' B' = P A B R^-1
AB = P^-1 C' R.
```

Write one seed rank term as coefficient matrices `U`, `V`, and `W`, so its input forms are the row-major dot products `<U,A'>` and `<V,B'>`, and `W` reconstructs its contribution to `C'`. Expanding indices gives

```text
<U, P A Q^-1> = <P^T U (Q^-1)^T, A>
<V, Q B R^-1> = <Q^T V (R^-1)^T, B>.
```

The output identity maps each reconstruction coefficient matrix to `P^-1 W R`. The implementation therefore applies

```text
U_new = P^T U (Q^-1)^T
V_new = Q^T V (R^-1)^T
W_new = P^-1 W R.
```

These formulas were derived from the actual index ordering rather than importing a column-major vectorization identity. Unit tests also evaluate a transformed decomposition on concrete integer matrices. Every exported result independently reconstructs `matrix_multiplication_tensor(2,2,2)` with exact integers.

## Bounded enumeration

There are 40 matrices in `{-1,0,1}^{2×2}` with determinant `+1` or `-1`. Their inverses are integral and are computed with the exact adjugate formula. The full triple product would contain 64,000 `(P,Q,R)` combinations, so the experiment uses seeded sampling without replacement.

The recorded configuration is:

```sh
PYTHONPATH=src python scripts/generate_tensor_game_change_of_basis.py \
  --output artifacts/tensor-game/change-of-basis-v1 \
  --entry-bound 1 --combination-limit 512 --mode sample --seed 20261016 \
  --maximum-coefficient 8 --maximum-total-support 96 \
  --maximum-pre-additions 64 --maximum-post-additions 32 \
  --representatives 12
```

The same 512 transformation triples are applied to conventional multiplication and Strassen, for 1,024 attempts. The generator also supports deterministic lexicographic enumeration. It records all three matrices, determinants, exact inverses, sample seed or enumeration index, parent family, limits, and complete coefficients.

## Exact validation and filtering

All 1,024 transformed decompositions passed exact tensor reconstruction before practical filtering. None exceeded the configured limits. Two representations were exact duplicates and collapsed by content hash, leaving 1,022 records. A stricter coefficient limit is covered in tests and rejects candidates by the recorded `maximum_coefficient` category.

Every accepted record also passes the existing `create_lowering()` contract. This validates rank, integer factors, exact reconstruction, and generation of static Triton lowering metadata without importing Triton, PyTorch, CUDA, or running a GPU.

## Dataset results

| Measure | Result |
|---|---:|
| Attempted seed/transformation pairs | 1,024 |
| Exact-valid before limits | 1,024 |
| Rejected by practical limits | 0 |
| Exact duplicates removed | 2 |
| Unique exact representations | 1,022 |
| Conventional / Strassen records | 511 / 511 |
| Limited canonical classes | 596 |
| Structural signatures | 598 |
| Primary arithmetic-cost points | 35 |
| Maximum coefficient magnitude | 2 |

Compared with Stage 1, all 598 structural signatures are new and 33 of the 35 primary `(rank, pre-additions, post-additions, coefficient-scalings)` points are new. The two old points are the conventional `(8,0,4,0)` and Strassen `(7,10,8,0)` costs. The apparent 596 canonical classes do not represent new algorithms: the existing canonicalizer only handles term order and per-term sign redistribution, not `GL(2,Z)^3` equivalence.

Preparation additions span 0 through 26. Reconstruction additions take values 4, 8, 10, 12, 14, and 16. Total coefficient support spans 24 through 58. Nonunit scaling counts are 0, 4, 6, 8, 10, 14, and 18. Of the retained candidates, 625 have maximum magnitude 1 and 397 have maximum magnitude 2.

The full distributions are in `manifest.json`. The largest groups are 270 candidates at 10 preparation additions, 442 at 14 reconstruction additions, 333 at support 54, and 625 requiring no nonunit coefficient scaling.

## Transformation complexity

As a simple transformation-complexity measure, count nonzero entries across `P`, `Q`, and `R`:

| Basis support | Candidates | Structural signatures | Cost points | Preparation range |
|---:|---:|---:|---:|---:|
| 6 | 6 | 4 | 2 | 0–10 |
| 7 | 96 | 67 | 6 | 2–18 |
| 8 | 408 | 242 | 14 | 4–22 |
| 9 | 512 | 285 | 18 | 10–26 |

Higher basis support broadens the observed static cost range in this sample. This is descriptive evidence, not a causal performance result.

## Recommended GPU-calibration subset

`representatives.json` contains 12 deterministic choices. It covers both parent families, low and high preparation/reconstruction work, support from 24 through 54, and coefficients of magnitude one and two. Useful anchors include:

| Candidate prefix | Family | Rank | Pre | Post | Scalings | Support | Max magnitude |
|---|---|---:|---:|---:|---:|---:|---:|
| `30b13f831111` | Conventional | 8 | 0 | 4 | 0 | 24 | 1 |
| `0f0a7a60a1ed` | Strassen | 7 | 10 | 8 | 0 | 36 | 1 |
| `05402bd09ac1` | Conventional | 8 | 4 | 14 | 0 | 42 | 1 |
| `0b88a39c7ca3` | Strassen | 7 | 14 | 12 | 4 | 44 | 2 |
| `10a894434fb7` | Strassen | 7 | 18 | 8 | 4 | 44 | 2 |

The full 12 should be used for initial compiler validation. GPU timing should then retain the two original anchors and candidates that isolate one cost change where possible. Static descriptors alone should not be used to predict a winner.

## Artifacts and reproducibility

The new dataset is under `artifacts/tensor-game/change-of-basis-v1/`. It does not overwrite Stage 1 or the diversity-search archive.

- `candidates.jsonl`: 1,022 complete records, SHA-256 `87efb33ee34a857d524a1429a716bc9d1039560805770aa5089f33340c00c52d`.
- `manifest.json`: configuration, environment, distributions, counts, and representative IDs, SHA-256 `e6e657d56eac6046c229a93f8a81eeadbd6c05e2d0e99aa40ac306578322660a`.
- `representatives.json`: 12 selected complete records, SHA-256 `d63ac247d2bb5e8c1e482506bceb6a9cf59752d512be47aaf26fde6c2f2b336b`.

Serialization and record ordering are deterministic. The manifest records commit `51653b0`, which is the input revision used to run the experiment; the implementation and evidence are committed together afterward.

## Limitations and next experiment

The sample covers only entry bound one and 512 of 64,000 possible triples. The canonicalizer cannot prove the known basis equivalence from coefficients alone, so provenance is authoritative here. The static scalar schedule counts unique input combinations but does not model common-subexpression sharing across different linear forms, device memory traffic, register pressure, compiler simplification, or latency. Coefficients of magnitude two may lower to addition, multiplication, or another compiler-selected sequence.

The next experiment should run CPU-side lowering validation over all 12 representatives, then perform a paired GPU study only after explicit approval. Use identical tensors and timing settings, preserve compiler artifacts, and compare complete-operation latency. Stratify results by rank, preparation work, reconstruction work, support, and coefficient scaling instead of treating change-of-basis variants as independent algorithms.
