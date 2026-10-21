# HA-Tensor-2A: direct exact rank-8/9 structural search

## Question and boundary

HA-Tensor-1's 123 exact change-of-basis rank-8 candidates had 60 structural
signatures, but every sampled candidate was statically dominated by the
conventional anchor. This study therefore solves for selected rank-one actions
*jointly* from finite pools, instead of only transforming that anchor. The
first hardware-oriented feature is operand-form reuse. No GPU is run here.
Exact rank-8 or rank-9 reconstruction does not imply faster GPU execution.

## Domain and exact finite-pool formulation

The existing ternary action domain has 80 nonzero vectors per factor,
512,000 raw factor triples, and 128,000 canonical rank-one tensors after
four-way sign normalization. The full domain can be streamed lazily, but no
unbounded solve is launched. The existing meet-in-the-middle convention
excludes repeated canonical actions. Each pool action therefore has one
Boolean selection variable `x_i`.

For fixed rank 8 or 9, CP-SAT enforces `sum(x_i) = rank` and one exact integer
equality for each of the 64 entries of the 2×2 multiplication tensor:
`sum(x_i * outer_product(action_i)[a,b,c]) = target[a,b,c]`. These are joint
constraints. No rank optimization or weighted latency score is used. Every
reported model is independently reconstructed with the repository's exact
integer tensor code, canonicalized by term order and sign, and SHA-256 hashed.
Invalid independent reconstruction is an error, never a candidate.

The optional solver is OR-Tools CP-SAT (`pip install 'matmul-inspector[constraint-search]'`).
The retained environment used OR-Tools 9.15.6755. The base package still has
no mandatory solver, GPU, or CUDA dependency. The default is one worker and
seed `20261018`. A wall-clock timeout and CP-SAT memory parameter are explicit.
Single-worker fixed-seed model hashes are reproducible in the tested version;
multi-worker scheduling is not promised deterministic. `EXACT` means an
independently verified model was returned. `INFEASIBLE` means CP-SAT proved the
finite-pool model infeasible. `UNKNOWN` and `TIME_LIMIT` are not infeasibility.
`RESOURCE_LIMIT` rejects an oversized preflight estimate before model
construction. `ERROR` includes missing dependencies or invalid recovered
models. An objective is called proven optimal only when CP-SAT reports
`OPTIMAL`. Per-attempt solver status history and enumeration stop reason are
retained separately from the overall result.

For reuse objectives, Boolean `y_u[f]` and `y_v[g]` variables are linked by
`AddMaxEquality` to all selected actions using each canonical input factor.
Thus a form variable is one *if and only if* at least one selected action uses
it. Supported modes are `FEASIBILITY`, `MIN_DISTINCT_INPUT_FORMS`,
`MIN_DISTINCT_U`, `MIN_DISTINCT_V`, and `MAX_TOTAL_REUSE`. The last maximizes
`2*rank - sum(y_u) - sum(y_v)` directly. Optional minimum u/v reuse constraints
also use these exact variables. After a proven optimum, bounded enumeration
fixes that optimum and blocks each selected-action set. Canonical hashes
deduplicate retained decompositions.

## Pools and controls

- `CONTROL` contains the known conventional rank-8 terms or a synthetic rank-9
  split, plus bounded distractors. These are mechanics tests, not discovery.
  The synthetic rank-9 control replaces one rank-8 term by two distinct legal
  terms with the same `u,v` and output factors summing exactly to the original
  `w`. All nine actions are distinct canonical ternary actions and pass exact
  reconstruction.
- `FROZEN_EXISTING` loads a deterministic union of named non-oracle pools from
  the retained algebraic-pools artifact. The default identifiers are residual,
  structural, and algebraic. Oracle-named sources are rejected. Each action
  retains source identifiers and indices. The artifact SHA-256 and the final
  canonical pool hash are recorded. No Strassen term is injected.
- `STRUCTURED_DOMAIN` streams canonical ternary actions under transparent
  maximum support bounds for `u`, `v`, and `w`. The default supports are
  `(1,1,2)`, giving 512 canonical actions when the pool cap is 512. It never
  inserts an oracle decomposition. A smaller cap takes the deterministic
  support-ordered prefix. Coefficient magnitude is fixed to one by the current
  ternary domain.

Controls and discovery use separate CLI modes. `CONTROL` is refused in
discovery mode. The preflight counts pool actions, action variables, distinct
form auxiliaries, and a conservative constraint upper bound. Configured caps
cover pool size, total variables, auxiliaries, constraints, time, CP-SAT memory,
models attempted, and unique models accepted. The solver summary records exact
initial and final model sizes when constructed.

## Reuse and static descriptors

For canonical terms, `distinct_u_forms` and `distinct_v_forms` count unique
stored input factors. `distinct_input_forms` is their sum. Each reuse count is
`rank - distinct forms`; total reuse is the sum. Maximum multiplicity and
number of forms used at least twice are reported separately for `u` and `v`.
Sorted form/product groups, input-form supports, and output-factor supports
are retained. Existing preparation additions, reconstruction additions,
coefficient scalings, support, coefficient L1, and maximum coefficient are
reported without a latency formula. These are algebraic reuse descriptors,
not evidence of cache or register residency.

The conventional rank-8 anchor has four distinct `u` forms, four distinct
`v` forms, total reuse eight, and maximum multiplicity two on each side.
Known Strassen rank-7 has seven plus seven distinct forms and zero reuse by
this definition. The synthetic rank-9 split has four plus four distinct forms,
total reuse ten, and maximum multiplicity three. Exact mode-flattening ranks
of the target are four for `u` and four for `v`. Therefore no exact
decomposition at either tested rank can reduce distinct input forms below the
conventional anchor's eight. Rank 9's larger raw reuse count can arise solely
from its extra product; it is not an improvement in distinct forms.

| Exact reference | Rank | Distinct u/v | Total reuse | Max u/v multiplicity | Pre / post additions | Scalings | Support |
|---|---:|---:|---:|---:|---:|---:|---:|
| Conventional | 8 | 4 / 4 | 8 | 2 / 2 | 0 / 4 | 0 | 24 |
| Known Strassen | 7 | 7 / 7 | 0 | 1 / 1 | 10 / 8 | 0 | 36 |
| Synthetic split | 9 | 4 / 4 | 10 | 3 / 3 | 0 / 6 | 0 | 28 |

## Reproduction

Default control smoke:

```sh
PYTHONPATH=src python scripts/run_hardware_aware_rank89_constraint_search.py \
  --rank 8 --study-mode control --pool-mode CONTROL \
  --output /tmp/ha-tensor-2a-control8
```

Bounded discovery example:

```sh
PYTHONPATH=src python scripts/run_hardware_aware_rank89_constraint_search.py \
  --rank 9 --study-mode discovery --pool-mode STRUCTURED_DOMAIN \
  --max-pool-size 512 --objective MIN_DISTINCT_INPUT_FORMS \
  --max-models 5 --max-accepted 5 --timeout-seconds 5 \
  --output artifacts/tensor-game/hardware-aware-rank89-v1/structured-rank9
```

Every run has configuration, provenance, a complete pool manifest, model-size
and solver-status summary, solution rows, conventional and synthetic controls,
baseline comparison, aggregate summary, notes, and SHA-256 file hashes.
The conventional anchor stays in every report, even if absent from a discovery
pool. Rank-8 and rank-9 runs have separate directories and counts.

## Catalog comparison, limitations, and next step

Canonical model hashes are compared with the conventional anchor, retained
HA-Tensor-1 candidates, and the explicit synthetic rank-9 control. Rank-9
models are also checked for an exact two-output-term merge into a cataloged
rank-8 decomposition. `NEW_TO_PROJECT_CATALOG` means only that the exact
canonical hash was not previously retained. It does not prove mathematical
novelty or non-equivalence under general change of basis. Strassen is a known
rank-7 contextual reference, not a rank-8/9 catalog match.

Solver failure is finite-pool evidence only. It does not exclude other pools,
larger coefficients, repeated-action policies, or the full domain. Hardware
descriptors omit instruction selection, register allocation, memory traffic,
and launch overhead. Rank 9 necessarily performs one more multiplicative term
than rank 8. No GPU measurements are made. A later HA-Tensor-2B would require
independently selected exact candidates, fused lowering, numerical validation,
interleaved RTX 4060 Ti timing against conventional Triton and cuBLAS, and
compiler/profile evidence. That work is not part of this milestone.

## Retained results and decision

The six retained CPU-only runs used implementation commit `2916313`, OR-Tools
9.15.6755, one worker, seed `20261018`, and a five-second limit per solve.
All were far inside that limit. Their exact solution counts are separate by
rank and pool; `INFEASIBLE` after enumerating a solution means no *additional*
solution in that finite pool.

| Run | Pool | Objective | Overall result | Exact unique models | Solve time |
|---|---:|---|---|---:|---:|
| Rank-8 control | 15 | Feasibility | Exact conventional | 1 | 0.0034 s |
| Rank-9 control | 16 | Feasibility | Exact synthetic split | 1 | 0.0036 s |
| Structured rank 8 | 512 | Feasibility | Exact anchor, then finite-pool exhaustion | 1 | 0.046 s |
| Structured rank 9 | 512 | Minimum distinct forms | Proven optimum 8, bounded five models | 5 | 0.154 s |
| Frozen rank 8 | 96 | Minimum distinct forms | Proven optimum 8, then finite-pool exhaustion | 3 | 0.014 s |
| Frozen rank 9 | 96 | Feasibility | Proven finite-pool infeasible | 0 | 0.0022 s |

Across the retained runs there are three unique exact rank-8 canonical
decompositions and six unique exact rank-9 decompositions. The control pools
recover their intended models. Removing a required action from the isolated
rank-8 control pool is proven infeasible in the focused test.

The two non-anchor frozen rank-8 hashes are `5e4118dd1125…` and
`1b8c69355bf5…`. Both are `NEW_TO_PROJECT_CATALOG`: neither is among the 123
retained HA-Tensor-1 hashes. They have four distinct `u` and four distinct `v`
forms, the same minimum eight distinct input forms as conventional. Both trade
one preparation addition, six reconstruction additions, and support 28 for
the anchor's zero, four, and 24. They are structurally distinct canonical
records, not demonstrated to be outside the full change-of-basis orbit or to
be mathematically new. A separate non-optimized frozen-pool feasibility probe
also found a rank-8 model with nine distinct forms and worse static costs; it
was not part of the retained optimized result.

The five structured rank-9 hashes are distinct from the explicit synthetic
control hash and from the rank-8 catalog. They share one structural-signature
class. Each has four distinct `u` and four distinct `v` forms, three-product
maximum multiplicity, ten total reuse counts, zero preparation additions, six
reconstruction additions, and support 28. An exact two-output-term merge of
each model reconstructs the conventional rank-8 anchor. They are therefore
known algebraic splits despite their `NEW_TO_PROJECT_CATALOG` hash label. Rank
9's apparent two extra reuse counts arise from its ninth multiplication. It
does not reduce the number of distinct input forms.

The structured rank-8 model used 512 action variables and 65 initial
constraints. Its complete enumeration found only the anchor. Structured
rank 9 used 512 action variables, eight form auxiliaries, and 73 initial
constraints. Frozen rank 8 used 96 action variables, 28 form auxiliaries, and
93 initial constraints. The retained model summaries contain final counts,
CP-SAT status sequences, conflicts, branches, and exact elapsed times. All
saved solutions and their SHA-256 manifests passed an independent post-run
check. The evidence is in `artifacts/tensor-game/hardware-aware-rank89-v1/`.

The decision is **NO_STRUCTURAL_SIGNAL** for operand-form reuse under these
bounded pools. Exact alternatives exist, but none lowers distinct input forms;
the target's flattening ranks show that the conventional anchor already meets
the lower bound. The rank-9 alternatives are simple known splits and add a
product. The two project-new rank-8 records have higher static costs. No GPU
candidate is recommended from this result alone. Other structural targets or
pool families would require a fresh rationale; finite-pool results cannot
settle their existence or hardware behavior.
