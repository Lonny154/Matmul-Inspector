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

Findings from the committed implementation will be added with the evidence.
