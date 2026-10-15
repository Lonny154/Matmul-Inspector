# Independent tensor decomposition discovery

TG-2B compares three action-proposal strategies inside the same sampled MCTS
planner. The policies receive the current exact residual, dimensions, coefficient
alphabet, and random generator. They do not receive standard multiplication or
Strassen factors.

This is a bounded CPU search study. It is not a reproduction of AlphaTensor and
does not use neural policy or value networks.

## Residual-informed proposals

For sampled integer factors `u` and `v`, the policy contracts the residual `S`:

```text
d_k = sum_ij S_ijk u_i v_j
q   = (sum_i u_i^2) (sum_j v_j^2)
w*_k = d_k / q
```

For fixed real-valued `u` and `v`, `w*` is the least-squares output direction.
TensorGame actions must remain exact integers, so the policy selects the allowed
coefficient `c` minimizing `|d_k - q*c|`. This comparison uses integers and does
not round through floating point. A zero `u` or `v` makes `q=0`; that proposal is
discarded safely.

The real-valued expression is only a proposal heuristic. Quantization can erase
useful directions or produce a zero output factor. A proposed action is accepted
only after normal TensorGame validation.

The policy also samples sparse slice-aligned candidates. It chooses a nonzero
residual `(i,j)` slice, uses signed basis factors for the first two modes, and
quantizes the corresponding output slice. This can exactly cancel a visible
residual contribution. It does not establish that the action belongs to a
minimum-rank decomposition.

Both heuristics are bounded by an attempt limit. They canonicalize sign-equivalent
factorizations and remove duplicates. They never enumerate the full triplet
space.

## Uniform and hybrid baselines

The existing uniform policy samples all three factor vectors from the same
coefficient alphabet. It remains the baseline.

The hybrid policy makes a seeded choice between uniform and residual-informed
proposal slots using a configurable mixing probability. Each component first
produces proposal weights. Duplicate actions are merged, then all weights are
normalized over the actions actually retained at that node. These normalized
values are MCTS priors. Raw heuristic scores are not described as calibrated
probabilities.

PUCT uses these priors through

```text
Q(s,a) + c_puct P(s,a) sqrt(N(s)) / (1 + N(s,a)).
```

The value evaluator, transition rules, exact transposition identity, and backup
semantics are identical across all three strategies.

## Curriculum

The checked-in curriculum contains:

1. A dense `2x2x2` tensor constructed from one rank-one term.
2. A `2x2x2` tensor constructed from two terms. Its mode-0 flattening has exact
   matrix rank two, which certifies tensor rank of at least two. The construction
   supplies the matching two-term upper bound.
3. The `1x2` by `2x1` matrix multiplication tensor with a two-action limit.
4. The 2x2 matrix multiplication tensor with an eight-action limit.
5. The same tensor with a seven-action stretch limit.

In general, constructing a tensor from two terms proves only an upper bound of
two. A lower-bound certificate such as the flattening rank is needed before
calling it exactly rank two.

Level 4 is independent discovery only because the proposal policies are not
initialized with the known eight terms. Level 5 failure is expected under small
budgets and says nothing about mathematical existence. A level 5 success would
still require inspection of its saved exact factors before any rediscovery claim.

## Evidence and cost accounting

Each trial records the target digest and shape, strategy, seed, MCTS limits,
termination reason, exact factors when successful, states, expansions,
transpositions, depth, proposal attempts, valid and accepted proposals,
duplicates, proposal time, total CPU wall time, and simulations per second.

The simulation allowance is matched across strategies. A wall-time limit may
cause a more expensive policy to complete fewer simulations. The record keeps
both permitted and completed counts so this cost difference remains visible.
Searches also have hard proposal and unique-node limits.

Every reported solution is independently reconstructed using exact integers.
Residual sparsity is only a heuristic signal. For example, cancelling one entry
of a two-term target lowers its nonzero count but still fails under a one-move
limit.

## Run the modest experiment

```sh
PYTHONPATH=src python examples/tensor_game_discovery.py
```

The default runs two deterministic seeds with 200 simulations per strategy and
target, a 200-state limit, and a two-second per-trial CPU limit. It writes a new
timestamped directory containing `results.json` and `report.md`. Existing
evidence directories are never overwritten.

An explicitly larger, still bounded experiment is available:

```sh
PYTHONPATH=src python examples/tensor_game_discovery.py \
  --extended --output artifacts/tensor-game/tg2b-extended
```

Inspect small-trial costs before increasing these limits.

## Limitations and future use

- Contraction and sparse slices provide local signals, not tensor-rank bounds.
- Integer quantization may discard useful real-valued directions.
- Duplicate-heavy residual states can spend many attempts producing few actions.
- The dense residual representation limits problem size.
- Wall timings include Python, proposal generation, and MCTS bookkeeping. They
  are CPU execution measurements, not accelerator results.
- Multiple seeded successes are search evidence, not proof of completeness or
  optimality.

Exact successful trajectories and unsuccessful state/action examples could later
form policy-training data. TG-2B does not train on them.

## Reference

Alhussein Fawzi et al., “Discovering faster matrix multiplication algorithms
with reinforcement learning,” *Nature* 610, 47–53 (2022).
[doi:10.1038/s41586-022-05172-4](https://doi.org/10.1038/s41586-022-05172-4).
