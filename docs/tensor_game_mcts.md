# Sampled MCTS for TensorGame

TG-2A adds a bounded CPU-only Monte Carlo tree search planner to the exact
TensorGame environment. It is a study of search mechanics. It is not a full
AlphaTensor reproduction and does not contain a learned policy or value model.

## From random sampling to tree search

Independent random sampling forgets earlier outcomes. MCTS instead builds a
tree of states, records action visits and values, and uses those observations to
allocate later simulations. Each simulation has four phases:

1. **Selection** follows actions chosen by PUCT through existing nodes.
2. **Expansion** samples a bounded action set at the first new state.
3. **Evaluation** assigns a deterministic value to that state.
4. **Backup** propagates the value through selected edges while subtracting the
   `-1` action cost at each step.

The implementation uses

```text
Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a)).
```

`Q` is the mean backed-up value, `P` is the normalized prior over actions that
were actually sampled at the node, and `N` stores visit counts. Exact score ties
use proposal insertion order. With a fixed random seed, proposals, ties, and
search results are reproducible.

Sampling avoids materializing the full action space. Even with coefficients
`{-1, 0, 1}`, a `(4, 4, 4)` action has `3^12` raw factor triplets before removing
zero vectors and equivalences. The uniform policy draws only the configured
number of candidates. It rejects zero factors and deduplicates the four factor
sign patterns that produce the same outer product.

## States and transpositions

A search state is the pair

```text
(exact residual tensor, remaining move budget).
```

The residual is immutable and hashable. Remaining depth is part of identity
because the same residual can be solvable with one budget and impossible with
another. Child residuals use the same exact subtraction routine as TensorGame.

Rank-one subtraction commutes. Applying actions `a` then `b` reaches the same
residual as `b` then `a`, with the same remaining budget. A transposition table
therefore reuses one node and its statistics for both paths. Depth always
decreases, so a traversal cannot cycle. A per-traversal identity set provides an
additional guard against accidental cycles.

## Values and outcomes

Every traversed action contributes `-1` during backup. A successful zero-residual
leaf has value `max_moves`, so its root return is `max_moves - trajectory_length`.
This makes successful shorter decompositions preferable within one configured
move limit. A nonzero leaf with no moves left has value `-(max_moves + 1)` and
can never be interpreted as success.

Unfinished leaves use `-residual_nonzeros / target_nonzeros`. This is a cheap,
deterministic guide, not a rank bound. A residual with fewer nonzero entries is
not necessarily closer to a minimum-rank decomposition.

Every candidate solution is independently checked by summing its outer products
and comparing the result exactly with the original target. Incomplete paths are
never returned as algorithms.

## Policies

The uniform policy is genuinely unguided. It samples bounded integer factors
from its configured coefficient alphabet and has no access to Strassen terms.
This search is expected to struggle because the useful actions are sparse in a
large combinatorial space and the initial heuristic is weak.

The reference-guided policy is a correctness fixture. It builds an exact subset
index for a supplied known decomposition, proposes factors that remain at a
matching residual, and may mix in separately sampled distractors with explicit
prior mass. It still traverses normal transitions and cannot report success
without reaching and reconstructing the zero residual. A guided Strassen result
is a replay found with privileged reference information, not a new discovery.

## Pseudocode

```text
repeat up to simulation_limit:
    node = root
    path = []
    while node is expanded and nonterminal:
        action = argmax PUCT(node, action)
        child = exact_transition(node, action)
        child = transposition_table.get_or_create(child)
        append (node, action) to path
        node = child
    if node is expandable and node_limit permits:
        node.actions = normalize(policy.propose_actions(node))
    value = terminal_or_heuristic_value(node)
    for edge in reversed(path):
        value = -1 + value
        update edge visits and mean value
```

## Running the demonstration

```sh
PYTHONPATH=src python examples/tensor_game_mcts.py
```

## Residual candidate providers

The installed CLI can run the residual-guided workflow on a target record or
manifest. Bounded residual ranking is the default:

```sh
matmul-inspector tensor-game-search \
  --target-json targets.json --target-id example

matmul-inspector tensor-game-search \
  --target-json targets.json --target-id example \
  --candidate-provider exhaustive
```

`bounded_128` streams all 6,400 `(u,v)` pairs
through a fixed 128-entry heap ranked by a continuous-relaxation bound, then
exactly evaluates those 128 pairs. It preserves
legal coefficients and exact final scores but is not guaranteed to reproduce
the exhaustive top 12.

Python callers can use:

```python
from matmul_inspector.tensor_game_bounded_residual import (
    create_residual_candidate_provider,
)

policy = create_residual_candidate_provider()  # bounded_128
exhaustive = create_residual_candidate_provider("exhaustive")
```

Both policies implement the existing `ActionPolicy` interface and can be passed
to `SampledMCTS`. Unsupported provider identifiers raise `ValueError`.
Bounded ranking is an approximation and may produce a different candidate set
from exhaustive ranking. The tested corpora showed no discovery regression, but
that does not prove universal equivalence. Performance depends on workload and
hardware.

The example runs a bounded reference-guided seven-term Strassen search and then
a small unguided budget. Failure in the unguided run is reported directly.

## Limits and TG-2B

- Dense residual tensors and per-node action dictionaries target small problems.
- Reference subset indexing is exponential and capped at 20 terms. It is only a
  fixture mechanism.
- The uniform proposal distribution and nonzero-count evaluator are weak.
- Search retains all unique states until completion or the node limit.
- No symmetry handling beyond factor-sign equivalence is implemented.
- There is no batching, neural inference, self-play, reinforcement learning, GPU
  acceleration, or hardware reward.

TG-2B can use this measured baseline for independent discovery experiments,
better mathematical action proposals, stronger rank-aware evaluation, and more
careful scaling studies.

## Reference

Alhussein Fawzi et al., “Discovering faster matrix multiplication algorithms
with reinforcement learning,” *Nature* 610, 47–53 (2022).
[doi:10.1038/s41586-022-05172-4](https://doi.org/10.1038/s41586-022-05172-4).
