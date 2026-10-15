# Neural value guidance study

TG-3D follows up the TG-3C observation that residual-informed MCTS found two
eight-term 2×2 matrix multiplication decompositions only when it used the trained
TG-3B value head. The study asks whether that result reflects learned value
ranking or a more general change in leaf-value scale.

The value head predicts a synthetic known-completion length. It does not predict
or certify minimum tensor rank. A nonterminal prediction `L` is clamped to
`[0, remaining_moves]` and converted to `max_moves - L`. Exact zero residuals and
exhausted move budgets are handled by MCTS before any heuristic evaluator runs.
Backup then subtracts one for each already-traversed action. Hand-calculated
two-edge tests confirm that incurred costs are not counted twice. Transposition
keys include both the residual and remaining move budget.

## Controls and method

Every configuration uses the same residual-informed proposal policy, coefficient
set, PUCT constant, proposal limit, node limit, target, and policy seed. The four
leaf evaluators are the original nonzero heuristic, the trained neural value,
constant zero, and an untrained copy of the same MLP initialized with seed
`20261013`. Terminal outcomes always override these controls.

The preregistered search uses seeds `3, 11, 19, 27, 35, 43`, 200 simulations,
12 proposals, 200 unique states, a five-second wall limit, and move limits eight
and seven. Identical seeds reproduce proposal randomness only until value-guided
tree traversal diverges. They do not force identical expanded states.

Diagnostics retain aggregate counts for every query and at most 64 query records
per trial. A record includes a residual digest, depth, remaining budget, raw and
clamped length, converted value, clipping reason, and residual nonzero count.
Reference factors are never supplied to search.

## Findings

On the eight-term target, trained and untrained neural values each succeeded for
all six seeds. The heuristic succeeded for two seeds and constant zero for one.
No configuration found a seven-term decomposition. All 15 saved successes were
independently reconstructed with exact integer arithmetic.

The trained evaluator clipped 131 of 1,245 queries (10.5%), always because its
predicted completion length exceeded the remaining budget. The untrained
evaluator clipped all 1,216 queries because every prediction was negative. Its
clamped length was therefore zero and its converted leaf value was maximally
optimistic. Despite having no learned weights, it matched the trained evaluator's
six eight-term successes and first-success simulation count of nine.

The TG-3C neural-value effect therefore replicated as a planner behavior but did
not isolate useful learned value representations. The untrained control is
consistent with the hypothesis that optimistic value scaling or clipping changes
PUCT traversal in a favorable way for this target. That is an interpretation,
not a causal proof. Constant zero is on a different MCTS value scale and did not
match the neural controls.

All trials completed 200 simulations without timeouts. Neural inference was a
small portion of total search time. Differences in total time mainly followed
different tree shapes and proposal-generation counts, so equal simulation limits
are not equal compute budgets.

The held-out synthetic diagnostic covers one test target at each construction
length from one through four. It illustrates out-of-distribution prediction and
clipping behavior but is not search-performance evidence and does not establish
true tensor rank.

## Limitations and next step

This is six paired seeds on one tensor at two move limits. Eight- and seven-term
targets are different search problems and are not pooled as interchangeable
evidence. Failure at seven terms is only a bounded-search result.

The next controlled study should preregister value-scale controls before running:
a constant `max_moves` evaluator, monotone transformations of trained values,
and trained-prediction permutations among comparable depths. That would separate
absolute optimism, within-depth ranking, and clipping effects before any
self-play or reinforcement learning work.

## Reference

Alhussein Fawzi et al., “Discovering faster matrix multiplication algorithms
with reinforcement learning,” *Nature* 610, 47–53 (2022),
<https://doi.org/10.1038/s41586-022-05172-4>.
