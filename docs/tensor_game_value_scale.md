# MCTS value scale and trajectory study

TG-3E separates absolute leaf-value optimism from learned state ranking. In this
MCTS, PUCT selects `Q + c_puct * prior * sqrt(parent_visits)/(1+child_visits)`.
Backup subtracts one per traversed action. Exact zero residuals receive
`max_moves`, exhausted nonterminal states receive `-(max_moves+1)`, and heuristic
leaf values describe the evaluated state before incurred edge costs are backed up.
Transposition keys include both residual and remaining moves.

The study fixes residual-informed proposals, seeds `3, 11, 19, 27, 35, 43`, 200
simulations, 12 proposals, 200 states, five seconds, PUCT 1.5, and move limits
eight and seven. It compares constants 0, `max_moves/2`, and `max_moves` with the
trained value and positive scales 0.5 and 2.0. Positive scaling preserves value
ordering after the existing prediction clamp, but changes Q magnitude relative
to the exploration bonus.

On the eight-term target, constant zero succeeded 1/6. Every other control
succeeded 6/6. Constant `max_moves`, original trained value, and expanded trained
value all succeeded at simulation nine and produced identical factor sequences
within each matched seed. The sequences differed across seeds. Their initial
proposal batches were identical within each seed, confirming that the repeated
paths arose from seeded proposals plus value-scale-driven traversal rather than
reference injection. Compressed values also succeeded 6/6, later at simulations
41–71. Constant half succeeded 6/6 at simulations 18–51.

No configuration found a seven-term solution. All 31 saved successes passed
independent exact-integer reconstruction. Terminal success occurred only after an
exact zero residual. Transposition reuse did not mix move budgets.

These observations strongly support absolute optimism as the main explanation
of the TG-3C/TG-3D result. Learned ranking was not necessary for success under
this experiment. Magnitude still affected how quickly the successful path was
traversed. This is a mechanistic result for one target and bounded search budget,
not a general statement about useful value learning.

A within-depth permutation was not run. The value-dependent searches do not
share a preregistered fixed residual pool. Assigning shuffled values online would
make state/value association depend on evaluation order and would fail the
intended control. A future study must first construct an external fixed pool and
freeze the mapping.

Bounded traces store the first 20 simulations, selected action digests, depth,
terminal outcome, root visits, Q values, priors, exploration bonuses, and PUCT
scores. Evidence also records value transformations, clipping, factor-sequence
digests, timing, and exact solutions.

## Reference

Alhussein Fawzi et al., “Discovering faster matrix multiplication algorithms
with reinforcement learning,” *Nature* 610, 47–53 (2022),
<https://doi.org/10.1038/s41586-022-05172-4>.
