# Offline adaptive sampling

Milestone 3A learns a sampling distribution from saved evaluation records. It
does not run kernels or update itself during a hardware experiment. The method
is deliberately small and interpretable. No neural network, reinforcement
learning, Bayesian optimization, or tree search is involved.

## Algorithm

Eligible evaluation latencies are first grouped by candidate. Repeated
evaluations of one candidate are reduced to their median, so a frequently
evaluated candidate does not get more votes than another candidate. Duplicate
evaluation IDs are discarded.

For candidate latency `L(c)`, the bounded reward is:

```text
R(c) = (L_max - L(c)) / (L_max - L_min)
```

This lies in `[0, 1]`, with lower latency receiving higher reward. If all
eligible latencies are equal, every reward is `0.5`.

For parameter `p` and value `v`, the smoothed score is:

```text
S(p=v) = (a * 0.5 + sum(R(c) for c containing p=v)) / (a + count(p=v))
```

The default prior strength `a=2` shrinks small and uneven samples toward `0.5`.
The learned candidate weight is the product of its five parameter scores. These
weights are normalized across valid candidates. Exploration is mixed at the
candidate level:

```text
P(c) = (1 - epsilon) * P_learned(c) + epsilon / valid_candidate_count
```

The default `epsilon=0.25` gives every valid candidate nonzero probability.
Unique sampling uses sequential probability-proportional selection without
replacement. By default, candidates contributing eligible feedback are marked
seen and excluded from future unique samples.

## Feedback eligibility

An evaluation influences rewards only when it completed, passed correctness,
contains a real nonsynthetic finite positive latency in microseconds, and uses a
supported CUDA-event median measurement. Workload shape, precision, backend,
hardware target, recovered configuration, candidate identity, and search-space
membership must match.

Failures are counted by category but never receive latency or reward. Synthetic,
host-clock, malformed, incompatible, and unsupported measurements are excluded
with diagnostic reasons. One adaptive state accepts one explicit feedback source
and one software-environment fingerprint. This prevents silent combination of
unmatched sessions. Create separate states when comparing environments.
Algorithm version 1 accepts one offline feedback update per state. To incorporate
additional evidence, reconstruct a new state from the complete deduplicated
record set. This avoids accidentally replacing earlier evidence with only the
latest batch.

## Persistence and exact continuation

The version-1 adaptive state contains the search definition, exact valid
candidate set, context, source and environment identity, parameter scores,
candidate probabilities, update diagnostics, seed, Python random-generator
state, and seen candidate IDs. Loading that state reconstructs constraints from
the saved valid set. Continuing with the same calls produces the same samples.
Changing Python's random implementation in a future interpreter may require a
state migration.

Run the offline example against the preserved controlled experiment:

```sh
python examples/offline_adaptive_sampling.py \
  --feedback controlled_triton_fc1 \
  --output offline_adaptive_search
```

It writes `adaptive_state.json` and `sampled_candidates.jsonl`. Existing output
paths are not overwritten. If hardware evidence is unavailable, first create a
controlled experiment; synthetic `EvaluationRecord` measurements remain
ineligible by design and are used only in unit tests to verify exclusion.

## Limitations and relationship to AlphaTensor

The product of independent parameter scores cannot represent interactions. For
example, two individually neutral tile dimensions may be fast only in specific
pairs. Tests preserve an XOR-like case where the model remains uniform despite a
strong interaction. Smoothing reduces overreaction but does not make a small
sample representative of other kernels or hardware.

The conceptual connection to AlphaTensor is feedback-guided sampled actions.
AlphaTensor searched mathematical tensor decompositions with learned policies.
This sampler only adjusts probabilities over conventional kernel configuration
values using offline measured latency. Milestone 3B must compare this strategy
with uniform sampling in a controlled experiment before any search-efficiency
claim is justified.
