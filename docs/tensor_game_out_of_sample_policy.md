# TG-4E out-of-sample policy evaluation

TG-4E compared frozen top-12, trained top-12, and a deterministic equal-budget
hybrid on a preregistered fresh target set. No model was retrained and production
search was unchanged.

## Frozen protocol

The evaluation used 16 generated `4×4×4` targets, with four targets at each
constructed depth from one through four. Constructed depth is a verified upper
bound, not a proven minimum tensor rank. Each policy used seeds 3, 19, and 35,
200 simulations, 12 candidates, PUCT 1.5, a 200-node limit, the nonzero heuristic,
and a five-second search cap. This produced 48 searches per policy and 144 total.

Exact targets and all six axis permutations were checked against the 200-target
TG-4B.2A curriculum and the earlier 2,000-example synthetic corpus. None of the
64 deterministic generator-pool candidates selected for the balanced set
collided. The preregistration was written before checkpoint evaluation and has
SHA-256 `b58a1ba725d19e38c78c340c2820c353ecb82683e2d3d0c5635678dfb02ed5ec`.

The hybrid alternated frozen and trained proposals by within-model rank, started
with frozen, canonicalized and deduplicated actions, and filled at most 12 slots.
It assigned uniform priors so incomparable cross-checkpoint scores were not mixed.

## Results

| Policy | Successes | Targets solved | Mean simulations | Mean expansions | Mean forwards | Mean CPU search wall |
|---|---:|---:|---:|---:|---:|---:|
| Frozen top-12 | 0/48 | 0/16 | 199.2 | 89.2 | 89.2 | 2.059 s |
| Trained top-12 | 0/48 | 0/16 | 199.1 | 90.3 | 90.3 | 2.090 s |
| Hybrid top-12 | 0/48 | 0/16 | 158.5 | 59.9 | 119.8 | 2.699 s |

Every paired comparison was a shared failure. This held separately at constructed
depths one, two, three, and four. The hybrid's lower simulation and expansion
counts are not an efficiency gain. Its two forward passes per expanded state made
the five-second cap bind more often at depths three and four.

There are no successful-run efficiency statistics because no run succeeded.
Likewise, there are no frozen-only, trained-only, hybrid-only, or shared successes.
The experiment cannot rank policies by exact discovery when all observed rates
are zero.

## Reference-action recall

The generated trajectories contain 40 reference decisions. They are one known
solution path, not an exhaustive oracle set.

| Policy | Top-1 | Top-12 |
|---|---:|---:|
| Frozen | 1/40 | 3/40 |
| Trained | 0/40 | 0/40 |
| Hybrid | 1/40 | 3/40 |

The hybrid preserved the three frozen reference actions but added no trained
reference-action coverage. Alternative valid actions were not exhaustively
enumerated, so a proposal differing from the reference is not labeled invalid.

## Hybrid attribution

No hybrid search succeeded, so there are no successful solution actions to
attribute and no justified candidate-removal ablation. A trained-exclusive action
appearing in an unsuccessful search would not establish useful contribution.

## Interpretation

- **H1 is unsupported:** trained top-12 did not outperform frozen at any depth.
- **H2 is unsupported:** neither policy succeeded, and their overall simulation,
  expansion, and wall-time measurements were similar.
- **H3 is unsupported:** the hybrid produced no additional discovery and no new
  reference-action recall.
- **H4 is supported for this benchmark:** improved imitation loss again produced
  no measured recall, success, or efficiency benefit.
- **H5 is supported operationally:** hybrid inference roughly doubled model
  forwards per expansion and caused earlier wall-time termination without benefit.

These are small-benchmark observations, not universal claims. Four targets per
depth provide little target-level precision. The search budget was also too weak
to solve even depth-one targets, so the experiment has a floor effect. That floor
effect prevents a strong conclusion that frozen and trained policies are equally
capable.

## Recommendation for TG-4F

Focus next on proposal generation and benchmark calibration, not additional
policy training. Preregister a CPU feasibility study using the same fresh-target
generator and reference-action diagnostics. Add a non-neural residual-informed
control and sweep search budget on validation-only targets until depth-one
problems have a nonzero but non-saturated success rate. Then freeze that budget
for a new policy comparison. This establishes an evaluation regime capable of
detecting improvements before spending another training run.

The ignored evidence bundle is under
`artifacts/tensor-game/tg4e-out-of-sample-v1/` and contains the preregistration,
hash, targets, results, reference recall, empty hybrid attribution, summary, and
full report.

Reproduce the two immutable stages from the repository root:

```sh
PYTHONPATH=src python examples/tensor_game_out_of_sample.py prepare \
  --output artifacts/tensor-game/tg4e-out-of-sample-v1 \
  --known artifacts/tensor-game/tg4b2a-diverse-full-v1/manifest.json \
  --known artifacts/tensor-game/tg3b-dataset-2000/decompositions.jsonl

PYTHONPATH=src python examples/tensor_game_out_of_sample.py evaluate \
  --output artifacts/tensor-game/tg4e-out-of-sample-v1 \
  --baseline-checkpoint artifacts/tensor-game/tg3b-mlp-256-seed42 \
  --trained-checkpoint artifacts/tensor-game/tg4b2b-offline-ablation-v1/candidates/policy_only_natural
```

The inspected Git commit was
`4a83b8d6677de426297226f2f72ab5c959bd3554`. Python, PyTorch, platform,
checkpoint hashes, and the preregistration hash are recorded in `summary.json`.
