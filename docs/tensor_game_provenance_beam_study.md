# Provenance-aware Tensor Game beam admission

## Research question

The quota provider protected baseline actions but previously lost rank-8 reliability under a global 128-state beam. This CPU study asks whether an optional admission quota can keep baseline-origin children while still carrying a bounded number of alternative-origin children.

The implementation succeeds under the requested 12-state beam. Every node-matched rank-8 configuration solves 3/3 trials. Rank 7 remains 0/3. The main limitation is that global 11/1 admission also solves 3/3 at this narrower beam. The experiment therefore proves compatibility and deterministic alternative retention. It does not isolate provenance admission as the cause of recovered reliability relative to the earlier 128-state study.

## Existing interference and provenance representation

The existing search expanded a layer, deduplicated by exact residual tensor, sorted all retained paths by residual norm and deterministic identity, then truncated the list globally. `QuotaMultiWProvider.__call__` returned plain `RankOneAction` objects. Its `protected_baseline`, `baseline_fallback`, and `quota_*_alternative` sources disappeared at that boundary.

The new `ProvenancedAction` labels each incoming edge as `baseline` or `alternative`. It does not label the residual permanently. A state reached through an alternative edge can use baseline edges at the next depth. The provider's old callable interface still returns plain actions, so existing callers and default global admission are unchanged.

## Admission and collision policy

`DiscoveryConfig.admission_policy="provenance_quota"` enables the experiment. Baseline and alternative capacities must be nonnegative and sum to the total beam width. Children keep the existing exact residual update, validity rules, score, and deterministic ordering.

Deduplication still uses the exact residual state and canonical path identity. A state reached through both provenance classes occupies one slot. Existing deterministic generation order chooses the retained path when `paths_per_state=1`; provenance does not override that owner. The collision's provenance set is retained for diagnostics. Admission first takes each class quota, then fills unused slots from the globally ranked remainder. No known solution affects admission.

## Controlled configuration

All node-matched trials use ranks 7 and 8, seeds 3, 11, and 29, beam width 12, maximum branching 12, and node budget 256. Quota providers use pair width 128 and two exact `w` candidates per pair. Residual and diversity alternative ranking are both tested. The 12/0 control uses the original baseline provider. The 11/1 quota provider is also tested with unchanged global admission.

Approximately evaluation-matched quota groups use 64 nodes. They score about 92,859–93,081 complete actions per three-trial group, close to the older full baseline study's roughly 95,000 evaluations. This is a proposal-evaluation match, not a wall-time or reachable-depth match.

Reproduce the study with:

```sh
PYTHONPATH=src python scripts/run_tensor_game_provenance_beam.py \
  --output artifacts/tensor-game/provenance-beam-v1
```

The command refuses to overwrite an existing output directory.

## Node-matched results

| Provider and admission | Rank 7 | Rank 8 | Max depth | Nodes, rank 8 | Complete actions, rank 8 | CPU time, rank 8 |
|---|---:|---:|---:|---:|---:|---:|
| Baseline global 12/0 | 0/3 | 3/3 | 8 | 225 | 28,800 | 6.89 s |
| Residual quota global 11/1 | 0/3 | 3/3 | 8 | 246 | 120,273 | 9.09 s |
| Residual provenance 11/1 | 0/3 | 3/3 | 8 | 246 | 120,099 | 8.88 s |
| Residual provenance 10/2 | 0/3 | 3/3 | 8 | 249 | 121,443 | 9.11 s |
| Residual provenance 8/4 | 0/3 | 3/3 | 8 | 255 | 124,287 | 9.51 s |
| Diversity quota global 11/1 | 0/3 | 3/3 | 8 | 246 | 120,375 | 9.36 s |
| Diversity provenance 11/1 | 0/3 | 3/3 | 8 | 246 | 120,300 | 9.32 s |
| Diversity provenance 10/2 | 0/3 | 3/3 | 8 | 249 | 121,611 | 9.49 s |
| Diversity provenance 8/4 | 0/3 | 3/3 | 8 | 255 | 124,770 | 9.48 s |

Times are measured Python CPU wall times on this run. They are not kernel benchmarks. All successful decompositions pass exact integer reconstruction.

Every approximately evaluation-matched rank-8 quota trial stops at the 64-node budget at depth 7 and fails 0/3. The node-matched success therefore depends on enough search horizon, not only admission composition.

## Beam occupancy and alternative survival

For the residual strategy, seed 3 rank-8 traces have the following `(beam, baseline, alternative)` occupancy by depth:

| Admission | Depth 1 | Depth 2 | Depth 3 | Depth 4 | Depth 5 | Depth 6 | Depth 7 |
|---|---|---|---|---|---|---|---|
| Baseline 12/0 | 8/8/0 | 12/12/0 | 12/12/0 | 12/12/0 | 12/12/0 | 12/12/0 | 6/6/0 |
| Global 11/1 | 9/8/1 | 12/12/0 | 12/12/0 | 12/12/0 | 12/12/0 | 12/12/0 | 12/6/6 |
| Provenance 11/1 | 9/8/1 | 12/11/1 | 12/11/1 | 12/11/1 | 12/11/1 | 12/11/1 | 12/8/4 |
| Provenance 10/2 | 10/8/2 | 12/10/2 | 12/10/2 | 12/10/2 | 12/10/2 | 12/10/2 | 12/7/5 |
| Provenance 8/4 | 12/8/4 | 12/8/4 | 12/8/4 | 12/8/4 | 12/8/4 | 12/8/4 | 12/8/4 |

Global 11/1 usually removes alternative children after the first layer. Provenance admission reliably carries them through subsequent layers. Fallback permits more than the nominal class quota when the other class cannot fill its allocation, as seen at depth 7.

## Known-trajectory diagnostics

The diagnostic follows exact conventional and Strassen trajectories offline. It records parent pruning, provider omission, invalid transitions, deduplication, and admission rejection without changing normal ranking.

The conventional rank-8 trajectory remains available at every depth for every node-matched rank-8 configuration. There is no observed baseline-path loss. The exact Strassen first action is absent from the bounded pair/provider output in every configuration, so the known Strassen trajectory is unavailable at depth 1. Provenance-aware admission cannot recover an action the provider never generates. This agrees with the earlier finding that pair coverage remains a separate bottleneck.

The rank-7 runs also trace the eight-term conventional path for diagnosis. Its seventh child is correctly marked ineligible at the rank-7 depth limit, not as a mathematical error.

## Diagnostics and evidence

Each depth records generated, valid, invalid, deduplicated, admitted, and beam-rejected counts by provenance. It also records mixed-provenance states, unused quota before fallback, beam occupancy, expanded nodes, cumulative provider calls, residual nonzero-count distributions, and final termination reason. Solution records preserve the full action path and exact validation metadata.

Raw evidence is under `artifacts/tensor-game/provenance-beam-v1/`:

- `trials.json`: all 78 trials with depth and known-trajectory traces.
- `summary.json`: matched aggregate results.
- `configuration.json`: frozen command and experiment matrix.
- `artifact_hashes.json`: SHA-256 hashes and byte sizes.

## Conclusion and limitations

The primary success criterion is met: every provenance allocation, including allocations with alternative capacity, preserves rank-8 reliability at 3/3 under the matched 12-state beam. Alternative children survive beyond the first layer. Rank 7 remains 0/3.

The result does not show that provenance admission alone fixes the earlier 128-state-beam failure because the global 11/1 control also recovers 3/3 here. The changed total beam width alters frontier composition and node use. The experiment also cannot improve Strassen trajectory survival when the first required action is missing from the candidate provider.

The next experiment should hold the earlier 128-state beam fixed and compare global versus provenance-aware admission with capacities scaled explicitly for that beam. It should first test whether the conventional trajectory is rejected by admission there. Only after that causal check should pair coverage be expanded for the missing first Strassen action.
