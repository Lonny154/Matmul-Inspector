# TG-4D candidate recall and policy coverage

TG-4D tested whether proposal coverage can recover the four critical one-step
targets from TG-4C without retraining or modifying production search.

The frozen policy recalled all four exact actions at top-12 and every larger
cutoff. The trained policy recalled none at top-1, 12, 32, 64, 128, or 256.
Ranks absent at 256 remain bounded observations, not exhaustive ranks.

Across the four targets and six original seeds, frozen top-12 recovered 24/24
discoveries. Trained top-12 recovered 0/24. A deterministic equal-budget hybrid
alternated frozen and trained candidates by source rank and stopped at 12. It
recovered 24/24. An expanded union of both top-12 lists used 24 candidates and
also recovered 24/24. Both combined policies needed two model inferences. Neither
showed a success or expansion advantage over frozen top-12 on this one-step set.
The union is not presented as compute-matched to the 12-candidate configurations.

Training coverage was weak. Three critical canonical actions never appeared in
the 1,400 training decisions' sampled supports. The fourth appeared four times
with 0.263158 total supervised target-probability mass. This supports
underrepresentation but does not prove causation. Exact actions were matched
across different residual states, and the held-out critical states were not part
of training.

The strongest observation is that candidate coverage is the immediate bottleneck
for these four targets. Increasing the trained decoder budget through 256 is
insufficient. There is no evidence here that trained proposals add useful ranking
information once frozen coverage is protected.

The full ignored evidence bundle is at
`artifacts/tensor-game/tg4d-candidate-coverage-v1`. It includes `report.md`,
`summary.json`, `candidate_recall.csv`, `proposal_union.csv`,
`training_coverage.json`, and `training_coverage.csv`.

The recommended TG-4E experiment is a preregistered comparison of frozen top-12
and the equal-budget hybrid on a fresh target set spanning one to four steps.
It should persist per-decision oracle-action recall, source attribution,
simulations to first success, expansions, and inference time. This tests whether
coverage protection generalizes and whether trained candidates help later in a
trajectory without tuning on the existing test targets.
