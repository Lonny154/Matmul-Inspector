# Tensor Game policy and candidate-provider optimization history

This is the decision trail from TG-4C through TG-4V. Results are not directly
comparable when corpora or budgets differ. Each linked report contains the
frozen inputs, raw measurements, and limitations.

| Study | Question and evidence | Decision carried forward |
|---|---|---|
| TG-4C | Traced the learned-policy regression to proposal coverage and ranking rather than MCTS mechanics. | Measure candidate recall before changing search. |
| TG-4D | Frozen top-12 and a frozen/trained hybrid recovered 24/24 historical discoveries. Trained top-12 recovered 0/24. | Retain frozen controls and test independently generated tensors. |
| TG-4E | Frozen, trained, and hybrid policies each solved 0/48 trials on 16 fresh targets. | Treat the learned policy as out-of-sample inadequate. |
| TG-4F | Calibrated search budgets and residual-oracle controls. | Separate candidate availability from downstream MCTS selection. |
| TG-4G | Learned providers solved 0/24 held-out trials. Residual search solved 24/24. | Move optimization effort to residual-guided proposals. |
| TG-4H | Verified residual supervision feasibility and its enumeration cost. | Test coverage-aware objectives without claiming a trained fix. |
| TG-4I–M | Set, retention, prefix, final-ranking, and broader coverage objectives improved pieces of the ranking problem but remained weak. The best broad validation recall was 7/44 frozen versus 2/44 for the broader prefix model. | Do not replace the frozen policy with these objectives. |
| TG-4N | Isolated local gradient gains from factorized decoding interference and combinatorial dilution. | Test conditional action factorization. |
| TG-4O | Conditional training improved component metrics but recalled only 1/44 complete validation actions, versus 7/44 frozen. | Reject the conditional model as a provider replacement. |
| TG-4P | Residual proposals covered 28/30 validation states versus 7/30 frozen. Adaptive routing matched residual coverage without lowering measured cost. | Optimize the residual provider directly. |
| TG-4Q | `bounded_128` matched exhaustive coverage at 28/30 states and 31/44 positives. Mean proposal latency was 29.03 ms versus 47.03 ms. | Carry `B=128` into held-out search. |
| TG-4R | Both providers succeeded in 36/36 trials on 12 fresh targets. Bounded mean search was 1.23× faster. | Integrate bounded as an opt-in provider. |
| TG-4S | On 16 new targets both succeeded in 33/48 trials. Bounded was 1.24× faster but used 1.93 MB mean traced peak memory. | Keep exhaustive default and fix bounded memory. |
| TG-4T | A fixed-size streaming heap preserved exact output on 49 states, cut mean traced peak 29.13×, and slightly improved latency. | Integrate the exact-output-equivalent implementation. |
| TG-4U | Replaced the opt-in bounded provider internals with the streaming selector and added permanent differential and search regressions. | Run an independent default-readiness study. |
| TG-4V | On 24 structurally targeted targets, both providers succeeded in 63/72 trials with no asymmetric outcomes. Integrated bounded was 2.16× faster on mean, faster in 71/72 pairs, and passed all frozen correctness, memory, timeout, compatibility, and uncertainty gates. | **GO** means the evidence supports a separate default-change review. Exhaustive remains the default in this study. |

The sequence contains important negative results. Learned-policy coverage work
did not solve held-out generalization. Adaptive mixing did not beat direct
residual proposals. The final positive result comes from bounded exact scoring
and memory engineering, not from a demonstrated learned-policy improvement.

See the [TG-4V readiness report](tensor_game_default_readiness.md) for the final
criteria and the recorded analysis-harness correction.
