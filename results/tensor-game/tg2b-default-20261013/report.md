# TG-2B independent discovery comparison

These CPU results compare matched MCTS budgets. No policy received known standard or Strassen factors.
A stopped or unsuccessful trial is not evidence that a decomposition does not exist.

Seeds: `[3, 11]`. Simulations per trial: `200`. Proposals per expansion: `12`. Node limit: `200`.

| Target | Strategy | Successes | Trials | Mean completed simulations | Mean wall time (s) |
|---|---|---:|---:|---:|---:|
| level1_rank_one_2x2x2 | uniform_sampled | 0 | 2 | 200.0 | 0.002818 |
| level1_rank_one_2x2x2 | residual_informed | 2 | 2 | 200.0 | 0.010617 |
| level1_rank_one_2x2x2 | hybrid | 1 | 2 | 200.0 | 0.003351 |
| level2_rank_two_2x2x2 | uniform_sampled | 0 | 2 | 200.0 | 0.008138 |
| level2_rank_two_2x2x2 | residual_informed | 2 | 2 | 200.0 | 0.021726 |
| level2_rank_two_2x2x2 | hybrid | 2 | 2 | 200.0 | 0.026567 |
| level3_matmul_1x2x1 | uniform_sampled | 1 | 2 | 200.0 | 0.008657 |
| level3_matmul_1x2x1 | residual_informed | 2 | 2 | 200.0 | 0.018291 |
| level3_matmul_1x2x1 | hybrid | 2 | 2 | 200.0 | 0.020503 |
| level4_matmul_2x2_eight | uniform_sampled | 0 | 2 | 200.0 | 0.046108 |
| level4_matmul_2x2_eight | residual_informed | 0 | 2 | 119.0 | 2.005236 |
| level4_matmul_2x2_eight | hybrid | 0 | 2 | 200.0 | 0.479427 |
| level5_matmul_2x2_seven | uniform_sampled | 0 | 2 | 200.0 | 0.039645 |
| level5_matmul_2x2_seven | residual_informed | 0 | 2 | 117.5 | 2.009469 |
| level5_matmul_2x2_seven | hybrid | 0 | 2 | 200.0 | 0.489176 |

Success means exact reconstruction within the configured move limit. Timing is CPU wall time and includes proposal generation and MCTS bookkeeping.
Level 5 success, if any, must be inspected from the exact saved factors before making a rediscovery claim.
