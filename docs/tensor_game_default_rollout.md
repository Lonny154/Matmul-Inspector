# TG-4Y default-provider rollout remediation

## Final decision

**GO. The public default was changed to `bounded_128`.** The repaired AWS installed-CLI audit passes every check and resolves the only failed TG-4Y gate. The original failed audit and original CONDITIONAL decision remain unchanged as historical evidence.

The frozen TG-4W and TG-4X evidence still contains 300 trials per host, 300 shared cross-machine keys, no missing records, no discovery mismatches, and 134/150 successes for each provider on each host. The confidence calculation still uses 150 independent targets. Timing was not pooled.

## Public behavior

Omitting the provider now selects `bounded_128` in the Python factory and installed CLI. Explicit `exhaustive` and explicit `bounded_128` remain available. Candidate budget, ranking implementations, MCTS, transitions, and exact-success logic were not changed. `provider_is_default` metadata now identifies bounded as the default.

```sh
matmul-inspector tensor-game-search \
  --target-json targets.json --target-id example

matmul-inspector tensor-game-search \
  --target-json targets.json --target-id example \
  --candidate-provider exhaustive
```

## Changed files

- `src/matmul_inspector/tensor_game_bounded_residual.py`: Python factory default.
- `src/matmul_inspector/tensor_game_cli.py`: CLI default and default metadata.
- `tests/tensor_game_provider_integration_test.py`: omitted and explicit-provider expectations.
- `docs/tensor_game_mcts.md` and `README.md`: public behavior and limitations.
- `examples/tensor_game_default_rollout_remediation.py`: reproducible remediation review and rollback check.
- `docs/tensor_game_default_rollout.md` and `artifacts/tensor-game/tg4y-default-rollout-remediation-v1/`: final evidence.

No candidate-ranking, MCTS, transition, or exact-success implementation changed.

## Risks

Bounded ranking is approximate and may return different candidates from exhaustive. Equal observed discovery does not prove universal equivalence. Performance depends on hardware and workload. Eight AWS exhaustive trials timed out, all on shared-failure targets; no timeout was reclassified as success.

## Rollback

Restore the factory and CLI parser defaults to `exhaustive`, and restore the metadata comparison to exhaustive. Keep both providers registered. Run the provider integration tests and installed CLI smokes. The procedure passed in an isolated temporary source copy without modifying the production tree. Roll back for an exhaustive-only discovery regression, correctness failure, persistent memory regression, material deployment incompatibility, or material bounded latency regression.
