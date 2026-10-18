# TG-Global-2: matched tempering ablation and repairability audit

## Questions and controls

This CPU study asks whether adjacent replica exchange changes search outcomes,
whether exact two-term repair changes them, and whether best states contain a
five-term core that the configured pool can complete.

The independent replica control runs the same local Metropolis mutation at the
same temperatures. It disables exchange without consuming exchange RNG draws.
For each seed, all four variants start from the same canonical state for each
replica. A dedicated initialization stream draws these states once. Each
replica then uses its own seeded local stream. Exchange and repair have separate
seeded streams. Local streams start alike across variants and diverge only as
exchanges move states between temperatures or repair changes the incumbent.

The variants are `independent_no_repair`, `tempered_no_repair`,
`independent_with_repair`, and `tempered_with_repair`. Every variant receives
the same fixed number of local mutation proposals: replicas × rounds × local
steps. Exact residual evaluations are counted separately because an identical
proposal requires none. Exchange and repair work, wall time, and table storage
are also reported. The pair-sum table is constructed once per study and shared
read-only by repair and audit. This avoids charging duplicate construction to
individual variants; the study records peak traced memory.

## Repairability audit

For each best state, the audit removes all 21 position pairs. It computes the
exact tensor required after freezing the other five terms and queries the same
bounded pair-sum table used by repair. A hash hit is checked by exact tensor
equality. Domain rules are applied and the completed seven-term tensor is
reconstructed independently. The report separates raw matches, rejected
matches, verified completions, and the number of removable cores that have at
least one verified completion. Details are capped at ten records per candidate.
The known Strassen control checks the mechanics.

## Pilot protocol

The default pilot uses 12 paired discovery seeds, two replicas at temperatures
1 and 2, four rounds, two local steps per replica per round, and `nnz_l1` with
mismatch weight 4. Repair runs every two rounds. The full profile supports 30
seeds and 100 rounds. Both profiles accept explicit seeds, bounds, and a
per-run time limit. The pilot is descriptive and does not test statistical
significance.

```sh
PYTHONPATH=src python scripts/run_tensor_game_tempered_study.py \
  --output artifacts/tensor-game/tempered-study-pilot-v1
```

The oracle pool requires `--pool strassen_oracle --oracle-positive-control`.
Discovery mode uses the existing frozen algebraic slice-projection pool and
does not inject Strassen actions. Artifacts contain the command, Git state,
pool hash, initial-state hashes, per-run counters, paired seed differences,
repairability records, and file hashes.

## Recorded pilot result

The retained 12-seed pilot used seeds 20261018–20261029 and the frozen
`algebraic_slice_projection_128` pool. Every variant used 192 local proposals
in total. Independent variants made 191 exact local residual evaluations each;
exchanged variants made 188 each because a few proposals repeated the current
state. The common pair table was ready for all repair and audit calls.

| Variant | Exact / 12 | Median best nnz | Median best L1 | Median best energy | Repairable best / 12 |
|---|---:|---:|---:|---:|---:|
| Independent, no repair | 0 | 10 | 10 | 50 | 0 |
| Tempered, no repair | 0 | 9.5 | 9.5 | 47.5 | 0 |
| Independent, repair | 0 | 10 | 10 | 50 | 0 |
| Tempered, repair | 0 | 9.5 | 9.5 | 47.5 | 0 |

Pairing by seed and initial-state hashes gives exchange 3 lower, 6 equal, and
3 higher best energies against the independent control. This is the same with
and without repair. Repair changed no best energy in either exchange setting.
The median paired energy difference is zero for all four required comparisons.
No discovery candidate was exact or two-term repairable. The oracle control
reconstructed known Strassen exactly and verified 21 of 21 removable cores.

Retained artifacts are under `artifacts/tensor-game/tempered-study-pilot-v1/`
and `artifacts/tensor-game/tempered-study-oracle-control-v1/`. Both were
generated from code commit `34bc8ab`. Small TG-Global-1 oracle and discovery
smokes were also regenerated from that commit under
`artifacts/tensor-game/tempered-search-*-smoke-v1/`.

## Interpretation and limits

A near-zero residual is not an exact decomposition. Exact oracle recovery
validates mechanics only. A better residual under unmatched compute is not
evidence that tempering is better. Even matched local proposals do not imply
equal computational cost: repair and exchange have their own costs. Wall times
are measured in sequential Python runs and depend on host load and run order.

Failure of two-term repair may reflect pool coverage rather than the search
algorithm. Absence of a repairable five-term core does not prove that no rank-7
decomposition exists in the full domain. No GPU acceleration is justified by
this experiment.
