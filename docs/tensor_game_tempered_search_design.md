# TG-Global-1: parallel tempering with exact two-term repair

## Scope and motivation

TG-Global-1 is a bounded CPU-only search-method experiment. It is not a new
matrix multiplication algorithm. It does not change the default Tensor Game
MCTS or candidate providers.

Sequential residual search scores one rank-one term at a time. A useful term in
a cancellation-heavy decomposition can initially increase the residual. Local
residual ranking can reject that term before complementary terms are present.
This prototype tests a different hypothesis: searching complete seven-term
states may preserve combinations whose value is only visible jointly.

Oracle recovery validates mechanics only. A lower residual is not an exact
decomposition. Every reported solution requires independent exact
reconstruction.

## State, pool, and energy

A state is a sorted tuple of seven indices into one immutable finite action
pool. Sorting removes term-order permutations. Each pool action uses the
existing sign canonicalization and SHA-256 action identity. Repeated actions
are disabled by default, matching the existing meet-in-the-middle experiment.
They can be enabled explicitly.

All tensors are immutable nested tuples of Python integers. For exact residual
`R = T - sum(a_i)`, the implementation records:

```text
nnz = count(R != 0)
l1 = sum(abs(R))
max_abs = max(abs(R))
```

The `l1` energy is `l1`. The `nnz_l1` energy is
`mismatch_weight * nnz + l1`. Energy zero means exact reconstruction, but the
candidate is still rebuilt independently before it receives `EXACT` status.

## Parallel tempering

Each replica has a fixed positive temperature. A local proposal replaces one
term with another pool action and canonicalizes the full state. For energy
change `delta`, an uphill proposal is accepted with
`exp(-delta / temperature)`. Equal or improving proposals are accepted.

For adjacent replicas `i` and `j`, the swap log acceptance ratio is:

```text
(1 / Ti - 1 / Tj) * (Ei - Ej)
```

The code compares logarithms to avoid exponent overflow and underflow. All RNGs
are local and derived deterministically from one master seed. Temperatures must
be finite and strictly positive. Exchanges alternate adjacent-pair parity.

## Exact two-term repair

The optional repair operator removes two positions and freezes the other five.
It computes the exact tensor required from the missing pair. A bounded table of
pool pair sums uses the existing meet-in-the-middle tensor hash and confirms
exact equality after the hash lookup. Domain rules are checked before a new
seven-term state is accepted. The full candidate is reconstructed again.

Repair distinguishes no complement, domain rejection, reconstruction failure,
verified exact completion, and pool, record, or construction-time budget
limits. Pool size and pair-record limits are checked before table allocation.

## Oracle separation and controls

The standalone command has two explicit pools:

- `strassen_oracle` is positive-control only and requires
  `--oracle-positive-control`. Its output says that it is oracle-assisted and
  not a discovery.
- `algebraic_slice_projection_128` reproduces the frozen algebraic-pair and
  slice-projection construction without consulting Strassen during generation.
  It retains the frozen-pair hash and generation audit.

Discovery mode rejects the oracle-only pool. No fallback injects oracle terms.
Artifacts record the pool hash, provenance, seed, full configuration, bounded
history, counters, best candidate, exact validation, branch, and commit.

## Reproduction

Small positive-control smoke run:

```sh
PYTHONPATH=src python scripts/run_tensor_game_tempered_search.py \
  --pool strassen_oracle --oracle-positive-control \
  --rounds 2 --local-steps 1 --temperatures 1,2 \
  --repair --repair-interval 1 \
  --output /tmp/tg-global-1-smoke
```

A future matched study should freeze pool, seeds, temperatures, proposal count,
wall-clock limit, and exact-validation rules before comparing this method with
sequential residual search. It should report success counts and work, not only
best residual.

## Limitations

This is a small Python prototype over finite pools. Parallel tempering does not
guarantee an exact result. Two-term repair can only find terms already present
in the active pool. An oracle-assisted exact result is not a discovery. No CPU
speed or search-quality advantage has been established.

CPU behavior must be established before GPU batching is attempted. GPU
acceleration would only make sense after the experiment demonstrates enough
large batched candidate evaluation to amortize transfer and launch costs.
