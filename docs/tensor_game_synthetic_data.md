# Synthetic demonstrations for neural tensor discovery

TG-3A builds exact supervised examples for future TensorGame policy and value
models. It implements the synthetic-demonstration direction used by AlphaTensor
at a much smaller scale. It does not define or train a neural network.

## Forward construction

For a configured construction length `R`, the generator samples nonzero integer
factor triplets and constructs

```text
T = sum(r=1..R) u_r ⊗ v_r ⊗ w_r.
```

Factors use `{-1, 0, 1}`. A configurable probability controls how often each
coefficient is zero. Every vector is forced to remain nonzero, and each action is
put in the canonical sign representation already used by TensorGame search. The
generator materializes only the sampled actions, never the full action space.

The stored construction proves that the tensor has a decomposition of length at
most `R`. It does not normally prove that its tensor rank is exactly `R`.

## Residual supervision

For factors `(a_1, ..., a_R)`, each nonterminal residual is stored:

```text
S_t = sum(r=t..R) a_r.
```

The full remaining factor list preserves multiplicity. Policy supervision lists
every distinct remaining action as a valid next target. If an action occurs
twice, it appears once in the policy target but twice in the remaining multiset.
Subtracting any labeled action is verified against the exact sum after removing
one occurrence.

This avoids presenting one arbitrary factor ordering as unique ground truth.
The known completion length is a value-training upper bound. It is not an exact
minimum-rank label, and other valid actions or decompositions may exist.

## Split and leakage rules

Splits operate on complete originating decompositions. No residual row from one
trajectory can move to another split. Before accepting a decomposition, the
generator checks:

- the target tensor has not appeared before;
- no nonterminal residual would cross a split boundary;
- the canonical factor multiset has not already been accepted;
- the constructed target is nonzero.

Collisions cause deterministic regeneration and are counted in the manifest.
Exact tensor and canonical-factor deduplication cannot detect every algebraically
equivalent decomposition. Held-out evaluation therefore means held-out exact
originating tensors, not a proof of algebraic novelty.

The 2x2 matrix multiplication tensor and its standard and Strassen decompositions
are not part of the default training data. They remain downstream evaluation and
reference fixtures.

## Versioned persistence

A dataset directory contains:

```text
manifest.json
decompositions.jsonl
samples.jsonl
```

The manifest records schema and generator versions, family, seed, configuration,
split counts, rejection statistics, record counts, and SHA-256 file digests.
Records include content digests. Writes use a temporary sibling directory and an
atomic rename. Existing output directories are rejected.

A decomposition record contains the exact target and factor sequence. A sample
record resembles:

```json
{
  "decomposition_id": "...",
  "split": "train",
  "residual": [[[1]]],
  "remaining_factors": [{"u": [-1], "v": [-1], "w": [1]}],
  "next_actions": [{"u": [-1], "v": [-1], "w": [1]}],
  "known_completion_length": 1,
  "generator_version": "tg3a.synthetic.v1"
}
```

Loading rejects unsupported versions, malformed schemas, changed file or record
digests, invalid coefficients, zero actions, broken transitions, reconstruction
failures, and split leakage.

## Optional PyTorch interface

`TorchSyntheticDataset(dataset, split="train", dtype="float32")` imports PyTorch
only when instantiated. Dataset generation and loading have no ML dependency.
Each item provides:

- `residual`: model input with the original tensor shape;
- `next_actions`: padded `int64` factor vectors with shape
  `(maximum construction length, sum(tensor shape))`;
- `action_mask`: which padded rows are valid multi-action labels;
- `known_completion_length`: separate `int64` value supervision;
- `allowed_coefficients`: the discrete action alphabet.

Residuals may use `float32`, `float64`, or `int64`. Float32 conversion rejects
integers outside its exact consecutive-integer range rather than silently losing
precision. Action targets remain integers. The fixed padded shapes work with a
standard future `torch.utils.data.DataLoader`.

## Run the example

```sh
PYTHONPATH=src python examples/tensor_game_synthetic_data.py
```

The default creates three decompositions at each construction length one through
four: 12 originating tensors and 30 residual examples. It validates, saves,
reloads, and validates again. A larger but still explicit option is:

```sh
PYTHONPATH=src python examples/tensor_game_synthetic_data.py \
  --larger --output artifacts/tensor-game/tg3a-larger
```

## Limits and TG-3B recommendation

JSONL repeats dense residual tensors and factor metadata, so storage grows with
`examples * product(shape)`. Generation also performs exact reconstruction and
cross-split digest checks in memory. The current design is appropriate for small
4x4x4 curricula, not millions of examples.

For TG-3B, start with roughly 1,000–5,000 originating 4x4x4 decompositions,
balanced across construction lengths one through four. Measure serialized size,
loader throughput, target ambiguity, and validation time before scaling further.

## Reference

Alhussein Fawzi et al., “Discovering faster matrix multiplication algorithms
with reinforcement learning,” *Nature* 610, 47–53 (2022).
[doi:10.1038/s41586-022-05172-4](https://doi.org/10.1038/s41586-022-05172-4).
