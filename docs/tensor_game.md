# TensorGame mathematical environment

TensorGame is a CPU-only research environment for expressing matrix
multiplication as exact tensor decomposition. This first milestone provides the
mathematics and known reference trajectories. It does not perform search or
learning.

## Matrix multiplication as a tensor

Multiplication of an `M x K` matrix `A` and a `K x N` matrix `B` is bilinear:
each output is linear in `A` when `B` is fixed and linear in `B` when `A` is
fixed. A third-order tensor records which flattened input pairs contribute to
each flattened output. With row-major indices, its shape is
`(M*K, K*N, M*N)` and

```text
T[i*K + k, k*N + j, i*N + j] = 1.
```

Every other coefficient is zero. The implementation stores immutable nested
tuples of Python integers. This keeps construction, transitions, and identity
checks exact.

## Rank-one actions

An action contains three integer coefficient vectors `(u, v, w)`. Their outer
product `u ⊗ v ⊗ w` represents one scalar multiplication: form one linear
combination of entries from each input, multiply the two values, then distribute
the product to outputs using `w`. A decomposition is correct exactly when the
sum of these outer products equals the target tensor.

TensorGame starts with the target as its residual. Each valid action applies

```text
residual_next = residual - u ⊗ v ⊗ w
```

and receives reward `-1`. A zero residual is successful termination. Reaching
the move limit with a nonzero residual is truncation, not a valid decomposition.
Invalid actions raise an exception before any state changes. The initial action
alphabet defaults to coefficients in `{-1, 0, 1}`. Callers can explicitly choose
a different finite set or allow arbitrary integers.

## Exact reference evidence

The package includes ordinary 2x2 multiplication as eight rank-one terms and
Strassen multiplication as seven explicit terms. Tests sum both decompositions
and compare them directly with the same `(4, 4, 4)` target. They also replay the
actions until the residual is exactly zero and evaluate both algorithms on small
integer matrices. No floating-point tolerance is involved.

Tensor rank describes the minimum number of rank-one terms needed to represent
the bilinear map. Reducing that count can improve the asymptotic arithmetic
complexity when the construction composes recursively. It does not by itself
predict practical hardware latency. Addition count, memory traffic, shapes,
parallelism, numerical behavior, compiler lowering, and the target device still
matter.

## Relationship to AlphaTensor

This state/action formulation follows the mathematical starting point used by
AlphaTensor: matrix multiplication algorithm discovery becomes a finite-horizon
game over a residual tensor. AlphaTensor adds learned representations and search
machinery. TG-1 intentionally adds neither. The next milestone is sampled tree
search over a carefully bounded action space.

## Run the reference example

From an installed package or a source checkout with `src` on `PYTHONPATH`:

```sh
PYTHONPATH=src python examples/tensor_game_reference.py
```

The output labels both trajectories as known algorithms rather than discovered
results and reports the exact residual after every move.

## Current limitations

- Dense nested tuples are intended only for small tensors.
- The environment has no action enumerator, symmetry reduction, search, neural
  network, reinforcement learning, or hardware cost model.
- The default coefficient alphabet is deliberately small.
- Lower tensor rank is not treated as a measured performance improvement.
- The multiplication evaluator currently expects integral inputs and factors for
  the exact arithmetic contract.

## Reference

Alhussein Fawzi et al., “Discovering faster matrix multiplication algorithms
with reinforcement learning,” *Nature* 610, 47–53 (2022).
[doi:10.1038/s41586-022-05172-4](https://doi.org/10.1038/s41586-022-05172-4).
