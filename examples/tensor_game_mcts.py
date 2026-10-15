"""Run bounded reference-guided and unguided TensorGame searches on CPU."""

from matmul_inspector.tensor_game import (
    matrix_multiplication_tensor,
    strassen_2x2_decomposition,
    subtract_action,
)
from matmul_inspector.tensor_game_search import (
    ReferenceGuidedPolicy,
    SampledMCTS,
    SearchConfig,
    UniformSampledPolicy,
)


def summarize(label, result):
    print(f"\n{label}")
    print(
        f"  simulations={result.simulations_run}, expanded={result.expanded_nodes}, "
        f"unique_states={result.unique_states}, transpositions={result.transposition_hits}"
    )
    print(f"  termination={result.termination_reason}, exact={result.exact_reconstruction}")
    if result.solution is None:
        print("  no complete decomposition found; no algorithm is reported")
        return
    residual = matrix_multiplication_tensor(2, 2, 2)
    for move, action in enumerate(result.solution, start=1):
        residual = subtract_action(residual, action)
        nonzeros = sum(value != 0 for plane in residual for row in plane for value in row)
        print(f"  move {move}: {action}, residual nonzeros={nonzeros}")
    print(f"  scalar multiplications={result.scalar_multiplications}")


def main():
    target = matrix_multiplication_tensor(2, 2, 2)
    guided = ReferenceGuidedPolicy(
        target,
        strassen_2x2_decomposition(),
        randomized_order=True,
    )
    guided_result = SampledMCTS(
        target,
        guided,
        SearchConfig(simulations=1200, max_actions=7, max_moves=7, seed=23),
    ).run()
    summarize("Reference-guided Strassen search (not independent discovery)", guided_result)

    unguided = UniformSampledPolicy((4, 4, 4))
    unguided_result = SampledMCTS(
        target,
        unguided,
        SearchConfig(simulations=100, max_actions=8, max_moves=7, seed=23, node_limit=100),
    ).run()
    summarize("Small unguided sampled search", unguided_result)


if __name__ == "__main__":
    main()
