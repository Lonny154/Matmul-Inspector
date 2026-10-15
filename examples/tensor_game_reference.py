"""Replay known 2x2 matrix multiplication decompositions exactly."""

from matmul_inspector.tensor_game import (
    TensorGame,
    matrix_multiplication_tensor,
    multiply_with_decomposition,
    standard_2x2_decomposition,
    strassen_2x2_decomposition,
)


def replay(name, actions, target):
    print(f"\n{name} (known reference algorithm, not a discovered result)")
    game = TensorGame(target, max_moves=len(actions))
    for move, action in enumerate(actions, start=1):
        result = game.step(action)
        print(f"  move {move}: residual nonzeros={game.residual_nonzeros}")
    print(f"  exact reconstruction: {game.reconstructs_target()}")
    print(f"  terminated={result.terminated}, truncated={result.truncated}")


def main():
    target = matrix_multiplication_tensor(2, 2, 2)
    nonzeros = sum(value != 0 for plane in target for row in plane for value in row)
    print(f"2x2 matrix multiplication tensor: shape=(4, 4, 4), nonzeros={nonzeros}")
    replay("Standard eight-move decomposition", standard_2x2_decomposition(), target)
    replay("Strassen seven-move decomposition", strassen_2x2_decomposition(), target)
    left = ((2, -3), (4, 5))
    right = ((-1, 6), (7, 2))
    result = multiply_with_decomposition(left, right, (2, 2, 2), strassen_2x2_decomposition())
    print(f"\nStrassen integer example: {left} @ {right} = {result}")


if __name__ == "__main__":
    main()
