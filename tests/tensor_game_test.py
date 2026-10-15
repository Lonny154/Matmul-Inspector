"""CPU-only exact tests for the TensorGame research environment."""

import unittest

from matmul_inspector.tensor_game import (
    RankOneAction,
    TensorGame,
    matrix_multiplication_tensor,
    multiply_with_decomposition,
    outer_product,
    standard_2x2_decomposition,
    strassen_2x2_decomposition,
    sum_actions,
    tensor_shape,
    validate_action,
)


class TensorGameTest(unittest.TestCase):
    def test_square_tensor_indices(self):
        target = matrix_multiplication_tensor(2, 2, 2)
        self.assertEqual(tensor_shape(target), (4, 4, 4))
        self.assertEqual(sum(x != 0 for plane in target for row in plane for x in row), 8)
        expected = {(i * 2 + k, k * 2 + j, i * 2 + j) for i in range(2) for k in range(2) for j in range(2)}
        actual = {(x, y, z) for x in range(4) for y in range(4) for z in range(4) if target[x][y][z]}
        self.assertEqual(actual, expected)

    def test_rectangular_tensor_and_dimensions(self):
        target = matrix_multiplication_tensor(2, 3, 1)
        self.assertEqual(tensor_shape(target), (6, 3, 2))
        self.assertEqual(sum(x != 0 for plane in target for row in plane for x in row), 6)
        for dimensions in ((0, 2, 2), (2, -1, 2), (2, 2, 1.5)):
            with self.assertRaises(ValueError):
                matrix_multiplication_tensor(*dimensions)

    def test_outer_product_and_validation(self):
        action = RankOneAction((1, -1), (0, 1), (1, 0))
        self.assertEqual(outer_product(action)[1][1][0], -1)
        validate_action(action, (2, 2, 2))
        with self.assertRaisesRegex(ValueError, "lengths"):
            validate_action(action, (3, 2, 2))
        with self.assertRaisesRegex(ValueError, "outside"):
            validate_action(RankOneAction((2,), (1,), (1,)), (1, 1, 1))
        with self.assertRaisesRegex(ValueError, "nonzero"):
            validate_action(RankOneAction((0,), (1,), (1,)), (1, 1, 1))
        with self.assertRaisesRegex(ValueError, "integers"):
            validate_action(RankOneAction((1.0,), (1,), (1,)), (1, 1, 1), None)

    def test_transition_invalid_atomicity_and_move_limit(self):
        game = TensorGame(matrix_multiplication_tensor(1, 1, 1), max_moves=2)
        before = (game.residual, tuple(game.history), game.terminated, game.truncated)
        with self.assertRaises(ValueError):
            game.step(RankOneAction((0,), (1,), (1,)))
        self.assertEqual((game.residual, tuple(game.history), game.terminated, game.truncated), before)
        result = game.step(RankOneAction((-1,), (1,), (1,)))
        self.assertEqual(result.reward, -1)
        self.assertFalse(result.terminated)
        result = game.step(RankOneAction((1,), (-1,), (1,)))
        self.assertFalse(result.terminated)
        self.assertTrue(result.truncated)
        self.assertFalse(game.reconstructs_target())

    def test_exact_success(self):
        game = TensorGame(matrix_multiplication_tensor(1, 1, 1), max_moves=1)
        result = game.step(RankOneAction((1,), (1,), (1,)))
        self.assertEqual((result.reward, result.terminated, result.truncated), (-1, True, False))
        self.assertEqual(game.residual_nonzeros, 0)
        self.assertTrue(game.reconstructs_target())

    def test_reference_decompositions_and_replay(self):
        target = matrix_multiplication_tensor(2, 2, 2)
        standard = standard_2x2_decomposition()
        strassen = strassen_2x2_decomposition()
        self.assertEqual(len(standard), 8)
        self.assertEqual(len(strassen), 7)
        self.assertEqual(sum_actions(standard, (4, 4, 4)), target)
        self.assertEqual(sum_actions(strassen, (4, 4, 4)), target)
        for actions in (standard, strassen):
            game = TensorGame(target, max_moves=len(actions))
            states = []
            for action in actions:
                states.append((game.step(action), game.residual))
            self.assertTrue(game.terminated)
            self.assertFalse(game.truncated)
            self.assertTrue(game.reconstructs_target())
            repeat = TensorGame(target, max_moves=len(actions))
            repeated = [(repeat.step(action), repeat.residual) for action in actions]
            self.assertEqual(repeated, states)

    def test_integer_matrix_multiplication(self):
        left = ((2, -3), (4, 5))
        right = ((-1, 6), (7, 2))
        expected = ((-23, 6), (31, 34))
        self.assertEqual(multiply_with_decomposition(left, right, (2, 2, 2), standard_2x2_decomposition()), expected)
        self.assertEqual(multiply_with_decomposition(left, right, (2, 2, 2), strassen_2x2_decomposition()), expected)


if __name__ == "__main__":
    unittest.main()
