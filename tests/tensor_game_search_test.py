"""Deterministic CPU tests for sampled TensorGame MCTS."""

import random
import unittest

from matmul_inspector.tensor_game import (
    RankOneAction,
    matrix_multiplication_tensor,
    outer_product,
    strassen_2x2_decomposition,
    sum_actions,
)
from matmul_inspector.tensor_game_search import (
    ActionStatistics,
    ReferenceGuidedPolicy,
    SampledMCTS,
    SearchConfig,
    SearchNode,
    UniformSampledPolicy,
    canonicalize_action,
    select_puct,
)


class PolicyTest(unittest.TestCase):
    def test_uniform_sampling_is_deterministic_valid_and_normalized(self):
        target = matrix_multiplication_tensor(1, 2, 1)
        policy = UniformSampledPolicy((2, 2, 1))
        first = policy.propose_actions(target, 3, random.Random(17), 12)
        second = policy.propose_actions(target, 3, random.Random(17), 12)
        self.assertEqual(first, second)
        self.assertEqual(len(first), len({item.action for item in first}))
        self.assertAlmostEqual(sum(item.prior for item in first), 1.0)
        for item in first:
            self.assertTrue(all(any(vector) for vector in (item.action.u, item.action.v, item.action.w)))
            self.assertTrue(set(item.action.u + item.action.v + item.action.w) <= {-1, 0, 1})

    def test_sign_equivalent_actions_are_canonicalized(self):
        action = RankOneAction((1, -1), (1,), (-1, 1))
        equivalent = RankOneAction((-1, 1), (-1,), (-1, 1))
        self.assertEqual(outer_product(action), outer_product(equivalent))
        self.assertEqual(canonicalize_action(action), canonicalize_action(equivalent))

    def test_reference_policy_only_claims_exact_reference_states(self):
        target = matrix_multiplication_tensor(2, 2, 2)
        reference = strassen_2x2_decomposition()
        policy = ReferenceGuidedPolicy(target, reference, randomized_order=False)
        proposals = policy.propose_actions(target, 7, random.Random(0), 7)
        self.assertEqual(len(proposals), 7)
        self.assertAlmostEqual(sum(item.prior for item in proposals), 1.0)
        unrelated = tuple(tuple(tuple(value + 1 for value in row) for row in plane) for plane in target)
        self.assertEqual(policy.propose_actions(unrelated, 7, random.Random(0), 7), ())

    def test_reference_policy_can_mix_positive_prior_distractors(self):
        target = matrix_multiplication_tensor(2, 2, 2)
        policy = ReferenceGuidedPolicy(
            target,
            strassen_2x2_decomposition(),
            reference_prior=0.8,
            distractor_policy=UniformSampledPolicy((4, 4, 4)),
            distractor_count=2,
            randomized_order=False,
        )
        proposals = policy.propose_actions(target, 7, random.Random(3), 9)
        self.assertLessEqual(len(proposals), 9)
        self.assertTrue(all(item.prior > 0 for item in proposals))
        self.assertAlmostEqual(sum(item.prior for item in proposals), 1.0)


class MCTSTest(unittest.TestCase):
    def test_state_identity_includes_remaining_budget(self):
        target = matrix_multiplication_tensor(1, 1, 1)
        search = SampledMCTS(target, UniformSampledPolicy((1, 1, 1)), SearchConfig(simulations=1))
        self.assertNotEqual(search._node(target, 1).key, search._node(target, 2).key)
        self.assertEqual(len(search.nodes), 2)

    def test_puct_tie_breaking_and_visit_preference_are_deterministic(self):
        residual = matrix_multiplication_tensor(1, 1, 1)
        first = RankOneAction((1,), (1,), (1,))
        second = RankOneAction((-1,), (1,), (1,))
        node = SearchNode(residual, 1, False, visits=0)
        node.actions = {first: ActionStatistics(0.5), second: ActionStatistics(0.5)}
        self.assertEqual(select_puct(node, 1.5), first)
        node.visits = 1
        node.actions[first].visits = 1
        self.assertEqual(select_puct(node, 1.5), second)

    def test_backup_accumulates_action_cost_and_visits(self):
        target = matrix_multiplication_tensor(1, 1, 1)
        action = RankOneAction((1,), (1,), (1,))
        search = SampledMCTS(target, UniformSampledPolicy((1, 1, 1)), SearchConfig(simulations=1, max_moves=1))
        parent = SearchNode(target, 1, False, actions={action: ActionStatistics(1.0)})
        leaf = SearchNode(sum_actions((), (1, 1, 1)), 0, True)
        search._backup(((parent, action),), leaf, 1.0)
        self.assertEqual((parent.visits, leaf.visits, parent.actions[action].visits), (1, 1, 1))
        self.assertEqual(parent.actions[action].value, 0.0)
        failed = SearchNode(target, 0, False)
        self.assertEqual(search._evaluate(failed), -2.0)

    def test_terminal_success_and_budget_enforcement(self):
        target = matrix_multiplication_tensor(1, 1, 1)
        reference = (RankOneAction((1,), (1,), (1,)),)
        policy = ReferenceGuidedPolicy(target, reference, randomized_order=False)
        result = SampledMCTS(target, policy, SearchConfig(simulations=4, max_actions=1, max_moves=1)).run()
        self.assertTrue(result.exact_reconstruction)
        self.assertEqual(result.scalar_multiplications, 1)
        self.assertEqual(result.termination_reason, "solution_found")
        self.assertEqual(result.simulations_run, 4)
        self.assertEqual(result.root_visits, 4)

    def test_reference_guided_strassen_is_reproducible_and_transposes(self):
        target = matrix_multiplication_tensor(2, 2, 2)
        policy = ReferenceGuidedPolicy(target, strassen_2x2_decomposition())
        config = SearchConfig(simulations=1200, max_actions=7, max_moves=7, seed=23)
        first = SampledMCTS(target, policy, config).run()
        second = SampledMCTS(target, policy, config).run()
        self.assertEqual(first, second)
        self.assertTrue(first.exact_reconstruction)
        self.assertEqual(first.scalar_multiplications, 7)
        self.assertEqual(sum_actions(first.solution, (4, 4, 4)), target)
        self.assertGreater(first.transposition_hits, 0)
        self.assertLessEqual(first.maximum_depth, 7)

    def test_insufficient_move_limit_rejects_incomplete_trajectory(self):
        target = matrix_multiplication_tensor(2, 2, 2)
        policy = ReferenceGuidedPolicy(target, strassen_2x2_decomposition())
        result = SampledMCTS(
            target,
            policy,
            SearchConfig(simulations=300, max_actions=7, max_moves=6, seed=5),
        ).run()
        self.assertFalse(result.exact_reconstruction)
        self.assertIsNone(result.solution)
        self.assertIsNone(result.scalar_multiplications)
        self.assertEqual(result.termination_reason, "simulation_budget_exhausted")

    def test_node_limit_is_respected(self):
        target = matrix_multiplication_tensor(1, 1, 1)
        policy = UniformSampledPolicy((1, 1, 1))
        result = SampledMCTS(
            target, policy, SearchConfig(simulations=10, max_actions=2, max_moves=2, node_limit=1)
        ).run()
        self.assertLessEqual(result.unique_states, 1)
        self.assertEqual(result.termination_reason, "node_limit_reached")


if __name__ == "__main__":
    unittest.main()
