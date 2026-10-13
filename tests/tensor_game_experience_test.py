"""CPU tests for TG-4A experience collection and exact replay."""

import random
import tempfile
from pathlib import Path
import unittest

from matmul_inspector.tensor_game import RankOneAction, matrix_multiplication_tensor
from matmul_inspector.tensor_game_experience import (
    CollectionConfig, ReplayBuffer, action_identity, load_buffer, play_episode,
    replay_episode, save_buffer, search_policy, select_behavior_action,
)
from matmul_inspector.tensor_game_search import (
    HeuristicValueEvaluator, ProposedAction, RootActionDiagnostic,
)


class FixedPolicy:
    name = "fixed_test_policy"
    def __init__(self, action): self.action = action
    def propose_actions(self, residual, remaining_moves, rng, max_actions):
        return (ProposedAction(self.action, 1.0),)


class ExperienceTest(unittest.TestCase):
    def test_search_policy_temperature_and_zero_visits(self):
        a = RankOneAction((1,), (1,), (1,))
        b = RankOneAction((-1,), (1,), (1,))
        roots = (RootActionDiagnostic(a, 0.25, 1, 2.0), RootActionDiagnostic(b, 0.75, 3, -1.0))
        policy = search_policy(roots, 1.0)
        self.assertEqual([x["search_probability"] for x in policy], [0.25, 0.75])
        zero = search_policy(tuple(RootActionDiagnostic(x.action, x.prior, 0, x.value) for x in roots), 1.0)
        self.assertEqual([x["search_probability"] for x in zero], [0.25, 0.75])
        deterministic = search_policy(roots, 0.0)
        self.assertEqual([x["search_probability"] for x in deterministic], [0.0, 1.0])
        chosen, probability = select_behavior_action(deterministic, random.Random(9), 0.0)
        self.assertEqual((chosen["action_id"], probability), (action_identity(b), 1.0))

    def _episode(self, action, *, max_moves=1, stop=None):
        target = matrix_multiplication_tensor(1, 1, 1)
        return play_episode(
            target, max_moves, policy_factory=lambda: FixedPolicy(action),
            evaluator_factory=HeuristicValueEvaluator,
            config=CollectionConfig(simulations=3, max_actions=1, node_limit=10),
            seed=3, experiment_id="test", configuration_id="fixed",
            model_provenance={"type": "none"}, stop_after_decisions=stop,
        )

    def test_successful_played_episode_and_exact_replay(self):
        episode, decisions = self._episode(RankOneAction((1,), (1,), (1,)))
        self.assertEqual((episode["outcome"], episode["exact_reconstruction"]), ("success", True))
        self.assertEqual(len(decisions), 1)
        self.assertFalse(episode["internal_simulations_are_episode_steps"])
        self.assertEqual(decisions[0]["observed_return_target"], 0)
        report = replay_episode(episode, {x["experience_id"]: x for x in decisions})
        self.assertTrue(report["valid"])

    def test_truncated_and_interrupted_outcomes_are_distinct(self):
        truncated, rows = self._episode(RankOneAction((-1,), (1,), (1,)))
        self.assertEqual(truncated["outcome"], "truncated")
        self.assertFalse(truncated["exact_reconstruction"])
        self.assertEqual(rows[0]["observed_return_target"], -3)
        interrupted, rows = self._episode(RankOneAction((1,), (1,), (1,)), stop=0)
        self.assertEqual(interrupted["outcome"], "interrupted")
        self.assertEqual(rows, [])

    def test_episode_wall_time_limit_is_censored(self):
        target = matrix_multiplication_tensor(1, 1, 1)
        episode, rows = play_episode(
            target, 1, policy_factory=lambda: FixedPolicy(RankOneAction((1,), (1,), (1,))),
            evaluator_factory=HeuristicValueEvaluator,
            config=CollectionConfig(simulations=3, max_actions=1, node_limit=10),
            seed=3, experiment_id="wall-limit", configuration_id="fixed",
            model_provenance={"type": "none"}, episode_wall_time_seconds=1e-12,
        )
        self.assertEqual((episode["outcome"], episode["stopping_reason"]), ("interrupted", "episode_wall_time_limit"))
        self.assertEqual(rows, [])

    def test_buffer_duplicates_capacity_corruption_and_reload(self):
        first, first_rows = self._episode(RankOneAction((1,), (1,), (1,)))
        buffer = ReplayBuffer(capacity_episodes=1)
        buffer.add(first, first_rows)
        with self.assertRaises(ValueError):
            buffer.add(first, first_rows)
        corrupted = dict(first_rows[0])
        corrupted["next_state_digest"] = "bad"
        self.assertFalse(replay_episode(first, {corrupted["experience_id"]: corrupted})["valid"])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "buffer"
            save_buffer(output, buffer, {"experiment": "test"})
            loaded = load_buffer(output)
            self.assertTrue(loaded.validate()["valid"])
            self.assertEqual((len(loaded.episodes), len(loaded.decisions)), (1, 1))
            with self.assertRaises(FileExistsError):
                save_buffer(output, buffer, {})

    def test_canonical_identity_collapses_sign_equivalence(self):
        first = RankOneAction((1,), (1,), (1,))
        equivalent = RankOneAction((-1,), (-1,), (1,))
        self.assertEqual(action_identity(first), action_identity(equivalent))


if __name__ == "__main__":
    unittest.main()
