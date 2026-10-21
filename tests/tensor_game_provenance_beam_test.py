import copy

import pytest

from matmul_inspector.tensor_game import (
    matrix_multiplication_tensor,
    standard_2x2_decomposition,
    subtract_action,
)
from matmul_inspector.tensor_game_diversity_search import DiscoveryConfig, discover
from matmul_inspector.tensor_game_quota_w import ProvenancedAction


class PathProvider:
    name = "test_provenance"

    def __init__(self, first="alternative"):
        self.actions = standard_2x2_decomposition()
        self.first = first
        self.calls = 0

    def propose_with_provenance(self, residual, limit):
        expected = matrix_multiplication_tensor(2, 2, 2)
        for depth, action in enumerate(self.actions):
            if residual == expected:
                self.calls += 1
                provenance = self.first if depth == 0 else "baseline"
                return (ProvenancedAction(action, provenance, "test"),)
            expected = subtract_action(expected, action)
        self.calls += 1
        return ()


def config(**changes):
    values = dict(
        rank=8,
        seed=3,
        node_budget=8,
        beam_width=1,
        max_actions=1,
        admission_policy="provenance_quota",
        baseline_capacity=1,
        alternative_capacity=0,
    )
    values.update(changes)
    return DiscoveryConfig(**values)


def stable(result):
    result = copy.deepcopy(result)
    result.pop("wall_seconds")
    return result


def test_provenance_is_per_edge_and_solution_path_reconstructs_exactly():
    result = discover(config(), PathProvider())
    assert result["success"]
    assert result["solutions"][0]["exact_validation"]
    assert result["depth_diagnostics"][0]["generated"] == {"alternative": 1}
    assert result["depth_diagnostics"][1]["generated"] == {"baseline": 1}


def test_quota_fill_and_determinism_when_a_class_is_empty():
    first = stable(discover(config(), PathProvider()))
    second = stable(discover(config(), PathProvider()))
    assert first == second
    assert first["depth_diagnostics"][0]["admitted"] == {"alternative": 1}
    assert first["depth_diagnostics"][0]["unused_quota_slots_before_fill"] == 1


def test_capacities_must_match_beam_width():
    with pytest.raises(ValueError, match="sum to beam_width"):
        config(beam_width=4, baseline_capacity=2, alternative_capacity=1)


def test_mixed_provenance_collision_uses_one_beam_slot():
    action = standard_2x2_decomposition()[0]

    class CollisionProvider:
        calls = 0

        def propose_with_provenance(self, residual, limit):
            self.calls += 1
            return (
                ProvenancedAction(action, "alternative", "test-alternative"),
                ProvenancedAction(action, "baseline", "test-baseline"),
            )

    result = discover(config(rank=2, node_budget=1), CollisionProvider())
    depth = result["depth_diagnostics"][0]
    assert depth["mixed_provenance_states"] == 1
    assert depth["beam_occupancy"] == 1
    assert sum(depth["removed_by_deduplication"].values()) == 1


def test_twelve_zero_matches_global_baseline_admission():
    common = dict(rank=8, seed=3, node_budget=32, beam_width=12, max_actions=12)
    global_result = discover(DiscoveryConfig(**common))
    quota_result = discover(DiscoveryConfig(
        **common,
        admission_policy="provenance_quota",
        baseline_capacity=12,
        alternative_capacity=0,
    ))
    for result in (global_result, quota_result):
        result.pop("wall_seconds")
        result.pop("depth_diagnostics")
        result["config"].pop("admission_policy")
        result["config"].pop("baseline_capacity")
        result["config"].pop("alternative_capacity")
        for solution in result["solutions"]:
            solution["reproduction_config"].pop("admission_policy")
            solution["reproduction_config"].pop("baseline_capacity")
            solution["reproduction_config"].pop("alternative_capacity")
    assert global_result == quota_result


def test_default_policy_and_serialization_are_stable():
    explicit = stable(discover(DiscoveryConfig(8, 3, node_budget=2, beam_width=4, max_actions=2)))
    assert explicit["config"]["admission_policy"] == "global"
    assert explicit["nodes_expanded"] <= 2
    assert isinstance(explicit["depth_diagnostics"], list)
