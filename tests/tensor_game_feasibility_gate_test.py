import copy
import json

import pytest

from matmul_inspector.tensor_game import (
    matrix_multiplication_tensor,
    standard_2x2_decomposition,
    strassen_2x2_decomposition,
    subtract_action,
    zero_tensor,
)
from matmul_inspector.tensor_game_diversity_search import (
    DiscoveryConfig,
    discover,
    feasibility_gate_decision,
)


def test_gate_is_off_by_default_and_explicit_off_is_equivalent():
    default = discover(DiscoveryConfig(7, 3, node_budget=8, beam_width=4, max_actions=3))
    explicit = discover(DiscoveryConfig(
        7, 3, node_budget=8, beam_width=4, max_actions=3, feasibility_gate="off"
    ))
    default.pop("wall_seconds")
    explicit.pop("wall_seconds")
    assert default == explicit
    assert default["feasibility_gate_diagnostics"]["checked_children"] == 0


def test_exact_gate_prunes_only_when_bound_strictly_exceeds_budget():
    target = matrix_multiplication_tensor(2, 2, 2)
    rejected = feasibility_gate_decision(target, 3, gate="flattening_rank")
    assert rejected["active"]
    assert rejected["flattening_ranks"] == [4, 4, 4]
    assert rejected["lower_bound"] == 4
    assert rejected["prune"]

    residual = target
    for action in standard_2x2_decomposition()[:5]:
        residual = subtract_action(residual, action)
    retained = feasibility_gate_decision(residual, 3, gate="flattening_rank")
    assert retained["active"] and retained["lower_bound"] == 3
    assert not retained["prune"]


def test_gate_activates_only_at_configured_late_depth_budget():
    target = matrix_multiplication_tensor(2, 2, 2)
    early = feasibility_gate_decision(target, 4, gate="flattening_rank")
    late = feasibility_gate_decision(target, 3, gate="flattening_rank")
    assert not early["active"] and early["lower_bound"] is None
    assert late["active"] and late["remaining_budget"] == 3
    with pytest.raises(ValueError, match="nonnegative"):
        feasibility_gate_decision(target, -1, gate="flattening_rank")


@pytest.mark.parametrize(
    ("rank", "actions"),
    ((8, standard_2x2_decomposition()), (7, strassen_2x2_decomposition())),
)
def test_known_valid_trajectory_children_survive(rank, actions):
    residual = matrix_multiplication_tensor(2, 2, 2)
    for depth, action in enumerate(actions, 1):
        residual = subtract_action(residual, action)
        remaining = rank - depth
        decision = feasibility_gate_decision(residual, remaining, gate="flattening_rank")
        assert not decision["prune"]
    assert residual == zero_tensor((4, 4, 4))
    terminal = feasibility_gate_decision(residual, 0, gate="flattening_rank")
    assert terminal["active"] and terminal["lower_bound"] == 0


def test_search_budget_accounting_and_rank8_terminal_solution_are_preserved():
    result = discover(DiscoveryConfig(
        8, 3, node_budget=256, beam_width=12, max_actions=12,
        feasibility_gate="flattening_rank",
    ))
    assert result["success"]
    assert result["nodes_expanded"] <= 256
    assert result["provider_calls"] == result["nodes_expanded"]
    checked_depths = [
        row for row in result["depth_diagnostics"]
        if row["feasibility"]["checked_children"]
    ]
    assert {row["feasibility"]["remaining_budget"] for row in checked_depths} == {1, 2, 3}
    assert not any(row["feasibility"]["pruned_children"] for row in checked_depths)


def test_gate_diagnostics_are_deterministic_after_removing_timings():
    config = DiscoveryConfig(
        7, 11, node_budget=32, beam_width=8, max_actions=6,
        feasibility_gate="flattening_rank",
    )
    first = discover(config)
    second = discover(config)

    def without_timings(value):
        value = copy.deepcopy(value)
        value.pop("wall_seconds")
        value["feasibility_gate_diagnostics"].pop("check_seconds")
        for row in value["depth_diagnostics"]:
            row["feasibility"].pop("check_seconds")
        return value

    first = without_timings(first)
    second = without_timings(second)
    assert first == second
    assert json.dumps(first, sort_keys=True, separators=(",", ":")) == json.dumps(
        second, sort_keys=True, separators=(",", ":")
    )
