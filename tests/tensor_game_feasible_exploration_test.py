import copy
import json

from matmul_inspector.tensor_game import standard_2x2_decomposition
from matmul_inspector.tensor_game_diversity_search import DiscoveryConfig, discover
from matmul_inspector.tensor_game_quota_w import ProvenancedAction, QuotaMultiWProvider


def quota_provider():
    return QuotaMultiWProvider(
        protected_baseline=11,
        alternative_slots=1,
        strategy="residual",
        trace_limit=0,
    )


def stable(result):
    value = copy.deepcopy(result)
    value.pop("wall_seconds")
    value["feasibility_gate_diagnostics"].pop("check_seconds")
    for depth in value["depth_diagnostics"]:
        depth["feasibility"].pop("check_seconds")
        depth.pop("beam_admission_seconds")
    return value


def test_quota_provider_provenance_admission_and_gate_compose():
    result = discover(
        DiscoveryConfig(
            rank=7,
            seed=3,
            node_budget=1,
            beam_width=12,
            max_actions=12,
            admission_policy="provenance_quota",
            baseline_capacity=11,
            alternative_capacity=1,
            feasibility_gate="flattening_rank",
        ),
        quota_provider(),
    )
    depth = result["depth_diagnostics"][0]
    assert depth["generated"] == {"alternative": 1, "baseline": 8}
    assert depth["admitted"] == {"alternative": 1, "baseline": 8}
    assert depth["beam_occupancy"] <= 12
    assert depth["feasibility"]["checked_children"] == 0
    assert result["nodes_expanded"] <= 1


def test_feasibility_pruning_is_counted_by_edge_provenance():
    first, second = standard_2x2_decomposition()[:2]

    class MixedProvider:
        calls = 0

        def propose_with_provenance(self, residual, limit):
            self.calls += 1
            return (
                ProvenancedAction(first, "baseline", "test"),
                ProvenancedAction(second, "alternative", "test"),
            )

    result = discover(
        DiscoveryConfig(
            rank=4,
            seed=0,
            node_budget=1,
            beam_width=2,
            max_actions=2,
            feasibility_gate="flattening_rank",
        ),
        MixedProvider(),
    )
    gate = result["depth_diagnostics"][0]["feasibility"]
    assert gate["remaining_budget"] == 3
    assert gate["checked_children"] == gate["pruned_children"] == 2
    assert gate["pruned_by_provenance"] == {"alternative": 1, "baseline": 1}
    assert result["depth_diagnostics"][0]["beam_occupancy"] == 0


def test_gate_off_quota_composition_is_exactly_equivalent_and_deterministic():
    common = dict(rank=7, seed=11, node_budget=3, beam_width=12, max_actions=12)
    default = discover(DiscoveryConfig(**common), quota_provider())
    explicit = discover(
        DiscoveryConfig(**common, feasibility_gate="off"), quota_provider()
    )
    assert stable(default) == stable(explicit)
    serialized = json.dumps(stable(default), sort_keys=True, separators=(",", ":"))
    repeated = json.dumps(stable(
        discover(DiscoveryConfig(**common), quota_provider())
    ), sort_keys=True, separators=(",", ":"))
    assert serialized == repeated


def test_duplicate_children_still_consume_one_slot_with_gate_enabled():
    action = standard_2x2_decomposition()[0]

    class DuplicateProvider:
        calls = 0

        def propose_with_provenance(self, residual, limit):
            self.calls += 1
            return (
                ProvenancedAction(action, "baseline", "one"),
                ProvenancedAction(action, "alternative", "two"),
            )

    result = discover(
        DiscoveryConfig(
            rank=8,
            seed=0,
            node_budget=1,
            beam_width=2,
            max_actions=2,
            feasibility_gate="flattening_rank",
        ),
        DuplicateProvider(),
    )
    depth = result["depth_diagnostics"][0]
    assert depth["beam_occupancy"] == 1
    assert depth["mixed_provenance_states"] == 1
    assert sum(depth["removed_by_deduplication"].values()) == 1
