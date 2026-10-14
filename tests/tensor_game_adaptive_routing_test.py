from matmul_inspector.tensor_game import RankOneAction
from matmul_inspector.tensor_game_adaptive_routing import adaptive_decision, aggregate_latency, frontier, merge_candidates, residual_features, select_threshold
from matmul_inspector.tensor_game_experience import action_identity
from matmul_inspector.tensor_game_search import canonicalize_action


def action(index):
    return RankOneAction((1,0,0,0),(0,1,0,0),(0,0,1 if index%2==0 else -1, index//2%2))


def residual_row(value):
    candidate=action(value)
    return {"action":{"u":list(candidate.u),"v":list(candidate.v),"w":list(candidate.w)},"action_id":action_identity(candidate),"score":value}


def test_routing_is_deterministic_and_uses_only_declared_feature():
    spec={"feature":"residual_nonzero","operator":">=","threshold":2,"provider_when_true":"residual","provider_when_false":"frozen"}
    features=residual_features([[[1,0],[0,1]]])
    assert adaptive_decision(features,spec)=="residual"
    assert adaptive_decision(features,spec)=="residual"
    assert "positive" not in features and "constructed_depth" not in features


def test_merge_canonicalizes_deduplicates_attributes_and_enforces_budget():
    shared=action(0); frozen=[(shared,-1.0),(action(1),-2.0)]
    residual=[residual_row(0),residual_row(1)]
    merged=merge_candidates(frozen,residual,2,2,budget=2)
    assert len(merged)==2
    assert merged[0].action==canonicalize_action(shared)
    assert merged[0].sources==("frozen","residual")
    assert merged[0].frozen_score==-1.0 and merged[0].residual_score==0


def test_threshold_selection_and_provider_call_semantics():
    rows=[{"residual_nonzero":1,"frozen_covered":True,"residual_covered":True},{"residual_nonzero":9,"frozen_covered":False,"residual_covered":True}]
    rule=select_threshold(rows,"residual_nonzero")
    assert adaptive_decision({"residual_nonzero":1},rule)=="frozen"
    assert adaptive_decision({"residual_nonzero":9},rule)=="residual"
    assert rule["training_residual_calls"]==1


def test_timing_and_cost_coverage_aggregation():
    values=aggregate_latency([1.0,2.0,3.0])
    assert values["mean_ms"]==2 and values["median_ms"]==2 and values["p95_ms"]==2.9
    rows=frontier([{"strategy":"slow","covered_states":1,"mean_total_ms":2.0},{"strategy":"fast","covered_states":1,"mean_total_ms":1.0}])
    assert rows[0]["pareto_dominated_by_mean"] is True
    assert rows[0]["dominated_by"]=="fast"
    assert rows[1]["pareto_dominated_by_mean"] is False
