from matmul_inspector.tensor_game import RankOneAction
from matmul_inspector.tensor_game_experience import action_identity
from matmul_inspector.tensor_game_heldout_bounded_search import AuditedResidualPolicy,divergence_rows,paired_outcome
from matmul_inspector.tensor_game_search_calibration import ExhaustiveResidualPolicy


def residual():return tuple(tuple(tuple(1 if i==j==k else 0 for k in range(4)) for j in range(4)) for i in range(4))


def test_provider_legality_budget_determinism_and_exhaustive_equivalence():
    bounded=AuditedResidualPolicy("bounded_128");a=bounded.propose_actions(residual(),1,None,12);b=AuditedResidualPolicy("bounded_128").propose_actions(residual(),1,None,12)
    assert a==b and len(a)<=12 and bounded.fully_evaluated_pairs==128 and bounded.enumerated_pairs==6400
    reference=ExhaustiveResidualPolicy().propose_actions(residual(),1,None,12);audited=AuditedResidualPolicy("exhaustive").propose_actions(residual(),1,None,12)
    assert [x.action for x in reference]==[x.action for x in audited]
    assert all(x.prior>0 for x in a) and abs(sum(x.prior for x in a)-1)<1e-12


def test_provider_configuration_is_frozen():
    try:AuditedResidualPolicy("bounded_128",512)
    except ValueError:pass
    else:raise AssertionError("non-preregistered budget accepted")


def test_divergence_compares_matching_states_only():
    base={"remaining_moves":1,"candidate_count":2,"proposal_seconds":.1,"enumerated_pairs":6400,"fully_evaluated_pairs":6400}
    left=[{**base,"state_digest":"same","call_index":0,"candidate_ids":["a","b"]},{**base,"state_digest":"left","call_index":1,"candidate_ids":["c"]}]
    right=[{**base,"state_digest":"same","call_index":0,"candidate_ids":["a","c"]},{**base,"state_digest":"right","call_index":1,"candidate_ids":["b"]}]
    rows=divergence_rows(left,right)
    assert len(rows)==1 and rows[0]["overlap"]==1 and not rows[0]["candidate_set_equal"]


def test_paired_outcome_categories_and_differences():
    base={"target_id":"t","constructed_depth":1,"seed":3,"search_wall_seconds":1.0,"proposal_generation_seconds":.5}
    row=paired_outcome({**base,"success":True},{**base,"success":False,"search_wall_seconds":.8,"proposal_generation_seconds":.3})
    assert row["outcome"]=="exhaustive_only" and row["proposal_difference_seconds"]==-.2
