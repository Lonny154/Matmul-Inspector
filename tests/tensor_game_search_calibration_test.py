from matmul_inspector.tensor_game import RankOneAction,outer_product
from matmul_inspector.tensor_game_search import canonicalize_action
from matmul_inspector.tensor_game_search_calibration import ExhaustiveResidualPolicy,ReferencePathPolicy,classify,replay_reference,run_search
def fixture():
 a=RankOneAction((1,),(1,),(1,));return a,outer_product(a)
def test_reference_replay_and_exact_residual():
 a,t=fixture();r=replay_reference({"target_id":"t","target":[[[1]]],"construction_length":1,"canonical_factors":[{"u":[1],"v":[1],"w":[1]}]});assert r["exact_success"] and r["final_residual_norm_squared"]==0
def test_reference_search_budget_and_insertion():
 a,t=fixture();p=ReferencePathPolicy(t,(a,));r=run_search(t,p,3,3,1);assert r["success"] and r["simulations_completed"]<=3 and r["reference_action_included"]
def test_failure_classification():
 assert classify({"success":False,"reference_action_included":False})=="reference_action_unavailable"
def test_residual_candidates_are_deterministic_valid_unique_and_budgeted():
 a=RankOneAction((1,0,0,0),(1,0,0,0),(1,0,0,0));t=outer_product(a);p=ExhaustiveResidualPolicy()
 import random
 one=p.propose_actions(t,1,random.Random(3),3);two=ExhaustiveResidualPolicy().propose_actions(t,1,random.Random(99),3)
 assert one==two and 0<len(one)<=3 and len({x.action for x in one})==len(one) and one[0].action==canonicalize_action(a)
