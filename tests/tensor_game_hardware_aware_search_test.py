from fractions import Fraction
import json
from matmul_inspector.tensor_game import matrix_multiplication_tensor,standard_2x2_decomposition
from matmul_inspector.tensor_game_bounded_residual import bounded_residual_candidates_memory_efficient,action_from_payload
from matmul_inspector.tensor_game_candidate_dataset import validate_decomposition
from matmul_inspector.tensor_game_diversity_search import DiscoveryConfig,discover
from matmul_inspector.tensor_game_hardware_aware_search import HardwareAwareProvider,pareto_order

def test_zero_penalty_reproduces_bounded_ranking_and_default_is_unchanged():
 residual=matrix_multiplication_tensor(2,2,2);rows,_=bounded_residual_candidates_memory_efficient(residual,128,12);expected=tuple(action_from_payload(x["action"]) for x in rows)
 assert HardwareAwareProvider(0)(residual,12)==expected
 rows2,_=bounded_residual_candidates_memory_efficient(residual,128,12);assert rows==rows2

def test_weighted_penalty_and_trace_are_exact_and_deterministic():
 residual=matrix_multiplication_tensor(2,2,2);first=HardwareAwareProvider("0.5",trace_limit=1);second=HardwareAwareProvider(Fraction(1,2),trace_limit=1)
 assert first(residual,4)==second(residual,4) and first.trace==second.trace
 for row in first.trace[0]["candidates"]:
  assert Fraction(row["weighted_score"])==row["residual_reduction"]-Fraction(1,2)*row["preparation_additions"]
 assert first.diagnostic_summary()["provider_calls"]==1

def test_pareto_never_selects_dominated_ahead_of_frontier():
 rows=[{"baseline_rank":1,"residual_reduction":2,"preparation_additions":2},{"baseline_rank":2,"residual_reduction":2,"preparation_additions":1},{"baseline_rank":3,"residual_reduction":1,"preparation_additions":0},{"baseline_rank":4,"residual_reduction":1,"preparation_additions":2}]
 ordered=pareto_order(rows)
 assert [x["baseline_rank"] for x in ordered]==[2,3,1,4]
 assert [x["pareto_optimal"] for x in ordered]==[True,True,False,False]

def test_budget_failure_and_exact_success_are_handled():
 failed=discover(DiscoveryConfig(8,0,node_budget=1,beam_width=1,max_actions=1),HardwareAwareProvider(1));assert not failed["success"] and failed["budget_exhausted"]
 actions=standard_2x2_decomposition();validate_decomposition(actions)

def test_trace_serialization_is_stable():
 provider=HardwareAwareProvider("0.25",trace_limit=1);provider(matrix_multiplication_tensor(2,2,2),3);first=json.dumps(provider.trace,sort_keys=True,separators=(",",":"));second=json.dumps(provider.trace,sort_keys=True,separators=(",",":"));assert first==second
