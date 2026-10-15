"""Held-out exhaustive versus bounded residual search instrumentation for TG-4R."""

from __future__ import annotations

from dataclasses import dataclass,field
import hashlib,json,random,time
from typing import Any

from .tensor_game import subtract_action,zero_tensor
from .tensor_game_bounded_residual import bounded_residual_candidates,bounded_residual_candidates_memory_efficient
from .tensor_game_experience import action_from_payload,action_identity,tensor_digest
from .tensor_game_out_of_sample import tensor_tuple
from .tensor_game_residual_supervision import enumerate_residual_teacher,squared_norm
from .tensor_game_search import HeuristicValueEvaluator,InstrumentedPolicy,ProposalMetrics,ProposedAction,SampledMCTS,SearchConfig,canonicalize_action


@dataclass
class AuditedResidualPolicy(InstrumentedPolicy):
    mode:str
    pair_budget:int=128
    metrics:ProposalMetrics=field(default_factory=ProposalMetrics,init=False)
    trace:list[dict[str,Any]]=field(default_factory=list,init=False)
    name:str=field(init=False)
    fully_evaluated_pairs:int=field(default=0,init=False)
    enumerated_pairs:int=field(default=0,init=False)

    def __post_init__(self):
        if self.mode not in {"exhaustive","bounded_128","bounded_128_memory","bounded_128_integrated"}:raise ValueError("unsupported provider")
        if self.mode.startswith("bounded_128") and self.pair_budget!=128:raise ValueError("bounded providers freeze B=128")
        self.name=self.mode
    def reset_metrics(self):
        self.metrics=ProposalMetrics();self.trace=[];self.fully_evaluated_pairs=0;self.enumerated_pairs=0
    def propose_actions(self,residual,remaining_moves,rng,max_actions):
        del rng;started=time.perf_counter()
        if self.mode=="exhaustive":rows,audit=enumerate_residual_teacher(residual,max_actions);fully=audit["enumerated_uv_pairs"]
        elif self.mode=="bounded_128":rows,audit=bounded_residual_candidates(residual,self.pair_budget,max_actions);fully=audit["fully_evaluated_pairs"]
        else:rows,audit=bounded_residual_candidates_memory_efficient(residual,self.pair_budget,max_actions);fully=audit["fully_evaluated_pairs"]
        actions=[canonicalize_action(action_from_payload(row["action"])) for row in rows];elapsed=time.perf_counter()-started
        self.enumerated_pairs+=audit["enumerated_uv_pairs"];self.fully_evaluated_pairs+=fully
        self.metrics.attempted+=fully;self.metrics.valid+=audit["unique_legal_candidates"];self.metrics.accepted+=len(actions);self.metrics.generation_seconds+=elapsed
        self.trace.append({"state_digest":tensor_digest(residual),"remaining_moves":remaining_moves,"call_index":len(self.trace),"candidate_ids":[action_identity(a) for a in actions],"candidate_scores":[row["score"] for row in rows],"candidate_count":len(actions),"proposal_seconds":elapsed,"enumerated_pairs":audit["enumerated_uv_pairs"],"fully_evaluated_pairs":fully})
        return tuple(ProposedAction(a,1/len(actions),(self.mode,)) for a in actions) if actions else ()


def run_trial(target:dict[str,Any],provider_name:str,seed:int,spec:dict[str,Any]):
    policy=AuditedResidualPolicy(provider_name);depth=target["construction_length"];tensor=tensor_tuple(target["target"])
    result=SampledMCTS(tensor,policy,SearchConfig(simulations=spec["simulation_budget_by_depth"][str(depth)],max_actions=spec["candidate_budget"],max_moves=depth,seed=seed,c_puct=spec["c_puct"],node_limit=spec["node_limit"],max_wall_time_seconds=spec["timeout_seconds"]),HeuristicValueEvaluator()).run()
    reference=action_identity(action_from_payload(target["canonical_factors"][0]));root={action_identity(x.action):x for x in result.root_actions};stat=root.get(reference)
    if result.exact_reconstruction:failure="success"
    elif stat is None:failure="initial_reference_action_missing"
    elif stat.visits==0:failure="initial_reference_action_available_not_selected"
    elif result.termination_reason=="wall_time_limit_reached":failure="wall_clock_timeout"
    elif result.termination_reason=="node_limit_reached":failure="node_limit_reached"
    else:failure="simulation_budget_exhausted_after_initial_reference_exploration"
    row={"target_id":target["target_id"],"target_digest":target["target_digest"],"constructed_depth":depth,"provider":provider_name,"seed":seed,"success":result.exact_reconstruction,"simulations_budget":spec["simulation_budget_by_depth"][str(depth)],"simulations_completed":result.simulations_run,"simulations_to_first_success":result.simulations_to_first_success,"time_to_first_success_seconds":"unavailable","expansions_to_first_success":"unavailable","expanded_nodes":result.expanded_nodes,"unique_states":result.unique_states,"candidate_generation_calls":len(policy.trace),"candidate_evaluations":result.accepted_proposals,"proposal_generation_seconds":result.proposal_generation_seconds,"mcts_excluding_proposals_seconds":result.search_wall_seconds-result.proposal_generation_seconds,"search_wall_seconds":result.search_wall_seconds,"enumerated_pairs":policy.enumerated_pairs,"fully_evaluated_pairs":policy.fully_evaluated_pairs,"termination_reason":result.termination_reason,"initial_reference_included":stat is not None,"initial_reference_visits":stat.visits if stat else 0,"failure_classification":failure,"final_residual_norm_squared":0 if result.exact_reconstruction else "unavailable"}
    costs=[{"target_id":target["target_id"],"seed":seed,"provider":provider_name,**x} for x in policy.trace]
    return row,costs


def divergence_rows(exhaustive:list[dict[str,Any]],bounded:list[dict[str,Any]])->list[dict[str,Any]]:
    left={(x["state_digest"],x["remaining_moves"]):x for x in exhaustive};right={(x["state_digest"],x["remaining_moves"]):x for x in bounded};rows=[]
    for key in sorted(set(left)&set(right)):
        a,b=left[key],right[key];aset,bset=set(a["candidate_ids"]),set(b["candidate_ids"])
        amap=dict(zip(a["candidate_ids"],a.get("candidate_scores",[])));bmap=dict(zip(b["candidate_ids"],b.get("candidate_scores",[])));shared=aset&bset;scores_available=shared<=amap.keys() and shared<=bmap.keys()
        rows.append({"state_digest":key[0],"remaining_moves":key[1],"exhaustive_call_index":a["call_index"],"bounded_call_index":b["call_index"],"exhaustive_candidates":len(aset),"bounded_candidates":len(bset),"overlap":len(shared),"same_order":a["candidate_ids"]==b["candidate_ids"],"candidate_set_equal":aset==bset,"shared_scores_equal":all(amap[x]==bmap[x] for x in shared) if scores_available else "unavailable","minimum_exhaustive_score_gap":min((y-x for x,y in zip(sorted(amap.values()),sorted(amap.values())[1:])),default="unavailable"),"exhaustive_only":";".join(sorted(aset-bset)),"bounded_only":";".join(sorted(bset-aset))})
    return rows


def paired_outcome(exhaustive:dict[str,Any],bounded:dict[str,Any])->dict[str,Any]:
    a,b=bool(exhaustive["success"]),bool(bounded["success"])
    return {"target_id":exhaustive["target_id"],"constructed_depth":exhaustive["constructed_depth"],"seed":exhaustive["seed"],"exhaustive_success":a,"bounded_success":b,"outcome":"both_succeed" if a and b else "exhaustive_only" if a else "bounded_only" if b else "both_fail","search_wall_difference_seconds":float(bounded["search_wall_seconds"])-float(exhaustive["search_wall_seconds"]),"proposal_difference_seconds":float(bounded["proposal_generation_seconds"])-float(exhaustive["proposal_generation_seconds"])}
