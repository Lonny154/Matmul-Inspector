"""Optional post-exact-score hardware-aware ranking for bounded_128."""
from __future__ import annotations
from dataclasses import dataclass,field
from fractions import Fraction
from collections import Counter
from typing import Literal
from .tensor_game import RankOneAction,Tensor
from .tensor_game_bounded_residual import action_from_payload,bounded_residual_candidates_memory_efficient,squared_norm
from .tensor_game_hardware_cost import action_cost

Strategy=Literal["weighted","pareto"]

def pareto_order(scored:list[dict])->list[dict]:
    """Return nondominated rows first, preserving baseline rank within groups."""
    for item in scored:
        item["pareto_optimal"]=not any(other["residual_reduction"]>=item["residual_reduction"] and other["preparation_additions"]<=item["preparation_additions"] and (other["residual_reduction"]>item["residual_reduction"] or other["preparation_additions"]<item["preparation_additions"]) for other in scored)
    return sorted(scored,key=lambda x:(not x["pareto_optimal"],x["baseline_rank"]))

@dataclass
class HardwareAwareProvider:
    penalty: Fraction|int|float|str=Fraction(0)
    strategy:Strategy="weighted"
    candidate_pool:int=128
    trace_limit:int=256
    trace:list[dict]=field(default_factory=list,init=False)
    depth:int|None=field(default=None,init=False)
    remaining_moves:int|None=field(default=None,init=False)
    calls:int=field(default=0,init=False)
    preparation_histogram:Counter=field(default_factory=Counter,init=False)
    reduction_histogram:Counter=field(default_factory=Counter,init=False)
    reranked_candidates:int=field(default=0,init=False)
    changed_selections:int=field(default=0,init=False)

    def __post_init__(self):
        self.penalty=Fraction(str(self.penalty))
        if self.penalty<0:raise ValueError("preparation penalty must be nonnegative")
        if self.strategy not in ("weighted","pareto"):raise ValueError("strategy must be weighted or pareto")
        if self.candidate_pool<=0 or self.trace_limit<0:raise ValueError("candidate pool must be positive and trace limit nonnegative")

    def set_context(self,depth:int,remaining_moves:int)->None:
        self.depth=depth;self.remaining_moves=remaining_moves

    def __call__(self,residual:Tensor,limit:int)->tuple[RankOneAction,...]:
        rows,_=bounded_residual_candidates_memory_efficient(residual,128,self.candidate_pool)
        residual_norm=squared_norm(residual); scored=[]
        for baseline_rank,row in enumerate(rows,1):
            action=action_from_payload(row["action"]);reduction=residual_norm-row["score"];prep=action_cost(action).preparation_additions;weighted=Fraction(reduction)-self.penalty*prep
            scored.append({"action":action,"action_id":row["action_id"],"residual_score":row["score"],"residual_reduction":reduction,"preparation_additions":prep,"weighted_score":weighted,"baseline_rank":baseline_rank})
        if self.strategy=="weighted":
            ordered=sorted(scored,key=lambda x:(-x["weighted_score"],x["baseline_rank"]))
        else:
            ordered=pareto_order(scored)
        selected=ordered[:limit]
        after={item["action_id"]:rank for rank,item in enumerate(ordered,1)}
        self.calls+=1;self.preparation_histogram.update(item["preparation_additions"] for item in scored);self.reduction_histogram.update(item["residual_reduction"] for item in scored)
        self.reranked_candidates+=sum(item["baseline_rank"]!=after[item["action_id"]] for item in scored)
        self.changed_selections+=len({item["action_id"] for item in selected}^{item["action_id"] for item in scored[:limit]})//2
        if len(self.trace)<self.trace_limit:
            self.trace.append({"depth":self.depth,"remaining_moves":self.remaining_moves,"remaining_residual_squared_norm":residual_norm,"strategy":self.strategy,"penalty":str(self.penalty),"candidate_pool_returned":len(rows),"selected_count":len(selected),"candidates":[{"action_id":item["action_id"],"residual_reduction":item["residual_reduction"],"preparation_additions":item["preparation_additions"],"weighted_score":str(item["weighted_score"]),"rank_before":item["baseline_rank"],"rank_after":after[item["action_id"]],"pareto_optimal":item.get("pareto_optimal"),"selected":item in selected} for item in scored]})
        return tuple(item["action"] for item in selected)

    def diagnostic_summary(self)->dict:
        return {"provider_calls":self.calls,"preparation_cost_histogram":dict(sorted(self.preparation_histogram.items())),"residual_reduction_histogram":dict(sorted(self.reduction_histogram.items())),"reranked_candidate_instances":self.reranked_candidates,"changed_selected_action_instances":self.changed_selections}
