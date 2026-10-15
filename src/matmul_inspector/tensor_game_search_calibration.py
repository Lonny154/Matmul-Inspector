"""TG-4F experiment-only search calibration and oracle controls."""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
import itertools, math, time
from typing import Any
from .tensor_game import RankOneAction, TensorGame, outer_product, subtract_action, tensor_shape, zero_tensor
from .tensor_game_experience import action_from_payload, action_identity, tensor_digest
from .tensor_game_search import (HeuristicValueEvaluator, InstrumentedPolicy, ProposalMetrics,
    ProposedAction, SampledMCTS, SearchConfig, canonicalize_action, quantize_ratio)
from .tensor_game_bounded_residual import ExhaustiveResidualPolicy

def squared_norm(tensor): return sum(x*x for plane in tensor for row in plane for x in row)

def replay_reference(record):
    target=tuple(tuple(tuple(row) for row in plane) for plane in record["target"])
    game=TensorGame(target,record["construction_length"],frozenset((-1,0,1)))
    error=None
    try:
        for value in record["canonical_factors"]: game.step(action_from_payload(value))
    except ValueError as exc: error=str(exc)
    return {"target_id":record["target_id"],"constructed_depth":record["construction_length"],
            "reference_length":len(record["canonical_factors"]),"reference_replay_valid":error is None,
            "final_residual_digest":tensor_digest(game.residual),"final_residual_norm_squared":squared_norm(game.residual),
            "exact_success":game.terminated and game.reconstructs_target(),"failure_reason":error}

@dataclass
class ReferencePathPolicy(InstrumentedPolicy):
    target: Any; factors: tuple[RankOneAction,...]; distractor: Any|None=None; augmented: bool=False
    name: str=field(init=False); metrics: ProposalMetrics=field(default_factory=ProposalMetrics,init=False)
    def __post_init__(self):
        self.name="reference_augmented" if self.augmented else "reference_only"; self.by_state={}
        residual=self.target
        for action in self.factors: self.by_state[residual]=action; residual=subtract_action(residual,action)
    def propose_actions(self,residual,remaining_moves,rng,max_actions):
        started=time.perf_counter(); ref=self.by_state.get(residual); proposals=[]
        if self.augmented and self.distractor:
            proposals=list(self.distractor.propose_actions(residual,remaining_moves,rng,max_actions))
        actions=[]
        if proposals: actions=[item.action for item in proposals]
        if ref is not None and ref not in actions:
            actions=(actions[:max_actions-1]+[ref]) if max_actions>1 else [ref]
        elif ref is not None and not actions: actions=[ref]
        actions=list(dict.fromkeys(canonicalize_action(a) for a in actions))[:max_actions]
        self.metrics.attempted+=len(actions); self.metrics.valid+=len(actions); self.metrics.accepted+=len(actions)
        self.metrics.generation_seconds+=time.perf_counter()-started
        return tuple(ProposedAction(a,1/len(actions),("reference",) if a==ref else ("frozen",)) for a in actions) if actions else ()

def run_search(target,policy,simulations,seed,max_moves,node_limit=2000,timeout=10.0):
    result=SampledMCTS(target,policy,SearchConfig(simulations=simulations,max_actions=12,max_moves=max_moves,
        seed=seed,c_puct=1.5,node_limit=node_limit,max_wall_time_seconds=timeout),HeuristicValueEvaluator()).run()
    ref=getattr(policy,"by_state",{}).get(target); root={action_identity(x.action):x for x in result.root_actions}; rr=root.get(action_identity(ref)) if ref else None
    return {"success":result.exact_reconstruction,"simulations_budget":simulations,"simulations_completed":result.simulations_run,
        "simulations_to_first_success":result.simulations_to_first_success,"expanded_nodes":result.expanded_nodes,
        "unique_states":result.unique_states,"candidate_evaluations":result.accepted_proposals,
        "search_wall_seconds":result.search_wall_seconds,"termination_reason":result.termination_reason,
        "reference_action_included":rr is not None,"reference_action_visits":rr.visits if rr else 0}

def classify(row):
    if row["success"]: return "success"
    if not row.get("reference_action_included",True): return "reference_action_unavailable"
    if row.get("reference_action_visits",0)==0: return "reference_action_available_not_selected"
    if row["termination_reason"]=="wall_time_limit_reached": return "wall_clock_timeout"
    if row["termination_reason"]=="node_limit_reached": return "expansion_limit_reached"
    return "simulation_budget_exhausted"
