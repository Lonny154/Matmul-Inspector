"""Bounded coarse-to-fine residual candidate generation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass,field
import hashlib
import itertools
import heapq
import json
import math
import time
from typing import Any, Sequence

from .tensor_game import RankOneAction, subtract_action
from .tensor_game_search import InstrumentedPolicy,ProposalMetrics,ProposedAction,canonicalize_action,quantize_ratio

COEFFICIENTS = (-1, 0, 1)
VECTORS = tuple(vector for vector in itertools.product(COEFFICIENTS, repeat=4) if any(vector))


def squared_norm(tensor: Any) -> int:
    return sum(value * value for plane in tensor for row in plane for value in row)


def action_identity(action: RankOneAction) -> str:
    action = canonicalize_action(action)
    payload = {"u": list(action.u), "v": list(action.v), "w": list(action.w)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def action_from_payload(value: dict[str, Sequence[int]]) -> RankOneAction:
    return canonicalize_action(RankOneAction(value["u"], value["v"], value["w"]))


@dataclass(frozen=True)
class PairScore:
    u: tuple[int, ...]
    v: tuple[int, ...]
    q: int
    contraction: tuple[int, ...]
    bound_numerator: int


def exact_score_from_contraction(residual_norm: int, contraction: Sequence[int], q: int, w: Sequence[int]) -> int:
    """Compute ||R-u⊗v⊗w||² without constructing the rank-one tensor."""
    return residual_norm - 2 * sum(d * value for d, value in zip(contraction, w)) + q * sum(value * value for value in w)


def coarse_pair_scores(residual: Any) -> list[PairScore]:
    """Rank all pairs by the continuous relaxation's achievable improvement."""
    rows=[]
    for u in VECTORS:
        uu=sum(value*value for value in u)
        for v in VECTORS:
            q=uu*sum(value*value for value in v)
            contraction=tuple(sum(residual[i][j][k]*u[i]*v[j] for i in range(4) for j in range(4)) for k in range(4))
            # Improvement is sum(d_k^2)/q. All q divide 16!, so this integer
            # numerator provides an exact common-denominator comparison.
            bound_numerator=(math.factorial(16)//q)*sum(value*value for value in contraction)
            rows.append(PairScore(u,v,q,contraction,bound_numerator))
    rows.sort(key=lambda row:(-row.bound_numerator,row.u,row.v))
    return rows


def coarse_pair_scores_streaming(residual:Any,pair_budget:int=128)->list[PairScore]:
    """Return the exact coarse-ranking prefix while retaining only B records."""
    if pair_budget<=0 or pair_budget>len(VECTORS)**2:raise ValueError("pair_budget must be in [1, 6400]")
    heap=[];common=math.factorial(16)
    for u in VECTORS:
        uu=sum(value*value for value in u);inverse_u=tuple(-value for value in u)
        for v in VECTORS:
            q=uu*sum(value*value for value in v)
            contraction=tuple(sum(residual[i][j][k]*u[i]*v[j] for i in range(4) for j in range(4)) for k in range(4))
            bound=(common//q)*sum(value*value for value in contraction);inverse_v=tuple(-value for value in v)
            pair=PairScore(u,v,q,contraction,bound);entry=(bound,inverse_u,inverse_v,pair)
            if len(heap)<pair_budget:heapq.heappush(heap,entry)
            elif entry[:3]>heap[0][:3]:heapq.heapreplace(heap,entry)
    return sorted((entry[3] for entry in heap),key=lambda row:(-row.bound_numerator,row.u,row.v))


def bounded_residual_candidates(residual: Any, pair_budget: int, limit: int=12) -> tuple[list[dict[str,Any]],dict[str,Any]]:
    """Coarse-rank all pairs, then exactly evaluate a bounded prefix."""
    if pair_budget <= 0 or pair_budget > len(VECTORS)**2: raise ValueError("pair_budget must be in [1, 6400]")
    if limit <= 0: raise ValueError("limit must be positive")
    total_start=time.perf_counter_ns(); start=total_start
    pairs=coarse_pair_scores(residual); coarse_ns=time.perf_counter_ns()-start
    residual_norm=squared_norm(residual); candidates={}; start=time.perf_counter_ns()
    for pair in pairs[:pair_budget]:
        w=tuple(quantize_ratio(value,pair.q,COEFFICIENTS) for value in pair.contraction)
        if not any(w): continue
        action=canonicalize_action(RankOneAction(pair.u,pair.v,w))
        score=exact_score_from_contraction(residual_norm,pair.contraction,pair.q,w)
        candidates[action]=min(score,candidates.get(action,math.inf))
    exact_ns=time.perf_counter_ns()-start; start=time.perf_counter_ns()
    ordered=sorted(candidates.items(),key=lambda item:(item[1],item[0].u,item[0].v,item[0].w)); ties=Counter(score for _,score in ordered)
    rows=[{"action":{"u":list(a.u),"v":list(a.v),"w":list(a.w)},"action_id":action_identity(a),"score":score,"rank":rank,"score_tie_count":ties[score]} for rank,(a,score) in enumerate(ordered[:limit],1)]
    ranking_ns=time.perf_counter_ns()-start
    return rows,{"enumerated_uv_pairs":len(pairs),"fully_evaluated_pairs":pair_budget,"unique_legal_candidates":len(candidates),"returned_candidates":len(rows),"coarse_scoring_ms":coarse_ns/1e6,"exact_rescoring_ms":exact_ns/1e6,"ranking_dedup_ms":ranking_ns/1e6,"total_ms":(time.perf_counter_ns()-total_start)/1e6,"bound_semantics":"continuous-relaxation lower bound on residual score; prefix selection is heuristic"}


def bounded_residual_candidates_memory_efficient(residual:Any,pair_budget:int=128,limit:int=12,*,analyze_costs:bool=False)->tuple[list[dict[str,Any]],dict[str,Any]]:
    """Exact-output-equivalent bounded ranking with O(pair_budget) records."""
    if limit<=0:raise ValueError("limit must be positive")
    total_start=time.perf_counter_ns();start=total_start;pairs=coarse_pair_scores_streaming(residual,pair_budget);coarse_ns=time.perf_counter_ns()-start
    residual_norm=squared_norm(residual);candidates={};start=time.perf_counter_ns()
    for pair in pairs:
        w=tuple(quantize_ratio(value,pair.q,COEFFICIENTS) for value in pair.contraction)
        if not any(w):continue
        action=canonicalize_action(RankOneAction(pair.u,pair.v,w));score=exact_score_from_contraction(residual_norm,pair.contraction,pair.q,w);candidates[action]=min(score,candidates.get(action,math.inf))
    exact_ns=time.perf_counter_ns()-start;start=time.perf_counter_ns();ordered=sorted(candidates.items(),key=lambda item:(item[1],item[0].u,item[0].v,item[0].w));ties=Counter(score for _,score in ordered)
    rows=[{"action":{"u":list(a.u),"v":list(a.v),"w":list(a.w)},"action_id":action_identity(a),"score":score,"rank":rank,"score_tie_count":ties[score]} for rank,(a,score) in enumerate(ordered[:limit],1)];ranking_ns=time.perf_counter_ns()-start
    audit={"enumerated_uv_pairs":len(VECTORS)**2,"fully_evaluated_pairs":pair_budget,"retained_coarse_records":len(pairs),"unique_legal_candidates":len(candidates),"returned_candidates":len(rows),"coarse_scoring_ms":coarse_ns/1e6,"exact_rescoring_ms":exact_ns/1e6,"ranking_dedup_ms":ranking_ns/1e6,"total_ms":(time.perf_counter_ns()-total_start)/1e6,"selection":"fixed_size_heap_exact_ties","bound_semantics":"identical to bounded_residual_candidates"}
    if analyze_costs:
        from .tensor_game_hardware_cost import action_cost, as_json, candidate_objectives, pareto_frontier
        analysis_start=time.perf_counter_ns();objectives=candidate_objectives(ordered,residual_norm);frontier=pareto_frontier(objectives);frontier_indices={item.candidate_index for item in frontier}
        analysis_rows=[]
        for index,((action,score),objective) in enumerate(zip(ordered,objectives)):
            analysis_rows.append({"action_id":action_identity(action),"score":score,"residual_reduction":objective.residual_reduction,"cost":as_json(action_cost(action)),"pareto_optimal":index in frontier_indices})
        analysis_ns=time.perf_counter_ns()-analysis_start
        audit["cost_analysis"]={"candidate_count":len(analysis_rows),"pareto_frontier_size":len(frontier),"analysis_ms":analysis_ns/1e6,"objectives":{"residual_reduction":"maximize","preparation_additions":"minimize"},"candidates":analysis_rows}
        audit["total_ms"]=(time.perf_counter_ns()-total_start)/1e6
    return rows,audit


def profile_exhaustive(residual: Any, limit: int=12) -> tuple[list[dict[str,Any]],dict[str,float|int]]:
    """Component timing clone of the preserved exhaustive reference."""
    total=time.perf_counter_ns(); contraction_ns=w_ns=action_ns=score_ns=0; candidates={}
    for u in VECTORS:
        uu=sum(x*x for x in u)
        for v in VECTORS:
            q=uu*sum(x*x for x in v); start=time.perf_counter_ns()
            d=tuple(sum(residual[i][j][k]*u[i]*v[j] for i in range(4) for j in range(4)) for k in range(4)); contraction_ns+=time.perf_counter_ns()-start
            start=time.perf_counter_ns(); w=tuple(quantize_ratio(x,q,COEFFICIENTS) for x in d); w_ns+=time.perf_counter_ns()-start
            if not any(w): continue
            start=time.perf_counter_ns(); action=canonicalize_action(RankOneAction(u,v,w)); action_ns+=time.perf_counter_ns()-start
            start=time.perf_counter_ns(); score=squared_norm(subtract_action(residual,action)); score_ns+=time.perf_counter_ns()-start
            candidates[action]=min(score,candidates.get(action,math.inf))
    start=time.perf_counter_ns(); ordered=sorted(candidates.items(),key=lambda item:(item[1],item[0].u,item[0].v,item[0].w)); ranking_ns=time.perf_counter_ns()-start
    rows=[{"action":{"u":list(a.u),"v":list(a.v),"w":list(a.w)},"action_id":action_identity(a),"score":s,"rank":i} for i,(a,s) in enumerate(ordered[:limit],1)]
    return rows,{"total_ms":(time.perf_counter_ns()-total)/1e6,"contraction_ms":contraction_ns/1e6,"w_selection_ms":w_ns/1e6,"canonicalization_object_ms":action_ns/1e6,"exact_tensor_score_ms":score_ns/1e6,"ranking_dedup_ms":ranking_ns/1e6,"evaluated_pairs":6400,"unique_candidates":len(candidates)}


@dataclass
class BoundedResidualPolicy(InstrumentedPolicy):
    """MCTS policy adapter for the validated B=128 provider."""
    shape:tuple[int,int,int]=(4,4,4)
    pair_budget:int=128
    name:str=field(default="bounded_128",init=False)
    metrics:ProposalMetrics=field(default_factory=ProposalMetrics,init=False,repr=False)
    fully_evaluated_pairs:int=field(default=0,init=False)

    def __post_init__(self):
        if self.shape!=(4,4,4):raise ValueError("bounded_128 currently supports shape (4, 4, 4)")
        if self.pair_budget!=128:raise ValueError("bounded_128 has a frozen pair budget of 128")

    def reset_metrics(self):
        super().reset_metrics();self.fully_evaluated_pairs=0

    def propose_actions(self,residual,remaining_moves,rng,max_actions):
        del rng
        if remaining_moves<=0 or max_actions<=0:return ()
        started=time.perf_counter();rows,audit=bounded_residual_candidates_memory_efficient(residual,self.pair_budget,max_actions)
        actions=[canonicalize_action(action_from_payload(row["action"])) for row in rows]
        self.fully_evaluated_pairs+=audit["fully_evaluated_pairs"]
        self.metrics.attempted+=audit["fully_evaluated_pairs"];self.metrics.valid+=audit["unique_legal_candidates"];self.metrics.accepted+=len(actions);self.metrics.generation_seconds+=time.perf_counter()-started
        return tuple(ProposedAction(action,1/len(actions),(self.name,)) for action in actions) if actions else ()


@dataclass
class ExhaustiveResidualPolicy(InstrumentedPolicy):
    """Exact residual policy retained as an explicit production fallback."""

    shape: tuple[int,int,int]=(4,4,4)
    name: str=field(default="exhaustive_residual",init=False)
    metrics: ProposalMetrics=field(default_factory=ProposalMetrics,init=False)

    def __post_init__(self):
        if self.shape != (4, 4, 4):
            raise ValueError("exhaustive currently supports shape (4, 4, 4)")

    def propose_actions(self,residual,remaining_moves,rng,max_actions):
        del rng
        if remaining_moves <= 0 or max_actions <= 0:
            return ()
        started=time.perf_counter();candidates={}
        for u in VECTORS:
            uu=sum(x*x for x in u)
            for v in VECTORS:
                denominator=uu*sum(x*x for x in v)
                contraction=tuple(sum(residual[i][j][k]*u[i]*v[j] for i in range(4) for j in range(4)) for k in range(4))
                w=tuple(quantize_ratio(x,denominator,COEFFICIENTS) for x in contraction)
                if not any(w):continue
                action=canonicalize_action(RankOneAction(u,v,w));score=squared_norm(subtract_action(residual,action))
                candidates[action]=min(score,candidates.get(action,math.inf))
        ranked=sorted(candidates,key=lambda action:(candidates[action],action.u,action.v,action.w))[:max_actions]
        self.metrics.attempted+=len(VECTORS)**2;self.metrics.valid+=len(candidates);self.metrics.accepted+=len(ranked)
        self.metrics.generation_seconds+=time.perf_counter()-started
        return tuple(ProposedAction(action,1/len(ranked),("residual_norm",)) for action in ranked) if ranked else ()


def create_residual_candidate_provider(identifier:str="bounded_128",shape:tuple[int,int,int]=(4,4,4)):
    """Create a residual provider; bounded_128 is the public default."""
    if identifier=="exhaustive":
        return ExhaustiveResidualPolicy(shape)
    if identifier=="bounded_128":return BoundedResidualPolicy(shape)
    raise ValueError(f"unsupported candidate provider {identifier!r}; expected 'exhaustive' or 'bounded_128'")
