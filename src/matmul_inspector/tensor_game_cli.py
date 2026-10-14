"""Residual-provider Tensor Game search command."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from typing import Sequence

from .tensor_game_bounded_residual import create_residual_candidate_provider
from .tensor_game_search import HeuristicValueEvaluator,SampledMCTS,SearchConfig

PROVIDERS=("exhaustive","bounded_128")

def tensor_tuple(value):return tuple(tuple(tuple(row) for row in plane) for plane in value)

def parser():
    p=argparse.ArgumentParser(prog="matmul-inspector tensor-game-search",description="Run CPU Tensor Game MCTS with an explicit residual candidate provider")
    p.add_argument("--target-json",type=Path,required=True,help="JSON target record, target list, or manifest")
    p.add_argument("--target-id",help="target ID when the input contains multiple targets")
    p.add_argument("--candidate-provider",choices=PROVIDERS,default="bounded_128")
    p.add_argument("--simulations",type=int,default=64);p.add_argument("--candidate-budget",type=int,default=12);p.add_argument("--seed",type=int,default=3);p.add_argument("--node-limit",type=int,default=2000);p.add_argument("--timeout-seconds",type=float,default=10.0);p.add_argument("--c-puct",type=float,default=1.5)
    return p

def load_target(path:Path,target_id:str|None):
    payload=json.loads(path.read_text());rows=payload.get("targets") if isinstance(payload,dict) else None
    if rows is None:row=payload
    elif target_id is None and len(rows)==1:row=rows[0]
    elif target_id is None:raise ValueError("--target-id is required for a multi-target manifest")
    else:
        matches=[x for x in rows if x.get("target_id")==target_id]
        if len(matches)!=1:raise ValueError(f"target ID {target_id!r} was not found uniquely")
        row=matches[0]
    target=row.get("target") if isinstance(row,dict) else None;depth=row.get("construction_length") if isinstance(row,dict) else None
    if target is None or not isinstance(depth,int) or depth<=0:raise ValueError("target record requires 'target' and positive 'construction_length'")
    return row,tensor_tuple(target),depth

def main(arguments:Sequence[str]|None=None)->int:
    args=parser().parse_args(arguments);record,target,depth=load_target(args.target_json,args.target_id);provider=create_residual_candidate_provider(args.candidate_provider)
    result=SampledMCTS(target,provider,SearchConfig(simulations=args.simulations,max_actions=args.candidate_budget,max_moves=depth,seed=args.seed,c_puct=args.c_puct,node_limit=args.node_limit,max_wall_time_seconds=args.timeout_seconds),HeuristicValueEvaluator()).run()
    output={"target_id":record.get("target_id"),"candidate_provider":args.candidate_provider,"provider_is_default":args.candidate_provider=="bounded_128","candidate_budget":args.candidate_budget,"seed":args.seed,"success":result.exact_reconstruction,"simulations_completed":result.simulations_run,"expanded_nodes":result.expanded_nodes,"proposal_generation_seconds":result.proposal_generation_seconds,"search_wall_seconds":result.search_wall_seconds,"termination_reason":result.termination_reason}
    print(json.dumps(output,sort_keys=True));return 0
