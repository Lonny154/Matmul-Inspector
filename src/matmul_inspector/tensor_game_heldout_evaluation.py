"""TG-4G frozen-protocol held-out evaluation helpers."""
from __future__ import annotations
from collections import Counter
import hashlib,json,random
from pathlib import Path
from typing import Any,Sequence
from .tensor_game import subtract_action
from .tensor_game_experience import action_from_payload,action_identity
from .tensor_game_out_of_sample import CONFIGURATIONS,EvaluationPolicy,tensor_tuple
from .tensor_game_search import HeuristicValueEvaluator,SampledMCTS,SearchConfig
from .tensor_game_search_calibration import ExhaustiveResidualPolicy

PROVIDERS=(*CONFIGURATIONS,"residual_norm_control")
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def verify(spec_path,manifest_path,spec_sha,manifest_sha):
 observed={"evaluation_spec":sha(spec_path),"heldout_targets":sha(manifest_path)}
 if observed!={"evaluation_spec":spec_sha,"heldout_targets":manifest_sha}:raise ValueError(f"frozen input hash mismatch: {observed}")
 spec=json.loads(spec_path.read_text());manifest=json.loads(manifest_path.read_text())
 counts=Counter(x["construction_length"] for x in manifest["targets"])
 if len(manifest["targets"])!=8 or counts!={1:4,2:4}:raise ValueError("held-out manifest shape mismatch")
 return spec,manifest,observed
def provider(name,models):
 if name=="residual_norm_control":return ExhaustiveResidualPolicy()
 return EvaluationPolicy(models["frozen"][0],models["frozen"][1],models["trained"][0],models["trained"][1],name)
def recall_rows(targets,models):
 rows=[]
 for target in targets:
  residual=tensor_tuple(target["target"]);factors=[action_from_payload(x) for x in target["canonical_factors"]]
  for step,reference in enumerate(factors):
   for name in PROVIDERS:
    p=provider(name,models);proposals=p.propose_actions(residual,len(factors)-step,random.Random(0),12)
    ids=[action_identity(x.action) for x in proposals];rid=action_identity(reference)
    rows.append({"target_id":target["target_id"],"constructed_depth":target["construction_length"],"reference_step":step,
     "provider":name,"reference_action_id":rid,"candidate_rank":ids.index(rid)+1 if rid in ids else None,
     "recall_top1":bool(ids and ids[0]==rid),"recall_top12":rid in ids,"candidate_count":len(ids)})
   residual=subtract_action(residual,reference)
 return rows
def run_trial(target,name,seed,spec,models):
 tensor=tensor_tuple(target["target"]);p=provider(name,models);sim=spec["simulation_budget_by_depth"][str(target["construction_length"])]
 result=SampledMCTS(tensor,p,SearchConfig(simulations=sim,max_actions=spec["candidate_budget"],max_moves=target["construction_length"],
  seed=seed,c_puct=spec["c_puct"],node_limit=spec["node_limit"],max_wall_time_seconds=spec["timeout_seconds"]),HeuristicValueEvaluator()).run()
 refs=[action_from_payload(x) for x in target["canonical_factors"]];initial=action_identity(refs[0]);root={action_identity(x.action):x for x in result.root_actions};stat=root.get(initial)
 if result.exact_reconstruction:failure="success"
 elif stat is None:failure="initial_reference_action_missing"
 elif stat.visits==0:failure="initial_reference_action_available_not_selected"
 elif result.termination_reason=="wall_time_limit_reached":failure="wall_clock_timeout"
 elif result.termination_reason=="node_limit_reached":failure="node_limit_reached"
 else:failure="simulation_budget_exhausted_after_initial_reference_exploration"
 forwards=getattr(p,"forward_passes",0);calls=result.expanded_nodes
 row={"target_id":target["target_id"],"target_digest":target["target_digest"],"constructed_depth":target["construction_length"],
  "provider":name,"seed":seed,"success":result.exact_reconstruction,"simulations_budget":sim,"simulations_completed":result.simulations_run,
  "simulations_to_first_success":result.simulations_to_first_success,"expansions_to_first_success":None,
  "expanded_nodes":result.expanded_nodes,"unique_states":result.unique_states,"candidate_generation_calls":calls,
  "candidate_evaluations":result.accepted_proposals,"model_forward_passes":forwards,
  "proposal_generation_seconds":result.proposal_generation_seconds,"search_wall_seconds":result.search_wall_seconds,
  "termination_reason":result.termination_reason,"initial_reference_included":stat is not None,
  "initial_reference_visits":stat.visits if stat else 0,"failure_classification":failure}
 attribution=[]
 if name=="alternating_hybrid12" and result.solution:
  residual=tensor
  for step,action in enumerate(result.solution):
   sources=p.attribution_by_state.get(hashlib.sha256(json.dumps(residual,sort_keys=True,separators=(",",":")).encode()).hexdigest(),{}).get(action_identity(action),())
   attribution.append({"target_id":target["target_id"],"seed":seed,"step":step,"action_id":action_identity(action),
    "source":"both" if len(sources)==2 else (sources[0]+"_only" if sources else "unknown")})
   residual=subtract_action(residual,action)
 return row,attribution
def aggregate(rows,recall):
 by_depth={}
 for depth in (1,2,"all"):
  scope=rows if depth=="all" else [x for x in rows if x["constructed_depth"]==depth]
  by_depth[str(depth)]={}
  for name in PROVIDERS:
   r=[x for x in scope if x["provider"]==name];s=[x for x in r if x["success"]]
   by_depth[str(depth)][name]={"trials":len(r),"unique_targets":len({x["target_id"] for x in r}),"successes":len(s),
    "targets_solved":len({x["target_id"] for x in s}),"mean_simulations":sum(x["simulations_completed"] for x in r)/len(r) if r else None,
    "mean_expansions":sum(x["expanded_nodes"] for x in r)/len(r) if r else None,"mean_candidate_evaluations":sum(x["candidate_evaluations"] for x in r)/len(r) if r else None,
    "mean_forwards":sum(x["model_forward_passes"] for x in r)/len(r) if r else None,"mean_search_wall":sum(x["search_wall_seconds"] for x in r)/len(r) if r else None,
    "mean_proposal_wall":sum(x["proposal_generation_seconds"] for x in r)/len(r) if r else None,"failures":dict(Counter(x["failure_classification"] for x in r))}
 recall_summary={}
 for name in PROVIDERS:
  r=[x for x in recall if x["provider"]==name];recall_summary[name]={"states":len(r),"top1":sum(x["recall_top1"] for x in r),"top12":sum(x["recall_top12"] for x in r)}
 return {"by_depth":by_depth,"reference_recall":recall_summary}
def paired(rows):
 idx={(x["target_id"],x["seed"],x["provider"]):x for x in rows};out=[]
 for a,b in (("frozen_top12","trained_top12"),("frozen_top12","alternating_hybrid12"),("frozen_top12","residual_norm_control"),("trained_top12","residual_norm_control")):
  for target,seed in sorted({(x["target_id"],x["seed"]) for x in rows}):
   left,right=idx[target,seed,a]["success"],idx[target,seed,b]["success"]
   out.append({"target_id":target,"constructed_depth":idx[target,seed,a]["constructed_depth"],"seed":seed,"baseline":a,"candidate":b,
    "baseline_success":left,"candidate_success":right,"transition":("shared_success" if left and right else "regression" if left else "improvement" if right else "shared_failure")})
 return out
