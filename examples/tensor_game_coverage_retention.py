#!/usr/bin/env python3
"""TG-4J coverage-loss and frozen-policy retention investigation."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import subprocess
import time

from matmul_inspector.tensor_game_candidate_coverage import decode
from matmul_inspector.tensor_game_coverage_training import (
    CoverageDataset,
    collate_coverage,
    first_decoder_pruning_position,
    per_positive_coverage_loss,
    selective_retention_kl,
)
from matmul_inspector.tensor_game_experience import action_from_payload, action_identity
from matmul_inspector.tensor_game_learning import load_checkpoint
from matmul_inspector.tensor_game_neural_mcts import canonical_action_log_probability
from matmul_inspector.tensor_game_offline_learning import (
    OfflineTrainingConfig,
    _move_tensors,
    configure_trainable,
    sampled_support_log_probabilities,
)
from matmul_inspector.tensor_game_out_of_sample import EvaluationPolicy, generate_fresh_targets, tensor_tuple
from matmul_inspector.tensor_game_search import HeuristicValueEvaluator, SampledMCTS, SearchConfig

VERSION=1; SEEDS=(3,19,35); COVERAGE_WEIGHT=0.5; RETENTION_WEIGHT=1.0


def load_json(path): return json.loads(path.read_text())
def write_json(path,value): path.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write_csv(path,rows):
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def load_raw(source, checkpoint=None):
    import torch
    model,config,_=load_checkpoint(source,"cpu")
    if checkpoint: model.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=True))
    model.eval();return model,config


def labels(path): return [json.loads(line) for line in path.read_text().splitlines()]


def diagnose(args):
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    rows=[]; all_labels=labels(args.labels)
    models={"frozen":load_raw(args.source),"mcts_only":load_raw(args.source,args.mcts),"set_coverage":load_raw(args.source,args.coverage)}
    import torch
    for label in [x for x in all_labels if x["split"]=="validation"]:
        residual=tensor_tuple(label["residual"]); positive_count=len(label["positive_actions"])
        decoded={}; logits={}; scores={}
        tensor=torch.tensor(residual,dtype=torch.float32).unsqueeze(0)
        for name,(model,config) in models.items():
            with torch.inference_mode(): value,_=model(tensor)
            logits[name]=value[0]; decoded[name]=[action_identity(x[0]) for x in decode(model,config,residual,256)]
            scores[name]={p["action_id"]:canonical_action_log_probability(value[0],action_from_payload(p["action"]),config.shape) for p in label["positive_actions"]}
        mass=[math.exp(value-max(scores["set_coverage"].values())) for value in scores["set_coverage"].values()]
        concentration=max(mass)/sum(mass)
        for positive in label["positive_actions"]:
            aid=positive["action_id"]; ranks={n:(ids.index(aid)+1 if aid in ids else None) for n,ids in decoded.items()}
            pruning=first_decoder_pruning_position(logits["set_coverage"],action_from_payload(positive["action"]))
            if pruning is not None: failure="early_sequence_pruning"
            elif positive_count>1 and concentration>0.9 and (not ranks["set_coverage"] or ranks["set_coverage"]>12): failure="positive_set_competition"
            elif ranks["set_coverage"] is None or ranks["set_coverage"]>12: failure="poor_relative_ranking"
            else: failure="covered_top12"
            rows.append({"state_id":label["state_id"],"target_id":label["target_id"],"constructed_depth":label["constructed_depth"],"action_id":aid,"positive_provenance":"+".join(positive["provenance"]),"positive_count":positive_count,"set_mass_concentration":concentration,"frozen_score":scores["frozen"][aid],"mcts_score":scores["mcts_only"][aid],"set_coverage_score":scores["set_coverage"][aid],"score_change_vs_frozen":scores["set_coverage"][aid]-scores["frozen"][aid],"score_change_vs_mcts":scores["set_coverage"][aid]-scores["mcts_only"][aid],"frozen_rank_256":ranks["frozen"],"mcts_rank_256":ranks["mcts_only"],"set_coverage_rank_256":ranks["set_coverage"],"set_top12":bool(ranks["set_coverage"] and ranks["set_coverage"]<=12),"set_top256":ranks["set_coverage"] is not None,"first_pruning_position":pruning,"failure_classification":failure})
    write_csv(args.output/"diagnostics.csv",rows)
    summary={"positive_actions":len(rows),"states":30,"multiple_positive_actions":sum(r["positive_count"]>1 for r in rows),"set_top12":sum(r["set_top12"] for r in rows),"set_top256":sum(r["set_top256"] for r in rows),"classifications":dict(Counter(r["failure_classification"] for r in rows)),"mean_score_change_vs_frozen":sum(r["score_change_vs_frozen"] for r in rows)/len(rows),"mean_score_change_vs_mcts":sum(r["score_change_vs_mcts"] for r in rows)/len(rows),"interpretation":"Bounded ranks are reported only when observed. Missing top-256 actions have no assigned exact rank."}
    write_json(args.output/"diagnostic_summary.json",summary);print(json.dumps(summary,indent=2))


def prepare(args):
    if not (args.output/"diagnostics.csv").exists(): raise ValueError("diagnostics must precede preregistration")
    fresh=generate_fresh_targets(known_paths=args.known,seed=20261017,per_depth=4)
    heldout={**{k:v for k,v in fresh.items() if k!="targets"},"reserved_for":"TG-4J final evaluation; sealed before training","targets":[x for x in fresh["targets"] if x["construction_length"] in (1,2)]}
    write_json(args.output/"heldout_targets.json",heldout); (args.output/"heldout_targets.sha256").write_text(f"{sha(args.output/'heldout_targets.json')}  heldout_targets.json\n")
    objective="""# TG-4J objective specification

The reference is TG-4I positive-set mass loss. The new individual objective is
the mean canonical sequence negative log probability over every verified
positive. It prevents an easy positive from satisfying the entire set loss.

The retention term is mean token KL `KL(frozen || student)`. It applies only to
coverage-labeled root states whose frozen top-12 contains a verified positive.
Frozen logits are computed without gradients. This targets inherited useful
coverage rather than retaining every frozen output.

Both new runs use per-positive weight 0.5. The retention run adds KL weight 1.0.
These fixed weights reflect the pre-held-out diagnostic loss scale and reduce the
coverage pressure that degraded TG-4I MCTS cross-entropy. No hard negatives are
used. Empty positive batches contribute zero coverage or retention loss.
"""
    (args.output/"objective_spec.md").write_text(objective)
    prereg={"version":VERSION,"status":"frozen_before_training","created_at_utc":datetime.now(timezone.utc).isoformat(),"git_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),"existing_configurations":{"frozen":sha(args.source/"best_model_state.pt"),"mcts_only":sha(args.mcts),"set_coverage_1.0":sha(args.coverage)},"new_configurations":{"per_positive":{"coverage_weight":COVERAGE_WEIGHT,"retention_weight":0.0},"per_positive_retention":{"coverage_weight":COVERAGE_WEIGHT,"retention_weight":RETENTION_WEIGHT}},"training":{"epochs":30,"batch_size":128,"learning_rate":0.001,"optimizer":"AdamW","seed":42,"gradient_clip":1.0,"trainable":"policy head only"},"checkpoint_selection":"within each new run minimize validation MCTS CE + 0.5 per-positive NLL + configured retention KL; no cross-configuration exclusion","heldout_sha256":sha(args.output/"heldout_targets.json"),"labels_sha256":sha(args.labels),"candidate_cutoffs":[1,12,32,256],"search":{"candidate_budget":12,"simulation_budget_by_depth":{"1":16,"2":64},"node_limit":2000,"timeout":10.0,"seeds":list(SEEDS),"c_puct":1.5,"value":"nonzero_heuristic"},"final_configurations":["frozen","mcts_only","set_coverage_1.0","per_positive","per_positive_retention"],"heldout_access":"once after both new checkpoints selected"}
    write_json(args.output/"preregistration.json",prereg);(args.output/"preregistration.sha256").write_text(f"{sha(args.output/'preregistration.json')}  preregistration.json\n");print(sha(args.output/"preregistration.json"))


def useful_frozen_states(all_labels,source):
    model,config=load_raw(source);result=set()
    for label in all_labels:
        ids={action_identity(x[0]) for x in decode(model,config,tensor_tuple(label["residual"]),12)}
        if ids&{x["action_id"] for x in label["positive_actions"]}:result.add(label["state_id"])
    return result


def eval_losses(model,frozen,dataset,config,useful):
    import torch
    loader=torch.utils.data.DataLoader(dataset,batch_size=128,shuffle=False,collate_fn=collate_coverage); totals=Counter()
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            batch=_move_tensors(batch,"cpu");logits,_=model(batch["residual"]);flogits,_=frozen(batch["residual"])
            lp=sampled_support_log_probabilities(logits,batch["actions"],batch["action_mask"],config.shape);mcts=-(batch["search_policy"]*torch.where(batch["action_mask"],lp,0.0)).sum(1).mean()
            cov=logits.sum()*0
            if len(batch["coverage_indices"]):cov=per_positive_coverage_loss(logits[batch["coverage_indices"]],batch["positive_actions"],batch["positive_mask"],config.shape)
            retain=selective_retention_kl(logits,flogits.detach(),torch.tensor([s in useful for s in batch["state_id"]]))
            totals.update(batches=1,mcts=float(mcts),coverage=float(cov),retention=float(retain))
    return {k:totals[k]/totals["batches"] for k in ("mcts","coverage","retention")}


def train_one(name,retention_weight,args,all_labels,useful):
    import torch
    config=OfflineTrainingConfig(mode="policy_only",epochs=30,batch_size=128,learning_rate=.001,policy_coefficient=1,value_coefficient=1,gradient_clip=1,patience=30,seed=42,device="cpu",strict_torso_freeze=True)
    model,mc,_=load_checkpoint(args.source,"cpu");frozen,_,_=load_checkpoint(args.source,"cpu");frozen.requires_grad_(False);frozen.eval()
    train=CoverageDataset(args.experience,"train",all_labels);val=CoverageDataset(args.experience,"validation",all_labels)
    random.seed(42);torch.manual_seed(42);torch.use_deterministic_algorithms(True);trainable=configure_trainable(model,"policy_only",True)
    loader=torch.utils.data.DataLoader(train,batch_size=128,shuffle=True,generator=torch.Generator().manual_seed(42),collate_fn=collate_coverage);opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=.001)
    directory=args.output/"checkpoints"/name;directory.mkdir(parents=True);checkpoint=directory/"best_model_state.pt";best=math.inf;best_epoch=None;history=[];started=time.perf_counter()
    for epoch in range(1,31):
        model.train(); totals=Counter()
        for batch in loader:
            batch=_move_tensors(batch,"cpu");opt.zero_grad(set_to_none=True);logits,_=model(batch["residual"])
            with torch.no_grad():flogits,_=frozen(batch["residual"])
            lp=sampled_support_log_probabilities(logits,batch["actions"],batch["action_mask"],mc.shape);mcts=-(batch["search_policy"]*torch.where(batch["action_mask"],lp,0.)).sum(1).mean();cov=logits.sum()*0
            if len(batch["coverage_indices"]):cov=per_positive_coverage_loss(logits[batch["coverage_indices"]],batch["positive_actions"],batch["positive_mask"],mc.shape)
            retain=selective_retention_kl(logits,flogits.detach(),torch.tensor([s in useful for s in batch["state_id"]]));loss=mcts+COVERAGE_WEIGHT*cov+retention_weight*retain
            if not torch.isfinite(loss):raise RuntimeError("nonfinite loss")
            loss.backward();grad=torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),1.0);opt.step();n=len(logits);totals.update(examples=n,total=float(loss.detach())*n,mcts=float(mcts.detach())*n,coverage=float(cov.detach())*n,retention=float(retain.detach())*n,gradient=float(grad)*n)
        vm=eval_losses(model,frozen,val,mc,useful);objective=vm["mcts"]+COVERAGE_WEIGHT*vm["coverage"]+retention_weight*vm["retention"]
        history.append({"configuration":name,"epoch":epoch,"train_total_loss":totals["total"]/totals["examples"],"train_mcts_loss":totals["mcts"]/totals["examples"],"train_coverage_loss":totals["coverage"]/totals["examples"],"train_retention_loss":totals["retention"]/totals["examples"],"gradient_norm":totals["gradient"]/totals["examples"],"validation_mcts_cross_entropy":vm["mcts"],"validation_per_positive_loss":vm["coverage"],"validation_retention_loss":vm["retention"],"validation_objective":objective})
        if objective<best:best=objective;best_epoch=epoch;torch.save(model.state_dict(),checkpoint)
    manifest={"version":VERSION,"configuration":name,"coverage_weight":COVERAGE_WEIGHT,"retention_weight":retention_weight,"source_checkpoint_sha256":sha(args.source/"best_model_state.pt"),"best_epoch":best_epoch,"best_validation_objective":best,"checkpoint_sha256":sha(checkpoint),"checkpoint":"best_model_state.pt","fixed_epochs_completed":30,"training_seconds":time.perf_counter()-started,"frozen_parameters_received_gradients":False,"useful_retention_states":len(useful),"trainable_parameters":trainable,"environment":{"python":platform.python_version(),"torch":torch.__version__,"device":"cpu"}}
    write_json(directory/"experiment.json",manifest);return manifest,history


def recall(models,all_labels):
    rows=[]
    for label in [x for x in all_labels if x["split"]=="validation"]:
        positive={x["action_id"] for x in label["positive_actions"]};res=tensor_tuple(label["residual"])
        for name,(model,config) in models.items():
            for cutoff in (1,12,32,256):
                started=time.perf_counter();ids=[action_identity(x[0]) for x in decode(model,config,res,cutoff)];elapsed=time.perf_counter()-started
                rows.append({"state_id":label["state_id"],"target_id":label["target_id"],"constructed_depth":label["constructed_depth"],"configuration":name,"cutoff":cutoff,"positive_count":len(positive),"positive_recalled":len(positive&set(ids)),"states_with_any":bool(positive&set(ids)),"generation_seconds":elapsed})
    return rows


def train(args):
    prereg=load_json(args.output/"preregistration.json");all_labels=labels(args.labels)
    if sha(args.output/"preregistration.json")!=(args.output/"preregistration.sha256").read_text().split()[0] or sha(args.labels)!=prereg["labels_sha256"]:raise ValueError("frozen input mismatch")
    useful=useful_frozen_states(all_labels,args.source);manifests={};metrics=[]
    for name,rw in (("per_positive",0.0),("per_positive_retention",RETENTION_WEIGHT)):
        manifests[name],rows=train_one(name,rw,args,all_labels,useful);metrics+=rows
    write_csv(args.output/"training_metrics.csv",metrics)
    models={"frozen":load_raw(args.source),"mcts_only":load_raw(args.source,args.mcts),"set_coverage_1.0":load_raw(args.source,args.coverage),**{n:load_raw(args.source,args.output/"checkpoints"/n/"best_model_state.pt") for n in manifests}}
    vr=recall(models,all_labels);write_csv(args.output/"validation_recall.csv",vr)
    write_json(args.output/"checkpoint_manifest.json",{"existing":{"frozen":sha(args.source/"best_model_state.pt"),"mcts_only":sha(args.mcts),"set_coverage_1.0":sha(args.coverage)},"new":manifests,"heldout_accessed":False,"retention_state_rule":"labeled roots with frozen top12 positive","retention_states":len(useful)});print(len(useful))


def search(targets,models):
    rows=[]
    for t in targets:
        depth=t["construction_length"];tensor=tensor_tuple(t["target"])
        for seed in SEEDS:
            for name,(model,config) in models.items():
                policy=EvaluationPolicy(model,config,model,config,"frozen_top12");result=SampledMCTS(tensor,policy,SearchConfig(simulations={1:16,2:64}[depth],max_actions=12,max_moves=depth,seed=seed,c_puct=1.5,node_limit=2000,max_wall_time_seconds=10),HeuristicValueEvaluator()).run()
                rows.append({"target_id":t["target_id"],"constructed_depth":depth,"configuration":name,"seed":seed,"success":result.exact_reconstruction,"simulations":result.simulations_run,"simulations_to_first_success":result.simulations_to_first_success,"expansions":result.expanded_nodes,"candidate_evaluations":result.accepted_proposals,"proposal_seconds":result.proposal_generation_seconds,"search_seconds":result.search_wall_seconds,"termination_reason":result.termination_reason})
    return rows


def evaluate(args):
    prereg=load_json(args.output/"preregistration.json");cm=load_json(args.output/"checkpoint_manifest.json")
    if cm["heldout_accessed"]:raise ValueError("heldout already accessed")
    held=load_json(args.output/"heldout_targets.json")
    if sha(args.output/"heldout_targets.json")!=prereg["heldout_sha256"]:raise ValueError("heldout hash mismatch")
    models={"frozen":load_raw(args.source),"mcts_only":load_raw(args.source,args.mcts),"set_coverage_1.0":load_raw(args.source,args.coverage),"per_positive":load_raw(args.source,args.output/"checkpoints/per_positive/best_model_state.pt"),"per_positive_retention":load_raw(args.source,args.output/"checkpoints/per_positive_retention/best_model_state.pt")}
    results=search(held["targets"],models);write_csv(args.output/"discovery_results.csv",results);pairs=[];index={(r["target_id"],r["seed"],r["configuration"]):r for r in results}
    for base in ("frozen","mcts_only","set_coverage_1.0"):
        for cand in ("per_positive","per_positive_retention"):
            for t in held["targets"]:
                for seed in SEEDS:
                    a=index[t["target_id"],seed,base]["success"];b=index[t["target_id"],seed,cand]["success"]
                    pairs.append({"target_id":t["target_id"],"constructed_depth":t["construction_length"],"seed":seed,"baseline":base,"candidate":cand,"baseline_success":a,"candidate_success":b,"transition":"shared_success" if a and b else "regression" if a else "improvement" if b else "shared_failure"})
    write_csv(args.output/"paired_outcomes.csv",pairs)
    reference=[]
    for t in held["targets"]:
        ref=action_identity(action_from_payload(t["canonical_factors"][0]));res=tensor_tuple(t["target"])
        for name,(model,config) in models.items():
            ids=[action_identity(x[0]) for x in decode(model,config,res,12)];reference.append({"target_id":t["target_id"],"constructed_depth":t["construction_length"],"configuration":name,"reference_top12":ref in ids,"reference_rank":ids.index(ref)+1 if ref in ids else None})
    summary={"results":{},"reference_recall":{},"paired":dict(Counter(f"{x['baseline']}_vs_{x['candidate']}:{x['transition']}" for x in pairs)),"heldout_opened_once":True,"preregistration_sha256":sha(args.output/"preregistration.json"),"heldout_sha256":sha(args.output/"heldout_targets.json")}
    for name in models:
        rr=[x for x in results if x["configuration"]==name];summary["results"][name]={"trial_successes":sum(x["success"] for x in rr),"trials":len(rr),"targets_solved":len({x["target_id"] for x in rr if x["success"]}),"unique_targets":8,"mean_search_seconds":sum(x["search_seconds"] for x in rr)/len(rr)};summary["reference_recall"][name]=sum(x["reference_top12"] for x in reference if x["configuration"]==name)
    cm["heldout_accessed"]=True;write_json(args.output/"checkpoint_manifest.json",cm);write_json(args.output/"summary.json",summary);print(json.dumps(summary,indent=2))


def main():
    p=argparse.ArgumentParser();s=p.add_subparsers(dest="cmd",required=True)
    for cmd in ("diagnose","prepare","train","evaluate"):
        q=s.add_parser(cmd);q.add_argument("--output",type=Path,required=True);q.add_argument("--source",type=Path,required=True);q.add_argument("--mcts",type=Path,required=True);q.add_argument("--coverage",type=Path,required=True);q.add_argument("--labels",type=Path,required=True);q.add_argument("--experience",type=Path,required=True)
        if cmd=="prepare":q.add_argument("--known",type=Path,action="append",required=True)
    a=p.parse_args();{"diagnose":diagnose,"prepare":prepare,"train":train,"evaluate":evaluate}[a.cmd](a)
if __name__=="__main__":main()
