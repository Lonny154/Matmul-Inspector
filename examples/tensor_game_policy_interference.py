#!/usr/bin/env python3
"""Run the bounded TG-4N policy interference diagnostics."""

from __future__ import annotations

import argparse, copy, csv, hashlib, json, math, platform, subprocess
from collections import Counter, defaultdict
from pathlib import Path

from matmul_inspector.tensor_game_coverage_training import CoverageDataset, collate_coverage, coverage_loss
from matmul_inspector.tensor_game_experience import action_from_payload, action_identity
from matmul_inspector.tensor_game_final_ranking import complete_action_score, final_ranking_loss
from matmul_inspector.tensor_game_learning import load_checkpoint, rank_complete_actions
from matmul_inspector.tensor_game_offline_learning import configure_trainable, sampled_support_log_probabilities
from matmul_inspector.tensor_game_policy_interference import architecture_semantics, component_log_scores, component_ranks, counterfactual_segment_logits, flattened_gradients, gradient_cosine, state_dict_digest
from matmul_inspector.tensor_game_prefix_coverage import prefix_survival_loss, trace_positive_prefixes


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def load_jsonl(path):return [json.loads(x) for x in path.read_text().splitlines()]
def write_json(path,value):path.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")
def write_csv(path,rows):
    with path.open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def load_model(source,checkpoint=None):
    import torch
    model,config,_=load_checkpoint(source,"cpu")
    if checkpoint:model.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=True))
    model.eval();return model,config


def balanced_sample(labels,count_per_depth):
    output=[];counts=Counter()
    for row in sorted(labels,key=lambda x:x["target_id"]):
        if counts[row["constructed_depth"]]<count_per_depth:output.append(row);counts[row["constructed_depth"]]+=1
    if set(counts.values())!={count_per_depth}:raise ValueError("insufficient balanced labels")
    return output


def support_map(experience):
    rows=json.loads((experience/"decisions.json").read_text());result=defaultdict(set)
    for row in rows:result[row["state_digest"]].update(x["action_id"] for x in row["sampled_actions"])
    return result


def action_metrics(logits,config,action,action_id):
    top12=rank_complete_actions(logits,config.shape,12);top256=rank_complete_actions(logits,config.shape,256);ids12=[action_identity(x[0]) for x in top12];ids256=[action_identity(x[0]) for x in top256]
    trace=trace_positive_prefixes(logits,[action]);scores=component_log_scores(logits,action);ranks=component_ranks(logits,action);complete=float(complete_action_score(logits,action).detach());cutoff=top12[-1][1]
    return {"complete_score":complete,"u_score":scores[0],"v_score":scores[1],"w_score":scores[2],"u_rank":ranks[0],"v_rank":ranks[1],"w_rank":ranks[2],"rank256":ids256.index(action_id)+1 if action_id in ids256 else None,"top12":action_id in ids12,"first_pruning_position":trace[-1]["first_pruning_position"],"prefix_survives":trace[-1]["survives"],"cutoff_margin":complete-cutoff}


def checkpoint_rows(models,samples,support):
    import torch
    rows=[]
    for split,labels in samples.items():
        for label in labels:
            residual=torch.tensor(label["residual"],dtype=torch.float32).unsqueeze(0)
            for name,(model,config) in models.items():
                with torch.inference_mode():logits,_=model(residual)
                for p in label["positive_actions"]:
                    action=action_from_payload(p["action"]);rows.append({"split":split,"state_id":label["state_id"],"target_id":label["target_id"],"configuration":name,"action_id":p["action_id"],"mcts_supported":p["action_id"] in support.get(label["state_id"],set()),"provenance":"+".join(p["provenance"]),**action_metrics(logits[0],config,action,p["action_id"])})
    return rows


def make_batches(experience,labels):
    dataset=CoverageDataset(experience,"train",labels);wanted={x["state_id"] for x in labels};found={}
    for i in range(len(dataset)):
        item=dataset[i]
        if item["state_id"] in wanted:found.setdefault(item["state_id"],item)
    ordered=[found[x["state_id"]] for x in labels]
    return [collate_coverage(ordered[i:i+8]) for i in range(0,len(ordered),8)]


def objective_terms(model,config,batch):
    logits,_=model(batch["residual"]);lp=sampled_support_log_probabilities(logits,batch["actions"],batch["action_mask"],config.shape);mcts=-(batch["search_policy"]*lp.masked_fill(~batch["action_mask"],0)).sum(1).mean();idx=batch["coverage_indices"]
    set_loss=coverage_loss(logits[idx],batch["positive_actions"],batch["positive_mask"]);prefix=prefix_survival_loss(logits[idx],batch["positive_actions"],batch["positive_mask"]);rank=final_ranking_loss(logits[idx],batch["positive_actions"],batch["positive_mask"])
    return logits,mcts,set_loss,prefix,rank


def mean_positive_score(logits,batch):
    import torch
    values=[]
    for b in range(logits.shape[0]):
        local=[]
        for j in batch["positive_mask"][b].nonzero(as_tuple=False).flatten().tolist():
            x=tuple(int(v) for v in batch["positive_actions"][b,j]);local.append(complete_action_score(logits[b],action_from_payload({"u":x[:4],"v":x[4:8],"w":x[8:]})))
        values.append(torch.stack(local).max())
    return torch.stack(values).mean()


def single_updates(source,batches,labels_by_state,support):
    import torch
    results=[];interactions=[];frozen,config=load_model(source);initial=state_dict_digest(frozen)
    for batch_index,batch in enumerate(batches):
        batch_labels=[labels_by_state[x] for x in batch["state_id"]]
        for objective in ("mcts","mcts_set","mcts_prefix_rank"):
            model,config=load_model(source);assert state_dict_digest(model)==initial;configure_trainable(model,"policy_only",True);model.train();params=tuple(p for p in model.parameters() if p.requires_grad);optimizer=torch.optim.AdamW(params,lr=.001)
            logits,mcts,set_loss,prefix,rank=objective_terms(model,config,batch);extra=set_loss if objective=="mcts_set" else prefix+rank if objective=="mcts_prefix_rank" else mcts*0;total=mcts+extra
            g_mcts=flattened_gradients(params,mcts);g_extra=flattened_gradients(params,extra) if objective!="mcts" else g_mcts*0;g_total=flattened_gradients(params,total);positive_metric=mean_positive_score(logits,batch);g_positive=flattened_gradients(params,positive_metric)
            predicted=float(-torch.dot(g_total,g_positive));before_params=torch.cat([p.detach().flatten() for p in params]);before=[]
            for i,label in enumerate(batch_labels):
                for p in label["positive_actions"]:before.append((label,p,action_metrics(logits[i],config,action_from_payload(p["action"]),p["action_id"])))
            optimizer.zero_grad(set_to_none=True);total.backward();gradient=float(torch.nn.utils.clip_grad_norm_(params,1.0));optimizer.step();after_params=torch.cat([p.detach().flatten() for p in params]);model.eval()
            with torch.inference_mode():after_logits,_=model(batch["residual"])
            deltas=[]
            for (label,p,old),i in zip(before,[i for i,l in enumerate(batch_labels) for _ in l["positive_actions"]]):
                new=action_metrics(after_logits[i],config,action_from_payload(p["action"]),p["action_id"]);deltas.append(new["complete_score"]-old["complete_score"]);results.append({"batch_index":batch_index,"objective":objective,"split":"train","state_id":label["state_id"],"target_id":label["target_id"],"action_id":p["action_id"],"mcts_supported":p["action_id"] in support.get(label["state_id"],set()),"before_score":old["complete_score"],"after_score":new["complete_score"],"score_delta":new["complete_score"]-old["complete_score"],"before_rank256":old["rank256"],"after_rank256":new["rank256"],"before_top12":old["top12"],"after_top12":new["top12"],"before_pruning":old["first_pruning_position"],"after_pruning":new["first_pruning_position"],"before_cutoff_margin":old["cutoff_margin"],"after_cutoff_margin":new["cutoff_margin"],"cutoff_margin_delta":new["cutoff_margin"]-old["cutoff_margin"]})
            interactions.append({"batch_index":batch_index,"objective":objective,"fresh_optimizer":True,"initial_digest":initial,"mcts_gradient_norm":float(g_mcts.norm()),"coverage_gradient_norm":float(g_extra.norm()),"total_gradient_norm":float(g_total.norm()),"mcts_coverage_cosine":gradient_cosine(g_mcts,g_extra) if objective!="mcts" else None,"predicted_positive_score_direction":predicted,"actual_mean_positive_score_delta":sum(deltas)/len(deltas),"clipped_gradient_norm":gradient,"parameter_update_norm":float((after_params-before_params).norm())})
    return results,interactions


def factorization_rows(models,samples):
    import torch
    rows=[]
    for split,labels in samples.items():
        for label in labels:
            residual=torch.tensor(label["residual"],dtype=torch.float32).unsqueeze(0);positive_ids={x["action_id"] for x in label["positive_actions"]}
            for name,(model,config) in models.items():
                with torch.inference_mode():logits,_=model(residual)
                top1=rank_complete_actions(logits[0],config.shape,1)[0][0];top1_positive=action_identity(top1) in positive_ids
                for p in label["positive_actions"]:
                    action=action_from_payload(p["action"]);metrics=action_metrics(logits[0],config,action,p["action_id"]);rows.append({"split":split,"state_id":label["state_id"],"configuration":name,"action_id":p["action_id"],**metrics,"all_component_ranks_le_12":all(metrics[x]<=12 for x in ("u_rank","v_rank","w_rank")),"independent_top1_is_verified_positive":top1_positive})
    return rows


def counterfactual_rows(frozen,trained,labels):
    import torch
    rows=[]
    for label in labels:
        residual=torch.tensor(label["residual"],dtype=torch.float32).unsqueeze(0)
        with torch.inference_mode():frozen_logits,_=frozen[0](residual);trained_logits,_=trained[0](residual)
        for p in label["positive_actions"]:
            action=action_from_payload(p["action"]);baseline=float(complete_action_score(trained_logits[0],action));
            for segment in ("u","v","w","all"):
                logits=counterfactual_segment_logits(trained_logits[0],frozen_logits[0],segment);score=float(complete_action_score(logits,action));ids={action_identity(x[0]) for x in rank_complete_actions(logits,trained[1].shape,12)};rows.append({"split":"validation","state_id":label["state_id"],"target_id":label["target_id"],"action_id":p["action_id"],"segment_replaced":segment,"trained_score":baseline,"counterfactual_score":score,"score_delta":score-baseline,"counterfactual_top12":p["action_id"] in ids,"synthetic_intervention":True})
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);p.add_argument("--source",type=Path,required=True);p.add_argument("--labels",type=Path,required=True);p.add_argument("--experience",type=Path,required=True);p.add_argument("--mcts",type=Path,required=True);p.add_argument("--coverage",type=Path,required=True);p.add_argument("--broader",type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True);all_labels=load_jsonl(a.labels);train=balanced_sample([x for x in all_labels if x["split"]=="train"],3);validation=balanced_sample([x for x in all_labels if x["split"]=="validation"],3);samples={"train":train,"validation":validation};support=support_map(a.experience)
    models={"frozen":load_model(a.source),"mcts_only":load_model(a.source,a.mcts),"coverage":load_model(a.source,a.coverage),"broader_prefix_rank":load_model(a.source,a.broader)}
    architecture=architecture_semantics();architecture.update({"model_parameters":sum(p.numel() for p in models["frozen"][0].parameters()),"policy_head_parameters":sum(p.numel() for p in models["frozen"][0].policy_head.parameters()),"training_sample_ids":[x["target_id"] for x in train],"validation_sample_ids":[x["target_id"] for x in validation],"sample_rule":"first three target IDs per constructed depth","checkpoint_hashes":{"frozen":sha(a.source/"best_model_state.pt"),"mcts_only":sha(a.mcts),"coverage":sha(a.coverage),"broader_prefix_rank":sha(a.broader)},"optimizer_replay":"historical AdamW state unavailable; single steps use fresh AdamW at 0.001"});write_json(a.output/"architecture_audit.json",architecture)
    comparison=checkpoint_rows(models,samples,support);write_csv(a.output/"checkpoint_comparison.csv",comparison)
    update_labels=balanced_sample([x for x in all_labels if x["split"]=="train"],6);batches=make_batches(a.experience,update_labels);single,interactions=single_updates(a.source,batches,{x["state_id"]:x for x in update_labels},support);write_csv(a.output/"single_update_results.csv",single);write_csv(a.output/"gradient_interactions.csv",interactions)
    factor=factorization_rows(models,samples);write_csv(a.output/"factorization_analysis.csv",factor);counter=counterfactual_rows(models["frozen"],models["broader_prefix_rank"],validation);write_csv(a.output/"counterfactual_scores.csv",counter)
    summary={"git":{"branch":subprocess.check_output(["git","branch","--show-current"],text=True).strip(),"commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()},"environment":{"python":platform.python_version(),"device":"cpu"},"samples":{"checkpoint_train_states":len(train),"checkpoint_validation_states":len(validation),"single_update_states":len(update_labels)},"checkpoint_hashes":architecture["checkpoint_hashes"],"single_update_initial_restoration":all(r["initial_digest"]==interactions[0]["initial_digest"] for r in interactions),"historical_optimizer_state_available":False,"heldout_accessed":False};write_json(a.output/"summary.json",summary);print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
