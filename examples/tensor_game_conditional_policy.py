#!/usr/bin/env python3
"""Run the bounded TG-4O conditional-policy comparison."""

from __future__ import annotations

import argparse,copy,csv,hashlib,json,platform,resource,subprocess,time,tracemalloc
from collections import Counter
from pathlib import Path

from matmul_inspector.tensor_game_conditional_policy import conditional_policy_loss,conditional_sequence_scores,create_conditional_policy,rank_conditional_actions
from matmul_inspector.tensor_game_coverage_training import CoverageDataset,canonical_action_log_probabilities,collate_coverage,coverage_loss
from matmul_inspector.tensor_game_experience import action_from_payload,action_identity
from matmul_inspector.tensor_game_learning import load_checkpoint,rank_complete_actions,rank_complete_actions_audited
from matmul_inspector.tensor_game_offline_learning import configure_trainable,sampled_support_log_probabilities
from matmul_inspector.tensor_game_prefix_coverage import sign_forms,trace_positive_prefixes


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def load_jsonl(path):return [json.loads(x) for x in path.read_text().splitlines()]
def write_json(path,value):path.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")
def write_csv(path,rows):
    with path.open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def frozen(source):return load_checkpoint(source,"cpu")[:2]


def batches(experience,labels,split="train"):
    dataset=CoverageDataset(experience,split,labels);wanted={x["state_id"] for x in labels};found={}
    for i in range(len(dataset)):
        item=dataset[i]
        if item["state_id"] in wanted:found.setdefault(item["state_id"],item)
    ordered=[found[x["state_id"]] for x in labels]
    return [collate_coverage(ordered[i:i+8]) for i in range(0,len(ordered),8)]


def independent_losses(model,config,batch):
    logits,_=model(batch["residual"]);lp=sampled_support_log_probabilities(logits,batch["actions"],batch["action_mask"],config.shape);mcts=-(batch["search_policy"]*lp.masked_fill(~batch["action_mask"],0)).sum(1).mean();idx=batch["coverage_indices"];coverage=coverage_loss(logits[idx],batch["positive_actions"],batch["positive_mask"]);return mcts,coverage,mcts+coverage


def conditional_losses(model,batch):
    mcts=conditional_policy_loss(model,batch["residual"],batch["actions"],batch["action_mask"],batch["search_policy"]);idx=batch["coverage_indices"];coverage=conditional_policy_loss(model,batch["residual"][idx],batch["positive_actions"],batch["positive_mask"]);return mcts,coverage,mcts+coverage


def decode_independent(model,config,label):
    import torch
    residual=torch.tensor(label["residual"],dtype=torch.float32);started=time.perf_counter()
    with torch.inference_mode():logits,_=model(residual[None])
    ranked,audit=rank_complete_actions_audited(logits[0],config.shape,12);return ranked,audit,logits[0],time.perf_counter()-started


def decode_conditional(model,config,label):
    import torch
    residual=torch.tensor(label["residual"],dtype=torch.float32);started=time.perf_counter();ranked,audit=rank_conditional_actions(model,residual,config,12,include_beam=True);elapsed=time.perf_counter()-started;return ranked,audit,residual,elapsed


def evaluate(models,labels,split):
    import torch
    recall=[];latency=[];factor=[]
    for label in labels:
        positives={x["action_id"] for x in label["positive_actions"]};reference=next((x["action_id"] for x in label["positive_actions"] if "validated_construction_continuation" in x["provenance"]),None)
        for name,(kind,model,config) in models.items():
            if kind=="independent":ranked,audit,logits,elapsed=decode_independent(model,config,label);beam=None
            else:ranked,audit,residual,elapsed=decode_conditional(model,config,label);beam=audit.pop("final_beam")
            ids=[action_identity(x[0]) for x in ranked];hit=positives&set(ids);sequences=[a.u+a.v+a.w for a,_ in ranked];pairs=[sum(x!=y for x,y in zip(a,b)) for i,a in enumerate(sequences) for b in sequences[i+1:]]
            recall.append({"split":split,"state_id":label["state_id"],"target_id":label["target_id"],"configuration":name,"positive_recalled":len(hit),"positive_count":len(positives),"any_positive_recalled":bool(hit),"reference_recalled":reference in ids,"candidate_count":len(ids),"mean_pairwise_token_distance":sum(pairs)/len(pairs) if pairs else 0})
            latency.append({"split":split,"state_id":label["state_id"],"configuration":name,"generation_seconds":elapsed,"conditional_forward_evaluations":audit.get("conditional_forward_evaluations",1),"beam_expansions":audit["internal_candidates_considered"] if kind=="independent" else audit["beam_expansions"],"decoded_candidates":audit["decoded_candidates"],"invalid_rejected":audit["invalid_rejected"],"canonical_duplicates_rejected":audit["canonical_duplicates_rejected"],"candidate_count":len(ids)})
            for p in label["positive_actions"]:
                action=action_from_payload(p["action"])
                if kind=="independent":
                    score=float(canonical_action_log_probabilities(logits[None],torch.tensor([[action.u+action.v+action.w]]))[0][0]);trace=trace_positive_prefixes(logits,[action]);survive=trace[-1]["survives"]
                else:
                    forms=torch.tensor(sign_forms(action));residuals=residual[None].expand(len(forms),-1,-1,-1);score=float(torch.logsumexp(conditional_sequence_scores(model,residuals,forms[:,None,:])[:,0],0).detach());beam_prefix={x for x,_ in beam};survive=any(form in beam_prefix for form in sign_forms(action))
                factor.append({"split":split,"state_id":label["state_id"],"target_id":label["target_id"],"configuration":name,"action_id":p["action_id"],"complete_score":score,"width384_survives":survive,"top12":p["action_id"] in ids,"reference_action":p["action_id"]==reference})
    return recall,latency,factor


def recalls_only(models,labels):
    return {name:sum(r["positive_recalled"] for r in evaluate({name:value},labels,"probe")[0]) for name,value in models.items()}


def train_pair(source,config,train_batches,training,validation,output):
    import torch
    base,_,_=load_checkpoint(source,"cpu");independent=base;configure_trainable(independent,"policy_only",True);conditional=create_conditional_policy(base,config)
    for p in conditional.features.parameters():p.requires_grad=False
    for p in conditional.value_head.parameters():p.requires_grad=False
    optimizers={"independent_control":torch.optim.AdamW((p for p in independent.parameters() if p.requires_grad),lr=.001),"conditional":torch.optim.AdamW((p for p in conditional.parameters() if p.requires_grad),lr=.001)}
    history=[];runs={};model_seconds={"independent_control":0.0,"conditional":0.0};torch.manual_seed(42);torch.use_deterministic_algorithms(True);tracemalloc.start();started=time.perf_counter()
    for update in range(1,101):
        batch=train_batches[(update-1)%len(train_batches)]
        for name,model in (("independent_control",independent),("conditional",conditional)):
            model_started=time.perf_counter()
            optimizer=optimizers[name];optimizer.zero_grad(set_to_none=True);mcts,coverage,total=independent_losses(model,config,batch) if name=="independent_control" else conditional_losses(model,batch);total.backward();gradient=torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),1.0);optimizer.step()
            model_seconds[name]+=time.perf_counter()-model_started
            if update==1 or update%20==0:history.append({"configuration":name,"update":update,"batch_index":(update-1)%len(train_batches),"mcts_loss":float(mcts.detach()),"coverage_loss":float(coverage.detach()),"total_loss":float(total.detach()),"gradient_norm":float(gradient),"elapsed_seconds":time.perf_counter()-started})
        if time.perf_counter()-started>240:raise RuntimeError("preregistered wall budget exceeded")
    _,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
    for name,model in (("independent_control",independent),("conditional",conditional)):
        directory=output/"checkpoints"/name;directory.mkdir(parents=True);path=directory/"model_state.pt";torch.save(model.state_dict(),path);runs[name]={"checkpoint_sha256":sha(path),"parameters":sum(p.numel() for p in model.parameters()),"trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),"training_seconds":model_seconds[name]}
    runs["shared"]={"runtime_seconds":time.perf_counter()-started,"python_peak_bytes":peak,"process_peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss};return independent,conditional,history,runs


def validation_ce(kind,model,config,experience,labels):
    import torch
    dataset=CoverageDataset(experience,"validation",labels);loader=torch.utils.data.DataLoader(dataset,batch_size=32,collate_fn=collate_coverage);total=count=0
    for batch in loader:
        with torch.inference_mode():loss=independent_losses(model,config,batch)[0] if kind=="independent" else conditional_losses(model,batch)[0]
        total+=float(loss)*len(batch["residual"]);count+=len(batch["residual"])
    return total/count


def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);p.add_argument("--source",type=Path,required=True);p.add_argument("--training-labels",type=Path,required=True);p.add_argument("--all-labels",type=Path,required=True);p.add_argument("--experience",type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True);training=load_jsonl(a.training_labels);all_labels=load_jsonl(a.all_labels);validation=[x for x in all_labels if x["split"]=="validation"];base,config=frozen(a.source);prototype=create_conditional_policy(base,config)
    spec={"version":1,"factorization":"p(u|R) p(v|u,R) p(w|u,v,R)","conditioning":"zero-initialized linear adapters over coefficient one-hot contexts","initial_equivalence":"base u/v/w logits copied exactly from frozen independent policy","encoder":"copied frozen residual MLP","coefficient_vocabulary":[-1,0,1],"conditional_parameters":sum(p.numel() for p in prototype.parameters()),"independent_parameters":sum(p.numel() for p in base.parameters()),"parameter_difference":sum(p.numel() for p in prototype.parameters())-sum(p.numel() for p in base.parameters()),"correctness_overfit":"targeted unit test lowers one-action positive-set loss in 10 updates","production_code_modified":False};write_json(a.output/"architecture_spec.json",spec)
    prereg={"version":1,"created_before_training":True,"training_labels_sha256":sha(a.training_labels),"training_states":32,"validation_states":30,"models":["frozen_independent","matched_independent_control","conditional"],"objective":"sampled MCTS cross-entropy + verified positive-set NLL","optimizer":"AdamW","learning_rate":.001,"batch_size":8,"updates":100,"seed":42,"checkpoint_selection":"final update only","decoder":{"candidate_budget":12,"beam_width":384},"primary_metric":"validation positive top-12 recall","hard_wall_seconds":240,"hard_peak_rss_bytes":2147483648,"heldout_access":False};write_json(a.output/"preregistration.json",prereg);(a.output/"preregistration.sha256").write_text(sha(a.output/"preregistration.json")+"  preregistration.json\n")
    train_batches=batches(a.experience,training);independent,conditional,metrics,runs=train_pair(a.source,config,train_batches,training,validation,a.output);write_csv(a.output/"training_metrics.csv",metrics)
    disabled=copy.deepcopy(conditional)
    import torch
    with torch.no_grad():
        disabled.uv_condition.weight.zero_();disabled.uv_condition.bias.zero_();disabled.uvw_condition.weight.zero_();disabled.uvw_condition.bias.zero_()
    models={"frozen_independent":("independent",base,config),"independent_control":("independent",independent,config),"conditional":("conditional",conditional,config),"conditional_disabled":("conditional",disabled,config)};train_eval=evaluate(models,training,"train");val_eval=evaluate(models,validation,"validation");recall=train_eval[0]+val_eval[0];latency=train_eval[1]+val_eval[1];factor=train_eval[2]+val_eval[2];write_csv(a.output/"validation_recall.csv",recall);write_csv(a.output/"candidate_latency.csv",latency);write_csv(a.output/"factorization_analysis.csv",factor)
    aggregates={}
    for split in ("train","validation"):
        aggregates[split]={}
        for name in models:
            rows=[x for x in recall if x["split"]==split and x["configuration"]==name];fr=[x for x in factor if x["split"]==split and x["configuration"]==name];aggregates[split][name]={"positive_recall":sum(x["positive_recalled"] for x in rows),"positive_denominator":sum(x["positive_count"] for x in rows),"state_recall":sum(x["any_positive_recalled"] for x in rows),"state_denominator":len(rows),"reference_recall":sum(x["reference_recalled"] for x in rows),"beam_survival":sum(x["width384_survives"] for x in fr),"mean_candidate_diversity":sum(x["mean_pairwise_token_distance"] for x in rows)/len(rows)}
    ce={name:validation_ce(kind,model,config,a.experience,validation) for name,(kind,model,config) in models.items()};summary={"git":{"branch":subprocess.check_output(["git","branch","--show-current"],text=True).strip(),"commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()},"environment":{"python":platform.python_version(),"device":"cpu"},"architecture":spec,"runs":runs,"evaluation":aggregates,"validation_mcts_cross_entropy":ce,"preregistration_sha256":sha(a.output/"preregistration.json"),"source_checkpoint_sha256":sha(a.source/"best_model_state.pt"),"heldout_accessed":False};write_json(a.output/"summary.json",summary);write_json(a.output/"checkpoint_manifest.json",{"source_checkpoint_sha256":summary["source_checkpoint_sha256"],"independent_control_sha256":runs["independent_control"]["checkpoint_sha256"],"conditional_sha256":runs["conditional"]["checkpoint_sha256"],"historical_checkpoints_modified":False});print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
