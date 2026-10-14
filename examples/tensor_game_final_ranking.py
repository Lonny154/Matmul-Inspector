#!/usr/bin/env python3
"""Run the bounded TG-4L final-ranking experiment."""

from __future__ import annotations

import argparse, csv, hashlib, json, platform, subprocess, time
from collections import Counter
from pathlib import Path

from matmul_inspector.tensor_game_candidate_coverage import decode
from matmul_inspector.tensor_game_coverage_training import CoverageDataset, collate_coverage, coverage_loss
from matmul_inspector.tensor_game_experience import action_from_payload, action_identity
from matmul_inspector.tensor_game_final_ranking import complete_action_score, final_ranking_loss
from matmul_inspector.tensor_game_learning import load_checkpoint, rank_complete_actions
from matmul_inspector.tensor_game_offline_learning import _move_tensors, configure_trainable, sampled_support_log_probabilities
from matmul_inspector.tensor_game_out_of_sample import tensor_tuple
from matmul_inspector.tensor_game_prefix_coverage import _advance_beam, prefix_survival_loss, sign_forms, trace_positive_prefixes
from matmul_inspector.tensor_game_search import canonicalize_action


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def load_rows(path): return [json.loads(x) for x in path.read_text().splitlines()]
def write_json(path, value): path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
def write_csv(path, rows):
    with path.open("w", newline="") as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def load_model(source, checkpoint=None):
    import torch
    model, config, _ = load_checkpoint(source, "cpu")
    if checkpoint: model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval(); return model, config


def tiny_batch(experience, labels):
    selected={x["state_id"] for x in labels}; dataset=CoverageDataset(experience,"train",labels); rows={}
    for index in range(len(dataset)):
        item=dataset[index]
        if item["state_id"] in selected: rows.setdefault(item["state_id"],item)
    if set(rows)!=selected: raise ValueError("tiny labels missing replay support")
    return collate_coverage([rows[x["state_id"]] for x in labels])


def components(model, config, batch):
    logits,_=model(batch["residual"]); lp=sampled_support_log_probabilities(logits,batch["actions"],batch["action_mask"],config.shape)
    mcts=-(batch["search_policy"]*lp.masked_fill(~batch["action_mask"],0)).sum(1).mean()
    idx=batch["coverage_indices"]; positives=batch["positive_actions"]; mask=batch["positive_mask"]
    selected=logits[idx]
    return logits,mcts,coverage_loss(selected,positives,mask),prefix_survival_loss(selected,positives,mask),final_ranking_loss(selected,positives,mask)


def train(name, mode, source, batch, output, updates=100):
    import torch
    model,config,_=load_checkpoint(source,"cpu");configure_trainable(model,"policy_only",True);model.train();optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=.001)
    torch.manual_seed(42);torch.use_deterministic_algorithms(True);history=[];started=time.perf_counter()
    for update in range(1,updates+1):
        optimizer.zero_grad(set_to_none=True);_,mcts,set_loss,prefix,rank=components(model,config,batch)
        total=mcts+(set_loss if mode=="set" else prefix)+(rank if mode=="combined" else 0)
        total.backward();gradient=torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),1.0);optimizer.step()
        if update==1 or update%10==0: history.append({"configuration":name,"update":update,"mcts_loss":float(mcts.detach()),"set_loss":float(set_loss.detach()),"prefix_loss":float(prefix.detach()),"ranking_loss":float(rank.detach()),"total_loss":float(total.detach()),"gradient_norm":float(gradient)})
    directory=output/"checkpoints"/name;directory.mkdir(parents=True);path=directory/"model_state.pt";torch.save(model.state_dict(),path);model.eval()
    return model,config,history,{"checkpoint_sha256":sha(path),"runtime_seconds":time.perf_counter()-started}


def diagnostic_rows(model, config, labels, configuration):
    import torch
    rows=[]
    for label in labels:
        residual=torch.tensor(label["residual"],dtype=torch.float32).unsqueeze(0)
        with torch.inference_mode(): logits,_=model(residual)
        logp=torch.log_softmax(logits[0].detach().cpu(),-1);beam=[((),torch.zeros(()))]
        for position in range(12): beam,_=_advance_beam(logp,beam,position,384)
        beam_prefixes={x:i+1 for i,(x,_) in enumerate(beam)}
        decoded=rank_complete_actions(logits[0],config.shape,12);decoded_ids=[action_identity(x[0]) for x in decoded];cutoff=decoded[-1][1] if decoded else None
        actions=[action_from_payload(x["action"]) for x in label["positive_actions"]];traces=trace_positive_prefixes(logits[0],actions)
        for i,(payload,action) in enumerate(zip(label["positive_actions"],actions)):
            trace=[x for x in traces if x["action_index"]==i];ranks=[beam_prefixes[f] for f in sign_forms(action) if f in beam_prefixes]
            teacher=float(complete_action_score(logits[0],action).detach());selected=payload["action_id"] in decoded_ids
            first=trace[-1]["first_pruning_position"]
            reason="selected" if selected else "prefix_pruning" if first else "final_ranking"
            rows.append({"split":label["split"],"state_id":label["state_id"],"target_id":label["target_id"],"configuration":configuration,"action_id":payload["action_id"],"token_sequence":json.dumps(action.u+action.v+action.w,separators=(",",":")),"teacher_forced_score":teacher,"width384_survives":first is None,"first_pruning_stage":first,"observed_complete_rank":min(ranks) if ranks else None,"final_top12":selected,"final_selection_cutoff":cutoff,"score_margin_to_cutoff":teacher-cutoff if cutoff is not None else None,"exclusion_mechanism":reason})
    return rows


def recall_rows(models, labels, split):
    rows=[]
    for label in labels:
        positives={x["action_id"] for x in label["positive_actions"]};res=tensor_tuple(label["residual"])
        for name,(model,config) in models.items():
            for cutoff in (1,12,32,256):
                started=time.perf_counter();ids={action_identity(x[0]) for x in decode(model,config,res,cutoff)}
                rows.append({"split":split,"state_id":label["state_id"],"target_id":label["target_id"],"configuration":name,"cutoff":cutoff,"positive_count":len(positives),"positive_recalled":len(positives&ids),"any_positive_recalled":bool(positives&ids),"candidate_count":len(ids),"generation_seconds":time.perf_counter()-started})
    return rows


def mcts_ce(model,config,experience,labels):
    import torch
    dataset=CoverageDataset(experience,"validation",labels);loader=torch.utils.data.DataLoader(dataset,batch_size=128,collate_fn=collate_coverage);total=count=0
    with torch.inference_mode():
        for batch in loader:
            logits,_=model(batch["residual"]);lp=sampled_support_log_probabilities(logits,batch["actions"],batch["action_mask"],config.shape);loss=-(batch["search_policy"]*lp.masked_fill(~batch["action_mask"],0)).sum(1);total+=float(loss.sum());count+=len(loss)
    return total/count


def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);p.add_argument("--source",type=Path,required=True);p.add_argument("--labels",type=Path,required=True);p.add_argument("--experience",type=Path,required=True);p.add_argument("--tg4k",type=Path,required=True);a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    a.output.mkdir(parents=True);all_labels=load_rows(a.labels);training=sorted((x for x in all_labels if x["split"]=="train"),key=lambda x:x["target_id"])[:8];validation=[x for x in all_labels if x["split"]=="validation"]
    prereg={"version":1,"created_before_training":True,"training_target_ids":[x["target_id"] for x in training],"configurations":{"matched_set":"mcts + set","matched_prefix":"mcts + prefix","matched_combined":"mcts + prefix + final_rank"},"weights":{"mcts":1.0,"coverage":1.0,"prefix":1.0,"rank":1.0},"ranking_margin":0.1,"prefix_margin":0.1,"updates":100,"optimizer":"AdamW","learning_rate":0.001,"seed":42,"policy_head_only":True,"hard_negatives":"regenerated each update; twelfth accepted width-384 proposal","primary_metric":"actual width-384 top-12 per-positive recall","heldout_access":False}
    write_json(a.output/"preregistration.json",prereg);(a.output/"preregistration.sha256").write_text(sha(a.output/"preregistration.json")+"  preregistration.json\n")
    batch=tiny_batch(a.experience,training);models={};history=[];runs={}
    for name,mode in (("matched_set","set"),("matched_prefix","prefix"),("matched_combined","combined")):
        model,config,local,run=train(name,mode,a.source,batch,a.output);models[name]=(model,config);history+=local;runs[name]=run
    write_csv(a.output/"tiny_training_metrics.csv",history)
    saved_set=load_model(a.source,a.tg4k/"checkpoints/tiny_set_coverage/model_state.pt");saved_prefix=load_model(a.source,a.tg4k/"checkpoints/tiny_prefix_coverage/model_state.pt")
    diagnostics=diagnostic_rows(*saved_set,training,"tg4k_tiny_set")+diagnostic_rows(*saved_prefix,training,"tg4k_tiny_prefix")
    for name,pair in models.items(): diagnostics+=diagnostic_rows(*pair,training,name)
    write_csv(a.output/"decoder_diagnostics.csv",diagnostics)
    evaluation={"tg4k_tiny_set":saved_set,"tg4k_tiny_prefix":saved_prefix,**models};recall=recall_rows(evaluation,training,"train")+recall_rows(evaluation,validation,"validation");write_csv(a.output/"validation_recall.csv",recall)
    failure=[]
    for name,pair in evaluation.items(): failure+=diagnostic_rows(*pair,validation,name)
    write_csv(a.output/"failure_analysis.csv",failure)
    ce={name:mcts_ce(*pair,a.experience,all_labels) for name,pair in evaluation.items()}
    aggregate={}
    for split,denom in (("train",sum(len(x["positive_actions"]) for x in training)),("validation",sum(len(x["positive_actions"]) for x in validation))):
        aggregate[split]={}
        for name in evaluation:
            aggregate[split][name]={str(c):{"positive_recall":sum(int(r["positive_recalled"]) for r in recall if r["split"]==split and r["configuration"]==name and int(r["cutoff"])==c),"state_recall":sum(r["any_positive_recalled"] in (True,"True") for r in recall if r["split"]==split and r["configuration"]==name and int(r["cutoff"])==c),"positive_denominator":denom,"state_denominator":len(training if split=="train" else validation)} for c in (1,12,32,256)}
    summary={"git":{"branch":subprocess.check_output(["git","branch","--show-current"],text=True).strip(),"commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()},"environment":{"python":platform.python_version()},"input_hashes":{"source_checkpoint":sha(a.source/"best_model_state.pt"),"labels":sha(a.labels),"tg4k_set":sha(a.tg4k/"checkpoints/tiny_set_coverage/model_state.pt"),"tg4k_prefix":sha(a.tg4k/"checkpoints/tiny_prefix_coverage/model_state.pt")},"runs":runs,"recall":aggregate,"validation_mcts_cross_entropy":ce,"diagnostic_mechanisms":{name:dict(Counter(x["exclusion_mechanism"] for x in diagnostics if x["configuration"]==name)) for name in ("tg4k_tiny_set","tg4k_tiny_prefix")},"heldout_accessed":False}
    write_json(a.output/"summary.json",summary);print(json.dumps(summary,indent=2))


if __name__=="__main__": main()
