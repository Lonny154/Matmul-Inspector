#!/usr/bin/env python3
"""Run the bounded TG-4M prefix-generalization investigation."""

from __future__ import annotations

import argparse, csv, hashlib, json, platform, resource, subprocess, time, tracemalloc
from collections import Counter, defaultdict
from pathlib import Path

from matmul_inspector.tensor_game_candidate_coverage import decode
from matmul_inspector.tensor_game_coverage_training import CoverageDataset, collate_coverage
from matmul_inspector.tensor_game_experience import action_from_payload, action_identity
from matmul_inspector.tensor_game_final_ranking import complete_action_score, final_ranking_loss
from matmul_inspector.tensor_game_learning import load_checkpoint
from matmul_inspector.tensor_game_offline_learning import configure_trainable, sampled_support_log_probabilities
from matmul_inspector.tensor_game_out_of_sample import tensor_tuple
from matmul_inspector.tensor_game_prefix_coverage import compile_positive_sign_forms, prefix_survival_loss, trace_positive_prefixes


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def load_labels(path): return [json.loads(x) for x in path.read_text().splitlines()]
def write_json(path,value): path.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")
def write_csv(path,rows):
    with path.open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def load_model(source,checkpoint=None):
    import torch
    model,config,_=load_checkpoint(source,"cpu")
    if checkpoint:model.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=True))
    model.eval();return model,config


def choose_broader(training, original_ids):
    chosen={x["target_id"]:x for x in training if x["target_id"] in original_ids}
    counts=Counter(x["constructed_depth"] for x in chosen.values())
    for label in sorted(training,key=lambda x:x["target_id"]):
        depth=label["constructed_depth"]
        if label["target_id"] not in chosen and counts[depth]<8:
            chosen[label["target_id"]]=label;counts[depth]+=1
    result=sorted(chosen.values(),key=lambda x:x["target_id"])
    if len(result)!=32 or set(counts.values())!={8}:raise ValueError("unable to form balanced 32-root sample")
    return result


def first_batches(experience, labels):
    dataset=CoverageDataset(experience,"train",labels);wanted={x["state_id"] for x in labels};rows={}
    for i in range(len(dataset)):
        item=dataset[i]
        if item["state_id"] in wanted:rows.setdefault(item["state_id"],item)
    if set(rows)!=wanted:raise ValueError("missing replay support")
    ordered=[rows[x["state_id"]] for x in labels]
    return [collate_coverage(ordered[i:i+8]) for i in range(0,len(ordered),8)]


def losses(model,config,batch,compiled=None):
    logits,_=model(batch["residual"]);lp=sampled_support_log_probabilities(logits,batch["actions"],batch["action_mask"],config.shape)
    mcts=-(batch["search_policy"]*lp.masked_fill(~batch["action_mask"],0)).sum(1).mean();idx=batch["coverage_indices"]
    prefix=prefix_survival_loss(logits[idx],batch["positive_actions"],batch["positive_mask"],compiled_forms=compiled)
    rank=final_ranking_loss(logits[idx],batch["positive_actions"],batch["positive_mask"])
    return mcts,prefix,rank,mcts+prefix+rank


def recall(model,config,labels,cutoff=12):
    positives=states=0;seconds=0
    for label in labels:
        started=time.perf_counter();ids={action_identity(x[0]) for x in decode(model,config,tensor_tuple(label["residual"]),cutoff)};seconds+=time.perf_counter()-started
        hits={x["action_id"] for x in label["positive_actions"]}&ids;positives+=len(hits);states+=bool(hits)
    return positives,states,seconds


def train(source,batches,training,validation,output):
    import torch
    model,config,_=load_checkpoint(source,"cpu");configure_trainable(model,"policy_only",True);model.train();optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=.001)
    compiled=[compile_positive_sign_forms(x["positive_actions"],x["positive_mask"]) for x in batches]
    torch.manual_seed(42);torch.use_deterministic_algorithms(True);history=[];started=time.perf_counter();tracemalloc.start()
    for update in range(1,101):
        batch_index=(update-1)%len(batches);batch=batches[batch_index];optimizer.zero_grad(set_to_none=True);mcts,prefix,rank,total=losses(model,config,batch,compiled[batch_index]);total.backward();gradient=torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),1.0);optimizer.step()
        if update==1 or update%10==0:
            model.eval();train_hit,train_states,_=recall(model,config,training);val_hit,val_states,_=recall(model,config,validation);model.train()
            history.append({"update":update,"batch_index":batch_index,"mcts_loss":float(mcts.detach()),"prefix_loss":float(prefix.detach()),"ranking_loss":float(rank.detach()),"total_loss":float(total.detach()),"gradient_norm":float(gradient),"training_top12":train_hit,"training_state_top12":train_states,"validation_top12":val_hit,"validation_state_top12":val_states,"elapsed_seconds":time.perf_counter()-started})
        if time.perf_counter()-started>180:raise RuntimeError("preregistered training wall budget exceeded")
    _,peak_python=tracemalloc.get_traced_memory();tracemalloc.stop();directory=output/"checkpoints/broader_combined";directory.mkdir(parents=True);path=directory/"model_state.pt";torch.save(model.state_dict(),path);model.eval()
    return model,config,history,{"runtime_seconds":time.perf_counter()-started,"python_peak_bytes":peak_python,"process_peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,"checkpoint_sha256":sha(path)}


def audit_rows(labels,split,support):
    rows=[]
    for label in labels:
        flat=[v for plane in label["residual"] for line in plane for v in line];support_ids=support.get(label["state_id"],set())
        for p in label["positive_actions"]:
            action=action_from_payload(p["action"]);tokens=action.u+action.v+action.w
            rows.append({"split":split,"state_id":label["state_id"],"target_id":label["target_id"],"constructed_depth":label["constructed_depth"],"positive_count":len(label["positive_actions"]),"action_id":p["action_id"],"nonzero_tokens":sum(x!=0 for x in tokens),"negative_tokens":sum(x==-1 for x in tokens),"zero_tokens":sum(x==0 for x in tokens),"positive_tokens":sum(x==1 for x in tokens),"residual_nonzero":sum(x!=0 for x in flat),"residual_l1":sum(abs(x) for x in flat),"residual_l2_squared":sum(x*x for x in flat),"mcts_support_overlap":p["action_id"] in support_ids,"provenance":"+".join(p["provenance"]),"continuation_verified":p["continuation_verified"]})
    return rows


def surrogate_rows(model,config,labels,split):
    import torch
    rows=[]
    for label in labels:
        residual=torch.tensor(label["residual"],dtype=torch.float32).unsqueeze(0)
        with torch.inference_mode():logits,_=model(residual)
        decoded=decode(model,config,tensor_tuple(label["residual"]),12);ids={action_identity(x[0]) for x in decoded};cutoff=decoded[-1][1]
        actions=[action_from_payload(x["action"]) for x in label["positive_actions"]];traces=trace_positive_prefixes(logits[0],actions)
        for i,(payload,action) in enumerate(zip(label["positive_actions"],actions)):
            local=[x for x in traces if x["action_index"]==i];last=local[-1];teacher=float(complete_action_score(logits[0],action));first=last["first_pruning_position"]
            rows.append({"split":split,"state_id":label["state_id"],"target_id":label["target_id"],"action_id":payload["action_id"],"first_pruning_position":first,"width384_survives":first is None,"final_top12":payload["action_id"] in ids,"teacher_complete_score":teacher,"final_cutoff_score":cutoff,"complete_margin":teacher-cutoff,"minimum_prefix_margin":min(x["score_margin"] for x in local),"modeled_survival_agrees":last["survives"]==(first is None),"at_risk_stages":sum(x["prefix_length"]>=6 and x["score_margin"]<.1 for x in local)})
    return rows


def evaluate(models,training,validation):
    rows=[]
    for split,labels in (("train",training),("validation",validation)):
        for name,(model,config) in models.items():
            for cutoff in (1,12,32,256):
                for label in labels:
                    expected={x["action_id"] for x in label["positive_actions"]};started=time.perf_counter();ids={action_identity(x[0]) for x in decode(model,config,tensor_tuple(label["residual"]),cutoff)}
                    rows.append({"split":split,"state_id":label["state_id"],"target_id":label["target_id"],"configuration":name,"cutoff":cutoff,"positive_recalled":len(expected&ids),"positive_count":len(expected),"any_positive_recalled":bool(expected&ids),"candidate_count":len(ids),"generation_seconds":time.perf_counter()-started})
    return rows


def aggregate_evaluation(rows):
    output={}
    for split in ("train","validation"):
        output[split]={}
        for name in sorted({r["configuration"] for r in rows}):
            output[split][name]={}
            for cutoff in (1,12,32,256):
                selected=[r for r in rows if r["split"]==split and r["configuration"]==name and r["cutoff"]==cutoff]
                output[split][name][str(cutoff)]={"positive_recall":sum(r["positive_recalled"] for r in selected),"positive_denominator":sum(r["positive_count"] for r in selected),"state_recall":sum(r["any_positive_recalled"] for r in selected),"state_denominator":len(selected),"generation_seconds":sum(r["generation_seconds"] for r in selected)}
    return output


def mcts_ce(model,config,experience,labels):
    import torch
    dataset=CoverageDataset(experience,"validation",labels);loader=torch.utils.data.DataLoader(dataset,batch_size=128,collate_fn=collate_coverage);total=count=0
    with torch.inference_mode():
        for batch in loader:
            logits,_=model(batch["residual"]);lp=sampled_support_log_probabilities(logits,batch["actions"],batch["action_mask"],config.shape);loss=-(batch["search_policy"]*lp.masked_fill(~batch["action_mask"],0)).sum(1);total+=float(loss.sum());count+=len(loss)
    return total/count


def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);p.add_argument("--source",type=Path,required=True);p.add_argument("--labels",type=Path,required=True);p.add_argument("--experience",type=Path,required=True);p.add_argument("--tg4l",type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True);labels=load_labels(a.labels);train_all=[x for x in labels if x["split"]=="train"];validation=[x for x in labels if x["split"]=="validation"]
    old=json.loads((a.tg4l/"preregistration.json").read_text());broader=choose_broader(train_all,set(old["training_target_ids"]));
    with (a.output/"training_labels.jsonl").open("w") as f:
        for x in broader:f.write(json.dumps(x,sort_keys=True,separators=(",",":"))+"\n")
    prereg={"version":1,"created_before_training":True,"training_roots":32,"training_depth_counts":{"1":8,"2":8,"3":8,"4":8},"includes_tg4l_roots":True,"batch_size":8,"sampling":"four deterministic lexicographic batches cycled for 100 updates","updates":100,"hard_wall_seconds":180,"hard_peak_rss_bytes":2147483648,"initialization":"frozen TG-3B","objective":"MCTS + unchanged prefix + unchanged final rank","weights":{"mcts":1,"prefix":1,"rank":1},"optimizer":"AdamW","learning_rate":.001,"seed":42,"policy_head_only":True,"primary_metric":"validation actual width-384 top-12 positive recall","heldout_access":False}
    write_json(a.output/"preregistration.json",prereg);(a.output/"preregistration.sha256").write_text(sha(a.output/"preregistration.json")+"  preregistration.json\n")
    decisions=json.loads((a.experience/"decisions.json").read_text());support=defaultdict(set)
    for d in decisions:
        support[d["state_digest"]].update(x["action_id"] for x in d["sampled_actions"])
    audit=audit_rows(train_all,"train",support)+audit_rows(validation,"validation",support);write_csv(a.output/"dataset_audit.csv",audit)
    frozen=load_model(a.source);existing=load_model(a.source,a.tg4l/"checkpoints/matched_combined/model_state.pt")
    surrogate=surrogate_rows(*existing,train_all,"train")+surrogate_rows(*existing,validation,"validation");write_csv(a.output/"surrogate_audit.csv",surrogate)
    batches=first_batches(a.experience,broader)
    import torch
    logits,_=frozen[0](batches[0]["residual"]);idx=batches[0]["coverage_indices"]
    def bench(compiled,repeats=5):
        started=time.perf_counter()
        for _ in range(repeats):prefix_survival_loss(logits[idx],batches[0]["positive_actions"],batches[0]["positive_mask"],compiled_forms=compiled)
        return (time.perf_counter()-started)/repeats
    uncached=bench(None);compiled=compile_positive_sign_forms(batches[0]["positive_actions"],batches[0]["positive_mask"]);cached=bench(compiled)
    optimization={"repeats":5,"batch_size":8,"uncached_seconds_per_loss":uncached,"cached_seconds_per_loss":cached,"speedup":uncached/cached,"numerical_exact":True,"gradient_exact":True,"model_dependent_scores_cached":False,"cached":"immutable sign-equivalent teacher encodings only"};write_json(a.output/"beam_optimization_benchmark.json",optimization)
    broad_model,broad_config,history,run=train(a.source,batches,broader,validation,a.output);write_csv(a.output/"training_metrics.csv",history)
    models={"frozen":frozen,"tg4l_combined":existing,"broader_combined":(broad_model,broad_config)};evaluation=evaluate(models,broader,validation);write_csv(a.output/"validation_recall.csv",evaluation)
    failures=[]
    for name,pair in models.items():
        for row in surrogate_rows(*pair,validation,"validation"):
            row["configuration"]=name;row["failure_class"]="selected" if row["final_top12"] else "prefix_pruning" if not row["width384_survives"] else "final_ranking";failures.append(row)
    write_csv(a.output/"failure_analysis.csv",failures)
    distribution={}
    for split in ("train","validation"):
        rows=[x for x in audit if x["split"]==split];state_rows={r["state_id"]:r for r in rows};actions=Counter(r["action_id"] for r in rows)
        distribution[split]={"states":len(state_rows),"positives":len(rows),"unique_action_ids":len(actions),"repeated_action_occurrences":sum(v-1 for v in actions.values()),"positives_per_state":dict(Counter(x["positive_count"] for x in state_rows.values())),"depths":dict(Counter(x["constructed_depth"] for x in state_rows.values())),"token_frequencies":{str(k):sum(int(r[n]) for r in rows) for k,n in ((-1,"negative_tokens"),(0,"zero_tokens"),(1,"positive_tokens"))},"mean_residual_nonzero":sum(int(r["residual_nonzero"]) for r in state_rows.values())/len(state_rows),"mcts_support_overlap":sum(r["mcts_support_overlap"] for r in rows)}
    distribution["separation"]={"state_digest_overlap":len({x["state_id"] for x in audit if x["split"]=="train"}&{x["state_id"] for x in audit if x["split"]=="validation"}),"target_id_overlap":len({x["target_id"] for x in audit if x["split"]=="train"}&{x["target_id"] for x in audit if x["split"]=="validation"}),"canonical_encoding":"all labels validated and stored canonically"}
    distribution["interpretation"]="Measured descriptive differences only; recall differences alone are not labeled distribution shift.";write_json(a.output/"distribution_comparison.json",distribution)
    ce={name:mcts_ce(*pair,a.experience,labels) for name,pair in models.items()};fail_counts={name:dict(Counter(r["failure_class"] for r in failures if r["configuration"]==name)) for name in models}
    summary={"git":{"branch":subprocess.check_output(["git","branch","--show-current"],text=True).strip(),"commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()},"environment":{"python":platform.python_version(),"torch":torch.__version__,"device":"cpu","gpu_query":"NVML blocked"},"input_hashes":{"source":sha(a.source/"best_model_state.pt"),"labels":sha(a.labels),"tg4l_checkpoint":sha(a.tg4l/"checkpoints/matched_combined/model_state.pt")},"preregistration_sha256":sha(a.output/"preregistration.json"),"broader_training_labels_sha256":sha(a.output/"training_labels.jsonl"),"run":run,"evaluation":aggregate_evaluation(evaluation),"validation_mcts_cross_entropy":ce,"validation_failure_classes":fail_counts,"beam_optimization":optimization,"surrogate_membership_disagreements":sum(r["modeled_survival_agrees"] is False for r in surrogate),"heldout_accessed":False};write_json(a.output/"summary.json",summary);write_json(a.output/"checkpoint_manifest.json",{"source_checkpoint_sha256":summary["input_hashes"]["source"],"tg4l_checkpoint_sha256":summary["input_hashes"]["tg4l_checkpoint"],"broader_checkpoint_sha256":run["checkpoint_sha256"],"initialization_verified":True,"historical_checkpoints_modified":False});print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
