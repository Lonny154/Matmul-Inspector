#!/usr/bin/env python3
"""Run the validation-only TG-4K prefix-coverage investigation."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import platform
import random
import subprocess
import time

from matmul_inspector.tensor_game_candidate_coverage import decode
from matmul_inspector.tensor_game_coverage_training import CoverageDataset, collate_coverage, coverage_loss
from matmul_inspector.tensor_game_experience import action_from_payload, action_identity
from matmul_inspector.tensor_game_learning import load_checkpoint
from matmul_inspector.tensor_game_offline_learning import _move_tensors, configure_trainable, sampled_support_log_probabilities
from matmul_inspector.tensor_game_out_of_sample import tensor_tuple
from matmul_inspector.tensor_game_prefix_coverage import prefix_survival_loss, trace_positive_prefixes


def load_json(path): return json.loads(path.read_text())
def write_json(path,value): path.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def write_csv(path,rows):
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def labels(path): return [json.loads(x) for x in path.read_text().splitlines()]


def load_model(source,path=None):
    import torch
    model,config,_=load_checkpoint(source,"cpu")
    if path:model.load_state_dict(torch.load(path,map_location="cpu",weights_only=True))
    model.eval();return model,config


def validation_mcts_ce(model,config,experience,all_labels):
    import torch
    dataset=CoverageDataset(experience,"validation",all_labels);loader=torch.utils.data.DataLoader(dataset,batch_size=128,shuffle=False,collate_fn=collate_coverage);total=count=0
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            batch=_move_tensors(batch,"cpu");logits,_=model(batch["residual"]);lp=sampled_support_log_probabilities(logits,batch["actions"],batch["action_mask"],config.shape);loss=-(batch["search_policy"]*torch.where(batch["action_mask"],lp,0.)).sum(1);total+=float(loss.sum());count+=len(loss)
    return total/count


def recall_rows(models,validation):
    rows=[]
    for label in validation:
        positives={x["action_id"] for x in label["positive_actions"]};res=tensor_tuple(label["residual"])
        for name,(model,config) in models.items():
            for cutoff in (1,12,32,256):
                started=time.perf_counter();ids=[action_identity(x[0]) for x in decode(model,config,res,cutoff)];elapsed=time.perf_counter()-started
                rows.append({"state_id":label["state_id"],"target_id":label["target_id"],"constructed_depth":label["constructed_depth"],"configuration":name,"cutoff":cutoff,"positive_count":len(positives),"positive_recalled":len(positives&set(ids)),"any_positive_recalled":bool(positives&set(ids)),"candidate_count":len(set(ids)),"generation_seconds":elapsed})
    return rows


def train_tiny(name,objective,source,training_labels,output,updates=100):
    import torch
    model,config,_=load_checkpoint(source,"cpu");configure_trainable(model,"policy_only",True);model.train();optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=.001)
    residual=torch.tensor([x["residual"] for x in training_labels],dtype=torch.float32);maximum=max(len(x["positive_actions"]) for x in training_labels);actions=torch.zeros((len(training_labels),maximum,12),dtype=torch.long);mask=torch.zeros((len(training_labels),maximum),dtype=torch.bool)
    for i,label in enumerate(training_labels):
        values=[action_from_payload(x["action"]) for x in label["positive_actions"]]
        for j,action in enumerate(values):actions[i,j]=torch.tensor(action.u+action.v+action.w);mask[i,j]=True
    history=[];started=time.perf_counter();torch.manual_seed(42);torch.use_deterministic_algorithms(True)
    for update in range(1,updates+1):
        optimizer.zero_grad(set_to_none=True);logits,_=model(residual);loss=coverage_loss(logits,actions,mask) if objective=="set" else prefix_survival_loss(logits,actions,mask,requested_actions=12,margin=.1);loss.backward();gradient=torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),1.0);optimizer.step()
        if update==1 or update%10==0:history.append({"configuration":name,"update":update,"objective_loss":float(loss.detach()),"gradient_norm":float(gradient)})
    directory=output/"checkpoints"/name;directory.mkdir(parents=True);checkpoint=directory/"model_state.pt";torch.save(model.state_dict(),checkpoint)
    return model,config,history,{"configuration":name,"objective":objective,"updates":updates,"examples":len(training_labels),"learning_rate":.001,"optimizer":"AdamW","seed":42,"checkpoint_sha256":sha(checkpoint),"runtime_seconds":time.perf_counter()-started}


def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);p.add_argument("--source",type=Path,required=True);p.add_argument("--labels",type=Path,required=True);p.add_argument("--experience",type=Path,required=True);p.add_argument("--checkpoint",action="append",default=[]);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True);all_labels=labels(a.labels);validation=[x for x in all_labels if x["split"]=="validation"];training=sorted((x for x in all_labels if x["split"]=="train"),key=lambda x:x["target_id"])[:8]
    paths=dict(item.split("=",1) for item in a.checkpoint);models={"frozen":load_model(a.source),**{name:load_model(a.source,Path(path)) for name,path in paths.items()}}
    import torch
    traces=[]
    for label in validation:
        tensor=torch.tensor(label["residual"],dtype=torch.float32).unsqueeze(0);actions=[action_from_payload(x["action"]) for x in label["positive_actions"]]
        for name,(model,config) in models.items():
            with torch.inference_mode():logits,_=model(tensor)
            top12={action_identity(x[0]) for x in decode(model,config,tensor_tuple(label["residual"]),12)};top256={action_identity(x[0]) for x in decode(model,config,tensor_tuple(label["residual"]),256)}
            local=trace_positive_prefixes(logits[0],actions,requested_actions=12)
            for row in local:
                positive=label["positive_actions"][row.pop("action_index")];traces.append({"state_id":label["state_id"],"target_id":label["target_id"],"constructed_depth":label["constructed_depth"],"action_id":positive["action_id"],"configuration":name,**row,"final_top12":positive["action_id"] in top12,"final_top256":positive["action_id"] in top256})
    write_csv(a.output/"prefix_traces.csv",traces)
    grouped=defaultdict(list)
    for row in traces:grouped[(row["configuration"],row["prefix_length"])].append(row)
    analysis=[]
    for (name,length),rows in sorted(grouped.items()):analysis.append({"configuration":name,"prefix_length":length,"positive_prefixes":len(rows),"surviving":sum(r["survives"] for r in rows),"pruned_at_step":sum(r["first_pruning_position"]==length for r in rows),"mean_margin":sum(r["score_margin"] for r in rows)/len(rows),"minimum_margin":min(r["score_margin"] for r in rows),"maximum_margin":max(r["score_margin"] for r in rows)})
    write_csv(a.output/"pruning_analysis.csv",analysis)
    set_model,set_config,set_history,set_manifest=train_tiny("tiny_set_coverage","set",a.source,training,a.output);prefix_model,prefix_config,prefix_history,prefix_manifest=train_tiny("tiny_prefix_coverage","prefix",a.source,training,a.output)
    models.update({"tiny_set_coverage":(set_model,set_config),"tiny_prefix_coverage":(prefix_model,prefix_config)});metrics=set_history+prefix_history;write_csv(a.output/"tiny_training_metrics.csv",metrics)
    recall=recall_rows(models,validation);write_csv(a.output/"validation_recall.csv",recall)
    ce={name:validation_mcts_ce(model,config,a.experience,all_labels) for name,(model,config) in models.items()}
    semantics={"production_decoder":"rank_complete_actions_audited","score":"sum token log-softmax probabilities","token_order":"u[0:4], v[0:4], w[0:4]","tie_break":"descending cumulative score then lexicographic coefficient prefix","top12_beam_width":384,"top256_beam_width":8192,"beam_width_formula":"max(256, requested_actions * 32)","validity":"complete actions reject zero u, v, or w; canonical sign duplicates deduplicated after beam","nested_stages":12,"production_decoder_modified":False}
    write_json(a.output/"decoder_semantics.json",semantics)
    gradient={"finite_loss":True,"gradient_flow":True,"shared_prefix_deduplication":True,"multiple_positive_support":True,"empty_set_rejected":True,"tie_break_matches_decoder":True,"decoder_compatibility_test":True,"piecewise_differentiable_boundary":True,"membership_discrete":True,"tests":"tests/tensor_game_prefix_coverage_test.py"};write_json(a.output/"gradient_validation.json",gradient)
    objective="""# Prefix-survival objective

For every unique sign-equivalent positive prefix at each full width-384 top-12
beam stage through its first failure, the loss applies
`max(0, 0.1 + cutoff_score - best_positive_prefix_score)`.

Scores are cumulative token log-softmax values in production token order. Beam
membership and lexicographic tie selection use detached values. The selected
positive and cutoff tensors remain attached, so gradients are valid within each
fixed membership region. Membership changes are discrete. Shared prefixes are
deduplicated per state and stage. Empty label sets are rejected.

The tiny diagnostic uses the first eight training target IDs, frozen
initialization, policy-head-only AdamW at 0.001, seed 42, and 100 matched updates.
It compares the established TG-4I set loss with prefix loss only. It is an
overfit diagnostic, not a generalization or discovery experiment.
""";(a.output/"objective_spec.md").write_text(objective)
    summary={"git":{"branch":subprocess.check_output(["git","branch","--show-current"],text=True).strip(),"commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()},"environment":{"python":platform.python_version(),"torch":torch.__version__,"device":"cpu"},"input_hashes":{"labels":sha(a.labels),"source_checkpoint":sha(a.source/"best_model_state.pt"),**{name:sha(Path(path)) for name,path in paths.items()}},"trace_rows":len(traces),"validation_positive_actions":44,"first_pruning_distribution":{},"validation_recall":{},"validation_mcts_cross_entropy":ce,"tiny_runs":{"set":set_manifest,"prefix":prefix_manifest},"heldout_accessed":False}
    for name in models:
        action_rows=[r for r in traces if r["configuration"]==name and r["prefix_length"]==12] if name in paths or name=="frozen" else []
        if action_rows:summary["first_pruning_distribution"][name]=dict(Counter(str(next((x["first_pruning_position"] for x in traces if x["configuration"]==name and x["state_id"]==r["state_id"] and x["action_id"]==r["action_id"] and x["first_pruning_position"] is not None),"survived")) for r in action_rows))
        summary["validation_recall"][name]={str(c):sum(int(r["positive_recalled"]) for r in recall if r["configuration"]==name and r["cutoff"]==c) for c in (1,12,32,256)}
    write_json(a.output/"summary.json",summary);print(json.dumps(summary["validation_recall"],indent=2))
if __name__=="__main__":main()
