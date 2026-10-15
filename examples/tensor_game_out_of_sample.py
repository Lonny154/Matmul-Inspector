#!/usr/bin/env python3
"""Prepare or execute the preregistered TG-4E experiment."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import subprocess
from matmul_inspector.tensor_game_learning import load_checkpoint
from matmul_inspector.tensor_game_offline_ablation import load_ablation_checkpoint
from matmul_inspector.tensor_game_out_of_sample import *

def main():
    p=argparse.ArgumentParser(); p.add_argument("stage",choices=("prepare","evaluate")); p.add_argument("--output",type=Path,required=True)
    p.add_argument("--known",type=Path,action="append",default=[]); p.add_argument("--baseline-checkpoint",type=Path); p.add_argument("--trained-checkpoint",type=Path)
    a=p.parse_args()
    if a.stage=="prepare":
        if a.output.exists(): raise FileExistsError(a.output)
        targets=generate_fresh_targets(known_paths=a.known); plan=preregistration(targets)
        a.output.mkdir(parents=True); (a.output/"targets.json").write_text(json.dumps(targets,indent=2,sort_keys=True)+"\n")
        (a.output/"preregistration.json").write_text(json.dumps(plan,indent=2,sort_keys=True)+"\n")
        sha=file_digest(a.output/"preregistration.json"); (a.output/"preregistration.sha256").write_text(sha+"  preregistration.json\n")
        print(sha); return
    plan_path=a.output/"preregistration.json"; recorded=(a.output/"preregistration.sha256").read_text().split()[0]
    if file_digest(plan_path)!=recorded: raise ValueError("preregistration hash mismatch")
    targets_payload=json.loads((a.output/"targets.json").read_text()); plan=json.loads(plan_path.read_text())
    if hashlib.sha256(canonical_json(targets_payload)).hexdigest()!=plan["target_content_sha256"]: raise ValueError("target hash mismatch")
    if (a.output/"summary.json").exists(): raise FileExistsError(a.output/"summary.json")
    fm,fc,_=load_checkpoint(a.baseline_checkpoint,"cpu"); tm,tc,_=load_ablation_checkpoint(a.trained_checkpoint,device="cpu")
    models={"frozen":(fm,fc),"trained":(tm,tc)}; targets=targets_payload["targets"]
    recall=reference_recall(targets,models); results,attribution=run_evaluation(targets,models)
    summary={"version":VERSION,"git_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
             "preregistration_sha256":recorded,"metrics":aggregate(results,recall),"environment":environment(),
             "checkpoint_sha256":{"frozen":file_digest(a.baseline_checkpoint/"best_model_state.pt"),"trained":file_digest(a.trained_checkpoint/"best_model_state.pt")}}
    (a.output/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    write_csv(a.output/"results.csv",results); write_csv(a.output/"candidate_recall.csv",recall)
    write_csv(a.output/"hybrid_attribution.csv",attribution,
              ("target_id","seed","step","action_id","attribution","trained_exclusive",
               "candidate_in_successful_solution","causal_necessity_tested","necessary_for_observed_success"))
    print(a.output)
if __name__=="__main__": main()
