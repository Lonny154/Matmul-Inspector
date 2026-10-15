#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,platform,subprocess,sys
from pathlib import Path
from matmul_inspector.tensor_game_heldout_evaluation import *
from matmul_inspector.tensor_game_learning import load_checkpoint
from matmul_inspector.tensor_game_offline_ablation import load_ablation_checkpoint
SPEC="4716b2531a9c8102778e49589891417df76335024717fbb95248525ba8705573";MANIFEST="dc5646879d626346ba0db08b13c0c375c0e0e52d5b2eec16249b02466aa66eb3"
def write(path,rows,fields=()):
 with path.open("w",newline="") as f:
  names=list(rows[0]) if rows else list(fields);w=csv.DictWriter(f,fieldnames=names)
  if names:w.writeheader()
  if rows:w.writerows(rows)
def main():
 p=argparse.ArgumentParser();p.add_argument("--spec",type=Path,required=True);p.add_argument("--targets",type=Path,required=True);p.add_argument("--baseline",type=Path,required=True);p.add_argument("--trained",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 spec,manifest,observed=verify(a.spec,a.targets,SPEC,MANIFEST);a.output.mkdir(parents=True);(a.output/"trials").mkdir()
 torch=__import__("torch"); integrity={"expected":{"evaluation_spec":SPEC,"heldout_targets":MANIFEST},"observed":observed,"valid":True,
  "git_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),"branch":subprocess.check_output(["git","branch","--show-current"],text=True).strip(),
  "working_tree_status":subprocess.check_output(["git","status","--short"],text=True).splitlines(),
  "checkpoint_sha256":{"frozen":sha(a.baseline/"best_model_state.pt"),"trained":sha(a.trained/"best_model_state.pt")},
  "environment":{"python":sys.version.split()[0],"torch":torch.__version__,"platform":platform.platform(),"device":"cpu"}}
 (a.output/"integrity_verification.json").write_text(json.dumps(integrity,indent=2,sort_keys=True)+"\n");(a.output/"evaluation_config.json").write_text(json.dumps(spec,indent=2,sort_keys=True)+"\n")
 fm,fc,_=load_checkpoint(a.baseline,"cpu");tm,tc,_=load_ablation_checkpoint(a.trained,device="cpu");models={"frozen":(fm,fc),"trained":(tm,tc)}
 recall=recall_rows(manifest["targets"],models);write(a.output/"reference_recall.csv",recall)
 rows=[];attrs=[]
 for target in manifest["targets"]:
  for seed in spec["seeds"]:
   for name in spec["policies"]:
    row,extra=run_trial(target,name,seed,spec,models);rows.append(row);attrs.extend(extra)
    (a.output/"trials"/f"{target['target_id']}-{seed}-{name}.json").write_text(json.dumps(row,indent=2,sort_keys=True)+"\n")
 write(a.output/"results.csv",rows);write(a.output/"search_costs.csv",rows);pairs=paired(rows);write(a.output/"paired_outcomes.csv",pairs)
 write(a.output/"hybrid_attribution.csv",attrs,("target_id","seed","step","action_id","source"))
 write(a.output/"failure_classification.csv",[{k:x[k] for k in ("target_id","constructed_depth","provider","seed","failure_classification","termination_reason")} for x in rows])
 summary={"integrity":integrity,"expected_trials":96,"completed_trials":len(rows),"metrics":aggregate(rows,recall),"hybrid_attribution":attrs,"paired_transition_counts":dict(Counter(x["baseline"]+"_vs_"+x["candidate"]+":"+x["transition"] for x in pairs))}
 (a.output/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n");print(a.output)
if __name__=="__main__":main()
