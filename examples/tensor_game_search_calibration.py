#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,hashlib,json,platform,subprocess,sys
from pathlib import Path
from matmul_inspector.tensor_game_learning import load_checkpoint
from matmul_inspector.tensor_game_neural_mcts import NeuralActionPolicy
from matmul_inspector.tensor_game_search_calibration import *

def write(path,rows):
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    p=argparse.ArgumentParser();p.add_argument("--targets",type=Path,required=True);p.add_argument("--checkpoint",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    payload=json.loads(a.targets.read_text());targets=payload["targets"];a.output.mkdir(parents=True)
    replay=[replay_reference(x) for x in targets];write(a.output/"reference_replay.csv",replay)
    if not all(x["exact_success"] for x in replay):raise RuntimeError("reference correctness failure")
    model,config,_=load_checkpoint(a.checkpoint,"cpu"); rows=[]
    # Include sub-16 budgets to locate the reference-path expansion threshold.
    # Omit 1024 because this MCTS records first success but continues all simulations.
    budgets=(4,8,16,64,256); seeds=(3,19,35)
    # Depth one first, then extend reference controls after correctness is established.
    for t in targets:
        target=tuple(tuple(tuple(r) for r in plane) for plane in t["target"]); factors=tuple(action_from_payload(x) for x in t["canonical_factors"])
        controls=("reference_only","reference_augmented","residual_norm") if t["construction_length"]==1 else ("reference_only","reference_augmented")
        for control in controls:
            for budget in budgets:
                for seed in seeds:
                    if control=="reference_only":policy=ReferencePathPolicy(target,factors)
                    elif control=="reference_augmented":policy=ReferencePathPolicy(target,factors,NeuralActionPolicy(model,config,"cpu"),True)
                    else:policy=ExhaustiveResidualPolicy()
                    row={"target_id":t["target_id"],"constructed_depth":t["construction_length"],"control":control,"seed":seed,**run_search(target,policy,budget,seed,t["construction_length"])}
                    row["failure_classification"]=classify(row);rows.append(row)
    write(a.output/"budget_sweep.csv",rows);write(a.output/"reference_injection.csv",[x for x in rows if x["control"].startswith("reference")]);write(a.output/"residual_control.csv",[x for x in rows if x["control"]=="residual_norm"])
    write(a.output/"failure_classification.csv",[{k:x[k] for k in ("target_id","constructed_depth","control","seed","simulations_budget","failure_classification")} for x in rows])
    groups={}
    for key in sorted({(x["constructed_depth"],x["control"],x["simulations_budget"]) for x in rows}):
        r=[x for x in rows if (x["constructed_depth"],x["control"],x["simulations_budget"])==key]
        groups["/".join(map(str,key))]={"successes":sum(x["success"] for x in r),"trials":len(r),"mean_wall_seconds":sum(x["search_wall_seconds"] for x in r)/len(r),"failures":dict(__import__('collections').Counter(x["failure_classification"] for x in r))}
    torch=__import__("torch")
    summary={"git_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),"targets_sha256":hashlib.sha256(a.targets.read_bytes()).hexdigest(),
      "checkpoint_sha256":hashlib.sha256((a.checkpoint/"best_model_state.pt").read_bytes()).hexdigest(),"groups":groups,
      "environment":{"python":sys.version.split()[0],"torch":torch.__version__,"platform":platform.platform(),"processor":platform.processor() or "unknown","device":"cpu"},"all_references_valid":True}
    (a.output/"calibration_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n");print(a.output)
if __name__=="__main__":main()
