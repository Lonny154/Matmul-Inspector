#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,platform,subprocess
from dataclasses import asdict
from pathlib import Path
from matmul_inspector.tensor_game_diversity_search import DiscoveryConfig,discover
from matmul_inspector.tensor_game_hardware_aware_search import HardwareAwareProvider

def encoded(value):return (json.dumps(value,sort_keys=True,separators=(",",":"))+"\n").encode()

def main():
 p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);p.add_argument("--seeds",type=int,nargs="+",default=(3,11,29));p.add_argument("--ranks",type=int,nargs="+",default=(7,8));p.add_argument("--penalties",nargs="+",default=("0","0.25","0.5","1"));p.add_argument("--node-budget",type=int,default=256);p.add_argument("--beam-width",type=int,default=128);p.add_argument("--max-actions",type=int,default=12);p.add_argument("--candidate-pool",type=int,default=128);p.add_argument("--trace-limit",type=int,default=8);a=p.parse_args()
 if a.output.exists():p.error("--output must not exist")
 policies=[("baseline","weighted","0")]+[(f"weighted_{x}","weighted",x) for x in a.penalties if x!="0"]+[("pareto","pareto","0")];trials=[]
 for name,strategy,penalty in policies:
  for rank in a.ranks:
   for seed in a.seeds:
    provider=HardwareAwareProvider(penalty,strategy,a.candidate_pool,a.trace_limit);config=DiscoveryConfig(rank,seed,a.node_budget,a.beam_width,a.max_actions,"baseline",1);result=discover(config,provider);result.update({"policy":name,"ranking_strategy":strategy,"preparation_penalty":penalty,"diagnostic_summary":provider.diagnostic_summary(),"diagnostic_trace":provider.trace});trials.append(result);print(json.dumps({"policy":name,"rank":rank,"seed":seed,"success":result["success"],"nodes":result["nodes_expanded"],"seconds":result["wall_seconds"]}),flush=True)
 records={row["candidate_id"]:row for trial in trials for row in trial["solutions"]};summary={}
 for name,_,_ in policies:
  for rank in a.ranks:
   group=[x for x in trials if x["policy"]==name and x["config"]["rank"]==rank];solutions=[r for x in group for r in x["solutions"]];summary[f"{name}:rank{rank}"]={"trials":len(group),"successes":sum(x["success"] for x in group),"nodes_expanded":sum(x["nodes_expanded"] for x in group),"wall_seconds":sum(x["wall_seconds"] for x in group),"unique_representations":len({r["candidate_id"] for r in solutions}),"canonical_classes":len({r["canonical_id"] for r in solutions}),"structural_signatures":len({r["structural_signature_id"] for r in solutions}),"arithmetic_cost_points":len({(r["rank"],r["static_cost"]["pre_additions"],r["static_cost"]["post_additions"],r["static_cost"]["coefficient_scalings"]) for r in solutions}),"pre_additions":sorted({r["static_cost"]["pre_additions"] for r in solutions}),"post_additions":sorted({r["static_cost"]["post_additions"] for r in solutions}),"total_support":sorted({sum(v!=0 for action in r["actions"] for v in action["u"]+action["v"]+action["w"]) for r in solutions})}
 a.output.mkdir(parents=True);trial_bytes=encoded({"schema":"tensor_game_hardware_aware_search.v1","configuration":vars(a)|{"output":str(a.output)},"environment":{"python":platform.python_version(),"platform":platform.platform()},"git_commit":subprocess.run(["git","rev-parse","HEAD"],capture_output=True,text=True).stdout.strip(),"trials":trials});candidate_bytes=b"".join(encoded(x) for x in sorted(records.values(),key=lambda x:x["candidate_id"]));(a.output/"trials.json").write_bytes(trial_bytes);(a.output/"candidates.jsonl").write_bytes(candidate_bytes);manifest={"schema":"tensor_game_hardware_aware_search_summary.v1","summary":summary,"unique_archived_solutions":len(records),"files":{"trials.json":{"bytes":len(trial_bytes),"sha256":hashlib.sha256(trial_bytes).hexdigest()},"candidates.jsonl":{"bytes":len(candidate_bytes),"sha256":hashlib.sha256(candidate_bytes).hexdigest()}}};(a.output/"summary.json").write_bytes(encoded(manifest));print(json.dumps(manifest,indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
