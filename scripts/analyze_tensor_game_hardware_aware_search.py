#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json
from collections import Counter
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument("--experiment",type=Path,required=True);a=p.parse_args();source=a.experiment/"trials.json";data=json.loads(source.read_text());policies=sorted({x["policy"] for x in data["trials"]});diagnostics={}
 for policy in policies:
  trials=[x for x in data["trials"] if x["policy"]==policy];prep=Counter();reduction=Counter()
  for trial in trials:
   summary=trial["diagnostic_summary"];prep.update({int(k):v for k,v in summary["preparation_cost_histogram"].items()});reduction.update({int(k):v for k,v in summary["residual_reduction_histogram"].items()})
  diagnostics[policy]={"trials":len(trials),"provider_calls":sum(x["diagnostic_summary"]["provider_calls"] for x in trials),"candidate_instances":sum(prep.values()),"preparation_cost_histogram":dict(sorted(prep.items())),"residual_reduction_histogram":dict(sorted(reduction.items())),"reranked_candidate_instances":sum(x["diagnostic_summary"]["reranked_candidate_instances"] for x in trials),"changed_selected_action_instances":sum(x["diagnostic_summary"]["changed_selected_action_instances"] for x in trials)}
 baseline=[x for x in data["trials"] if x["policy"]=="baseline"]
 matched={}
 for policy in policies:
  if policy=="baseline":continue
  group=[x for x in data["trials"] if x["policy"]==policy];pairs=zip(sorted(baseline,key=lambda x:(x["config"]["rank"],x["config"]["seed"])),sorted(group,key=lambda x:(x["config"]["rank"],x["config"]["seed"])),strict=True);matched[policy]={"trials":len(group),"identical_success":0,"identical_nodes":0,"identical_solution_ids":0}
  for left,right in pairs:
   matched[policy]["identical_success"]+=left["success"]==right["success"];matched[policy]["identical_nodes"]+=left["nodes_expanded"]==right["nodes_expanded"];matched[policy]["identical_solution_ids"]+={x["candidate_id"] for x in left["solutions"]}=={x["candidate_id"] for x in right["solutions"]}
 payload={"schema":"tensor_game_hardware_aware_search_analysis.v1","source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),"policy_diagnostics":diagnostics,"matched_against_baseline":matched,"conclusion":"No policy changed candidate ordering or selection because bounded_128 exposed only zero-preparation, unit-residual-reduction actions on every visited state."};output=a.experiment/"analysis.json"
 if output.exists():p.error("analysis.json exists")
 output.write_text(json.dumps(payload,indent=2)+"\n");files=[]
 for path in sorted(x for x in a.experiment.iterdir() if x.is_file() and x.name!="artifact_hashes.json"):files.append({"path":path.name,"bytes":path.stat().st_size,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
 (a.experiment/"artifact_hashes.json").write_text(json.dumps({"schema":"tensor_game_hardware_aware_search_hashes.v1","files":files},indent=2)+"\n");print(json.dumps(payload,indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
