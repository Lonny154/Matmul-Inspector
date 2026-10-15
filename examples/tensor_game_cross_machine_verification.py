#!/usr/bin/env python3
"""Portable TG-4X verification using the exact frozen TG-4W inputs."""
from __future__ import annotations
import argparse,csv,hashlib,importlib.metadata,json,os,platform,shutil,statistics,subprocess,sys,tempfile,time
from collections import Counter
from pathlib import Path

from matmul_inspector.tensor_game_bounded_residual import create_residual_candidate_provider
from matmul_inspector.tensor_game_search import HeuristicValueEvaluator,SampledMCTS,SearchConfig

TARGET_SHA="c55792e3866d2064eae411c5427fdb7b2513b910f5670bbb37d439ea03776e75"
SPEC_SHA="1f51262e5b1737812d091c8183fd1eb09c26a30b21c89d876d1a873d325ebdab"
RESULT_FIELDS=("target_id","target_digest","constructed_depth","category","provider","public_provider","seed","execution_order","environment_id","success","simulations_budget","simulations_completed","expanded_nodes","unique_states","candidate_evaluations","proposal_generation_seconds","search_overhead_seconds","search_wall_seconds","termination_reason")

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def tensor(value):return tuple(tuple(tuple(row) for row in plane) for plane in value)
def atomic_json(path,value):
 path.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.NamedTemporaryFile("w",dir=path.parent,delete=False) as f:json.dump(value,f,indent=2,sort_keys=True);f.write("\n");temporary=Path(f.name)
 temporary.replace(path)
def atomic_csv(path,rows):
 path.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.NamedTemporaryFile("w",newline="",dir=path.parent,delete=False) as f:w=csv.DictWriter(f,fieldnames=RESULT_FIELDS);w.writeheader();w.writerows(rows);temporary=Path(f.name)
 temporary.replace(path)
def load_inputs(root):
 target_path=root/"target_manifest.json";spec_path=root/"evaluation_spec.json"
 observed={"target_manifest.json":digest(target_path),"evaluation_spec.json":digest(spec_path)}
 expected={"target_manifest.json":TARGET_SHA,"evaluation_spec.json":SPEC_SHA}
 if observed!=expected:raise ValueError(f"frozen input hash mismatch: expected {expected}, observed {observed}")
 manifest=json.loads(target_path.read_text());spec=json.loads(spec_path.read_text())
 if manifest["target_count"]!=150 or spec["seeds"]!=[3] or spec["candidate_budget"]!=12 or spec["production_default"]!="exhaustive":raise ValueError("frozen TG-4W semantics mismatch")
 return manifest,spec,observed
def dependency_versions():
 values={}
 for name in ("matmul-inspector","numpy","torch"):
  try:values[name]=importlib.metadata.version(name)
  except importlib.metadata.PackageNotFoundError:values[name]="not installed"
 return values
def cpu_model():
 try:
  for line in Path("/proc/cpuinfo").read_text().splitlines():
   if line.startswith("model name") or line.startswith("Model"):
    return line.split(":",1)[1].strip()
 except OSError:pass
 return platform.processor() or "unavailable"
def command_output(args):
 try:return subprocess.check_output(args,text=True,stderr=subprocess.STDOUT).strip()
 except (OSError,subprocess.CalledProcessError) as exc:return f"unavailable: {exc}"
def host_metadata(hashes,runner):
 commit=command_output(["git","rev-parse","HEAD"]);status=command_output(["git","status","--porcelain"])
 identity={"cpu_model":cpu_model(),"machine":platform.machine(),"logical_cpus":os.cpu_count(),"os":platform.platform(),"python":sys.version.split()[0]}
 environment_id=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16]
 return {"schema_version":"tg4x.host.v1","environment_id":environment_id,**identity,"dependencies":dependency_versions(),"process_count":1,"git_revision":commit,"git_dirty":bool(status) and not status.startswith("unavailable:"),"git_status":status,"frozen_artifact_hashes":hashes,"runner_sha256":digest(runner),"captured_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}
def compatibility(manifest,cli):
 failures=[];default=create_residual_candidate_provider();explicit=create_residual_candidate_provider("exhaustive");bounded=create_residual_candidate_provider("bounded_128")
 if type(default) is not type(explicit):failures.append("default factory differs from explicit exhaustive")
 if bounded.name!="bounded_128":failures.append("bounded provider identity mismatch")
 try:create_residual_candidate_provider("invalid")
 except ValueError as exc:invalid_error=str(exc)
 else:invalid_error="not rejected";failures.append("invalid Python provider accepted")
 executable=shutil.which(cli);smokes={}
 if not executable:failures.append(f"installed CLI not found: {cli}")
 else:
  target=manifest["targets"][0];common=[executable,"tensor-game-search","--target-json",str(compatibility.target_path),"--target-id",target["target_id"],"--simulations","1"]
  for label,extra in (("default",[]),("bounded",["--candidate-provider","bounded_128"])):
   try:smokes[label]=json.loads(subprocess.check_output(common+extra,text=True,stderr=subprocess.STDOUT))
   except (subprocess.CalledProcessError,json.JSONDecodeError) as exc:failures.append(f"installed CLI {label} smoke failed: {exc}")
  if smokes.get("default",{}).get("candidate_provider")!="exhaustive" or smokes.get("default",{}).get("provider_is_default") is not True:failures.append("installed CLI default changed")
  if smokes.get("bounded",{}).get("candidate_provider")!="bounded_128" or smokes.get("bounded",{}).get("provider_is_default") is not False:failures.append("installed CLI bounded metadata mismatch")
  invalid=subprocess.run(common+["--candidate-provider","invalid"],text=True,capture_output=True)
  invalid_cli={"returncode":invalid.returncode,"stderr":invalid.stderr.strip()}
  if invalid.returncode==0 or "invalid choice" not in invalid.stderr:failures.append("installed CLI invalid provider did not fail clearly")
 return {"default_provider":"exhaustive","default_preserved":not any("default" in x for x in failures),"explicit_exhaustive_available":type(default) is type(explicit),"bounded_available":bounded.name=="bounded_128","invalid_python_provider_error":invalid_error,"installed_cli":executable,"installed_cli_smokes":smokes,"invalid_installed_cli_provider":invalid_cli if executable else "not run","failures":failures}

def schedule(manifest,spec,limit=None):
 targets=manifest["targets"][:limit]
 rows=[]
 for ti,target in enumerate(targets):
  for si,seed in enumerate(spec["seeds"]):
   providers=spec["providers"] if (ti+si)%2==0 else list(reversed(spec["providers"]))
   for order,provider in enumerate(providers):rows.append((target,seed,provider,order))
 return rows
def key_from_row(row):return row["target_id"],int(row["seed"]),row["provider"]
def validate_resume(rows,planned):
 if len(rows)>len(planned):raise ValueError("results contain more rows than the frozen schedule")
 keys=[]
 for row in rows:
  missing=set(RESULT_FIELDS)-set(row)
  if missing:raise ValueError(f"resume row misses fields: {sorted(missing)}")
  keys.append(key_from_row(row))
 expected=[(target["target_id"],seed,provider) for target,seed,provider,_ in planned[:len(rows)]]
 if keys!=expected:raise ValueError("completed rows are not an exact prefix of the deterministic schedule")
 if len(set(keys))!=len(keys):raise ValueError("duplicate completed trial")
def trial(target_record,seed,provider_name,order,spec,environment_id):
 public="bounded_128" if provider_name=="bounded_128_integrated" else provider_name;provider=create_residual_candidate_provider(public);depth=target_record["construction_length"]
 result=SampledMCTS(tensor(target_record["target"]),provider,SearchConfig(simulations=spec["simulation_budget_by_depth"][str(depth)],max_actions=spec["candidate_budget"],max_moves=depth,seed=seed,c_puct=spec["c_puct"],node_limit=spec["node_limit"],max_wall_time_seconds=spec["timeout_seconds"]),HeuristicValueEvaluator()).run()
 return {"target_id":target_record["target_id"],"target_digest":target_record["target_digest"],"constructed_depth":depth,"category":target_record["workload"]["primary_category"],"provider":provider_name,"public_provider":public,"seed":seed,"execution_order":order,"environment_id":environment_id,"success":result.exact_reconstruction,"simulations_budget":spec["simulation_budget_by_depth"][str(depth)],"simulations_completed":result.simulations_run,"expanded_nodes":result.expanded_nodes,"unique_states":result.unique_states,"candidate_evaluations":result.accepted_proposals,"proposal_generation_seconds":result.proposal_generation_seconds,"search_overhead_seconds":result.search_wall_seconds-result.proposal_generation_seconds,"search_wall_seconds":result.search_wall_seconds,"termination_reason":result.termination_reason}
def summarize(rows,metadata,mode):
 aggregates={}
 for provider in ("exhaustive","bounded_128_integrated"):
  selected=[x for x in rows if x["provider"]==provider];times=[float(x["search_wall_seconds"]) for x in selected]
  aggregates[provider]={"trials":len(selected),"successes":sum(str(x["success"]).lower()=="true" for x in selected),"mean_search_seconds":statistics.mean(times),"median_search_seconds":statistics.median(times),"minimum_search_seconds":min(times),"maximum_search_seconds":max(times),"mean_proposal_seconds":statistics.mean(float(x["proposal_generation_seconds"]) for x in selected),"timeouts":sum(x["termination_reason"]=="wall_time_limit_reached" for x in selected)}
 by_key={(x["target_id"],int(x["seed"]),x["provider"]):x for x in rows};pairs=[]
 for target_id,seed in sorted({(x["target_id"],int(x["seed"])) for x in rows}):
  a,b=by_key[target_id,seed,"exhaustive"],by_key[target_id,seed,"bounded_128_integrated"];sa,sb=(str(a["success"]).lower()=="true",str(b["success"]).lower()=="true")
  pairs.append({"target_id":target_id,"seed":seed,"outcome":"both_succeed" if sa and sb else "exhaustive_only" if sa else "bounded_only" if sb else "both_fail","bounded_minus_exhaustive_seconds":float(b["search_wall_seconds"])-float(a["search_wall_seconds"])})
 return {"schema_version":"tg4x.summary.v1","mode":mode,"host":metadata,"completed_trials":len(rows),"aggregates":aggregates,"paired_outcomes":dict(Counter(x["outcome"] for x in pairs)),"bounded_faster_pairs":sum(x["bounded_minus_exhaustive_seconds"]<0 for x in pairs),"paired_trials":len(pairs),"timing_policy":"per-host only; do not pool heterogeneous hosts","pairs":pairs}
def execute(args,smoke):
 root=args.inputs.resolve();manifest,spec,hashes=load_inputs(root);compatibility.target_path=root/"target_manifest.json";metadata=host_metadata(hashes,Path(__file__));out=args.output.resolve();out.mkdir(parents=True,exist_ok=True);atomic_json(out/"host_metadata.json",metadata);atomic_json(out/"compatibility.json",compatibility(manifest,args.cli))
 planned=schedule(manifest,spec,args.smoke_targets if smoke else None);result_path=out/"results.csv";rows=list(csv.DictReader(result_path.open())) if result_path.exists() else [];validate_resume(rows,planned)
 if not rows:
  first=manifest["targets"][0]
  for public in ("exhaustive","bounded_128"):create_residual_candidate_provider(public).propose_actions(tensor(first["target"]),first["construction_length"],None,spec["candidate_budget"])
 for target_record,seed,provider,order in planned[len(rows):]:rows.append(trial(target_record,seed,provider,order,spec,metadata["environment_id"]));atomic_csv(result_path,rows)
 summary=summarize(rows,metadata,"smoke" if smoke else "full");atomic_json(out/"summary.json",summary);return summary
def compare(args):
 left=list(csv.DictReader(args.left.open()));right=list(csv.DictReader(args.right.open()));lk={(x["target_id"],x["provider"],int(x["seed"])):x for x in left};rk={(x["target_id"],x["provider"],int(x["seed"])):x for x in right};shared=sorted(set(lk)&set(rk));mismatches=[{"target_id":k[0],"provider":k[1],"seed":k[2],"left_success":lk[k]["success"],"right_success":rk[k]["success"]} for k in shared if str(lk[k]["success"]).lower()!=str(rk[k]["success"]).lower()]
 def stats(rows):
  return {p:{"trials":len(v),"mean_search_seconds":statistics.mean(float(x["search_wall_seconds"]) for x in v),"median_search_seconds":statistics.median(float(x["search_wall_seconds"]) for x in v)} for p in ("exhaustive","bounded_128_integrated") if (v:=[x for x in rows if x["provider"]==p])}
 payload={"schema_version":"tg4x.comparison.v1","shared_trial_keys":len(shared),"missing_from_left":[list(x) for x in sorted(set(rk)-set(lk))],"missing_from_right":[list(x) for x in sorted(set(lk)-set(rk))],"discovery_mismatches":mismatches,"left_timing":stats(left),"right_timing":stats(right),"timing_pooled":False};atomic_json(args.output,payload);return payload
def inspect(args):
 manifest,spec,hashes=load_inputs(args.inputs.resolve());return {"integrity":"pass","hashes":hashes,"targets":manifest["target_count"],"seeds":spec["seeds"],"candidate_budget":spec["candidate_budget"],"simulation_budget_by_depth":spec["simulation_budget_by_depth"],"planned_full_trials":len(schedule(manifest,spec))}
def main():
 p=argparse.ArgumentParser();sub=p.add_subparsers(dest="command",required=True)
 for name in ("inspect","smoke","run"):
  q=sub.add_parser(name);q.add_argument("--inputs",type=Path,required=True)
  if name!="inspect":q.add_argument("--output",type=Path,required=True);q.add_argument("--cli",default="matmul-inspector");q.add_argument("--smoke-targets",type=int,default=4)
 q=sub.add_parser("compare");q.add_argument("--left",type=Path,required=True);q.add_argument("--right",type=Path,required=True);q.add_argument("--output",type=Path,required=True)
 a=p.parse_args();payload=inspect(a) if a.command=="inspect" else execute(a,a.command=="smoke") if a.command in {"smoke","run"} else compare(a);print(json.dumps(payload,indent=2,sort_keys=True))
if __name__=="__main__":main()
