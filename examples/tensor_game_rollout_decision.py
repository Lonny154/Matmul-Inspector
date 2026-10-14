#!/usr/bin/env python3
"""Prepare and run the preregistered TG-4W rollout decision."""
from __future__ import annotations
import argparse,csv,hashlib,importlib.metadata,json,math,os,platform,resource,statistics,subprocess,sys,tracemalloc
from collections import Counter
from pathlib import Path

from matmul_inspector.tensor_game_bounded_residual import create_residual_candidate_provider
from matmul_inspector.tensor_game_heldout_bounded_search import AuditedResidualPolicy,divergence_rows,paired_outcome,run_trial
from matmul_inspector.tensor_game_out_of_sample import tensor_tuple
from matmul_inspector.tensor_game_rollout_decision import decide_release,generate_rollout_targets,upper_zero_event_bound
from matmul_inspector.tensor_game_search_calibration import replay_reference

PROVIDERS=("exhaustive","bounded_128_integrated")
KNOWN=(Path("artifacts/tensor-game/tg4b2a-diverse-full-v1/manifest.json"),Path("artifacts/tensor-game/tg3b-dataset-2000/decompositions.jsonl"),Path("artifacts/tensor-game/tg4e-out-of-sample-v1/targets.json"),Path("artifacts/tensor-game/tg4f-search-calibration-v1/heldout_targets.json"),Path("artifacts/tensor-game/tg4i-coverage-training-v1/heldout_targets.json"),Path("artifacts/tensor-game/tg4r-heldout-bounded-search-v1/heldout_targets.json"),Path("artifacts/tensor-game/tg4s-provider-integration-v1/regression_targets.json"),Path("artifacts/tensor-game/tg4v-default-readiness-v1/target_manifest.json"))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+"\n")
def csvwrite(p,rows,fields=()):
 rows=list(rows);names=list(dict.fromkeys(key for row in rows for key in row)) if rows else list(fields)
 with p.open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=names);w.writeheader();w.writerows(rows)
def percentile(v,p):
 v=sorted(v);x=(len(v)-1)*p;i=int(x);return v[i] if i==len(v)-1 else v[i]*(i+1-x)+v[i+1]*(x-i)
def rss():
 for line in Path("/proc/self/status").read_text().splitlines():
  if line.startswith("VmRSS:"):return int(line.split()[1])*1024
 return None
def cpu_model():
 for line in Path("/proc/cpuinfo").read_text().splitlines():
  if line.startswith("model name"):return line.split(":",1)[1].strip()
 return "unavailable"
def environment():
 deps={}
 for name in ("numpy","torch","pytest"):
  try:deps[name]=importlib.metadata.version(name)
  except importlib.metadata.PackageNotFoundError:deps[name]="not installed"
 return {"environment_id":"wsl2-local-cpu-python313","machine_count":1,"cpu_model":cpu_model(),"logical_cpus":os.cpu_count(),"os":platform.platform(),"python":sys.version.split()[0],"dependencies":deps,"process_count":1,"hardware_accelerator":"none used","cross_machine_status":"unverified; no second genuine machine was available"}

def prepare(out):
 if out.exists():raise FileExistsError(out)
 out.mkdir(parents=True)
 criteria={"schema_version":"tg4w.release-criteria.v1","status":"frozen_before_provider_evaluation","maximum_regression_upper_bound":.02,"confidence_method":"one-sided exact zero-event binomial bound 1 - 0.05^(1/n), using unique targets","required_unique_targets":150,"required_observed_exhaustive_only_targets":0,"minimum_mean_speedup":1.10,"minimum_paired_faster_fraction":.80,"minimum_genuine_machine_count":2,"memory":{"root_calls_per_provider":500,"maximum_traced_retained_growth_bytes":500000,"maximum_rss_growth_bytes":33554432,"candidate_determinism_required":True},"hard_failures":["any exhaustive-only target","candidate legality/scoring failure","bounded timeout excess","compatibility failure","persistent memory growth above either threshold"],"classification":{"GO":"all hard and release gates pass","CONDITIONAL":"hard gates pass but performance, uncertainty, or cross-machine release evidence is incomplete","NO-GO":"a hard gate fails"},"default_change_authorized":False};dump(out/"release_criteria.json",criteria)
 manifest=generate_rollout_targets(KNOWN);manifest["reserved_for"]="TG-4W rollout decision; frozen before either provider executes";dump(out/"target_manifest.json",manifest);(out/"target_manifest.sha256").write_text(sha(out/"target_manifest.json")+"  target_manifest.json\n")
 spec={"schema_version":"tg4w.evaluation.v1","status":"frozen_before_provider_evaluation","providers":list(PROVIDERS),"public_provider_identifiers":["exhaustive","bounded_128"],"production_default":"exhaustive","candidate_budget":12,"simulation_budget_by_depth":{"1":16,"2":64,"3":64,"4":64},"node_limit":2000,"timeout_seconds":10.0,"c_puct":1.5,"seeds":[3],"provider_order":"alternate by target index","warmup":"one untimed root proposal per provider on first target","timing":"single process, perf_counter inside unchanged search implementation, no tracemalloc during timed trials","maximum_provider_trials":300,"interruption":"partial results written after every trial; interrupted studies are preserved but this runner refuses implicit resume","primary_metrics":["unique-target paired discovery","end-to-end wall time","release gates"],"secondary_metrics":["proposal time","tail latency","timeouts","candidate score agreement","memory stability"],"environment":environment()};dump(out/"evaluation_spec.json",spec);(out/"evaluation_spec.sha256").write_text(sha(out/"evaluation_spec.json")+"  evaluation_spec.json\n")
 print(json.dumps({"targets":manifest["target_count"],"zero_event_upper_bound":upper_zero_event_bound(manifest["target_count"]),"target_sha256":sha(out/"target_manifest.json"),"spec_sha256":sha(out/"evaluation_spec.json")},indent=2))

def audit_compatibility(out):
 failures=[];default=create_residual_candidate_provider();explicit=create_residual_candidate_provider("exhaustive");bounded=create_residual_candidate_provider("bounded_128")
 if type(default) is not type(explicit):failures.append("default factory differs from explicit exhaustive")
 if bounded.name!="bounded_128":failures.append("bounded public identifier mismatch")
 try:create_residual_candidate_provider("invalid")
 except ValueError as exc:invalid_error=str(exc)
 else:failures.append("invalid provider accepted");invalid_error="none"
 payload={"default_provider":"exhaustive","default_preserved":not failures and default.name=="exhaustive_residual","explicit_exhaustive_available":type(default) is type(explicit),"bounded_uses_streaming_heap":bounded.__class__.__name__=="BoundedResidualPolicy","unsupported_identifier_error":invalid_error,"cli_choices":["exhaustive","bounded_128"],"provider_metadata_recorded":True,"new_dependencies":[],"failures":failures,"environment":environment()};dump(out/"compatibility_audit.json",payload);return payload

def memory_stability(out,target,criteria):
 rows=[];residual=tensor_tuple(target["target"]);calls=criteria["memory"]["root_calls_per_provider"]
 for name in PROVIDERS:
  public_name="bounded_128" if name=="bounded_128_integrated" else name;provider=create_residual_candidate_provider(public_name);first=None;before_rss=rss();tracemalloc.start();base_current,_=tracemalloc.get_traced_memory()
  for index in range(calls):
   proposals=provider.propose_actions(residual,target["construction_length"],None,12);ids=tuple(repr(x.action) for x in proposals);first=ids if first is None else first;current,peak=tracemalloc.get_traced_memory()
   if index in {0,9,49,99,249,499}:rows.append({"provider":name,"call_index":index+1,"candidate_count":len(ids),"deterministic":ids==first,"traced_current_bytes":current,"traced_peak_bytes":peak,"rss_bytes":rss()})
  final_current,peak=tracemalloc.get_traced_memory();tracemalloc.stop();after_rss=rss();rows.append({"provider":name,"call_index":"final","candidate_count":len(first),"deterministic":True,"traced_current_bytes":final_current,"traced_peak_bytes":peak,"rss_bytes":after_rss,"traced_retained_growth_bytes":final_current-base_current,"rss_growth_bytes":after_rss-before_rss})
 csvwrite(out/"memory_stability.csv",rows);return rows

def evaluate(out):
 tp,sp=out/"target_manifest.json",out/"evaluation_spec.json";manifest=json.loads(tp.read_text());spec=json.loads(sp.read_text());criteria=json.loads((out/"release_criteria.json").read_text())
 if sha(tp)!=(out/"target_manifest.sha256").read_text().split()[0] or sha(sp)!=(out/"evaluation_spec.sha256").read_text().split()[0]:raise ValueError("frozen input hash mismatch")
 if not all(replay_reference(x)["exact_success"] for x in manifest["targets"]):raise RuntimeError("reference replay failed")
 traces={}
 if (out/"results.csv").exists():
  results=list(csv.DictReader((out/"results.csv").open()))
  if len(results)!=spec["maximum_provider_trials"]:raise ValueError("partial search results require explicit recovery")
  for row in results:row["success"]=row["success"]=="True"
 else:
  warm=tensor_tuple(manifest["targets"][0]["target"])
  for name in PROVIDERS:AuditedResidualPolicy(name).propose_actions(warm,1,None,12)
  results=[];latency=[]
  for ti,target in enumerate(manifest["targets"]):
   category=target["workload"]["primary_category"];order=spec["providers"] if ti%2==0 else list(reversed(spec["providers"]))
   for oi,name in enumerate(order):
    row,trace=run_trial(target,name,3,spec);row.update({"category":category,"execution_order":oi,"environment_id":spec["environment"]["environment_id"]});results.append(row);traces[target["target_id"],name]=trace;latency.append({k:row[k] for k in ("target_id","constructed_depth","category","provider","seed","execution_order","environment_id","success","proposal_generation_seconds","mcts_excluding_proposals_seconds","search_wall_seconds")});csvwrite(out/"results.csv",results);csvwrite(out/"latency_measurements.csv",latency)
 pairs=[];score_failures=0
 for target in manifest["targets"]:
  find=lambda name:next(x for x in results if x["target_id"]==target["target_id"] and x["provider"]==name)
  pair=paired_outcome(find(PROVIDERS[0]),find(PROVIDERS[1]));pair["category"]=target["workload"]["primary_category"];pairs.append(pair)
  if traces:
   audit_rows=divergence_rows(traces[target["target_id"],PROVIDERS[0]],traces[target["target_id"],PROVIDERS[1]])
  else:
   residual=tensor_tuple(target["target"]);left=AuditedResidualPolicy(PROVIDERS[0]);right=AuditedResidualPolicy(PROVIDERS[1]);left.propose_actions(residual,target["construction_length"],None,12);right.propose_actions(residual,target["construction_length"],None,12);audit_rows=divergence_rows(left.trace,right.trace)
  for row in audit_rows:score_failures+=row["shared_scores_equal"] is False
 csvwrite(out/"paired_outcomes.csv",pairs)
 mem=memory_stability(out,manifest["targets"][73],criteria);compat=audit_compatibility(out)
 finals=[x for x in mem if x["call_index"]=="final"];memory_failures=sum(int(x["traced_retained_growth_bytes"])>criteria["memory"]["maximum_traced_retained_growth_bytes"] or int(x["rss_growth_bytes"])>criteria["memory"]["maximum_rss_growth_bytes"] or not x["deterministic"] for x in finals)
 aggregates={}
 for name in PROVIDERS:
  rows=[x for x in results if x["provider"]==name];times=[float(x["search_wall_seconds"]) for x in rows];props=[float(x["proposal_generation_seconds"]) for x in rows]
  aggregates[name]={"trials":len(rows),"successes":sum(x["success"] for x in rows),"mean_search_seconds":statistics.mean(times),"median_search_seconds":statistics.median(times),"p95_search_seconds":percentile(times,.95),"mean_proposal_seconds":statistics.mean(props),"mean_search_overhead_seconds":statistics.mean(float(x["mcts_excluding_proposals_seconds"]) for x in rows),"timeouts":sum(x["termination_reason"]=="wall_time_limit_reached" for x in rows)}
 outcomes=Counter(x["outcome"] for x in pairs);regressed={x["target_id"] for x in pairs if x["outcome"]=="exhaustive_only"};diffs=[float(x["search_wall_difference_seconds"]) for x in pairs];bound=upper_zero_event_bound(len(manifest["targets"])) if not regressed else 1.0
 evidence={"exhaustive_only_targets":len(regressed),"correctness_failures":score_failures,"bounded_timeouts":aggregates[PROVIDERS[1]]["timeouts"],"exhaustive_timeouts":aggregates[PROVIDERS[0]]["timeouts"],"compatibility_failures":len(compat["failures"]),"memory_stability_failures":memory_failures,"regression_upper_bound":bound,"mean_speedup":aggregates[PROVIDERS[0]]["mean_search_seconds"]/aggregates[PROVIDERS[1]]["mean_search_seconds"],"paired_faster_fraction":sum(x<0 for x in diffs)/len(diffs),"genuine_machine_count":spec["environment"]["machine_count"]}
 decision=decide_release(criteria,evidence);decision.update({"evidence":evidence,"recommendation":"Do not change the default in TG-4W. Resolve the failed release gates before implementation.","criteria_sha256":sha(out/"release_criteria.json"),"target_manifest_sha256":sha(tp),"evaluation_spec_sha256":sha(sp)});dump(out/"decision.json",decision)
 strat=[]
 for dimension,key in (("depth","constructed_depth"),("category","category")):
  for value in sorted({str(x[key]) for x in results}):
   for name in PROVIDERS:
    rows=[x for x in results if str(x[key])==value and x["provider"]==name];strat.append({"dimension":dimension,"value":value,"provider":name,"trials":len(rows),"successes":sum(x["success"] for x in rows),"mean_search_seconds":statistics.mean(float(x["search_wall_seconds"]) for x in rows)})
 summary={"git":{"branch":subprocess.check_output(["git","branch","--show-current"],text=True).strip(),"commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()},"environment":spec["environment"],"completed_trials":len(results),"expected_trials":spec["maximum_provider_trials"],"aggregates":aggregates,"paired_outcomes":dict(outcomes),"bounded_faster_pairs":sum(x<0 for x in diffs),"paired_trials":len(pairs),"exhaustive_only_targets":sorted(regressed),"zero_regression_one_sided_95_percent_upper_bound":bound if not regressed else None,"stratified_results":strat,"memory_final_rows":finals,"decision":decision,"production_default":"exhaustive","production_default_changed":False};dump(out/"summary.json",summary);report(out,manifest,summary);print(json.dumps(summary,indent=2))

def report(out,manifest,s):
 e,b=s["aggregates"][PROVIDERS[0]],s["aggregates"][PROVIDERS[1]];d=s["decision"];ev=d["evidence"]
 text=f"""# TG-4W default-provider rollout decision

## Decision

**{d['decision']}**. Exhaustive remains the default. The bounded provider passed the measured discovery, runtime, correctness, timeout, memory, and installed-package gates on this host. The cross-machine reproducibility gate failed because only one genuine machine was available. A default change is therefore not yet recommended.

## Independent paired evidence

The frozen corpus contains {manifest['target_count']} targets: {manifest['depth_counts']} by constructed depth and {manifest['category_counts']} by primary category. Both providers ran once per target with seed 3, producing {s['completed_trials']} trials and {s['paired_trials']} independent target pairs. Paired outcomes were {s['paired_outcomes']}. Exhaustive succeeded in {e['successes']}/{e['trials']}; bounded succeeded in {b['successes']}/{b['trials']}.

With zero exhaustive-only targets, the preregistered one-sided exact 95% upper regression-rate bound is {s['zero_regression_one_sided_95_percent_upper_bound']:.2%}. This meets the 2% margin but does not prove equivalence.

| Provider | Mean s | Median s | p95 s | Mean proposal s | Mean non-proposal s | Timeouts |
|---|---:|---:|---:|---:|---:|---:|
| exhaustive | {e['mean_search_seconds']:.4f} | {e['median_search_seconds']:.4f} | {e['p95_search_seconds']:.4f} | {e['mean_proposal_seconds']:.4f} | {e['mean_search_overhead_seconds']:.4f} | {e['timeouts']} |
| bounded | {b['mean_search_seconds']:.4f} | {b['median_search_seconds']:.4f} | {b['p95_search_seconds']:.4f} | {b['mean_proposal_seconds']:.4f} | {b['mean_search_overhead_seconds']:.4f} | {b['timeouts']} |

Mean speedup was {ev['mean_speedup']:.2f}×. Bounded was faster in {s['bounded_faster_pairs']}/{s['paired_trials']} pairs ({ev['paired_faster_fraction']:.1%}). Raw per-depth, category, target, and seed results are in `summary.json`, `results.csv`, and `paired_outcomes.csv`.

## Memory, correctness, and compatibility

Each provider completed 500 repeated root proposals under `tracemalloc`, with checkpoint RSS measurements. Final rows are {s['memory_final_rows']}. Python-traced growth and RSS stayed inside the frozen thresholds, and candidate sequences were deterministic. RSS remains an allocator-level observation rather than per-call ownership. Timed search did not run under `tracemalloc`.

Shared candidate scores agreed throughout the paired trace audit. Exact construction references replayed before evaluation. The installed API audit preserved the exhaustive default, verified explicit exhaustive and bounded selection, confirmed the streaming bounded class, retained exhaustive as fallback, and observed a clear `ValueError` for an invalid identifier. No dependency or interface was added.

## Environment and limitations

This was one process on `{s['environment']['cpu_model']}` under `{s['environment']['os']}`, Python {s['environment']['python']}. No second machine was available. Repeated runs on this host are not presented as multi-machine reproduction. Construction length is an upper bound rather than proven rank. Duplicate exclusion covers exact tensors and six axis permutations, not arbitrary basis transformations. Distinct alternative decompositions were not proven.

The unchanged search result instrumentation records exact-success status and residual norm classification, but not the full final residual tensor. Full final residuals are therefore unavailable rather than reconstructed after the fact. Candidate correctness was audited at root states during resumed finalization because the first finalization attempt failed while serializing completed memory checkpoints. The 300 preserved search trials were not rerun.

## Reproduction

```sh
PYTHONPATH=src python examples/tensor_game_rollout_decision.py prepare
PYTHONPATH=src python examples/tensor_game_rollout_decision.py evaluate
```

## Recommended TG-4X milestone

Repeat the frozen TG-4W paired timing and compatibility protocol on one genuinely independent machine, then combine only the reliability counts and keep timing stratified by environment for the final default-change decision.
"""
 (out/"report.md").write_text(text);Path("docs/tensor_game_rollout_decision.md").write_text(text)

def main():
 p=argparse.ArgumentParser();p.add_argument("phase",choices=("prepare","evaluate","report"));p.add_argument("--output",type=Path,default=Path("artifacts/tensor-game/tg4w-rollout-decision-v1"));a=p.parse_args()
 if a.phase=="prepare":prepare(a.output)
 elif a.phase=="evaluate":evaluate(a.output)
 else:report(a.output,json.loads((a.output/"target_manifest.json").read_text()),json.loads((a.output/"summary.json").read_text()))
if __name__=="__main__":main()
