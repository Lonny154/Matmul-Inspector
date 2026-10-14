#!/usr/bin/env python3
"""Prepare and run the preregistered TG-4V default-provider readiness study."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,platform,resource,statistics,subprocess,sys,tracemalloc
from collections import Counter,defaultdict
from pathlib import Path

from matmul_inspector.tensor_game_bounded_residual import create_residual_candidate_provider
from matmul_inspector.tensor_game_default_readiness import decide_readiness,generate_readiness_targets
from matmul_inspector.tensor_game_heldout_bounded_search import AuditedResidualPolicy,divergence_rows,paired_outcome,run_trial
from matmul_inspector.tensor_game_out_of_sample import tensor_tuple
from matmul_inspector.tensor_game_search_calibration import replay_reference

KNOWN=(Path("artifacts/tensor-game/tg4b2a-diverse-full-v1/manifest.json"),Path("artifacts/tensor-game/tg3b-dataset-2000/decompositions.jsonl"),Path("artifacts/tensor-game/tg4e-out-of-sample-v1/targets.json"),Path("artifacts/tensor-game/tg4f-search-calibration-v1/heldout_targets.json"),Path("artifacts/tensor-game/tg4i-coverage-training-v1/heldout_targets.json"),Path("artifacts/tensor-game/tg4r-heldout-bounded-search-v1/heldout_targets.json"),Path("artifacts/tensor-game/tg4s-provider-integration-v1/regression_targets.json"))
PROVIDERS=("exhaustive","bounded_128_integrated")
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write_json(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+"\n")
def write_csv(p,rows,fields=()):
 rows=list(rows);names=list(rows[0]) if rows else list(fields)
 with p.open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=names);w.writeheader();w.writerows(rows)
def pct(v,p):
 v=sorted(v);x=(len(v)-1)*p;i=int(x);return v[i] if i==len(v)-1 else v[i]*(i+1-x)+v[i+1]*(x-i)
def rss_bytes():
 for line in Path("/proc/self/status").read_text().splitlines():
  if line.startswith("VmRSS:"):return int(line.split()[1])*1024
 return None

def prepare(output):
 if output.exists():raise FileExistsError(output)
 output.mkdir(parents=True)
 criteria={"schema_version":"tg4v.readiness.v1","status":"frozen_before_provider_evaluation","decision_scope":"whether evidence supports a later separately reviewed default change; this study never changes the default","hard_gates":["zero invalid references or candidate-score correctness failures","zero unique targets with exhaustive success and bounded failure","bounded timeout count no greater than exhaustive","default-provider and CLI compatibility tests pass"],"minimum_mean_speedup":1.05,"minimum_paired_faster_fraction":.75,"maximum_mean_python_peak_bytes":100000,"maximum_mean_python_retained_bytes":30000,"maximum_zero_regression_upper_bound":.125,"outcomes":{"GO":"all hard, performance, memory, and uncertainty gates pass","CONDITIONAL":"hard gates pass but at least one promotion gate misses","NO-GO":"at least one hard gate fails"},"statistical_unit":"24 independently generated target tensors; repeated seeds are paired repetitions, not independent samples","claim_limit":"a zero-event upper bound is uncertainty, not proof of noninferiority"}
 write_json(output/"readiness_criteria.json",criteria)
 manifest=generate_readiness_targets(KNOWN);manifest["reserved_for"]="TG-4V primary default-readiness evaluation; frozen before provider execution";write_json(output/"target_manifest.json",manifest);(output/"target_manifest.sha256").write_text(sha(output/"target_manifest.json")+"  target_manifest.json\n")
 spec={"schema_version":"tg4v.evaluation.v1","status":"frozen_before_provider_evaluation","providers":list(PROVIDERS),"production_default":"exhaustive","candidate_budget":12,"bounded_pair_budget":128,"simulation_budget_by_depth":{"1":16,"2":64,"3":64,"4":64},"node_limit":2000,"timeout_seconds":10.0,"c_puct":1.5,"seeds":[3,19,35],"provider_order":"alternate by target index plus seed index","maximum_provider_trials":len(manifest["targets"])*3*2,"primary_metrics":["paired exact discovery","search wall latency","correctness"],"secondary_metrics":["proposal latency","candidate divergence","allocation peak and retained bytes","process RSS","timeouts","per-depth and per-category success"],"memory_measurement":"separate untimed root proposal; tracemalloc and process RSS are distinct","stopping":"run every frozen trial once; save partial results after every trial"};write_json(output/"evaluation_spec.json",spec);(output/"evaluation_spec.sha256").write_text(sha(output/"evaluation_spec.json")+"  evaluation_spec.json\n")
 print(json.dumps({"targets":len(manifest["targets"]),"categories":manifest["category_counts"],"target_sha256":sha(output/"target_manifest.json"),"spec_sha256":sha(output/"evaluation_spec.json")},indent=2))

def evaluate(output):
 tp,sp=output/"target_manifest.json",output/"evaluation_spec.json";manifest=json.loads(tp.read_text());spec=json.loads(sp.read_text());criteria=json.loads((output/"readiness_criteria.json").read_text())
 if sha(tp)!=(output/"target_manifest.sha256").read_text().split()[0] or sha(sp)!=(output/"evaluation_spec.sha256").read_text().split()[0]:raise ValueError("frozen input hash mismatch")
 if (output/"results.csv").exists():raise FileExistsError("evaluation already started")
 replay=[{"target_id":x["target_id"],**replay_reference(x)} for x in manifest["targets"]];write_csv(output/"reference_replay.csv",replay)
 if not all(x["exact_success"] for x in replay):raise RuntimeError("reference replay failed")
 results=[];traces={};latency=[]
 for ti,target in enumerate(manifest["targets"]):
  category=target["workload"]["primary_category"]
  for si,seed in enumerate(spec["seeds"]):
   order=spec["providers"] if (ti+si)%2==0 else list(reversed(spec["providers"]))
   for oi,name in enumerate(order):
    row,trace=run_trial(target,name,seed,spec);row.update({"category":category,"execution_order":oi});results.append(row);traces[target["target_id"],seed,name]=trace;latency.append({k:row[k] for k in ("target_id","constructed_depth","category","provider","seed","execution_order","proposal_generation_seconds","mcts_excluding_proposals_seconds","search_wall_seconds")});write_csv(output/"results.csv",results);write_csv(output/"latency_measurements.csv",latency)
 pairs=[];div=[]
 for target in manifest["targets"]:
  for seed in spec["seeds"]:
   find=lambda provider:next(x for x in results if x["target_id"]==target["target_id"] and x["seed"]==seed and x["provider"]==provider)
   pair=paired_outcome(find(PROVIDERS[0]),find(PROVIDERS[1]));pair["category"]=target["workload"]["primary_category"];pairs.append(pair)
   rows=divergence_rows(traces[target["target_id"],seed,PROVIDERS[0]],traces[target["target_id"],seed,PROVIDERS[1]]);first=next((x for x in sorted(rows,key=lambda z:(z["exhaustive_call_index"],z["bounded_call_index"])) if not x["same_order"]),None)
   for x in rows:div.append({"target_id":target["target_id"],"constructed_depth":target["construction_length"],"category":target["workload"]["primary_category"],"seed":seed,"first_divergence":bool(first and (x["state_digest"],x["remaining_moves"])==(first["state_digest"],first["remaining_moves"])),**x})
 write_csv(output/"paired_outcomes.csv",pairs);write_csv(output/"candidate_divergence.csv",div)
 memory=[]
 for target in manifest["targets"]:
  residual=tensor_tuple(target["target"])
  for name in PROVIDERS:
   provider=AuditedResidualPolicy(name);before=rss_bytes();tracemalloc.start();proposals=provider.propose_actions(residual,target["construction_length"],None,12);current,peak=tracemalloc.get_traced_memory();tracemalloc.stop();after=rss_bytes();memory.append({"target_id":target["target_id"],"constructed_depth":target["construction_length"],"category":target["workload"]["primary_category"],"provider":name,"candidate_count":len(proposals),"python_retained_bytes":current,"python_peak_bytes":peak,"process_rss_before_bytes":before,"process_rss_after_bytes":after,"process_rss_delta_bytes":after-before if before is not None and after is not None else "unavailable","process_peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024})
 write_csv(output/"memory_measurements.csv",memory)
 aggregates={}
 for name in PROVIDERS:
  r=[x for x in results if x["provider"]==name];wall=[float(x["search_wall_seconds"]) for x in r];prop=[float(x["proposal_generation_seconds"]) for x in r];mem=[x for x in memory if x["provider"]==name]
  aggregates[name]={"trials":len(r),"successes":sum(x["success"] for x in r),"unique_targets_solved":len({x["target_id"] for x in r if x["success"]}),"mean_search_seconds":statistics.mean(wall),"median_search_seconds":statistics.median(wall),"p95_search_seconds":pct(wall,.95),"mean_proposal_seconds":statistics.mean(prop),"timeouts":sum(x["termination_reason"]=="wall_time_limit_reached" for x in r),"mean_python_peak_bytes":statistics.mean(int(x["python_peak_bytes"]) for x in mem),"mean_python_retained_bytes":statistics.mean(int(x["python_retained_bytes"]) for x in mem)}
 breakdown=[]
 for dimension,key in (("depth","constructed_depth"),("category","category")):
  for value in sorted({str(x[key]) for x in results}):
   for name in PROVIDERS:
    rows=[x for x in results if str(x[key])==value and x["provider"]==name];breakdown.append({"dimension":dimension,"value":value,"provider":name,"trials":len(rows),"successes":sum(x["success"] for x in rows),"mean_search_seconds":statistics.mean(float(x["search_wall_seconds"]) for x in rows)})
 write_csv(output/"stratified_results.csv",breakdown)
 diffs=[float(x["search_wall_difference_seconds"]) for x in pairs];outcomes=Counter(x["outcome"] for x in pairs);regressed={x["target_id"] for x in pairs if x["outcome"]=="exhaustive_only"};upper=1-math.pow(.05,1/len(manifest["targets"])) if not regressed else 1.0
 correctness=sum(not x["shared_scores_equal"] for x in div if x["shared_scores_equal"]!="unavailable")
 integration_failures=0
 try:
  integration_failures+=type(create_residual_candidate_provider()) is not type(create_residual_candidate_provider("exhaustive"));integration_failures+=create_residual_candidate_provider("bounded_128").name!="bounded_128"
 except Exception:integration_failures+=1
 evidence={"correctness_failures":correctness,"exhaustive_only_targets":len(regressed),"bounded_timeouts":aggregates[PROVIDERS[1]]["timeouts"],"exhaustive_timeouts":aggregates[PROVIDERS[0]]["timeouts"],"integration_failures":integration_failures,"mean_speedup":aggregates[PROVIDERS[0]]["mean_search_seconds"]/aggregates[PROVIDERS[1]]["mean_search_seconds"],"paired_faster_fraction":sum(x<0 for x in diffs)/len(diffs),"bounded_mean_python_peak_bytes":aggregates[PROVIDERS[1]]["mean_python_peak_bytes"],"bounded_mean_python_retained_bytes":aggregates[PROVIDERS[1]]["mean_python_retained_bytes"],"regression_upper_bound":upper}
 decision=decide_readiness(criteria,evidence);decision.update({"evidence":evidence,"criteria_sha256":sha(output/"readiness_criteria.json"),"target_manifest_sha256":sha(tp),"evaluation_spec_sha256":sha(sp)});write_json(output/"readiness_decision.json",decision)
 summary={"git":{"branch":subprocess.check_output(["git","branch","--show-current"],text=True).strip(),"commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()},"environment":{"python":sys.version.split()[0],"platform":platform.platform(),"device":"cpu"},"completed_trials":len(results),"expected_trials":spec["maximum_provider_trials"],"aggregates":aggregates,"paired_outcomes":dict(outcomes),"bounded_faster_pairs":sum(x<0 for x in diffs),"paired_trials":len(pairs),"matched_states":len(div),"different_candidate_states":sum(not x["same_order"] for x in div),"target_success_regressions":sorted(regressed),"zero_regression_one_sided_95_percent_upper_bound":upper if not regressed else None,"stratified_results":breakdown,"readiness":decision,"production_default":"exhaustive","production_default_changed":False};write_json(output/"summary.json",summary)
 write_report(output,manifest,summary)
 print(json.dumps(summary,indent=2))

def reanalyze(output):
 """Correct the preregistered decision after the documented harness assertion bug."""
 summary=json.loads((output/"summary.json").read_text());criteria=json.loads((output/"readiness_criteria.json").read_text());old=summary["readiness"];evidence=dict(old["evidence"])
 if evidence["integration_failures"] not in (0,1):raise ValueError("unexpected prior integration result")
 default=create_residual_candidate_provider();explicit=create_residual_candidate_provider("exhaustive");bounded=create_residual_candidate_provider("bounded_128")
 evidence["integration_failures"]=int(type(default) is not type(explicit) or bounded.name!="bounded_128")
 corrected=decide_readiness(criteria,evidence);corrected.update({"evidence":evidence,"criteria_sha256":old["criteria_sha256"],"target_manifest_sha256":old["target_manifest_sha256"],"evaluation_spec_sha256":old["evaluation_spec_sha256"],"analysis_correction":"The original harness compared the established internal exhaustive policy name 'exhaustive_residual' with the public provider identifier 'exhaustive'. The corrected check compares default and explicit factory types and retains the bounded identifier check. Provider trials and frozen inputs were not rerun."})
 correction_path=output/"analysis_correction.json";original=json.loads(correction_path.read_text()) if correction_path.exists() else {"original_decision":old["decision"],"original_integration_failures":old["evidence"]["integration_failures"]}
 write_json(correction_path,{**original,"corrected_decision":corrected["decision"],"corrected_integration_failures":evidence["integration_failures"],"provider_trials_rerun":False,"reason":corrected["analysis_correction"]})
 summary["readiness"]=corrected;write_json(output/"readiness_decision.json",corrected);write_json(output/"summary.json",summary);write_report(output,json.loads((output/"target_manifest.json").read_text()),summary);print(json.dumps(corrected,indent=2))

def write_report(output,manifest,s):
 e,b=s["aggregates"][PROVIDERS[0]],s["aggregates"][PROVIDERS[1]];d=s["readiness"]
 rows=list(csv.DictReader((output/"results.csv").open()));pairs=list(csv.DictReader((output/"paired_outcomes.csv").open()))
 def grouped_latency(provider,success):
  values=[float(x["search_wall_seconds"]) for x in rows if x["provider"]==provider and x["success"].lower()==str(success).lower()]
  return statistics.mean(values) if values else None
 slower=[x for x in pairs if float(x["search_wall_difference_seconds"])>=0]
 depth_lines=[]
 for depth in range(1,5):
  er=[x for x in rows if x["provider"]==PROVIDERS[0] and int(x["constructed_depth"])==depth];br=[x for x in rows if x["provider"]==PROVIDERS[1] and int(x["constructed_depth"])==depth]
  depth_lines.append(f"| {depth} | {sum(x['success']=='True' for x in er)}/{len(er)} | {sum(x['success']=='True' for x in br)}/{len(br)} | {statistics.mean(float(x['search_wall_seconds']) for x in er):.4f} | {statistics.mean(float(x['search_wall_seconds']) for x in br):.4f} |")
 text=f"""# TG-4V default-provider readiness and independent regression study

## Result

The preregistered decision is **{d['decision']}**. Exhaustive remains the production default. The study ran {s['completed_trials']} trials on {manifest['target_count']} independently generated targets. Exhaustive succeeded in {e['successes']}/{e['trials']} trials and integrated bounded residual succeeded in {b['successes']}/{b['trials']}. Paired outcomes were {s['paired_outcomes']}.

Mean end-to-end search time was {e['mean_search_seconds']:.4f} seconds for exhaustive and {b['mean_search_seconds']:.4f} seconds for bounded. That is a {d['evidence']['mean_speedup']:.2f}× speed ratio. Bounded was faster in {s['bounded_faster_pairs']}/{s['paired_trials']} paired trials. Mean bounded Python allocation was {b['mean_python_peak_bytes']/1024:.1f} KiB peak and {b['mean_python_retained_bytes']/1024:.1f} KiB retained.

| Depth | Exhaustive success | Bounded success | Exhaustive mean s | Bounded mean s |
|---:|---:|---:|---:|---:|
{chr(10).join(depth_lines)}

Mean latency on successful trials was {grouped_latency(PROVIDERS[0],True):.4f} seconds exhaustive and {grouped_latency(PROVIDERS[1],True):.4f} bounded. On failed trials it was {grouped_latency(PROVIDERS[0],False):.4f} and {grouped_latency(PROVIDERS[1],False):.4f} seconds. There was one non-improving pair: {slower[0]['target_id']} seed {slower[0]['seed']} ({float(slower[0]['search_wall_difference_seconds']):+.6f} seconds). This case had the same discovery outcome for both providers.

## Frozen decision gates

The criteria, target manifest, and evaluation spec were written and hashed before either provider ran. Hard gates cover reference and scoring correctness, exhaustive-only target regressions, timeouts, and integration compatibility. Promotion gates require at least 1.05× mean speedup, bounded faster in 75% of pairs, bounded mean allocation at most 100,000 peak and 30,000 retained bytes, and a zero-regression one-sided 95% upper bound at most 12.5%.

Failed gates: {d['failed_gates'] or 'none'}.

## Corpus and limitations

The manifest contains six targets at each construction depth from one through four. Primary categories are {manifest['category_counts']}. Exact tensors and all six axis permutations were excluded against the listed historical manifests and within this corpus. This does not establish exclusion under arbitrary basis changes. Construction length is a verified upper bound, not proven tensor rank. “Near-equivalent decomposition” is structural stress labeling; distinct alternative decompositions were not proven.

Repeated seeds are paired repetitions. The independent unit for the regression bound is the {manifest['target_count']} target tensors. A zero-event bound is uncertainty rather than proof of noninferiority. Timing is a single CPU-process observation and Python allocation tracking does not measure all native allocations.

Candidate scores agreed at all {s['matched_states']} matched states where they were available. Candidate ordering or membership differed at {s['different_candidate_states']} matched states, with no asymmetric discovery outcome. Root memory was measured separately from timing for all target/provider pairs. Process RSS deltas are allocator observations, not per-call peaks. Long-running repeated-search memory stability was not measured and is not inferred from those root samples.

## Reproduction

```sh
PYTHONPATH=src python examples/tensor_game_default_readiness.py prepare
PYTHONPATH=src python examples/tensor_game_default_readiness.py evaluate
```

## Recommended TG-4W milestone

Run a separately reviewed default-provider rollout decision using the frozen TG-4V gates plus a larger multi-machine latency corpus, without changing candidate semantics.
"""
 (output/"report.md").write_text(text);Path("docs/tensor_game_default_readiness.md").write_text(text)

def main():
 p=argparse.ArgumentParser();p.add_argument("phase",choices=("prepare","evaluate","reanalyze"));p.add_argument("--output",type=Path,default=Path("artifacts/tensor-game/tg4v-default-readiness-v1"));a=p.parse_args();prepare(a.output) if a.phase=="prepare" else evaluate(a.output) if a.phase=="evaluate" else reanalyze(a.output)
if __name__=="__main__":main()
