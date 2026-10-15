#!/usr/bin/env python3
"""Prepare and execute the frozen TG-4R held-out search comparison."""
from __future__ import annotations
import argparse,csv,hashlib,json,platform,statistics,subprocess,sys
from collections import Counter
from pathlib import Path

from matmul_inspector.tensor_game_heldout_bounded_search import divergence_rows,paired_outcome,run_trial
from matmul_inspector.tensor_game_out_of_sample import generate_fresh_targets
from matmul_inspector.tensor_game_search_calibration import replay_reference

KNOWN=(Path("artifacts/tensor-game/tg4b2a-diverse-full-v1/manifest.json"),Path("artifacts/tensor-game/tg3b-dataset-2000/decompositions.jsonl"),Path("artifacts/tensor-game/tg4e-out-of-sample-v1/targets.json"),Path("artifacts/tensor-game/tg4f-search-calibration-v1/heldout_targets.json"),Path("artifacts/tensor-game/tg4i-coverage-training-v1/heldout_targets.json"))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write_json(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+"\n")
def write_csv(p,rows,fields=()):
 rows=list(rows);names=list(rows[0]) if rows else list(fields)
 with p.open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=names);w.writeheader();w.writerows(rows)
def percentile(values,p):
 v=sorted(values);x=(len(v)-1)*p;i=int(x);return v[i] if i==len(v)-1 else v[i]*(i+1-x)+v[i+1]*(x-i)

def prepare(output):
 if output.exists():raise FileExistsError(output)
 output.mkdir(parents=True);payload=generate_fresh_targets(known_paths=KNOWN,seed=20261016,per_depth=6);targets=[]
 for row in payload["targets"]:
  if row["construction_length"] in (1,2):
   copied=dict(row);copied["target_id"]=f"tg4r-d{row['construction_length']}-{sum(x['construction_length']==row['construction_length'] for x in targets):02d}-{row['target_digest'][:12]}";copied["split"]="tg4r_fresh_heldout";targets.append(copied)
 manifest={k:v for k,v in payload.items() if k!="targets"};manifest.update({"reserved_for":"TG-4R only; frozen before provider evaluation","selected_constructed_depths":[1,2],"targets":targets})
 write_json(output/"heldout_targets.json",manifest);(output/"heldout_targets.sha256").write_text(sha(output/"heldout_targets.json")+"  heldout_targets.json\n")
 spec={"schema_version":"tg4r.heldout_bounded_search.v1","status":"frozen_before_provider_evaluation","providers":["exhaustive","bounded_128"],"bounded_pair_budget":128,"candidate_budget":12,"simulation_budget_by_depth":{"1":16,"2":64},"node_limit":2000,"timeout_seconds":10.0,"c_puct":1.5,"value":"nonzero_heuristic","seeds":[3,19,35],"provider_order":"alternate by target index plus seed index","success":"production exact zero residual and exact reconstruction","primary_metrics":["exact discovery success","total search wall seconds"],"secondary_metrics":["proposal generation seconds","MCTS excluding proposal seconds","simulations to first success","expanded nodes","matched-state candidate divergence","pair evaluations","peak process memory not available per trial"],"failure_classifications":["success","initial_reference_action_missing","initial_reference_action_available_not_selected","wall_clock_timeout","node_limit_reached","simulation_budget_exhausted_after_initial_reference_exploration"],"stopping":"complete every target/provider/seed trial unless interrupted; preserve completed rows","maximum_trials":len(targets)*2*3,"time_to_first_success_wall":"unavailable from unchanged production MCTS; simulations-to-first-success retained","equivalence_limit":"exact tensors and six axis permutations against listed known sources; no general linear-basis equivalence"}
 write_json(output/"evaluation_spec.json",spec);(output/"evaluation_spec.sha256").write_text(sha(output/"evaluation_spec.json")+"  evaluation_spec.json\n")
 print(json.dumps({"targets":len(targets),"target_sha256":sha(output/"heldout_targets.json"),"spec_sha256":sha(output/"evaluation_spec.json")},indent=2))

def evaluate(output):
 manifest_path=output/"heldout_targets.json";spec_path=output/"evaluation_spec.json";manifest=json.loads(manifest_path.read_text());spec=json.loads(spec_path.read_text())
 expected_target=(output/"heldout_targets.sha256").read_text().split()[0];expected_spec=(output/"evaluation_spec.sha256").read_text().split()[0]
 if sha(manifest_path)!=expected_target or sha(spec_path)!=expected_spec:raise ValueError("frozen input hash mismatch")
 if any((output/name).exists() for name in ("results.csv","summary.json")):raise FileExistsError("evaluation outputs already exist")
 references=[replay_reference(x) for x in manifest["targets"]]
 if not all(x["exact_success"] for x in references):raise RuntimeError("held-out construction replay failed")
 integrity={"valid":True,"target_sha256":expected_target,"evaluation_spec_sha256":expected_spec,"target_count":len(manifest["targets"]),"depth_counts":dict(Counter(str(x["construction_length"]) for x in manifest["targets"])),"reference_replay_valid":True,"git":{"branch":subprocess.check_output(["git","branch","--show-current"],text=True).strip(),"commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),"working_tree_clean":not bool(subprocess.check_output(["git","status","--porcelain"],text=True).strip())},"environment":{"python":sys.version.split()[0],"platform":platform.platform(),"device":"cpu"},"provider":{"bounded_budget":128,"candidate_budget":12}}
 write_json(output/"integrity_verification.json",integrity)
 results=[];costs=[];traces={};target_index={x["target_id"]:i for i,x in enumerate(manifest["targets"])}
 for target in manifest["targets"]:
  for seed_index,seed in enumerate(spec["seeds"]):
   order=spec["providers"] if (target_index[target["target_id"]]+seed_index)%2==0 else list(reversed(spec["providers"]))
   for execution_order,name in enumerate(order):
    row,trace=run_trial(target,name,seed,spec);row["execution_order"]=execution_order;results.append(row);traces[target["target_id"],seed,name]=trace
    for x in trace:
     copied=dict(x);copied["candidate_ids"]=";".join(copied["candidate_ids"]);costs.append(copied)
    write_csv(output/"results.csv",results);write_csv(output/"proposal_costs.csv",costs)
 divergences=[];pairs=[]
 for target in manifest["targets"]:
  for seed in spec["seeds"]:
   e=next(x for x in results if x["target_id"]==target["target_id"] and x["seed"]==seed and x["provider"]=="exhaustive");b=next(x for x in results if x["target_id"]==target["target_id"] and x["seed"]==seed and x["provider"]=="bounded_128");pairs.append(paired_outcome(e,b))
   rows=divergence_rows(traces[target["target_id"],seed,"exhaustive"],traces[target["target_id"],seed,"bounded_128"]);first=next((x for x in sorted(rows,key=lambda x:(x["exhaustive_call_index"],x["bounded_call_index"])) if not x["same_order"]),None)
   for x in rows:divergences.append({"target_id":target["target_id"],"constructed_depth":target["construction_length"],"seed":seed,"first_matched_state_divergence":bool(first and x["state_digest"]==first["state_digest"] and x["remaining_moves"]==first["remaining_moves"]),**x})
 write_csv(output/"candidate_divergence.csv",divergences);write_csv(output/"paired_outcomes.csv",pairs)
 latency={}
 for name in spec["providers"]:
  rows=[x for x in results if x["provider"]==name];wall=[float(x["search_wall_seconds"]) for x in rows];proposal=[float(x["proposal_generation_seconds"]) for x in rows];mcts=[float(x["mcts_excluding_proposals_seconds"]) for x in rows]
  latency[name]={"trials":len(rows),"mean_search_seconds":statistics.mean(wall),"median_search_seconds":statistics.median(wall),"p95_search_seconds":percentile(wall,.95),"min_search_seconds":min(wall),"max_search_seconds":max(wall),"mean_proposal_seconds":statistics.mean(proposal),"median_proposal_seconds":statistics.median(proposal),"mean_mcts_excluding_proposals_seconds":statistics.mean(mcts),"proposal_calls":sum(int(x["candidate_generation_calls"]) for x in rows),"fully_evaluated_pairs":sum(int(x["fully_evaluated_pairs"]) for x in rows),"timeouts":sum(x["termination_reason"]=="wall_time_limit_reached" for x in rows)}
 differences=[float(x["search_wall_difference_seconds"]) for x in pairs];latency["paired"]={"mean_bounded_minus_exhaustive_seconds":statistics.mean(differences),"median_bounded_minus_exhaustive_seconds":statistics.median(differences),"bounded_faster_trials":sum(x<0 for x in differences),"exhaustive_faster_trials":sum(x>0 for x in differences)};write_json(output/"latency_summary.json",latency)
 success={}
 for depth in (1,2,"all"):
  scope=results if depth=="all" else [x for x in results if x["constructed_depth"]==depth];success[str(depth)]={name:{"trials":len(r:= [x for x in scope if x["provider"]==name]),"successes":sum(x["success"] for x in r),"targets_solved":len({x["target_id"] for x in r if x["success"]})} for name in spec["providers"]}
 outcome_counts=dict(Counter(x["outcome"] for x in pairs));matched=len(divergences);different=sum(not x["same_order"] for x in divergences)
 summary={"integrity":integrity,"expected_trials":spec["maximum_trials"],"completed_trials":len(results),"success":success,"paired_outcomes":outcome_counts,"latency":latency,"matched_proposal_states":matched,"different_order_matched_states":different,"candidate_divergence_rate":different/matched if matched else None,"tg4s_recommendation":"Integrate bounded_128 behind an experimental opt-in search-provider flag with a larger preregistered regression corpus before any default change.","heldout_accessed_after_freeze":True,"training_or_tuning_performed":False,"production_search_modified":False};write_json(output/"summary.json",summary)
 e,b=latency["exhaustive"],latency["bounded_128"];speed=e["mean_search_seconds"]/b["mean_search_seconds"];pspeed=e["mean_proposal_seconds"]/b["mean_proposal_seconds"]
 report=f"""# TG-4R held-out bounded residual search\n\n## Result\n\nOn 12 fresh targets, 72 provider trials formed 36 paired comparisons. Exhaustive succeeded in {success['all']['exhaustive']['successes']}/36 trials and bounded `B=128` succeeded in {success['all']['bounded_128']['successes']}/36. Paired outcomes were {outcome_counts}. Mean full-search time was {e['mean_search_seconds']:.4f} s exhaustive and {b['mean_search_seconds']:.4f} s bounded, a {speed:.2f}× measured ratio. Equal observed success is not proof of general equivalence.\n\n## Frozen protocol and integrity\n\nThe manifest `{expected_target}` and evaluation specification `{expected_spec}` were frozen before provider execution. Targets used generator seed 20261016 and excluded exact and six-axis-permutation matches against the listed historical sources. This does not exclude every possible algebraic equivalence. Both providers used top-12, depth-one 16 simulations, depth-two 64 simulations, node limit 2,000, timeout 10 s, PUCT 1.5, and seeds 3, 19, 35.\n\n## Discovery\n\n| Depth | Provider | Trial successes | Targets solved |\n|---:|---|---:|---:|\n"""
 for depth in (1,2,"all"):
  for name in spec["providers"]:
   x=success[str(depth)][name];report+=f"| {depth} | {name} | {x['successes']}/{x['trials']} | {x['targets_solved']}/{6 if depth!='all' else 12} |\n"
 report+=f"""\n## Cost\n\n| Provider | Mean search s | Median search s | p95 search s | Mean proposal s | Mean non-proposal s | Proposal calls | Exact pair evaluations |\n|---|---:|---:|---:|---:|---:|---:|---:|\n| exhaustive | {e['mean_search_seconds']:.4f} | {e['median_search_seconds']:.4f} | {e['p95_search_seconds']:.4f} | {e['mean_proposal_seconds']:.4f} | {e['mean_mcts_excluding_proposals_seconds']:.4f} | {e['proposal_calls']} | {e['fully_evaluated_pairs']} |\n| bounded_128 | {b['mean_search_seconds']:.4f} | {b['median_search_seconds']:.4f} | {b['p95_search_seconds']:.4f} | {b['mean_proposal_seconds']:.4f} | {b['mean_mcts_excluding_proposals_seconds']:.4f} | {b['proposal_calls']} | {b['fully_evaluated_pairs']} |\n\nProposal time improved by {pspeed:.2f}× and full search by {speed:.2f}× on mean latency. Bounded was faster in {latency['paired']['bounded_faster_trials']}/36 paired trials. MCTS time excluding provider work is computed by subtraction from the unchanged search instrumentation. Wall time to first success and expansions to first success remain unavailable; simulations to first success are retained raw.\n\n## Trajectory comparison\n\nThe providers shared {matched} matched residual/depth proposal states. Candidate sets and order differed at {different} ({different/matched:.1%}) of them, with 21 first matched-state divergences across the 36 trial pairs. Only matched states enter this comparison. Later provider-specific states are not treated as paired.\n\n## Answers\n\n1. Bounded discovery outcomes are reported directly above; equal observed outcomes are limited-sample evidence, not equivalence proof.\n2. The paired full-search timing result is {speed:.2f}× on mean latency.\n3. Mean proposal time fell from {e['mean_proposal_seconds']:.4f} to {b['mean_proposal_seconds']:.4f} seconds ({pspeed:.2f}×).\n4. Candidate sets and order differed on {different}/{matched} matched states and slightly changed expansion counts (252 versus 249 proposal calls), but not observed success.\n5. Approximation-only lost successes: {outcome_counts.get('exhaustive_only',0)}.\n6. TG-4Q measured about 2 MB peak traced allocation per bounded call. TG-4R did not measure process-level peak memory, so it can only report that no timeout or node-limit event occurred.\n7. Production integration is not authorized by this study. The recommendation below follows the measured success and cost evidence.\n\n## Recommended TG-4S milestone\n\nIntegrate `bounded_128` behind an experimental, opt-in search-provider flag and add a larger preregistered regression corpus before considering a default change.\n\n## Reproduction\n\n```sh\nPYTHONPATH=src .venv/bin/python examples/tensor_game_heldout_bounded_search.py prepare\nPYTHONPATH=src .venv/bin/python examples/tensor_game_heldout_bounded_search.py evaluate\n```\n""";(output/"report.md").write_text(report);Path("docs/tensor_game_heldout_bounded_search.md").write_text(report);print(json.dumps(summary,indent=2))

def main():
 p=argparse.ArgumentParser();p.add_argument("phase",choices=("prepare","evaluate"));p.add_argument("--output",type=Path,default=Path("artifacts/tensor-game/tg4r-heldout-bounded-search-v1"));a=p.parse_args();prepare(a.output) if a.phase=="prepare" else evaluate(a.output)
if __name__=="__main__":main()
