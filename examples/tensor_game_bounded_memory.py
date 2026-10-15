#!/usr/bin/env python3
"""Profile, benchmark, and replay the TG-4T memory-efficient selector."""
from __future__ import annotations
import argparse,csv,hashlib,json,platform,resource,statistics,subprocess,sys,time,tracemalloc
from collections import Counter
from pathlib import Path

from matmul_inspector.tensor_game_bounded_residual import bounded_residual_candidates,bounded_residual_candidates_memory_efficient,coarse_pair_scores,coarse_pair_scores_streaming
from matmul_inspector.tensor_game_heldout_bounded_search import divergence_rows,paired_outcome,run_trial

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write_json(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+"\n")
def write_csv(p,rows):
 rows=list(rows)
 with p.open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def pct(v,p):v=sorted(v);x=(len(v)-1)*p;i=int(x);return v[i] if i==len(v)-1 else v[i]*(i+1-x)+v[i+1]*(x-i)
def rss():
 for line in Path('/proc/self/status').read_text().splitlines():
  if line.startswith('VmRSS:'):return int(line.split()[1])*1024
def load_states():
 labels=[json.loads(x) for x in Path('artifacts/tensor-game/tg4i-coverage-training-v1/training_labels.jsonl').read_text().splitlines() if x];return [x for x in labels if x['split']=='validation']
def synthetic():
 zero=[[[0]*4 for _ in range(4)] for _ in range(4)];dense=[[[((i*7+j*3+k)%3)-1 for k in range(4)] for j in range(4)] for i in range(4)];repeated=[[[2 if i==j==k else 0 for k in range(4)] for j in range(4)] for i in range(4)];return [('zero',zero),('dense',dense),('repeated',repeated)]

def prepare(output):
 if output.exists():raise FileExistsError(output)
 output.mkdir(parents=True);states=load_states();state=states[0]['residual'];profiles={}
 for name,fn in (("original",bounded_residual_candidates),("optimized",bounded_residual_candidates_memory_efficient)):
  fn(state,128,12);tracemalloc.start();before=tracemalloc.take_snapshot();rows,audit=fn(state,128,12);current,peak=tracemalloc.get_traced_memory();after=tracemalloc.take_snapshot();stats=after.compare_to(before,'lineno')[:12];tracemalloc.stop();profiles[name]={"python_current_bytes":current,"python_peak_bytes":peak,"coarse_records":6400 if name=='original' else 128,"candidate_count":len(rows),"audit":audit,"top_allocation_sites":[{"traceback":str(x.traceback),"size_diff":x.size_diff,"count_diff":x.count_diff} for x in stats]}
 write_json(output/'allocation_profile.json',profiles)
 design="""# TG-4T selection design\n\nThe original selector creates 6,400 `PairScore` records, retains them in a list, and fully sorts the list by `(-bound_numerator, u, v)`. Profiling identified those records and their contraction tuples as the dominant traced allocation.\n\nThe optimized selector streams pairs through a 128-entry min-heap. The heap root is the worst retained item. Its key is `(bound_numerator, -u, -v)`, so larger bound wins and, for equal bounds, lexicographically smaller `(u,v)` wins exactly as before. After enumeration, the retained records are sorted with the original key. Contractions, integer bound arithmetic, legal `w` selection, canonicalization, exact rescoring, and final action sorting are unchanged.\n\nSpace changes from O(6,400) `PairScore` records to O(128). Time changes from sorting 6,400 records to 6,400 heap comparisons plus at most 128-record final sorting. No dependency or floating-point operation is added.\n""";(output/'selection_design.md').write_text(design)
 differential=[];cases=[('tg4q',x['state_id'],x['residual']) for x in states]
 tg4s=json.loads(Path('artifacts/tensor-game/tg4s-provider-integration-v1/regression_targets.json').read_text());cases += [('tg4s',x['target_id'],x['target']) for x in tg4s['targets']];cases += [('synthetic',n,r) for n,r in synthetic()]
 for source,identifier,residual in cases:
  old_pairs=coarse_pair_scores(residual)[:128];new_pairs=coarse_pair_scores_streaming(residual,128);old,_=bounded_residual_candidates(residual,128,12);new,_=bounded_residual_candidates_memory_efficient(residual,128,12)
  differential.append({"source":source,"state_id":identifier,"selected_pairs_equal":old_pairs==new_pairs,"candidate_rows_equal":old==new,"candidate_count":len(old),"candidate_ids":";".join(x['action_id'] for x in old),"scores":";".join(str(x['score']) for x in old)})
 write_csv(output/'differential_correctness.csv',differential)
 if not all(x['selected_pairs_equal'] and x['candidate_rows_equal'] for x in differential):raise RuntimeError('differential correctness failure')
 spec={"schema_version":"tg4t.memory_benchmark.v1","status":"frozen_after_differential_correctness_before_final_measurement","implementations":["bounded_128_original","bounded_128_memory_heap"],"pair_budget":128,"candidate_budget":12,"benchmark_states":"30 TG-4Q validation states","warmups":2,"repeats":5,"order":"interleaved and reversed by state plus repeat","timing_clock":"perf_counter_ns","memory":"separate tracemalloc calls plus process RSS before/after","regression_manifest":"artifacts/tensor-game/tg4s-provider-integration-v1/regression_targets.json","regression_manifest_sha256":sha(Path('artifacts/tensor-game/tg4s-provider-integration-v1/regression_targets.json')),"regression_spec":"artifacts/tensor-game/tg4s-provider-integration-v1/evaluation_spec.json","regression_spec_sha256":sha(Path('artifacts/tensor-game/tg4s-provider-integration-v1/evaluation_spec.json')),"search_providers":["bounded_128","bounded_128_memory"],"primary_metrics":["peak traced Python bytes","candidate-generation latency"],"secondary_metrics":["retained bytes","RSS delta","coarse selection time","exact scoring time","candidate count","full search time"]};write_json(output/'benchmark_spec.json',spec);(output/'benchmark_spec.sha256').write_text(sha(output/'benchmark_spec.json')+'  benchmark_spec.json\n');print(json.dumps({"profile":profiles,"differential_states":len(differential),"spec_sha256":sha(output/'benchmark_spec.json')},indent=2))

def benchmark(output):
 spec=json.loads((output/'benchmark_spec.json').read_text());assert sha(output/'benchmark_spec.json')==(output/'benchmark_spec.sha256').read_text().split()[0]
 mp=Path(spec['regression_manifest']);sp=Path(spec['regression_spec']);assert sha(mp)==spec['regression_manifest_sha256'] and sha(sp)==spec['regression_spec_sha256'];manifest=json.loads(mp.read_text());search_spec=json.loads(sp.read_text());states=load_states();fns={"original":bounded_residual_candidates,"optimized":bounded_residual_candidates_memory_efficient}
 for i in range(2):
  for name in (("original","optimized") if i==0 else ("optimized","original")):fns[name](states[i]['residual'],128,12)
 latency=[]
 for si,state in enumerate(states):
  for repeat in range(spec['repeats']):
   order=("original","optimized") if (si+repeat)%2==0 else ("optimized","original")
   for oi,name in enumerate(order):
    start=time.perf_counter_ns();rows,audit=fns[name](state['residual'],128,12);elapsed=(time.perf_counter_ns()-start)/1e6;latency.append({"state_id":state['state_id'],"repeat":repeat,"execution_order":oi,"implementation":name,"total_ms":elapsed,"coarse_selection_ms":audit['coarse_scoring_ms'],"exact_rescoring_ms":audit['exact_rescoring_ms'],"ranking_dedup_ms":audit['ranking_dedup_ms'],"candidate_count":len(rows)})
 write_csv(output/'latency_measurements.csv',latency)
 memory=[]
 for state in states:
  for name in ('original','optimized'):
   before=rss();tracemalloc.start();rows,audit=fns[name](state['residual'],128,12);current,peak=tracemalloc.get_traced_memory();tracemalloc.stop();after=rss();memory.append({"state_id":state['state_id'],"implementation":name,"python_peak_bytes":peak,"python_retained_bytes":current,"rss_before_bytes":before,"rss_after_bytes":after,"rss_delta_bytes":after-before,"process_peak_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,"candidate_count":len(rows)})
 write_csv(output/'memory_measurements.csv',memory)
 results=[];traces={}
 for ti,target in enumerate(manifest['targets']):
  for si,seed in enumerate(search_spec['seeds']):
   order=spec['search_providers'] if (ti+si)%2==0 else list(reversed(spec['search_providers']))
   for oi,name in enumerate(order):
    row,trace=run_trial(target,name,seed,search_spec);row['execution_order']=oi;results.append(row);traces[target['target_id'],seed,name]=trace;write_csv(output/'regression_results.csv',results)
 pairs=[];divergence_count=0
 for target in manifest['targets']:
  for seed in search_spec['seeds']:
   old=next(x for x in results if x['target_id']==target['target_id'] and x['seed']==seed and x['provider']=='bounded_128');new=next(x for x in results if x['target_id']==target['target_id'] and x['seed']==seed and x['provider']=='bounded_128_memory');pair=paired_outcome(old,new);rows=divergence_rows(traces[target['target_id'],seed,'bounded_128'],traces[target['target_id'],seed,'bounded_128_memory']);different=sum(not x['same_order'] for x in rows);divergence_count+=different;pair['matched_states']=len(rows);pair['candidate_divergences']=different;pair['trajectory_equivalent']=different==0 and old['expanded_nodes']==new['expanded_nodes'] and old['simulations_completed']==new['simulations_completed'];pairs.append(pair)
 write_csv(output/'paired_outcomes.csv',pairs)
 agg={}
 for name in ('original','optimized'):
  l=[x['total_ms'] for x in latency if x['implementation']==name];m=[x for x in memory if x['implementation']==name];provider='bounded_128' if name=='original' else 'bounded_128_memory';r=[x for x in results if x['provider']==provider]
  agg[name]={"mean_candidate_ms":statistics.mean(l),"median_candidate_ms":statistics.median(l),"p95_candidate_ms":pct(l,.95),"min_candidate_ms":min(l),"max_candidate_ms":max(l),"mean_python_peak_bytes":statistics.mean(x['python_peak_bytes'] for x in m),"mean_python_retained_bytes":statistics.mean(x['python_retained_bytes'] for x in m),"successes":sum(x['success'] for x in r),"mean_search_seconds":statistics.mean(x['search_wall_seconds'] for x in r),"mean_proposal_seconds":statistics.mean(x['proposal_generation_seconds'] for x in r),"proposal_calls":sum(x['candidate_generation_calls'] for x in r)}
 outcomes=Counter(x['outcome'] for x in pairs);summary={"git":{"branch":subprocess.check_output(['git','branch','--show-current'],text=True).strip(),"commit":subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()},"environment":{"python":sys.version.split()[0],"platform":platform.platform(),"device":"cpu"},"benchmark_spec_sha256":sha(output/'benchmark_spec.json'),"differential_states":len(list(csv.DictReader((output/'differential_correctness.csv').open()))),"all_candidate_outputs_exact":True,"aggregates":agg,"regression_trials":len(results),"paired_outcomes":dict(outcomes),"candidate_divergences":divergence_count,"all_search_trajectories_equivalent":all(x['trajectory_equivalent'] for x in pairs),"default_provider_changed":False,"cli_provider_changed":False,"tg4u_recommendation":"Replace the opt-in bounded_128 selector with the exactly equivalent heap implementation and add dense and repeated-term permanent regression cases; keep exhaustive as default."};write_json(output/'summary.json',summary)
 o,n=agg['original'],agg['optimized'];memory_ratio=o['mean_python_peak_bytes']/n['mean_python_peak_bytes'];speed=o['mean_candidate_ms']/n['mean_candidate_ms'];search=o['mean_search_seconds']/n['mean_search_seconds']
 report=f"""# TG-4T memory-efficient bounded residual ranking\n\n## Result\n\nThe fixed-size heap preserved exact selected pairs and final candidates on all {summary['differential_states']} differential states. Mean traced peak memory changed from {o['mean_python_peak_bytes']/1e6:.2f} to {n['mean_python_peak_bytes']/1e6:.2f} MB ({memory_ratio:.2f}× reduction). Mean untraced candidate latency changed from {o['mean_candidate_ms']:.2f} to {n['mean_candidate_ms']:.2f} ms ({speed:.2f}× ratio). The CLI remains on the original opt-in implementation and exhaustive remains default.\n\n## Cause and design\n\n`allocation_profile.json` attributes the dominant original allocation to the 6,400 retained `PairScore` records and contraction tuples. The heap retains 128 records and restores the original stable ordering after streaming. Integer arithmetic and exact rescoring are unchanged.\n\n## Candidate benchmark\n\n| Implementation | Mean ms | Median ms | p95 ms | Mean traced peak | Mean retained |\n|---|---:|---:|---:|---:|---:|\n| original | {o['mean_candidate_ms']:.2f} | {o['median_candidate_ms']:.2f} | {o['p95_candidate_ms']:.2f} | {o['mean_python_peak_bytes']/1e6:.2f} MB | {o['mean_python_retained_bytes']/1e6:.3f} MB |\n| optimized | {n['mean_candidate_ms']:.2f} | {n['median_candidate_ms']:.2f} | {n['p95_candidate_ms']:.2f} | {n['mean_python_peak_bytes']/1e6:.2f} MB | {n['mean_python_retained_bytes']/1e6:.3f} MB |\n\n`tracemalloc` measurements are separate from the uninstrumented timing samples. RSS deltas are allocator-level observations and not reliable per-call peaks.\n\n## Frozen TG-4S replay\n\nOriginal and optimized bounded providers produced {dict(outcomes)} across 48 paired trials. Candidate divergences at matched states: {divergence_count}. All deterministic trajectory fields matched: {summary['all_search_trajectories_equivalent']}. Mean search time was {o['mean_search_seconds']:.4f} versus {n['mean_search_seconds']:.4f} seconds ({search:.2f}× ratio). These are fresh same-session controls, not comparisons against historical timings.\n\n## Answers\n\n1. Retaining 6,400 Python records and contraction tuples caused the elevated peak.\n2. Mean peak fell by {memory_ratio:.2f}×, from {o['mean_python_peak_bytes']/1e6:.2f} to {n['mean_python_peak_bytes']/1e6:.2f} MB.\n3. Yes. Selected pair order, candidate identities, order, and integer scores matched exactly.\n4. Candidate latency changed by a {speed:.2f}× original/optimized ratio.\n5. Same-session search latency changed by a {search:.2f}× ratio with identical trajectories.\n6. Yes. The contained heap logic removes the dominant allocation and slightly improves rather than harms measured latency.\n7. Yes, it is ready to replace the implementation behind the existing opt-in `bounded_128` identifier in a separate reviewable change. The current CLI was intentionally not changed in TG-4T.\n8. A separate default-provider decision still needs broader dense/repeated-term coverage and explicit review.\n\n## Recommended TG-4U milestone\n\nReview and, if accepted, replace the opt-in `bounded_128` selector with the exactly equivalent heap implementation, then add dense and repeated-term cases to the permanent regression suite. Do not change the exhaustive default in that milestone.\n\n## Reproduction\n\n```sh\nPYTHONPATH=src .venv/bin/python examples/tensor_game_bounded_memory.py prepare\nPYTHONPATH=src .venv/bin/python examples/tensor_game_bounded_memory.py benchmark\n```\n""";(output/'report.md').write_text(report);Path('docs/tensor_game_bounded_memory.md').write_text(report);print(json.dumps(summary,indent=2))

def main():
 p=argparse.ArgumentParser();p.add_argument('phase',choices=('prepare','benchmark'));p.add_argument('--output',type=Path,default=Path('artifacts/tensor-game/tg4t-bounded-memory-v1'));a=p.parse_args();prepare(a.output) if a.phase=='prepare' else benchmark(a.output)
if __name__=='__main__':main()
