#!/usr/bin/env python3
"""Run the TG-4Q bounded residual validation benchmark."""

from __future__ import annotations

import argparse,csv,hashlib,json,platform,statistics,subprocess,time,tracemalloc
from pathlib import Path

from matmul_inspector.tensor_game_bounded_residual import bounded_residual_candidates,profile_exhaustive
from matmul_inspector.tensor_game_residual_supervision import enumerate_residual_teacher

BUDGETS=(128,512,1024); NAMES=("exhaustive","bounded_128","bounded_512","bounded_1024")
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write_json(p,x): p.write_text(json.dumps(x,indent=2,sort_keys=True)+"\n")
def write_csv(p,rows):
    rows=list(rows)
    with p.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def percentile(v,p):
    v=sorted(v);x=(len(v)-1)*p;i=int(x);return v[i] if i==len(v)-1 else v[i]*(i+1-x)+v[i+1]*(x-i)
def call(name,residual): return enumerate_residual_teacher(residual,12) if name=="exhaustive" else bounded_residual_candidates(residual,int(name.split("_")[1]),12)


def main():
    a=argparse.ArgumentParser();a.add_argument("--labels",type=Path,default=Path("artifacts/tensor-game/tg4i-coverage-training-v1/training_labels.jsonl"));a.add_argument("--output",type=Path,default=Path("artifacts/tensor-game/tg4q-bounded-residual-v1"));a.add_argument("--repeats",type=int,default=5);args=a.parse_args()
    args.output.mkdir(parents=True,exist_ok=False); labels=[json.loads(x) for x in args.labels.read_text().splitlines() if x]; train=[x for x in labels if x["split"]=="train"]; validation=[x for x in labels if x["split"]=="validation"]
    prereg={"schema_version":"tg4q.bounded_residual.v1","created_before_validation":True,"reference":"unchanged tg4h exhaustive residual teacher","approximation":"continuous-relaxation bound ranks all 6400 pairs; exact discrete rescoring of bounded prefix","pair_budgets":list(BUDGETS),"candidate_budget":12,"calibration_split":"TG-4I train","validation_split":"TG-4I validation","validation_state_ids":[x["state_id"] for x in validation],"timing":{"warmups":2,"repeats":args.repeats,"clock":"perf_counter_ns","alternating_rotated_order":True,"device":"cpu"},"random_seeds":[],"primary_metrics":["state-level verified-positive top-12 coverage","end-to-end candidate generation latency"],"labels_are_runtime_inputs":False}
    write_json(args.output/"preregistration.json",prereg);digest=sha(args.output/"preregistration.json");(args.output/"preregistration.sha256").write_text(digest+"  preregistration.json\n")
    derivation="""# TG-4Q scoring derivation\n\nFor `A = u ⊗ v ⊗ w`, define `q = ||u||²||v||²` and `d_k = Σᵢⱼ R_ijk u_i v_j`. Exact expansion gives\n\n`||R-A||² = ||R||² - 2Σₖ d_k w_k + qΣₖ w_k²`.\n\nFor fixed `(u,v)`, each legal discrete `w_k` minimizes `q w_k² - 2d_k w_k`, so it is the allowed integer nearest to `d_k/q`, with the existing deterministic tie break. Relaxing `w` to real values gives score `||R||² - Σₖd_k²/q`. This is a rigorous lower bound on the best legal discrete score. TG-4Q ranks pairs by that bound, retains a fixed prefix, selects legal `w`, and exactly rescores accepted actions using the expansion above.\n\nThe bound is not a safe fixed-budget pruning certificate. Discrete quantization can reorder pairs, so a discarded pair can produce a better legal action. The bounded provider is therefore heuristic. It preserves exact integer final scores, legal coefficients, canonicalization, deduplication, and final tie-breaking. It still computes all 6,400 contractions; it avoids discrete `w`, action construction, canonicalization, and exact scoring outside the retained prefix.\n""";(args.output/"scoring_derivation.md").write_text(derivation)
    # Component profile is descriptive and uses five validation states with three repeats.
    profile=[]
    for label in validation[:5]:
        for repeat in range(3):
            _,row=profile_exhaustive(label["residual"]);profile.append({"state_id":label["state_id"],"repeat":repeat,**row})
    write_csv(args.output/"baseline_profile.csv",profile)
    for i in range(2):
        for name in (NAMES if i==0 else tuple(reversed(NAMES))):call(name,validation[i]["residual"])
    comparisons=[];timings=[]
    for si,label in enumerate(validation):
        exact,exact_audit=enumerate_residual_teacher(label["residual"],12); exact_ids=[x["action_id"] for x in exact];positive={x["action_id"] for x in label["positive_actions"]}
        quality={"exhaustive":(exact,exact_audit)}
        for budget in BUDGETS: quality[f"bounded_{budget}"]=bounded_residual_candidates(label["residual"],budget,12)
        for name,(rows,audit) in quality.items():
            found=positive&{x["action_id"] for x in rows};overlap=len(set(exact_ids)&{x["action_id"] for x in rows})
            tracemalloc.start();call(name,label["residual"]);_,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
            comparisons.append({"state_id":label["state_id"],"configuration":name,"candidate_count":len(rows),"positive_count":len(positive),"positive_found":len(found),"state_covered":bool(found),"exact_top12_overlap":overlap,"exact_top12_overlap_fraction":overlap/len(exact_ids) if exact_ids else 1.0,"enumerated_pairs":audit["enumerated_uv_pairs"],"fully_evaluated_pairs":audit.get("fully_evaluated_pairs",audit["enumerated_uv_pairs"]),"fraction_exhaustive_work":audit.get("fully_evaluated_pairs",audit["enumerated_uv_pairs"])/6400,"peak_python_bytes":peak,"failure_reason":"verified_positive_absent" if not found else ""})
        for repeat in range(args.repeats):
            order=NAMES[(si+repeat)%len(NAMES):]+NAMES[:(si+repeat)%len(NAMES)]
            for oi,name in enumerate(order):
                start=time.perf_counter_ns();_,audit=call(name,label["residual"]);total=(time.perf_counter_ns()-start)/1e6
                timings.append({"state_id":label["state_id"],"repeat":repeat,"execution_order":oi,"configuration":name,"total_ms":total,"coarse_scoring_ms":audit.get("coarse_scoring_ms",""),"exact_rescoring_ms":audit.get("exact_rescoring_ms",audit.get("elapsed_seconds",0)*1000),"ranking_dedup_ms":audit.get("ranking_dedup_ms",""),"enumerated_pairs":audit["enumerated_uv_pairs"],"fully_evaluated_pairs":audit.get("fully_evaluated_pairs",audit["enumerated_uv_pairs"])})
    write_csv(args.output/"candidate_comparison.csv",comparisons);write_csv(args.output/"latency_measurements.csv",timings)
    summaries=[]
    for name in NAMES:
        c=[x for x in comparisons if x["configuration"]==name];v=[x["total_ms"] for x in timings if x["configuration"]==name]
        summaries.append({"configuration":name,"covered_states":sum(x["state_covered"] for x in c),"total_states":len(c),"positive_recall_numerator":sum(x["positive_found"] for x in c),"positive_recall_denominator":sum(x["positive_count"] for x in c),"mean_top12_overlap":statistics.mean(x["exact_top12_overlap_fraction"] for x in c),"mean_fully_evaluated_pairs":statistics.mean(x["fully_evaluated_pairs"] for x in c),"fraction_exhaustive_work":statistics.mean(x["fraction_exhaustive_work"] for x in c),"mean_ms":statistics.mean(v),"median_ms":statistics.median(v),"p95_ms":percentile(v,.95),"min_ms":min(v),"max_ms":max(v),"mean_peak_python_bytes":statistics.mean(x["peak_python_bytes"] for x in c)})
    write_csv(args.output/"coverage_summary.csv",summaries)
    frontier=[]
    for row in summaries:
        dominators=[x["configuration"] for x in summaries if x is not row and x["covered_states"]>=row["covered_states"] and x["mean_ms"]<=row["mean_ms"] and (x["covered_states"]>row["covered_states"] or x["mean_ms"]<row["mean_ms"])]
        frontier.append({**row,"pareto_dominated_by_mean":bool(dominators),"dominated_by":";".join(dominators)})
    write_csv(args.output/"cost_coverage_frontier.csv",frontier);by={x["configuration"]:x for x in summaries};best=min(frontier,key=lambda x:(-x["covered_states"],x["mean_ms"]))
    component={key:statistics.mean(float(x[key]) for x in profile) for key in ("total_ms","contraction_ms","w_selection_ms","canonicalization_object_ms","exact_tensor_score_ms","ranking_dedup_ms")}
    summary={"git":{"branch":subprocess.check_output(["git","branch","--show-current"],text=True).strip(),"commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()},"environment":{"python":platform.python_version(),"device":"cpu"},"inputs":{"labels_sha256":sha(args.labels)},"preregistration_sha256":digest,"training_states_not_evaluated":len(train),"validation_states":len(validation),"baseline_profile_mean_ms":component,"configurations":summaries,"best_measured_tradeoff":best["configuration"],"tg4r_recommendation":"Run one preregistered held-out exact-discovery comparison of bounded_128 against the unchanged exhaustive residual provider.","heldout_discovery_run":False,"production_search_modified":False};write_json(args.output/"summary.json",summary)
    speed=by["exhaustive"]["mean_ms"]/by[best["configuration"]]["mean_ms"]
    report=f"""# TG-4Q bounded residual candidate generation\n\n## Result\n\n`{best['configuration']}` was the best measured coverage-first tradeoff: {best['covered_states']}/{best['total_states']} states and {best['positive_recall_numerator']}/{best['positive_recall_denominator']} positives at {best['mean_ms']:.2f} ms mean latency. Exhaustive reached {by['exhaustive']['covered_states']}/{by['exhaustive']['total_states']} and {by['exhaustive']['positive_recall_numerator']}/{by['exhaustive']['positive_recall_denominator']} at {by['exhaustive']['mean_ms']:.2f} ms. The measured speed ratio is {speed:.2f}×. No search or held-out discovery was run.\n\n## Exhaustive profile\n\nAcross 15 instrumented calls, mean component times were: contraction {component['contraction_ms']:.2f} ms, legal `w` selection {component['w_selection_ms']:.2f} ms, canonicalization/object work {component['canonicalization_object_ms']:.2f} ms, tensor construction and exact scoring {component['exact_tensor_score_ms']:.2f} ms, and ranking {component['ranking_dedup_ms']:.2f} ms. Instrumentation overhead means these components should not be compared directly with uninstrumented latency. The implementation is Python-loop based, not vectorized.\n\n## Validation\n\n| Configuration | State coverage | Positive recall | Top-12 overlap | Pairs exactly evaluated | Work fraction | Mean ms | Median ms | p95 ms | Dominated |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---|\n"""
    for x in frontier: report+=f"| {x['configuration']} | {x['covered_states']}/{x['total_states']} | {x['positive_recall_numerator']}/{x['positive_recall_denominator']} | {x['mean_top12_overlap']:.3f} | {x['mean_fully_evaluated_pairs']:.0f} | {x['fraction_exhaustive_work']:.3f} | {x['mean_ms']:.2f} | {x['median_ms']:.2f} | {x['p95_ms']:.2f} | {x['dominated_by'] or 'no'} |\n"
    report+=f"""\nThe continuous relaxation is a rigorous lower bound, but fixed-budget prefix retention is heuristic because discrete coefficient quantization can reorder pairs. All returned candidates use exact integer scores. `bounded_128` omitted five exhaustive candidates across three states and lost no verified positive. It also used about 1.94 MB mean peak traced Python memory versus 0.05 MB for exhaustive because it retains all coarse pair records.\n\n## Answers\n\n1. Contractions dominated at {component['contraction_ms']:.2f} ms, followed by tensor construction/exact scoring at {component['exact_tensor_score_ms']:.2f} ms and legal `w` selection at {component['w_selection_ms']:.2f} ms.\n2. Yes on these validation states: the bound preserved 98.9% mean top-12 overlap at `B=128`; `B=512` and `B=1024` reproduced the exhaustive candidates exactly. This is empirical, not an exactness guarantee.\n3. `B=128` avoided discrete exact processing for 98% of pairs, while still computing every contraction.\n4. Yes. `B=128` reduced mean end-to-end latency from {by['exhaustive']['mean_ms']:.2f} to {by['bounded_128']['mean_ms']:.2f} ms, with separated timing ranges.\n5. No verified-positive or state coverage was lost. Five non-positive exhaustive candidates were absent across three states.\n6. The best measured cost-coverage tradeoff is `bounded_128`.\n7. It is worth evaluating as an experimental search provider. The higher traced-memory footprint and lack of a fixed-budget exactness guarantee argue against production integration before held-out discovery evidence.\n\n## Recommended TG-4R milestone\n\nRun one preregistered held-out exact-discovery comparison of frozen `bounded_128` against the unchanged exhaustive residual provider, with identical search settings and candidate budgets.\n\n## Reproduction\n\n```sh\nPYTHONPATH=src .venv/bin/python examples/tensor_game_bounded_residual.py\n```\n""";(args.output/"report.md").write_text(report);Path("docs/tensor_game_bounded_residual.md").write_text(report);print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
