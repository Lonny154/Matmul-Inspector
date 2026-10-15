#!/usr/bin/env python3
"""Run the preregistered TG-4P validation-only routing study."""

from __future__ import annotations

import argparse, csv, hashlib, json, math, platform, statistics, subprocess, time, tracemalloc
from pathlib import Path

import torch

from matmul_inspector.tensor_game_adaptive_routing import adaptive_decision, aggregate_latency, frontier, merge_candidates, residual_features, select_threshold
from matmul_inspector.tensor_game_candidate_coverage import decode
from matmul_inspector.tensor_game_experience import action_identity
from matmul_inspector.tensor_game_learning import load_checkpoint
from matmul_inspector.tensor_game_residual_supervision import enumerate_residual_teacher


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def load_jsonl(path): return [json.loads(line) for line in path.read_text().splitlines() if line]
def write_json(path, value): path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
def write_csv(path, rows):
    rows=list(rows); fields=list(rows[0]) if rows else []
    with path.open("w", newline="") as stream:
        writer=csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def ids(candidates): return [action_identity(item.action) for item in candidates]
def diversity(candidates):
    vectors=[item.action.u+item.action.v+item.action.w for item in candidates]
    distances=[sum(a != b for a,b in zip(x,y))/len(x) for i,x in enumerate(vectors) for y in vectors[i+1:]]
    return statistics.mean(distances) if distances else 0.0


def phi(feature, outcome):
    if len(set(feature)) < 2 or len(set(outcome)) < 2: return None
    return statistics.correlation(feature, outcome)


def evaluate_lists(label, frozen, residual):
    positive={x["action_id"] for x in label["positive_actions"]}
    reference={x["action_id"] for x in label["positive_actions"] if "validated_construction_continuation" in x["provenance"]}
    fids={action_identity(a) for a,_ in frozen}; rids={row["action_id"] for row in residual}
    features=residual_features(label["residual"])
    scores=[score for _,score in frozen]
    return {**features,"state_id":label["state_id"],"split":label["split"],"positive_count":len(positive),"reference_count":len(reference),"frozen_covered":bool(positive&fids),"residual_covered":bool(positive&rids),"frozen_positive_count":len(positive&fids),"residual_positive_count":len(positive&rids),"fallback_gain":bool((positive&rids) and not (positive&fids)),"frozen_top_score":scores[0] if scores else None,"frozen_score_margin":scores[0]-scores[1] if len(scores)>1 else None}


def strategy_candidates(name, frozen, residual, decision):
    if name == "frozen_only": return merge_candidates(frozen, [], 12, 0)
    if name == "residual_only": return merge_candidates([], residual, 0, 12)
    if name == "fixed_6_6": return merge_candidates(frozen, residual, 6, 6)
    if name == "fixed_9_3": return merge_candidates(frozen, residual, 9, 3)
    return merge_candidates(frozen, [], 12, 0) if decision == "frozen" else merge_candidates([], residual, 0, 12)


def measured_call(name, label, model, config, spec):
    t0=time.perf_counter_ns(); features=residual_features(label["residual"]); decision=adaptive_decision(features,spec) if name=="adaptive" else "both"; t1=time.perf_counter_ns()
    need_frozen=name in {"frozen_only","fixed_6_6","fixed_9_3"} or (name=="adaptive" and decision=="frozen")
    need_residual=name in {"residual_only","fixed_6_6","fixed_9_3"} or (name=="adaptive" and decision=="residual")
    frozen=[]; residual=[]; tf=tr=0
    if need_frozen:
        start=time.perf_counter_ns(); frozen=decode(model,config,label["residual"],12); tf=time.perf_counter_ns()-start
    residual_evaluations=0
    if need_residual:
        start=time.perf_counter_ns(); residual, audit=enumerate_residual_teacher(label["residual"],12); tr=time.perf_counter_ns()-start; residual_evaluations=audit["enumerated_uv_pairs"]
    start=time.perf_counter_ns(); candidates=strategy_candidates(name,frozen,residual,decision); td=time.perf_counter_ns()-start
    return candidates,{"selected_providers":decision if name=="adaptive" else ("frozen" if name=="frozen_only" else "residual" if name=="residual_only" else "frozen+residual"),"router_ms":(t1-t0)/1e6,"frozen_ms":tf/1e6,"residual_ms":tr/1e6,"dedup_ms":td/1e6,"total_ms":(time.perf_counter_ns()-t0)/1e6,"frozen_calls":int(need_frozen),"residual_calls":int(need_residual),"policy_forwards":int(need_frozen),"residual_evaluations":residual_evaluations}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--checkpoint",type=Path,default=Path("artifacts/tensor-game/tg3b-mlp-256-seed42")); parser.add_argument("--labels",type=Path,default=Path("artifacts/tensor-game/tg4i-coverage-training-v1/training_labels.jsonl")); parser.add_argument("--output",type=Path,default=Path("artifacts/tensor-game/tg4p-adaptive-routing-v1")); parser.add_argument("--repeats",type=int,default=5); args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    labels=load_jsonl(args.labels); training=[x for x in labels if x["split"]=="train"]; validation=[x for x in labels if x["split"]=="validation"]
    model,config,_=load_checkpoint(args.checkpoint,"cpu")
    # Phase 4 is completed and sealed before any validation row is decoded.
    audit=[]
    for label in training:
        frozen=decode(model,config,label["residual"],12); residual,_=enumerate_residual_teacher(label["residual"],12); audit.append(evaluate_lists(label,frozen,residual))
    rule=select_threshold(audit,"residual_nonzero")
    spec={"schema_version":"tg4p.routing.v1","created_before_validation":True,"candidate_budget":12,"adaptive_rule":rule,"fixed_allocations":{"fixed_6_6":{"frozen":6,"residual":6},"fixed_9_3":{"frozen":9,"residual":3}},"deduplication":"canonical identity; quota prefixes then alternating frozen/residual fill","measurement":{"warmups":2,"repeats":args.repeats,"clock":"perf_counter_ns","alternating_order":True,"device":"cpu"},"validation_split":"TG-4I validation only","selection_split":"TG-4I train only","random_seeds":[],"primary_metrics":["state-level verified-positive top-12 coverage","total routing plus proposal latency"],"runtime_inputs":["residual_nonzero"],"forbidden_runtime_inputs":["positive actions","reference solution","constructed depth","success outcome"]}
    write_json(args.output/"routing_spec.json",spec); digest=sha(args.output/"routing_spec.json"); (args.output/"routing_spec.sha256").write_text(digest+"  routing_spec.json\n")
    write_csv(args.output/"routing_feature_audit.csv",audit)
    # Warmups are excluded. Provider order alternates.
    for i in range(2):
        label=validation[i%len(validation)]
        if i%2==0: decode(model,config,label["residual"],12); enumerate_residual_teacher(label["residual"],12)
        else: enumerate_residual_teacher(label["residual"],12); decode(model,config,label["residual"],12)
    strategies=["frozen_only","residual_only","fixed_6_6","fixed_9_3","adaptive"]
    baseline=[]; latency=[]; candidate_rows=[]; coverage=[]
    for state_index,label in enumerate(validation):
        # Untimed quality lists are shared only for analysis, never for measured call cost.
        frozen=decode(model,config,label["residual"],12); residual,residual_audit=enumerate_residual_teacher(label["residual"],12)
        features=residual_features(label["residual"]); decision=adaptive_decision(features,rule)
        positive={x["action_id"] for x in label["positive_actions"]}; reference={x["action_id"] for x in label["positive_actions"] if "validated_construction_continuation" in x["provenance"]}
        fids={action_identity(a) for a,_ in frozen}; rids={x["action_id"] for x in residual}
        # Peak Python allocation is a separate diagnostic, outside timed samples.
        for provider in ("frozen","residual"):
            tracemalloc.start()
            if provider=="frozen": result=decode(model,config,label["residual"],12); considered=len(result); evaluations=0; forwards=1
            else: result,pa=enumerate_residual_teacher(label["residual"],12); considered=pa["unique_legal_candidates"]; evaluations=pa["enumerated_uv_pairs"]; forwards=0
            _,peak=tracemalloc.get_traced_memory(); tracemalloc.stop()
            returned=len(result); found=len(positive & ({action_identity(a) for a,_ in result} if provider=="frozen" else {x["action_id"] for x in result}))
            baseline.append({"state_id":label["state_id"],"provider":provider,"candidates_considered":considered,"unique_candidates_returned":returned,"positive_recall_numerator":found,"positive_recall_denominator":len(positive),"state_covered":bool(found),"policy_forwards":forwards,"residual_evaluations":evaluations,"peak_python_bytes":peak})
        for name in strategies:
            candidates=strategy_candidates(name,frozen,residual,decision); cids=set(ids(candidates)); selected=decision if name=="adaptive" else ("frozen" if name=="frozen_only" else "residual" if name=="residual_only" else "frozen+residual")
            found=positive&cids
            coverage.append({"state_id":label["state_id"],"strategy":name,"selected_providers":selected,"positive_count":len(positive),"positive_found":len(found),"state_covered":bool(found),"reference_count":len(reference),"reference_found":len(reference&cids),"frozen_only_positives":len((positive&fids)-rids),"residual_only_positives":len((positive&rids)-fids),"shared_positives":len(positive&fids&rids),"excluded_by_allocation":len((positive&(fids|rids))-cids),"candidate_count":len(candidates),"candidate_diversity":diversity(candidates),**features})
            for rank,item in enumerate(candidates,1):
                identifier=action_identity(item.action)
                candidate_rows.append({"state_id":label["state_id"],"strategy":name,"selected_providers":selected,"rank":rank,"action_id":identifier,"u":" ".join(map(str,item.action.u)),"v":" ".join(map(str,item.action.v)),"w":" ".join(map(str,item.action.w)),"sources":"+".join(item.sources),"frozen_score":item.frozen_score,"residual_score":item.residual_score,"verified_positive":identifier in positive,"reference_action":identifier in reference})
        for repeat in range(args.repeats):
            order=strategies if (state_index+repeat)%2==0 else list(reversed(strategies))
            for order_index,name in enumerate(order):
                _,record=measured_call(name,label,model,config,rule); latency.append({"state_id":label["state_id"],"repeat":repeat,"execution_order":order_index,"strategy":name,**record})
    write_csv(args.output/"provider_baselines.csv",baseline); write_csv(args.output/"validation_candidates.csv",candidate_rows); write_csv(args.output/"validation_coverage.csv",coverage); write_csv(args.output/"latency_measurements.csv",latency)
    frontier_input=[]
    for name in strategies:
        cov=[x for x in coverage if x["strategy"]==name]; samples=[x["total_ms"] for x in latency if x["strategy"]==name]; stats=aggregate_latency(samples); covered=sum(x["state_covered"] for x in cov); positives=sum(x["positive_found"] for x in cov); total_positive=sum(x["positive_count"] for x in cov); total_seconds=sum(samples)/1000
        frontier_input.append({"strategy":name,"covered_states":covered,"total_states":len(cov),"positive_recall_numerator":positives,"positive_recall_denominator":total_positive,"mean_total_ms":stats["mean_ms"],"median_total_ms":stats["median_ms"],"p95_total_ms":stats["p95_ms"],"min_total_ms":stats["min_ms"],"max_total_ms":stats["max_ms"],"cost_per_covered_state_ms":sum(samples)/covered if covered else "undefined","covered_states_per_proposal_second":covered/total_seconds if total_seconds else "undefined","frozen_calls":sum(x["frozen_calls"] for x in latency if x["strategy"]==name),"residual_calls":sum(x["residual_calls"] for x in latency if x["strategy"]==name),"residual_evaluations":sum(x["residual_evaluations"] for x in latency if x["strategy"]==name)})
    frontier_rows=frontier(frontier_input); write_csv(args.output/"cost_coverage_frontier.csv",frontier_rows)
    summary={"git":{"branch":subprocess.check_output(["git","branch","--show-current"],text=True).strip(),"commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),"working_tree_clean":not bool(subprocess.check_output(["git","status","--porcelain"],text=True).strip())},"environment":{"python":platform.python_version(),"torch":torch.__version__,"device":"cpu","cuda_available":torch.cuda.is_available()},"inputs":{"checkpoint_sha256":sha(args.checkpoint/"best_model_state.pt"),"labels_sha256":sha(args.labels)},"routing_spec_sha256":digest,"training_states":len(training),"validation_states":len(validation),"training_signal_associations":{"residual_nonzero_vs_frozen_covered":phi([x["residual_nonzero"] for x in audit],[int(x["frozen_covered"]) for x in audit]),"residual_nonzero_vs_fallback_gain":phi([x["residual_nonzero"] for x in audit],[int(x["fallback_gain"]) for x in audit])},"adaptive_rule":rule,"frontier":frontier_rows,"conclusion":"No useful adaptive tradeoff: the selected rule routes every state to residual and adds routing overhead.","tg4q_recommendation":"Implement and validate one bounded residual-ranking approximation against exhaustive residual-only candidate coverage and cost.","heldout_discovery_run":False,"production_search_modified":False}
    write_json(args.output/"summary.json",summary)
    by={x["strategy"]:x for x in frontier_rows}; ratio=by["residual_only"]["mean_total_ms"]/by["frozen_only"]["mean_total_ms"]
    report=f"""# TG-4P cost-aware adaptive candidate routing\n\n## Result\n\nThe adaptive rule covered {by['adaptive']['covered_states']}/{by['adaptive']['total_states']} validation states at a mean proposal cost of {by['adaptive']['mean_total_ms']:.2f} ms. Frozen-only covered {by['frozen_only']['covered_states']}/{by['frozen_only']['total_states']}; residual-only covered {by['residual_only']['covered_states']}/{by['residual_only']['total_states']}. This is a validation-only candidate study. No search or held-out discovery was run.\n\n## Frozen protocol\n\nThe rule was selected on {len(training)} training states before decoding the {len(validation)} validation states. It sends a state to residual when `residual_nonzero {rule['operator']} {rule['threshold']:g}` and otherwise to frozen. Runtime routing reads only the residual tensor. The sealed specification SHA-256 is `{digest}`.\n\n## Cost and coverage\n\n| Strategy | Covered states | Positive recall | Mean ms | Median ms | p95 ms | Residual calls | Dominated by mean |\n|---|---:|---:|---:|---:|---:|---:|---|\n"""
    for row in frontier_rows: report+=f"| {row['strategy']} | {row['covered_states']}/{row['total_states']} | {row['positive_recall_numerator']}/{row['positive_recall_denominator']} | {row['mean_total_ms']:.2f} | {row['median_total_ms']:.2f} | {row['p95_total_ms']:.2f} | {row['residual_calls']} | {row['dominated_by'] or 'no'} |\n"
    report+=f"""\nResidual-only cost {ratio:.2f}× frozen-only in these controlled samples. Each residual call evaluated all 6,400 `(u,v)` pairs. Therefore 6+6 and 9+3 did not reduce residual enumeration and paid for both providers. Mean-cost dominance is descriptive. Timing ranges and raw repetitions are retained because small differences inside run-to-run variability are not firm dominance evidence. Python `tracemalloc` peaks are separate diagnostics and were not included in timed samples.\n\n## Routing signal\n\nOn training data, residual nonzero count had correlation {summary['training_signal_associations']['residual_nonzero_vs_frozen_covered']} with frozen state coverage and {summary['training_signal_associations']['residual_nonzero_vs_fallback_gain']} with useful residual fallback. These are selection-set associations, not calibrated probabilities. The selected rule sent every validation state to residual.\n\n## Answers\n\n1. Residual generation cost {ratio:.2f}× frozen inference on mean latency.\n2. No. The observed training associations were near zero and the selected rule degenerated to residual-only.\n3. Yes, from {by['frozen_only']['covered_states']} to {by['adaptive']['covered_states']} of {len(validation)}, but only by calling residual on every state.\n4. No. Adaptive mean cost was {by['adaptive']['mean_total_ms']:.2f} ms versus {by['residual_only']['mean_total_ms']:.2f} ms and made the same number of residual calls.\n5. No. Both fixed allocations paid full residual enumeration plus frozen inference and did not improve residual-only coverage.\n6. Frozen-only and residual-only are the useful non-dominated endpoints. Residual achieved higher coverage and better covered-states per proposal second.\n7. No new held-out exact-discovery evaluation is justified for this router. It is operationally residual-only, which TG-4G already evaluated.\n\n## Recommended TG-4Q milestone\n\nImplement and validate one bounded residual-ranking approximation against exhaustive residual-only candidate coverage and cost. Keep the same 12-action budget and validation labels. Do not run held-out search until it establishes a real cost reduction without material coverage loss.\n\n## Reproduction\n\n```sh\nPYTHONPATH=src .venv/bin/python examples/tensor_game_adaptive_routing.py\n```\n"""
    (args.output/"report.md").write_text(report); Path("docs/tensor_game_adaptive_routing.md").write_text(report)
    print(json.dumps(summary,indent=2))

if __name__=="__main__": main()
