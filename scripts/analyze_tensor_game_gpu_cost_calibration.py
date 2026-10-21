#!/usr/bin/env python3
"""Build interpretable within-rank analysis from GPU calibration evidence."""
from __future__ import annotations
import argparse,hashlib,json,math,statistics
from pathlib import Path

def ranks(values):
    order=sorted(range(len(values)),key=lambda i:values[i]); result=[0.0]*len(values); start=0
    while start<len(order):
        end=start+1
        while end<len(order) and values[order[end]]==values[order[start]]: end+=1
        rank=(start+end-1)/2+1
        for index in order[start:end]: result[index]=rank
        start=end
    return result

def pearson(left,right):
    if len(left)<2:return None
    lm,rm=statistics.mean(left),statistics.mean(right); numerator=sum((x-lm)*(y-rm) for x,y in zip(left,right)); denominator=math.sqrt(sum((x-lm)**2 for x in left)*sum((y-rm)**2 for y in right))
    return numerator/denominator if denominator else None

def spearman(left,right):return pearson(ranks(left),ranks(right))

def main():
    p=argparse.ArgumentParser();p.add_argument("--experiment",type=Path,required=True);a=p.parse_args(); result_path=a.experiment/"results.json";data=json.loads(result_path.read_text()); manifests={x["candidate_id"]:x for x in data["candidate_manifest"]}; candidates=[x for x in manifests if not x.startswith("original_")]
    correlations=[];order_stability=[]
    descriptors={"pre_additions":lambda x:x["static_cost"]["pre_additions"],"post_additions":lambda x:x["static_cost"]["post_additions"],"coefficient_scalings":lambda x:x["static_cost"]["coefficient_scalings"],"total_support":lambda x:x["total_coefficient_support"],"maximum_coefficient":lambda x:x["maximum_coefficient_magnitude"]}
    for size in data["configuration"]["benchmark_sizes"]:
        for rank in (7,8):
            group=[x for x in candidates if manifests[x]["rank"]==rank]; latency=[next(s["median_ms"] for s in data["summaries"] if s["size"]==size and s["mode"]=="device" and s["candidate_id"]==x) for x in group]
            correlations.append({"size":size,"rank":rank,"candidate_count":len(group),"spearman":{name:spearman([function(manifests[x]) for x in group],latency) for name,function in descriptors.items()}})
            direct=[];reverse=[]
            for identifier in group:
                direct.append(statistics.median(r["medians_ms"]["device"][identifier] for r in data["rounds"] if r["size"]==size and r["order"]=="randomized_direct"));reverse.append(statistics.median(r["medians_ms"]["device"][identifier] for r in data["rounds"] if r["size"]==size and r["order"]=="reversed_prior"))
            order_stability.append({"size":size,"rank":rank,"direct_vs_reversed_spearman":spearman(direct,reverse),"direct_medians_ms":dict(zip(group,direct)),"reversed_medians_ms":dict(zip(group,reverse))})
    compiler=[]
    compiler_root=a.experiment/"compiler"
    if compiler_root.exists():
        for path in sorted(compiler_root.glob("*/summary.json")):
            value=json.loads(path.read_text()); kernels=value["kernels"]
            compiler.append({"candidate_id":value["candidate_id"],"rank":value["rank"],"kernel_specializations":len(kernels),"maximum_registers_per_thread":max(x["registers_per_thread"] for x in kernels),"maximum_dynamic_shared_bytes":max(x["dynamic_shared_bytes"] or 0 for x in kernels),"maximum_local_bytes":max(x["local_bytes"] or 0 for x in kernels),"maximum_stack_bytes":max(x["stack_bytes"] or 0 for x in kernels),"summed_ptx_static":{key:sum(x["ptx_static"][key] for x in kernels) for key in ("fma_f32","global_loads","global_stores")},"summed_sass_static":{key:sum(x["sass_static"][key] for x in kernels) for key in ("ffma","global_loads","global_stores")}})
    correctness=data["correctness"]
    payload={"schema":"tensor_game_gpu_cost_analysis.v1","source":{"results_sha256":hashlib.sha256(result_path.read_bytes()).hexdigest()},"candidate_count":len(manifests),"passing_candidate_count":sum(x["correctness_passed"] for x in manifests.values()),"correctness_error_range":{"maximum_absolute_error":[min(x["max_absolute_error"] for x in correctness),max(x["max_absolute_error"] for x in correctness)],"maximum_mean_absolute_error":max(x["mean_absolute_error"] for x in correctness),"maximum_relative_error_nonzero_reference":max((x["max_relative_error_nonzero_reference"] or 0) for x in correctness),"relative_error_note":"maximum elementwise relative error is unstable near small nonzero reference values"},"within_rank_correlations":correlations,"execution_order_stability":order_stability,"compiler_summary":compiler,"notes":["Spearman correlations use candidate-level medians within rank; transformed variants are not independent algorithm families.","Static compiler counts are summed text occurrences across specialized kernels, not dynamic instructions."]}
    output=a.experiment/"analysis.json"
    if output.exists():p.error("analysis.json already exists")
    output.write_text(json.dumps(payload,indent=2)+"\n")
    files=[]
    for path in sorted(x for x in a.experiment.rglob("*") if x.is_file() and x.name!="artifact_hashes.json"):
        files.append({"path":str(path.relative_to(a.experiment)),"bytes":path.stat().st_size,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
    (a.experiment/"artifact_hashes.json").write_text(json.dumps({"schema":"tensor_game_gpu_cost_artifact_hashes.v1","files":files},indent=2)+"\n")
    print(json.dumps(payload,indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
