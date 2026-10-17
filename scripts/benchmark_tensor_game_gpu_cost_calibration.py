#!/usr/bin/env python3
"""Paired GPU calibration for exact dataset-backed Tensor Game lowerings."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import statistics
import subprocess
import time
from pathlib import Path


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] if lower == upper else ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def distribution(values):
    mean = statistics.mean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    return {"median_ms": statistics.median(values), "iqr_ms": percentile(values, .75) - percentile(values, .25), "mean_ms": mean, "stdev_ms": stdev, "cv": stdev / mean if mean else 0.0, "min_ms": min(values), "max_ms": max(values)}


def bootstrap_median_ci(values, seed, replicates=5000):
    rng = random.Random(seed)
    medians = sorted(statistics.median(rng.choices(values, k=len(values))) for _ in range(replicates))
    return [percentile(medians, .025), percentile(medians, .975)]


def telemetry():
    fields = "timestamp,temperature.gpu,power.draw,power.limit,clocks.current.graphics,clocks.current.memory,utilization.gpu,memory.used"
    result = subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"], capture_output=True, text=True)
    return {"available": result.returncode == 0, "fields": fields.split(","), "values": result.stdout.strip().split(", ") if result.returncode == 0 else None, "error": result.stderr.strip() or None}


def actions_from_record(record, RankOneAction):
    return tuple(RankOneAction(row["u"], row["v"], row["w"]) for row in record["actions"])


def load_candidates(dataset, RankOneAction, create_lowering, validate_decomposition, standard, strassen):
    selected = json.loads((dataset / "representatives.json").read_text())
    rows = []
    for record in selected:
        actions = actions_from_record(record, RankOneAction)
        validate_decomposition(actions)
        lowering = create_lowering(record["candidate_id"], actions)
        rows.append((record["candidate_id"], record, lowering))
    for name, actions in (("original_conventional", standard()), ("original_strassen", strassen())):
        validate_decomposition(actions)
        payload = [{"u": list(a.u), "v": list(a.v), "w": list(a.w)} for a in actions]
        digest = hashlib.sha256((json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()).hexdigest()
        if digest not in {identifier for identifier, _, _ in rows}:
            lowering = create_lowering(name, actions)
            cost = lowering.metadata
            support = sum(value != 0 for action in actions for value in action.u + action.v + action.w)
            record = {"candidate_id": name, "coefficient_hash": digest, "family_id": name.removeprefix("original_"), "rank": cost.rank, "actions": payload, "exact_validation": True, "static_cost": {"pre_additions": cost.preparation_additions, "post_additions": cost.reconstruction_additions, "coefficient_scalings": cost.coefficient_scalings}, "total_coefficient_support": support, "structural_signature": {"max_abs_coefficient": 1}}
            rows.append((name, record, lowering))
    return rows


def error_metrics(torch, actual, expected):
    difference = (actual - expected).abs()
    meaningful = expected.abs() > 1e-6
    return {"max_absolute_error": difference.max().item(), "mean_absolute_error": difference.mean().item(), "max_relative_error_nonzero_reference": (difference[meaningful] / expected[meaningful].abs()).max().item() if meaningful.any() else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--correctness-sizes", type=int, nargs="+", default=(128,256,512,1024)); parser.add_argument("--benchmark-sizes", type=int, nargs="+", default=(256,512,1024))
    parser.add_argument("--warmups", type=int, default=20); parser.add_argument("--iterations", type=int, default=30); parser.add_argument("--host-iterations", type=int, default=5); parser.add_argument("--rounds", type=int, default=15); parser.add_argument("--seed", type=int, default=20261016)
    args = parser.parse_args()
    if args.output.exists(): parser.error("--output must not already exist")
    if min(args.warmups,args.iterations,args.host_iterations,args.rounds)<=0: parser.error("timing counts must be positive")
    import torch, triton
    if not torch.cuda.is_available(): parser.error("CUDA is unavailable")
    from matmul_inspector.tensor_game import RankOneAction, standard_2x2_decomposition, strassen_2x2_decomposition
    from matmul_inspector.tensor_game_candidate_dataset import validate_decomposition
    from matmul_inspector.tensor_game_triton_lowering import create_lowering, lowering_manifest
    candidates = load_candidates(args.dataset,RankOneAction,create_lowering,validate_decomposition,standard_2x2_decomposition,strassen_2x2_decomposition)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed); torch.backends.cuda.matmul.allow_tf32=False
    correctness=[]; passing={identifier for identifier,_,_ in candidates}
    for size in args.correctness_sizes:
        generator=torch.Generator(device="cuda"); generator.manual_seed(args.seed+size)
        cases={"zero":(torch.zeros((size,size),device="cuda"),torch.zeros((size,size),device="cuda")),"identity":(torch.eye(size,device="cuda"),torch.eye(size,device="cuda")),"negative":(-torch.rand((size,size),generator=generator,device="cuda"),torch.rand((size,size),generator=generator,device="cuda")-.5),"seeded_random":(torch.randn((size,size),generator=generator,device="cuda"),torch.randn((size,size),generator=generator,device="cuda"))}
        for case,(left,right) in cases.items():
            expected=left@right
            for identifier,record,lowering in candidates:
                actual=lowering.execute(left,right,"ieee","fused_input","fused")
                metrics=error_metrics(torch,actual,expected); passed=bool(torch.allclose(actual,expected,rtol=5e-3,atol=5e-3))
                correctness.append({"size":size,"case":case,"candidate_id":identifier,**metrics,"rtol":5e-3,"atol":5e-3,"passed":passed})
                if not passed: passing.discard(identifier)
    torch.cuda.synchronize()
    if not passing: raise RuntimeError("no candidate passed correctness")
    raw=[]; rounds=[]; randomizer=random.Random(args.seed)
    by_id={identifier:(record,lowering) for identifier,record,lowering in candidates if identifier in passing}
    for size in args.benchmark_sizes:
        generator=torch.Generator(device="cuda"); generator.manual_seed(args.seed+size+10000)
        left=torch.randn((size,size),generator=generator,device="cuda"); right=torch.randn((size,size),generator=generator,device="cuda")
        implementations={identifier:(lambda lowering=lowering:lowering.execute(left,right,"ieee","fused_input","fused")) for identifier,(record,lowering) in by_id.items()}
        implementations["pytorch_cublas"]=lambda:left@right
        for implementation in implementations.values():
            for _ in range(args.warmups): implementation()
        torch.cuda.synchronize()
        direct=list(implementations); randomizer.shuffle(direct)
        for round_index in range(args.rounds):
            if round_index%2==0:
                if round_index: randomizer.shuffle(direct)
                order=list(direct); order_label="randomized_direct"
            else:
                order=list(reversed(direct)); order_label="reversed_prior"
            before=telemetry()
            for _ in range(args.iterations):
                for identifier in order:
                    start=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
                    start.record(); implementations[identifier](); end.record(); end.synchronize()
                    raw.append({"size":size,"round":round_index,"order":order_label,"mode":"device","candidate_id":identifier,"elapsed_ms":start.elapsed_time(end)})
            for identifier in order:
                for _ in range(args.host_iterations):
                    torch.cuda.synchronize(); started=time.perf_counter_ns(); implementations[identifier](); torch.cuda.synchronize()
                    raw.append({"size":size,"round":round_index,"order":order_label,"mode":"host_synchronized","candidate_id":identifier,"elapsed_ms":(time.perf_counter_ns()-started)/1e6})
            after=telemetry(); medians={}
            for mode in ("device","host_synchronized"):
                medians[mode]={identifier:statistics.median(row["elapsed_ms"] for row in raw if row["size"]==size and row["round"]==round_index and row["mode"]==mode and row["candidate_id"]==identifier) for identifier in implementations}
            rounds.append({"size":size,"round":round_index,"order":order_label,"execution_order":order,"telemetry_before":before,"telemetry_after":after,"medians_ms":medians})
    summaries=[]; ratios=[]
    for size in args.benchmark_sizes:
        for mode in ("device","host_synchronized"):
            for identifier in (*by_id,"pytorch_cublas"):
                values=[row["medians_ms"][mode][identifier] for row in rounds if row["size"]==size]
                item={"size":size,"mode":mode,"candidate_id":identifier,"independent_rounds":len(values),"round_medians_ms":values,**distribution(values)}
                if identifier in by_id:
                    family=by_id[identifier][0]["family_id"]; reference="original_strassen" if family=="strassen" else "original_conventional"
                    paired=[row["medians_ms"][mode][identifier]/row["medians_ms"][mode][reference] for row in rounds if row["size"]==size]
                    item.update({"same_rank_reference":reference,"paired_candidate_over_reference_median":statistics.median(paired),"paired_ratio_bootstrap_95_ci":bootstrap_median_ci(paired,args.seed+size+len(summaries))})
                    ratios.extend({"size":size,"mode":mode,"candidate_id":identifier,"reference":reference,"round":index,"ratio":value} for index,value in enumerate(paired))
                summaries.append(item)
    manifest=[]
    for identifier,record,lowering in candidates:
        manifest.append({"candidate_id":identifier,"coefficient_hash":record.get("coefficient_hash",record["candidate_id"]),"family_id":record["family_id"],"rank":record["rank"],"exact_validation":record["exact_validation"],"correctness_passed":identifier in passing,"static_cost":record["static_cost"],"total_coefficient_support":record["total_coefficient_support"],"maximum_coefficient_magnitude":record["structural_signature"]["max_abs_coefficient"],"structural_signature":record.get("structural_signature"),"lowering":lowering_manifest(lowering,"ieee","fused_input","fused"),"static_execution":{"gemm_launches":record["rank"],"reconstruction_launches":1,"total_device_kernel_launches":record["rank"]+1,"rank_product_buffers":record["rank"],"full_size_input_combination_buffers":0,"intermediate_output_block_buffers":0}})
    payload={"schema":"tensor_game_gpu_cost_calibration.v1","configuration":vars(args)|{"dataset":str(args.dataset),"output":str(args.output),"precision":"ieee","strategy":"fused_input","reconstruction":"fused","tile_geometry":"32x32x16, 4 warps, 3 stages","device_timing":"CUDA events around complete operation; compilation excluded","host_timing":"perf_counter_ns with synchronization before and after","replicate_unit":"independent round median","order":"randomized direct order on even rounds; exact reversal on following odd round"},"environment":{"python":platform.python_version(),"platform":platform.platform(),"torch":torch.__version__,"triton":triton.__version__,"cuda_runtime":torch.version.cuda,"device":torch.cuda.get_device_name(0),"capability":torch.cuda.get_device_capability(0)},"initial_telemetry":telemetry(),"candidate_manifest":manifest,"correctness":correctness,"passing_candidate_ids":sorted(passing),"rounds":rounds,"summaries":summaries,"paired_ratios":ratios,"raw_timings":raw}
    args.output.mkdir(parents=True,exist_ok=False); (args.output/"results.json").write_text(json.dumps(payload,sort_keys=True,separators=(",",":"),default=str)+"\n")
    print(json.dumps({"output":str(args.output),"candidates":len(candidates),"passing":len(passing),"correctness_rows":len(correctness),"timing_rows":len(raw)},indent=2)); return 0

if __name__=="__main__": raise SystemExit(main())
