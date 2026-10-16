#!/usr/bin/env python3
"""Paired device/host timing and telemetry for Tensor Game GPU lowerings."""

from __future__ import annotations

import argparse, json, math, platform, random, statistics, subprocess, time
from pathlib import Path


def percentile(values, q):
    values = sorted(values); pos = (len(values) - 1) * q
    lo = math.floor(pos); hi = math.ceil(pos)
    return values[lo] if lo == hi else values[lo] + (values[hi] - values[lo]) * (pos - lo)


def summary(values):
    mean = statistics.mean(values)
    return {"median": statistics.median(values), "q1": percentile(values, .25), "q3": percentile(values, .75), "iqr": percentile(values, .75)-percentile(values, .25), "min": min(values), "max": max(values), "mean": mean, "stdev": statistics.stdev(values) if len(values)>1 else 0.0, "coefficient_of_variation": statistics.stdev(values)/mean if len(values)>1 and mean else 0.0}


def bootstrap_ci(values, seed, samples=10000):
    rng=random.Random(seed); medians=[]
    for _ in range(samples): medians.append(statistics.median(rng.choice(values) for _ in values))
    return [percentile(medians,.025), percentile(medians,.975)]


def telemetry():
    fields="timestamp,name,driver_version,temperature.gpu,clocks.current.graphics,clocks.current.memory,power.draw,power.limit,utilization.gpu,clocks_throttle_reasons.active"
    result=subprocess.run(["nvidia-smi",f"--query-gpu={fields}","--format=csv,noheader,nounits"],capture_output=True,text=True)
    return {"available":result.returncode==0,"fields":fields.split(","),"value":result.stdout.strip() or None,"error":result.stderr.strip() or None}


def main():
    p=argparse.ArgumentParser(); p.add_argument("--output",type=Path,required=True); p.add_argument("--sizes",type=int,nargs="+",default=(512,1024)); p.add_argument("--warmups",type=int,default=20); p.add_argument("--iterations",type=int,default=30); p.add_argument("--rounds",type=int,default=15); p.add_argument("--seed",type=int,default=20261016); a=p.parse_args()
    if a.output.exists(): p.error("--output must not already exist")
    import torch, triton
    from matmul_inspector.tensor_game_triton_lowering import reference_lowering
    from matmul_inspector.tensor_game_triton_runtime import execute_fused_preallocated
    if not torch.cuda.is_available(): p.error("CUDA is unavailable")
    torch.manual_seed(a.seed); torch.backends.cuda.matmul.allow_tf32=False
    raw=[]; rounds=[]
    for size in a.sizes:
        left=torch.randn((size,size),device="cuda"); right=torch.randn((size,size),device="cuda")
        lowerings={n:reference_lowering(n) for n in ("conventional","strassen")}
        allocated={n:(lambda l=l: l.execute(left,right,"ieee","fused_input","fused")) for n,l in lowerings.items()}
        buffers={n:([torch.empty((size//2,size//2),device="cuda") for _ in l.actions],torch.empty_like(left)) for n,l in lowerings.items()}
        preallocated={n:(lambda l=l,b=b: execute_fused_preallocated(l,left,right,*b)) for (n,l),b in zip(lowerings.items(),buffers.values())}
        implementations={"pytorch":lambda:left@right,**{f"{n}_allocated":f for n,f in allocated.items()},**{f"{n}_preallocated":f for n,f in preallocated.items()}}
        for f in implementations.values():
            for _ in range(a.warmups): f()
        torch.cuda.synchronize()
        direct=list(implementations); reversed_order=list(reversed(direct))
        for round_index in range(a.rounds):
            order=direct if round_index%2==0 else reversed_order
            before=telemetry()
            for _ in range(a.iterations):
                for name in order:
                    start=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
                    start.record(); implementations[name](); end.record(); end.synchronize()
                    raw.append({"size":size,"round":round_index,"order":"direct" if round_index%2==0 else "reversed","mode":"device","implementation":name,"elapsed_ms":start.elapsed_time(end)})
            for _ in range(a.iterations):
                for name in order:
                    torch.cuda.synchronize(); start_ns=time.perf_counter_ns(); implementations[name](); torch.cuda.synchronize(); end_ns=time.perf_counter_ns()
                    raw.append({"size":size,"round":round_index,"order":"direct" if round_index%2==0 else "reversed","mode":"host_synchronized","implementation":name,"elapsed_ms":(end_ns-start_ns)/1e6})
            after=telemetry()
            row={"size":size,"round":round_index,"order":"direct" if round_index%2==0 else "reversed","telemetry_before":before,"telemetry_after":after,"medians_ms":{}}
            for mode in ("device","host_synchronized"):
                row["medians_ms"][mode]={name:statistics.median(x["elapsed_ms"] for x in raw if x["size"]==size and x["round"]==round_index and x["mode"]==mode and x["implementation"]==name) for name in implementations}
            row["paired_conventional_over_strassen"]={mode:row["medians_ms"][mode]["conventional_allocated"]/row["medians_ms"][mode]["strassen_allocated"] for mode in ("device","host_synchronized")}
            rounds.append(row)
    summaries=[]
    for size in a.sizes:
      for mode in ("device","host_synchronized"):
       for name in ("pytorch","conventional_allocated","strassen_allocated","conventional_preallocated","strassen_preallocated"):
        vals=[r["medians_ms"][mode][name] for r in rounds if r["size"]==size]
        summaries.append({"size":size,"mode":mode,"implementation":name,**summary(vals)})
       ratios=[r["paired_conventional_over_strassen"][mode] for r in rounds if r["size"]==size]
       summaries.append({"size":size,"mode":mode,"implementation":"paired_conventional_over_strassen_allocated",**summary(ratios),"bootstrap_95_percent_ci_for_median":bootstrap_ci(ratios,a.seed+size+(0 if mode=="device" else 1))})
    payload={"schema":"tensor_game_timing_diagnostics.v1","configuration":vars(a)|{"output":str(a.output),"order":"direct on even rounds, reversed on odd rounds","device_window":"CUDA events around complete invocation; end event synchronized","host_window":"perf_counter_ns around invocation with synchronization immediately before and after","replicate_unit":"independent round median"},"environment":{"python":platform.python_version(),"torch":torch.__version__,"triton":triton.__version__,"cuda_runtime":torch.version.cuda,"device":torch.cuda.get_device_name(0),"capability":torch.cuda.get_device_capability(0)},"initial_telemetry":telemetry(),"rounds":rounds,"summaries":summaries,"raw_samples":raw}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(payload,indent=2)+"\n"); print(json.dumps(summaries,indent=2)); return 0


if __name__=="__main__": raise SystemExit(main())
