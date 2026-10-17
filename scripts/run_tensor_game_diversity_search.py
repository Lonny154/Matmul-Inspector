#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, platform, subprocess
from pathlib import Path
from matmul_inspector.tensor_game_diversity_search import DiscoveryConfig, DiversityArchive, discover

def dump(value): return (json.dumps(value, sort_keys=True, separators=(",", ":"))+"\n").encode()

def main():
    p=argparse.ArgumentParser(description="Run bounded CPU-only Tensor Game decomposition discovery")
    p.add_argument("--output",type=Path,required=True); p.add_argument("--seeds",type=int,nargs="+",default=[3,11,29]); p.add_argument("--ranks",type=int,nargs="+",default=[7,8]); p.add_argument("--node-budget",type=int,default=256); p.add_argument("--beam-width",type=int,default=128); p.add_argument("--max-actions",type=int,default=12); p.add_argument("--archive-size",type=int,default=64)
    a=p.parse_args()
    if a.output.exists(): p.error(f"output already exists: {a.output}")
    a.output.mkdir(parents=True); archive=DiversityArchive(a.archive_size); trials=[]
    for mode in ("baseline","diversity"):
        for rank in a.ranks:
            for seed in a.seeds:
                config=DiscoveryConfig(rank,seed,a.node_budget,a.beam_width,a.max_actions,mode,4 if mode=="diversity" else 1)
                result=discover(config); trials.append(result)
                for row in result["solutions"]: archive.add(row)
                print(json.dumps({k:result[k] for k in ("config","success","nodes_expanded","wall_seconds")}),flush=True)
    candidates=b"".join(dump(row) for row in archive.records); (a.output/"candidates.jsonl").write_bytes(candidates)
    raw={"schema":"tensor_game_diversity_search.v1","git_commit":subprocess.run(["git","rev-parse","HEAD"],capture_output=True,text=True).stdout.strip(),"environment":{"python":platform.python_version(),"platform":platform.platform()},"provider":"bounded_128 (frozen pair budget 128)","trials":trials}
    raw_bytes=dump(raw); (a.output/"trials.json").write_bytes(raw_bytes)
    signatures={r["structural_signature_id"] for r in archive.records}; costs={(r["static_cost"]["pre_additions"],r["static_cost"]["post_additions"],r["static_cost"]["coefficient_scalings"],r["rank"]) for r in archive.records}
    summary={"attempts":len(trials),"successful_attempts":sum(x["success"] for x in trials),"unique_exact_representations":len(archive.records),"limited_canonical_classes":len({r["canonical_id"] for r in archive.records}),"structural_signatures":len(signatures),"arithmetic_cost_points":len(costs),"nodes_expanded":sum(x["nodes_expanded"] for x in trials),"wall_seconds":sum(x["wall_seconds"] for x in trials),"files":{"trials.json":{"sha256":hashlib.sha256(raw_bytes).hexdigest()},"candidates.jsonl":{"sha256":hashlib.sha256(candidates).hexdigest()}}}
    (a.output/"summary.json").write_bytes(dump(summary)); print(json.dumps(summary,indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
