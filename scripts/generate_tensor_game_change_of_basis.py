#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, platform, subprocess
from dataclasses import asdict
from pathlib import Path
from matmul_inspector.tensor_game_candidate_dataset import write_dataset
from matmul_inspector.tensor_game_change_of_basis import ChangeOfBasisConfig, generate_change_of_basis_candidates, select_cost_representatives

def main():
    p=argparse.ArgumentParser(description="Generate exact bounded GL(2,Z)^3 Tensor Game variants")
    p.add_argument("--output",type=Path,required=True); p.add_argument("--entry-bound",type=int,default=1); p.add_argument("--combination-limit",type=int,default=512); p.add_argument("--mode",choices=("enumerate","sample"),default="sample"); p.add_argument("--seed",type=int,default=20261016); p.add_argument("--maximum-coefficient",type=int,default=8); p.add_argument("--maximum-total-support",type=int,default=96); p.add_argument("--maximum-pre-additions",type=int,default=64); p.add_argument("--maximum-post-additions",type=int,default=32); p.add_argument("--representatives",type=int,default=12)
    a=p.parse_args(); config=ChangeOfBasisConfig(a.entry_bound,a.combination_limit,a.mode,a.seed,a.maximum_coefficient,a.maximum_total_support,a.maximum_pre_additions,a.maximum_post_additions)
    records,summary=generate_change_of_basis_candidates(config); representatives=select_cost_representatives(records,a.representatives)
    manifest={"schema":"tensor_game_change_of_basis.v1","generator_configuration":asdict(config),"git_commit":subprocess.run(["git","rev-parse","HEAD"],capture_output=True,text=True).stdout.strip(),"environment":{"python":platform.python_version(),"platform":platform.platform()},"seed_families":["conventional","strassen"],"summary":summary,"representative_candidate_ids":[row["candidate_id"] for row in representatives],"limitations":["All variants retain known seed-family equivalence.","Canonical IDs do not recognize general change-of-basis equivalence.","Static costs are not performance measurements."]}
    write_dataset(a.output,records,manifest); (a.output/"representatives.json").write_text(json.dumps(representatives,sort_keys=True,separators=(",", ":"))+"\n",encoding="utf-8"); print(json.dumps(manifest,indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
