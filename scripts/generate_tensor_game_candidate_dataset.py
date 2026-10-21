#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, platform, subprocess
from pathlib import Path
from matmul_inspector.tensor_game_candidate_dataset import GENERATOR_VERSION, TRANSFORMATIONS, generate_candidates, write_dataset

def main():
    p=argparse.ArgumentParser(); p.add_argument("--output",type=Path,required=True); p.add_argument("--seed",type=int,default=20261016); p.add_argument("--maximum-candidates",type=int,default=128); p.add_argument("--transformations",nargs="+",choices=TRANSFORMATIONS,default=TRANSFORMATIONS); p.add_argument("--input-solution-archive",type=Path); a=p.parse_args()
    if a.maximum_candidates<=0:p.error("--maximum-candidates must be positive")
    records,summary=generate_candidates(a.seed,a.maximum_candidates,a.transformations,a.input_solution_archive)
    commit=subprocess.run(["git","rev-parse","HEAD"],capture_output=True,text=True).stdout.strip() or None
    manifest={"schema":GENERATOR_VERSION,"generator_version":GENERATOR_VERSION,"configuration":{"seed":a.seed,"maximum_candidates":a.maximum_candidates,"transformations":list(a.transformations),"input_solution_archive":str(a.input_solution_archive) if a.input_solution_archive else None},"git_commit":commit,"environment":{"python":platform.python_version(),"platform":platform.platform()},"exact_target":"2x2 matrix multiplication tensor, shape 4x4x4","seed_inventory":[{"seed_id":"conventional","rank":8,"source":"built-in exact reference"},{"seed_id":"strassen","rank":7,"source":"built-in exact reference"}],"summary":summary,"limitations":["Only two independent built-in seed families are available.","Canonical classes prove only term-order and supported sign-scaling equivalence; index symmetries and general basis changes are not canonicalized.","Execution variants within one algebraic class may perform differently."]}
    write_dataset(a.output,records,manifest); print(json.dumps(manifest,indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
