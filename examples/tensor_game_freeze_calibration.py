#!/usr/bin/env python3
"""Freeze TG-4F evaluation settings and an unevaluated TG-4G target manifest."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from matmul_inspector.tensor_game_out_of_sample import canonical_json,generate_fresh_targets
def main():
 p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);p.add_argument("--known",type=Path,action="append",default=[]);a=p.parse_args()
 spec={"version":1,"status":"frozen_after_calibration_before_tg4g","target_generation":"filtered_tg4b2a_diverse_targets_v1 seed 20261015",
  "difficulty":"constructed depth upper bound, not proven minimum rank","primary_depth":1,"stress_depth":2,
  "candidate_budget":12,"simulation_budget_by_depth":{"1":16,"2":64},"node_limit":2000,"timeout_seconds":10.0,
  "c_puct":1.5,"value":"nonzero_heuristic","success":"production exact zero residual and exact reconstruction",
  "seeds":[3,19,35],"policies":["frozen_top12","trained_top12","alternating_hybrid12","residual_norm_control"],
  "required_metrics":["success","reference_action_recall","simulations_to_first_success","expanded_nodes","unique_states","candidate_evaluations","model_forward_passes","wall_seconds","termination_reason"],
  "hardware_assumption":"CPU-compatible; 10-second per-search hard cap","calibration_targets_are_not_generalization_evidence":True}
 fresh=generate_fresh_targets(known_paths=a.known,seed=20261015,per_depth=4)
 heldout={**{k:v for k,v in fresh.items() if k!="targets"},"reserved_for":"TG-4G; not evaluated during TG-4F",
          "targets":[x for x in fresh["targets"] if x["construction_length"] in (1,2)]}
 for name,value in (("evaluation_spec",spec),("heldout_targets",heldout)):
  path=a.output/f"{name}.json";path.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")
  (a.output/f"{name}.sha256").write_text(hashlib.sha256(path.read_bytes()).hexdigest()+f"  {name}.json\n")
if __name__=="__main__":main()
