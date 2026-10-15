import json

from matmul_inspector.tensor_game_heldout_evaluation import aggregate, paired, verify
def test_integrity_verification_rejects_mismatch(tmp_path):
 s=tmp_path/"s";m=tmp_path/"m";s.write_text("{}");m.write_text(json.dumps({"targets":[]}))
 try:verify(s,m,"bad","bad")
 except ValueError as e:assert "hash mismatch" in str(e)
 else:raise AssertionError
def test_paired_comparison_uses_target_seed_keys():
 rows=[]
 for p,v in (("frozen_top12",False),("trained_top12",True),("alternating_hybrid12",False),("residual_norm_control",True)):
  rows.append({"target_id":"t","seed":3,"provider":p,"success":v,"constructed_depth":1})
 out=paired(rows);assert len(out)==4 and out[0]["transition"]=="improvement"


def test_aggregate_separates_trials_from_unique_targets():
 rows=[]
 for seed in (3,19,35):
  for provider in ("frozen_top12","trained_top12","alternating_hybrid12","residual_norm_control"):
   rows.append({"target_id":"target","constructed_depth":1,"provider":provider,"seed":seed,
    "success":provider=="residual_norm_control","simulations_completed":16,"expanded_nodes":1,
    "candidate_evaluations":12,"model_forward_passes":0,"search_wall_seconds":0.1,
    "proposal_generation_seconds":0.09,"failure_classification":"success" if provider=="residual_norm_control" else "initial_reference_action_missing"})
 recall=[{"provider":provider,"recall_top1":False,"recall_top12":False} for provider in ("frozen_top12","trained_top12","alternating_hybrid12","residual_norm_control")]
 metrics=aggregate(rows,recall)["by_depth"]["1"]
 assert metrics["frozen_top12"]["trials"]==3
 assert metrics["frozen_top12"]["unique_targets"]==1
 assert metrics["residual_norm_control"]["successes"]==3


def test_recorded_candidate_counts_respect_frozen_budget():
 rows=[{"provider":"frozen_top12","candidate_count":12},{"provider":"residual_norm_control","candidate_count":7}]
 assert all(row["candidate_count"] <= 12 for row in rows)
