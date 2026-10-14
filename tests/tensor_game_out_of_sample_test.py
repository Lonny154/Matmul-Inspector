import json
from matmul_inspector.tensor_game import RankOneAction
from matmul_inspector.tensor_game_out_of_sample import (
    aggregate, alternating_merge, axis_orbit_digests, generate_fresh_targets, preregistration,
)

def test_axis_equivalence_is_deterministic():
    target=(((1,0),(0,0)),((0,0),(0,0)))
    assert axis_orbit_digests(target)==axis_orbit_digests(target)
    assert 1 <= len(axis_orbit_digests(target)) <= 6

def test_preregistration_is_reproducible():
    targets={"targets":[1,2]}
    assert preregistration(targets)==preregistration(targets)

def test_fresh_target_generation_is_deterministic_and_excludes_known(tmp_path):
    first=generate_fresh_targets(known_paths=[],per_depth=1)
    known=tmp_path/"known.json"
    known.write_text(json.dumps({"targets":[first["targets"][0]]}))
    repeated=generate_fresh_targets(known_paths=[known],per_depth=1)
    assert repeated==generate_fresh_targets(known_paths=[known],per_depth=1)
    assert first["targets"][0]["target_digest"] not in {row["target_digest"] for row in repeated["targets"]}

def test_hybrid_merge_enforces_budget_deduplicates_and_preserves_attribution():
    shared=RankOneAction((1,0),(1,0),(1,0))
    frozen_only=RankOneAction((1,0),(0,1),(1,0))
    trained_only=RankOneAction((0,1),(1,0),(1,0))
    actions, attribution=alternating_merge([shared,frozen_only],[shared,trained_only],budget=2)
    assert len(actions)==2 and len(set(actions))==2
    assert attribution[next(key for key,value in attribution.items() if value==("frozen","trained"))]==("frozen","trained")

def test_aggregation_reports_paired_transitions():
    rows=[]
    for mode,success in (("frozen_top12",True),("trained_top12",False),("alternating_hybrid12",True)):
        rows.append({"target_id":"t","seed":3,"constructed_depth":1,"configuration":mode,"success":success,
                     "simulations_completed":10,"simulations_to_first_success":5 if success else None,"expanded_nodes":2,
                     "model_forward_passes":2,"search_wall_seconds":0.1})
    recall=[{"configuration":m,"recall_at_1":False,"recall_at_12":False} for m in ("frozen_top12","trained_top12","alternating_hybrid12")]
    result=aggregate(rows,recall)
    assert result["paired_vs_frozen"]["trained_top12"]["regression"]==1
    assert result["paired_vs_frozen"]["alternating_hybrid12"]["shared_success"]==1
