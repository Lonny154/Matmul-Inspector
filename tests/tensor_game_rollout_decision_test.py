from collections import Counter
from matmul_inspector.tensor_game_rollout_decision import decide_release,generate_rollout_targets,upper_zero_event_bound
from matmul_inspector.tensor_game_search_calibration import replay_reference

def test_rollout_corpus_is_balanced_unique_deterministic_and_valid():
    a=generate_rollout_targets([],123,24);b=generate_rollout_targets([],123,24)
    assert a==b and len(a["targets"])==24
    assert Counter(x["workload"]["primary_category"] for x in a["targets"])==Counter({x:4 for x in a["category_minimums"]})
    assert set(x["construction_length"] for x in a["targets"])=={1,2,3,4}
    assert all(x["construction_length"]>=2 for x in a["targets"] if x["workload"]["primary_category"] in {"repeated_term","sign_equivalent"})
    assert len({x["target_digest"] for x in a["targets"]})==24
    assert all(replay_reference(x)["exact_success"] for x in a["targets"])

def test_release_decision_and_target_bound():
    assert upper_zero_event_bound(150)<.02
    criteria={"maximum_regression_upper_bound":.02,"minimum_mean_speedup":1.1,"minimum_paired_faster_fraction":.8,"minimum_genuine_machine_count":2}
    evidence={"exhaustive_only_targets":0,"correctness_failures":0,"bounded_timeouts":0,"exhaustive_timeouts":0,"compatibility_failures":0,"memory_stability_failures":0,"regression_upper_bound":.0198,"mean_speedup":1.2,"paired_faster_fraction":.9,"genuine_machine_count":2}
    assert decide_release(criteria,evidence)["decision"]=="GO"
    evidence["genuine_machine_count"]=1;assert decide_release(criteria,evidence)["decision"]=="CONDITIONAL"
    evidence["exhaustive_only_targets"]=1;assert decide_release(criteria,evidence)["decision"]=="NO-GO"
