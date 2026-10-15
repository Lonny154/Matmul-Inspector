from collections import Counter

from matmul_inspector.tensor_game_default_readiness import decide_readiness,generate_readiness_targets
from matmul_inspector.tensor_game_search_calibration import replay_reference

def test_structural_corpus_is_deterministic_balanced_and_valid():
    first=generate_readiness_targets([],1234);second=generate_readiness_targets([],1234)
    assert first==second and len(first["targets"])==24
    assert Counter(x["construction_length"] for x in first["targets"])==Counter({1:6,2:6,3:6,4:6})
    counts=Counter(x["workload"]["primary_category"] for x in first["targets"])
    for category,minimum in first["category_minimums"].items():assert counts[category]>=minimum
    assert all(replay_reference(x)["exact_success"] for x in first["targets"])
    assert len({x["target_digest"] for x in first["targets"]})==24

def test_repeated_and_sign_equivalent_categories_are_real_constructions():
    manifest=generate_readiness_targets([],5678)
    repeated=[x for x in manifest["targets"] if x["workload"]["primary_category"]=="repeated_term"]
    signed=[x for x in manifest["targets"] if x["workload"]["primary_category"]=="sign_equivalent"]
    assert all(x["workload"]["repeated_reference_actions"]>0 for x in repeated)
    assert all(x["workload"]["sign_equivalent_reference"] for x in signed)

def test_readiness_gates_distinguish_go_conditional_and_no_go():
    criteria={"minimum_mean_speedup":1.05,"minimum_paired_faster_fraction":.75,"maximum_mean_python_peak_bytes":100000,"maximum_mean_python_retained_bytes":30000,"maximum_zero_regression_upper_bound":.125}
    evidence={"correctness_failures":0,"exhaustive_only_targets":0,"bounded_timeouts":0,"exhaustive_timeouts":0,"integration_failures":0,"mean_speedup":1.1,"paired_faster_fraction":.8,"bounded_mean_python_peak_bytes":90000,"bounded_mean_python_retained_bytes":20000,"regression_upper_bound":.12}
    assert decide_readiness(criteria,evidence)["decision"]=="GO"
    evidence["mean_speedup"]=1.0
    assert decide_readiness(criteria,evidence)["decision"]=="CONDITIONAL"
    evidence["exhaustive_only_targets"]=1
    assert decide_readiness(criteria,evidence)["decision"]=="NO-GO"
