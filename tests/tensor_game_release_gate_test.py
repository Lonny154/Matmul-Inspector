from matmul_inspector.tensor_game_release_gate import decide_default_rollout

def evidence():
    return {"frozen_input_integrity":True,"complete_cross_machine_trials":True,"cross_machine_discovery_mismatches":0,"missing_cross_machine_records":0,"exhaustive_only_targets":0,"local_speedup":2.1,"aws_speedup":2.0,"memory_stability":True,"candidate_correctness":True,"python_api_failures":0,"local_cli_failures":0,"aws_cli_failures":0,"explicit_exhaustive_fallback":True}

def test_all_release_gates_authorize_a_later_default_change():
    result=decide_default_rollout(evidence());assert result["decision"]=="GO" and result["default_change_authorized"]

def test_aws_installed_cli_failure_forces_conditional():
    values=evidence();values["aws_cli_failures"]=5;result=decide_default_rollout(values)
    assert result["decision"]=="CONDITIONAL" and not result["default_change_authorized"]
    assert result["failed_gates"]==["installed_cli_compatibility_both_hosts"]
