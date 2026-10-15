import json

from matmul_inspector.tensor_game_policy_regression import BASELINE, TRAINED, analyze_trials


def _trial(configuration, success, action, support):
    return {
        "configuration_id": configuration, "target_group_id": "target-0", "seed": 3,
        "trial_id": configuration, "status": "completed_success" if success else "bounded_search_failure",
        "termination_reason": "exact_reconstruction" if success else "move_limit_reached",
        "exact_reconstruction": success,
        "result": {"episode": {"initial_target_digest": "state", "max_moves": 1}, "decisions": [{
            "state_digest": "state", "next_state_digest": action, "selected_action_id": action,
            "selected_action": {"u": [1], "v": [1], "w": [1]},
            "sampled_actions": [{"action_id": item, "prior": 1.0, "visits": 1, "q_value": 0.0,
                                 "search_probability": 1.0, "action": {"u": [1], "v": [1], "w": [1]}}
                                for item in support],
        }]},
    }


def test_analyze_trials_identifies_unrecoverable_first_step_regression(tmp_path):
    trials = tmp_path / "trials"
    trials.mkdir()
    (trials / "baseline.json").write_text(json.dumps(_trial(BASELINE, True, "gold", ["gold", "b"])))
    (trials / "trained.json").write_text(json.dumps(_trial(TRAINED, False, "wrong", ["wrong", "t"])))
    result = analyze_trials(trials)
    assert result["success_transitions"]["regression"]["count"] == 1
    record = result["regression_records"][0]
    assert record["first_divergence"]["same_reference_state"]
    assert record["first_divergence"]["available_action_overlap"] == 0
    assert not record["first_divergence"]["gold_in_trained_support"]
    assert not record["recoverable_after_first_wrong_action"]


def test_analyze_trials_requires_complete_pairs(tmp_path):
    trials = tmp_path / "trials"
    trials.mkdir()
    (trials / "baseline.json").write_text(json.dumps(_trial(BASELINE, True, "gold", ["gold"])))
    try:
        analyze_trials(trials)
    except ValueError as error:
        assert "complete" in str(error)
    else:
        raise AssertionError("missing pair was accepted")
