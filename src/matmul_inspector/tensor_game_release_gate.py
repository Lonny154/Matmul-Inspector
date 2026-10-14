"""Final release-gate decision helpers for the Tensor Game provider rollout."""
from __future__ import annotations
from typing import Any

def decide_default_rollout(evidence:dict[str,Any])->dict[str,Any]:
    gates={
        "frozen_input_integrity":evidence["frozen_input_integrity"],
        "complete_cross_machine_trials":evidence["complete_cross_machine_trials"],
        "zero_cross_machine_discovery_mismatches":evidence["cross_machine_discovery_mismatches"]==0,
        "zero_missing_cross_machine_records":evidence["missing_cross_machine_records"]==0,
        "zero_exhaustive_only_targets":evidence["exhaustive_only_targets"]==0,
        "performance_gain_both_hosts":evidence["local_speedup"]>1 and evidence["aws_speedup"]>1,
        "memory_stability":evidence["memory_stability"],
        "candidate_correctness":evidence["candidate_correctness"],
        "python_api_compatibility":evidence["python_api_failures"]==0,
        "installed_cli_compatibility_both_hosts":evidence["local_cli_failures"]==0 and evidence["aws_cli_failures"]==0,
        "explicit_exhaustive_fallback":evidence["explicit_exhaustive_fallback"],
    }
    hard_failures=[name for name,passed in gates.items() if not passed]
    decision="GO" if not hard_failures else "CONDITIONAL"
    return {"decision":decision,"default_change_authorized":decision=="GO","default_changed":False,"gates":gates,"failed_gates":hard_failures}
