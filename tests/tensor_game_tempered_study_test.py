import os
from pathlib import Path
import subprocess
import sys

import pytest

from matmul_inspector.tensor_game import matrix_multiplication_tensor, standard_2x2_decomposition, strassen_2x2_decomposition, sum_actions
from matmul_inspector.tensor_game_meet_in_middle import canonical_action_pool
from matmul_inspector.tensor_game_tempered_search import (
    PairRepairTable, TemperedSearchConfig, audit_two_term_repairability,
    build_pair_repair_table, canonical_state, initial_states_for_seed,
    parallel_tempering_search, state_identity,
)
from matmul_inspector.tensor_game_tempered_study import aggregate_matched_runs, run_matched_study

TARGET = matrix_multiplication_tensor(2, 2, 2)
STRASSEN = canonical_action_pool(strassen_2x2_decomposition())
POOL = canonical_action_pool((*STRASSEN, *standard_2x2_decomposition()))


def small_config():
    return TemperedSearchConfig(temperatures=(1.0, 2.0), replicas=2, rounds=3,
                                local_steps_per_round=2, repair_interval=1, seed=41)


def test_four_variants_share_initial_states_and_local_proposal_budget():
    initial, runs, audits = run_matched_study(TARGET, POOL, small_config(), (41, 42),
                                               audit_enabled=True, oracle_assisted=True)
    assert len(initial) == 2 and len(runs) == 8 and len(audits) == 8
    for seed in (41, 42):
        paired = [row for row in runs if row["seed"] == seed]
        assert len({tuple(row["initial_state_hashes"]) for row in paired}) == 1
        assert {row["counters"]["local_proposals"] for row in paired} == {12}
        assert all(row["counters"]["exchange_attempts"] == 0 for row in paired
                   if row["variant"].startswith("independent"))
    summary = aggregate_matched_runs(runs)
    assert len(summary["paired_comparisons"]) == 4
    assert all(len(item["per_seed"]) == 2 for item in summary["paired_comparisons"].values())


def test_same_seed_reproduces_trajectory_and_summary_without_wall_times():
    config = small_config()
    states = initial_states_for_seed(len(POOL), config)
    first = parallel_tempering_search(TARGET, POOL, config, states)
    second = parallel_tempering_search(TARGET, POOL, config, states)
    for key in ("best_state", "best_metrics", "best_candidate_hash", "initial_state_hashes",
                "counters", "history", "unique_canonical_states_visited"):
        assert first[key] == second[key]


def test_audit_checks_all_21_pairs_and_recovers_five_term_strassen_core():
    extras = canonical_action_pool(standard_2x2_decomposition())[:2]
    pool = canonical_action_pool((*STRASSEN, *extras))
    indices = {action: index for index, action in enumerate(pool)}
    state = tuple(indices[action] for action in STRASSEN[:5]) + tuple(indices[action] for action in extras)
    table = build_pair_repair_table(pool, allow_repeated_actions=False,
                                    maximum_pool_size=20, maximum_pair_records=100,
                                    maximum_seconds=2)
    audit = audit_two_term_repairability(TARGET, state, pool, table, False, oracle_assisted=True)
    assert audit["term_pairs_audited"] == audit["total_term_pairs"] == 21
    assert audit["valid_completion_cores"] >= 1
    assert audit["independently_verified"] >= 1
    assert audit["oracle_assisted"] is True
    assert all(row["exact_reconstruction"] for row in audit["details"])


def test_missing_action_and_duplicate_rule_prevent_false_repairability():
    pool = canonical_action_pool((*STRASSEN[:5], *standard_2x2_decomposition()[:2]))
    table = build_pair_repair_table(pool, allow_repeated_actions=False,
                                    maximum_pool_size=20, maximum_pair_records=100,
                                    maximum_seconds=2)
    audit = audit_two_term_repairability(TARGET, tuple(range(7)), pool, table, False)
    assert audit["term_pairs_audited"] == 21
    assert audit["repairable"] is False
    # An exact raw complement using an already frozen action must be rejected.
    repeated_target = sum_actions(tuple(pool[i] for i in (0, 1, 2, 3, 4, 0, 1)), (4, 4, 4))
    repeated_audit = audit_two_term_repairability(repeated_target, tuple(range(7)), pool, table, False)
    assert repeated_audit["raw_pair_complements"] > 0
    assert repeated_audit["domain_rejections"] > 0
    assert repeated_audit["independently_verified"] == 0


def test_pair_table_budget_is_bounded_in_audit():
    table = build_pair_repair_table(STRASSEN, allow_repeated_actions=False,
                                    maximum_pool_size=6, maximum_pair_records=100,
                                    maximum_seconds=1)
    audit = audit_two_term_repairability(TARGET, tuple(range(7)), STRASSEN, table, False)
    assert audit["lookup_status"] == "BUDGET_EXCEEDED_POOL_SIZE"
    assert audit["term_pairs_audited"] == 0
    assert audit["repairable"] is False


def test_aggregation_rejects_unpaired_hashes():
    _, rows, _ = run_matched_study(TARGET, POOL, small_config(), (41,),
                                    audit_enabled=False, oracle_assisted=False)
    rows[0]["initial_state_hashes"] = ["mismatch"]
    with pytest.raises(ValueError, match="different initial states"):
        aggregate_matched_runs(rows)


def test_cli_discovery_rejects_oracle_pool(tmp_path):
    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src")
    command = [sys.executable, str(root / "scripts/run_tensor_game_tempered_study.py"),
               "--pool", "strassen_oracle", "--seed-count", "1", "--output", str(tmp_path / "run")]
    result = subprocess.run(command, cwd=root, env=environment, capture_output=True, text=True)
    assert result.returncode != 0
    assert "requires --oracle-positive-control" in result.stderr
    assert not (tmp_path / "run").exists()
