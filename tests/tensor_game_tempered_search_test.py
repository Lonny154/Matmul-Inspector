import json
import os
from pathlib import Path
import random
import subprocess
import sys

import pytest

from matmul_inspector.tensor_game import (
    matrix_multiplication_tensor, standard_2x2_decomposition,
    strassen_2x2_decomposition, sum_actions,
)
from matmul_inspector.tensor_game_meet_in_middle import canonical_action_pool
from matmul_inspector.tensor_game_tempered_search import (
    TemperedSearchConfig, build_pair_repair_table, canonical_state,
    exact_two_term_repair, exchange_accept, parallel_tempering_search,
    residual_metrics, validate_pool_mode,
)


TARGET = matrix_multiplication_tensor(2, 2, 2)
STRASSEN = canonical_action_pool(strassen_2x2_decomposition())


def test_exact_and_perturbed_energy_components():
    exact = residual_metrics(TARGET, STRASSEN, "nnz_l1", 7)
    assert (exact.nonzero_entries, exact.l1_norm, exact.max_absolute_entry, exact.energy) == (0, 0, 0, 0)
    perturbed = residual_metrics(TARGET, STRASSEN[:-1], "nnz_l1", 7)
    assert perturbed.nonzero_entries == 4
    assert perturbed.l1_norm == 4
    assert perturbed.max_absolute_entry == 1
    assert perturbed.energy == 32
    assert residual_metrics(TARGET, STRASSEN[:-1], "l1", 999).energy == 4


def test_same_seed_has_same_bounded_trajectory_and_valid_local_states():
    pool = canonical_action_pool((*STRASSEN, *standard_2x2_decomposition()))
    config = TemperedSearchConfig(temperatures=(1.0, 2.0), rounds=5,
                                  local_steps_per_round=2, seed=91)
    first = parallel_tempering_search(TARGET, pool, config)
    second = parallel_tempering_search(TARGET, pool, config)
    for key in ("status", "best_state", "best_metrics", "counters", "history",
                "unique_canonical_states_visited"):
        assert first[key] == second[key]
    assert len(first["best_state"]) == 7
    assert len(set(first["best_state"])) == 7
    assert first["best_state"] == canonical_state(first["best_state"])


def test_exchange_decision_is_deterministic_and_temperatures_are_validated():
    assert exchange_accept(10, 5, 1.0, 2.0, random.Random(3)) == exchange_accept(
        10, 5, 1.0, 2.0, random.Random(3))
    with pytest.raises(ValueError):
        exchange_accept(1, 2, 0.0, 1.0, random.Random(1))
    with pytest.raises(ValueError):
        TemperedSearchConfig(temperatures=(1.0, -1.0)).validate(7)
    with pytest.raises(ValueError):
        TemperedSearchConfig(temperatures=(1.0, 2.0), replicas=3).validate(7)


def test_two_term_repair_recovers_and_independently_reconstructs_strassen():
    extras = canonical_action_pool(standard_2x2_decomposition())[:2]
    pool = canonical_action_pool((*STRASSEN, *extras))
    indices = {action: index for index, action in enumerate(pool)}
    state = tuple(indices[action] for action in STRASSEN[:5]) + tuple(indices[action] for action in extras)
    table = build_pair_repair_table(pool, allow_repeated_actions=False,
                                    maximum_pool_size=20, maximum_pair_records=200,
                                    maximum_seconds=2.0)
    result = exact_two_term_repair(TARGET, state, (5, 6), pool, table, False)
    assert result["status"] == "EXACT_VERIFIED"
    repaired = tuple(pool[index] for index in result["state"])
    assert sum_actions(repaired, (4, 4, 4)) == TARGET


def test_two_term_repair_reports_missing_complement():
    pool = canonical_action_pool((*STRASSEN[:5], *standard_2x2_decomposition()[:2]))
    table = build_pair_repair_table(pool, allow_repeated_actions=False,
                                    maximum_pool_size=20, maximum_pair_records=200,
                                    maximum_seconds=2.0)
    result = exact_two_term_repair(TARGET, tuple(range(7)), (5, 6), pool, table, False)
    assert result["status"] == "NO_COMPLEMENT"


def test_repair_limits_fail_bounded_before_pair_allocation():
    assert build_pair_repair_table(STRASSEN, allow_repeated_actions=False,
                                   maximum_pool_size=6, maximum_pair_records=100,
                                   maximum_seconds=1.0).status == "BUDGET_EXCEEDED_POOL_SIZE"
    assert build_pair_repair_table(STRASSEN, allow_repeated_actions=False,
                                   maximum_pool_size=7, maximum_pair_records=20,
                                   maximum_seconds=1.0).status == "BUDGET_EXCEEDED_PAIR_RECORDS"


def test_oracle_pool_is_rejected_in_discovery_mode_and_labeled_in_result():
    with pytest.raises(ValueError, match="not permitted"):
        validate_pool_mode("strassen_oracle", False)
    validate_pool_mode("strassen_oracle", True)
    result = parallel_tempering_search(TARGET, STRASSEN, TemperedSearchConfig(
        temperatures=(1.0,), rounds=1, local_steps_per_round=1, seed=4))
    assert result["status"] == "EXACT"
    assert result["exact_reconstruction"] is True


def test_oracle_assisted_artifacts_are_explicitly_labeled(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "artifacts"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src")
    subprocess.run([
        sys.executable, str(root / "scripts/run_tensor_game_tempered_search.py"),
        "--pool", "strassen_oracle", "--oracle-positive-control",
        "--temperatures", "1", "--replicas", "1", "--rounds", "1",
        "--local-steps", "1", "--output", str(output),
    ], check=True, cwd=root, env=environment, capture_output=True, text=True)
    summary = json.loads((output / "summary.json").read_text())
    provenance = json.loads((output / "provenance.json").read_text())
    candidate = json.loads((output / "best_candidate.json").read_text())
    assert summary["oracle_assisted"] is True
    assert "not a discovery" in summary["interpretation"]
    assert provenance["oracle_data_permitted"] is True
    assert provenance["pool_provenance"]["known_strassen_used_for_construction"] is True
    assert candidate["exact_reconstruction"] is True
