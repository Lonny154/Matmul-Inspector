from dataclasses import replace
from pathlib import Path
import sys

import pytest

from matmul_inspector.tensor_game import matrix_multiplication_tensor, standard_2x2_decomposition, strassen_2x2_decomposition, sum_actions
from matmul_inspector.tensor_game_meet_in_middle import canonical_action_pool, partial_sum_identity
from matmul_inspector.tensor_game_tempered_search import (
    PairRepairTable, TemperedSearchConfig, parallel_tempering_search,
    residual_metrics, state_identity,
)
from matmul_inspector.tensor_game_completion_depth import (
    CompletionConfig, CompletionLookup, audit_completion_depth, build_completion_lookup,
    complete_core, recombine_parents,
)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_tensor_game_completion_depth_study import aggregate, load_candidates

TARGET = matrix_multiplication_tensor(2, 2, 2)
STRASSEN = canonical_action_pool(strassen_2x2_decomposition())


def config(**changes):
    return replace(CompletionConfig(maximum_pool_size=20, maximum_pair_records=200,
                                    maximum_unique_pair_sums=200,
                                    maximum_complement_probes=2000,
                                    maximum_seconds=2), **changes)


def test_strassen_has_all_core_counts_and_all_completion_depths():
    lookup = build_completion_lookup(STRASSEN, config(), oracle_assisted=True)
    audit = audit_completion_depth(TARGET, tuple(range(7)), STRASSEN, lookup, config(), oracle_assisted=True)
    assert [audit["depths"][str(depth)]["frozen_cores_tested"] for depth in (2, 3, 4)] == [21, 35, 35]
    assert [audit["depths"][str(depth)]["cores_with_verified_matches"] for depth in (2, 3, 4)] == [21, 35, 35]
    assert [audit["depths"][str(depth)]["unique_exact_completions"] for depth in (2, 3, 4)] == [1, 1, 1]
    assert audit["unique_exact_completions"] == 1
    assert audit["exact_candidates"][0]["exact_reconstruction"] is True


@pytest.mark.parametrize("missing", (3, 4))
def test_mitm_finds_known_complement_and_rejects_missing_actions(missing):
    frozen = tuple(range(7 - missing))
    lookup = build_completion_lookup(STRASSEN, config(), oracle_assisted=True)
    hit = complete_core(TARGET, frozen, missing, STRASSEN, lookup, config(), oracle_assisted=True)
    assert hit["status"] == "COMPLETE" and hit["verified_matches"] >= 1
    limited_pool = STRASSEN[:-1]
    limited_lookup = build_completion_lookup(limited_pool, config(), oracle_assisted=True)
    miss = complete_core(TARGET, frozen, missing, limited_pool, limited_lookup, config(), oracle_assisted=True)
    assert miss["status"] == "NO_COMPLETION"
    assert miss["verified_matches"] == 0


def test_domain_rules_and_hash_equality_are_enforced():
    pool = STRASSEN
    lookup = build_completion_lookup(pool, config(), oracle_assisted=True)
    repeated_target = sum_actions(tuple(pool[i] for i in (0, 1, 2, 3, 4, 0, 1)), (4, 4, 4))
    result = complete_core(repeated_target, (0, 1, 2, 3, 4), 2, pool, lookup, config(), oracle_assisted=True)
    assert result["raw_matches"] > 0 and result["domain_valid_matches"] == 0
    wrong = sum_actions((pool[0], pool[0]), (4, 4, 4))
    required = sum_actions((pool[5], pool[6]), (4, 4, 4))
    digest = partial_sum_identity(required)
    forged = replace(lookup, pair_table=PairRepairTable("READY", 1, {digest: ((wrong, 5, 6),)}))
    collision = complete_core(TARGET, tuple(range(5)), 2, pool, forged, config(), oracle_assisted=True)
    assert collision["raw_matches"] == collision["verified_matches"] == 0
    assert collision["work"]["exact_tensor_equalities"] == 1


def test_pair_limits_and_lookup_limits_are_explicit():
    limited = config(maximum_pair_records=20)
    lookup = build_completion_lookup(STRASSEN, limited, oracle_assisted=True)
    assert lookup.status == "PAIR_TABLE_LIMIT"
    result = complete_core(TARGET, tuple(range(3)), 4, STRASSEN, lookup, limited, oracle_assisted=True)
    assert result["status"] == "PAIR_TABLE_LIMIT"
    lookup = build_completion_lookup(STRASSEN, config(), oracle_assisted=True)
    result = complete_core(TARGET, tuple(range(3)), 4, STRASSEN, lookup,
                           config(),
                           oracle_assisted=True, maximum_probes=0)
    assert result["status"] == "LOOKUP_LIMIT"


def test_audit_is_deterministic_excluding_timing():
    lookup = build_completion_lookup(STRASSEN, config(), oracle_assisted=True)
    first = audit_completion_depth(TARGET, tuple(range(7)), STRASSEN, lookup, config(), oracle_assisted=True)
    second = audit_completion_depth(TARGET, tuple(range(7)), STRASSEN, lookup, config(), oracle_assisted=True)
    for result in (first, second):
        result.pop("wall_clock_seconds")
        for depth in result["depths"].values():
            depth.pop("wall_clock_seconds")
            depth.pop("pair_table_build_seconds")
            for core in depth["core_records"]:
                core["work"].pop("probe_seconds", None)
                core["work"].pop("validation_seconds", None)
                core["work"].pop("wall_clock_seconds", None)
            depth["work"].pop("probe_seconds", None)
            depth["work"].pop("validation_seconds", None)
    assert first == second


def test_recombination_positive_control_is_legal_and_deduplicated():
    extras = canonical_action_pool(standard_2x2_decomposition())
    pool = canonical_action_pool((*STRASSEN, *extras))
    indices = {action: index for index, action in enumerate(pool)}
    left = tuple(sorted(indices[action] for action in (*STRASSEN[:4], *extras[:3])))
    right = tuple(sorted(indices[action] for action in (*STRASSEN[4:], *extras[3:7])))
    parents = [{"candidate_hash": state_identity(state, pool), "state": state,
                "metrics": vars(residual_metrics(TARGET, tuple(pool[i] for i in state), "nnz_l1", 4))}
               for state in (left, right)]
    lookup = build_completion_lookup(pool, config(), oracle_assisted=True)
    rows = recombine_parents(TARGET, parents, pool, lookup, config(), oracle_assisted=True,
                             top_k=2, maximum_parent_pairs=1, maximum_partial_children_per_pair=300)
    children = rows[0]["partial_children"]
    assert any(child["verified_matches"] for child in children)
    assert all(len(child["partial_state"]) == len(set(child["partial_state"])) == 4 for child in children)
    exact = {tuple(state) for child in children for state in child["exact_states"]}
    assert len(exact) == 1
    assert all(sum_actions(tuple(pool[i] for i in state), (4, 4, 4)) == TARGET for state in exact)


def test_discovery_rejects_oracle_pair_table_and_candidate_source():
    oracle = build_completion_lookup(STRASSEN, config(), oracle_assisted=True)
    with pytest.raises(ValueError, match="oracle pair table"):
        audit_completion_depth(TARGET, tuple(range(7)), STRASSEN, oracle, config(), oracle_assisted=False)
    source = Path(__file__).resolve().parents[1] / "artifacts/tensor-game/tempered-study-oracle-control-v1"
    with pytest.raises(ValueError, match="oracle status"):
        load_candidates(source, STRASSEN, oracle_assisted=False)


def test_aggregate_counts_match_raw_audit_and_recombination():
    lookup = build_completion_lookup(STRASSEN, config(), oracle_assisted=True)
    audit = audit_completion_depth(TARGET, tuple(range(7)), STRASSEN, lookup, config(), oracle_assisted=True)
    summary = aggregate([audit], [], lookup, oracle_assisted=True)
    for depth in (2, 3, 4):
        assert summary["completion_depths"][str(depth)]["verified_matches"] == audit["depths"][str(depth)]["verified_matches"]
    assert summary["candidates_audited"] == 1


def test_bounded_retention_records_actual_strict_incumbents():
    pool = canonical_action_pool((*STRASSEN, *standard_2x2_decomposition()))
    settings = TemperedSearchConfig(temperatures=(1.0, 2.0), replicas=2,
                                    rounds=5, local_steps_per_round=2,
                                    retain_improvements_limit=4, seed=71)
    first = parallel_tempering_search(TARGET, pool, settings)["retained_improvements"]
    second = parallel_tempering_search(TARGET, pool, settings)["retained_improvements"]
    assert first == second
    assert 1 <= len(first) <= 4
    assert first[0]["round"] == 0
    assert all(first[i]["metrics"]["energy"] > first[i + 1]["metrics"]["energy"]
               for i in range(len(first) - 1))
