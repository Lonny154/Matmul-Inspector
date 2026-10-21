import json

import pytest

from matmul_inspector.hardware_aware_rank89_search import (
    DEFAULT_CATALOG, SolveLimits, StructuredBounds, _solution, catalog_ids,
    classify_candidate, control_pool, distinct_input_lower_bounds,
    frozen_existing_pool, model_size, rank9_split_parent, reuse_descriptors,
    solve_exact_decomposition, structured_pool, synthetic_rank9_control,
)
from matmul_inspector.tensor_game import (
    standard_2x2_decomposition, strassen_2x2_decomposition,
)
from matmul_inspector.tensor_game_candidate_dataset import validate_decomposition


def needs_solver():
    pytest.importorskip("ortools")


def small_limits(**overrides):
    values = {"max_models": 1, "max_accepted": 1, "timeout_seconds": 2.0}
    values.update(overrides)
    return SolveLimits(**values)


def test_reuse_reference_vectors_and_exact_flattening_bounds():
    conventional = reuse_descriptors(standard_2x2_decomposition())
    strassen = reuse_descriptors(strassen_2x2_decomposition())
    synthetic = reuse_descriptors(synthetic_rank9_control())
    assert distinct_input_lower_bounds() == {
        "distinct_u_forms": 4, "distinct_v_forms": 4, "distinct_input_forms": 8}
    assert (conventional["distinct_u_forms"], conventional["distinct_v_forms"],
            conventional["total_reuse_count"], conventional["max_u_multiplicity"]) == (4, 4, 8, 2)
    assert (strassen["distinct_u_forms"], strassen["distinct_v_forms"],
            strassen["total_reuse_count"]) == (7, 7, 0)
    assert (synthetic["distinct_u_forms"], synthetic["distinct_v_forms"],
            synthetic["total_reuse_count"], synthetic["max_u_multiplicity"]) == (4, 4, 10, 3)
    assert conventional == reuse_descriptors(standard_2x2_decomposition())


def test_synthetic_rank9_is_legal_distinct_and_exact():
    actions = synthetic_rank9_control()
    assert len(actions) == len(set(actions)) == 9
    validate_decomposition(actions)
    assert all(value in (-1, 0, 1) for action in actions
               for value in action.u + action.v + action.w)
    assert rank9_split_parent(actions, catalog_ids()) is not None


def test_canonical_form_reuse_counts_sign_equivalent_terms():
    from matmul_inspector.tensor_game_candidate_dataset import sign_scale_term
    actions = list(standard_2x2_decomposition())
    actions[0] = sign_scale_term(actions[0], (-1, -1, 1))
    assert reuse_descriptors(actions) == reuse_descriptors(standard_2x2_decomposition())


def test_rank8_control_exact_and_required_action_infeasible():
    needs_solver()
    pool, _ = control_pool(8, distractors=0)
    exact = solve_exact_decomposition(pool, 8, limits=small_limits(), catalog_hashes=catalog_ids())
    assert exact["status"] == "EXACT" and exact["unique_solution_count"] == 1
    assert exact["solutions"][0]["exact_reconstruction_status"] == "PASS"
    assert exact["solutions"][0]["project_catalog_classification"] == "CONVENTIONAL_ANCHOR"
    missing = solve_exact_decomposition(pool[1:], 8, limits=small_limits())
    assert missing["status"] == "INFEASIBLE"


def test_rank9_control_exact_and_derived_classification():
    needs_solver()
    pool, _ = control_pool(9, distractors=0)
    result = solve_exact_decomposition(pool, 9, limits=small_limits(), catalog_hashes=catalog_ids())
    assert result["status"] == "EXACT"
    solution = result["solutions"][0]
    assert solution["project_catalog_classification"] == "SYNTHETIC_CONTROL"
    assert solution["known_rank8_split_parent_hash"] is not None
    assert solution["rank"] == 9


def test_timeout_and_preflight_are_not_infeasible():
    needs_solver()
    pool, _ = control_pool(8, distractors=0)
    limited = solve_exact_decomposition(pool, 8, limits=small_limits(max_pool_actions=7))
    assert limited["status"] == "RESOURCE_LIMIT"
    assert limited["solver_status"] == "NOT_CONSTRUCTED"
    timed = solve_exact_decomposition(pool, 8, limits=small_limits(timeout_seconds=1e-9))
    assert timed["status"] in {"TIME_LIMIT", "UNKNOWN"}
    assert timed["status"] != "INFEASIBLE"


def test_auxiliary_variable_linkage_and_optimum():
    needs_solver()
    pool, _ = control_pool(8, distractors=0)
    size = model_size(pool, "MIN_DISTINCT_INPUT_FORMS")
    assert size["action_variables"] == 8
    assert size["auxiliary_variables"] == 8
    minimized = solve_exact_decomposition(pool, 8, objective="MIN_DISTINCT_INPUT_FORMS",
                                           limits=small_limits())
    assert minimized["status"] == "EXACT"
    assert minimized["objective_value"] == 8
    assert minimized["solutions"][0]["reuse_descriptor_vector"]["distinct_input_forms"] == 8
    assert minimized["solutions"][0]["objective_proven_optimal"]
    constrained = solve_exact_decomposition(pool, 8, objective="MIN_DISTINCT_INPUT_FORMS",
                                             min_u_reuse=5, limits=small_limits())
    assert constrained["status"] == "INFEASIBLE"
    maximized = solve_exact_decomposition(pool, 8, objective="MAX_TOTAL_REUSE", limits=small_limits())
    assert maximized["objective_value"] == 8


def test_enumeration_dedup_and_determinism():
    needs_solver()
    pool, _ = structured_pool(StructuredBounds(), 512)
    settings = small_limits(max_pool_actions=512, max_models=3, max_accepted=3)
    first = solve_exact_decomposition(pool, 8, limits=settings, seed=7)
    second = solve_exact_decomposition(pool, 8, limits=settings, seed=7)
    hashes = [row["canonical_decomposition_hash"] for row in first["solutions"]]
    assert hashes == [row["canonical_decomposition_hash"] for row in second["solutions"]]
    assert len(hashes) == len(set(hashes)) == 1
    assert first["status"] == "EXACT"
    assert first["enumeration_stop_reason"] == "exhausted_finite_pool"
    nine = solve_exact_decomposition(pool, 9, objective="MIN_DISTINCT_INPUT_FORMS",
                                     limits=settings, seed=7, catalog_hashes=catalog_ids())
    assert nine["status"] == "EXACT" and nine["solutions"]
    assert all(row["rank"] == 9 for row in nine["solutions"])
    assert all(row["known_rank8_split_parent_hash"] for row in nine["solutions"])


def test_independent_reconstruction_failure_rejects_model():
    pool, _ = control_pool(8, distractors=0)
    with pytest.raises(ValueError, match="exactly reconstruct"):
        _solution(pool, tuple(range(7)), {}, set())


def test_catalog_hash_classification_and_rank_separation():
    known = catalog_ids()
    anchor = next(iter(json.loads(line)["canonical_id"] for line in DEFAULT_CATALOG.read_text().splitlines()
                       if json.loads(line)["static_cost"]["pre_additions"] == 0))
    assert classify_candidate(anchor, 8, known, anchor, "other") == "CONVENTIONAL_ANCHOR"
    assert classify_candidate("known", 8, {"known"}, anchor, "other") == "EXISTING_PROJECT_CANDIDATE"
    assert classify_candidate("unknown", 8, known, anchor, "other") == "NEW_TO_PROJECT_CATALOG"
    assert classify_candidate("synthetic", 9, known, anchor, "synthetic") == "SYNTHETIC_CONTROL"


def test_discovery_pool_provenance_and_no_oracle_injection(tmp_path):
    data = {"residual_ranked": {"actions": [{"u": [1, 0, 0, 0], "v": [1, 0, 0, 0], "w": [1, 0, 0, 0]}]},
            "strassen_oracle": {"actions": []}}
    source = tmp_path / "pools.json"
    source.write_text(json.dumps(data))
    pool, provenance = frozen_existing_pool(source, ("residual_ranked",), 5)
    assert len(pool) == 1
    assert all(item["source"] == "frozen_existing" for group in provenance.values() for item in group)
    with pytest.raises(ValueError, match="oracle-only"):
        frozen_existing_pool(source, ("strassen_oracle",), 5)
    structured, origins = structured_pool(StructuredBounds(), 20)
    assert len(structured) == 20
    assert all(item["source"] == "structured_domain" for group in origins.values() for item in group)
