import copy

import pytest

from matmul_inspector.hardware_aware_tensor_candidates import (
    Rank8Config, annotate_pareto, descriptor_vector, dominates, run_study,
    select_representatives, validated_candidate,
)
from matmul_inspector.tensor_game import RankOneAction, standard_2x2_decomposition
from matmul_inspector.tensor_game_candidate_dataset import permute_terms, sign_scale_term, validate_decomposition
from matmul_inspector.tensor_game_change_of_basis import transform_decomposition


def test_baseline_exact_and_known_descriptors():
    validate_decomposition(standard_2x2_decomposition())
    result = run_study(Rank8Config(max_attempts=1))
    baseline = result["baseline"]
    assert baseline["exact_reconstruction_status"] == "PASS"
    assert baseline["rank"] == 8
    assert baseline["descriptor_vector"] == {
        "pre_additions": 0, "post_additions": 4, "coefficient_scalings": 0,
        "total_support": 24, "nonzero_coefficients": 24,
        "coefficient_l1": 24, "max_abs_coefficient": 1,
    }
    assert result["rank7_reference"]["rank"] == 7
    assert result["aggregate_summary"]["unique_canonical_rank8_candidates"] == 1


def test_transformed_exact_and_invalid_rejected():
    base = standard_2x2_decomposition()
    identity = ((1, 0), (0, 1))
    shear = ((1, 1), (0, 1))
    transformed = transform_decomposition(base, shear, identity, identity)
    candidate = validated_candidate(transformed, {"type": "test"}, Rank8Config())
    assert candidate["exact_reconstruction_status"] == "PASS"
    assert candidate["descriptor_vector"]["pre_additions"] > 0
    broken = list(transformed)
    first = broken[0]
    broken[0] = RankOneAction((first.u[0] + 1, *first.u[1:]), first.v, first.w)
    with pytest.raises(ValueError, match="exactly reconstruct"):
        validated_candidate(broken, {"type": "test"}, Rank8Config())
    with pytest.raises(ValueError, match="eight terms"):
        validated_candidate(transformed[:-1], {"type": "test"}, Rank8Config())


def test_permutation_sign_equivalence_canonical_dedup_and_signatures():
    base = standard_2x2_decomposition()
    permuted = permute_terms(base, tuple(reversed(range(8))))
    changed = list(permuted)
    changed[0] = sign_scale_term(changed[0], (-1, -1, 1))
    rows = [validated_candidate(actions, {"type": "test"}, Rank8Config())
            for actions in (base, permuted, changed)]
    assert len({row["canonical_candidate_hash"] for row in rows}) == 1
    assert len({row["structural_signature_id"] for row in rows}) == 1
    assert rows[0]["descriptor_vector"] == descriptor_vector(rows[0])


def test_repeated_paths_and_seed_reproducibility():
    config = Rank8Config(seed=19, max_attempts=48, max_accepted=48)
    first = run_study(config)
    second = run_study(config)
    assert first == second
    assert first["aggregate_summary"]["generation_attempts"] == 48
    assert first["aggregate_summary"]["exact_valid_candidates"] == 48
    assert first["aggregate_summary"]["duplicate_candidates"] == 48 - len(first["candidates"])
    assert all(row["equivalent_generation_path_count"] == len(row["generation_paths"])
               for row in first["candidates"])


def test_limits_and_rejections():
    result = run_study(Rank8Config(seed=7, max_attempts=20, maximum_coefficient=1,
                                    maximum_total_support=24))
    assert result["aggregate_summary"]["rejected_candidates"] > 0
    assert result["aggregate_summary"]["exact_valid_candidates"] == result["aggregate_summary"]["generation_attempts"]
    assert all(row["descriptor_vector"]["total_support"] <= 24 for row in result["candidates"])
    anchor_only = run_study(Rank8Config(max_attempts=1, maximum_total_support=1))
    assert anchor_only["baseline"]["descriptor_vector"]["total_support"] == 24


def test_synthetic_pareto_dominance_and_ties():
    def row(identifier, pre, post, scale, support):
        return {"canonical_candidate_hash": identifier,
                "descriptor_vector": {"pre_additions": pre, "post_additions": post,
                                      "coefficient_scalings": scale, "total_support": support}}
    rows = [row("a", 0, 5, 0, 24), row("b", 1, 6, 0, 30),
            row("c", 2, 4, 0, 28), row("d", 0, 5, 0, 24)]
    assert dominates(rows[0]["descriptor_vector"], rows[1]["descriptor_vector"])
    assert not dominates(rows[0]["descriptor_vector"], rows[2]["descriptor_vector"])
    assert not dominates(rows[0]["descriptor_vector"], rows[3]["descriptor_vector"])
    frontier = annotate_pareto(rows)
    assert [item["canonical_candidate_hash"] for item in frontier] == ["a", "c", "d"]
    assert rows[1]["pareto_dominator_count"] == 2


def test_anchor_and_deterministic_unique_representatives():
    result = run_study(Rank8Config(max_attempts=64, max_accepted=64))
    frontier = result["pareto_frontier"]
    baseline_hash = result["baseline"]["canonical_candidate_hash"]
    selected = select_representatives(frontier, baseline_hash, 16)
    assert selected == select_representatives(copy.deepcopy(frontier), baseline_hash, 16)
    assert selected[0]["canonical_candidate_hash"] == baseline_hash
    assert len({item["canonical_candidate_hash"] for item in selected}) == len(selected)
    assert len(selected) <= 16
    assert all(item["canonical_candidate_hash"] in {row["canonical_candidate_hash"] for row in frontier}
               for item in selected)
    assert all(row["rank"] == 8 for row in frontier)
    assert result["rank7_reference"]["canonical_id"] not in {row["canonical_id"] for row in frontier}
