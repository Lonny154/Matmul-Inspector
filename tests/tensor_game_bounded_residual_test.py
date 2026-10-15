import json

import pytest

from matmul_inspector.tensor_game import RankOneAction, subtract_action, validate_action
from matmul_inspector.tensor_game_bounded_residual import bounded_residual_candidates, coarse_pair_scores, exact_score_from_contraction
from matmul_inspector.tensor_game_experience import action_from_payload
from matmul_inspector.tensor_game_residual_supervision import enumerate_residual_teacher, squared_norm
from matmul_inspector.tensor_game_search import canonicalize_action, contract_residual


def tensor():
    return tuple(tuple(tuple(1 if i==j==k else 0 for k in range(4)) for j in range(4)) for i in range(4))


def test_exact_score_identity_and_canonical_legal_candidates():
    residual=tensor(); action=RankOneAction((1,0,0,0),(1,0,0,0),(1,0,0,0))
    d=contract_residual(residual,action.u,action.v); q=sum(x*x for x in action.u)*sum(x*x for x in action.v)
    assert exact_score_from_contraction(squared_norm(residual),d,q,action.w)==squared_norm(subtract_action(residual,action))
    rows,_=bounded_residual_candidates(residual,128)
    assert len(rows)<=12
    for row in rows:
        candidate=action_from_payload(row["action"]); validate_action(candidate,(4,4,4),frozenset((-1,0,1)))
        assert candidate==canonicalize_action(candidate)


def test_determinism_deduplication_ordering_and_exact_rescoring():
    first,audit=bounded_residual_candidates(tensor(),512); second,_=bounded_residual_candidates(tensor(),512)
    assert first==second
    assert len({row["action_id"] for row in first})==len(first)
    assert [row["score"] for row in first]==sorted(row["score"] for row in first)
    assert audit["fully_evaluated_pairs"]==512 and audit["enumerated_uv_pairs"]==6400
    for row in first:
        action=action_from_payload(row["action"])
        assert row["score"]==squared_norm(subtract_action(tensor(),action))


def test_bound_is_continuous_relaxation_and_approximation_discrepancy_is_allowed():
    pairs=coarse_pair_scores(tensor())
    assert len(pairs)==6400 and pairs[0].bound_numerator>=pairs[-1].bound_numerator
    bounded,_=bounded_residual_candidates(tensor(),128); exact,_=enumerate_residual_teacher(tensor(),12)
    assert 0 <= len({x["action_id"] for x in bounded}&{x["action_id"] for x in exact}) <= 12


def test_degenerate_residual_budget_and_no_label_input():
    zero=tuple(tuple(tuple(0 for _ in range(4)) for _ in range(4)) for _ in range(4))
    rows,audit=bounded_residual_candidates(zero,128)
    assert rows==[] and audit["returned_candidates"]==0
    with pytest.raises(ValueError): bounded_residual_candidates(zero,0)
    with pytest.raises(ValueError): bounded_residual_candidates(zero,6401)
