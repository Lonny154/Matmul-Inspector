import copy
import pytest
from matmul_inspector.tensor_game import RankOneAction, standard_2x2_decomposition
from matmul_inspector.tensor_game_candidate_dataset import _record, permute_terms, sign_scale_term
from matmul_inspector.tensor_game_diversity_search import DiscoveryConfig, DiversityArchive, discover

def test_seeded_search_is_deterministic_and_budgeted():
    config=DiscoveryConfig(rank=8,seed=9,node_budget=3,beam_width=4,max_actions=2,mode="diversity",paths_per_state=2)
    first=discover(config); second=discover(config)
    for row in (first,second): row.pop("wall_seconds")
    assert first==second and first["nodes_expanded"]<=3 and first["budget_exhausted"]

def test_archive_validates_deduplicates_and_prefers_novel_signatures():
    actions=standard_2x2_decomposition(); base=_record("standard","known",actions,[]); archive=DiversityArchive(2)
    assert archive.add(base) and not archive.add(base)
    permuted=_record("permuted","known",permute_terms(actions,tuple(reversed(range(8)))),[])
    assert archive.add(permuted)
    changed=list(actions); changed[0]=sign_scale_term(changed[0],(-1,1,-1))
    distinct=_record("scaled","known",tuple(changed),[]); assert archive.add(distinct)
    assert len(archive.records)==2 and distinct["candidate_id"] in {x["candidate_id"] for x in archive.records}
    broken=copy.deepcopy(base); broken["actions"]=broken["actions"][:-1]
    with pytest.raises(ValueError): archive.add(broken)

def test_incomplete_trajectory_is_not_a_solution():
    action=RankOneAction((1,0,0,0),(1,0,0,0),(1,0,0,0))
    result=discover(DiscoveryConfig(1,0,node_budget=1,beam_width=1,max_actions=1),lambda residual,limit:(action,))
    assert not result["success"] and result["solutions"]==[]
