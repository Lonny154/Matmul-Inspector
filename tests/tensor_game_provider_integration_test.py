import json,random

import pytest

from matmul_inspector.tensor_game import RankOneAction,validate_action
from matmul_inspector.tensor_game_bounded_residual import BoundedResidualPolicy,bounded_residual_candidates_memory_efficient,create_residual_candidate_provider
from matmul_inspector.tensor_game_experience import action_identity
from matmul_inspector.tensor_game_search_calibration import ExhaustiveResidualPolicy
from matmul_inspector.tensor_game_cli import load_target,main
from matmul_inspector.cli import COMMANDS

def residual():return tuple(tuple(tuple(1 if i==j==k else 0 for k in range(4)) for j in range(4)) for i in range(4))

def test_default_is_bounded_and_explicit_exhaustive_remains_available():
    default=create_residual_candidate_provider();explicit=create_residual_candidate_provider("exhaustive")
    assert isinstance(default,BoundedResidualPolicy) and isinstance(explicit,ExhaustiveResidualPolicy)
    assert explicit.propose_actions(residual(),1,random.Random(1),12)==create_residual_candidate_provider("exhaustive").propose_actions(residual(),1,random.Random(1),12)

def test_bounded_selection_metadata_legality_budget_dedup_and_determinism():
    policy=create_residual_candidate_provider("bounded_128");assert isinstance(policy,BoundedResidualPolicy) and policy.name=="bounded_128"
    first=policy.propose_actions(residual(),1,random.Random(1),12);second=create_residual_candidate_provider("bounded_128").propose_actions(residual(),1,random.Random(1),12)
    assert first==second and len(first)<=12 and len({action_identity(x.action) for x in first})==len(first)
    for item in first:validate_action(item.action,(4,4,4),frozenset((-1,0,1)))
    expected,_=bounded_residual_candidates_memory_efficient(residual(),128,12)
    assert [action_identity(x.action) for x in first]==[x["action_id"] for x in expected]

def test_unsupported_provider_is_clear():
    with pytest.raises(ValueError,match="unsupported candidate provider"):create_residual_candidate_provider("unknown")
    with pytest.raises(SystemExit):main(["--target-json","missing.json","--candidate-provider","unknown"])

def test_cli_default_and_explicit_metadata(tmp_path,capsys):
    assert "tensor-game-search" in COMMANDS
    path=tmp_path/"target.json";path.write_text(json.dumps({"target_id":"one","construction_length":1,"target":residual()}))
    assert main(["--target-json",str(path),"--simulations","1"])==0;default=json.loads(capsys.readouterr().out);assert default["candidate_provider"]=="bounded_128" and default["provider_is_default"]
    assert main(["--target-json",str(path),"--candidate-provider","bounded_128","--simulations","1"])==0;bounded=json.loads(capsys.readouterr().out);assert bounded["candidate_provider"]=="bounded_128" and bounded["provider_is_default"]
    assert main(["--target-json",str(path),"--candidate-provider","exhaustive","--simulations","1"])==0;exhaustive=json.loads(capsys.readouterr().out);assert exhaustive["candidate_provider"]=="exhaustive" and not exhaustive["provider_is_default"]

def test_manifest_requires_unambiguous_target(tmp_path):
    path=tmp_path/"targets.json";path.write_text(json.dumps({"targets":[{"target_id":"a","construction_length":1,"target":residual()},{"target_id":"b","construction_length":1,"target":residual()}]}))
    with pytest.raises(ValueError,match="--target-id"):load_target(path,None)
    row,_,_=load_target(path,"b");assert row["target_id"]=="b"
