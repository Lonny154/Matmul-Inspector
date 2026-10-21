import json
import pytest
from matmul_inspector.tensor_game import RankOneAction, standard_2x2_decomposition, strassen_2x2_decomposition
from matmul_inspector.tensor_game_candidate_dataset import canonical_actions, generate_candidates, index_symmetry, permute_terms, sign_scale_term, validate_decomposition, write_dataset
from matmul_inspector.tensor_game_triton_lowering import create_lowering

def test_exact_transformations_and_lowering_compatibility():
    base=strassen_2x2_decomposition(); order=tuple(reversed(range(7)))
    variants=[permute_terms(base,order),index_symmetry(base,True,False,True)]
    changed=list(base); changed[0]=sign_scale_term(changed[0],(-1,-1,1)); variants.append(tuple(changed))
    for actions in variants: validate_decomposition(actions); assert create_lowering("candidate",actions).metadata.rank==7
    assert canonical_actions(base)==canonical_actions(variants[0])==canonical_actions(variants[2])

def test_invalid_transformations_are_rejected():
    with pytest.raises(ValueError,match="permutation"): permute_terms(strassen_2x2_decomposition(),range(6))
    with pytest.raises(ValueError,match="product"): sign_scale_term(strassen_2x2_decomposition()[0],(-1,1,1))
    with pytest.raises(ValueError,match="reconstruct"): validate_decomposition(strassen_2x2_decomposition()[:-1])

def test_generation_is_deterministic_deduplicated_and_provenanced(tmp_path):
    first,summary=generate_candidates(maximum=64); second,summary2=generate_candidates(maximum=64)
    assert first==second and summary==summary2 and len(first)==summary["unique_exact_representations"]
    assert {r["family_id"] for r in first}=={"conventional","strassen"}
    assert all(r["exact_validation"] and r["transformation_provenance"] for r in first)
    output=tmp_path/"dataset"; write_dataset(output,first,{"summary":summary})
    before=(output/"candidates.jsonl").read_bytes(); assert len(before.splitlines())==len(first)
    with pytest.raises(FileExistsError): write_dataset(output,first,{})

def test_invalid_archive_is_counted_not_exported(tmp_path):
    archive=tmp_path/"archive.json"; archive.write_text(json.dumps({"solutions":[{"actions":[{"u":[1,0,0,0],"v":[1,0,0,0],"w":[1,0,0,0]}]}]}))
    records,summary=generate_candidates(maximum=16,archive=archive)
    assert records and summary["invalid_attempts"]==1 and summary["archive_records_considered"]==1
