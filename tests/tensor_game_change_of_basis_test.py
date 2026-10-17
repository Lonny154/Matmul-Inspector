import json
import random
import pytest
from matmul_inspector.tensor_game import matrix_multiplication_tensor, multiply_with_decomposition, standard_2x2_decomposition, strassen_2x2_decomposition
from matmul_inspector.tensor_game_bounded_residual import create_residual_candidate_provider
from matmul_inspector.tensor_game_candidate_dataset import validate_decomposition, write_dataset
from matmul_inspector.tensor_game_change_of_basis import ChangeOfBasisConfig, determinant, generate_change_of_basis_candidates, inverse_unimodular, multiply, transform_decomposition, transformation_triples, unimodular_matrices
from matmul_inspector.tensor_game_triton_lowering import create_lowering

def test_exact_unimodular_inverse_and_singular_rejection():
    matrix=((1,1),(1,0)); inverse=inverse_unimodular(matrix)
    assert determinant(matrix)==-1 and multiply(matrix,inverse)==((1,0),(0,1))
    with pytest.raises(ValueError,match="determinant"): inverse_unimodular(((1,2),(2,4)))

def test_change_of_basis_reconstructs_and_executes_identity():
    p=((1,1),(0,1)); q=((1,0),(-1,1)); r=((0,1),(1,0))
    transformed=transform_decomposition(strassen_2x2_decomposition(),p,q,r)
    validate_decomposition(transformed); assert create_lowering("changed",transformed).metadata.rank==7
    assert multiply_with_decomposition(((2,3),(5,7)),((11,13),(17,19)),(2,2,2),transformed)==((73,83),(174,198))

def test_enumeration_sampling_limits_and_determinism():
    assert len(unimodular_matrices(1))==40
    enumerated=transformation_triples(ChangeOfBasisConfig(combination_limit=5,mode="enumerate"))
    sampled=transformation_triples(ChangeOfBasisConfig(combination_limit=5,mode="sample",seed=4))
    assert len(enumerated)==len(sampled)==5 and sampled==transformation_triples(ChangeOfBasisConfig(combination_limit=5,mode="sample",seed=4))
    strict=ChangeOfBasisConfig(combination_limit=8,mode="enumerate",maximum_coefficient=1)
    first,summary=generate_change_of_basis_candidates(strict); second,summary2=generate_change_of_basis_candidates(strict)
    assert first==second and summary==summary2 and summary["attempted_transformations"]==16
    assert summary["rejection_reasons"].get("maximum_coefficient",0)>0

def test_dedup_provenance_and_stable_serialization(tmp_path):
    records,summary=generate_change_of_basis_candidates(ChangeOfBasisConfig(combination_limit=4,mode="sample",seed=8))
    assert len({row["candidate_id"] for row in records})==len(records)
    assert all(row["parent_decomposition"]==row["family_id"] and row["change_of_basis"]["inverses"] for row in records)
    first=tmp_path/"first"; second=tmp_path/"second"
    write_dataset(first,records,{"summary":summary}); write_dataset(second,records,{"summary":summary})
    assert (first/"candidates.jsonl").read_bytes()==(second/"candidates.jsonl").read_bytes()
    loaded=[json.loads(line) for line in (first/"candidates.jsonl").read_text().splitlines()]
    assert len(loaded)==len(records) and loaded[0]["candidate_id"]==records[0]["candidate_id"]

def test_default_reference_is_unchanged():
    residual=matrix_multiplication_tensor(2,2,2)
    first=create_residual_candidate_provider().propose_actions(residual,8,random.Random(1),12)
    second=create_residual_candidate_provider().propose_actions(residual,8,random.Random(99),12)
    assert first==second and first
