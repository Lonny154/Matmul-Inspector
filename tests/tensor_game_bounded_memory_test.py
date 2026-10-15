import random

from matmul_inspector.tensor_game_bounded_residual import bounded_residual_candidates,bounded_residual_candidates_memory_efficient,coarse_pair_scores,coarse_pair_scores_streaming

def diagonal():return tuple(tuple(tuple(1 if i==j==k else 0 for k in range(4)) for j in range(4)) for i in range(4))
def zero():return tuple(tuple(tuple(0 for _ in range(4)) for _ in range(4)) for _ in range(4))
def dense():return tuple(tuple(tuple(((i*7+j*3+k)%3)-1 for k in range(4)) for j in range(4)) for i in range(4))
def repeated():
    base=diagonal();return tuple(tuple(tuple(base[i][j][k]*2 for k in range(4)) for j in range(4)) for i in range(4))
def near_tie():return tuple(tuple(tuple(1 if (i+j+k)%5==0 else -1 if (i*3+j+k)%7==0 else 0 for k in range(4)) for j in range(4)) for i in range(4))

def assert_equivalent(residual,budget=128):
    original,a=bounded_residual_candidates(residual,budget,12);optimized,b=bounded_residual_candidates_memory_efficient(residual,budget,12)
    assert optimized==original
    assert b["retained_coarse_records"]==budget and b["fully_evaluated_pairs"]==a["fully_evaluated_pairs"]
    assert bounded_residual_candidates_memory_efficient(residual,budget,12)[0]==optimized

def test_selected_pair_identity_order_and_ties_are_exact():
    for residual in (zero(),diagonal(),dense(),repeated(),near_tie()):
        old=coarse_pair_scores(residual)[:128];new=coarse_pair_scores_streaming(residual,128)
        assert new==old
        assert_equivalent(residual)

def test_sparse_dense_random_and_degenerate_differential_suite():
    rng=random.Random(42)
    states=[tuple(tuple(tuple(rng.choice((-1,0,0,0,1)) for _ in range(4)) for _ in range(4)) for _ in range(4)) for _ in range(5)]
    for residual in states:assert_equivalent(residual)

def test_seeded_regression_states_are_exact_without_external_artifacts():
    rng=random.Random(20261015)
    states=[tuple(tuple(tuple(rng.randint(-3,3) for _ in range(4)) for _ in range(4)) for _ in range(4)) for _ in range(8)]
    for residual in states:assert_equivalent(residual)
