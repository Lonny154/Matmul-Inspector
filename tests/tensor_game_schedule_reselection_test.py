"""CPU-only HA-Schedule-10 replay tests. All new timing fixtures are synthetic."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from matmul_inspector.tensor_game_schedule_reselection import (
    compare_reselection, evidence_votes, load_retained_corpus, make_observation,
    select_batch1, select_batch2, validate_observations,
)

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def retained():
    candidates, provenance = load_retained_corpus(REPO)
    return candidates, select_batch1(candidates), provenance


def synthetic(left, right, *, mode="EAGER", reversed=False, equal=False):
    a, b = ([1.0] * 5, [2.0] * 5)
    if reversed:
        a, b = b, a
    if equal:
        b = a
    return make_observation(left, right, mode=mode, left_ms=a, right_ms=b,
        provenance={"evidence_kind": "SYNTHETIC_TEST_FIXTURE"},
        benchmark={"rounds": 5, "timing": "synthetic fixture"},
        correctness={left["schedule_hash"]: "PASS", right["schedule_hash"]: "PASS"})


def test_retained_corpus_and_stable_batch1(retained):
    candidates, batch1, provenance = retained
    assert len(candidates) == 156
    assert sum(row["retained_correctness"] == "PASS" for row in candidates) == 132
    assert len(batch1) == 8
    assert {row["matrix_size"] for row in batch1} == {256}
    assert len({row["schedule_hash"] for row in batch1}) == 8
    assert select_batch1(list(reversed(candidates))) == batch1
    assert all(len(row["schedule_hash"]) == 64 for row in batch1)
    assert all(len(value) == 64 for value in provenance.values())


def test_serialization_replay_and_mode_isolation(retained):
    candidates, batch1, _ = retained
    observation = synthetic(batch1[0], batch1[1])
    loaded = json.loads(json.dumps([observation]))
    validate_observations(loaded, candidates, "EAGER")
    frozen = select_batch2(candidates, batch1, [], mode="EAGER")
    updated = select_batch2(candidates, batch1, loaded, mode="EAGER")
    assert compare_reselection(frozen, updated)["changed"]
    assert updated == select_batch2(list(reversed(candidates)), batch1, loaded, mode="EAGER")
    assert select_batch2(candidates, batch1, loaded, mode="CUDA_GRAPH") == select_batch2(
        candidates, batch1, [], mode="CUDA_GRAPH")
    assert len(updated) == 4
    assert all(row["supporting_tokens"] for row in updated if row["ordinal_support"])
    assert all(len(token["observation_ids"][0]) == 64 for row in updated
               for token in row["supporting_tokens"].values())


def test_failed_candidate_cannot_supply_preference_or_enter_batch2(retained):
    candidates, batch1, _ = retained
    failed = next(row for row in candidates if row["retained_correctness"] != "PASS")
    failed_peer = next(row for row in candidates if row["retained_correctness"] != "PASS" and
                       row["matrix_size"] == failed["matrix_size"] and
                       row["schedule_hash"] != failed["schedule_hash"])
    with pytest.raises(ValueError, match="failed"):
        make_observation(failed_peer, failed, mode="EAGER", left_ms=[1.0] * 5,
            right_ms=[2.0] * 5, provenance={"evidence_kind": "SYNTHETIC_TEST_FIXTURE"},
            benchmark={"rounds": 5},
            correctness={failed_peer["schedule_hash"]: "PASS", failed["schedule_hash"]: "FAIL"})
    assert failed["schedule_hash"] not in {row["schedule_hash"] for row in
        select_batch2(candidates, batch1, [], mode="EAGER")}
    observation = synthetic(batch1[0], batch1[1])
    changed = deepcopy(observation)
    changed["left_round_medians_ms"] = [3.0] * 5
    with pytest.raises(ValueError, match="hash"):
        validate_observations([changed], candidates, "EAGER")


def test_contradiction_and_indistinguishable_pair(retained):
    candidates, batch1, _ = retained
    first = synthetic(batch1[0], batch1[1])
    with pytest.raises(ValueError, match="duplicate"):
        validate_observations([first, first], candidates, "EAGER")
    opposite = synthetic(batch1[0], batch1[1], reversed=True)
    votes = evidence_votes([first, opposite], candidates, "EAGER")
    assert votes and all(item["net_votes"] == 0 for item in votes.values())
    assert select_batch2(candidates, batch1, [first, opposite], mode="EAGER") == select_batch2(
        candidates, batch1, [], mode="EAGER")
    parity = synthetic(batch1[0], batch1[1], equal=True)
    assert parity["preferred"] is None and parity["confidence"] == "INDETERMINATE"
    assert select_batch2(candidates, batch1, [parity], mode="EAGER") == select_batch2(
        candidates, batch1, [], mode="EAGER")


def test_missing_evidence_budget_and_context(retained):
    candidates, batch1, _ = retained
    assert select_batch2(candidates, batch1, [], mode="EAGER", budget=0) == []
    assert len(select_batch2(candidates, batch1, [], mode="EAGER", budget=2)) == 2
    with pytest.raises(ValueError, match="budget"):
        select_batch2(candidates, batch1, [], mode="EAGER", budget=5)
    with pytest.raises(ValueError, match="algorithm"):
        synthetic(batch1[0], batch1[4])
    observation = synthetic(batch1[0], batch1[1])
    unrelated = batch1[0].copy()
    unrelated["schedule_hash"] = "a" * 64
    bad = observation.copy()
    bad["left"] = unrelated["schedule_hash"]
    with pytest.raises(ValueError, match="Batch 1"):
        select_batch2(candidates, batch1, [bad], mode="EAGER")
