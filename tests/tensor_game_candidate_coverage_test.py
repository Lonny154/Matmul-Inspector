from matmul_inspector.tensor_game import RankOneAction
from matmul_inspector.tensor_game_candidate_coverage import aggregate_recall, _normalize
from matmul_inspector.tensor_game_search import canonicalize_action


def test_recall_aggregation_keeps_seed_and_target_denominators_distinct():
    rows = []
    for policy, recalled in (("frozen", True), ("trained", False)):
        for seed in (3, 11):
            row = {"target_group_id": "target", "seed": seed, "policy": policy}
            for cutoff in (1, 12, 32, 64, 128, 256):
                row[f"recall_at_{cutoff}"] = recalled
            rows.append(row)
    result = aggregate_recall(rows)
    assert result["frozen"]["per_run_recall"]["12"] == 2
    assert result["frozen"]["unique_target_recall"]["12"] == 1
    assert result["trained"]["unique_target_recall"]["256"] == 0


def test_combined_weights_are_normalized_without_changing_order():
    first = RankOneAction((1,), (1,), (1,))
    second = RankOneAction((-1,), (1,), (1,))
    result = _normalize([(first, 0.25, "frozen"), (second, 0.75, "trained")])
    assert [row.action for row in result] == [canonicalize_action(first), canonicalize_action(second)]
    assert [row.prior for row in result] == [0.25, 0.75]
