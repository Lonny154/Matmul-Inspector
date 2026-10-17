"""Bounded exact meet-in-the-middle diagnostics for the Tensor Game."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from itertools import combinations
import json
import math
import random
import time
from typing import Iterable, Iterator, Sequence

from .tensor_game import RankOneAction, Tensor, outer_product, sum_actions, tensor_shape, validate_action
from .tensor_game_experience import action_identity
from .tensor_game_search import canonicalize_action


def add_tensors(left: Tensor, right: Tensor) -> Tensor:
    """Add equally shaped integer tensors exactly."""
    shape = tensor_shape(left)
    if tensor_shape(right) != shape:
        raise ValueError("tensor shapes must match")
    return tuple(tuple(tuple(left[i][j][k] + right[i][j][k] for k in range(shape[2]))
                       for j in range(shape[1])) for i in range(shape[0]))


def subtract_tensors(left: Tensor, right: Tensor) -> Tensor:
    """Subtract equally shaped integer tensors exactly."""
    shape = tensor_shape(left)
    if tensor_shape(right) != shape:
        raise ValueError("tensor shapes must match")
    return tuple(tuple(tuple(left[i][j][k] - right[i][j][k] for k in range(shape[2]))
                       for j in range(shape[1])) for i in range(shape[0]))


def partial_sum_identity(tensor: Tensor) -> str:
    """Return a stable identity for an exact partial sum."""
    return hashlib.sha256(json.dumps(tensor, separators=(",", ":")).encode()).hexdigest()


def canonical_action_pool(actions: Iterable[RankOneAction]) -> tuple[RankOneAction, ...]:
    """Validate, sign-canonicalize, deduplicate, and deterministically order actions."""
    unique = set()
    for action in actions:
        validate_action(action, (4, 4, 4), frozenset((-1, 0, 1)))
        unique.add(canonicalize_action(action))
    return tuple(sorted(unique, key=lambda action: action.u + action.v + action.w))


def _sum_indices(tensors: Sequence[Tensor], indices: Sequence[int]) -> Tensor:
    total = tensors[indices[0]]
    for index in indices[1:]:
        total = add_tensors(total, tensors[index])
    return total


def _index_stream(pool_size: int, depth: int, maximum: int, seed: int) -> tuple[Iterator[tuple[int, ...]], str, int]:
    if depth <= 0 or depth > pool_size:
        raise ValueError("depth must be in [1, pool size]")
    if maximum <= 0:
        raise ValueError("maximum attempts must be positive")
    total = math.comb(pool_size, depth)
    if total <= maximum:
        return iter(combinations(range(pool_size), depth)), "complete", total

    def sampled() -> Iterator[tuple[int, ...]]:
        rng = random.Random(seed)
        seen: set[tuple[int, ...]] = set()
        while len(seen) < maximum:
            item = tuple(sorted(rng.sample(range(pool_size), depth)))
            if item not in seen:
                seen.add(item)
                yield item

    return sampled(), "sampled", total


@dataclass(frozen=True)
class MeetInMiddleConfig:
    left_depth: int = 3
    right_depth: int = 4
    maximum_left_attempts: int = 25_000
    maximum_right_attempts: int = 25_000
    maximum_stored_sums: int = 25_000
    maximum_seconds: float = 10.0
    seed: int = 20261017

    def validate(self) -> None:
        if self.left_depth <= 0 or self.right_depth <= 0:
            raise ValueError("depths must be positive")
        if min(self.maximum_left_attempts, self.maximum_right_attempts, self.maximum_stored_sums) <= 0:
            raise ValueError("work and storage budgets must be positive")
        if not math.isfinite(self.maximum_seconds) or self.maximum_seconds <= 0:
            raise ValueError("maximum_seconds must be finite and positive")


def meet_in_middle(target: Tensor, actions: Sequence[RankOneAction], config: MeetInMiddleConfig) -> dict:
    """Search one bounded action pool for exact left/right complementary sums."""
    config.validate()
    shape = tensor_shape(target)
    pool = canonical_action_pool(actions)
    if config.left_depth + config.right_depth > len(pool):
        raise ValueError("pool is too small for disjoint partial sums")
    tensors = tuple(outer_product(action) for action in pool)
    left_stream, left_mode, left_space = _index_stream(len(pool), config.left_depth, config.maximum_left_attempts, config.seed)
    right_stream, right_mode, right_space = _index_stream(len(pool), config.right_depth, config.maximum_right_attempts, config.seed + 1)
    started = time.perf_counter()
    deadline = started + config.maximum_seconds
    table: dict[str, list[tuple[Tensor, tuple[int, ...]]]] = {}
    left_attempts = left_duplicates = hash_collisions = 0
    stored_count = 0
    stop_reason = "completed"
    for indices in left_stream:
        if time.perf_counter() >= deadline:
            stop_reason = "time_budget"
            break
        value = _sum_indices(tensors, indices)
        digest = partial_sum_identity(value)
        bucket = table.setdefault(digest, [])
        if bucket and all(existing != value for existing, _ in bucket):
            hash_collisions += 1
        if any(existing == value for existing, _ in bucket):
            left_duplicates += 1
        else:
            if stored_count >= config.maximum_stored_sums:
                if not bucket:
                    del table[digest]
                stop_reason = "storage_budget"
                break
            bucket.append((value, indices))
            stored_count += 1
        left_attempts += 1

    right_attempts = complement_matches = overlap_rejections = 0
    decompositions: dict[str, dict] = {}
    if stop_reason != "time_budget":
        for right_indices in right_stream:
            if time.perf_counter() >= deadline:
                stop_reason = "time_budget"
                break
            right_value = _sum_indices(tensors, right_indices)
            complement = subtract_tensors(target, right_value)
            for stored, left_indices in table.get(partial_sum_identity(complement), ()):
                if stored != complement:
                    hash_collisions += 1
                    continue
                complement_matches += 1
                if set(left_indices) & set(right_indices):
                    overlap_rejections += 1
                    continue
                selected = tuple(pool[index] for index in left_indices + right_indices)
                if sum_actions(selected, shape) != target:
                    continue
                ids = tuple(sorted(action_identity(action) for action in selected))
                decomposition_id = hashlib.sha256("".join(ids).encode()).hexdigest()
                decompositions.setdefault(decomposition_id, {
                    "decomposition_id": decomposition_id,
                    "left_indices": list(left_indices), "right_indices": list(right_indices),
                    "action_ids": list(ids), "exact_reconstruction": True,
                })
            right_attempts += 1
    elapsed = time.perf_counter() - started
    stored = stored_count
    return {
        "schema": "tensor_game_meet_in_middle_result.v1",
        "config": asdict(config), "pool_size": len(pool),
        "left": {"mode": left_mode, "combination_space": left_space, "attempts": left_attempts,
                 "unique_sums_stored": stored, "duplicate_sums": left_duplicates},
        "right": {"mode": right_mode, "combination_space": right_space, "attempts": right_attempts},
        "complement_matches": complement_matches, "overlap_rejections": overlap_rejections,
        "hash_collisions": hash_collisions, "exact_decomposition_count": len(decompositions),
        "decompositions": list(decompositions.values()), "stop_reason": stop_reason,
        "runtime_seconds": elapsed,
    }


def action_domain_complexity(vector_count: int = 80) -> dict:
    """Return exact combinatorial sizes for the canonical ternary action domain."""
    raw = vector_count ** 3
    unique = raw // 4
    unordered = {str(depth): math.comb(unique + depth - 1, depth) for depth in range(2, 5)}
    return {
        "nonzero_ternary_vectors": vector_count, "raw_factor_triples": raw,
        "sign_equivalence_orbit_size": 4, "unique_rank_one_tensors": unique,
        "ordered_two_term_sums": unique ** 2, "unordered_multisets_with_repetition": unordered,
        "minimum_64_byte_storage_estimates": {depth: count * 64 for depth, count in unordered.items()},
        "full_3_plus_4_enumeration_attempts": unordered["3"] + unordered["4"],
    }
