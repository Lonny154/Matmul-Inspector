"""Bounded parallel-tempering experiments over complete Tensor Game states."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from itertools import combinations
import json
import math
import random
import time
from typing import Sequence

from .tensor_game import RankOneAction, Tensor, outer_product, sum_actions, tensor_shape
from .tensor_game_experience import action_identity
from .tensor_game_meet_in_middle import add_tensors, partial_sum_identity, subtract_tensors
from .tensor_game_search import canonicalize_action


@dataclass(frozen=True)
class ResidualMetrics:
    nonzero_entries: int
    l1_norm: int
    max_absolute_entry: int
    energy: int


def residual_metrics(target: Tensor, actions: Sequence[RankOneAction], mode: str = "l1",
                     mismatch_weight: int = 1) -> ResidualMetrics:
    """Reconstruct a candidate exactly and return all energy components."""
    if mode not in ("l1", "nnz_l1"):
        raise ValueError("energy mode must be 'l1' or 'nnz_l1'")
    if mismatch_weight < 0:
        raise ValueError("mismatch_weight must be nonnegative")
    residual = subtract_tensors(target, sum_actions(actions, tensor_shape(target)))
    flat = tuple(value for plane in residual for row in plane for value in row)
    nonzero = sum(value != 0 for value in flat)
    l1 = sum(abs(value) for value in flat)
    maximum = max(map(abs, flat), default=0)
    energy = l1 if mode == "l1" else mismatch_weight * nonzero + l1
    return ResidualMetrics(nonzero, l1, maximum, energy)


def canonical_state(indices: Sequence[int]) -> tuple[int, ...]:
    """Canonicalize a complete decomposition because term order is irrelevant."""
    return tuple(sorted(indices))


@dataclass(frozen=True)
class TemperedSearchConfig:
    rank: int = 7
    temperatures: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
    replicas: int | None = None
    exchange_enabled: bool = True
    stop_on_exact: bool = True
    rounds: int = 100
    local_steps_per_round: int = 4
    exchange_interval: int = 1
    energy_mode: str = "nnz_l1"
    mismatch_weight: int = 4
    seed: int = 20261018
    allow_repeated_actions: bool = False
    repair_enabled: bool = False
    repair_interval: int = 10
    repair_maximum_pool_size: int = 256
    repair_maximum_pair_records: int = 32_896
    repair_maximum_seconds: float = 5.0
    wall_clock_seconds: float | None = None
    history_interval: int = 1
    retain_improvements_limit: int = 0

    def validate(self, pool_size: int) -> None:
        if self.rank <= 0 or pool_size <= 0:
            raise ValueError("rank and pool size must be positive")
        if not self.allow_repeated_actions and self.rank > pool_size:
            raise ValueError("pool is smaller than rank while repeats are disabled")
        if not self.temperatures or any(not math.isfinite(t) or t <= 0 for t in self.temperatures):
            raise ValueError("temperatures must be finite and positive")
        if self.replicas is not None and (self.replicas <= 0 or self.replicas != len(self.temperatures)):
            raise ValueError("replicas must equal the temperature count")
        if tuple(sorted(self.temperatures)) != self.temperatures:
            raise ValueError("temperatures must be in nondecreasing order")
        if min(self.rounds, self.local_steps_per_round, self.exchange_interval,
               self.repair_interval, self.history_interval) <= 0:
            raise ValueError("round and interval settings must be positive")
        if self.retain_improvements_limit < 0:
            raise ValueError("retain_improvements_limit must be nonnegative")
        if self.energy_mode not in ("l1", "nnz_l1") or self.mismatch_weight < 0:
            raise ValueError("invalid energy configuration")
        if self.repair_maximum_pool_size <= 0 or self.repair_maximum_pair_records <= 0:
            raise ValueError("repair limits must be positive")
        if not math.isfinite(self.repair_maximum_seconds) or self.repair_maximum_seconds <= 0:
            raise ValueError("repair_maximum_seconds must be finite and positive")
        if self.wall_clock_seconds is not None and (
            not math.isfinite(self.wall_clock_seconds) or self.wall_clock_seconds <= 0
        ):
            raise ValueError("wall_clock_seconds must be finite and positive")


@dataclass
class SearchCounters:
    local_proposals: int = 0
    local_residual_evaluations: int = 0
    local_accepts: int = 0
    improving_accepts: int = 0
    equal_energy_accepts: int = 0
    worsening_accepts: int = 0
    exchange_attempts: int = 0
    exchange_accepts: int = 0
    repair_attempts: int = 0
    repair_pair_lookups: int = 0
    repair_candidates_constructed: int = 0
    repair_successes: int = 0
    repair_budget_exceeded: int = 0


def metropolis_accept(current: int, proposed: int, temperature: float, rng: random.Random) -> bool:
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    delta = proposed - current
    return delta <= 0 or math.log(max(rng.random(), float.fromhex("0x0.0000000000001p-1022"))) < -delta / temperature


def exchange_accept(first_energy: int, second_energy: int, first_temperature: float,
                    second_temperature: float, rng: random.Random) -> bool:
    if any(not math.isfinite(t) or t <= 0 for t in (first_temperature, second_temperature)):
        raise ValueError("temperatures must be finite and positive")
    log_ratio = (1.0 / first_temperature - 1.0 / second_temperature) * (first_energy - second_energy)
    return log_ratio >= 0 or math.log(max(rng.random(), float.fromhex("0x0.0000000000001p-1022"))) < log_ratio


@dataclass(frozen=True)
class PairRepairTable:
    status: str
    records: int
    table: dict[str, tuple[tuple[Tensor, int, int], ...]]


def build_pair_repair_table(actions: Sequence[RankOneAction], *, allow_repeated_actions: bool,
                            maximum_pool_size: int, maximum_pair_records: int,
                            maximum_seconds: float) -> PairRepairTable:
    """Build a bounded exact pair-sum table using the existing tensor digest."""
    if len(actions) > maximum_pool_size:
        return PairRepairTable("BUDGET_EXCEEDED_POOL_SIZE", 0, {})
    pair_count = len(actions) * (len(actions) + (1 if allow_repeated_actions else -1)) // 2
    if pair_count > maximum_pair_records:
        return PairRepairTable("BUDGET_EXCEEDED_PAIR_RECORDS", 0, {})
    started = time.perf_counter()
    tensors = tuple(outer_product(action) for action in actions)
    mutable: dict[str, list[tuple[Tensor, int, int]]] = {}
    for left in range(len(actions)):
        for right in range(left if allow_repeated_actions else left + 1, len(actions)):
            if time.perf_counter() - started >= maximum_seconds:
                return PairRepairTable("BUDGET_EXCEEDED_TIME", sum(map(len, mutable.values())), {})
            value = add_tensors(tensors[left], tensors[right])
            mutable.setdefault(partial_sum_identity(value), []).append((value, left, right))
    return PairRepairTable("READY", pair_count, {key: tuple(value) for key, value in mutable.items()})


def exact_two_term_repair(target: Tensor, state: Sequence[int], positions: tuple[int, int],
                          actions: Sequence[RankOneAction], table: PairRepairTable,
                          allow_repeated_actions: bool) -> dict:
    """Freeze all but two terms and look up the exact missing tensor sum."""
    if table.status != "READY":
        return {"status": "BUDGET_EXCEEDED", "detail": table.status}
    if len(positions) != 2 or positions[0] == positions[1]:
        raise ValueError("repair positions must be distinct")
    frozen = tuple(value for index, value in enumerate(state) if index not in positions)
    required = subtract_tensors(target, sum_actions(tuple(actions[i] for i in frozen), tensor_shape(target)))
    found_domain_rejection = False
    for exact_sum, left, right in table.table.get(partial_sum_identity(required), ()):
        if exact_sum != required:
            continue
        candidate = canonical_state((*frozen, left, right))
        if not allow_repeated_actions and len(set(candidate)) != len(candidate):
            found_domain_rejection = True
            continue
        selected = tuple(actions[i] for i in candidate)
        if sum_actions(selected, tensor_shape(target)) != target:
            return {"status": "RECONSTRUCTION_FAILED", "state": candidate}
        return {"status": "EXACT_VERIFIED", "state": candidate, "exact_reconstruction": True}
    return {"status": "DOMAIN_REJECTED" if found_domain_rejection else "NO_COMPLEMENT"}


def pool_identity(actions: Sequence[RankOneAction]) -> str:
    return hashlib.sha256(json.dumps([action_identity(action) for action in actions], separators=(",", ":")).encode()).hexdigest()


def state_identity(state: Sequence[int], actions: Sequence[RankOneAction]) -> str:
    """Stable identity of the canonical action multiset, independent of pool order."""
    ids = sorted(action_identity(actions[index]) for index in state)
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def initial_states_for_seed(pool_size: int, config: TemperedSearchConfig) -> tuple[tuple[int, ...], ...]:
    """Draw paired initial states without using the local proposal RNG streams."""
    config.validate(pool_size)
    master = random.Random(config.seed)
    return tuple(canonical_state(master.choices(range(pool_size), k=config.rank)
                   if config.allow_repeated_actions else master.sample(range(pool_size), config.rank))
                 for _ in config.temperatures)


def audit_two_term_repairability(target: Tensor, state: Sequence[int], actions: Sequence[RankOneAction],
                                 table: PairRepairTable, allow_repeated_actions: bool,
                                 *, oracle_assisted: bool = False, maximum_details: int = 10) -> dict:
    """Audit every removable pair and verify all exact completions independently."""
    if maximum_details < 0:
        raise ValueError("maximum_details must be nonnegative")
    shape = tensor_shape(target)
    records = []
    raw = rejected = verified = constructed = lookups = cores = 0
    for positions in combinations(range(len(state)), 2):
        if table.status != "READY":
            break
        frozen = tuple(value for index, value in enumerate(state) if index not in positions)
        required = subtract_tensors(target, sum_actions(tuple(actions[i] for i in frozen), shape))
        lookups += 1
        core_verified = 0
        for exact_sum, left, right in table.table.get(partial_sum_identity(required), ()):
            if exact_sum != required:
                continue
            raw += 1
            candidate = canonical_state((*frozen, left, right))
            if not allow_repeated_actions and len(set(candidate)) != len(candidate):
                rejected += 1
                continue
            constructed += 1
            if sum_actions(tuple(actions[i] for i in candidate), shape) == target:
                verified += 1
                core_verified += 1
                if len(records) < maximum_details:
                    records.append({"removed_positions": positions, "completion_action_ids":
                                    [action_identity(actions[i]) for i in (left, right)],
                                    "completed_state_hash": state_identity(candidate, actions),
                                    "exact_reconstruction": True})
        cores += core_verified > 0
    return {"schema": "tensor_game_repairability_audit.v1",
            "candidate_hash": state_identity(state, actions), "term_pairs_audited": lookups,
            "total_term_pairs": len(state) * (len(state) - 1) // 2,
            "valid_completion_cores": cores, "raw_pair_complements": raw,
            "domain_rejections": rejected, "candidates_constructed": constructed,
            "independently_verified": verified, "repairable": cores > 0,
            "oracle_assisted": oracle_assisted, "lookup_status": table.status,
            "details": records}


def validate_pool_mode(pool_name: str, oracle_permitted: bool) -> None:
    """Prevent an oracle-only pool from entering a discovery-mode run."""
    if pool_name == "strassen_oracle" and not oracle_permitted:
        raise ValueError("the Strassen oracle pool is not permitted in discovery mode")


def parallel_tempering_search(target: Tensor, pool: Sequence[RankOneAction], config: TemperedSearchConfig,
                              initial_states: Sequence[Sequence[int]] | None = None,
                              pair_table: PairRepairTable | None = None) -> dict:
    """Run deterministic bounded parallel tempering over complete decompositions."""
    actions = tuple(canonicalize_action(action) for action in pool)
    if len(set(actions)) != len(actions):
        raise ValueError("action pool must contain unique canonical actions")
    config.validate(len(actions))
    master = random.Random(config.seed)
    rngs = tuple(random.Random(master.getrandbits(64)) for _ in config.temperatures)
    exchange_rng = random.Random(master.getrandbits(64))
    repair_rng = random.Random(master.getrandbits(64))
    states = list(initial_states_for_seed(len(actions), config) if initial_states is None
                  else (canonical_state(state) for state in initial_states))
    if len(states) != len(config.temperatures) or any(len(state) != config.rank or
            any(index < 0 or index >= len(actions) for index in state) or
            (not config.allow_repeated_actions and len(set(state)) != len(state)) for state in states):
        raise ValueError("initial states must match replicas, rank, pool, and domain")
    initial_hashes = [state_identity(state, actions) for state in states]
    metrics = [residual_metrics(target, tuple(actions[i] for i in state), config.energy_mode,
                                config.mismatch_weight) for state in states]
    best_index = min(range(len(states)), key=lambda i: (metrics[i].energy, states[i]))
    best_state, best_metrics = states[best_index], metrics[best_index]
    retained = ([{"state": best_state, "metrics": asdict(best_metrics), "round": 0,
                  "proposal_index": 0}] if config.retain_improvements_limit else [])
    started = time.perf_counter()
    time_to_best = 0.0
    best_round = best_proposal_index = 0
    counters = SearchCounters()
    visited = set(states)
    history = []
    repaired = False
    repair_table = (pair_table if pair_table is not None else build_pair_repair_table(actions, allow_repeated_actions=config.allow_repeated_actions,
                    maximum_pool_size=config.repair_maximum_pool_size,
                    maximum_pair_records=config.repair_maximum_pair_records,
                    maximum_seconds=config.repair_maximum_seconds)) if config.repair_enabled else None
    stop_reason = "maximum_rounds"
    rounds_completed = 0
    for round_index in range(1, config.rounds + 1):
        if config.wall_clock_seconds is not None and time.perf_counter() - started >= config.wall_clock_seconds:
            stop_reason = "wall_clock_budget"; break
        for replica, (temperature, rng) in enumerate(zip(config.temperatures, rngs)):
            for _ in range(config.local_steps_per_round):
                counters.local_proposals += 1
                position = rng.randrange(config.rank)
                frozen = states[replica][:position] + states[replica][position + 1:]
                available = tuple(index for index in range(len(actions))
                                  if config.allow_repeated_actions or index not in frozen)
                replacement = rng.choice(available)
                proposed = canonical_state((*states[replica][:position], *states[replica][position + 1:], replacement))
                if proposed == states[replica]:
                    continue
                proposed_metrics = residual_metrics(target, tuple(actions[i] for i in proposed),
                                                    config.energy_mode, config.mismatch_weight)
                counters.local_residual_evaluations += 1
                old_energy = metrics[replica].energy
                if metropolis_accept(old_energy, proposed_metrics.energy, temperature, rng):
                    states[replica], metrics[replica] = proposed, proposed_metrics
                    counters.local_accepts += 1
                    counters.improving_accepts += proposed_metrics.energy < old_energy
                    counters.equal_energy_accepts += proposed_metrics.energy == old_energy
                    counters.worsening_accepts += proposed_metrics.energy > old_energy
                    visited.add(proposed)
                    if (proposed_metrics.energy, proposed) < (best_metrics.energy, best_state):
                        strict_improvement = proposed_metrics.energy < best_metrics.energy
                        best_state, best_metrics = proposed, proposed_metrics
                        if strict_improvement and len(retained) < config.retain_improvements_limit:
                            retained.append({"state": proposed, "metrics": asdict(proposed_metrics),
                                             "round": round_index, "proposal_index": counters.local_proposals})
                        time_to_best = time.perf_counter() - started
                        best_round, best_proposal_index = round_index, counters.local_proposals
        if config.exchange_enabled and round_index % config.exchange_interval == 0:
            parity = (round_index // config.exchange_interval - 1) % 2
            for replica in range(parity, len(states) - 1, 2):
                counters.exchange_attempts += 1
                if exchange_accept(metrics[replica].energy, metrics[replica + 1].energy,
                                   config.temperatures[replica], config.temperatures[replica + 1], exchange_rng):
                    states[replica], states[replica + 1] = states[replica + 1], states[replica]
                    metrics[replica], metrics[replica + 1] = metrics[replica + 1], metrics[replica]
                    counters.exchange_accepts += 1
        if config.repair_enabled and round_index % config.repair_interval == 0:
            counters.repair_attempts += 1
            positions = tuple(sorted(repair_rng.sample(range(config.rank), 2)))
            counters.repair_pair_lookups += int(repair_table.status == "READY")
            repair = exact_two_term_repair(target, best_state, positions, actions, repair_table,
                                           config.allow_repeated_actions)  # type: ignore[arg-type]
            counters.repair_candidates_constructed += repair["status"] in ("EXACT_VERIFIED", "RECONSTRUCTION_FAILED")
            counters.repair_budget_exceeded += repair["status"] == "BUDGET_EXCEEDED"
            if repair["status"] == "EXACT_VERIFIED":
                strict_improvement = best_metrics.energy > 0
                repaired = repaired or repair["state"] != best_state
                if repair["state"] != best_state:
                    time_to_best = time.perf_counter() - started
                    best_round, best_proposal_index = round_index, counters.local_proposals
                best_state = repair["state"]
                visited.add(best_state)
                best_metrics = residual_metrics(target, tuple(actions[i] for i in best_state),
                                                config.energy_mode, config.mismatch_weight)
                if strict_improvement and len(retained) < config.retain_improvements_limit:
                    retained.append({"state": best_state, "metrics": asdict(best_metrics),
                                     "round": round_index, "proposal_index": counters.local_proposals})
                counters.repair_successes += 1
        rounds_completed = round_index
        if round_index % config.history_interval == 0 or best_metrics.energy == 0:
            history.append({"round": round_index, "best": asdict(best_metrics)})
        if best_metrics.energy == 0 and config.stop_on_exact:
            stop_reason = "exact_candidate"; break
    selected = tuple(actions[i] for i in best_state)
    independently_exact = sum_actions(selected, tensor_shape(target)) == target
    status = "EXACT" if best_metrics.energy == 0 and independently_exact else "BUDGET_EXHAUSTED"
    return {
        "schema": "tensor_game_tempered_search_result.v1", "status": status,
        "stop_reason": stop_reason, "rounds_completed": rounds_completed,
        "best_state": best_state, "best_action_ids": [action_identity(action) for action in selected],
        "best_metrics": asdict(best_metrics), "exact_reconstruction": independently_exact,
        "repair_modified_best": repaired, "counters": asdict(counters),
        "initial_state_hashes": initial_hashes,
        "best_candidate_hash": state_identity(best_state, actions),
        "time_to_best_seconds": time_to_best, "best_round": best_round,
        "best_proposal_index": best_proposal_index,
        "wall_clock_seconds": time.perf_counter() - started,
        "repair_table_status": repair_table.status if repair_table else "DISABLED",
        "unique_canonical_states_visited": len(visited), "history": history,
        "pool_hash": pool_identity(actions), "config": asdict(config),
        "retained_improvements": retained,
    }
