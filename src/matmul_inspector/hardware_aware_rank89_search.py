"""Bounded exact finite-pool rank-8/9 search with optional OR-Tools CP-SAT."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
import math
from pathlib import Path
import time
from typing import Iterable, Sequence

from .hardware_aware_tensor_candidates import descriptor_vector
from .tensor_game import (
    RankOneAction, matrix_multiplication_tensor, outer_product,
    standard_2x2_decomposition, strassen_2x2_decomposition,
)
from .tensor_game_bounded_residual import VECTORS
from .tensor_game_candidate_dataset import (
    _digest, _record, actions_payload, canonical_actions, validate_decomposition,
)
from .tensor_game_experience import action_identity
from .tensor_game_hardware_cost import decomposition_cost
from .tensor_game_meet_in_middle import canonical_action_pool
from .tensor_game_residual_lookahead import exact_matrix_rank
from .tensor_game_search import canonicalize_action

OBJECTIVES = ("FEASIBILITY", "MIN_DISTINCT_INPUT_FORMS", "MIN_DISTINCT_U",
              "MIN_DISTINCT_V", "MAX_TOTAL_REUSE")
DEFAULT_CATALOG = Path("artifacts/tensor-game/hardware-aware-rank8-v1/candidates.jsonl")
DEFAULT_FROZEN = Path("artifacts/tensor-game/algebraic-pools-v1/pools.json")


@dataclass(frozen=True)
class SolveLimits:
    max_pool_actions: int = 512
    max_variables: int = 2048
    max_auxiliary_variables: int = 512
    max_constraints: int = 1024
    timeout_seconds: float = 5.0
    memory_limit_mb: int = 1024
    max_models: int = 2
    max_accepted: int = 2

    def __post_init__(self) -> None:
        values = (self.max_pool_actions, self.max_variables, self.max_auxiliary_variables,
                  self.max_constraints, self.memory_limit_mb, self.max_models, self.max_accepted)
        if min(values) <= 0 or not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("all resource limits must be positive and finite")


@dataclass(frozen=True)
class StructuredBounds:
    max_u_support: int = 1
    max_v_support: int = 1
    max_w_support: int = 2
    max_abs_coefficient: int = 1

    def __post_init__(self) -> None:
        if min(self.max_u_support, self.max_v_support, self.max_w_support,
               self.max_abs_coefficient) <= 0:
            raise ValueError("structural bounds must be positive")
        if self.max_abs_coefficient != 1:
            raise ValueError("the existing canonical action domain is ternary; coefficient bound must be one")


def _action_key(action: RankOneAction) -> tuple[int, ...]:
    return action.u + action.v + action.w


def _action_payload(action: RankOneAction) -> dict:
    return {"u": list(action.u), "v": list(action.v), "w": list(action.w)}


def _row_action(value: dict) -> RankOneAction:
    return canonicalize_action(RankOneAction(value["u"], value["v"], value["w"]))


def synthetic_rank9_control() -> tuple[RankOneAction, ...]:
    """Split one unit output into two distinct legal ternary output factors."""
    conventional = list(standard_2x2_decomposition())
    first = conventional.pop(0)
    occupied = next(index for index, value in enumerate(first.w) if value)
    extra = (occupied + 1) % 4
    left = list(first.w)
    right = [0] * 4
    left[extra] = 1
    right[extra] = -1
    split = (RankOneAction(first.u, first.v, left), RankOneAction(first.u, first.v, right))
    actions = tuple(canonicalize_action(action) for action in (*conventional, *split))
    if len(set(actions)) != 9:
        raise ValueError("synthetic split repeats a canonical action")
    canonical_action_pool(actions)  # Enforces the existing nonzero ternary domain.
    validate_decomposition(actions)
    return canonical_actions(actions)


def reuse_descriptors(actions: Sequence[RankOneAction]) -> dict:
    """Count equality of the stored canonical input factors, not device residency."""
    canonical = canonical_actions(actions)
    u_counts = Counter(action.u for action in canonical)
    v_counts = Counter(action.v for action in canonical)

    def groups(counts: Counter) -> list[dict]:
        return [{"form": list(form), "products": count} for form, count in sorted(counts.items())]

    return {
        "distinct_u_forms": len(u_counts), "distinct_v_forms": len(v_counts),
        "distinct_input_forms": len(u_counts) + len(v_counts),
        "u_reuse_count": len(canonical) - len(u_counts),
        "v_reuse_count": len(canonical) - len(v_counts),
        "total_reuse_count": 2 * len(canonical) - len(u_counts) - len(v_counts),
        "max_u_multiplicity": max(u_counts.values()),
        "max_v_multiplicity": max(v_counts.values()),
        "reused_u_forms": sum(count >= 2 for count in u_counts.values()),
        "reused_v_forms": sum(count >= 2 for count in v_counts.values()),
        "product_groups_by_u": groups(u_counts), "product_groups_by_v": groups(v_counts),
        "input_form_support": {"u": sorted(Counter(sum(x != 0 for x in form) for form in u_counts).items()),
                               "v": sorted(Counter(sum(x != 0 for x in form) for form in v_counts).items())},
        "output_factor_support": sorted(Counter(sum(x != 0 for x in action.w) for action in canonical).items()),
    }


def distinct_input_lower_bounds() -> dict[str, int]:
    """Exact mode-flattening ranks lower-bound distinct input factors."""
    tensor = matrix_multiplication_tensor(2, 2, 2)
    u_matrix = [tuple(tensor[i][j][k] for j in range(4) for k in range(4)) for i in range(4)]
    v_matrix = [tuple(tensor[i][j][k] for i in range(4) for k in range(4)) for j in range(4)]
    u_bound, v_bound = exact_matrix_rank(u_matrix), exact_matrix_rank(v_matrix)
    return {"distinct_u_forms": u_bound, "distinct_v_forms": v_bound,
            "distinct_input_forms": u_bound + v_bound}


def _pool(rows: Iterable[tuple[RankOneAction, dict]], maximum: int) -> tuple[tuple[RankOneAction, ...], dict[str, list[dict]]]:
    by_action: dict[RankOneAction, list[dict]] = {}
    for action, provenance in rows:
        action = canonicalize_action(action)
        if action not in by_action and len(by_action) >= maximum:
            break
        by_action.setdefault(action, []).append(provenance)
    actions = canonical_action_pool(by_action)
    return actions, {action_identity(action): by_action[action] for action in actions}


def control_pool(rank: int, maximum: int = 32, distractors: int = 8) -> tuple[tuple[RankOneAction, ...], dict]:
    required = standard_2x2_decomposition() if rank == 8 else synthetic_rank9_control()
    if rank not in (8, 9) or maximum < rank or distractors < 0:
        raise ValueError("invalid control pool configuration")
    source = [(action, {"source": "known_positive_control", "rank": rank}) for action in required]
    for action, _ in itertools.islice(structured_domain(StructuredBounds()), distractors):
        source.append((action, {"source": "control_distractor", "restriction": "unit inputs, output support <=2"}))
    return _pool(source, maximum)


def structured_domain(bounds: StructuredBounds) -> Iterable[tuple[RankOneAction, dict]]:
    """Stream the bounded ternary domain; no known decomposition is injected."""
    vectors = tuple(vector for vector in VECTORS if max(map(abs, vector)) <= bounds.max_abs_coefficient)
    by_support = lambda maximum: sorted((vector for vector in vectors if sum(x != 0 for x in vector) <= maximum),
                                         key=lambda vector: (sum(x != 0 for x in vector), vector))
    u_vectors, v_vectors, w_vectors = (by_support(bounds.max_u_support),
                                       by_support(bounds.max_v_support), by_support(bounds.max_w_support))
    for u, v, w in itertools.product(u_vectors, v_vectors, w_vectors):
        action = RankOneAction(u, v, w)
        if canonicalize_action(action) == action:
            yield action, {"source": "structured_domain", "restrictions": asdict(bounds)}


def structured_pool(bounds: StructuredBounds, maximum: int) -> tuple[tuple[RankOneAction, ...], dict]:
    if maximum <= 0:
        raise ValueError("maximum pool size must be positive")
    return _pool(structured_domain(bounds), maximum)


def frozen_existing_pool(path: Path, identifiers: Sequence[str], maximum: int) -> tuple[tuple[RankOneAction, ...], dict]:
    """Load prior non-oracle action pools without reference-action injection."""
    data = json.loads(path.read_text())
    if not identifiers:
        raise ValueError("at least one frozen pool identifier is required")
    rows = []
    for identifier in identifiers:
        if "oracle" in identifier.lower() or identifier not in data:
            raise ValueError(f"unknown or oracle-only frozen pool: {identifier}")
        for index, payload in enumerate(data[identifier]["actions"]):
            rows.append((_row_action(payload), {"source": "frozen_existing", "artifact": str(path),
                                                "pool_identifier": identifier, "source_index": index}))
    return _pool(rows, maximum)


def pool_manifest(actions: Sequence[RankOneAction], provenance: dict) -> dict:
    rows = [{"action_id": action_identity(action), "action": _action_payload(action),
             "provenance": provenance[action_identity(action)]} for action in actions]
    return {"pool_hash": _digest(rows), "action_count": len(rows), "actions": rows,
            "repeated_action_policy": "at most once per canonical rank-one action"}


def model_size(actions: Sequence[RankOneAction], objective: str = "FEASIBILITY",
               min_u_reuse: int = 0, min_v_reuse: int = 0, max_models: int = 1) -> dict:
    forms_u = {action.u for action in actions}
    forms_v = {action.v for action in actions}
    use_aux = objective != "FEASIBILITY" or min_u_reuse or min_v_reuse
    auxiliary = len(forms_u) + len(forms_v) if use_aux else 0
    return {"action_variables": len(actions), "auxiliary_variables": auxiliary,
            "variable_count": len(actions) + auxiliary,
            "constraint_count_upper_bound": 65 + auxiliary + int(bool(min_u_reuse)) + int(bool(min_v_reuse)) + max_models,
            "tensor_equality_constraints": 64, "rank_constraints": 1,
            "distinct_u_forms_available": len(forms_u), "distinct_v_forms_available": len(forms_v)}


def classify_candidate(canonical_hash: str, rank: int, catalog_hashes: set[str],
                       conventional_hash: str, synthetic_hash: str) -> str:
    if rank == 8 and canonical_hash == conventional_hash:
        return "CONVENTIONAL_ANCHOR"
    if rank == 9 and canonical_hash == synthetic_hash:
        return "SYNTHETIC_CONTROL"
    if canonical_hash in catalog_hashes:
        return "EXISTING_PROJECT_CANDIDATE"
    return "NEW_TO_PROJECT_CATALOG"


def catalog_ids(path: Path = DEFAULT_CATALOG) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(f"retained HA-Tensor-1 catalog is missing: {path}")
    return {json.loads(line)["canonical_id"] for line in path.read_text().splitlines()}


def _canonical_hash(actions: Sequence[RankOneAction]) -> str:
    return _digest(actions_payload(canonical_actions(actions)))


def rank9_split_parent(actions: Sequence[RankOneAction], catalog_hashes: set[str]) -> str | None:
    """Recognize an exact two-output-term split of a cataloged rank-8 form."""
    if len(actions) != 9:
        return None
    conventional = _canonical_hash(standard_2x2_decomposition())
    for left, right in itertools.combinations(range(9), 2):
        first, second = actions[left], actions[right]
        if first.u != second.u or first.v != second.v:
            continue
        merged = RankOneAction(first.u, first.v, tuple(a + b for a, b in zip(first.w, second.w)))
        if not any(merged.w):
            continue
        parent = [action for index, action in enumerate(actions) if index not in (left, right)]
        parent.append(merged)
        key = _canonical_hash(parent)
        if key == conventional or key in catalog_hashes:
            validate_decomposition(parent)
            return key
    return None


def _solution(actions: Sequence[RankOneAction], selected: Sequence[int], source: dict,
              catalog_hashes: set[str]) -> dict:
    chosen = canonical_actions(tuple(actions[index] for index in selected))
    validate_decomposition(chosen)  # Independent exact 64-entry check, after CP-SAT.
    row = _record("direct-finite-pool", "direct-finite-pool", chosen, [])
    key = row["canonical_id"]
    conventional_hash = _canonical_hash(standard_2x2_decomposition())
    synthetic_hash = _canonical_hash(synthetic_rank9_control())
    return {"canonical_decomposition_hash": key, "rank": len(chosen),
            "exact_reconstruction_status": "PASS", "action_ids": sorted(action_identity(action) for action in chosen),
            "actions": row["actions"], "structural_signature": row["structural_signature"],
            "structural_signature_id": row["structural_signature_id"],
            "static_cost": row["static_cost"], "descriptor_vector": descriptor_vector(row),
            "reuse_descriptor_vector": reuse_descriptors(chosen),
            "known_rank8_split_parent_hash": rank9_split_parent(chosen, catalog_hashes),
            "project_catalog_classification": classify_candidate(key, len(chosen), catalog_hashes,
                                                                  conventional_hash, synthetic_hash),
            "source": source}


def solve_exact_decomposition(actions: Sequence[RankOneAction], rank: int, *,
                              objective: str = "FEASIBILITY", limits: SolveLimits = SolveLimits(),
                              seed: int = 20261018, workers: int = 1,
                              min_u_reuse: int = 0, min_v_reuse: int = 0,
                              catalog_hashes: set[str] | None = None,
                              source: dict | None = None) -> dict:
    """Solve exact joint Boolean selection; never infer UNSAT from a timeout."""
    if rank not in (8, 9) or objective not in OBJECTIVES or workers <= 0:
        raise ValueError("unsupported rank, objective, or worker count")
    if min(min_u_reuse, min_v_reuse) < 0:
        raise ValueError("reuse minima must be nonnegative")
    pool = canonical_action_pool(actions)
    estimate = model_size(pool, objective, min_u_reuse, min_v_reuse, limits.max_models)
    summary = {"rank": rank, "pool_size": len(pool), "objective": objective,
               "model_size": estimate, "seed": seed, "worker_count": workers,
               "timeout_seconds": limits.timeout_seconds, "memory_limit_mb": limits.memory_limit_mb,
               "max_models": limits.max_models, "max_accepted": limits.max_accepted,
               "min_u_reuse": min_u_reuse, "min_v_reuse": min_v_reuse,
               "solver": "OR-Tools CP-SAT", "solver_status": "NOT_STARTED",
               "status": "UNKNOWN", "solutions": [], "solve_wall_seconds": 0.0}
    if (len(pool) > limits.max_pool_actions or estimate["variable_count"] > limits.max_variables
            or estimate["auxiliary_variables"] > limits.max_auxiliary_variables
            or estimate["constraint_count_upper_bound"] > limits.max_constraints):
        summary.update(status="RESOURCE_LIMIT", solver_status="NOT_CONSTRUCTED")
        return summary
    try:
        import ortools
        from ortools.sat.python import cp_model
    except ImportError:
        summary.update(status="ERROR", solver_status="DEPENDENCY_MISSING",
                       error="Install the optional constraint-search extra (OR-Tools).")
        return summary
    summary["solver_version"] = ortools.__version__
    model = cp_model.CpModel()
    selected = [model.NewBoolVar(f"action_{index}") for index in range(len(pool))]
    model.Add(sum(selected) == rank)
    target = matrix_multiplication_tensor(2, 2, 2)
    tensors = [outer_product(action) for action in pool]
    for i, j, k in itertools.product(range(4), repeat=3):
        model.Add(sum(tensor[i][j][k] * selected[index] for index, tensor in enumerate(tensors)
                      if tensor[i][j][k]) == target[i][j][k])
    y_u: dict[tuple[int, ...], object] = {}
    y_v: dict[tuple[int, ...], object] = {}
    if estimate["auxiliary_variables"]:
        for label, factor, storage in (("u", "u", y_u), ("v", "v", y_v)):
            for form in sorted({getattr(action, factor) for action in pool}):
                y = model.NewBoolVar(f"{label}_{len(storage)}")
                model.AddMaxEquality(y, [selected[index] for index, action in enumerate(pool)
                                         if getattr(action, factor) == form])
                storage[form] = y
        if min_u_reuse:
            model.Add(rank - sum(y_u.values()) >= min_u_reuse)
        if min_v_reuse:
            model.Add(rank - sum(y_v.values()) >= min_v_reuse)
    objective_expr = None
    if objective == "MIN_DISTINCT_INPUT_FORMS":
        objective_expr = sum(y_u.values()) + sum(y_v.values())
    elif objective == "MAX_TOTAL_REUSE":
        objective_expr = 2 * rank - sum(y_u.values()) - sum(y_v.values())
    elif objective == "MIN_DISTINCT_U":
        objective_expr = sum(y_u.values())
    elif objective == "MIN_DISTINCT_V":
        objective_expr = sum(y_v.values())
    if objective_expr is not None:
        if objective == "MAX_TOTAL_REUSE":
            model.Maximize(objective_expr)
        else:
            model.Minimize(objective_expr)
    summary["actual_model_size_initial"] = {"variables": len(model.Proto().variables),
                                             "constraints": len(model.Proto().constraints),
                                             "auxiliary_variables": len(y_u) + len(y_v)}
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = workers
    solver.parameters.random_seed = seed
    solver.parameters.max_memory_in_mb = limits.memory_limit_mb
    solver.parameters.log_search_progress = False
    begun = time.monotonic()
    seen: set[str] = set()
    catalog = catalog_hashes if catalog_hashes is not None else set()
    optimum = None
    statuses = []
    stop_reason = "model_limit"
    for _ in range(limits.max_models):
        remaining = limits.timeout_seconds - (time.monotonic() - begun)
        if remaining <= 0:
            stop_reason = "time_limit"
            if not summary["solutions"]:
                summary["status"] = "TIME_LIMIT"
            break
        solver.parameters.max_time_in_seconds = remaining
        try:
            code = solver.Solve(model)
        except Exception as exc:
            summary.update(status="ERROR", error=f"{type(exc).__name__}: {exc}")
            stop_reason = "solver_error"
            break
        status_name = solver.StatusName(code)
        statuses.append(status_name)
        summary.update(solver_status=status_name, solve_wall_seconds=time.monotonic() - begun,
                       conflicts=solver.NumConflicts(), branches=solver.NumBranches())
        if code in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            chosen_indices = tuple(index for index, variable in enumerate(selected) if solver.Value(variable))
            try:
                solution = _solution(pool, chosen_indices, source or {}, catalog)
            except ValueError as exc:
                summary.update(status="ERROR", error=f"independent exact reconstruction failed: {exc}")
                stop_reason = "invalid_model"
                break
            key = solution["canonical_decomposition_hash"]
            if key not in seen:
                seen.add(key)
                solution["solver_objective_value"] = int(round(solver.ObjectiveValue())) if objective_expr is not None else None
                solution["objective_proven_optimal"] = code == cp_model.OPTIMAL
                summary["solutions"].append(solution)
            summary["status"] = "EXACT"
            if objective_expr is not None:
                if code != cp_model.OPTIMAL:
                    stop_reason = "objective_not_proven_optimal"
                    break  # A feasible model need not have the true optimum.
                if optimum is None:
                    optimum = int(round(solver.ObjectiveValue()))
                    model.Add(objective_expr == optimum)
                    model.ClearObjective()
            if len(summary["solutions"]) >= limits.max_accepted:
                stop_reason = "accepted_limit"
                break
            chosen_set = set(chosen_indices)
            model.Add(sum(selected[i] if i in chosen_set else 1 - selected[i]
                          for i in range(len(pool))) <= len(pool) - 1)
        elif code == cp_model.INFEASIBLE:
            stop_reason = "exhausted_finite_pool"
            if not summary["solutions"]:
                summary["status"] = "INFEASIBLE"
            break
        elif code == cp_model.UNKNOWN:
            timed_out = time.monotonic() - begun >= limits.timeout_seconds * 0.95
            stop_reason = "time_limit" if timed_out else "unknown"
            if not summary["solutions"]:
                summary["status"] = "TIME_LIMIT" if timed_out else "UNKNOWN"
            break
        else:
            summary["status"] = "ERROR"
            stop_reason = "solver_error"
            break
    summary["solver_status_history"] = statuses
    summary["enumeration_stop_reason"] = stop_reason
    summary["actual_model_size_final"] = {"variables": len(model.Proto().variables),
                                           "constraints": len(model.Proto().constraints)}
    summary["solve_wall_seconds"] = time.monotonic() - begun
    summary["objective_value"] = optimum
    summary["unique_solution_count"] = len(summary["solutions"])
    return summary
