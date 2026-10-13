"""Bounded sampled Monte Carlo tree search for exact TensorGame states."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
import time
from typing import Protocol, Sequence

from .tensor_game import (
    RankOneAction,
    Tensor,
    outer_product,
    subtract_action,
    sum_actions,
    tensor_shape,
    validate_action,
    zero_tensor,
)

StateKey = tuple[Tensor, int]


@dataclass
class ProposalMetrics:
    attempted: int = 0
    valid: int = 0
    accepted: int = 0
    duplicates: int = 0
    generation_seconds: float = 0.0


class InstrumentedPolicy:
    """Shared counters for proposal cost and acceptance diagnostics."""

    metrics: ProposalMetrics

    def reset_metrics(self) -> None:
        self.metrics = ProposalMetrics()


def canonicalize_action(action: RankOneAction) -> RankOneAction:
    """Choose one representation among the four sign-equivalent factorizations."""
    u, v, w = action.u, action.v, action.w
    forms = (
        (u, v, w),
        (tuple(-x for x in u), tuple(-x for x in v), w),
        (tuple(-x for x in u), v, tuple(-x for x in w)),
        (u, tuple(-x for x in v), tuple(-x for x in w)),
    )
    return RankOneAction(*min(forms))


@dataclass(frozen=True)
class ProposedAction:
    action: RankOneAction
    prior: float
    provenance: tuple[str, ...] = ()


class ActionPolicy(Protocol):
    name: str

    def propose_actions(
        self,
        residual: Tensor,
        remaining_moves: int,
        rng: random.Random,
        max_actions: int,
    ) -> tuple[ProposedAction, ...]: ...


@dataclass
class ValueMetrics:
    calls: int = 0
    clipped: int = 0
    inference_seconds: float = 0.0
    predictions: list[float] = field(default_factory=list)


class LeafEvaluator(Protocol):
    name: str
    metrics: ValueMetrics

    def reset_metrics(self) -> None: ...

    def evaluate(
        self,
        residual: Tensor,
        remaining_moves: int,
        max_moves: int,
        target_nonzeros: int,
    ) -> float: ...


@dataclass
class HeuristicValueEvaluator:
    """Original nonzero-count leaf estimate, exposed as a configurable adapter."""

    name: str = field(default="nonzero_heuristic", init=False)
    metrics: ValueMetrics = field(default_factory=ValueMetrics, init=False, repr=False)

    def reset_metrics(self) -> None:
        self.metrics = ValueMetrics()

    def evaluate(self, residual, remaining_moves, max_moves, target_nonzeros):
        nonzeros = sum(value != 0 for plane in residual for row in plane for value in row)
        value = -nonzeros / max(1, target_nonzeros)
        self.metrics.calls += 1
        self.metrics.predictions.append(value)
        return value


def _normalize(proposals: Sequence[tuple[RankOneAction, float]]) -> tuple[ProposedAction, ...]:
    unique: dict[RankOneAction, float] = {}
    for action, prior in proposals:
        canonical = canonicalize_action(action)
        unique[canonical] = unique.get(canonical, 0.0) + prior
    total = sum(unique.values())
    if not unique or not math.isfinite(total) or total <= 0:
        return ()
    return tuple(ProposedAction(action, prior / total) for action, prior in unique.items())


@dataclass
class UniformSampledPolicy(InstrumentedPolicy):
    """Sample bounded factor vectors without enumerating their Cartesian product."""

    shape: tuple[int, int, int]
    coefficients: tuple[int, ...] = (-1, 0, 1)
    attempt_multiplier: int = 50
    name: str = field(default="uniform_sampled", init=False)
    metrics: ProposalMetrics = field(default_factory=ProposalMetrics, init=False, repr=False)

    def __post_init__(self) -> None:
        if len(self.shape) != 3 or any(value <= 0 for value in self.shape):
            raise ValueError("shape must contain three positive dimensions")
        if not self.coefficients or any(not isinstance(value, int) or isinstance(value, bool) for value in self.coefficients):
            raise ValueError("coefficients must be a nonempty tuple of integers")
        if self.attempt_multiplier <= 0:
            raise ValueError("attempt_multiplier must be positive")

    def propose_actions(self, residual, remaining_moves, rng, max_actions):
        started = time.perf_counter()
        if tensor_shape(residual) != self.shape or remaining_moves <= 0 or max_actions <= 0:
            return ()
        actions: dict[RankOneAction, float] = {}
        attempts = 0
        while len(actions) < max_actions and attempts < max_actions * self.attempt_multiplier:
            attempts += 1
            self.metrics.attempted += 1
            vectors = tuple(
                tuple(rng.choice(self.coefficients) for _ in range(length))
                for length in self.shape
            )
            action = RankOneAction(*vectors)
            try:
                validate_action(action, self.shape, frozenset(self.coefficients))
            except ValueError:
                continue
            self.metrics.valid += 1
            canonical = canonicalize_action(action)
            if canonical in actions:
                self.metrics.duplicates += 1
            actions.setdefault(canonical, 1.0)
        result = _normalize(tuple(actions.items()))
        self.metrics.accepted += len(result)
        self.metrics.generation_seconds += time.perf_counter() - started
        return result


def contract_residual(residual: Tensor, u: Sequence[int], v: Sequence[int]) -> tuple[int, ...]:
    """Compute d_k = sum_ij residual[i,j,k] u_i v_j exactly."""
    x, y, z = tensor_shape(residual)
    if len(u) != x or len(v) != y:
        raise ValueError("factor lengths must match the first two residual dimensions")
    if any(not isinstance(value, int) or isinstance(value, bool) for value in (*u, *v)):
        raise ValueError("factor coefficients must be integers")
    return tuple(
        sum(residual[i][j][k] * u[i] * v[j] for i in range(x) for j in range(y))
        for k in range(z)
    )


def quantize_ratio(numerator: int, denominator: int, coefficients: Sequence[int]) -> int:
    """Choose an allowed integer nearest to numerator/denominator without floats."""
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    if not coefficients:
        raise ValueError("coefficients must not be empty")
    return min(coefficients, key=lambda value: (abs(numerator - denominator * value), abs(value), value))


@dataclass
class ResidualInformedPolicy(InstrumentedPolicy):
    """Propose integer actions from residual contractions and sparse slices."""

    shape: tuple[int, int, int]
    coefficients: tuple[int, ...] = (-1, 0, 1)
    sparse_probability: float = 0.25
    attempt_multiplier: int = 50
    name: str = field(default="residual_informed", init=False)
    metrics: ProposalMetrics = field(default_factory=ProposalMetrics, init=False, repr=False)

    def __post_init__(self) -> None:
        if len(self.shape) != 3 or any(value <= 0 for value in self.shape):
            raise ValueError("shape must contain three positive dimensions")
        if not self.coefficients or any(not isinstance(value, int) or isinstance(value, bool) for value in self.coefficients):
            raise ValueError("coefficients must be a nonempty tuple of integers")
        if not 0 <= self.sparse_probability <= 1:
            raise ValueError("sparse_probability must be in [0, 1]")
        if self.attempt_multiplier <= 0:
            raise ValueError("attempt_multiplier must be positive")

    def _candidate(self, residual: Tensor, rng: random.Random) -> RankOneAction | None:
        x, y, _ = self.shape
        nonzeros = [
            (i, j)
            for i in range(x)
            for j in range(y)
            if any(residual[i][j][k] != 0 for k in range(self.shape[2]))
        ]
        if nonzeros and rng.random() < self.sparse_probability:
            i, j = rng.choice(nonzeros)
            sign = 1 if residual[i][j][next(k for k in range(self.shape[2]) if residual[i][j][k])] > 0 else -1
            u = tuple(sign if index == i else 0 for index in range(x))
            v = tuple(1 if index == j else 0 for index in range(y))
        else:
            u = tuple(rng.choice(self.coefficients) for _ in range(x))
            v = tuple(rng.choice(self.coefficients) for _ in range(y))
        q = sum(value * value for value in u) * sum(value * value for value in v)
        if q == 0:
            return None
        contraction = contract_residual(residual, u, v)
        w = tuple(quantize_ratio(value, q, self.coefficients) for value in contraction)
        return RankOneAction(u, v, w)

    def propose_actions(self, residual, remaining_moves, rng, max_actions):
        started = time.perf_counter()
        if tensor_shape(residual) != self.shape or remaining_moves <= 0 or max_actions <= 0:
            return ()
        actions: dict[RankOneAction, float] = {}
        attempts = 0
        while len(actions) < max_actions and attempts < max_actions * self.attempt_multiplier:
            attempts += 1
            self.metrics.attempted += 1
            action = self._candidate(residual, rng)
            if action is None:
                continue
            try:
                validate_action(action, self.shape, frozenset(self.coefficients))
            except ValueError:
                continue
            self.metrics.valid += 1
            canonical = canonicalize_action(action)
            if canonical in actions:
                self.metrics.duplicates += 1
            actions.setdefault(canonical, 1.0)
        result = _normalize(tuple(actions.items()))
        self.metrics.accepted += len(result)
        self.metrics.generation_seconds += time.perf_counter() - started
        return result


@dataclass
class HybridSampledPolicy(InstrumentedPolicy):
    """Mix bounded uniform and residual-informed proposal batches."""

    shape: tuple[int, int, int]
    residual_probability: float = 0.5
    coefficients: tuple[int, ...] = (-1, 0, 1)
    name: str = field(default="hybrid", init=False)
    metrics: ProposalMetrics = field(default_factory=ProposalMetrics, init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0 <= self.residual_probability <= 1:
            raise ValueError("residual_probability must be in [0, 1]")
        self.uniform = UniformSampledPolicy(self.shape, self.coefficients)
        self.informed = ResidualInformedPolicy(self.shape, self.coefficients)

    def reset_metrics(self) -> None:
        super().reset_metrics()
        self.uniform.reset_metrics()
        self.informed.reset_metrics()

    def propose_actions(self, residual, remaining_moves, rng, max_actions):
        started = time.perf_counter()
        informed_count = sum(rng.random() < self.residual_probability for _ in range(max_actions))
        uniform_count = max_actions - informed_count
        before_uniform = (self.uniform.metrics.attempted, self.uniform.metrics.valid)
        before_informed = (self.informed.metrics.attempted, self.informed.metrics.valid)
        uniform = self.uniform.propose_actions(residual, remaining_moves, rng, uniform_count)
        informed = self.informed.propose_actions(residual, remaining_moves, rng, informed_count)
        proposals = [
            (item.action, (1 - self.residual_probability) * item.prior) for item in uniform
        ] + [(item.action, self.residual_probability * item.prior) for item in informed]
        result = _normalize(proposals)
        self.metrics.attempted += (
            self.uniform.metrics.attempted - before_uniform[0]
            + self.informed.metrics.attempted - before_informed[0]
        )
        self.metrics.valid += (
            self.uniform.metrics.valid - before_uniform[1]
            + self.informed.metrics.valid - before_informed[1]
        )
        child_accepted = len(uniform) + len(informed)
        self.metrics.duplicates += child_accepted - len(result)
        self.metrics.accepted += len(result)
        self.metrics.generation_seconds += time.perf_counter() - started
        return result


class ReferenceGuidedPolicy:
    """Exact known-decomposition fixture with optional unrelated distractors."""

    name = "reference_guided"

    def __init__(
        self,
        target: Tensor,
        reference: Sequence[RankOneAction],
        *,
        reference_prior: float = 0.9,
        distractor_policy: ActionPolicy | None = None,
        distractor_count: int = 0,
        randomized_order: bool = True,
    ) -> None:
        self.target = target
        self.shape = tensor_shape(target)
        self.reference = tuple(canonicalize_action(action) for action in reference)
        if not self.reference or sum_actions(self.reference, self.shape) != target:
            raise ValueError("reference actions must reconstruct the target exactly")
        if not 0 < reference_prior <= 1:
            raise ValueError("reference_prior must be in (0, 1]")
        if distractor_count < 0:
            raise ValueError("distractor_count must be nonnegative")
        self.reference_prior = reference_prior
        self.distractor_policy = distractor_policy
        self.distractor_count = distractor_count
        self.randomized_order = randomized_order
        self._remaining_by_residual = self._build_subset_index()

    def _build_subset_index(self) -> dict[Tensor, tuple[RankOneAction, ...]]:
        index: dict[Tensor, tuple[RankOneAction, ...]] = {}
        count = len(self.reference)
        if count > 20:
            raise ValueError("reference subset indexing is limited to 20 actions")
        zero = zero_tensor(self.shape)
        for mask in range(1 << count):
            remaining = tuple(self.reference[i] for i in range(count) if mask & (1 << i))
            residual = sum_actions(remaining, self.shape) if remaining else zero
            index.setdefault(residual, remaining)
        return index

    def propose_actions(self, residual, remaining_moves, rng, max_actions):
        if tensor_shape(residual) != self.shape or remaining_moves <= 0 or max_actions <= 0:
            return ()
        reference = list(self._remaining_by_residual.get(residual, ()))
        if self.randomized_order:
            rng.shuffle(reference)
        reference = reference[:max_actions]
        proposals: list[tuple[RankOneAction, float]] = []
        if reference:
            each = self.reference_prior / len(reference)
            proposals.extend((action, each) for action in reference)
        room = max_actions - len(reference)
        distractor_count = min(room, self.distractor_count)
        if self.distractor_policy is not None and distractor_count:
            distractors = self.distractor_policy.propose_actions(
                residual, remaining_moves, rng, distractor_count
            )
            mass = 1.0 - self.reference_prior if reference else 1.0
            if mass > 0:
                proposals.extend((item.action, mass * item.prior) for item in distractors)
        return _normalize(proposals)


@dataclass
class ActionStatistics:
    prior: float
    visits: int = 0
    value_sum: float = 0.0
    child: StateKey | None = None

    @property
    def value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass
class SearchNode:
    residual: Tensor
    remaining_moves: int
    terminal: bool
    visits: int = 0
    expanded: bool = False
    actions: dict[RankOneAction, ActionStatistics] = field(default_factory=dict)

    @property
    def key(self) -> StateKey:
        return (self.residual, self.remaining_moves)


def select_puct(node: SearchNode, c_puct: float) -> RankOneAction:
    """Select by PUCT; insertion order is the deterministic final tie-breaker."""
    if not node.actions:
        raise ValueError("cannot select from a node without actions")
    scale = math.sqrt(node.visits)
    return max(
        node.actions,
        key=lambda action: (
            node.actions[action].value
            + c_puct * node.actions[action].prior * scale / (1 + node.actions[action].visits),
            -list(node.actions).index(action),
        ),
    )


@dataclass(frozen=True)
class SearchConfig:
    simulations: int = 1000
    max_actions: int = 16
    max_moves: int = 7
    seed: int = 0
    c_puct: float = 1.5
    node_limit: int | None = None
    max_wall_time_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.simulations <= 0 or self.max_actions <= 0 or self.max_moves <= 0:
            raise ValueError("simulation, action, and move limits must be positive")
        if self.c_puct < 0 or not math.isfinite(self.c_puct):
            raise ValueError("c_puct must be finite and nonnegative")
        if self.node_limit is not None and self.node_limit <= 0:
            raise ValueError("node_limit must be positive when supplied")
        if self.max_wall_time_seconds is not None and self.max_wall_time_seconds <= 0:
            raise ValueError("max_wall_time_seconds must be positive when supplied")


@dataclass(frozen=True)
class RootActionDiagnostic:
    action: RankOneAction
    prior: float
    visits: int
    value: float


@dataclass(frozen=True)
class SearchResult:
    config: SearchConfig
    policy_type: str
    simulations_run: int
    expanded_nodes: int
    unique_states: int
    transposition_hits: int
    maximum_depth: int
    solution: tuple[RankOneAction, ...] | None
    exact_reconstruction: bool
    scalar_multiplications: int | None
    termination_reason: str
    root_visits: int
    root_actions: tuple[RootActionDiagnostic, ...]
    proposal_attempts: int = field(compare=False)
    valid_proposals: int = field(compare=False)
    accepted_proposals: int = field(compare=False)
    duplicate_proposals: int = field(compare=False)
    proposal_generation_seconds: float = field(compare=False)
    search_wall_seconds: float = field(compare=False)
    simulations_per_second: float = field(compare=False)
    value_evaluator_type: str = "nonzero_heuristic"
    value_evaluation_calls: int = field(default=0, compare=False)
    value_prediction_clips: int = field(default=0, compare=False)
    value_inference_seconds: float = field(default=0.0, compare=False)
    value_predictions: tuple[float, ...] = field(default=(), compare=False)


class SampledMCTS:
    """Deterministic seeded MCTS with exact residual/depth transpositions."""

    def __init__(
        self,
        target: Tensor,
        policy: ActionPolicy,
        config: SearchConfig,
        value_evaluator: LeafEvaluator | None = None,
    ):
        self.target = target
        self.shape = tensor_shape(target)
        self.policy = policy
        self.value_evaluator = value_evaluator or HeuristicValueEvaluator()
        self.config = config
        self.rng = random.Random(config.seed)
        self.nodes: dict[StateKey, SearchNode] = {}
        self.expanded_nodes = 0
        self.transposition_hits = 0
        self.maximum_depth = 0
        self.solution: tuple[RankOneAction, ...] | None = None
        self._target_nonzeros = sum(value != 0 for plane in target for row in plane for value in row)
        self._zero = zero_tensor(self.shape)

    def _node(self, residual: Tensor, remaining_moves: int) -> SearchNode:
        key = (residual, remaining_moves)
        existing = self.nodes.get(key)
        if existing is not None:
            return existing
        node = SearchNode(residual, remaining_moves, residual == self._zero)
        self.nodes[key] = node
        return node

    def _expand(self, node: SearchNode) -> None:
        if node.expanded or node.terminal or node.remaining_moves == 0:
            return
        proposals = self.policy.propose_actions(
            node.residual, node.remaining_moves, self.rng, self.config.max_actions
        )
        total = sum(item.prior for item in proposals)
        if proposals and (not math.isclose(total, 1.0, abs_tol=1e-12) or any(item.prior <= 0 for item in proposals)):
            raise ValueError("policy proposals must have positive normalized priors")
        node.actions = {item.action: ActionStatistics(item.prior) for item in proposals}
        node.expanded = True
        self.expanded_nodes += 1

    def _evaluate(self, node: SearchNode) -> float:
        if node.terminal:
            return float(self.config.max_moves)
        if node.remaining_moves == 0:
            return float(-(self.config.max_moves + 1))
        return self.value_evaluator.evaluate(
            node.residual,
            node.remaining_moves,
            self.config.max_moves,
            self._target_nonzeros,
        )

    def _backup(self, path: Sequence[tuple[SearchNode, RankOneAction]], leaf: SearchNode, value: float) -> None:
        leaf.visits += 1
        backed = value
        for node, action in reversed(path):
            backed = -1.0 + backed
            statistics = node.actions[action]
            statistics.visits += 1
            statistics.value_sum += backed
            node.visits += 1

    def run(self) -> SearchResult:
        if isinstance(self.policy, InstrumentedPolicy):
            self.policy.reset_metrics()
        self.value_evaluator.reset_metrics()
        started = time.perf_counter()
        root = self._node(self.target, self.config.max_moves)
        simulations_run = 0
        node_limited = False
        timed_out = False
        for _ in range(self.config.simulations):
            if (
                self.config.max_wall_time_seconds is not None
                and time.perf_counter() - started >= self.config.max_wall_time_seconds
            ):
                timed_out = True
                break
            simulations_run += 1
            node = root
            path: list[tuple[SearchNode, RankOneAction]] = []
            trajectory: list[RankOneAction] = []
            seen = {node.key}
            while not node.terminal and node.remaining_moves > 0:
                if not node.expanded:
                    self._expand(node)
                    break
                if not node.actions:
                    break
                action = select_puct(node, self.config.c_puct)
                statistics = node.actions[action]
                child_key = statistics.child
                if child_key is None:
                    residual = subtract_action(node.residual, action)
                    child_key = (residual, node.remaining_moves - 1)
                    existed = child_key in self.nodes
                    if (
                        not existed
                        and self.config.node_limit is not None
                        and len(self.nodes) >= self.config.node_limit
                    ):
                        node_limited = True
                        break
                    child = self._node(*child_key)
                    if existed:
                        self.transposition_hits += 1
                    statistics.child = child_key
                else:
                    child = self.nodes[child_key]
                path.append((node, action))
                trajectory.append(action)
                self.maximum_depth = max(self.maximum_depth, len(path))
                if child.key in seen:
                    break
                seen.add(child.key)
                node = child
            value = self._evaluate(node)
            self._backup(path, node, value)
            if node.terminal:
                candidate = tuple(trajectory)
                exact = sum_actions(candidate, self.shape) == self.target
                if exact and (self.solution is None or len(candidate) < len(self.solution)):
                    self.solution = candidate

        exact = self.solution is not None and sum_actions(self.solution, self.shape) == self.target
        if exact:
            reason = "solution_found"
        elif timed_out:
            reason = "wall_time_limit_reached"
        elif node_limited:
            reason = "node_limit_reached"
        else:
            reason = "simulation_budget_exhausted"
        diagnostics = tuple(
            RootActionDiagnostic(action, stats.prior, stats.visits, stats.value)
            for action, stats in root.actions.items()
        )
        wall_seconds = time.perf_counter() - started
        metrics = getattr(self.policy, "metrics", ProposalMetrics())
        value_metrics = self.value_evaluator.metrics
        return SearchResult(
            config=self.config,
            policy_type=self.policy.name,
            simulations_run=simulations_run,
            expanded_nodes=self.expanded_nodes,
            unique_states=len(self.nodes),
            transposition_hits=self.transposition_hits,
            maximum_depth=self.maximum_depth,
            solution=self.solution if exact else None,
            exact_reconstruction=exact,
            scalar_multiplications=len(self.solution) if exact else None,
            termination_reason=reason,
            root_visits=root.visits,
            root_actions=diagnostics,
            proposal_attempts=metrics.attempted,
            valid_proposals=metrics.valid,
            accepted_proposals=metrics.accepted,
            duplicate_proposals=metrics.duplicates,
            proposal_generation_seconds=metrics.generation_seconds,
            search_wall_seconds=wall_seconds,
            simulations_per_second=simulations_run / wall_seconds if wall_seconds else 0.0,
            value_evaluator_type=self.value_evaluator.name,
            value_evaluation_calls=value_metrics.calls,
            value_prediction_clips=value_metrics.clipped,
            value_inference_seconds=value_metrics.inference_seconds,
            value_predictions=tuple(value_metrics.predictions),
        )
