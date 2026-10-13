"""Bounded sampled Monte Carlo tree search for exact TensorGame states."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
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


class ActionPolicy(Protocol):
    name: str

    def propose_actions(
        self,
        residual: Tensor,
        remaining_moves: int,
        rng: random.Random,
        max_actions: int,
    ) -> tuple[ProposedAction, ...]: ...


def _normalize(proposals: Sequence[tuple[RankOneAction, float]]) -> tuple[ProposedAction, ...]:
    unique: dict[RankOneAction, float] = {}
    for action, prior in proposals:
        canonical = canonicalize_action(action)
        unique[canonical] = unique.get(canonical, 0.0) + prior
    total = sum(unique.values())
    if not unique or not math.isfinite(total) or total <= 0:
        return ()
    return tuple(ProposedAction(action, prior / total) for action, prior in unique.items())


@dataclass(frozen=True)
class UniformSampledPolicy:
    """Sample bounded factor vectors without enumerating their Cartesian product."""

    shape: tuple[int, int, int]
    coefficients: tuple[int, ...] = (-1, 0, 1)
    attempt_multiplier: int = 50
    name: str = field(default="uniform_sampled", init=False)

    def __post_init__(self) -> None:
        if len(self.shape) != 3 or any(value <= 0 for value in self.shape):
            raise ValueError("shape must contain three positive dimensions")
        if not self.coefficients or any(not isinstance(value, int) or isinstance(value, bool) for value in self.coefficients):
            raise ValueError("coefficients must be a nonempty tuple of integers")
        if self.attempt_multiplier <= 0:
            raise ValueError("attempt_multiplier must be positive")

    def propose_actions(self, residual, remaining_moves, rng, max_actions):
        if tensor_shape(residual) != self.shape or remaining_moves <= 0 or max_actions <= 0:
            return ()
        actions: dict[RankOneAction, float] = {}
        attempts = 0
        while len(actions) < max_actions and attempts < max_actions * self.attempt_multiplier:
            attempts += 1
            vectors = tuple(
                tuple(rng.choice(self.coefficients) for _ in range(length))
                for length in self.shape
            )
            action = RankOneAction(*vectors)
            try:
                validate_action(action, self.shape, frozenset(self.coefficients))
            except ValueError:
                continue
            actions.setdefault(canonicalize_action(action), 1.0)
        return _normalize(tuple(actions.items()))


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

    def __post_init__(self) -> None:
        if self.simulations <= 0 or self.max_actions <= 0 or self.max_moves <= 0:
            raise ValueError("simulation, action, and move limits must be positive")
        if self.c_puct < 0 or not math.isfinite(self.c_puct):
            raise ValueError("c_puct must be finite and nonnegative")
        if self.node_limit is not None and self.node_limit <= 0:
            raise ValueError("node_limit must be positive when supplied")


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


class SampledMCTS:
    """Deterministic seeded MCTS with exact residual/depth transpositions."""

    def __init__(self, target: Tensor, policy: ActionPolicy, config: SearchConfig):
        self.target = target
        self.shape = tensor_shape(target)
        self.policy = policy
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
        nonzeros = sum(value != 0 for plane in node.residual for row in plane for value in row)
        return -nonzeros / max(1, self._target_nonzeros)

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
        root = self._node(self.target, self.config.max_moves)
        simulations_run = 0
        node_limited = False
        for _ in range(self.config.simulations):
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
        elif node_limited:
            reason = "node_limit_reached"
        else:
            reason = "simulation_budget_exhausted"
        diagnostics = tuple(
            RootActionDiagnostic(action, stats.prior, stats.visits, stats.value)
            for action, stats in root.actions.items()
        )
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
        )
