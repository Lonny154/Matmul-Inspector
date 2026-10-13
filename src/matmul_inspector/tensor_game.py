"""Exact, CPU-only tensor environment for matrix multiplication algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Iterable, Sequence

Tensor = tuple[tuple[tuple[int, ...], ...], ...]
Matrix = Sequence[Sequence[int]]


def _positive_dimension(name: str, value: int) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def zero_tensor(shape: tuple[int, int, int]) -> Tensor:
    """Return an immutable integer tensor with the requested positive shape."""
    x, y, z = (_positive_dimension("tensor dimension", value) for value in shape)
    return tuple(tuple(tuple(0 for _ in range(z)) for _ in range(y)) for _ in range(x))


def matrix_multiplication_tensor(m: int, k: int, n: int) -> Tensor:
    """Construct the exact row-major tensor for (M x K) @ (K x N)."""
    m = _positive_dimension("M", m)
    k = _positive_dimension("K", k)
    n = _positive_dimension("N", n)
    values = [[[0 for _ in range(m * n)] for _ in range(k * n)] for _ in range(m * k)]
    for i in range(m):
        for inner in range(k):
            for j in range(n):
                values[i * k + inner][inner * n + j][i * n + j] = 1
    return tuple(tuple(tuple(row) for row in plane) for plane in values)


def tensor_shape(tensor: Tensor) -> tuple[int, int, int]:
    """Return a validated, non-ragged third-order tensor shape."""
    if not tensor or not tensor[0] or not tensor[0][0]:
        raise ValueError("tensor must be nonempty and third-order")
    shape = (len(tensor), len(tensor[0]), len(tensor[0][0]))
    if any(len(plane) != shape[1] for plane in tensor) or any(
        len(row) != shape[2] for plane in tensor for row in plane
    ):
        raise ValueError("tensor must have a rectangular third-order shape")
    if any(not isinstance(value, Integral) or isinstance(value, bool) for plane in tensor for row in plane for value in row):
        raise ValueError("tensor coefficients must be integers")
    return shape


@dataclass(frozen=True)
class RankOneAction:
    """Integer factors whose outer product is one tensor-game action."""

    u: tuple[int, ...]
    v: tuple[int, ...]
    w: tuple[int, ...]

    def __init__(self, u: Iterable[int], v: Iterable[int], w: Iterable[int]):
        object.__setattr__(self, "u", tuple(u))
        object.__setattr__(self, "v", tuple(v))
        object.__setattr__(self, "w", tuple(w))


def validate_action(
    action: RankOneAction,
    shape: tuple[int, int, int],
    allowed_coefficients: frozenset[int] | None = frozenset({-1, 0, 1}),
) -> None:
    """Validate lengths, integer coefficients, bounds, and nonzero factors."""
    if not isinstance(action, RankOneAction):
        raise TypeError("action must be a RankOneAction")
    if tuple(map(len, (action.u, action.v, action.w))) != shape:
        raise ValueError(f"action vector lengths must match tensor shape {shape}")
    coefficients = action.u + action.v + action.w
    if any(not isinstance(value, Integral) or isinstance(value, bool) for value in coefficients):
        raise ValueError("action coefficients must be integers")
    if allowed_coefficients is not None and any(value not in allowed_coefficients for value in coefficients):
        raise ValueError(f"action coefficient is outside {sorted(allowed_coefficients)}")
    if any(not any(vector) for vector in (action.u, action.v, action.w)):
        raise ValueError("rank-one action must have three nonzero factor vectors")


def outer_product(action: RankOneAction) -> Tensor:
    """Construct u tensor-product v tensor-product w exactly."""
    return tuple(
        tuple(tuple(int(x * y * z) for z in action.w) for y in action.v)
        for x in action.u
    )


def sum_actions(actions: Sequence[RankOneAction], shape: tuple[int, int, int]) -> Tensor:
    """Sum rank-one actions into an exact immutable tensor."""
    total = [[[0 for _ in range(shape[2])] for _ in range(shape[1])] for _ in range(shape[0])]
    for action in actions:
        validate_action(action, shape, allowed_coefficients=None)
        for x, u_value in enumerate(action.u):
            for y, v_value in enumerate(action.v):
                for z, w_value in enumerate(action.w):
                    total[x][y][z] += u_value * v_value * w_value
    return tuple(tuple(tuple(row) for row in plane) for plane in total)


@dataclass(frozen=True)
class StepResult:
    reward: int
    terminated: bool
    truncated: bool


class TensorGame:
    """Deterministic residual-tensor game with exact integer transitions."""

    def __init__(
        self,
        target: Tensor,
        max_moves: int,
        allowed_coefficients: frozenset[int] | None = frozenset({-1, 0, 1}),
    ) -> None:
        self.shape = tensor_shape(target)
        self.target = target
        self.max_moves = _positive_dimension("max_moves", max_moves)
        self.allowed_coefficients = allowed_coefficients
        self.residual = target
        self.history: list[RankOneAction] = []
        self.terminated = False
        self.truncated = False

    def step(self, action: RankOneAction) -> StepResult:
        """Subtract one outer product. Invalid actions leave all state unchanged."""
        if self.terminated or self.truncated:
            raise RuntimeError("cannot step a finished game")
        validate_action(action, self.shape, self.allowed_coefficients)
        updated = tuple(
            tuple(
                tuple(
                    self.residual[x][y][z] - action.u[x] * action.v[y] * action.w[z]
                    for z in range(self.shape[2])
                )
                for y in range(self.shape[1])
            )
            for x in range(self.shape[0])
        )
        self.residual = updated
        self.history.append(action)
        self.terminated = all(value == 0 for plane in updated for row in plane for value in row)
        self.truncated = len(self.history) >= self.max_moves and not self.terminated
        return StepResult(reward=-1, terminated=self.terminated, truncated=self.truncated)

    @property
    def residual_nonzeros(self) -> int:
        return sum(value != 0 for plane in self.residual for row in plane for value in row)

    def reconstructs_target(self) -> bool:
        return sum_actions(self.history, self.shape) == self.target


def standard_2x2_decomposition() -> tuple[RankOneAction, ...]:
    """The ordinary eight-scalar-multiplication decomposition."""
    actions = []
    for i in range(2):
        for inner in range(2):
            for j in range(2):
                u = [0] * 4
                v = [0] * 4
                w = [0] * 4
                u[i * 2 + inner] = v[inner * 2 + j] = w[i * 2 + j] = 1
                actions.append(RankOneAction(u, v, w))
    return tuple(actions)


def strassen_2x2_decomposition() -> tuple[RankOneAction, ...]:
    """Strassen's seven explicit factor triplets in row-major order."""
    return (
        RankOneAction((1, 0, 0, 1), (1, 0, 0, 1), (1, 0, 0, 1)),
        RankOneAction((0, 0, 1, 1), (1, 0, 0, 0), (0, 0, 1, -1)),
        RankOneAction((1, 0, 0, 0), (0, 1, 0, -1), (0, 1, 0, 1)),
        RankOneAction((0, 0, 0, 1), (-1, 0, 1, 0), (1, 0, 1, 0)),
        RankOneAction((1, 1, 0, 0), (0, 0, 0, 1), (-1, 1, 0, 0)),
        RankOneAction((-1, 0, 1, 0), (1, 1, 0, 0), (0, 0, 0, 1)),
        RankOneAction((0, 1, 0, -1), (0, 0, 1, 1), (1, 0, 0, 0)),
    )


def multiply_with_decomposition(
    left: Matrix,
    right: Matrix,
    dimensions: tuple[int, int, int],
    actions: Sequence[RankOneAction],
) -> tuple[tuple[int, ...], ...]:
    """Evaluate a bilinear algorithm described by factor triplets."""
    m, k, n = dimensions
    if len(left) != m or any(len(row) != k for row in left):
        raise ValueError(f"left matrix must have shape ({m}, {k})")
    if len(right) != k or any(len(row) != n for row in right):
        raise ValueError(f"right matrix must have shape ({k}, {n})")
    left_flat = tuple(value for row in left for value in row)
    right_flat = tuple(value for row in right for value in row)
    output = [0] * (m * n)
    shape = (m * k, k * n, m * n)
    for action in actions:
        validate_action(action, shape, allowed_coefficients=None)
        scalar = sum(a * x for a, x in zip(action.u, left_flat)) * sum(
            b * x for b, x in zip(action.v, right_flat)
        )
        for index, coefficient in enumerate(action.w):
            output[index] += coefficient * scalar
    return tuple(tuple(output[i * n : (i + 1) * n]) for i in range(m))
