"""Experimental action-mass momentum-center closure, c=1, (+---), no recoil.

State: x^mu, stored P^mu=p^mu+q A^mu, six spin-tensor components. These
coordinates are NOT asserted to have canonical Poisson brackets. Physical
momentum obeys S p=0 and p^2=m0^2+(gq/2) F S. Dipole D=k S with
k=(gq/2)*(p.u)/p^2. No expansion in spin; not a production default.
"""

import numpy as np

from dataclasses import dataclass
from itertools import permutations
from typing import Any, Callable

Provider = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]


class VelocityDomainError(ValueError):
    """The model has no usable future timelike velocity at this state."""

    def __init__(
        self,
        message: str,
        *,
        beta_squared: float | None = None,
        direction_norm_squared: float | None = None,
    ) -> None:
        super().__init__(message)
        self.beta_squared = beta_squared
        self.direction_norm_squared = direction_norm_squared


METRIC = np.array([1.0, -1.0, -1.0, -1.0])
PAIRS = tuple((i, j) for i in range(4) for j in range(i + 1, 4))
EPS = np.zeros((4, 4, 4, 4))
for indices in permutations(range(4)):
    inversions = sum(indices[i] > indices[j] for i in range(4) for j in range(i + 1, 4))
    EPS[indices] = -((-1.0) ** inversions)


def dot(a: np.ndarray, b: np.ndarray) -> float:
    return float((METRIC * a) @ b)


def contract(field: np.ndarray, tensor: np.ndarray) -> float:
    return float(np.einsum("ab,a,b,ab", field, METRIC, METRIC, tensor))


def spin_tensor(u: np.ndarray, spin: np.ndarray) -> np.ndarray:
    return np.einsum("abcd,c,d->ab", EPS, METRIC * u, METRIC * spin)


def pack(tensor: np.ndarray) -> np.ndarray:
    return np.array([tensor[i, j] for i, j in PAIRS])


def unpack(values: np.ndarray) -> np.ndarray:
    tensor = np.zeros((4, 4), dtype=np.asarray(values).dtype)
    for value, (i, j) in zip(values, PAIRS):
        tensor[i, j], tensor[j, i] = value, -value
    return tensor


@dataclass(frozen=True)
class Particle:
    """Length-time Gaussian parameters; native charge must be divided by c."""

    charge: float = 1.0
    bare_mass: float = 1.0
    g: float = 2.0

    def __post_init__(self) -> None:
        if (
            not np.isfinite([self.charge, self.bare_mass, self.g]).all()
            or self.bare_mass <= 0
        ):
            raise ValueError("Finite parameters and positive bare mass required")

    @property
    def coupling(self) -> float:
        return self.g * self.charge / (2 * self.bare_mass)


def velocity_direction(
    momentum: np.ndarray,
    tensor: np.ndarray,
    field: np.ndarray,
    gradient: np.ndarray,
    particle: Particle,
) -> tuple[Any, ...]:
    """Unnormalized velocity for the exact constraint solve and boundary audit."""
    for value, shape in (
        (momentum, (4,)),
        (tensor, (4, 4)),
        (field, (4, 4)),
        (gradient, (4, 4, 4)),
    ):
        if np.shape(value) != shape or not np.isfinite(value).all():
            raise VelocityDomainError(
                "Finite momentum, spin, field and gradient required"
            )
    mass2 = dot(momentum, momentum)
    if not np.isfinite(mass2) or mass2 <= 0 or momentum[0] <= 0:
        raise VelocityDomainError("Future timelike kinetic momentum required")
    # RK trial states need an off-constraint extension. Report the constraint
    # and its rate; never project the stored spin or momentum to hide drift.
    mixed = field * METRIC[None, :]
    unit_torque = mixed @ tensor + tensor @ mixed.T
    unit_force = (
        0.5 * METRIC * np.array([contract(entry, tensor) for entry in gradient])
    )
    matrix = mass2 * np.eye(4) - particle.charge * (tensor * METRIC[None, :]) @ mixed
    coefficient = particle.bare_mass * particle.coupling
    spin_right = (
        coefficient
        / mass2
        * (unit_torque @ (METRIC * momentum) + tensor @ (METRIC * unit_force))
    )
    spin_right += (
        particle.charge / mass2 * (tensor * METRIC[None, :]) @ mixed @ momentum
    )
    if not np.isfinite(matrix).all() or not np.isfinite(spin_right).all():
        raise VelocityDomainError("Nonfinite momentum-to-velocity system")
    try:
        correction = np.linalg.solve(matrix, spin_right)
    except np.linalg.LinAlgError as error:
        raise VelocityDomainError("Singular momentum-to-velocity system") from error
    direction = momentum / mass2 + correction
    return (
        direction,
        unit_force,
        unit_torque,
        coefficient / mass2,
        np.linalg.cond(matrix),
        correction,
    )


def velocity(
    momentum: np.ndarray,
    tensor: np.ndarray,
    field: np.ndarray,
    gradient: np.ndarray,
    particle: Particle,
) -> tuple[Any, ...]:
    """Differentiate S p=0 and solve normalization without expanding in spin."""
    direction, unit_force, unit_torque, coefficient, condition, correction = (
        velocity_direction(momentum, tensor, field, gradient, particle)
    )
    norm_squared = dot(direction, direction)
    beta_squared = (
        float(np.dot(direction[1:] / direction[0], direction[1:] / direction[0]))
        if np.isfinite(direction).all() and direction[0] != 0
        else None
    )
    if not np.isfinite(norm_squared) or norm_squared <= 0:
        raise VelocityDomainError(
            "No continuous timelike velocity branch",
            beta_squared=beta_squared,
            direction_norm_squared=norm_squared,
        )
    omega = 1 / np.sqrt(norm_squared)
    u = omega * direction
    if not np.isfinite(u).all() or u[0] <= 0:
        raise VelocityDomainError(
            "Nonfinite or nonfuture velocity branch",
            beta_squared=beta_squared,
            direction_norm_squared=norm_squared,
        )
    coupling = coefficient * omega
    return (
        u,
        coupling * unit_force,
        coupling * unit_torque,
        coupling,
        condition,
        omega * correction,
    )


def initial_state(
    x: Any,
    momentum_direction: Any,
    rest_spin: Any,
    particle: Particle,
    provider: Provider,
) -> np.ndarray:
    """Specify momentum direction and its rest-frame spin; mass follows action."""
    x, momentum, rest = map(np.asarray, (x, momentum_direction, rest_spin))
    if x.shape != (4,) or momentum.shape != (4,) or rest.shape != (3,):
        raise ValueError("Invalid initial shape")
    if (
        not all(np.isfinite(v).all() for v in (x, momentum, rest))
        or momentum[0] <= 0
        or dot(momentum, momentum) <= 0
    ):
        raise ValueError("Finite data and timelike initial momentum required")
    reference = momentum / np.sqrt(dot(momentum, momentum))
    s0 = reference[1:] @ rest
    spin = np.r_[s0, rest + reference[1:] * s0 / (1 + reference[0])]
    tensor = spin_tensor(reference, spin)
    a, _, field, _ = provider(x)
    mass2 = particle.bare_mass**2 + particle.bare_mass * particle.coupling * contract(
        field, tensor
    )
    if mass2 <= 0:
        raise ValueError("Action mass relation is not timelike")
    momentum = np.sqrt(mass2) * reference
    state = np.r_[x, momentum + particle.charge * a, pack(tensor)]
    evaluate(state, particle, provider)
    return state


def evaluate(
    state: np.ndarray, particle: Particle, provider: Provider
) -> tuple[np.ndarray, dict[str, Any]]:
    state = np.asarray(state, dtype=float)
    if state.shape != (14,) or not np.isfinite(state).all():
        raise ValueError("Finite 14-component momentum-center state required")
    a, da, field, gradient = provider(state[:4])
    for value, shape in (
        (a, (4,)),
        (da, (4, 4)),
        (field, (4, 4)),
        (gradient, (4, 4, 4)),
    ):
        if np.shape(value) != shape or not np.isfinite(value).all():
            raise ValueError("Finite potential response with correct shapes required")
    momentum = state[4:8] - particle.charge * a
    tensor = unpack(state[8:14])
    u, dipole_force, torque, coupling, condition, velocity_correction = velocity(
        momentum, tensor, field, gradient, particle
    )
    momentum_rate = particle.charge * field @ (METRIC * u) + dipole_force
    # The omitted p wedge p term vanishes analytically. Do not manufacture
    # spin from cancellation of large parallel momenta in a charge-only run.
    spin_rate = (
        np.outer(momentum, velocity_correction)
        - np.outer(velocity_correction, momentum)
        + torque
    )
    stored_rate = momentum_rate + particle.charge * np.einsum("a,ab->b", u, da)
    current = particle.charge * field @ (METRIC * u) - coupling * np.einsum(
        "bn,nab->a", METRIC[:, None] * tensor, gradient
    )
    scalar_rate = (
        contract(field, spin_rate)
        + np.array([contract(entry, tensor) for entry in gradient]) @ u
    )
    coefficient = particle.bare_mass * particle.coupling
    return np.r_[u, stored_rate, pack(spin_rate)], dict(
        kinetic_momentum=momentum,
        proper_velocity=u,
        momentum_rate=momentum_rate,
        current_residual=momentum_rate - current,
        spin_constraint=tensor @ (METRIC * momentum),
        constraint_rate=spin_rate @ (METRIC * momentum)
        + tensor @ (METRIC * momentum_rate),
        velocity_spin_contraction=tensor @ (METRIC * u),
        spin_invariant=contract(tensor, tensor) / 2,
        kinetic_mass=np.sqrt(dot(momentum, momentum)),
        momentum_rest_frame_beta_squared=1.0
        - dot(momentum, momentum) / dot(momentum, u) ** 2,
        action_interaction_relative=coefficient
        * contract(field, tensor)
        / particle.bare_mass**2,
        velocity_matrix_condition=condition,
        proper_dipole=coupling * tensor,
        proper_dipole_coupling=coupling,
        mass_constraint=dot(momentum, momentum)
        - particle.bare_mass**2
        - coefficient * contract(field, tensor),
        mass_constraint_rate=2 * dot(momentum, momentum_rate)
        - coefficient * scalar_rate,
    )


def rk4(
    state: np.ndarray, width: float, particle: Particle, provider: Provider
) -> np.ndarray:
    if not np.isfinite(width) or width <= 0:
        raise ValueError("Positive proper-time step required")

    def rhs(value: np.ndarray) -> np.ndarray:
        return evaluate(value, particle, provider)[0]

    k1 = rhs(state)
    k2 = rhs(state + width * k1 / 2)
    k3 = rhs(state + width * k2 / 2)
    k4 = rhs(state + width * k3)
    result = state + width * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    evaluate(result, particle, provider)
    return result
