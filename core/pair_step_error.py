"""Physical error comparison for internal full-spin steps and their integrals."""

import numpy as np
from dataclasses import dataclass
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from .momentum_center_pair import MomentumCenterParticle

from .constants import C_MMNS as c
from . import momentum_center as model
from .nonlinear_pair_adaptive import _physical_state, _diagnostics
from .step_doubling import (
    StepDoublingState,
    StepDoublingTolerances,
    ErrorScale,
    assess_step_doubling,
)


@dataclass(frozen=True)
class InternalStepSettings:
    position_mm: ErrorScale
    momentum_native: ErrorScale
    spin_native: ErrorScale
    impulse_native: ErrorScale
    torque_native: ErrorScale
    mass_exchange: ErrorScale
    maximum_trials: int = 512

    def __post_init__(self) -> None:
        for key in (
            "position_mm",
            "momentum_native",
            "spin_native",
            "impulse_native",
            "torque_native",
            "mass_exchange",
        ):
            if not isinstance(getattr(self, key), ErrorScale):
                raise ValueError("Explicit physical error scales required")
        if type(self.maximum_trials) is not int or not 1 <= self.maximum_trials <= 4096:
            raise ValueError("Internal trial budget must be between one and 4096")

    @classmethod
    def from_payload(cls, value):
        values = dict(value)
        for key in (
            "position_mm",
            "momentum_native",
            "spin_native",
            "impulse_native",
            "torque_native",
            "mass_exchange",
        ):
            values[key] = ErrorScale(**values[key])
        return cls(**values)


def reaction_rate_vector(data):
    """41 rates: four impulses, four torques, and action-mass exchange."""
    reaction = data.get("reaction")
    if reaction is None:
        return np.zeros(41)
    rows = [
        reaction,
        *[reaction["sectors"][key] for key in ("q_squared", "qD", "D_squared")],
    ]
    factor = c**2 / data["proper_velocity"][0]
    return np.r_[
        np.concatenate([np.asarray(row["force"]) for row in rows]) * factor,
        np.concatenate([model.pack(row["torque"]) for row in rows]) * factor,
        c * reaction["external_action_mass_rate"] / data["proper_velocity"][0],
    ]


def reaction_ledger(values):
    values = np.asarray(values)
    if values.shape != (41,):
        raise ValueError("41 reaction integral components required")
    impulses, torques = values[:16].reshape(4, 4), values[16:40].reshape(4, 6)
    return dict(
        applied_impulse_native=impulses[0].tolist(),
        applied_torque_native=torques[0].tolist(),
        external_action_mass_exchange=float(values[40]),
        sectors={
            key: dict(
                impulse_native=impulses[i].tolist(), torque_native=torques[i].tolist()
            )
            for i, key in enumerate(("q_squared", "qD", "D_squared"), 1)
        },
    )


def physical_error_norm(
    proposed, error, particle, provider, settings, *, preserve_error=False
):
    """Compare an embedded error estimate in physical, not stored-momentum units."""
    if preserve_error:
        return _preserved_physical_error_norm(
            proposed, error, particle, provider, settings
        )
    alternative = proposed - error
    states = []
    for value in (proposed, alternative):
        potential = provider(value[0], value[1:4])[0]
        kinetic = value[4:8] - particle.charge_native * potential / c
        states.append(
            _physical_state(
                dict(states=[value[:14]]),
                [dict(particles=[dict(kinetic_momentum_native=kinetic)])],
            )
        )
    ledgers = [
        dict(dipole_reaction_ledger=[reaction_ledger(value[14:])])
        for value in (proposed, alternative)
    ]
    diagnostics = _diagnostics(ledgers[0], ledgers[1], settings)
    a, b = [
        StepDoublingState(state[0], state[1], state[2], diag)
        for state, diag in zip(states, diagnostics)
    ]
    return assess_step_doubling(
        a,
        b,
        method_order=1,
        tolerances=StepDoublingTolerances(
            settings.position_mm,
            settings.momentum_native,
            settings.spin_native,
            ErrorScale(1, 0),
        ),
    ).normalized_error


def _preserved_physical_error_norm(
    proposed: np.ndarray,
    error: np.ndarray,
    particle: "MomentumCenterParticle",
    provider: Callable[[float, np.ndarray], tuple[np.ndarray, ...]],
    settings: InternalStepSettings,
) -> float:
    """Retain small error estimates through subtraction and rest-frame conversion.

    Decimal is used only for error bookkeeping, not the potential solver. The
    provider still evaluates rounded float64 coordinates, as in the existing
    physical adapter. This does not certify sub-float64 field accuracy.
    """
    from decimal import Decimal, localcontext
    from .preserved_source import decimal_array

    if (
        np.shape(proposed) != (55,)
        or np.shape(error) != (55,)
        or not np.isfinite([proposed, error]).all()
    ):
        raise ValueError("Finite 55-component state and error required")
    with localcontext() as context:
        context.prec = 80
        first = decimal_array(proposed)
        second = first - decimal_array(error)
        groups = []
        for value in (first, second):
            rounded = np.asarray(value, dtype=float)
            potential = np.asarray(provider(rounded[0], rounded[1:4])[0])
            if potential.shape != (4,) or not np.isfinite(potential).all():
                raise ValueError("Finite four-potential required for physical errors")
            momentum = value[4:8] - Decimal.from_float(
                particle.charge_native
            ) * decimal_array(potential) / Decimal.from_float(c)
            mass2 = momentum[0] ** 2 - sum(v * v for v in momentum[1:])
            if mass2 <= 0 or momentum[0] <= 0:
                raise ValueError(
                    "Adaptive spin comparison requires timelike kinetic momentum"
                )
            mass = mass2.sqrt()
            boost = decimal_array(np.eye(4))
            boost[0, 0] = momentum[0] / mass
            boost[0, 1:] = boost[1:, 0] = -momentum[1:] / mass
            boost[1:, 1:] += np.outer(momentum[1:], momentum[1:]) / (
                mass * (momentum[0] + mass)
            )
            rest = boost @ model.unpack(value[8:14]) @ boost.T
            groups.append(
                (
                    value[1:4],
                    momentum,
                    np.array([rest[2, 3], rest[3, 1], rest[1, 2]]),
                    value[14:30],
                    value[30:54],
                    value[54:],
                )
            )
        maximum = Decimal(0)
        scales = (
            settings.position_mm,
            settings.momentum_native,
            settings.spin_native,
            settings.impulse_native,
            settings.torque_native,
            settings.mass_exchange,
        )
        for a, b, scale in zip(*groups, scales):
            absolute, relative = Decimal.from_float(
                float(scale.absolute)
            ), Decimal.from_float(float(scale.relative))
            for x, y in zip(a, b):
                denominator = absolute + relative * max(abs(x), abs(y))
                difference = abs(x - y)
                if denominator:
                    maximum = max(maximum, difference / denominator)
                elif difference:
                    return float("inf")
        return float(maximum)
