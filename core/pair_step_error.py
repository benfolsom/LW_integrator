"""Physical error comparison for internal full-spin steps and their integrals."""

import numpy as np
from dataclasses import dataclass

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


def physical_error_norm(proposed, error, particle, provider, settings):
    """Compare an embedded error estimate in physical, not stored-momentum units."""
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
