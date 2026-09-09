"""Experimental nonlinear, radiation-off reciprocal pair in native units.

Inputs are accepted states and histories, not an invented interacting past.
Both particles advance against the same frozen history; publication is atomic.
The 14-component state is [t_ns, x_mm(3), P_native(4), S_native(6)].
"""

from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np

from .constants import C_MMNS as c
from .full_dipole_history import FullDipoleHistory
from .full_dipole_response import response
from . import momentum_center as model

MODEL = "experimental_momentum_center_pair_v1"
UNITS = "mm_ns_amu_scaled_gaussian"
NativeProvider = Callable[
    [float, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
]


@dataclass(frozen=True)
class MomentumCenterParticle:
    charge_native: float
    mass_amu: float
    g: float = 2.0
    reaction_mode: str = "off"

    def __post_init__(self) -> None:
        self.length_time_particle()
        if self.reaction_mode != "off":
            raise ValueError("Nonlinear pair radiation reaction is not implemented")

    def length_time_particle(self) -> model.Particle:
        return model.Particle(self.charge_native / c, self.mass_amu, self.g)


def _to_length_time(state: Any) -> np.ndarray:
    value = np.asarray(state, dtype=float).copy()
    if value.shape != (14,) or not np.isfinite(value).all():
        raise ValueError("Finite 14-component native state required")
    value[0] *= c
    value[4:] /= c
    return value


def _from_length_time(state: np.ndarray) -> np.ndarray:
    value = np.asarray(state, dtype=float).copy()
    value[0] /= c
    value[4:] *= c
    return value


def _provider_length_time(provider: NativeProvider) -> model.Provider:
    def supplied(
        event: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        values = provider(event[0] / c, event[1:])
        shapes = ((4,), (4, 4), (4, 4), (4, 4, 4))
        if len(values) != 4:
            raise ValueError("Provider must return A, partial A, F, partial F")
        result = tuple(np.asarray(v, dtype=float) / c for v in values)
        if any(
            v.shape != shape or not np.isfinite(v).all()
            for v, shape in zip(result, shapes)
        ):
            raise ValueError("Invalid native potential response")
        return result[0], result[1], result[2], result[3]

    return supplied


def initial_state_native(
    event: Any,
    momentum_direction: Any,
    rest_spin_native: Any,
    particle: MomentumCenterParticle,
    provider: NativeProvider,
) -> np.ndarray:
    """Rest spin is in the momentum rest frame; direction is dimensionless."""
    event = np.asarray(event, dtype=float).copy()
    if event.shape != (4,):
        raise ValueError("Native event must be [t_ns, x_mm, y_mm, z_mm]")
    event[0] *= c
    return _from_length_time(
        model.initial_state(
            event,
            momentum_direction,
            np.asarray(rest_spin_native) / c,
            particle.length_time_particle(),
            _provider_length_time(provider),
        )
    )


def dynamics_native(
    state: np.ndarray, particle: MomentumCenterParticle, provider: NativeProvider
) -> tuple[np.ndarray, dict[str, Any]]:
    """Lab-time derivative plus explicitly separated native/length-time diagnostics."""
    rhs, diagnostic = model.evaluate(
        _to_length_time(state),
        particle.length_time_particle(),
        _provider_length_time(provider),
    )
    rate = rhs * (c / rhs[0])
    rate[0] /= c
    rate[4:] *= c
    return rate, dict(
        kinetic_momentum_native=c * diagnostic["kinetic_momentum"],
        proper_velocity_mm_ns=c * diagnostic["proper_velocity"],
        proper_dipole_native=c * diagnostic["proper_dipole"],
        length_time=diagnostic,
    )


class FullDipoleProvider:
    """Native A and derivatives from the published, full-tensor source history.

    Derivative indices refer to (ct, x, y, z), not (t, x, y, z).
    This reference path reuses analytical jets, not the sparse compiled kernel.
    """

    def __init__(self, history: FullDipoleHistory, charge_native: float) -> None:
        if history.speed_limit != c or not np.isfinite(charge_native):
            raise ValueError(
                "Native history requires speed_limit=C_MMNS and finite charge"
            )
        self.history = history
        self.charge_native = float(charge_native)

    def __call__(
        self, time_ns: float, position_mm: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        event = np.r_[c * time_ns, position_mm]
        if event.shape != (4,) or not np.isfinite(event).all():
            raise ValueError("Finite observer event required")
        for segment in reversed(self.history.segments):

            def cone(time: float) -> float:
                position, _ = segment.sample(time)
                return float(
                    c * (time_ns - time) - np.linalg.norm(event[1:] - position)
                )

            if cone(segment.start) >= 0 and cone(segment.end) <= 0:
                result = response(
                    event,
                    c * segment.start,
                    c * segment.duration,
                    segment.position,
                    segment.dipole / c,
                    charge=self.charge_native / c,
                    allow_boundary=True,
                )
                values = tuple(
                    c * result[key]
                    for key in (
                        "four_potential",
                        "partial_a",
                        "field_tensor",
                        "partial_f",
                    )
                )
                return values[0], values[1], values[2], values[3]
        raise ValueError(
            "Retarded source time is outside published history; no extrapolation"
        )


def initialize_pair(
    particles: list[MomentumCenterParticle],
    states: Any,
    histories: list[FullDipoleHistory],
) -> dict[str, Any]:
    """Checkpoint accepted native states and their already prepared source past."""
    if len(particles) != 2 or len(histories) != 2:
        raise ValueError("Exactly two particles and histories required")
    payload = dict(
        model=MODEL,
        units=UNITS,
        particles=[asdict(p) for p in particles],
        states=np.asarray(states, dtype=float).tolist(),
        histories=[h.to_checkpoint_payload() for h in histories],
        accepted_steps=0,
    )
    _restore(payload)
    return payload


def _restore(
    payload: dict[str, Any]
) -> tuple[list[MomentumCenterParticle], np.ndarray, list[FullDipoleHistory]]:
    if payload.get("model") != MODEL or payload.get("units") != UNITS:
        raise ValueError("Wrong nonlinear pair model or units")
    particles = [MomentumCenterParticle(**p) for p in payload["particles"]]
    histories = [
        FullDipoleHistory.from_checkpoint_payload(h) for h in payload["histories"]
    ]
    states = np.asarray(payload["states"], dtype=float)
    count = payload["accepted_steps"]
    if (
        len(particles) != 2
        or len(histories) != 2
        or states.shape != (2, 14)
        or not np.isfinite(states).all()
    ):
        raise ValueError("Invalid nonlinear pair checkpoint")
    if type(count) is not int or count < 0 or states[0, 0] != states[1, 0]:
        raise ValueError("Invalid step count or unequal accepted times")
    for state, history in zip(states, histories):
        if (
            history.speed_limit != c
            or history.time[-1] != state[0]
            or not np.array_equal(history.position[-1], state[1:4])
        ):
            raise ValueError("History endpoint must match the native accepted state")
    providers = [
        FullDipoleProvider(histories[1 - i], particles[1 - i].charge_native)
        for i in range(2)
    ]
    for state, particle, provider, history in zip(
        states, particles, providers, histories
    ):
        rate, diagnostic = dynamics_native(state, particle, provider)
        for stored, expected in (
            (history.velocity[-1], rate[1:4]),
            (history.dipole[-1], diagnostic["proper_dipole_native"]),
        ):
            scale = max(np.linalg.norm(stored), np.linalg.norm(expected))
            if np.linalg.norm(stored - expected) > 128 * np.finfo(float).eps * scale:
                raise ValueError(
                    "History endpoint velocity or dipole disagrees with the accepted state"
                )
    return particles, states, histories


def advance_pair(
    payload: dict[str, Any], width_ns: float, steps: int = 1
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Fixed lab-time RK4, returning new checkpoint and accepted-end diagnostics.

    Failure returns no partial checkpoint and never mutates the input. A smaller
    step does not necessarily repair insufficient published source history.
    """
    if (
        not np.isfinite(width_ns)
        or width_ns <= 0
        or type(steps) is not int
        or steps < 1
    ):
        raise ValueError("Positive finite step width and integer step count required")
    particles, states, histories = _restore(payload)
    records = []
    for _ in range(steps):
        providers = [
            FullDipoleProvider(histories[1 - i], particles[1 - i].charge_native)
            for i in range(2)
        ]
        trials, diagnostics, candidate_histories = [], [], []
        endpoint = states[0, 0] + width_ns
        for state, particle, provider, history in zip(
            states, particles, providers, histories
        ):

            def rhs(value: np.ndarray) -> np.ndarray:
                return dynamics_native(value, particle, provider)[0]

            k1 = rhs(state)
            k2 = rhs(state + width_ns * k1 / 2)
            k3 = rhs(state + width_ns * k2 / 2)
            k4 = rhs(state + width_ns * k3)
            trial = state + width_ns * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            trial[0] = endpoint
            rate, diagnostic = dynamics_native(trial, particle, provider)
            candidate_histories.append(
                history.append(
                    endpoint,
                    trial[1:4],
                    rate[1:4],
                    diagnostic["proper_dipole_native"],
                )
            )
            trials.append(trial)
            diagnostics.append(diagnostic)
        states, histories = np.asarray(trials), candidate_histories
        records.append(dict(time_ns=endpoint, particles=diagnostics))
    result = dict(
        model=MODEL,
        units=UNITS,
        particles=[asdict(p) for p in particles],
        states=states.tolist(),
        histories=[h.to_checkpoint_payload() for h in histories],
        accepted_steps=payload["accepted_steps"] + steps,
    )
    return result, records
