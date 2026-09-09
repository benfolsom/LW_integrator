"""Experimental nonlinear reciprocal pair with an optional zero-spin charge recoil.

Inputs are accepted states and histories, not an invented interacting past.
Both particles advance against the same frozen history; publication is atomic.
The 14-component state is [t_ns, x_mm(3), P_native(4), S_native(6)].
"""

from dataclasses import asdict, dataclass
import copy
from typing import Any, Callable

import numpy as np

from .constants import C_MMNS as c
from .full_dipole_history import FullDipoleHistory, SourcePositionError
from .full_dipole_response import response
from . import momentum_center as model
from .momentum_center_reaction import charge_reaction

MODEL = "experimental_momentum_center_pair_v1"
UNITS = "mm_ns_amu_scaled_gaussian"
NativeProvider = Callable[
    [float, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
]


def smooth_start_native(provider: NativeProvider, duration_ns: float) -> NativeProvider:
    """Explicit driven startup: multiply A by a C5 ramp, including derivatives.

    Ramp time starts at lab t=0. Derivatives are with respect to ct, not t.
    This compensating external potential vanishes after the ramp; it is not
    an isolated interacting prehistory or a Lorentz-invariant preparation.
    """
    if not np.isfinite(duration_ns) or duration_ns < 0:
        raise ValueError("Finite nonnegative startup duration required")
    coefficients = np.array([0.0, 0, 0, 0, 0, 0, 462, -1980, 3465, -3080, 1386, -252])

    def wrapped(
        time_ns: float, position: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if duration_ns == 0 or time_ns >= duration_ns:
            return provider(time_ns, position)
        if time_ns <= 0:
            return np.zeros(4), np.zeros((4, 4)), np.zeros((4, 4)), np.zeros((4, 4, 4))
        r, dr, ddr = [
            np.polynomial.polynomial.polyval(
                time_ns / duration_ns, np.polynomial.polynomial.polyder(coefficients, n)
            )
            / (c * duration_ns) ** n
            for n in range(3)
        ]
        a, da, field, gradient = provider(time_ns, position)
        wedge = np.zeros((4, 4))
        wedge[0, :] += a
        wedge[:, 0] -= a
        new_da = r * da
        new_da[0] += dr * a
        new_gradient = r * gradient
        new_gradient[0] += dr * field + ddr * wedge
        for k in range(4):
            new_gradient[k, 0, :] += dr * da[k]
            new_gradient[k, :, 0] -= dr * da[k]
        return r * a, new_da, r * field + dr * wedge, new_gradient

    return wrapped


@dataclass(frozen=True)
class MomentumCenterParticle:
    charge_native: float
    mass_amu: float
    g: float = 2.0
    reaction_mode: str = "off"

    def __post_init__(self) -> None:
        self.length_time_particle()
        if self.reaction_mode not in ("off", "charge_ll"):
            raise ValueError(
                "Requested reaction is not implemented; supported modes are off and zero-spin charge_ll"
            )

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
    if particle.reaction_mode == "charge_ll" and np.any(rest_spin_native):
        raise ValueError(
            "charge_ll requires exactly zero spin; finite-spin reaction is not implemented"
        )
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
    scaled_state = _to_length_time(state)
    if particle.reaction_mode == "charge_ll" and np.any(scaled_state[8:]):
        raise ValueError("charge_ll requires exactly zero spin; no spin was discarded")
    values = _provider_length_time(provider)(scaled_state[:4])
    rhs, diagnostic = model.evaluate(
        scaled_state, particle.length_time_particle(), lambda event: values
    )
    if particle.reaction_mode == "charge_ll":
        reaction = charge_reaction(
            diagnostic["proper_velocity"],
            values[2],
            values[3],
            charge=particle.charge_native / c,
            mass=particle.mass_amu,
        )
        force = np.asarray(reaction["force"])
        rhs[4:8] += force
        diagnostic["momentum_rate"] += force
        diagnostic["current_residual"] += force
        diagnostic["mass_constraint_rate"] += 2 * model.dot(
            diagnostic["kinetic_momentum"], force
        )
        diagnostic["reaction"] = reaction
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

    def __init__(
        self,
        history: FullDipoleHistory,
        charge_native: float,
        startup_duration_ns: float = 0.0,
    ) -> None:
        if history.speed_limit != c or not np.isfinite(charge_native):
            raise ValueError(
                "Native history requires speed_limit=C_MMNS and finite charge"
            )
        self.history = history
        self.charge_native = float(charge_native)
        self.samples: list[tuple[float, float, float]] = []
        self.prepared = smooth_start_native(self._response, startup_duration_ns)

    def __call__(
        self, time_ns: float, position_mm: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.prepared(time_ns, position_mm)

    def _response(
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
                self.samples.append(
                    (time_ns, result["retarded_time"] / c, segment.start)
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
    *,
    startup_duration_ns: float = 0.0,
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
        startup_duration_ns=startup_duration_ns,
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
        FullDipoleProvider(
            histories[1 - i],
            particles[1 - i].charge_native,
            payload.get("startup_duration_ns", 0.0),
        )
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
    ledger_keys = (
        "applied_impulse_native",
        "outward_radiation_native",
        "bound_rate_integral_native",
    )
    ledger = copy.deepcopy(
        payload.get(
            "reaction_ledger",
            [{key: [0.0] * 4 for key in ledger_keys} for _ in range(2)],
        )
    )
    if (
        not isinstance(ledger, list)
        or len(ledger) != 2
        or any(
            not isinstance(row, dict)
            or set(row) != set(ledger_keys)
            or any(
                np.shape(row[key]) != (4,) or not np.isfinite(row[key]).all()
                for key in ledger_keys
            )
            for row in ledger
        )
    ):
        raise ValueError("Invalid reaction ledger in checkpoint")
    records = []
    for _ in range(steps):
        providers = [
            FullDipoleProvider(
                histories[1 - i],
                particles[1 - i].charge_native,
                payload.get("startup_duration_ns", 0.0),
            )
            for i in range(2)
        ]
        trials, diagnostics, candidate_histories = [], [], []
        endpoint = states[0, 0] + width_ns
        for i, (state, particle, provider, history) in enumerate(
            zip(states, particles, providers, histories)
        ):
            stage_diagnostics = []

            def rhs(value: np.ndarray) -> np.ndarray:
                rate, data = dynamics_native(value, particle, provider)
                stage_diagnostics.append(data["length_time"])
                return rate

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
            if particle.reaction_mode != "off":
                for key, source in zip(
                    ledger_keys,
                    ("force", "outward_radiation_rate", "bound_momentum_rate"),
                ):
                    increment = (
                        width_ns
                        * c**2
                        / 6
                        * sum(
                            weight
                            * np.asarray(d["reaction"][source])
                            / d["proper_velocity"][0]
                            for weight, d in zip((1, 2, 2, 1), stage_diagnostics)
                        )
                    )
                    ledger[i][key] = (np.asarray(ledger[i][key]) + increment).tolist()
        states, histories = np.asarray(trials), candidate_histories
        records.append(
            dict(
                time_ns=endpoint,
                particles=diagnostics,
                source_sampling=[
                    dict(
                        count=len(p.samples),
                        evolved_count=sum(s[2] >= 0 for s in p.samples),
                        minimum_delay_ns=min(
                            (s[0] - s[1] for s in p.samples), default=None
                        ),
                    )
                    for p in providers
                ],
            )
        )
    result = dict(
        model=MODEL,
        units=UNITS,
        particles=[asdict(p) for p in particles],
        states=states.tolist(),
        histories=[h.to_checkpoint_payload() for h in histories],
        accepted_steps=payload["accepted_steps"] + steps,
        startup_duration_ns=payload.get("startup_duration_ns", 0.0),
    )
    if "reaction_ledger" in payload or any(p.reaction_mode != "off" for p in particles):
        result["reaction_ledger"] = ledger
    return result, records


def advance_pair_refined(
    payload: dict[str, Any], width_ns: float, max_halvings: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Try one interval, subdividing only source-position budget failures.

    A velocity-domain error, causal-history gap or other failure is not retried.
    A failed interval (including either subdivided half) leaves input untouched.
    This is bounded accuracy-budget recovery, not a local-truncation estimator.
    """
    if type(max_halvings) is not int or not 0 <= max_halvings <= 10:
        raise ValueError("Source-position recovery requires 0 to 10 step halvings")
    try:
        return advance_pair(payload, width_ns, 1)
    except SourcePositionError:
        if max_halvings == 0:
            raise
        middle, first = advance_pair_refined(payload, width_ns / 2, max_halvings - 1)
        final, second = advance_pair_refined(middle, width_ns / 2, max_halvings - 1)
        return final, first + second
