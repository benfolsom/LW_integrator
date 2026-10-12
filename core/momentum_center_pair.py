"""Nonlinear reciprocal pair with optional charge/full-dipole recoil.

Inputs are accepted states and histories, not an invented interacting past.
Both particles advance against the same frozen history; publication is atomic.
The 14-component state is [t_ns, x_mm(3), P_native(4), S_native(6)].
"""

from dataclasses import asdict, dataclass
import copy
from typing import Any, Callable, Iterator, TYPE_CHECKING

if TYPE_CHECKING:
    from .stochastic_emission import StochasticEmissionConfig

import numpy as np
from scipy.special import betainc

from .constants import C_MMNS as c
from .full_dipole_history import FullDipoleHistory, SourcePositionError
from .full_dipole_response import response
from . import momentum_center as model
from .momentum_center_reaction import charge_reaction
from .full_dipole_reaction import evaluate_reaction
from .coupled_dipole_reaction import evaluate_coupled_reaction

MODEL = "experimental_momentum_center_pair_v1"
MULTIPARTICLE_MODEL = "momentum_center_particles_v1"
UNITS = "mm_ns_amu_scaled_gaussian"
FULL_DIPOLE_REACTION_MODES = ("full_dipole_rr", "full_dipole_coupled")
NativeResponse = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
NativeProvider = Callable[[float, np.ndarray], NativeResponse]


def smooth_start_native(
    provider: NativeProvider, duration_ns: float, smoothness: int = 5
) -> NativeProvider:
    """Explicit driven startup with selectable C5 or C7 potential smoothness.

    Ramp time starts at lab t=0. Derivatives are with respect to ct, not t.
    This compensating external potential vanishes after the ramp; it is not
    an isolated interacting prehistory or a Lorentz-invariant preparation.
    """
    if not np.isfinite(duration_ns) or duration_ns < 0:
        raise ValueError("Finite nonnegative startup duration required")
    if smoothness not in (5, 7):
        raise ValueError("Startup smoothness must be 5 or 7")
    from math import comb

    coefficients = np.zeros(2 * smoothness + 2)
    for k in range(smoothness + 1):
        coefficients[smoothness + 1 + k] = (
            (-1) ** k
            * comb(smoothness + k, k)
            * comb(2 * smoothness + 1, smoothness - k)
        )

    def wrapped(
        time_ns: float, position: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if duration_ns == 0 or time_ns >= duration_ns:
            return provider(time_ns, position)
        if time_ns <= 0:
            return np.zeros(4), np.zeros((4, 4)), np.zeros((4, 4)), np.zeros((4, 4, 4))
        s = time_ns / duration_ns
        if smoothness == 5:
            # Preserve the old ramp for replay of pre-reaction checkpoints.
            r, dr, ddr = [
                np.polynomial.polynomial.polyval(
                    s, np.polynomial.polynomial.polyder(coefficients, n)
                )
                / (c * duration_ns) ** n
                for n in range(3)
            ]
        else:
            # The expanded polynomial cancels badly near its upper endpoint.
            # This is the identical smoothstep evaluated as a beta CDF; its
            # first two derivatives are factored, with no large cancellations.
            n = smoothness
            normalization = (2 * n + 1) * comb(2 * n, n)
            r = betainc(n + 1, n + 1, s)
            dr = normalization * (s * (1 - s)) ** n / (c * duration_ns)
            ddr = (
                normalization
                * n
                * (1 - 2 * s)
                * (s * (1 - s)) ** (n - 1)
                / (c * duration_ns) ** 2
            )
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

    if duration_ns:
        setattr(wrapped, "zero_before_ns", 0.0)
    return wrapped


@dataclass(frozen=True)
class MomentumCenterParticle:
    charge_native: float
    mass_amu: float
    g: float = 2.0
    reaction_mode: str = "off"
    reaction_window_ns: float | None = None
    reaction_derivative_method: str = "centered"
    reaction_source_order: int | None = None

    def __post_init__(self) -> None:
        self.length_time_particle()
        if self.reaction_source_order is not None and (
            type(self.reaction_source_order) is not int
            or not 2 <= self.reaction_source_order <= 7
            or self.reaction_mode not in FULL_DIPOLE_REACTION_MODES
            or self.reaction_derivative_method != "analytic"
        ):
            raise ValueError(
                "reaction_source_order must be an integer 2-7 and requires a full-dipole reaction with analytic derivatives"
            )
        if self.reaction_derivative_method not in (
            "centered",
            "backward",
            "boundary_aware",
            "analytic",
        ):
            raise ValueError(
                "Reaction derivative method must be analytic, centered, backward or boundary_aware"
            )
        if self.reaction_mode not in ("off", "charge_ll", *FULL_DIPOLE_REACTION_MODES):
            raise ValueError(
                "Requested reaction is not implemented; supported modes: off, zero-spin charge_ll, full_dipole_coupled, and legacy full_dipole_rr"
            )
        if self.reaction_window_ns is not None and (
            not np.isfinite(self.reaction_window_ns) or self.reaction_window_ns <= 0
        ):
            raise ValueError("Positive proper-time reaction_window_ns required")
        if (
            self.reaction_mode in FULL_DIPOLE_REACTION_MODES
            and self.reaction_derivative_method != "analytic"
            and self.reaction_window_ns is None
        ):
            raise ValueError(
                f"{self.reaction_mode} requires an explicit reaction_window_ns"
            )
        if (
            self.reaction_mode == "full_dipole_coupled"
            and self.reaction_derivative_method
            not in ("analytic", "centered", "boundary_aware")
        ):
            raise ValueError(
                "full_dipole_coupled requires analytic, centered or boundary_aware source derivatives"
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

    zero_until = getattr(provider, "zero_before_ns", None)
    taylor = getattr(provider, "taylor_response_length_time", None)
    if taylor is not None:
        supplied.taylor_response = taylor
    if zero_until is not None:
        setattr(supplied, "zero_before_length_time", c * zero_until)
    boundaries = getattr(provider, "reaction_startup_boundaries_ns", ())
    if boundaries:
        boundaries = np.asarray(boundaries) * c

        def domain(event, velocity):
            region = int(np.searchsorted(boundaries, event[0], side="right"))
            left = boundaries[region - 1] if region else -np.inf
            right = boundaries[region] if region < len(boundaries) else np.inf
            return (
                region,
                (event[0] - left) / velocity[0],
                (right - event[0]) / velocity[0],
            )

        supplied.reaction_fit_domain = domain
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
            "charge_ll requires exactly zero spin; use full_dipole_coupled for finite-spin reaction"
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
    state: np.ndarray,
    particle: MomentumCenterParticle,
    provider: NativeProvider,
    *,
    suppress_radiation: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Lab-time derivative plus explicitly separated native/length-time diagnostics."""
    scaled_state = _to_length_time(state)
    if particle.reaction_mode == "charge_ll" and np.any(scaled_state[8:]):
        raise ValueError("charge_ll requires exactly zero spin; no spin was discarded")
    supplied = _provider_length_time(provider)
    values = supplied(scaled_state[:4])
    if particle.reaction_mode in FULL_DIPOLE_REACTION_MODES and not suppress_radiation:
        evaluator = (
            evaluate_coupled_reaction
            if particle.reaction_mode == "full_dipole_coupled"
            else evaluate_reaction
        )
        rhs, diagnostic = evaluator(
            scaled_state,
            particle.length_time_particle(),
            supplied,
            window=(
                None
                if particle.reaction_window_ns is None
                else c * particle.reaction_window_ns
            ),
            method=particle.reaction_derivative_method,
        )
    else:
        rhs, diagnostic = model.evaluate(
            scaled_state, particle.length_time_particle(), lambda event: values
        )
    if particle.reaction_mode == "charge_ll" and not suppress_radiation:
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
    Analytical full-tensor response reuses strict compiled Taylor arithmetic
    when Numba is available; it is not the earlier sparse full-response kernel.
    """

    def __init__(
        self,
        history: FullDipoleHistory,
        charge_native: float,
        startup_duration_ns: float = 0.0,
        startup_smoothness: int = 5,
        *,
        one_sided_derivatives: bool = False,
        taylor_source_order: int | None = None,
    ) -> None:
        if history.speed_limit != c or not np.isfinite(charge_native):
            raise ValueError(
                "Native history requires speed_limit=C_MMNS and finite charge"
            )
        self.history = history
        self.charge_native = float(charge_native)
        self.startup_duration_ns = startup_duration_ns
        self.startup_smoothness = startup_smoothness
        if not isinstance(one_sided_derivatives, bool):
            raise ValueError("Explicit one-sided derivative policy required")
        self.one_sided_derivatives = one_sided_derivatives
        # Optional: build Taylor data from source derivatives through this
        # order at the retarded point (dipole one order lower), matching the
        # C4/C3 continuity of evolved history joins. Higher segment-polynomial
        # derivatives are interpolation artifacts that jump at every join and
        # grow as the recording step shrinks. Point responses are unchanged.
        self.taylor_source_order = taylor_source_order
        self.samples: list[tuple[float, float, float]] = []
        self.prepared = smooth_start_native(
            self._response, startup_duration_ns, startup_smoothness
        )
        if startup_duration_ns:
            self.zero_before_ns = 0.0
            self.reaction_startup_boundaries_ns = (0.0, startup_duration_ns)

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

    def taylor_response_length_time(self, events):
        """Analytical response along observer motion; strict at nonsmooth joins.

        Events and responses already use length-time units. No native-unit
        conversion is applied a second time by the provider adapter.
        """
        from math import comb
        from .full_dipole_response import response_taylor
        from .taylor_series import Series, to_series, from_series

        events = np.asarray(events, dtype=float)
        if events.shape != (5, 4) or not np.isfinite(events).all():
            raise ValueError("Finite order-four observer coefficients required")
        duration = c * self.startup_duration_ns
        if duration and events[0, 0] <= duration and self.startup_smoothness != 7:
            raise ValueError("Analytical reaction derivatives require C7 startup")
        if duration and events[0, 0] <= 0:
            return tuple(
                np.zeros((5,) + shape) for shape in ((4,), (4, 4), (4, 4), (4, 4, 4))
            )
        for segment in reversed(self.history.segments):

            def cone(time):
                x, _ = segment.sample(time)
                return events[0, 0] - c * time - np.linalg.norm(events[0, 1:] - x)

            if cone(segment.start) >= 0 and cone(segment.end) <= 0:
                # Declared coasting intervals have unique derivatives at their
                # mutual knots. Evolved C4/C3 history joins do not: fail there.
                inertial = (
                    self.history.inertial_until is not None
                    and segment.end < self.history.inertial_until
                )
                position, dipole = segment.position, segment.dipole / c
                allow_boundary = inertial or self.one_sided_derivatives
                if self.taylor_source_order is not None and not inertial:
                    from .full_dipole_response import local_source_polynomial

                    retarded = response(
                        events[0],
                        c * segment.start,
                        c * segment.duration,
                        position,
                        dipole,
                        charge=self.charge_native / c,
                        allow_boundary=allow_boundary,
                    )["retarded_time"]
                    point = (retarded - c * segment.start) / (c * segment.duration)
                    order = self.taylor_source_order
                    position = local_source_polynomial(position, point, order)
                    dipole = local_source_polynomial(dipole, point, order - 1)
                values = response_taylor(
                    events,
                    c * segment.start,
                    c * segment.duration,
                    position,
                    dipole,
                    charge=self.charge_native / c,
                    allow_boundary=allow_boundary,
                )
                break
        else:
            raise ValueError(
                "Retarded source time is outside published history; no extrapolation"
            )
        if not duration or events[0, 0] >= duration:
            return values
        s = Series(events[:, 0]) / duration
        z = 1 - s if s.c[0] > 0.5 else s
        ramp = sum(
            (-1) ** k * comb(7 + k, k) * comb(15, 7 - k) * z ** (8 + k)
            for k in range(8)
        )
        if s.c[0] > 0.5:
            ramp = 1 - ramp
        normalization = 15 * comb(14, 7)
        dr = normalization * (s * (1 - s)) ** 7 / duration
        ddr = normalization * 7 * (1 - 2 * s) * (s * (1 - s)) ** 6 / duration**2
        a, da, field, gradient = map(to_series, values)
        wedge = np.full((4, 4), Series.constant(0), dtype=object)
        wedge[0, :] += a
        wedge[:, 0] -= a
        new_da = ramp * da
        new_da[0] += dr * a
        new_gradient = ramp * gradient
        new_gradient[0] += dr * field + ddr * wedge
        for k in range(4):
            new_gradient[k, 0, :] += dr * da[k]
            new_gradient[k, :, 0] -= dr * da[k]
        return tuple(
            from_series(v)
            for v in (ramp * a, new_da, ramp * field + dr * wedge, new_gradient)
        )


class CombinedDipoleProvider:
    """Sum independent non-self sources before evaluating nonlinear response.

    Each source retains its own retarded time and derivative boundary checks.
    Summing forces instead would lose nonlinear cross-source contributions.
    """

    def __init__(self, providers: list[FullDipoleProvider]) -> None:
        if not providers:
            raise ValueError("At least one non-self source required")
        self.providers = providers
        # All sources are constructed with the same driven-start policy.
        first = providers[0]
        if any(
            p.startup_duration_ns != first.startup_duration_ns
            or p.startup_smoothness != first.startup_smoothness
            for p in providers
        ):
            raise ValueError("Combined sources require a common startup policy")
        for name in ("zero_before_ns", "reaction_startup_boundaries_ns"):
            if hasattr(first, name):
                setattr(self, name, getattr(first, name))

    @property
    def samples(self) -> list[tuple[float, float, float]]:
        return [sample for provider in self.providers for sample in provider.samples]

    @staticmethod
    def _sum(values: Iterator[NativeResponse]) -> NativeResponse:
        result = [value.copy() for value in next(values)]
        for contribution in values:
            for total, value in zip(result, contribution):
                total += value
        return result[0], result[1], result[2], result[3]

    def __call__(self, time_ns: float, position_mm: np.ndarray) -> NativeResponse:
        return self._sum(p(time_ns, position_mm) for p in self.providers)

    def taylor_response_length_time(self, events: np.ndarray) -> NativeResponse:
        return self._sum(p.taylor_response_length_time(events) for p in self.providers)


def _providers(
    payload: dict[str, Any],
    particles: list[MomentumCenterParticle],
    histories: list[FullDipoleHistory],
) -> list[FullDipoleProvider | CombinedDipoleProvider]:
    result: list[FullDipoleProvider | CombinedDipoleProvider] = []
    for observer in range(len(particles)):
        sources = [
            FullDipoleProvider(
                history,
                particle.charge_native,
                payload.get("startup_duration_ns", 0.0),
                payload.get("startup_smoothness", 5),
                one_sided_derivatives="internal_step_control" in payload,
                taylor_source_order=particles[observer].reaction_source_order,
            )
            for index, (particle, history) in enumerate(zip(particles, histories))
            if index != observer
        ]
        result.append(
            sources[0] if len(sources) == 1 else CombinedDipoleProvider(sources)
        )
    return result


def initialize_pair(
    particles: list[MomentumCenterParticle],
    states: Any,
    histories: list[FullDipoleHistory],
    *,
    startup_duration_ns: float = 0.0,
    startup_smoothness: int | None = None,
    inertial_prehistory: bool = False,
    preserve_source_increments: bool = False,
    integration_method: str = "rk4",
    internal_step_settings=None,
    stochastic_emission: Any = None,
    photon_transport: Any = None,
) -> dict[str, Any]:
    """Initialize the backward-compatible, exactly two-particle runner."""
    if len(particles) != 2 or len(histories) != 2:
        raise ValueError("Exactly two particles and histories required")
    return _initialize_particles(
        particles,
        states,
        histories,
        startup_duration_ns=startup_duration_ns,
        startup_smoothness=startup_smoothness,
        inertial_prehistory=inertial_prehistory,
        preserve_source_increments=preserve_source_increments,
        integration_method=integration_method,
        internal_step_settings=internal_step_settings,
        stochastic_emission=stochastic_emission,
        photon_transport=photon_transport,
    )


def _initialize_particles(
    particles: list[MomentumCenterParticle],
    states: Any,
    histories: list[FullDipoleHistory],
    *,
    startup_duration_ns: float = 0.0,
    startup_smoothness: int | None = None,
    inertial_prehistory: bool = False,
    preserve_source_increments: bool = False,
    integration_method: str = "rk4",
    internal_step_settings: Any = None,
    stochastic_emission: Any = None,
    photon_transport: Any = None,
    checkpoint_model: str = MODEL,
) -> dict[str, Any]:
    """Checkpoint accepted native states and their already prepared source past."""
    if len(particles) < 2 or len(histories) != len(particles):
        raise ValueError("At least two particles with one history each required")
    if integration_method not in ("rk4", "dop853"):
        raise ValueError("Integration method must be rk4 or dop853")
    if integration_method == "dop853" and not preserve_source_increments:
        raise ValueError("DOP853 requires preserved source increments")
    if not isinstance(inertial_prehistory, bool):
        raise ValueError("Explicit boolean inertial_prehistory required")
    if inertial_prehistory:
        from dataclasses import replace

        # Explicit caller declaration, validated against all supplied samples.
        # Existing checkpoints are never silently reclassified on restore.
        histories = [replace(h, inertial_until=float(h.time[-1])) for h in histories]
    payload = dict(
        model=checkpoint_model,
        units=UNITS,
        particles=[asdict(p) for p in particles],
        states=np.asarray(states, dtype=float).tolist(),
        histories=[h.to_checkpoint_payload() for h in histories],
        accepted_steps=0,
        startup_duration_ns=startup_duration_ns,
        startup_smoothness=(
            (
                7
                if any(p.reaction_mode in FULL_DIPOLE_REACTION_MODES for p in particles)
                else 5
            )
            if startup_smoothness is None
            else startup_smoothness
        ),
    )
    _restore(payload)
    if preserve_source_increments:
        payload = _initialize_source_precision(payload)
        _restore(payload)
    if integration_method != "rk4":
        payload["integration_method"] = integration_method
    if internal_step_settings is not None:
        from .pair_step_error import InternalStepSettings

        if not isinstance(internal_step_settings, InternalStepSettings):
            raise ValueError("Explicit InternalStepSettings required")
        payload["internal_step_control"] = asdict(internal_step_settings)
        _restore(payload)
    from .stochastic_emission import EmissionRuntime, StochasticEmissionConfig

    emission = StochasticEmissionConfig.from_dict(stochastic_emission)
    if emission.enabled:
        _validate_emission_pair(payload, emission)
        payload["stochastic_emission"] = EmissionRuntime(emission).to_payload()
    from .photon_transport import PhotonTransportConfig, PhotonTransportRuntime
    from .photon_transport_adapters import validate_transport_emission

    transport = PhotonTransportConfig.from_dict(photon_transport)
    validate_transport_emission(transport, emission)
    if transport.enabled:
        payload["photon_transport"] = PhotonTransportRuntime(transport).to_payload()
    return payload


def _preserved_dipole(reference, high, low, particle, provider, diagnostic, dref):
    from .preserved_source import dipole_parts, rounded_state

    state = rounded_state(reference, high, low)
    a, _, field, gradient = _provider_length_time(provider)(_to_length_time(state)[:4])
    if particle.reaction_mode in FULL_DIPOLE_REACTION_MODES:
        reaction = diagnostic["length_time"]["reaction"]
        field = field + reaction["self_field"]
        gradient = gradient + reaction["self_gradient"]
    return dipole_parts(
        reference, high, low, particle.length_time_particle(), a, field, gradient, dref
    )


def _initialize_source_precision(payload):
    """Explicit fresh-run policy; never manufacture lost checkpoint precision."""
    from dataclasses import replace
    from .preserved_source import rounded_state

    particles, states, histories = _restore(payload)
    if (
        payload["accepted_steps"]
        or np.any(states[:, 0])
        or not payload["startup_duration_ns"]
    ):
        raise ValueError(
            "Preserved source initialization requires a fresh driven start at zero"
        )
    if any(h.inertial_until != 0 or h.derivative_degree != 10 for h in histories):
        raise ValueError(
            "Preserved source initialization requires declared inertial degree-ten histories"
        )
    result = copy.deepcopy(payload)
    source = dict(
        format=_source_format(payload),
        reference=states.tolist(),
        high=np.zeros_like(states).tolist(),
        low=np.zeros_like(states).tolist(),
    )
    updated = []
    for i, (history, provider) in enumerate(
        zip(histories, _providers(payload, particles, histories))
    ):
        _, diagnostic = dynamics_native(states[i], particles[i], provider)
        reference = history.dipole[-1]
        high, low = _preserved_dipole(
            states[i],
            np.zeros(14),
            np.zeros(14),
            particles[i],
            provider,
            diagnostic,
            reference,
        )
        value = rounded_state(reference, high, low)
        updated.append(
            replace(
                history,
                segments=(),
                dipole=np.broadcast_to(value, history.dipole.shape),
                dipole_reference=reference,
                dipole_high=np.broadcast_to(high, history.dipole.shape),
                dipole_low=np.broadcast_to(low, history.dipole.shape),
            ).completed()
        )
    result["histories"] = [h.to_checkpoint_payload() for h in updated]
    result["source_precision"] = source
    return result


def _source_format(payload: dict[str, Any]) -> str:
    return (
        "preserved-particles-source-v1"
        if payload.get("model") == MULTIPARTICLE_MODEL
        else "preserved-pair-source-v1"
    )


def _source_components(payload, states, histories):
    from .preserved_source import rounded_state

    source = payload.get("source_precision")
    if source is None:
        if any(h.dipole_reference is not None for h in histories):
            raise ValueError(
                "Preserved pair histories require state increment metadata"
            )
        return None
    if not isinstance(source, dict) or source.get("format") != _source_format(payload):
        raise ValueError("Invalid preserved pair source format")
    values = tuple(
        np.asarray(source.get(key), dtype=float) for key in ("reference", "high", "low")
    )
    if any(v.shape != (len(histories), 14) or not np.isfinite(v).all() for v in values):
        raise ValueError("Invalid preserved pair state components")
    if not np.array_equal(rounded_state(*values), states) or any(
        h.dipole_reference is None for h in histories
    ):
        raise ValueError("Preserved pair state components disagree with checkpoint")
    return values


def _restore(
    payload: dict[str, Any],
) -> tuple[list[MomentumCenterParticle], np.ndarray, list[FullDipoleHistory]]:
    if (
        payload.get("model") not in (MODEL, MULTIPARTICLE_MODEL)
        or payload.get("units") != UNITS
    ):
        raise ValueError("Wrong nonlinear pair model or units")
    method = payload.get("integration_method", "rk4")
    if method not in ("rk4", "dop853"):
        raise ValueError("Integration method must be rk4 or dop853")
    if method == "dop853" and payload.get("source_precision") is None:
        raise ValueError("DOP853 requires preserved source increments")
    particles = [
        MomentumCenterParticle(
            **dict(
                dict(
                    reaction_derivative_method=(
                        "centered"
                        if p.get("reaction_mode") == "full_dipole_coupled"
                        else "backward"
                    )
                ),
                **p,
            )
        )
        for p in payload["particles"]
    ]
    if "internal_step_control" in payload:
        from .pair_step_error import InternalStepSettings

        InternalStepSettings.from_payload(payload["internal_step_control"])
        if payload.get("source_precision") is None or any(
            p.reaction_mode == "charge_ll" for p in particles
        ):
            raise ValueError(
                "Internal error control requires preserved RK4 or DOP853 with off or full-dipole reaction"
            )
    histories = [
        FullDipoleHistory.from_checkpoint_payload(h) for h in payload["histories"]
    ]
    states = np.asarray(payload["states"], dtype=float)
    count = payload["accepted_steps"]
    if (
        len(particles) < 2
        or (payload["model"] == MODEL and len(particles) != 2)
        or len(histories) != len(particles)
        or states.shape != (len(particles), 14)
        or not np.isfinite(states).all()
    ):
        raise ValueError("Invalid nonlinear pair checkpoint")
    if type(count) is not int or count < 0 or not np.all(states[:, 0] == states[0, 0]):
        raise ValueError("Invalid step count or unequal accepted times")
    for state, history in zip(states, histories):
        if (
            history.speed_limit != c
            or history.time[-1] != state[0]
            or not np.array_equal(history.position[-1], state[1:4])
        ):
            raise ValueError("History endpoint must match the native accepted state")
    _source_components(payload, states, histories)
    providers = _providers(payload, particles, histories)
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
    """Advance an existing pair checkpoint without changing its schema."""
    if payload.get("model") != MODEL:
        raise ValueError("Wrong nonlinear pair model or units")
    return _advance_particles(payload, width_ns, steps)


def _relative_constraint_residuals(
    state: np.ndarray,
    particle: MomentumCenterParticle,
    diagnostic: dict[str, Any],
) -> tuple[float, float]:
    """Dimensionless accepted-state checks in the simulation frame."""
    data = diagnostic["length_time"]
    momentum = np.asarray(data["kinetic_momentum"])
    spin = model.unpack(np.asarray(state[8:]) / c)
    mass_scale = max(particle.mass_amu**2, data["kinetic_mass"] ** 2)
    spin_scale = max(
        np.linalg.norm(spin) * np.linalg.norm(momentum), np.finfo(float).tiny
    )
    return (
        abs(float(data["mass_constraint"])) / mass_scale,
        float(np.linalg.norm(data["spin_constraint"])) / spin_scale,
    )


def _validate_emission_pair(
    payload: dict[str, Any], config: "StochasticEmissionConfig"
) -> None:
    if not config.enabled:
        return
    if np.any(np.asarray(payload["states"])[:, 8:]):
        raise ValueError("Stochastic pair recoil requires zero intrinsic spin")
    for history in payload["histories"]:
        if (
            history.get("geometry_reconstruction", "endpoint") != "endpoint"
            or history.get("dipole_reconstruction", "endpoint") != "endpoint"
            or history.get("derivative_sampling", "consecutive") != "consecutive"
        ):
            raise ValueError(
                "Stochastic pair recoil requires endpoint source histories "
                "with consecutive sampling"
            )
    if (
        payload.get("integration_method", "rk4") != "rk4"
        or "internal_step_control" in payload
    ):
        raise ValueError("Stochastic pair emission currently requires fixed RK4 steps")


def _pair_emission_indicators(
    state: np.ndarray, particle: MomentumCenterParticle, provider: NativeProvider
) -> dict[str, Any]:
    from .stochastic_emission import indicators_from_force

    _, data = dynamics_native(state, particle, provider, suppress_radiation=True)
    u = data["proper_velocity_mm_ns"] / c
    # F native and q native give dP/dtau in amu mm/ns^2.
    force = (
        particle.charge_native * provider(state[0], state[1:4])[2] @ (model.METRIC * u)
    )
    return indicators_from_force(
        data["kinetic_momentum_native"][1:], particle.mass_amu, force[1:] / u[0]
    )


def _advance_particles(
    payload: dict[str, Any],
    width_ns: float,
    steps: int = 1,
    *,
    passive_selector: Callable[[np.ndarray, int], frozenset[int]] | None = None,
    passive_constraint_budget_relative: float | None = None,
    passive_method: str = "midpoint",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Fixed lab-time RK4 or DOP853, with optional lower-order passive updates.

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
    emission_runtime = None
    if "stochastic_emission" in payload:
        from .stochastic_emission import (
            EmissionRuntime,
            StochasticEmissionConfig,
            is_electron,
            quantum_power_suppression,
        )

        emission_config = StochasticEmissionConfig.from_dict(
            payload["stochastic_emission"]["config"]
        )
        _validate_emission_pair(payload, emission_config)
        if emission_config.enabled:
            if passive_selector is not None:
                raise ValueError(
                    "Stochastic emission does not support passive pair updates"
                )
            emission_runtime = EmissionRuntime(
                emission_config, payload["stochastic_emission"]
            )
    transport_runtime = None
    if "photon_transport" in payload:
        from .photon_transport import PhotonTransportConfig, PhotonTransportRuntime
        from .photon_transport_adapters import (
            validate_transport_emission,
            commit_pair_transport,
        )

        transport_config = PhotonTransportConfig.from_dict(
            payload["photon_transport"]["config"]
        )
        if emission_runtime is None:
            raise ValueError(
                "Photon transport checkpoint requires enabled stochastic emission"
            )
        validate_transport_emission(transport_config, emission_config)
        if transport_config.enabled:
            transport_runtime = PhotonTransportRuntime(
                transport_config, payload["photon_transport"]
            )
    if passive_method not in ("midpoint", "rk3"):
        raise ValueError("Passive method must be midpoint or rk3")
    if passive_constraint_budget_relative is not None and (
        not np.isscalar(passive_constraint_budget_relative)
        or isinstance(passive_constraint_budget_relative, (bool, str))
        or not np.isfinite(passive_constraint_budget_relative)
        or passive_constraint_budget_relative <= 0
        or passive_selector is None
    ):
        raise ValueError("Positive passive constraint budget and selector required")
    source_components = _source_components(payload, states, histories)
    if source_components is not None:
        from .preserved_source import rounded_state

        source_reference, source_high, source_low = (
            v.copy() for v in source_components
        )
    ledger_keys = (
        "applied_impulse_native",
        "outward_radiation_native",
        "bound_rate_integral_native",
    )
    ledger = copy.deepcopy(
        payload.get(
            "reaction_ledger",
            [{key: [0.0] * 4 for key in ledger_keys} for _ in particles],
        )
    )
    if (
        not isinstance(ledger, list)
        or len(ledger) != len(particles)
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
    internal_settings = None
    if "internal_step_control" in payload:
        from .pair_step_error import InternalStepSettings

        internal_settings = InternalStepSettings.from_payload(
            payload["internal_step_control"]
        )
    dipole_ledger = copy.deepcopy(
        payload.get(
            "dipole_reaction_ledger",
            [
                dict(
                    applied_impulse_native=[0.0] * 4,
                    applied_torque_native=[0.0] * 6,
                    external_action_mass_exchange=0.0,
                    sectors={
                        key: dict(impulse_native=[0.0] * 4, torque_native=[0.0] * 6)
                        for key in ("q_squared", "qD", "D_squared")
                    },
                )
                for _ in particles
            ],
        )
    )
    if not isinstance(dipole_ledger, list) or len(dipole_ledger) != len(particles):
        raise ValueError("Invalid dipole reaction ledger")
    for row in dipole_ledger:
        if (
            not isinstance(row, dict)
            or set(row)
            != {
                "applied_impulse_native",
                "applied_torque_native",
                "external_action_mass_exchange",
                "sectors",
            }
            or np.shape(row["applied_impulse_native"]) != (4,)
            or np.shape(row["applied_torque_native"]) != (6,)
            or not np.isfinite(row["applied_impulse_native"]).all()
            or not np.isfinite(row["applied_torque_native"]).all()
            or not np.isscalar(row["external_action_mass_exchange"])
            or not np.isfinite(row["external_action_mass_exchange"])
            or not isinstance(row["sectors"], dict)
            or set(row["sectors"]) != {"q_squared", "qD", "D_squared"}
        ):
            raise ValueError("Invalid dipole reaction ledger")
        for sector in row["sectors"].values():
            if (
                not isinstance(sector, dict)
                or set(sector) != {"impulse_native", "torque_native"}
                or np.shape(sector["impulse_native"]) != (4,)
                or np.shape(sector["torque_native"]) != (6,)
                or not np.isfinite(sector["impulse_native"]).all()
                or not np.isfinite(sector["torque_native"]).all()
            ):
                raise ValueError("Invalid dipole reaction sector ledger")
    for step_index in range(steps):
        selected_passive = (
            passive_selector(states.copy(), payload["accepted_steps"] + step_index)
            if passive_selector is not None
            else None
        )
        if selected_passive is not None:
            if not isinstance(selected_passive, frozenset) or any(
                type(i) is not int or not 0 <= i < len(particles)
                for i in selected_passive
            ):
                raise ValueError("Passive indices must identify valid particle indices")
            if selected_passive and (
                payload.get("integration_method", "rk4") != "rk4"
                or ("internal_step_control" in payload and passive_method != "rk3")
                or (
                    passive_method != "rk3"
                    and any(p.reaction_mode != "off" for p in particles)
                )
            ):
                raise ValueError(
                    "Reduced updates require RK4; reaction and internal steps require passive_rk3"
                )
        providers = _providers(payload, particles, histories)
        trials, diagnostics, candidate_histories = [], [], []
        endpoint = states[0, 0] + width_ns
        for i, (state, particle, provider, history) in enumerate(
            zip(states, particles, providers, histories)
        ):
            emission_indicators = None
            reaction_suppressed = False
            quantum_reaction_scale = 1.0
            if emission_runtime is not None and is_electron(
                particle.mass_amu, particle.charge_native
            ):
                emission_indicators = _pair_emission_indicators(
                    state, particle, provider
                )
                reaction_suppressed = emission_runtime.observe(
                    str(i), emission_indicators, state[0]
                )
                if (
                    not reaction_suppressed
                    and emission_runtime.config.rate_model == "quantum_lcfa"
                ):
                    quantum_reaction_scale = quantum_power_suppression(
                        emission_indicators["chi"]
                    )

            def step_dynamics(
                value: np.ndarray,
                selected_particle: MomentumCenterParticle,
                selected_provider: NativeProvider,
            ) -> tuple[np.ndarray, dict[str, Any]]:
                if reaction_suppressed:
                    return dynamics_native(
                        value,
                        selected_particle,
                        selected_provider,
                        suppress_radiation=True,
                    )
                rate, data = dynamics_native(
                    value, selected_particle, selected_provider
                )
                if quantum_reaction_scale != 1.0:
                    free_rate, _ = dynamics_native(
                        value,
                        selected_particle,
                        selected_provider,
                        suppress_radiation=True,
                    )
                    rate = free_rate + quantum_reaction_scale * (rate - free_rate)
                    data["quantum_reaction_scale"] = quantum_reaction_scale
                return rate, data

            stage_diagnostics = []
            passive = selected_passive is not None and i in selected_passive
            forced_fallback = False
            weights: tuple[float, ...] = (1, 2, 2, 1)
            divisor = 6
            if source_components is not None:
                start_high = source_high[i].copy()
                start_low = source_low[i].copy()

            def finish_trial(
                candidate: np.ndarray | None,
                rates: tuple[np.ndarray, ...] | None = None,
                stage_weights: tuple[int, ...] = (1, 2, 2, 1),
                stage_divisor: int = 6,
            ) -> np.ndarray:
                nonlocal weights, divisor
                if rates is not None:
                    weights, divisor = stage_weights, stage_divisor
                if source_components is not None:
                    from .preserved_source import rk_increment, rounded_state

                    if rates is not None:
                        source_high[i], source_low[i] = rk_increment(
                            start_high,
                            start_low,
                            rates,
                            width_ns,
                            weights=stage_weights,
                            divisor=stage_divisor,
                        )
                    source_high[i, 0] = endpoint - source_reference[i, 0]
                    source_low[i, 0] = 0.0
                    candidate = rounded_state(
                        source_reference[i], source_high[i], source_low[i]
                    )
                assert candidate is not None
                candidate[0] = endpoint
                return candidate

            def rhs(value: np.ndarray) -> np.ndarray:
                rate, data = step_dynamics(value, particle, provider)
                stage_diagnostics.append(data["length_time"])
                return rate

            def rk4_from_first_two() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
                # Discard any reduced candidate stages before the RK4 retry.
                del stage_diagnostics[2:]
                k3 = rhs(state + width_ns * k2 / 2)
                k4 = rhs(state + width_ns * k3)
                return state + width_ns * (k1 + 2 * k2 + 2 * k3 + k4) / 6, k3, k4

            trial: np.ndarray | None = None
            trial_rates: tuple[np.ndarray, ...] | None = None
            trial_weights: tuple[int, ...] = (1, 2, 2, 1)
            trial_divisor = 6
            integrated_reaction, internal_statistics = None, None

            def internal_trial(method: str) -> np.ndarray:
                nonlocal integrated_reaction, internal_statistics, weights, divisor
                from .pair_internal_step import integrate_particle

                (
                    source_high[i],
                    source_low[i],
                    integrated_reaction,
                    internal_statistics,
                ) = integrate_particle(
                    source_reference[i],
                    start_high,
                    start_low,
                    width_ns,
                    particle,
                    provider,
                    internal_settings,
                    method=method,
                )
                weights, divisor = (), 1
                return finish_trial(None)

            def retry_rk4() -> np.ndarray:
                if internal_settings is not None:
                    return internal_trial("rk4")
                candidate, k3, k4 = rk4_from_first_two()
                return finish_trial(candidate, (k1, k2, k3, k4))

            if internal_settings is not None:
                from .preserved_adaptive import InternalAccuracyError

                try:
                    trial = internal_trial(
                        "rk3" if passive else payload.get("integration_method", "rk4")
                    )
                except (model.VelocityDomainError, InternalAccuracyError):
                    if not passive or passive_constraint_budget_relative is None:
                        raise
                    forced_fallback = True
                    trial = retry_rk4()
            elif payload.get("integration_method", "rk4") == "dop853":
                from .preserved_runge_kutta import dop853_step

                source_high[i], source_low[i], weights = dop853_step(
                    source_reference[i], source_high[i], source_low[i], width_ns, rhs
                )
                divisor = 1
            elif passive:
                k1 = rhs(state)
                k2 = rhs(state + width_ns * k1 / 2)
                if passive_method == "rk3":
                    try:
                        k3_reduced = rhs(state + width_ns * (2 * k2 - k1))
                        trial = state + width_ns * (k1 + 4 * k2 + k3_reduced) / 6
                        trial_rates = (k1, k2, k3_reduced)
                        trial_weights = (1, 4, 1)
                    except model.VelocityDomainError:
                        if passive_constraint_budget_relative is None:
                            raise
                        forced_fallback = True
                        trial, k3, k4 = rk4_from_first_two()
                        trial_rates = (k1, k2, k3, k4)
                else:
                    trial = state + width_ns * k2
                    trial_rates, trial_weights, trial_divisor = (k2,), (1,), 1
            else:
                k1 = rhs(state)
                k2 = rhs(state + width_ns * k1 / 2)
                trial, k3, k4 = rk4_from_first_two()
                trial_rates = (k1, k2, k3, k4)
            trial = finish_trial(trial, trial_rates, trial_weights, trial_divisor)
            try:
                rate, diagnostic = step_dynamics(trial, particle, provider)
            except model.VelocityDomainError:
                if (
                    not passive
                    or passive_constraint_budget_relative is None
                    or forced_fallback
                ):
                    raise
                forced_fallback = True
                trial = retry_rk4()
                rate, diagnostic = step_dynamics(trial, particle, provider)
            if passive and passive_constraint_budget_relative is not None:
                if forced_fallback:
                    candidate_mass, candidate_spin = None, None
                    fallback = True
                else:
                    candidate_mass, candidate_spin = _relative_constraint_residuals(
                        trial, particle, diagnostic
                    )
                    fallback = (
                        max(candidate_mass, candidate_spin)
                        > passive_constraint_budget_relative
                    )
                if fallback and not forced_fallback:
                    # Fixed stepping reuses the first two stages; adaptive
                    # stepping retries the entire interval from its original
                    # increments and replaces all reaction integrals.
                    trial = retry_rk4()
                    rate, diagnostic = step_dynamics(trial, particle, provider)
                accepted_mass, accepted_spin = _relative_constraint_residuals(
                    trial, particle, diagnostic
                )
                if (
                    max(accepted_mass, accepted_spin)
                    > passive_constraint_budget_relative
                ):
                    raise ValueError(
                        "RK4 fallback could not satisfy the passive constraint budget"
                    )
                diagnostic["passive_reduced"] = dict(
                    method=passive_method,
                    fallback_to_rk4=fallback,
                    candidate_mass_relative=candidate_mass,
                    candidate_spin_relative=candidate_spin,
                    accepted_mass_relative=accepted_mass,
                    accepted_spin_relative=accepted_spin,
                )
            velocity_before_jump = None
            if emission_indicators is not None:
                before = diagnostic["kinetic_momentum_native"][1:].copy()
                after = emission_runtime.emit(
                    str(i),
                    before,
                    particle.mass_amu,
                    particle.charge_native,
                    emission_indicators,
                    width_ns,
                    endpoint,
                    trial[1:4],
                )
                if not np.array_equal(before, after):
                    velocity_before_jump = rate[1:4].copy()
                    mc = particle.mass_amu * c
                    # Preserve the event's existing canonical potential offset.
                    trial[5:8] += after - before
                    trial[4] += (
                        np.hypot(mc, np.linalg.norm(after))
                        - diagnostic["kinetic_momentum_native"][0]
                    )
                    if source_components is not None:
                        from decimal import localcontext
                        from .preserved_source import decimal_array, split_array

                        with localcontext() as context:
                            context.prec = 80
                            source_high[i, 4:8], source_low[i, 4:8] = split_array(
                                decimal_array(trial[4:8])
                                - decimal_array(source_reference[i, 4:8])
                            )
                    rate, diagnostic = step_dynamics(trial, particle, provider)
                diagnostic["stochastic_emission"] = copy.deepcopy(
                    emission_runtime.particles[str(i)]
                )
            parts = None
            dipole = diagnostic["proper_dipole_native"]
            if source_components is not None:
                parts = _preserved_dipole(
                    source_reference[i],
                    source_high[i],
                    source_low[i],
                    particle,
                    provider,
                    diagnostic,
                    history.dipole_reference,
                )
                dipole = rounded_state(history.dipole_reference, *parts)
            jump_options = (
                {"velocity_before_jump": velocity_before_jump}
                if velocity_before_jump is not None
                else {}
            )
            candidate_histories.append(
                history.append(
                    endpoint,
                    trial[1:4],
                    rate[1:4],
                    dipole,
                    dipole_parts=parts,
                    **jump_options,
                )
            )
            trials.append(trial)
            if internal_statistics is not None:
                diagnostic["internal_step_control"] = internal_statistics
            diagnostics.append(diagnostic)
            if particle.reaction_mode != "off" and len(weights) != len(
                stage_diagnostics
            ):
                raise ValueError(
                    "Reaction integration requires one weight per accepted stage"
                )
            if particle.reaction_mode == "charge_ll" and not reaction_suppressed:
                for key, source in zip(
                    ledger_keys,
                    ("force", "outward_radiation_rate", "bound_momentum_rate"),
                ):
                    increment = (
                        width_ns
                        * quantum_reaction_scale
                        * c**2
                        / divisor
                        * sum(
                            weight
                            * np.asarray(d["reaction"][source])
                            / d["proper_velocity"][0]
                            for weight, d in zip(weights, stage_diagnostics)
                        )
                    )
                    ledger[i][key] = (np.asarray(ledger[i][key]) + increment).tolist()
            if (
                particle.reaction_mode in FULL_DIPOLE_REACTION_MODES
                and not reaction_suppressed
            ):
                if integrated_reaction is not None:
                    from .nonlinear_pair_adaptive import _add_ledgers

                    dipole_ledger[i] = _add_ledgers(
                        dipole_ledger[i], integrated_reaction
                    )
                row = dipole_ledger[i]
                for weight, data in zip(weights, stage_diagnostics):
                    factor = (
                        quantum_reaction_scale
                        * weight
                        * width_ns
                        * c
                        / (divisor * data["proper_velocity"][0])
                    )
                    reaction = data["reaction"]
                    for target, source in (
                        ("applied_impulse_native", "force"),
                        ("applied_torque_native", "torque"),
                    ):
                        value = (
                            reaction[source]
                            if source == "force"
                            else model.pack(reaction[source])
                        )
                        row[target] = (
                            np.asarray(row[target]) + c * factor * value
                        ).tolist()
                    row["external_action_mass_exchange"] += (
                        factor * reaction["external_action_mass_rate"]
                    )
                    for key, sector in reaction["sectors"].items():
                        for target, source in (
                            ("impulse_native", "force"),
                            ("torque_native", "torque"),
                        ):
                            value = (
                                sector[source]
                                if source == "force"
                                else model.pack(sector[source])
                            )
                            row["sectors"][key][target] = (
                                np.asarray(row["sectors"][key][target])
                                + c * factor * value
                            ).tolist()
        if transport_runtime is not None:
            commit_pair_transport(
                transport_runtime,
                emission_runtime,
                payload,
                particles,
                histories,
                trials,
                diagnostics,
                candidate_histories,
                (
                    (source_reference, source_high, source_low)
                    if source_components is not None
                    else None
                ),
            )
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
        model=payload["model"],
        units=UNITS,
        particles=[asdict(p) for p in particles],
        states=states.tolist(),
        histories=[h.to_checkpoint_payload() for h in histories],
        accepted_steps=payload["accepted_steps"] + steps,
        startup_duration_ns=payload.get("startup_duration_ns", 0.0),
        startup_smoothness=payload.get("startup_smoothness", 5),
    )
    if "dipole_budget_policy" in payload:
        from copy import deepcopy

        result["dipole_budget_policy"] = deepcopy(payload["dipole_budget_policy"])
    if "reaction_ledger" in payload or any(
        p.reaction_mode == "charge_ll" for p in particles
    ):
        result["reaction_ledger"] = ledger
    if source_components is not None:
        result["source_precision"] = dict(
            format=_source_format(payload),
            reference=source_reference.tolist(),
            high=source_high.tolist(),
            low=source_low.tolist(),
        )
    if "integration_method" in payload:
        result["integration_method"] = payload["integration_method"]
    if "internal_step_control" in payload:
        result["internal_step_control"] = copy.deepcopy(
            payload["internal_step_control"]
        )
    if "dipole_reaction_ledger" in payload or any(
        p.reaction_mode in FULL_DIPOLE_REACTION_MODES for p in particles
    ):
        result["dipole_reaction_ledger"] = dipole_ledger
    if emission_runtime is not None:
        result["stochastic_emission"] = emission_runtime.to_payload()
    elif "stochastic_emission" in payload:
        result["stochastic_emission"] = copy.deepcopy(payload["stochastic_emission"])
    if transport_runtime is not None:
        result["photon_transport"] = transport_runtime.to_payload()
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
    if max_halvings and "internal_step_control" in payload:
        raise ValueError(
            "Internal error control keeps source-recording cadence fixed; "
            "combining it with source-position step halving is not supported"
        )
    try:
        return advance_pair(payload, width_ns, 1)
    except SourcePositionError:
        if max_halvings == 0:
            raise
        middle, first = advance_pair_refined(payload, width_ns / 2, max_halvings - 1)
        final, second = advance_pair_refined(middle, width_ns / 2, max_halvings - 1)
        return final, first + second
