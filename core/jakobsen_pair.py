"""Experimental reciprocal Jakobsen pair with frozen source intervals.

Both roles see the same accepted history. Steps requiring a future source
state are rejected, not extrapolated. This is not the CLI/GUI pair runner.
The initial coasting history is prescribed cold-start data, not a claimed
past solution of the mutually interacting equations.
"""

from dataclasses import asdict
from types import SimpleNamespace
import numpy as np

from .constants import C_MMNS as c
from .jakobsen_step import (
    JakobsenParticle,
    initial_canonical_state,
    canonical_dynamics,
    canonical_constraint_residual,
)
from .retarded_fields import (
    _quintic_position_coefficients_mm,
    _segment_beta_bernstein_bound,
)
from .dipole_hertz_jet import (
    _spin_coefficients,
    polynomial_dipole_hertz_response_jet_native,
)
from .retarded_potential_directional_jet import (
    quintic_charge_response_directional_gradient_native,
)
from .dipole_hertz_jet_numba import (
    quintic_dipole_hertz_sparse_potential_rate_strict_serial,
    quintic_dipole_hertz_response_coefficients_strict_serial,
)
from .charge_response_jet_numba import (
    quintic_charge_response_coefficients_strict_serial,
)
from .exact_retarded_numba import evaluate_source_roots_exact_serial
from .antisymmetric_response_rfs import (
    materialize_antisymmetric_response_native,
    materialize_partial_antisymmetric_response_native,
)


MODEL = "experimental_jakobsen_reciprocal_frozen_v1"


class FrozenSource:
    """Per-interval derivatives; adding a knot never rewrites old polynomials."""

    def __init__(self, particle, rows, segments, spin_segments):
        self.particle = particle
        self.rows = [np.asarray(row, dtype=float).copy() for row in rows]
        self.segments = [np.asarray(x, dtype=float).copy() for x in segments]
        self.spin_segments = [np.asarray(x, dtype=float).copy() for x in spin_segments]
        if (
            not self.rows
            or len(self.segments) != len(self.rows) - 1
            or len(self.spin_segments) != len(self.segments)
            or any(r.shape != (16,) or not np.isfinite(r).all() for r in self.rows)
            or any(x.shape != (6, 3) or not np.isfinite(x).all() for x in self.segments)
            or any(
                x.shape != (4, 3) or not np.isfinite(x).all()
                for x in self.spin_segments
            )
            or any(np.dot(r[4:7], r[4:7]) >= 1 for r in self.rows)
        ):
            raise ValueError("Invalid frozen source checkpoint")
        self.refresh()
        if np.any(self.durations <= 0):
            raise ValueError("Source times must increase")
        for i in range(len(self.durations)):
            self.check_segment(self.coefficients[i], self.durations[i])

    @staticmethod
    def check_segment(coefficient, duration):
        source = SimpleNamespace(
            segment_duration_ns=np.array([duration]),
            position_coefficients_mm=np.asarray(coefficient)[None, :],
        )
        if _segment_beta_bernstein_bound(source, 0) >= 1:
            raise ValueError("Source interpolation cannot be certified timelike")

    def refresh(self):
        self.times = np.array([row[0] for row in self.rows])
        self.positions = np.array([row[1:4] for row in self.rows])
        self.durations = np.diff(self.times)
        self.coefficients = np.asarray(self.segments)

    def append(self, row):
        row = np.asarray(row, dtype=float)
        start = self.rows[-1]
        duration = row[0] - start[0]
        if row.shape != (16,) or not np.isfinite(row).all() or duration <= 0:
            raise ValueError("Finite forward source row required")
        _, coefficient = _quintic_position_coefficients_mm(
            np.array([start[0], row[0]]),
            np.array([start[1:4], row[1:4]]),
            np.array([start[4:7], row[4:7]]),
            np.array([start[7:10], row[7:10]]),
        )
        spin = _spin_coefficients(
            start[10:13], row[10:13], start[13:16], row[13:16], duration
        )
        self.check_segment(coefficient[0], duration)
        self.segments.append(coefficient[0])
        self.spin_segments.append(spin)
        self.rows.append(row.copy())
        self.refresh()

    def payload(self):
        return dict(
            rows=[r.tolist() for r in self.rows],
            segments=[x.tolist() for x in self.segments],
            spin_segments=[x.tolist() for x in self.spin_segments],
        )


class RetardedSourceProvider:
    def __init__(self, source, *, sparse=True):
        self.source = source
        self.sparse = sparse
        self.calls = 0
        self.minimum_history_margin_ns = float("inf")

    def _segment(self, t, x):
        """Select one accepted smooth interval; never bridge a derivative jump."""
        source = self.source
        x = np.asarray(x, dtype=float)
        batch = evaluate_source_roots_exact_serial(
            source.times,
            source.positions,
            source.durations,
            source.coefficients,
            False,
            np.array([t]),
            x[None, :],
            1e-13,
            96,
        )
        if batch[0][0] != 0:
            raise ValueError(
                "Retarded source history unavailable; reduce step or extend prehistory"
            )
        root = float(batch[2][0])
        self.calls += 1
        self.minimum_history_margin_ns = min(
            self.minimum_history_margin_ns, source.times[-1] - root
        )
        index = int(
            np.clip(
                np.searchsorted(source.times, root, side="right") - 1,
                0,
                len(source.durations) - 1,
            )
        )
        fraction = (root - source.times[index]) / source.durations[index]
        if not 1e-8 < fraction < 1 - 1e-8:
            raise ValueError("Retarded root at nonsmooth frozen interval boundary")
        return index, root

    def gradient_proper_rate(self, t, x, u):
        """Analytical derivative of the selected interval, along observer motion.

        This experimental reference path differentiates potentials directly.
        It does not sample displaced fields, refit history, or smooth a join.
        It is deliberately not a claim that adjacent high derivatives agree.
        """
        index, root = self._segment(t, x)
        source = self.source
        args = dict(
            observer_time_ns=t,
            observer_position_mm=x,
            segment_start_time_ns=source.times[index],
            segment_duration_ns=source.durations[index],
            position_coefficients_mm=source.coefficients[index],
            retarded_time_ns=root,
        )
        rate = quintic_charge_response_directional_gradient_native(
            **args,
            charge_native=source.particle.charge_native,
            four_velocity_mm_ns=u,
        ).partial_antisymmetric_response_along_velocity.copy()
        if np.any(source.spin_segments[index]):
            magnetic = polynomial_dipole_hertz_response_jet_native(
                **args,
                magnetic_moment_native=(
                    source.particle.g
                    * source.particle.charge_native
                    / (2 * source.particle.mass_amu * c)
                ),
                rest_spin_coefficients=source.spin_segments[index],
                preserved_rest_spin_magnitude=None,
                observer_four_velocity_mm_ns=u,
            )
            rate += magnetic.partial_antisymmetric_response_along_velocity
        return materialize_partial_antisymmetric_response_native(rate)

    def __call__(self, t, x):
        index, root = self._segment(t, x)
        source = self.source
        x = np.asarray(x, dtype=float)
        base = (
            t,
            x,
            source.times[index],
            source.durations[index],
            source.coefficients[index],
            root,
        )
        a, da, packed, gradient, kappa, _ = (
            quintic_charge_response_coefficients_strict_serial(
                base[0], base[1], source.particle.charge_native, *base[2:]
            )
        )
        if kappa <= 1e-14:
            raise ValueError("Singular retarded charge response")
        field = materialize_antisymmetric_response_native(packed)
        df = materialize_partial_antisymmetric_response_native(gradient)
        # Spin coefficients are physical angular momentum; the multiplier
        # converts them into magnetic moment, without normalizing the spin.
        moment_coefficient = (
            source.particle.g
            * source.particle.charge_native
            / (2 * source.particle.mass_amu * c)
        )
        if np.any(source.spin_segments[index]):
            arguments = (
                t,
                x,
                moment_coefficient,
                source.times[index],
                source.durations[index],
                source.coefficients[index],
                source.spin_segments[index],
                False,
                0.0,
                root,
            )
            if self.sparse:
                status, ma, mf, mdf, _, _, mda = (
                    quintic_dipole_hertz_sparse_potential_rate_strict_serial(
                        *arguments, np.zeros(4)
                    )
                )
                if status:
                    raise ValueError("Invalid sparse magnetic source response")
                mf = materialize_antisymmetric_response_native(mf)
                mdf = materialize_partial_antisymmetric_response_native(mdf)
            else:
                result = quintic_dipole_hertz_response_coefficients_strict_serial(
                    *arguments
                )
                if result[0]:
                    raise ValueError("Invalid dense magnetic source response")
                # Native dense kernel tuple layout is checked in unit tests.
                _, _, ma, mda, mf, mdf, *_ = result
            a = a + ma
            da = da + mda
            field = field + mf
            df = df + mdf
        return a, da, field, df


def source_row(state, particle, provider):
    rhs, response, u = canonical_dynamics(state, particle=particle, provider=provider)
    gamma = u[0] / c
    beta = u[1:] / u[0]
    rest = state[8:11]
    acceleration = response.four_force / particle.mass_amu
    beta_prime = (acceleration[1:] - beta * acceleration[0]) / (gamma**2 * c**2)
    rest_rate = rhs[8:] / gamma
    return np.r_[state[0], state[1:4], beta, beta_prime, rest, rest_rate]


def _lab_rk4(state, width, particle, provider):
    """Fourth-order lab-time integration of the existing proper-time equation.

    Higher accuracy of endpoint positions is needed before differentiating
    interpolated source histories; this does not change the force equation.
    """

    def rhs(y):
        state = y[:11]
        result, response, u = canonical_dynamics(
            state, particle=particle, provider=provider
        )
        gamma = u[0] / c
        return np.r_[result / result[0], response.four_force / gamma]

    state = np.r_[state, np.zeros(4)]
    k1 = rhs(state)
    k2 = rhs(state + 0.5 * width * k1)
    k3 = rhs(state + 0.5 * width * k2)
    k4 = rhs(state + width * k3)
    result = state + width * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    result[0] = state[0] + width
    return result[:11], result[11:]


def _mechanical_momentum(sources):
    total = np.zeros(4)
    for source in sources:
        beta = source.rows[-1][4:7]
        total += (
            source.particle.mass_amu * c * np.r_[1.0, beta] / np.sqrt(1 - beta @ beta)
        )
    return total


def initialize_pair(
    *, particles, positions_mm, betas, rest_spins_native, prehistory_ns, sparse=True
):
    if len(particles) != 2:
        raise ValueError("Exactly two particles required")
    if not np.isfinite(prehistory_ns) or prehistory_ns <= 0:
        raise ValueError("Positive finite coasting prehistory required")
    sources = []
    for particle, x, beta, spin in zip(
        particles, positions_mm, betas, rest_spins_native
    ):
        x = np.asarray(x, dtype=float)
        beta = np.asarray(beta, dtype=float)
        spin = np.asarray(spin, dtype=float)
        if (
            x.shape != (3,)
            or beta.shape != (3,)
            or spin.shape != (3,)
            or beta @ beta >= 1
        ):
            raise ValueError("Finite timelike initial data required")
        first = np.r_[
            -prehistory_ns,
            x - c * beta * prehistory_ns,
            beta,
            np.zeros(3),
            spin,
            np.zeros(3),
        ]
        source = FrozenSource(particle, [first], [], [])
        source.append(np.r_[0.0, x, beta, np.zeros(3), spin, np.zeros(3)])
        sources.append(source)
    providers = [
        RetardedSourceProvider(sources[1 - i], sparse=sparse) for i in range(2)
    ]
    states = []
    for i, particle in enumerate(particles):
        beta = np.asarray(betas[i])
        u = c * np.r_[1.0, beta] / np.sqrt(1 - beta @ beta)
        states.append(
            initial_canonical_state(
                time_ns=0,
                position_mm=positions_mm[i],
                four_velocity_mm_ns=u,
                rest_spin_angular_momentum=rest_spins_native[i],
                particle=particle,
                provider=providers[i],
            )
        )
    # Set right-hand initial derivatives only. The already frozen coasting
    # interval keeps its left-hand derivatives; there is no prehistory rewrite.
    rows = [source_row(states[i], particles[i], providers[i]) for i in range(2)]
    for source, row in zip(sources, rows):
        source.rows[-1] = row
    return dict(
        model=MODEL,
        particles=[asdict(p) for p in particles],
        sparse=bool(sparse),
        states=[s.tolist() for s in states],
        sources=[s.payload() for s in sources],
        accepted=0,
    )


def advance_pair(payload, width_ns, steps=1):
    """Pure batch: failures and rejected trials leave the input payload unchanged."""
    if (
        payload.get("model") != MODEL
        or not np.isfinite(width_ns)
        or width_ns <= 0
        or isinstance(steps, bool)
        or not isinstance(steps, int)
        or steps < 1
    ):
        raise ValueError("Valid pair checkpoint and positive advance required")
    particles = [JakobsenParticle(**p) for p in payload["particles"]]
    if len(particles) != 2:
        raise ValueError("Exactly two particles required")
    sources = [
        FrozenSource(p, **data) for p, data in zip(particles, payload["sources"])
    ]
    providers = [
        RetardedSourceProvider(sources[1 - i], sparse=payload["sparse"])
        for i in range(2)
    ]
    states = [np.asarray(s, dtype=float).copy() for s in payload["states"]]
    records = []
    if (
        len(states) != 2
        or len(sources) != 2
        or any(s.shape != (11,) or not np.isfinite(s).all() for s in states)
        or states[0][0] != states[1][0]
        or any(source.times[-1] != state[0] for source, state in zip(sources, states))
    ):
        raise ValueError("Checkpoint states and histories need a common accepted time")
    mechanical_initial = np.asarray(
        payload.get("mechanical_initial", _mechanical_momentum(sources))
    )
    mechanical_impulse = np.asarray(
        payload.get("mechanical_impulse", np.zeros(4))
    ).copy()
    if any(
        x.shape != (4,) or not np.isfinite(x).all()
        for x in (mechanical_initial, mechanical_impulse)
    ):
        raise ValueError("Invalid mechanical-accounting checkpoint")
    for _ in range(steps):
        trials = [
            _lab_rk4(states[i], width_ns, particles[i], providers[i]) for i in range(2)
        ]
        candidates = [trial[0] for trial in trials]
        mechanical_impulse += trials[0][1] + trials[1][1]
        rows = [source_row(candidates[i], particles[i], providers[i]) for i in range(2)]
        constraints = [
            canonical_constraint_residual(
                candidates[i], particle=particles[i], provider=providers[i]
            )
            for i in range(2)
        ]
        # Both trials and both endpoint derivatives are complete before either
        # accepted source history is changed: no order-dependent feedback.
        for source, row in zip(sources, rows):
            source.append(row)
        states = candidates
        records.append(
            dict(
                time_ns=float(states[0][0]),
                distance_mm=float(np.linalg.norm(states[0][1:4] - states[1][1:4])),
                canonical_residual=constraints,
                mechanical_change=(
                    _mechanical_momentum(sources) - mechanical_initial
                ).tolist(),
                mechanical_impulse_residual=(
                    _mechanical_momentum(sources)
                    - mechanical_initial
                    - mechanical_impulse
                ).tolist(),
                minimum_history_margin_ns=min(
                    p.minimum_history_margin_ns for p in providers
                ),
            )
        )
    result = dict(
        model=MODEL,
        particles=payload["particles"],
        sparse=payload["sparse"],
        states=[s.tolist() for s in states],
        sources=[s.payload() for s in sources],
        accepted=payload["accepted"] + steps,
        mechanical_initial=mechanical_initial.tolist(),
        mechanical_impulse=mechanical_impulse.tolist(),
    )
    return result, records
