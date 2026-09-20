"""Experimental reduced-order full-dipole self-reaction and ordinary predictor.

Full spin is retained. Only the reaction is reduced in order: local source
derivatives are computed on an auxiliary reaction-off trajectory, integrated
locally from the current state. No future source history is queried; a
forward auxiliary trajectory still uses the frozen retarded provider.
The time-symmetric singular self-field is excluded by prescription, not solved.
"""

from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

from . import momentum_center as model
from .full_dipole_self_field import regular_self_field, reaction_sectors


def inherit_reaction_domain(provider, wrapped):
    """Keep boundary information when adding the trial self-field."""
    domain = getattr(provider, "reaction_fit_domain", None)
    if domain is not None:
        wrapped.reaction_fit_domain = domain
    return wrapped


def with_local_self_field(provider, field, gradient, origin):
    """Same local linear self-field for numeric and analytical evaluation."""

    def supplied(event):
        a, da, f, df = provider(event)
        return (
            a,
            da,
            f + field + np.einsum("k,kij->ij", event - origin, gradient),
            df + gradient,
        )

    taylor = getattr(provider, "taylor_response", None)
    if taylor is not None:

        def expanded(events):
            a, da, f, df = (np.array(v, copy=True) for v in taylor(events))
            delta = np.array(events, copy=True)
            delta[0] -= origin
            f += np.einsum("nk,kij->nij", delta, gradient)
            f[0] += field
            df[0] += gradient
            return a, da, f, df

        supplied.taylor_response = expanded
    return inherit_reaction_domain(provider, supplied)


def ordinary_source_coefficients(
    state: np.ndarray,
    particle: model.Particle,
    provider: model.Provider,
    *,
    window: float,
    method: str = "centered",
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Order-four coefficients from a local ordinary-motion predictor.

    window is a caller-specified proper length-time, NOT a physical radius.
    Eleven Chebyshev samples fit degree eight to u and D. Centered sampling
    uses +/- window; legacy backward sampling uses [-window,0]. Sampling sensitivity
    must be checked by changing window; this routine does not claim an error
    bound on high derivatives from the ODE state tolerance.

    boundary_aware is an opt-in diagnostic: it integrates kinetic increments,
    compensates velocity differences, and avoids declared startup boundaries.
    It does not yet repair general history-join sampling or guarantee accurate
    one-sided fourth derivatives near a ramp endpoint. Not the default.
    """
    if method == "analytic":
        from .momentum_center_derivatives import source_coefficients

        return source_coefficients(state, particle, provider)
    if window is None or not np.isfinite(window) or window <= 0:
        raise ValueError("Positive self-reaction derivative window required")
    if method not in ("centered", "backward", "boundary_aware"):
        raise ValueError(
            "Reaction derivative method must be centered, backward or boundary_aware"
        )
    state = np.asarray(state, dtype=float)
    _, center = model.evaluate(state, particle, provider)
    nodes = np.cos(np.arange(11) * np.pi / 10)
    if method == "backward":
        nodes = (nodes - 1) / 2
    else:
        nodes[5] = 0.0
    # A declared zero-field pre-start interval has exactly coasting ordinary
    # motion. Do not fit across its boundary and manufacture self-reaction.
    zero_until = getattr(provider, "zero_before_length_time", None)
    if zero_until is not None and state[0] <= zero_until:
        u = np.zeros((5, 4))
        d = np.zeros((5, 4, 4))
        u[0], d[0] = center["proper_velocity"], center["proper_dipole"]
        return (
            u,
            d,
            dict(
                proper_window=window,
                derivative_method=method,
                fit_residuals=[0.0, 0.0],
                ordinary_evaluations=1,
                ordinary_velocity=u[0],
                coasting_control=True,
            ),
        )
    sampling_side = "backward" if method == "backward" else "centered"
    domain = getattr(provider, "reaction_fit_domain", None)
    if method == "boundary_aware" and domain is not None:
        velocity = center["proper_velocity"]
        region, before, after = domain(state[:4], velocity)
        if min(before, after) < window:
            if before >= window and before >= after:
                sampling_side = "backward"
                nodes = (np.cos(np.arange(11) * np.pi / 10) - 1) / 2
            elif after >= window:
                sampling_side = "forward"
                nodes = (np.cos(np.arange(11) * np.pi / 10) + 1) / 2
            else:
                raise ValueError(
                    "Requested reaction window does not fit a smooth startup region"
                )
        original_provider = provider

        def guarded(event):
            # Initial-velocity distances select a side; every actual sampled
            # event is checked too, including adaptive ODE substeps.
            if domain(event, velocity)[0] != region:
                raise ValueError(
                    "Reaction derivative sampling crossed a startup boundary"
                )
            return original_provider(event)

        provider = guarded
    # Integrate changes, avoiding large absolute position/momentum offsets in
    # the adaptive solver's absolute error control.
    predictor_state = state.copy()
    evaluate_predictor = model.evaluate
    if method == "boundary_aware":
        predictor_state[4:8] = center["kinetic_momentum"]
        evaluate_predictor = model.evaluate_kinetic
    deltas = np.zeros((len(nodes), 14))
    evaluations = 0
    for sign in (-1, 1):
        indices = np.flatnonzero(sign * nodes > 0)
        indices = indices[np.argsort(abs(nodes[indices]))]
        if not len(indices):
            continue
        solution = solve_ivp(
            lambda t, delta: evaluate_predictor(
                predictor_state + delta, particle, provider
            )[0],
            (0.0, sign * window),
            np.zeros(14),
            t_eval=window * nodes[indices],
            method="DOP853",
            rtol=3e-13,
            atol=3e-15,
            max_step=window / 8,
        )
        if not solution.success:
            raise ValueError(
                "Ordinary self-reaction predictor failed: " + solution.message
            )
        evaluations += solution.nfev
        deltas[indices] = solution.y.T
    responses = [provider((state + delta)[:4]) for delta in deltas]
    values = [
        evaluate_predictor(predictor_state + delta, particle, lambda event: response)[1]
        for delta, response in zip(deltas, responses)
    ]
    centered_changes = {}
    if method != "backward":
        a0, _, f0, df0 = provider(state[:4])
        p0, u0, s0 = (
            center["kinetic_momentum"],
            center["proper_velocity"],
            model.unpack(state[8:]),
        )
        mass2 = model.dot(p0, p0)
        coefficient = particle.bare_mass * particle.coupling
        correction0 = model.velocity_direction(p0, s0, f0, df0, particle)[-1]
        eta0 = 2 * model.dot(p0, correction0) + mass2 * model.dot(
            correction0, correction0
        )
        radius0 = mass2 * (1 + eta0)
        uc, dc = [], []
        for delta, response, data in zip(deltas, responses, values):
            # Never obtain the small kinetic-energy change by subtracting
            # two momenta near the rest mass. Keep the integrated increment.
            dp = (
                delta[4:8]
                if method == "boundary_aware"
                else delta[4:8] - particle.charge * (response[0] - a0)
            )
            dm2 = 2 * model.dot(p0, dp) + model.dot(dp, dp)
            ds = model.unpack(delta[8:])
            p = p0 + dp
            correction = model.velocity_direction(
                p, s0 + ds, response[2], response[3], particle
            )[-1]
            eta = 2 * model.dot(p, correction) + (mass2 + dm2) * model.dot(
                correction, correction
            )
            dradius = dm2 * (1 + eta) + mass2 * (eta - eta0)
            root0, root = np.sqrt(radius0), np.sqrt(radius0 + dradius)
            # k=K/sqrt(p^2*(1+eta)); rationalize the difference of roots.
            dk = -coefficient * dradius / (root * root0 * (root + root0))
            dc.append((center["proper_dipole_coupling"] + dk) * ds + dk * s0)
            if method == "boundary_aware":
                # u=(p+p^2*correction)/sqrt(p^2*(1+eta)). Form its
                # increment from the integrated dp instead of subtracting
                # two nearly equal relativistic velocities. Rationalize
                # the normalization difference, as for the dipole above.
                numerator_change = (
                    dp + dm2 * correction + mass2 * (correction - correction0)
                )
                du = numerator_change / root - u0 * dradius / (root * (root + root0))
            else:
                du = data["proper_velocity"] - u0
            du[0] = (2 * u0[1:] @ du[1:] + du[1:] @ du[1:]) / (
                data["proper_velocity"][0] + u0[0]
            )
            uc.append(du)
        centered_changes = dict(
            proper_velocity=np.asarray(uc), proper_dipole=np.asarray(dc)
        )
    coefficients = []
    residuals = []
    for key in ("proper_velocity", "proper_dipole"):
        samples = np.asarray([v[key] for v in values])
        changes = (
            centered_changes[key] if method != "backward" else samples - center[key]
        )
        # Fix the known center value, fit only its changes.
        matrix = np.polynomial.polynomial.polyvander(nodes, 8)[:, 1:]
        higher = np.linalg.lstsq(matrix, changes.reshape(11, -1), rcond=None)[0]
        fitted = np.concatenate((center[key].reshape(1, -1), higher))
        residuals.append(
            float(np.linalg.norm(matrix @ higher - changes.reshape(11, -1)))
        )
        result = fitted[:5] / window ** np.arange(5)[:, None]
        coefficients.append(result.reshape((5,) + samples.shape[1:]))
    # u0 is not independent: u.u=1 fixes its derivatives algebraically.
    # Fitting gamma~1 directly loses its tiny changes to rounding, then
    # differentiation amplifies that noise. This reconstructs derivatives,
    # not the particle state or a changed velocity law.
    velocity = coefficients[0]
    for n in range(1, 5) if method != "backward" else ():
        spatial = sum(velocity[k, 1:] @ velocity[n - k, 1:] for k in range(n + 1))
        temporal = sum(velocity[k, 0] * velocity[n - k, 0] for k in range(1, n))
        velocity[n, 0] = (spatial - temporal) / (2 * velocity[0, 0])
    return (
        coefficients[0],
        coefficients[1],
        dict(
            proper_window=window,
            derivative_method=method,
            sampling_side=sampling_side,
            fit_residuals=residuals,
            ordinary_evaluations=evaluations + len(values) + 1,
            ordinary_velocity=center["proper_velocity"],
        ),
    )


def evaluate_with_self_field(
    state: np.ndarray,
    particle: model.Particle,
    provider: model.Provider,
    self_field: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Couple the regular field through the full nonlinear velocity solve.

    Stored P remains p+q A_external. The constitutive D=kS rule is retained,
    but the external-only action mass is allowed to exchange energy with the
    regular field. Its actual rate is reported, never projected away. This
    is an explicit experimental constitutive extension, not proven full
    renormalized particle-plus-field conservation.
    """
    a, da, f, df = provider(state[:4])
    total = (a, da, f + self_field["field"], df + self_field["gradient"])
    rhs, data = model.evaluate(state, particle, lambda event: total)
    u, p, d = data["proper_velocity"], data["kinetic_momentum"], data["proper_dipole"]
    s, sd = model.unpack(state[8:]), model.unpack(rhs[8:])
    sectors = reaction_sectors(self_field, u, d, charge=particle.charge)
    force = sum((row["force"] for row in sectors.values()), np.zeros(4))
    torque = sum((row["torque"] for row in sectors.values()), np.zeros((4, 4)))
    coefficient = particle.bare_mass * particle.coupling
    external_mass = (
        model.dot(p, p) - particle.bare_mass**2 - coefficient * model.contract(f, s)
    )
    external_rate = 2 * model.dot(p, data["momentum_rate"]) - coefficient * (
        model.contract(f, sd) + np.array([model.contract(entry, s) for entry in df]) @ u
    )
    data["mass_constraint"] = external_mass
    data["mass_constraint_rate"] = external_rate
    data["mass_law"] = (
        "dynamic_external_action_mass; compare accumulated exchange, not zero"
    )
    data["reaction"] = dict(
        mode="full_dipole_rr",
        force=force,
        torque=torque,
        sectors=sectors,
        external_action_mass_rate=external_rate,
        self_field=self_field["field"],
        self_gradient=self_field["gradient"],
        approximation="Radiative finite part; ordinary-source reduction in reaction strength, no spin expansion",
        conservation_status="Bound momentum and independent total balance remain unvalidated",
    )
    return rhs, data


def evaluate_reaction(
    state: np.ndarray,
    particle: model.Particle,
    provider: model.Provider,
    *,
    window: float,
    method: str = "centered",
) -> tuple[np.ndarray, dict[str, Any]]:
    u, d, predictor = ordinary_source_coefficients(
        state, particle, provider, window=window, method=method
    )
    self_field = regular_self_field(u, d, charge=particle.charge)
    rhs, data = evaluate_with_self_field(state, particle, provider, self_field)
    data["reaction"]["predictor"] = predictor
    data["reaction"]["source_balance"] = self_field["source_balance"]
    data["reaction"][
        "source_balance_scope"
    ] = "Ordinary-source identity; not an exact balance on the reaction-corrected trajectory"
    data["reaction"]["source_velocity_change_relative"] = float(
        np.linalg.norm(data["proper_velocity"] - u[0]) / np.linalg.norm(u[0])
    )
    data["reaction"]["source_dipole_change_relative"] = (
        float(np.linalg.norm(data["proper_dipole"] - d[0]) / np.linalg.norm(d[0]))
        if np.linalg.norm(d[0])
        else 0.0
    )
    return rhs, data
