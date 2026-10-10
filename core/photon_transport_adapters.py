"""Accepted-step photon adapters; photons remain outside charged source history."""

from __future__ import annotations

import numpy as np

from .constants import C_MMNS as C
from .photon_transport import PhotonTarget


def validate_transport_emission(config, emission_config):
    if config.enabled and not emission_config.enabled:
        raise ValueError("Photon transport requires stochastic_emission.enabled=true")


def _pair_photon_response(provider, time, position):
    """Use published history, or a cubic between accepted unpublished tail knots.

    The pair C5 derivative window delays publication by six knots. A photon's
    null ray can reach that tail immediately after birth. Only zero-spin charge
    histories are allowed here; split incoming/outgoing velocities are retained.
    No source extrapolation or charged-history publication occurs.
    """
    try:
        return provider(time, position)[2]
    except ValueError as error:
        if (
            str(error)
            != "Retarded source time is outside published history; no extrapolation"
        ):
            raise
        history = provider.history
        from .full_dipole_history import _endpoint_polynomial
        from .full_dipole_response import response

        jumps = dict(history.velocity_jumps)
        event = np.r_[C * time, position]
        for left in range(len(history.time) - 2, max(-1, len(history.time) - 8), -1):
            start, stop = C * history.time[left : left + 2]
            if time < history.time[left]:
                continue
            # The response's endpoint gate uses the same coordinate roundoff scale.
            tol = (
                8
                * np.finfo(float).eps
                * max(
                    abs(event[0]),
                    abs(start),
                    abs(stop),
                    np.linalg.norm(position),
                    np.linalg.norm(history.position[left]),
                    np.linalg.norm(history.position[left + 1]),
                )
            )
            a = event[0] - start - np.linalg.norm(position - history.position[left])
            b = event[0] - stop - np.linalg.norm(position - history.position[left + 1])
            if a < -tol or b > tol:
                continue
            incoming = np.asarray(jumps.get(left + 1, history.velocity[left + 1]))
            polynomial = _endpoint_polynomial(
                np.array([history.position[left], history.velocity[left] / C]),
                np.array([history.position[left + 1], incoming / C]),
                stop - start,
            )
            result = response(
                event,
                start,
                stop - start,
                polynomial,
                np.zeros((1, 4, 4)),
                charge=provider.charge_native / C,
                allow_boundary=True,
            )
            from .momentum_center_pair import smooth_start_native

            values = tuple(
                C * result[key]
                for key in ("four_potential", "partial_a", "field_tensor", "partial_f")
            )
            ramped = smooth_start_native(
                lambda _time, _position: values,
                provider.startup_duration_ns,
                provider.startup_smoothness,
            )
            return ramped(time, position)[2]
        raise error


def _apply_general_recoil(state, index, before, after):
    from .integration_runner import _clear_medina_force_history

    mass = float(state["m"][index])
    before_pt = np.hypot(mass * C, np.linalg.norm(before))
    after_pt = np.hypot(mass * C, np.linalg.norm(after))
    state["Pt"][index] += after_pt - before_pt
    state["gamma"][index] = after_pt / (mass * C)
    proper = after / (mass * C)
    for j, axis in enumerate("xyz"):
        state[f"P{axis}"][index] += after[j] - before[j]
        state[f"b{axis}"][index] = after[j] / after_pt
        if f"source_u_{axis}" in state:
            state[f"source_u_{axis}"][index] = proper[j]
    if "source_speed_deficit" in state:
        state["source_speed_deficit"][index] = 1 / (
            state["gamma"][index] * (state["gamma"][index] + np.linalg.norm(proper))
        )
    if "source_kinematics_ready" in state:
        state["source_kinematics_ready"][index] = 1.0
    _clear_medina_force_history(
        state, particle_mask=np.arange(len(state["x"])) == index
    )


def _sampled_photon_fields(histories, time, position):
    """Nearest causal null-time sample, without interpolation or extrapolation.

    Like the sampled charge path, this is a diagnostic field approximation.
    Missing prehistory contributes zero. Continuous stored bdot excludes kicks.
    """
    electric, magnetic = np.zeros(3), np.zeros(3)
    for history in histories:
        for j in range(len(history[0]["x"])):
            candidates = []
            for row in history:
                t = float(row["t"][j])
                if t > time or bool(
                    row.get("_dead_particles", np.zeros(len(row["x"]), dtype=bool))[j]
                ):
                    continue
                rvec = position - np.array([row[a][j] for a in "xyz"])
                radius = np.linalg.norm(rvec)
                residual = time - t - radius / C
                if radius > 0 and residual >= 0:
                    candidates.append((residual, row, rvec, radius))
            if not candidates:
                continue
            _, row, rvec, radius = min(candidates, key=lambda item: item[0])
            n = rvec / radius
            beta = np.array([row[f"b{a}"][j] for a in "xyz"])
            bdot = np.array([row[f"bdot{a}"][j] for a in "xyz"])
            kappa = 1 - n @ beta
            charge = float(row.get("q_source", row["q"])[j])
            e = (
                charge
                / kappa**3
                * (
                    (n - beta) / (row["gamma"][j] ** 2 * radius**2)
                    + np.cross(n, np.cross(n - beta, bdot)) / radius
                )
            )
            electric += e
            magnetic += np.cross(n, e)
    return electric, magnetic


def commit_general_transport(
    runtime, emission, rider_history, driver_history, external_field=None
):
    """Transport to latest charge endpoint; targets freeze at accepted endpoints."""
    runtime.ingest(emission.events)
    targets, owners, before = [], [], []
    for role, history in (("rider", rider_history), ("driver", driver_history)):
        state = history[-1]
        for i in range(len(state["x"])):
            if bool(
                state.get("_dead_particles", np.zeros(len(state["x"]), dtype=bool))[i]
            ):
                continue
            mass = float(state["m"][i])
            p = (
                mass
                * C
                * state["gamma"][i]
                * np.array([state[f"b{a}"][i] for a in "xyz"])
            )
            targets.append(
                PhotonTarget(
                    f"{role}:{i}",
                    np.array([state[a][i] for a in "xyz"]),
                    p.copy(),
                    mass,
                    float(state.get("q_observer", state["q"])[i]),
                )
            )
            owners.append((state, i))
            before.append(p)

    def fields(time, position):
        e, b = _sampled_photon_fields((rider_history, driver_history), time, position)
        if external_field is not None and external_field.enabled:
            from .external_fields import evaluate_external_field_native

            ext_e, ext_b, _ = evaluate_external_field_native(
                external_field, position_mm=tuple(position), time_ns=time
            )
            e, b = e + ext_e, b + ext_b
        return e, b

    endpoint = max(float(np.max(h[-1]["t"])) for h in (rider_history, driver_history))
    runtime.advance(endpoint, targets, fields)
    for target, (state, i), original in zip(targets, owners, before):
        if not np.array_equal(original, target.momentum):
            _apply_general_recoil(state, i, original, target.momentum)


def commit_pair_transport(
    runtime,
    emission,
    payload,
    particles,
    histories,
    trials,
    diagnostics,
    candidate_histories,
    source_components=None,
):
    """Atomic endpoint recoil; preserve pre-emission incoming velocity knots."""
    from .momentum_center_pair import FullDipoleProvider, dynamics_native

    runtime.ingest(emission.events)
    targets = [
        PhotonTarget(
            str(i),
            trial[1:4].copy(),
            data["kinetic_momentum_native"][1:].copy(),
            particle.mass_amu,
            particle.charge_native,
        )
        for i, (trial, data, particle) in enumerate(zip(trials, diagnostics, particles))
    ]
    originals = [target.momentum.copy() for target in targets]
    # All sources, including the emitter, contribute fields at a photon query.
    sources = [
        FullDipoleProvider(
            h,
            p.charge_native,
            payload.get("startup_duration_ns", 0.0),
            payload.get("startup_smoothness", 5),
        )
        for h, p in zip(histories, particles)
    ]
    # Photon null roots at a birth knot need the outgoing accepted segment as
    # well as the incoming one. The full charge barrier is already accepted;
    # freeze these pre-transport histories for all photon field queries.
    field_sources = [
        FullDipoleProvider(
            h,
            p.charge_native,
            payload.get("startup_duration_ns", 0.0),
            payload.get("startup_smoothness", 5),
        )
        for h, p in zip(candidate_histories, particles)
    ]

    def fields(time, position):
        f = sum(
            (
                _pair_photon_response(provider, time, position)
                for provider in field_sources
            ),
            np.zeros((4, 4)),
        )
        return f[1:, 0], -np.array([f[2, 3], f[3, 1], f[1, 2]])

    runtime.advance(float(trials[0][0]), targets, fields)
    for i, (target, before, trial, particle) in enumerate(
        zip(targets, originals, trials, particles)
    ):
        if np.array_equal(target.momentum, before):
            continue
        mc = particle.mass_amu * C
        trial[5:8] += target.momentum - before
        trial[4] += np.hypot(mc, np.linalg.norm(target.momentum)) - np.hypot(
            mc, np.linalg.norm(before)
        )
        if source_components is not None:
            from decimal import localcontext
            from .preserved_source import decimal_array, split_array

            reference, high, low = source_components
            with localcontext() as context:
                context.prec = 80
                high[i, 4:8], low[i, 4:8] = split_array(
                    decimal_array(trial[4:8]) - decimal_array(reference[i, 4:8])
                )
        # Zero-spin constraint makes the canonical mechanical offset sufficient.
        # Use only other charged sources when reconstructing the target state.
        provider = sources[1 - i] if len(sources) == 2 else None
        if provider is None:
            from .momentum_center_pair import CombinedDipoleProvider

            provider = CombinedDipoleProvider(
                [p for j, p in enumerate(sources) if j != i]
            )
        rate, data = dynamics_native(
            trial, particle, provider, suppress_radiation=emission.active(str(i))
        )
        if "stochastic_emission" in diagnostics[i]:
            data["stochastic_emission"] = diagnostics[i]["stochastic_emission"]
        diagnostics[i] = data
        accepted = candidate_histories[i]
        incoming = np.asarray(
            dict(accepted.velocity_jumps).get(
                len(histories[i].time), accepted.velocity[-1]
            )
        )
        parts = (
            (accepted.dipole_high[-1], accepted.dipole_low[-1])
            if accepted.dipole_reference is not None
            else None
        )
        candidate_histories[i] = histories[i].append(
            trial[0],
            trial[1:4],
            rate[1:4],
            data["proper_dipole_native"],
            dipole_parts=parts,
            velocity_before_jump=incoming,
        )
