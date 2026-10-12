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


def commit_general_transport(*args, **kwargs):
    """Reject unsynchronised general-runner transport, including direct callers."""
    raise NotImplementedError(
        "Photon transport requires the native lab-time pair runner; "
        "the general runner has unsynchronised particle lab clocks"
    )


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
            start_position=histories[i].position[-1].copy(),
            start_time_ns=float(histories[i].time[-1]),
            end_time_ns=float(trial[0]),
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
        delta_energy = C * (
            np.hypot(particle.mass_amu * C, np.linalg.norm(target.momentum))
            - np.hypot(particle.mass_amu * C, np.linalg.norm(before))
        )
        diagnostics[i]["photon_scattering_energy_change_native"] = float(delta_energy)
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
        data["photon_scattering_energy_change_native"] = float(delta_energy)
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
