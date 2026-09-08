"""Experimental canonical adapter reusing maintained time and error controllers.

Single observer with a supplied potential provider. This is not yet the CLI/
GUI's coupled pair mode. Checkpoints carry an explicit model and provider ID;
they are intentionally incompatible with old RFS checkpoint payloads.
"""

from dataclasses import asdict

import numpy as np

from .jakobsen_step import midpoint_step, state_velocity, canonical_constraint_residual
from .shared_lab_time import solve_proper_step_to_lab_time
from .step_doubling import (
    StepDoublingState,
    assess_step_doubling,
    propose_next_step_ns,
    StepControllerConfig,
)

MODEL = "experimental_jakobsen_canonical_linear_spin_v1"


def advance_to_time(state, target_time_ns, *, particle, provider):
    u, _ = state_velocity(state, particle, provider)

    def advance(h):
        candidate = midpoint_step(state, h, particle=particle, provider=provider)
        return {"t": np.array([candidate[0]]), "canonical_state": candidate}

    from .constants import C_MMNS

    endpoint = solve_proper_step_to_lab_time(
        advance,
        role="Jakobsen observer",
        start_time_ns=state[0],
        target_time_ns=target_time_ns,
        initial_proper_step_ns=(target_time_ns - state[0]) * C_MMNS / u[0],
        absolute_tolerance_ns=1e-18,
        relative_tolerance=1e-13,
    )
    return endpoint.state["canonical_state"]


def error_state(state, particle, provider):
    u, _ = state_velocity(state, particle, provider)
    return StepDoublingState(
        position_mm=state[1:4],
        mechanical_momentum_native=particle.mass_amu * u[1:],
        rest_spin=state[8:11],
        diagnostics_native=np.array(
            [canonical_constraint_residual(state, particle=particle, provider=provider)]
        ),
    )


def checkpoint(state, *, particle, provider_id, next_step_ns, accepted=0, rejected=0):
    if not isinstance(provider_id, str) or not provider_id:
        raise ValueError("Explicit provider identity required")
    if any(
        isinstance(count, (bool, np.bool_))
        or not isinstance(count, (int, np.integer))
        or count < 0
        for count in (accepted, rejected)
    ):
        raise ValueError("Nonnegative integer controller counts required")
    state = np.asarray(state, dtype=float)
    if (
        state.shape != (11,)
        or not np.isfinite(state).all()
        or not np.isfinite(next_step_ns)
        or next_step_ns <= 0
    ):
        raise ValueError("Finite state and positive next step required")
    return dict(
        model=MODEL,
        particle=asdict(particle),
        provider_id=provider_id,
        state=state.tolist(),
        next_step_ns=float(next_step_ns),
        accepted=int(accepted),
        rejected=int(rejected),
    )


def restore(payload, *, particle, provider_id):
    if (
        payload.get("model") != MODEL
        or payload.get("particle") != asdict(particle)
        or payload.get("provider_id") != provider_id
    ):
        raise ValueError("Checkpoint model, particle or provider mismatch")
    validated = checkpoint(
        payload["state"],
        particle=particle,
        provider_id=provider_id,
        next_step_ns=payload["next_step_ns"],
        accepted=payload["accepted"],
        rejected=payload["rejected"],
    )
    if validated["accepted"] < 0 or validated["rejected"] < 0:
        raise ValueError("Invalid controller counts")
    return validated


def integrate(
    payload,
    end_time_ns,
    *,
    particle,
    provider,
    provider_id,
    tolerances,
    maximum_step_ns,
    minimum_step_ns=1e-14,
):
    """Pure accepted-state loop. Rejected trial arrays never enter the checkpoint."""
    payload = restore(payload, particle=particle, provider_id=provider_id)
    if (
        not np.isfinite([minimum_step_ns, maximum_step_ns]).all()
        or not 0 < minimum_step_ns <= maximum_step_ns
    ):
        raise ValueError("Positive ordered step bounds required")
    state = np.asarray(payload["state"])
    if not np.isfinite(end_time_ns) or end_time_ns < state[0]:
        raise ValueError("Forward finite endpoint required")
    width = payload["next_step_ns"]
    accepted, rejected = payload["accepted"], payload["rejected"]
    records = []
    for _ in range(100000):
        remaining = end_time_ns - state[0]
        if remaining <= max(1e-18, abs(end_time_ns) * 2e-14):
            return (
                checkpoint(
                    state,
                    particle=particle,
                    provider_id=provider_id,
                    next_step_ns=width,
                    accepted=accepted,
                    rejected=rejected,
                ),
                records,
            )
        width = min(width, maximum_step_ns, remaining)
        target = state[0] + width
        full = advance_to_time(state, target, particle=particle, provider=provider)
        half = advance_to_time(
            state, state[0] + 0.5 * width, particle=particle, provider=provider
        )
        refined = advance_to_time(half, target, particle=particle, provider=provider)
        assessment = assess_step_doubling(
            error_state(full, particle, provider),
            error_state(refined, particle, provider),
            method_order=2,
            tolerances=tolerances,
        )
        proposed = propose_next_step_ns(
            width,
            assessment.normalized_error,
            accepted=assessment.accepted,
            config=StepControllerConfig(method_order=2),
            minimum_step_ns=minimum_step_ns,
            maximum_step_ns=maximum_step_ns,
        )
        if assessment.accepted:
            state = refined
            accepted += 1
            records.append(
                dict(
                    time_ns=float(state[0]),
                    error=assessment.normalized_error,
                    canonical_constraint_residual=canonical_constraint_residual(
                        state, particle=particle, provider=provider
                    ),
                )
            )
        else:
            rejected += 1
            if width <= minimum_step_ns:
                raise RuntimeError("Requested error not met at minimum step")
        width = proposed
    raise RuntimeError("Canonical adaptive trial limit exceeded")
