"""Instantaneous Plummer interactions and accepted lab-time source sampling."""

from __future__ import annotations

import numpy as np


def plummer_force_potential(
    displacement: np.ndarray,
    source_charge: np.ndarray,
    observer_charge: float,
    softening_squared: float | np.ndarray,
) -> tuple[np.ndarray, float]:
    """Return laboratory force and Phi, with observer/source weights separate.

    Zero-charge sources are excluded before division, including the self-source.
    The force is the negative observer gradient of q Phi, even at r = epsilon.
    """
    selected = np.asarray(source_charge) != 0.0
    r = np.asarray(displacement, dtype=float)[selected]
    charge = np.asarray(source_charge, dtype=float)[selected]
    eps2 = np.broadcast_to(softening_squared, np.shape(source_charge))[selected]
    rho2 = np.sum(r * r, axis=1) + eps2
    # An unsoftened coincident point has no defined interaction. Match the
    # sampled kernel's exclusion rather than divide by zero.
    valid = rho2 > 0.0
    r, charge, rho2 = r[valid], charge[valid], rho2[valid]
    rho = np.sqrt(rho2)
    force = observer_charge * np.sum(
        charge[:, None] * r / (rho2 * rho)[:, None], axis=0
    )
    return force, float(np.sum(charge / rho))


class MissingLabTimeHistoryError(RuntimeError):
    """An instantaneous source cannot bracket the observer's lab event."""


def sample_lab_time_sources(history, time_ns, *, required_charges=None):
    """Linearly sample accepted source positions without extrapolating history.

    Exact knots, including the common bunch barrier, avoid interpolation.
    Zero-charge and dead sources do not require a bracket.
    """
    from .vectorized_interactions import gather_external_samples

    if not history or "t" not in history[-1]:
        raise MissingLabTimeHistoryError(
            "Instantaneous sources have no accepted lab-time history"
        )
    last = history[-1]
    count = len(last["t"])
    required = (
        np.asarray(
            last.get("q_source", last["q"])
            if required_charges is None
            else required_charges
        )
        != 0
    )
    required &= ~np.asarray(last.get("_dead_particles", np.zeros(count, dtype=bool)))
    indices = np.full(count, len(history) - 1, dtype=int)
    next_indices = indices.copy()
    weights = np.ones(count)
    for j in np.flatnonzero(required):
        if float(last["t"][j]) == time_ns:
            continue
        times = np.array([state["t"][j] for state in history], dtype=float)
        if (
            not np.all(np.isfinite(times))
            or np.any(np.diff(times) < 0)
            or time_ns < times[0]
            or time_ns > times[-1]
        ):
            raise MissingLabTimeHistoryError(
                f"Instantaneous source {j} has no history bracketing observer "
                f"lab time {time_ns:.17g} ns (available "
                f"{times[0]:.17g} to {times[-1]:.17g} ns)"
            )
        lower = int(np.searchsorted(times, time_ns, side="right") - 1)
        indices[j] = lower
        if times[lower] != time_ns:
            next_indices[j] = lower + 1
            weights[j] = (times[lower + 1] - time_ns) / (
                times[lower + 1] - times[lower]
            )
        else:
            next_indices[j] = lower
    return gather_external_samples(
        history,
        indices,
        indices_next=next_indices,
        weights=weights,
        interpolate_positions=True,
        include_positions=True,
    )
