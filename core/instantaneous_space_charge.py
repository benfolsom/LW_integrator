"""Instantaneous Plummer interactions on the shared proper-time source clock."""

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
