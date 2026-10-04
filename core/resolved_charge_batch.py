"""Strict source loops over already resolved light-cone samples.

Root geometry remains authoritative in the maintained resolved provider. This
kernel fuses the source jet calls after that geometry has been resolved, without
changing the scalar jet arithmetic or the caller's compensated source sums.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from .precise_charge import precise_charge_jets_strict_serial, resolved_separation_frame


@njit(cache=True, fastmath=False)
def resolved_charge_jets_strict_serial(
    charges: np.ndarray,
    separations: np.ndarray,
    velocities: np.ndarray,
    accelerations: np.ndarray,
    jerks: np.ndarray,
    frames: np.ndarray,
    snaps: np.ndarray,
    has_frame: np.ndarray,
    has_snap: np.ndarray,
    ballistic_frames: np.ndarray | None = None,
    ballistic_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate sources in their original order with one compiled dispatch."""
    count = len(charges)
    potentials = np.empty((count, 4, 21))
    electric = np.empty((count, 3, 21))
    magnetic = np.empty((count, 3, 21))
    kappas = np.empty(count)
    for index in range(count):
        frame = frames[index] if has_frame[index] else None
        if ballistic_frames is not None and ballistic_mask is not None:
            if ballistic_mask[index]:
                frame = resolved_separation_frame(
                    ballistic_frames[index, 0],
                    ballistic_frames[index, 1:],
                    velocities[index],
                )
        potential, e, b, _root, _residual, kappa = precise_charge_jets_strict_serial(
            charges[index],
            separations[index],
            velocities[index],
            accelerations[index],
            jerks[index],
            frame,
            snaps[index] if has_snap[index] else None,
        )
        potentials[index] = potential
        electric[index] = e
        magnetic[index] = b
        kappas[index] = kappa
    return potentials, electric, magnetic, kappas
