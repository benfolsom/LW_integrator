"""Strict source loops over already resolved light-cone samples.

Root geometry remains authoritative in the maintained resolved provider. This
kernel fuses the source jet calls after that geometry has been resolved, without
changing the scalar jet arithmetic or the caller's compensated source sums.
"""

from __future__ import annotations

import numpy as np
from numba import get_num_threads, njit, prange, set_num_threads

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


@njit(cache=True, fastmath=False)
def _resolved_charge_jet_row(
    index,
    charges,
    separations,
    velocities,
    accelerations,
    jerks,
    frames,
    snaps,
    has_frame,
    has_snap,
    ballistic_frames,
    ballistic_mask,
):
    # Numba's parallel loop cannot propagate callee exceptions reliably. Catch
    # them within the scalar row and publish a disjoint status instead; the
    # host then repeats exceptional batches through the ordered authority.
    try:
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
    except Exception:
        return False, np.zeros((4, 21)), np.zeros((3, 21)), np.zeros((3, 21)), 0.0
    return True, potential, e, b, kappa


@njit(cache=True, fastmath=False, parallel=True)
def _resolved_charge_jets_parallel(
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate disjoint pair rows; the caller retains source reduction order."""
    count = len(charges)
    potentials = np.empty((count, 4, 21))
    electric = np.empty((count, 3, 21))
    magnetic = np.empty((count, 3, 21))
    kappas = np.empty(count)
    valid_rows = np.empty(count, dtype=np.bool_)
    for index in prange(count):
        valid, potential, e, b, kappa = _resolved_charge_jet_row(
            index,
            charges,
            separations,
            velocities,
            accelerations,
            jerks,
            frames,
            snaps,
            has_frame,
            has_snap,
            ballistic_frames,
            ballistic_mask,
        )
        valid_rows[index] = valid
        potentials[index] = potential
        electric[index] = e
        magnetic[index] = b
        kappas[index] = kappa
    return potentials, electric, magnetic, kappas, valid_rows


# Complete jet batches, not observer or history counts. Remeasured by the
# production seam study; small calls retain the maintained serial dispatcher.
MIN_PARALLEL_PAIRS = 64


def resolved_charge_jets_strict(
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
    """Use at most two workers on owned pair arrays, synchronously.

    Each row retains the serial arithmetic. No parallel source reduction or
    history access occurs here. Restore the calling thread's Numba mask even
    when a singular pair raises; one-worker callers always remain serial.
    """
    args = (
        charges,
        separations,
        velocities,
        accelerations,
        jerks,
        frames,
        snaps,
        has_frame,
        has_snap,
        ballistic_frames,
        ballistic_mask,
    )
    workers = get_num_threads()
    if len(args[0]) < MIN_PARALLEL_PAIRS or workers == 1:
        return resolved_charge_jets_strict_serial(*args)
    try:
        set_num_threads(min(workers, 2))
        result = _resolved_charge_jets_parallel(*args)
        if not np.all(result[4]):
            # Re-evaluate exceptional rows in source order so worker scheduling
            # cannot select a different first failing pair or exception text.
            return resolved_charge_jets_strict_serial(*args)
        return result[:4]
    finally:
        set_num_threads(workers)
