"""Bounded diagnostic iteration matching a moment update to its Medina force."""

import numpy as np


def match_medina_force(step, *, particle_count, maximum_iterations=6):
    """Repeat the same uncommitted step; never carry a trial into history.

    The supplied step consumes a lab three-force estimate and returns the
    Medina three-force it actually applied. This closes the dependence of the
    moment correction on reaction acceleration without adding another kick.
    It is diagnostic only: unprimed or capped Medina steps are rejected.
    """
    guess = np.zeros((particle_count, 3))
    for iteration in range(maximum_iterations):
        result = step(guess.copy())
        actual = np.asarray(result["_moment_applied_medina_force_native"])
        if actual.shape != guess.shape or not np.all(np.isfinite(actual)):
            raise ValueError("invalid applied Medina force in moment diagnostic")
        if np.any(result["medina_impulse_capped"]) or not np.all(
            result["medina_force_derivative_ready"]
        ):
            raise ValueError("moment diagnostic requires primed, uncapped Medina")
        scale = np.maximum(
            np.linalg.norm(actual, axis=1), np.linalg.norm(guess, axis=1)
        )
        mismatch = np.linalg.norm(actual - guess, axis=1)
        if np.all(mismatch <= 1e-6 * scale):
            result["_moment_medina_iterations"] = iteration + 1
            result["_moment_medina_force_relative_mismatch"] = np.divide(
                mismatch, scale, out=np.zeros_like(scale), where=scale > 0
            )
            return result
        guess = actual.copy()
    raise ValueError("moment/Medina force iteration did not converge")
