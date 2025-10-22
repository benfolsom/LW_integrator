"""Mirror-image computations used by the retarded integrator.

Conducting boundaries can be represented via image charges. The helpers here
construct those synthetic bunches while preserving the exact behaviour of the
legacy solver (including its stochastic aperture spill model).
"""

from __future__ import annotations

import random
import numpy as np

from .constants import NUMERICAL_EPSILON
from .types import ParticleState


def _random_sign() -> int:
    """Return +1 or -1 with equal probability."""

    return 1 if random.random() < 0.5 else -1


def zeros_like_state(vector: ParticleState) -> ParticleState:
    """Return an empty particle state dictionary with the same layout."""

    result: ParticleState = {
        "x": np.zeros_like(vector["x"]),
        "y": np.zeros_like(vector["y"]),
        "z": np.zeros_like(vector["z"]),
        "t": np.zeros_like(vector["t"]),
        "Px": np.zeros_like(vector["Px"]),
        "Py": np.zeros_like(vector["Py"]),
        "Pz": np.zeros_like(vector["Pz"]),
        "Pt": np.zeros_like(vector["Pt"]),
        "gamma": np.zeros_like(vector["gamma"]),
        "bx": np.zeros_like(vector["bx"]),
        "by": np.zeros_like(vector["by"]),
        "bz": np.zeros_like(vector["bz"]),
        "bdotx": np.zeros_like(vector["bdotx"]),
        "bdoty": np.zeros_like(vector["bdoty"]),
        "bdotz": np.zeros_like(vector["bdotz"]),
        "q": np.copy(vector["q"]),
    }

    if "m" in vector:
        result["m"] = np.copy(vector["m"])
    if "char_time" in vector:
        result["char_time"] = np.copy(vector["char_time"])

    return result


def generate_conducting_image(
    vector: ParticleState, wall_z: float, aperture_radius: float
) -> ParticleState:
    """Generate mirror charges for a conducting wall boundary.

    Parameters
    ----------
    vector:
        Particle bunch used to generate the mirror image.
    wall_z:
        Location of the conducting wall in the simulation coordinate system.
    aperture_radius:
        Radius of the circular aperture carved into the wall.
    """

    result = zeros_like_state(vector)

    for i in range(len(vector["x"])):
        # TODO: Verify that this is properly deprecated
        # r = np.sqrt(vector["x"][i] ** 2 + vector["y"][i] ** 2)

        if vector["z"][i] <= wall_z:
            result["z"][i] = wall_z + abs(wall_z - vector["z"][i])
        else:
            result["z"][i] = wall_z - abs(wall_z - vector["z"][i])

        R_dist = abs(result["z"][i] - vector["z"][i])
        denom = max(R_dist**2, NUMERICAL_EPSILON)
        cos_argument = 1.0 - 2.0 * (aperture_radius**2) / denom
        cos_argument = float(np.clip(cos_argument, -1.0, 1.0))
        theta = np.arccos(cos_argument)

        if theta < np.pi / 4:
            shift = 2 * R_dist * np.tan(theta)
            hypo = np.sqrt(R_dist**2 + shift**2)
            result["q"] = result["q"] * (
                1 - 2 * (aperture_radius**2) / (hypo**2) * 1 / (1 - np.cos(np.pi / 2))
            )
        else:
            shift = 0
            result["q"].fill(0.0)

        result["x"][i] = vector["x"][i]
        result["y"][i] = vector["y"][i]
        result["Px"][i] = vector["Px"][i]
        result["Py"][i] = vector["Py"][i]
        result["Pz"][i] = -vector["Pz"][i]
        result["Pt"][i] = vector["Pt"][i]
        result["gamma"][i] = vector["gamma"][i]
        result["bx"][i] = vector["bx"][i]
        result["by"][i] = vector["by"][i]
        result["bz"][i] = -vector["bz"][i]
        result["bdotx"][i] = vector["bdotx"][i]
        result["bdoty"][i] = vector["bdoty"][i]
        result["bdotz"][i] = -vector["bdotz"][i]
        result["t"][i] = vector["t"][i]

    return result


def generate_switching_image(
    vector: ParticleState, wall_z: float, aperture_radius: float, cut_z: float
) -> ParticleState:
    """Generate mirror charges for a switching wall boundary.

    The switching wall behaves like a conductor until particles pass the
    longitudinal ``cut_z`` threshold, after which the mirror image is removed
    to emulate an opening absorber.
    """

    result = zeros_like_state(vector)
    result["q"] = -np.copy(vector["q"])

    for i in range(len(vector["x"])):
        if vector["z"][i] >= cut_z:
            result["q"].fill(0.0)
        else:
            result["x"][i] = vector["x"][i]
            result["y"][i] = vector["y"][i]
            if vector["z"][i] <= wall_z:
                result["z"][i] = wall_z + abs(wall_z - vector["z"][i])
            else:
                result["z"][i] = wall_z - abs(wall_z - vector["z"][i])

        result["Px"][i] = vector["Px"][i]
        result["Py"][i] = vector["Py"][i]
        result["Pz"][i] = -vector["Pz"][i]
        result["Pt"][i] = vector["Pt"][i]
        result["gamma"][i] = vector["gamma"][i]
        result["bx"][i] = vector["bx"][i]
        result["by"][i] = vector["by"][i]
        result["bz"][i] = -vector["bz"][i]
        result["bdotx"][i] = vector["bdotx"][i]
        result["bdoty"][i] = vector["bdoty"][i]
        result["bdotz"][i] = -vector["bdotz"][i]
        result["t"][i] = vector["t"][i]

    return result


__all__ = [
    "generate_conducting_image",
    "generate_switching_image",
    "zeros_like_state",
]
