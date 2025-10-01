"""Mirror image computations used by the retarded integrator."""

from __future__ import annotations

import random
from typing import Dict

import numpy as np

from .types import ParticleState


def _random_sign() -> int:
    """Return +1 or -1 with equal probability."""

    return 1 if random.random() < 0.5 else -1


def zeros_like_state(vector: ParticleState) -> ParticleState:
    """Create an empty particle state dictionary with the same array layout."""

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
    """Generate mirror charges for a conducting wall boundary."""

    result = zeros_like_state(vector)

    for i in range(len(vector["x"])):
        r = np.sqrt(vector["x"][i] ** 2 + vector["y"][i] ** 2)

        if vector["z"][i] >= wall_z:
            result["q"].fill(0.0)
        else:
            result["q"] = -vector["q"]
            result["z"][i] = wall_z + abs(wall_z - vector["z"][i])

        R_dist = abs(result["z"][i] - vector["z"][i])

        if R_dist / 2 > aperture_radius:
            theta = np.arccos(-2 * (aperture_radius**2) / (R_dist**2) + 1)
            sign_x = _random_sign()
            sign_y = _random_sign()

            if theta < np.pi / 4:
                shift = 2 * R_dist * np.tan(theta)
                result["x"][i] = (
                    vector["x"][i] + (aperture_radius + shift / np.sqrt(2)) * sign_x
                )
                result["y"][i] = (
                    vector["y"][i] + (aperture_radius + shift / np.sqrt(2)) * sign_y
                )
                result["q"] = result["q"] * (
                    1
                    - 2
                    * (aperture_radius**2)
                    / (R_dist**2)
                    * 1
                    / (1 - np.cos(np.pi / 2))
                )
            else:
                shift = 0
                result["q"].fill(0.0)
                result["x"][i] = vector["x"][i]
                result["y"][i] = vector["y"][i]
        else:
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
    """Generate mirror charges for a switching wall boundary."""

    result = zeros_like_state(vector)
    result["q"] = -np.copy(vector["q"])

    for i in range(len(vector["x"])):
        if vector["z"][i] >= cut_z:
            result["q"].fill(0.0)
        else:
            result["x"][i] = vector["x"][i]
            result["y"][i] = vector["y"][i]
            result["z"][i] = wall_z + abs(wall_z - vector["z"][i])

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
