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


def _radial_weight(
    x: np.ndarray,
    y: np.ndarray,
    aperture_radius: float,
    shift: float,
    plateau: float,
) -> np.ndarray:
    """Return attenuation factors driven by the ``rho - shift`` displacement.

    Behaviour by construction:
    - If ``rho - shift == 0``                -> weight = 0 (center of aperture)
    - If ``|rho - shift| == aperture_radius`` -> weight = plateau (∈ [0.5, 1.0])
    - If ``|rho - shift| >= 2*aperture_radius`` -> weight → 1.0
    """

    rho = np.hypot(x, y)
    delta = rho - shift
    weights = np.zeros_like(delta, dtype=float)

    # If no meaningful aperture scale: give plateau everywhere except the exact center.
    if aperture_radius <= NUMERICAL_EPSILON:
        plateau_clamped = float(np.clip(plateau, 0.5, 1.0))
        weights[np.abs(delta) > 0.0] = plateau_clamped
        return weights

    a_r = float(aperture_radius)
    plateau_clamped = float(np.clip(plateau, 0.5, 1.0))

    # Symmetric ramp around the center of the aperture
    nonzero = np.abs(delta) > 0.0
    if not np.any(nonzero):
        return weights

    # Use |delta| to ramp symmetrically; clamp to [0, 2]
    dn = np.clip(np.abs(delta[nonzero]) / a_r, 0.0, 2.0)
    w = np.zeros_like(dn)

    # 0 < dn < 1: ramp up to plateau
    mask_ramp = dn < 1.0
    if np.any(mask_ramp):
        w[mask_ramp] = dn[mask_ramp] * plateau_clamped

    # 1 <= dn < 2: ramp from plateau to 1
    mask_mid = (dn >= 1.0) & (dn < 2.0)
    if np.any(mask_mid):
        w[mask_mid] = plateau_clamped + (dn[mask_mid] - 1.0) * (1.0 - plateau_clamped)

    # dn >= 2: saturate at 1
    w[dn >= 2.0] = 1.0

    weights[nonzero] = np.clip(w, 0.0, 1.0)
    return weights


def generate_conducting_image(
    vector: ParticleState,
    wall_z: float,
    aperture_radius: float,
    subcharge_count: int = 12,
    *,
    use_weighting: bool = True,
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
    use_weighting:
        When ``True`` (default) apply radial attenuation to the subcharges based on
        their cylindrical distance from the aperture axis.
    """

    count = int(subcharge_count)
    if count < 4 or count > 128:
        raise ValueError("subcharge_count must be between 4 and 128 inclusive")

    base_len = len(vector["x"])
    total_len = base_len * count

    def _alloc_like(key: str, *, fill: float = 0.0) -> np.ndarray:
        template = np.asarray(vector[key])
        return np.full(total_len, fill, dtype=template.dtype)

    result: ParticleState = {
        "x": _alloc_like("x"),
        "y": _alloc_like("y"),
        "z": _alloc_like("z"),
        "t": _alloc_like("t"),
        "Px": _alloc_like("Px"),
        "Py": _alloc_like("Py"),
        "Pz": _alloc_like("Pz"),
        "Pt": _alloc_like("Pt"),
        "gamma": _alloc_like("gamma"),
        "bx": _alloc_like("bx"),
        "by": _alloc_like("by"),
        "bz": _alloc_like("bz"),
        "bdotx": _alloc_like("bdotx"),
        "bdoty": _alloc_like("bdoty"),
        "bdotz": _alloc_like("bdotz"),
        "q": _alloc_like("q"),
    }

    if "m" in vector:
        result["m"] = np.repeat(np.asarray(vector["m"]), count)
    if "char_time" in vector:
        result["char_time"] = np.repeat(np.asarray(vector["char_time"]), count)

    charges_suppressed = False

    for i in range(base_len):
        start = i * count
        end = start + count

        mirrored_z = (
            wall_z + abs(wall_z - vector["z"][i])
            if vector["z"][i] <= wall_z
            else wall_z - abs(wall_z - vector["z"][i])
        )

        R_dist = abs(mirrored_z - vector["z"][i])
        denom = max(R_dist**2, NUMERICAL_EPSILON)
        cos_argument = 1.0 - 2.0 * (aperture_radius**2) / denom
        cos_argument = float(np.clip(cos_argument, -1.0, 1.0))
        theta = float(np.arccos(cos_argument))

        shift = 0.0
        weighting_enabled = use_weighting
        base_charge_per_sub = 0.0
        charge_values = np.zeros(count, dtype=result["q"].dtype)

        if theta < np.pi / 4:
            reduction = 1 - 2 * (aperture_radius**2) / denom * 1 / (
                1 - np.cos(np.pi / 2)
            )
            effective_charge = vector["q"][i] * reduction
            shift = float(2 * R_dist * np.tan(theta))
            base_charge_per_sub = float(effective_charge) / count
            charge_values.fill(base_charge_per_sub)
        else:
            charges_suppressed = True

        angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
        center_x = vector["x"][i]
        center_y = vector["y"][i]

        if shift <= NUMERICAL_EPSILON:
            x_positions = np.full(count, center_x, dtype=result["x"].dtype)
            y_positions = np.full(count, center_y, dtype=result["y"].dtype)
        else:
            x_positions = center_x + shift * np.cos(angles)
            y_positions = center_y + shift * np.sin(angles)

        if weighting_enabled and base_charge_per_sub != 0.0:
            # Geometry-dependent plateau: plateau = (G / 2) with G in [1, 2),
            # increasing with R_dist / aperture_radius
            ratio = float(R_dist / max(aperture_radius, NUMERICAL_EPSILON))
            geometry_factor = 1.0 + ratio / (1.0 + ratio)
            plateau = float(np.clip(0.5 * geometry_factor, 0.5, 1.0))

            displacement = float(np.hypot(center_x, center_y))
            if displacement > NUMERICAL_EPSILON or shift > NUMERICAL_EPSILON:
                weights = _radial_weight(
                    x_positions,
                    y_positions,
                    aperture_radius,
                    shift,
                    plateau,
                )
            else:
                # Perfectly on-axis: enforce zero weight
                weights = np.zeros(count, dtype=float)

            # Combine legacy solid-angle reduction (in base_charge_per_sub)
            # with rho-shift weighting that emphasizes off-axis behaviour.
            charge_values = (base_charge_per_sub * weights).astype(
                result["q"].dtype, copy=False
            )

        result["x"][start:end] = x_positions
        result["y"][start:end] = y_positions
        result["z"][start:end] = mirrored_z
        result["t"][start:end] = vector["t"][i]

        result["Px"][start:end] = vector["Px"][i]
        result["Py"][start:end] = vector["Py"][i]
        result["Pz"][start:end] = -vector["Pz"][i]
        result["Pt"][start:end] = vector["Pt"][i]
        result["gamma"][start:end] = vector["gamma"][i]

        result["bx"][start:end] = vector["bx"][i]
        result["by"][start:end] = vector["by"][i]
        result["bz"][start:end] = -vector["bz"][i]

        result["bdotx"][start:end] = vector["bdotx"][i]
        result["bdoty"][start:end] = vector["bdoty"][i]
        result["bdotz"][start:end] = -vector["bdotz"][i]

        result["q"][start:end] = charge_values

        if "m" in result:
            result["m"][start:end] = vector["m"][i]
        if "char_time" in result:
            result["char_time"][start:end] = vector["char_time"][i]

    if charges_suppressed:
        result["q"].fill(0.0)

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
