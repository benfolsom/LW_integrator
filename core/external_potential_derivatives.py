"""Exact potential derivatives for unbounded, uniform prescribed fields.

Use phi=-E dot x and A=(B cross x)/2 in the solver's native field units.
The potential is linear in (ct,x,y,z), so all second and higher derivatives
vanish. This is a local derivative provider, not a new canonical state gauge.
Hard field windows and magnetic gradients remain explicitly unsupported here.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .retarded_potential_directional_jet import PotentialDirectionalDerivatives
from .types import ExternalFieldConfig


def supports_uniform_external_potential(field: ExternalFieldConfig) -> bool:
    """Return whether a globally smooth uniform potential describes this config."""
    if not field.enabled:
        return True
    return not np.any(field.magnetic_field_gradient_t_per_m) and all(
        getattr(field, f"{axis}_{side}") is None
        for axis in "xyzt"
        for side in ("min", "max")
    )


def uniform_external_potential_derivatives_native(
    field: ExternalFieldConfig, *, position_mm: Sequence[float]
) -> PotentialDirectionalDerivatives:
    """Return exact derivatives without numerical field differentiation.

    Reject bounded or nonuniform enabled fields rather than assigning zero
    higher derivatives to them. Disabled fields return exactly zero.
    """
    if not supports_uniform_external_potential(field):
        raise ValueError(
            "analytical external potential requires an unbounded uniform field"
        )
    position = np.asarray(position_mm, dtype=float)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError("position_mm must have three finite components")
    electric = np.asarray(field.electric_field_native, dtype=float)
    magnetic = np.asarray(field.magnetic_field_native, dtype=float)
    if any(v.shape != (3,) or not np.all(np.isfinite(v)) for v in (electric, magnetic)):
        raise ValueError("external field vectors must have three finite components")
    potential = np.zeros(4)
    gradient = np.zeros((4, 4))
    if field.enabled:
        potential[0] = -electric @ position
        potential[1:] = 0.5 * np.cross(magnetic, position)
        gradient[1:, 0] = -electric
        gradient[1:, 1:] = 0.5 * np.cross(magnetic, np.eye(3))
    return PotentialDirectionalDerivatives(
        four_potential=potential,
        partial_a=gradient,
        partial2_a=np.zeros((4, 4, 4)),
        partial3_a_along_velocity=np.zeros((4, 4, 4)),
        partial3_a_along_acceleration=np.zeros((4, 4, 4)),
        partial4_a_along_velocity_twice=np.zeros((4, 4, 4)),
    )
