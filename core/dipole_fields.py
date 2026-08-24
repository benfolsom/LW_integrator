"""Exact exterior field of a static point magnetic dipole.

This module is a deliberately small source-field oracle.  It evaluates the
vacuum, Coulomb-gauge point-dipole potential

``A = m x r / R^3``

and its curl in the solver's native scaled-Gaussian units.  The signed scalar
``magnetic_moment_native`` multiplies a unit rest-spin direction, so a negative
electron moment is antiparallel to the supplied spin without changing the
spin convention.

Only the exterior solution is represented.  The contact delta distribution,
finite-size regularization, softening, motion, retardation, and dipole
self-reaction are intentionally absent.  The result is suitable as a static
validation oracle and as a reusable rest-frame kernel for a future retarded
dipole provider.

Gradient arrays use ``gradient[field_component, coordinate]``.  The returned
``partial_f[lambda, mu, nu]`` is ``partial_lambda F^(mu nu)`` for coordinates
``(c t, x, y, z)`` in millimetres; its temporal slice is exactly zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Sequence, cast

import numpy as np

from .rfs import electromagnetic_field_tensor_native

VectorLike = Sequence[float] | np.ndarray

_UNIT_VECTOR_RTOL = 1.0e-12
_UNIT_VECTOR_ATOL = 1.0e-15


class DipoleFieldDomainError(ValueError):
    """Raised when a point-dipole field is requested outside its domain."""


class DipoleSelfFieldError(DipoleFieldDomainError):
    """Raised when a particle's intrinsic dipole self-field is requested."""


@dataclass(frozen=True)
class StaticDipoleFieldResult:
    """Static exterior dipole potential, field, and analytic derivatives.

    ``vector_potential_native`` has native magnetic-field times millimetre
    units.  Consequently ``curl(A)`` is ``magnetic_field_native`` and
    ``q A / c`` has native canonical-momentum units.  ``four_potential_native``
    is contravariant ``A^mu = (0, A)`` in this static gauge.

    An identity-excluded result has ``excluded=True`` and exactly zero
    potentials, fields, and derivatives.  Its separation diagnostics are
    retained so exclusion remains explicit rather than masquerading as a
    distant zero field.
    """

    magnetic_moment_vector_native: np.ndarray
    separation_vector_mm: np.ndarray
    separation_mm: float
    four_potential_native: np.ndarray
    vector_potential_native: np.ndarray
    vector_potential_gradient_native: np.ndarray
    electric_field_native: np.ndarray
    magnetic_field_native: np.ndarray
    magnetic_gradient_native_per_mm: np.ndarray
    field_tensor: np.ndarray
    partial_f: np.ndarray
    excluded: bool = False


def _vector3(value: VectorLike, *, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,):
        raise ValueError(f"{name} must contain exactly three components")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return cast(np.ndarray, vector)


def _moment_vector_native(
    magnetic_moment_native: float, rest_spin_direction: VectorLike
) -> np.ndarray:
    moment = float(magnetic_moment_native)
    if not np.isfinite(moment):
        raise ValueError("magnetic_moment_native must be finite")
    spin = _vector3(rest_spin_direction, name="rest_spin_direction")
    norm = float(np.linalg.norm(spin))
    if not np.isclose(
        norm,
        1.0,
        rtol=_UNIT_VECTOR_RTOL,
        atol=_UNIT_VECTOR_ATOL,
    ):
        raise ValueError("rest_spin_direction must be a unit vector")
    return cast(np.ndarray, moment * spin)


def _validated_minimum_separation(minimum_separation_mm: float) -> float:
    minimum = float(minimum_separation_mm)
    if not np.isfinite(minimum) or minimum < 0.0:
        raise ValueError("minimum_separation_mm must be finite and non-negative")
    return minimum


def _zero_result(
    *, moment_vector: np.ndarray, separation_vector: np.ndarray
) -> StaticDipoleFieldResult:
    zeros3: np.ndarray = np.zeros(3, dtype=float)
    zeros4: np.ndarray = np.zeros(4, dtype=float)
    zeros33: np.ndarray = np.zeros((3, 3), dtype=float)
    zeros44: np.ndarray = np.zeros((4, 4), dtype=float)
    zeros444: np.ndarray = np.zeros((4, 4, 4), dtype=float)
    return StaticDipoleFieldResult(
        magnetic_moment_vector_native=moment_vector,
        separation_vector_mm=separation_vector,
        separation_mm=float(np.linalg.norm(separation_vector)),
        four_potential_native=zeros4,
        vector_potential_native=zeros3,
        vector_potential_gradient_native=zeros33,
        electric_field_native=zeros3.copy(),
        magnetic_field_native=zeros3.copy(),
        magnetic_gradient_native_per_mm=zeros33.copy(),
        field_tensor=zeros44,
        partial_f=zeros444,
        excluded=True,
    )


def static_point_dipole_field_native(
    *,
    separation_vector_mm: VectorLike,
    magnetic_moment_native: float,
    rest_spin_direction: VectorLike,
    minimum_separation_mm: float = 0.0,
) -> StaticDipoleFieldResult:
    """Return the strict exterior field for one static point dipole.

    Args:
        separation_vector_mm: Observer position minus source position.
        magnetic_moment_native: Signed fully polarized magnetic moment in
            native ``charge * mm`` units.
        rest_spin_direction: Unit spin direction in the source rest frame.
        minimum_separation_mm: Hard exterior-model cutoff.  Evaluation at or
            inside this radius raises :class:`DipoleFieldDomainError`; no
            softening is applied.

    The exact native scaled-Gaussian expressions are

    ``A = m x n / R^2`` and
    ``B = (3 n (m.n) - m) / R^3``.
    """

    separation_vector = _vector3(separation_vector_mm, name="separation_vector_mm")
    moment_vector = _moment_vector_native(magnetic_moment_native, rest_spin_direction)
    minimum = _validated_minimum_separation(minimum_separation_mm)
    separation = float(np.linalg.norm(separation_vector))
    if separation <= minimum:
        if separation == 0.0:
            detail = "the observer coincides with the point-dipole source"
        else:
            detail = (
                f"separation {separation:.16g} mm is at or inside the hard "
                f"minimum {minimum:.16g} mm"
            )
        raise DipoleFieldDomainError(f"{detail}; the exterior field is undefined")

    direction = separation_vector / separation
    moment_on_direction = float(moment_vector @ direction)
    inverse_r2 = separation**-2
    inverse_r3 = separation**-3
    inverse_r4 = separation**-4

    vector_potential = np.cross(moment_vector, direction) * inverse_r2
    magnetic_field = (
        3.0 * direction * moment_on_direction - moment_vector
    ) * inverse_r3

    # dA_i/dx_k = epsilon_ijk m_j/R^3 - 3 A_i n_k/R.
    # Construct its first term by differentiating m x r component by
    # component.  Column k is the derivative along coordinate k.
    vector_potential_gradient: np.ndarray = np.empty((3, 3), dtype=float)
    coordinate_basis: np.ndarray = np.eye(3, dtype=float)
    moment_cross_separation = np.cross(moment_vector, separation_vector)
    for coordinate in range(3):
        vector_potential_gradient[:, coordinate] = (
            np.cross(moment_vector, coordinate_basis[:, coordinate]) * inverse_r3
            - 3.0
            * moment_cross_separation
            * separation_vector[coordinate]
            * separation**-5
        )

    identity: np.ndarray = np.eye(3, dtype=float)
    magnetic_gradient = (
        3.0
        * inverse_r4
        * (
            identity * moment_on_direction
            + np.outer(direction, moment_vector)
            + np.outer(moment_vector, direction)
            - 5.0 * moment_on_direction * np.outer(direction, direction)
        )
    )

    electric_field: np.ndarray = np.zeros(3, dtype=float)
    field_tensor = electromagnetic_field_tensor_native(electric_field, magnetic_field)
    partial_f: np.ndarray = np.zeros((4, 4, 4), dtype=float)
    for coordinate in range(3):
        partial_f[coordinate + 1] = electromagnetic_field_tensor_native(
            electric_field,
            magnetic_gradient[:, coordinate],
        )

    four_potential = np.concatenate(([0.0], vector_potential))
    return StaticDipoleFieldResult(
        magnetic_moment_vector_native=moment_vector,
        separation_vector_mm=separation_vector,
        separation_mm=separation,
        four_potential_native=four_potential,
        vector_potential_native=vector_potential,
        vector_potential_gradient_native=vector_potential_gradient,
        electric_field_native=electric_field,
        magnetic_field_native=magnetic_field,
        magnetic_gradient_native_per_mm=magnetic_gradient,
        field_tensor=field_tensor,
        partial_f=partial_f,
    )


def evaluate_static_point_dipole_field_native(
    *,
    source_position_mm: VectorLike,
    observer_position_mm: VectorLike,
    magnetic_moment_native: float,
    rest_spin_direction: VectorLike,
    minimum_separation_mm: float = 0.0,
    source_particle_id: Hashable | None = None,
    observer_particle_id: Hashable | None = None,
    exclude_self: bool = False,
) -> StaticDipoleFieldResult:
    """Evaluate a positioned static source with explicit identity exclusion.

    If both particle identifiers compare equal, the intrinsic self-field is
    never evaluated.  ``exclude_self=True`` returns an explicit zero result;
    otherwise :class:`DipoleSelfFieldError` is raised.  Different identities
    still obey the coincidence and hard minimum-separation checks.
    """

    source_position = _vector3(source_position_mm, name="source_position_mm")
    observer_position = _vector3(observer_position_mm, name="observer_position_mm")
    moment_vector = _moment_vector_native(magnetic_moment_native, rest_spin_direction)
    _validated_minimum_separation(minimum_separation_mm)
    separation_vector = observer_position - source_position

    identities_supplied = (
        source_particle_id is not None and observer_particle_id is not None
    )
    same_particle = bool(
        identities_supplied and source_particle_id == observer_particle_id
    )
    if same_particle:
        if exclude_self:
            return _zero_result(
                moment_vector=moment_vector,
                separation_vector=separation_vector,
            )
        raise DipoleSelfFieldError(
            "a particle's intrinsic dipole self-field is outside this provider; "
            "pass exclude_self=True to return an explicit excluded result"
        )

    return static_point_dipole_field_native(
        separation_vector_mm=separation_vector,
        magnetic_moment_native=magnetic_moment_native,
        rest_spin_direction=rest_spin_direction,
        minimum_separation_mm=minimum_separation_mm,
    )


__all__ = [
    "DipoleFieldDomainError",
    "DipoleSelfFieldError",
    "StaticDipoleFieldResult",
    "evaluate_static_point_dipole_field_native",
    "static_point_dipole_field_native",
]
