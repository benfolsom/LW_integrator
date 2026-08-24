"""Native scaled-Gaussian Rafelski--Formanek--Steinmetz kernel.

This module evaluates the local response of one classical point particle to a
*supplied* electromagnetic field. It deliberately does not construct charge
or magnetic-dipole source fields and it does not advance an integrator state.

The translational equation uses the full antisymmetric ``G`` tensor from
Rafelski, Formanek, and Steinmetz, Eur. Phys. J. C 78, 6 (2018), Eqs. (14),
(17), and (18), https://doi.org/10.1140/epjc/s10052-017-5493-2. The homogeneous
spin coefficients use the signed minimal choice stated by Formanek, Steinmetz,
and Rafelski, Phys. Rev. A 103, 052218 (2021), Eqs. (3) and (8),
https://doi.org/10.1103/PhysRevA.103.052218. The spin-gradient term uses the
full 2018 ``G`` tensor; it reduces to the compact 2021 Eq. (11) form in vacuum
but is an explicit full-G extension inside a current distribution.

Conventions
-----------

* Solver-native scaled-Gaussian units throughout: amu, mm, ns, and native
  charge. ``c`` is therefore :data:`core.constants.C_MMNS`.
* Coordinates are ``x = (c t, x, y, z)`` in mm and the metric is
  ``diag(+1,-1,-1,-1)``.
* ``F`` means contravariant ``F^(mu nu)`` with ``F^(0i) = -E_i`` and
  ``F^(ij) = -epsilon_ijk B_k``. Native ``E`` and ``B`` have the same units
  and the Lorentz force is ``q (E + beta x B)``.
* ``partial_f[lambda, mu, nu]`` means ``partial_lambda F^(mu nu)`` per mm.
  The temporal derivative is ``partial_0 = (1/c) partial_t``.
* ``u`` is the contravariant four-velocity in mm/ns. ``a`` is the
  dimensionless normalized spin/polarization four-vector; its rest-frame norm
  is one for a fully polarized particle.
* The signed magnetic moment is in native ``charge * mm`` and the invariant
  spin magnitude is in native ``amu * mm^2 / ns``.

With normalized spin, the equations implemented here are

``dp/dtau = (q/c) F.u + (mu/c) G[a].u``

and

``da/dtau = q/(m c) F.a + (mu/S - q/(m c))
             [F.a - u (u.F.a)/c^2] + mu/(m c) G[a].a``.

The partial derivative used to form ``G`` acts on the supplied field while
holding the observer spin fixed. Source-spin retardation and derivatives,
self-field removal, and radiation reaction are responsibilities of the caller.
"""

from __future__ import annotations

from itertools import permutations
from typing import Sequence, Tuple, Union, cast

import numpy as np

from .constants import C_MMNS

MINKOWSKI_METRIC = np.diag((1.0, -1.0, -1.0, -1.0))
"""Minkowski metric with signature ``(+---)``."""

VectorLike = Union[Sequence[float], np.ndarray]
TensorLike = Union[Sequence[Sequence[float]], np.ndarray]
GradientLike = Union[Sequence[Sequence[Sequence[float]]], np.ndarray]


def _permutation_sign(indices: Sequence[int]) -> float:
    inversions = sum(
        indices[left] > indices[right]
        for left in range(4)
        for right in range(left + 1, 4)
    )
    return -1.0 if inversions % 2 else 1.0


_LEVI_CIVITA_LOWER = np.zeros((4, 4, 4, 4), dtype=float)
for _indices in permutations(range(4)):
    _LEVI_CIVITA_LOWER[_indices] = _permutation_sign(_indices)

# Raising all four indices with (+---) changes the sign.
_LEVI_CIVITA_UPPER = -_LEVI_CIVITA_LOWER


def _four_vector(value: VectorLike, *, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (4,):
        raise ValueError(f"{name} must have shape (4,)")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def _field_tensor(value: TensorLike, *, name: str = "field_tensor") -> np.ndarray:
    tensor = np.asarray(value, dtype=float)
    if tensor.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4)")
    if not np.all(np.isfinite(tensor)):
        raise ValueError(f"{name} must contain only finite values")
    if not np.allclose(tensor + tensor.T, 0.0, rtol=1.0e-12, atol=1.0e-15):
        raise ValueError(f"{name} must be antisymmetric")
    return tensor


def _field_gradient(value: GradientLike) -> np.ndarray:
    gradient = np.asarray(value, dtype=float)
    if gradient.shape != (4, 4, 4):
        raise ValueError("partial_f must have shape (4, 4, 4)")
    if not np.all(np.isfinite(gradient)):
        raise ValueError("partial_f must contain only finite values")
    antisymmetry_error = gradient + np.swapaxes(gradient, 1, 2)
    if not np.allclose(antisymmetry_error, 0.0, rtol=1.0e-12, atol=1.0e-15):
        raise ValueError("partial_f must be antisymmetric in its field indices")
    return gradient


def lower_four_vector(vector: VectorLike) -> np.ndarray:
    """Lower a contravariant four-vector index with ``diag(+1,-1,-1,-1)``."""

    return cast(np.ndarray, MINKOWSKI_METRIC @ _four_vector(vector, name="vector"))


def minkowski_dot(left: VectorLike, right: VectorLike) -> float:
    """Return the ``(+---)`` inner product of two contravariant vectors."""

    left_vector = _four_vector(left, name="left")
    right_vector = _four_vector(right, name="right")
    return float(left_vector @ MINKOWSKI_METRIC @ right_vector)


def electromagnetic_field_tensor_native(
    electric_field_native: Sequence[float], magnetic_field_native: Sequence[float]
) -> np.ndarray:
    """Construct native Gaussian ``F^(mu nu)`` from equally scaled ``E`` and ``B``."""

    electric = np.asarray(electric_field_native, dtype=float)
    magnetic = np.asarray(magnetic_field_native, dtype=float)
    if electric.shape != (3,) or not np.all(np.isfinite(electric)):
        raise ValueError("electric_field_native must contain three finite components")
    if magnetic.shape != (3,) or not np.all(np.isfinite(magnetic)):
        raise ValueError("magnetic_field_native must contain three finite components")

    ex, ey, ez = electric
    bx, by, bz = magnetic
    return np.array(
        [
            [0.0, -ex, -ey, -ez],
            [ex, 0.0, -bz, by],
            [ey, bz, 0.0, -bx],
            [ez, -by, bx, 0.0],
        ],
        dtype=float,
    )


def fields_from_tensor_native(
    field_tensor: TensorLike,
) -> Tuple[np.ndarray, np.ndarray]:
    """Recover native Gaussian ``(E, B)`` from a contravariant field tensor."""

    field = _field_tensor(field_tensor)
    electric = -field[0, 1:4]
    magnetic = np.array((-field[2, 3], field[1, 3], -field[1, 2]))
    return cast(Tuple[np.ndarray, np.ndarray], (electric, magnetic))


def hodge_dual(field_tensor: TensorLike) -> np.ndarray:
    """Return contravariant ``F*`` with ``epsilon_0123 = +1``.

    The definition is ``F*^(mu nu) = 1/2 epsilon^(mu nu alpha beta)
    F_(alpha beta)``. With the conventions above, ``F*^(0i) = B_i`` and
    applying the dual twice returns ``-F``.
    """

    field = _field_tensor(field_tensor)
    field_lower = MINKOWSKI_METRIC @ field @ MINKOWSKI_METRIC
    return cast(
        np.ndarray,
        0.5 * np.einsum("mnab,ab->mn", _LEVI_CIVITA_UPPER, field_lower),
    )


def magnetic_four_potential_covariant(
    field_tensor: TensorLike, spin_four_vector: VectorLike
) -> np.ndarray:
    """Return covariant ``B_mu = F*_(mu nu) a^nu`` in RFS notation."""

    dual_contravariant = hodge_dual(field_tensor)
    dual_covariant = MINKOWSKI_METRIC @ dual_contravariant @ MINKOWSKI_METRIC
    spin = _four_vector(spin_four_vector, name="spin_four_vector")
    return cast(np.ndarray, dual_covariant @ spin)


def rfs_g_tensor(partial_f: GradientLike, spin_four_vector: VectorLike) -> np.ndarray:
    """Return the full contravariant RFS tensor ``G^(mu nu)[a]``.

    The input derivative has ordering ``[lambda, mu, nu]`` and represents
    ``partial_lambda F^(mu nu)``. The observer spin is held fixed under this
    partial derivative, exactly as in the local RFS response law.
    """

    gradient = _field_gradient(partial_f)
    spin = _four_vector(spin_four_vector, name="spin_four_vector")

    # Lower only the two field indices before applying the Hodge dual. The
    # leading derivative index remains the supplied covariant lambda index.
    gradient_field_lower = np.einsum(
        "ma,lab,bn->lmn", MINKOWSKI_METRIC, gradient, MINKOWSKI_METRIC
    )
    dual_gradient_contravariant = 0.5 * np.einsum(
        "mnab,lab->lmn", _LEVI_CIVITA_UPPER, gradient_field_lower
    )
    dual_gradient_covariant = np.einsum(
        "ma,lab,bn->lmn",
        MINKOWSKI_METRIC,
        dual_gradient_contravariant,
        MINKOWSKI_METRIC,
    )

    # partial_lambda B_nu, with a treated as an observer variable rather than
    # as part of the supplied spacetime field.
    partial_b_covariant = np.einsum("lnr,r->ln", dual_gradient_covariant, spin)
    g_covariant = partial_b_covariant - partial_b_covariant.T
    return cast(np.ndarray, MINKOWSKI_METRIC @ g_covariant @ MINKOWSKI_METRIC)


def rfs_four_force_native(
    *,
    four_velocity_mm_ns: VectorLike,
    spin_four_vector: VectorLike,
    field_tensor: TensorLike,
    partial_f: GradientLike,
    charge_native: float,
    magnetic_moment_native: float,
) -> np.ndarray:
    """Return native ``dp^mu/dtau`` from the full RFS translational equation.

    The result has native force units ``amu mm/ns^2`` in all four components.
    ``spin_four_vector`` is normalized and ``magnetic_moment_native`` is signed
    relative to it.
    """

    charge = float(charge_native)
    moment = float(magnetic_moment_native)
    if not np.isfinite(charge) or not np.isfinite(moment):
        raise ValueError("charge_native and magnetic_moment_native must be finite")

    velocity = _four_vector(four_velocity_mm_ns, name="four_velocity_mm_ns")
    spin = _four_vector(spin_four_vector, name="spin_four_vector")
    field = _field_tensor(field_tensor)
    g_tensor = rfs_g_tensor(partial_f, spin)
    velocity_covariant = MINKOWSKI_METRIC @ velocity
    return cast(
        np.ndarray,
        (
            charge * (field @ velocity_covariant)
            + moment * (g_tensor @ velocity_covariant)
        )
        / C_MMNS,
    )


def rfs_spin_rhs_native(
    *,
    four_velocity_mm_ns: VectorLike,
    spin_four_vector: VectorLike,
    field_tensor: TensorLike,
    partial_f: GradientLike,
    charge_native: float,
    mass_amu: float,
    magnetic_moment_native: float,
    invariant_spin_native: float,
) -> np.ndarray:
    """Return the signed minimal ``da^mu/dtau`` in inverse nanoseconds."""

    charge = float(charge_native)
    mass = float(mass_amu)
    moment = float(magnetic_moment_native)
    invariant_spin = float(invariant_spin_native)
    if not np.isfinite(charge) or not np.isfinite(moment):
        raise ValueError("charge_native and magnetic_moment_native must be finite")
    if not np.isfinite(mass) or mass <= 0.0:
        raise ValueError("mass_amu must be finite and positive")
    if not np.isfinite(invariant_spin) or invariant_spin <= 0.0:
        raise ValueError("invariant_spin_native must be finite and positive")

    velocity = _four_vector(four_velocity_mm_ns, name="four_velocity_mm_ns")
    spin = _four_vector(spin_four_vector, name="spin_four_vector")
    field = _field_tensor(field_tensor)
    g_tensor = rfs_g_tensor(partial_f, spin)

    velocity_covariant = MINKOWSKI_METRIC @ velocity
    spin_covariant = MINKOWSKI_METRIC @ spin
    field_on_spin = field @ spin_covariant
    g_on_spin = g_tensor @ spin_covariant
    u_dot_f_dot_s = float(velocity_covariant @ field_on_spin)

    charge_to_mass_c = charge / (mass * C_MMNS)
    moment_to_spin = moment / invariant_spin
    orthogonal_field_on_spin = field_on_spin - (velocity * u_dot_f_dot_s / C_MMNS**2)

    return cast(
        np.ndarray,
        charge_to_mass_c * field_on_spin
        + (moment_to_spin - charge_to_mass_c) * orthogonal_field_on_spin
        + moment / (mass * C_MMNS) * g_on_spin,
    )


__all__ = [
    "MINKOWSKI_METRIC",
    "electromagnetic_field_tensor_native",
    "fields_from_tensor_native",
    "hodge_dual",
    "lower_four_vector",
    "magnetic_four_potential_covariant",
    "minkowski_dot",
    "rfs_four_force_native",
    "rfs_g_tensor",
    "rfs_spin_rhs_native",
]
