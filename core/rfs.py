"""Isolated Rafelski--Formanek--Steinmetz point-dipole kernel.

This module evaluates the local response of one classical point particle to a
*supplied* electromagnetic field.  It deliberately does not construct charge
or magnetic-dipole source fields and it does not advance an integrator state.

The translational equation uses the full antisymmetric ``G`` tensor from
Rafelski, Formanek, and Steinmetz, Eur. Phys. J. C 78, 6 (2018), Eqs. (14),
(17), and (18), https://doi.org/10.1140/epjc/s10052-017-5493-2.  The spin
right-hand side is their signed minimal solution as stated by Formanek,
Steinmetz, and Rafelski, Phys. Rev. A 103, 052218 (2021), Eqs. (3), (8), and
(11), https://doi.org/10.1103/PhysRevA.103.052218.

Conventions
-----------

* SI units throughout.
* Coordinates are ``x = (c t, x, y, z)`` and the metric is ``diag(+1,-1,-1,-1)``.
* ``F`` means contravariant ``F^(mu nu)`` with ``F^(0i) = -E_i/c`` and
  ``F^(ij) = -epsilon_ijk B_k``.
* ``partial_f[lambda, mu, nu]`` means ``partial_lambda F^(mu nu)``.  Its
  temporal derivative is therefore ``partial_0 = (1/c) partial_t`` and every
  derivative component has units of inverse metres times the field tensor.
* ``u`` is the contravariant four-velocity in m/s and ``s`` is the physical
  contravariant spin four-vector in J s.

The partial derivative used to form ``G`` acts on the supplied field while
holding the observer's spin fixed.  Source-spin retardation and derivatives,
self-field removal, and radiation reaction are responsibilities of the caller.
"""

from __future__ import annotations

from itertools import permutations
from typing import Sequence, Tuple, Union, cast

import numpy as np

SPEED_OF_LIGHT_M_S = 299_792_458.0
"""Speed of light in vacuum, exact in SI."""

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

    return cast(
        np.ndarray,
        MINKOWSKI_METRIC @ _four_vector(vector, name="vector"),
    )


def minkowski_dot(left: VectorLike, right: VectorLike) -> float:
    """Return the ``(+---)`` inner product of two contravariant vectors."""

    left_vector = _four_vector(left, name="left")
    right_vector = _four_vector(right, name="right")
    return float(left_vector @ MINKOWSKI_METRIC @ right_vector)


def electromagnetic_field_tensor_si(
    electric_field_v_m: Sequence[float], magnetic_field_t: Sequence[float]
) -> np.ndarray:
    """Construct contravariant ``F^(mu nu)`` from SI electric and magnetic fields."""

    electric = np.asarray(electric_field_v_m, dtype=float)
    magnetic = np.asarray(magnetic_field_t, dtype=float)
    if electric.shape != (3,) or not np.all(np.isfinite(electric)):
        raise ValueError("electric_field_v_m must contain three finite components")
    if magnetic.shape != (3,) or not np.all(np.isfinite(magnetic)):
        raise ValueError("magnetic_field_t must contain three finite components")

    ex, ey, ez = electric / SPEED_OF_LIGHT_M_S
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


def fields_from_tensor_si(field_tensor: TensorLike) -> Tuple[np.ndarray, np.ndarray]:
    """Recover SI ``(E, B)`` from a contravariant field tensor."""

    field = _field_tensor(field_tensor)
    electric = -SPEED_OF_LIGHT_M_S * field[0, 1:4]
    magnetic = np.array((-field[2, 3], field[1, 3], -field[1, 2]))
    return cast(Tuple[np.ndarray, np.ndarray], (electric, magnetic))


def hodge_dual(field_tensor: TensorLike) -> np.ndarray:
    """Return contravariant ``F*`` with ``epsilon_0123 = +1``.

    The definition is ``F*^(mu nu) = 1/2 epsilon^(mu nu alpha beta)
    F_(alpha beta)``.  With the conventions above, ``F*^(0i) = B_i`` and
    applying the dual twice returns ``-F``.
    """

    field = _field_tensor(field_tensor)
    field_lower = MINKOWSKI_METRIC @ field @ MINKOWSKI_METRIC
    return cast(
        np.ndarray,
        0.5 * np.einsum("mnab,ab->mn", _LEVI_CIVITA_UPPER, field_lower),
    )


def dipole_charge_from_moment_si(
    magnetic_moment_j_per_t: float, rest_spin_magnitude_j_s: float
) -> float:
    """Return the signed RFS dipole charge ``d = mu/(c S)``.

    ``magnetic_moment_j_per_t`` is signed relative to spin.  The invariant
    stretched-state spin magnitude ``S`` must be positive.  A partially
    polarized spin vector may have a smaller magnitude while retaining this
    species coupling.
    """

    moment = float(magnetic_moment_j_per_t)
    spin_magnitude = float(rest_spin_magnitude_j_s)
    if not np.isfinite(moment):
        raise ValueError("magnetic_moment_j_per_t must be finite")
    if not np.isfinite(spin_magnitude) or spin_magnitude <= 0.0:
        raise ValueError("rest_spin_magnitude_j_s must be finite and positive")
    return moment / (SPEED_OF_LIGHT_M_S * spin_magnitude)


def magnetic_four_potential_covariant(
    field_tensor: TensorLike, spin_four_vector_j_s: VectorLike
) -> np.ndarray:
    """Return covariant ``B_mu = F*_(mu nu) s^nu`` in RFS notation."""

    dual_contravariant = hodge_dual(field_tensor)
    dual_covariant = MINKOWSKI_METRIC @ dual_contravariant @ MINKOWSKI_METRIC
    spin = _four_vector(spin_four_vector_j_s, name="spin_four_vector_j_s")
    return cast(np.ndarray, dual_covariant @ spin)


def rfs_g_tensor(
    partial_f: GradientLike, spin_four_vector_j_s: VectorLike
) -> np.ndarray:
    """Return the full contravariant RFS tensor ``G^(mu nu)``.

    The input derivative has ordering ``[lambda, mu, nu]`` and represents
    ``partial_lambda F^(mu nu)``.  The observer spin is held fixed under this
    partial derivative, exactly as in the local RFS response law.
    """

    gradient = _field_gradient(partial_f)
    spin = _four_vector(spin_four_vector_j_s, name="spin_four_vector_j_s")

    # Lower only the two field indices before applying the Hodge dual.  The
    # leading derivative index remains the supplied covariant lambda index.
    gradient_field_lower = np.einsum(
        "ma,lab,bn->lmn",
        MINKOWSKI_METRIC,
        gradient,
        MINKOWSKI_METRIC,
    )
    dual_gradient_contravariant = 0.5 * np.einsum(
        "mnab,lab->lmn",
        _LEVI_CIVITA_UPPER,
        gradient_field_lower,
    )
    dual_gradient_covariant = np.einsum(
        "ma,lab,bn->lmn",
        MINKOWSKI_METRIC,
        dual_gradient_contravariant,
        MINKOWSKI_METRIC,
    )

    # partial_lambda B_nu, with s treated as an observer variable rather than
    # as part of the supplied spacetime field.
    partial_b_covariant = np.einsum(
        "lnr,r->ln",
        dual_gradient_covariant,
        spin,
    )
    g_covariant = partial_b_covariant - partial_b_covariant.T
    g_contravariant = MINKOWSKI_METRIC @ g_covariant @ MINKOWSKI_METRIC
    return cast(np.ndarray, g_contravariant)


def rfs_four_force_si(
    *,
    four_velocity_m_s: VectorLike,
    spin_four_vector_j_s: VectorLike,
    field_tensor: TensorLike,
    partial_f: GradientLike,
    charge_coulomb: float,
    dipole_charge: float,
) -> np.ndarray:
    """Return ``dp^mu/dtau`` from the full RFS translational equation.

    The result has force units (N) in all four components.  ``dipole_charge``
    is the signed coupling returned by :func:`dipole_charge_from_moment_si`.
    """

    charge = float(charge_coulomb)
    coupling = float(dipole_charge)
    if not np.isfinite(charge) or not np.isfinite(coupling):
        raise ValueError("charge_coulomb and dipole_charge must be finite")

    velocity = _four_vector(four_velocity_m_s, name="four_velocity_m_s")
    spin = _four_vector(spin_four_vector_j_s, name="spin_four_vector_j_s")
    field = _field_tensor(field_tensor)
    g_tensor = rfs_g_tensor(partial_f, spin)
    velocity_covariant = MINKOWSKI_METRIC @ velocity
    return cast(
        np.ndarray,
        charge * (field @ velocity_covariant)
        + coupling * (g_tensor @ velocity_covariant),
    )


def rfs_spin_rhs_si(
    *,
    four_velocity_m_s: VectorLike,
    spin_four_vector_j_s: VectorLike,
    field_tensor: TensorLike,
    partial_f: GradientLike,
    charge_coulomb: float,
    mass_kg: float,
    dipole_charge: float,
) -> np.ndarray:
    """Return the signed 2021 minimal ``ds^mu/dtau`` in SI units.

    The anomalous coefficient is not supplied independently.  It follows the
    RFS relation ``a_tilde = c d - q/m``, so charged, neutral, positive-moment,
    and negative-moment particles all share the same regular equation.
    """

    charge = float(charge_coulomb)
    mass = float(mass_kg)
    coupling = float(dipole_charge)
    if not np.isfinite(charge) or not np.isfinite(coupling):
        raise ValueError("charge_coulomb and dipole_charge must be finite")
    if not np.isfinite(mass) or mass <= 0.0:
        raise ValueError("mass_kg must be finite and positive")

    velocity = _four_vector(four_velocity_m_s, name="four_velocity_m_s")
    spin = _four_vector(spin_four_vector_j_s, name="spin_four_vector_j_s")
    field = _field_tensor(field_tensor)
    g_tensor = rfs_g_tensor(partial_f, spin)

    velocity_covariant = MINKOWSKI_METRIC @ velocity
    spin_covariant = MINKOWSKI_METRIC @ spin
    field_on_spin = field @ spin_covariant
    g_on_spin = g_tensor @ spin_covariant
    u_dot_f_dot_s = float(velocity_covariant @ field_on_spin)

    charge_to_mass = charge / mass
    moment_excess = SPEED_OF_LIGHT_M_S * coupling - charge_to_mass
    orthogonal_field_on_spin = field_on_spin - (
        velocity * u_dot_f_dot_s / SPEED_OF_LIGHT_M_S**2
    )

    return cast(
        np.ndarray,
        charge_to_mass * field_on_spin
        + moment_excess * orthogonal_field_on_spin
        + coupling / mass * g_on_spin,
    )


__all__ = [
    "MINKOWSKI_METRIC",
    "SPEED_OF_LIGHT_M_S",
    "dipole_charge_from_moment_si",
    "electromagnetic_field_tensor_si",
    "fields_from_tensor_si",
    "hodge_dual",
    "lower_four_vector",
    "magnetic_four_potential_covariant",
    "minkowski_dot",
    "rfs_four_force_si",
    "rfs_g_tensor",
    "rfs_spin_rhs_si",
]
