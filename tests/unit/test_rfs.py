from __future__ import annotations

from itertools import permutations

import numpy as np
import pytest

from core.constants import C_MMNS
from core.external_fields import AMU_KG, NATIVE_FORCE_UNIT_NEWTON
from core.magnetic_dipole import (
    HBAR_NATIVE,
    MAGNETIC_FIELD_NATIVE_TO_TESLA,
    NATIVE_ACTION_UNIT_J_S,
    NATIVE_ENERGY_UNIT_J,
    magnetic_moment_j_per_t_to_native,
)
from core.rfs import (
    MINKOWSKI_METRIC,
    electromagnetic_field_tensor_native,
    fields_from_tensor_native,
    hodge_dual,
    magnetic_four_potential_covariant,
    minkowski_dot,
    rfs_four_force_native,
    rfs_g_tensor,
    rfs_spin_rhs_native,
)

_C_SI = C_MMNS * 1.0e6
_LENGTH_UNIT_M = 1.0e-3
_TIME_UNIT_S = 1.0e-9
_VELOCITY_UNIT_M_S = _LENGTH_UNIT_M / _TIME_UNIT_S
_CHARGE_UNIT_C = NATIVE_FORCE_UNIT_NEWTON / (MAGNETIC_FIELD_NATIVE_TO_TESLA * _C_SI)
_MOMENT_UNIT_J_T = NATIVE_ENERGY_UNIT_J / MAGNETIC_FIELD_NATIVE_TO_TESLA


def _four_velocity(beta: np.ndarray) -> np.ndarray:
    gamma = 1.0 / np.sqrt(1.0 - float(beta @ beta))
    return np.concatenate(([gamma * C_MMNS], gamma * C_MMNS * beta))


def _boost_rest_spin(rest_spin: np.ndarray, beta: np.ndarray) -> np.ndarray:
    gamma = 1.0 / np.sqrt(1.0 - float(beta @ beta))
    projection = float(beta @ rest_spin)
    spatial = rest_spin + gamma**2 / (gamma + 1.0) * projection * beta
    return np.concatenate(([gamma * projection], spatial))


def _tensor_gradient_from_field_gradients(
    electric_gradient_native_per_mm: np.ndarray,
    magnetic_gradient_native_per_mm: np.ndarray,
) -> np.ndarray:
    partial_f = np.zeros((4, 4, 4), dtype=float)
    for coordinate in range(3):
        partial_f[coordinate + 1] = electromagnetic_field_tensor_native(
            electric_gradient_native_per_mm[:, coordinate],
            magnetic_gradient_native_per_mm[:, coordinate],
        )
    return partial_f


def _permutation_sign(indices: tuple[int, ...]) -> float:
    inversions = sum(
        indices[left] > indices[right]
        for left in range(4)
        for right in range(left + 1, 4)
    )
    return -1.0 if inversions % 2 else 1.0


_EPS_LOWER = np.zeros((4, 4, 4, 4))
for _indices in permutations(range(4)):
    _EPS_LOWER[_indices] = _permutation_sign(_indices)
_EPS_UPPER = -_EPS_LOWER


def _si_g_tensor(partial_f_si_per_m: np.ndarray, spin: np.ndarray) -> np.ndarray:
    gradient_lower = np.einsum(
        "ma,lab,bn->lmn",
        MINKOWSKI_METRIC,
        partial_f_si_per_m,
        MINKOWSKI_METRIC,
    )
    dual_gradient_up = 0.5 * np.einsum("mnab,lab->lmn", _EPS_UPPER, gradient_lower)
    dual_gradient_down = np.einsum(
        "ma,lab,bn->lmn",
        MINKOWSKI_METRIC,
        dual_gradient_up,
        MINKOWSKI_METRIC,
    )
    partial_b_down = np.einsum("lnr,r->ln", dual_gradient_down, spin)
    g_down = partial_b_down - partial_b_down.T
    return MINKOWSKI_METRIC @ g_down @ MINKOWSKI_METRIC


def _si_oracle(
    *,
    velocity_native: np.ndarray,
    spin: np.ndarray,
    field_native: np.ndarray,
    partial_native_per_mm: np.ndarray,
    charge_native: float,
    mass_amu: float,
    moment_native: float,
    invariant_spin_native: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the published SI equations as an independent unit oracle."""

    velocity_si = velocity_native * _VELOCITY_UNIT_M_S
    field_si = field_native * MAGNETIC_FIELD_NATIVE_TO_TESLA
    partial_si = partial_native_per_mm * MAGNETIC_FIELD_NATIVE_TO_TESLA / _LENGTH_UNIT_M
    charge_si = charge_native * _CHARGE_UNIT_C
    mass_si = mass_amu * AMU_KG
    moment_si = moment_native * _MOMENT_UNIT_J_T
    spin_invariant_si = invariant_spin_native * NATIVE_ACTION_UNIT_J_S

    velocity_down = MINKOWSKI_METRIC @ velocity_si
    spin_down = MINKOWSKI_METRIC @ spin
    g_si = _si_g_tensor(partial_si, spin)
    field_on_spin = field_si @ spin_down
    g_on_spin = g_si @ spin_down
    u_dot_f_dot_a = float(velocity_down @ field_on_spin)

    force_si = charge_si * (field_si @ velocity_down) + moment_si / _C_SI * (
        g_si @ velocity_down
    )
    charge_to_mass = charge_si / mass_si
    spin_rhs_si = (
        charge_to_mass * field_on_spin
        + (moment_si / spin_invariant_si - charge_to_mass)
        * (field_on_spin - velocity_si * u_dot_f_dot_a / _C_SI**2)
        + moment_si / (_C_SI * mass_si) * g_on_spin
    )
    return force_si / NATIVE_FORCE_UNIT_NEWTON, spin_rhs_si * _TIME_UNIT_S


def _assert_zero_relative(value: float, scale: float, *, rtol: float = 3.0e-13) -> None:
    assert abs(value) <= rtol * max(scale, np.finfo(float).tiny)


def test_field_tensor_and_dual_conventions_are_gaussian_native() -> None:
    electric = np.array((2.0, -3.0, 5.0))
    magnetic = np.array((0.7, -1.1, 0.4))
    field = electromagnetic_field_tensor_native(electric, magnetic)
    dual = hodge_dual(field)

    recovered_electric, recovered_magnetic = fields_from_tensor_native(field)
    np.testing.assert_allclose(recovered_electric, electric)
    np.testing.assert_allclose(recovered_magnetic, magnetic)
    np.testing.assert_allclose(field + field.T, 0.0)
    np.testing.assert_allclose(dual[0, 1:4], magnetic)
    assert dual[1, 2] == pytest.approx(-electric[2])
    assert dual[1, 3] == pytest.approx(electric[1])
    np.testing.assert_allclose(hodge_dual(dual), -field)


def test_tensor_convention_reproduces_native_lorentz_force() -> None:
    charge = -1.3e-5
    electric = np.array((3.0, -2.0, 4.0))
    magnetic = np.array((0.2, 0.5, -0.4))
    beta = np.array((0.2, -0.1, 0.3))
    velocity = _four_velocity(beta)
    gamma = velocity[0] / C_MMNS
    field = electromagnetic_field_tensor_native(electric, magnetic)

    four_force = rfs_four_force_native(
        four_velocity_mm_ns=velocity,
        spin_four_vector=(0.0, 0.0, 0.0, 1.0),
        field_tensor=field,
        partial_f=np.zeros((4, 4, 4)),
        charge_native=charge,
        magnetic_moment_native=0.0,
    )
    expected_spatial = charge * gamma * (electric + np.cross(beta, magnetic))
    np.testing.assert_allclose(four_force[1:], expected_spatial)


def test_magnetic_potential_has_negative_rest_interaction_energy() -> None:
    magnetic = np.array((0.3, -0.4, 0.8))
    spin = np.array((0.0, 0.2, -0.3, 0.5))
    moment = -2.1e-15
    potential_covariant = magnetic_four_potential_covariant(
        electromagnetic_field_tensor_native(np.zeros(3), magnetic), spin
    )
    rest_velocity = np.array((C_MMNS, 0.0, 0.0, 0.0))

    interaction_energy = moment / C_MMNS * float(potential_covariant @ rest_velocity)
    assert interaction_energy == pytest.approx(-moment * float(spin[1:] @ magnetic))


def test_neutral_signed_moment_precesses_in_native_uniform_b() -> None:
    spin_invariant = 0.5 * HBAR_NATIVE
    moment = magnetic_moment_j_per_t_to_native(-9.662_365_3e-27)
    magnetic_field = 2.0 / MAGNETIC_FIELD_NATIVE_TO_TESLA
    spin = np.array((0.0, 1.0, 0.0, 0.0))
    velocity = np.array((C_MMNS, 0.0, 0.0, 0.0))
    field = electromagnetic_field_tensor_native(
        (0.0, 0.0, 0.0), (0.0, 0.0, magnetic_field)
    )

    spin_rhs = rfs_spin_rhs_native(
        four_velocity_mm_ns=velocity,
        spin_four_vector=spin,
        field_tensor=field,
        partial_f=np.zeros((4, 4, 4)),
        charge_native=0.0,
        mass_amu=1.008_664_915_95,
        magnetic_moment_native=moment,
        invariant_spin_native=spin_invariant,
    )

    np.testing.assert_allclose(spin_rhs[[0, 1, 3]], 0.0, atol=0.0)
    assert spin_rhs[2] == pytest.approx(-moment * magnetic_field / spin_invariant)
    assert spin_rhs[2] > 0.0


def test_full_g_native_rest_force_is_gradient_of_mu_dot_b() -> None:
    moment = magnetic_moment_j_per_t_to_native(-9.662_365_3e-27)
    dbz_dx = 7.5 / MAGNETIC_FIELD_NATIVE_TO_TESLA * 1.0e-3
    spin = np.array((0.0, 0.0, 0.0, 1.0))
    velocity = np.array((C_MMNS, 0.0, 0.0, 0.0))
    magnetic_gradient = np.zeros((3, 3))
    magnetic_gradient[2, 0] = dbz_dx
    partial_f = _tensor_gradient_from_field_gradients(
        np.zeros((3, 3)), magnetic_gradient
    )

    force = rfs_four_force_native(
        four_velocity_mm_ns=velocity,
        spin_four_vector=spin,
        field_tensor=np.zeros((4, 4)),
        partial_f=partial_f,
        charge_native=0.0,
        magnetic_moment_native=moment,
    )

    np.testing.assert_allclose(force[[0, 2, 3]], 0.0, atol=0.0)
    assert force[1] == pytest.approx(moment * dbz_dx, rel=2.0e-13)


def test_randomized_native_kernel_matches_published_si_equations() -> None:
    rng = np.random.default_rng(20260824)
    for _ in range(32):
        beta = rng.uniform(-0.25, 0.25, 3)
        velocity = _four_velocity(beta)
        rest_spin = rng.normal(size=3)
        rest_spin /= np.linalg.norm(rest_spin)
        spin = _boost_rest_spin(rest_spin, beta)
        field = electromagnetic_field_tensor_native(
            rng.uniform(-4.0, 4.0, 3), rng.uniform(-3.0, 3.0, 3)
        )
        partial_f = np.stack(
            [
                electromagnetic_field_tensor_native(
                    rng.uniform(-0.8, 0.8, 3), rng.uniform(-0.6, 0.6, 3)
                )
                for _ in range(4)
            ]
        )
        charge = float(rng.uniform(-2.0e-5, 2.0e-5))
        mass = float(rng.uniform(0.2, 4.0))
        moment = float(rng.uniform(-5.0e-15, 5.0e-15))
        invariant_spin = float(rng.uniform(0.25, 2.0) * HBAR_NATIVE)

        native_force = rfs_four_force_native(
            four_velocity_mm_ns=velocity,
            spin_four_vector=spin,
            field_tensor=field,
            partial_f=partial_f,
            charge_native=charge,
            magnetic_moment_native=moment,
        )
        native_spin_rhs = rfs_spin_rhs_native(
            four_velocity_mm_ns=velocity,
            spin_four_vector=spin,
            field_tensor=field,
            partial_f=partial_f,
            charge_native=charge,
            mass_amu=mass,
            magnetic_moment_native=moment,
            invariant_spin_native=invariant_spin,
        )
        oracle_force, oracle_spin_rhs = _si_oracle(
            velocity_native=velocity,
            spin=spin,
            field_native=field,
            partial_native_per_mm=partial_f,
            charge_native=charge,
            mass_amu=mass,
            moment_native=moment,
            invariant_spin_native=invariant_spin,
        )
        np.testing.assert_allclose(native_force, oracle_force, rtol=8.0e-14, atol=1e-28)
        np.testing.assert_allclose(
            native_spin_rhs, oracle_spin_rhs, rtol=8.0e-14, atol=1e-28
        )


def test_native_rhs_preserves_all_three_covariant_constraints() -> None:
    beta = np.array((0.18, 0.11, -0.07))
    velocity = _four_velocity(beta)
    spin = _boost_rest_spin(np.array((0.557, -0.371, 0.743)), beta)
    mass = 1.67
    charge = 1.2e-5
    moment = 2.4e-15
    invariant_spin = 0.5 * HBAR_NATIVE
    field = electromagnetic_field_tensor_native((-3.0, 5.0, 2.0), (0.6, -0.1, 0.25))
    partial_f = np.stack(
        [
            electromagnetic_field_tensor_native((0.1, -0.2, 0.3), (0.5, 0.2, -0.3)),
            electromagnetic_field_tensor_native((0.4, 0.15, -0.25), (-0.4, 0.7, 0.1)),
            electromagnetic_field_tensor_native((-0.05, 0.07, 0.12), (0.6, -0.2, 0.9)),
            electromagnetic_field_tensor_native((0.2, -0.1, 0.08), (-0.3, 0.4, 0.2)),
        ]
    )

    force = rfs_four_force_native(
        four_velocity_mm_ns=velocity,
        spin_four_vector=spin,
        field_tensor=field,
        partial_f=partial_f,
        charge_native=charge,
        magnetic_moment_native=moment,
    )
    acceleration = force / mass
    spin_rhs = rfs_spin_rhs_native(
        four_velocity_mm_ns=velocity,
        spin_four_vector=spin,
        field_tensor=field,
        partial_f=partial_f,
        charge_native=charge,
        mass_amu=mass,
        magnetic_moment_native=moment,
        invariant_spin_native=invariant_spin,
    )

    _assert_zero_relative(
        2.0 * minkowski_dot(velocity, acceleration),
        2.0 * np.linalg.norm(velocity) * np.linalg.norm(acceleration),
    )
    _assert_zero_relative(
        2.0 * minkowski_dot(spin, spin_rhs),
        2.0 * np.linalg.norm(spin) * np.linalg.norm(spin_rhs),
    )
    _assert_zero_relative(
        minkowski_dot(acceleration, spin) + minkowski_dot(velocity, spin_rhs),
        np.linalg.norm(acceleration) * np.linalg.norm(spin)
        + np.linalg.norm(velocity) * np.linalg.norm(spin_rhs),
    )
