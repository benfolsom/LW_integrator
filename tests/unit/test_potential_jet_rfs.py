"""Equivalence tests for the potential-jet RFS algebra."""

from __future__ import annotations

import numpy as np
import pytest

from core.constants import C_MMNS
from core.magnetic_dipole import boost_rest_polarization
from core.potential_jet_rfs import potential_derivative_rfs_response_native
from core.rfs import rfs_four_force_native, rfs_spin_rhs_native

_SIGNS = np.asarray((1.0, -1.0, -1.0, -1.0), dtype=float)


def _field_tensor(partial_a: np.ndarray) -> np.ndarray:
    partial_up_a = _SIGNS[:, np.newaxis] * partial_a
    return partial_up_a - partial_up_a.T


def _field_gradient(partial2_a: np.ndarray) -> np.ndarray:
    return _SIGNS[np.newaxis, :, np.newaxis] * partial2_a - _SIGNS[
        np.newaxis, np.newaxis, :
    ] * np.swapaxes(partial2_a, 1, 2)


@pytest.mark.parametrize("seed", range(32))
def test_potential_derivative_response_matches_field_tensor_path(seed: int) -> None:
    rng = np.random.default_rng(seed)
    beta = rng.uniform(-0.75, 0.75, size=3)
    beta *= 0.92 / max(0.92, float(np.linalg.norm(beta)))
    gamma = 1.0 / np.sqrt(1.0 - float(beta @ beta))
    velocity = C_MMNS * gamma * np.concatenate(([1.0], beta))
    rest_spin = rng.normal(size=3)
    rest_spin /= np.linalg.norm(rest_spin)
    spin = boost_rest_polarization(rest_spin, beta)

    partial_a = rng.normal(scale=3.0e-4, size=(4, 4))
    raw_hessian = rng.normal(scale=8.0e-5, size=(4, 4, 4))
    partial2_a = 0.5 * (raw_hessian + np.swapaxes(raw_hessian, 0, 1))
    charge = float(rng.uniform(-2.0, 2.0))
    moment = float(rng.uniform(-3.0e-3, 3.0e-3))
    mass = float(rng.uniform(0.25, 4.0))
    invariant_spin = float(rng.uniform(0.1, 2.0))

    field_tensor = _field_tensor(partial_a)
    partial_f = _field_gradient(partial2_a)
    expected_force = rfs_four_force_native(
        four_velocity_mm_ns=velocity,
        spin_four_vector=spin,
        field_tensor=field_tensor,
        partial_f=partial_f,
        charge_native=charge,
        magnetic_moment_native=moment,
    )
    expected_spin = rfs_spin_rhs_native(
        four_velocity_mm_ns=velocity,
        spin_four_vector=spin,
        field_tensor=field_tensor,
        partial_f=partial_f,
        charge_native=charge,
        mass_amu=mass,
        magnetic_moment_native=moment,
        invariant_spin_native=invariant_spin,
    )

    result = potential_derivative_rfs_response_native(
        four_velocity_mm_ns=velocity,
        spin_four_vector=spin,
        partial_a=partial_a,
        partial2_a=partial2_a,
        charge_native=charge,
        mass_amu=mass,
        magnetic_moment_native=moment,
        invariant_spin_native=invariant_spin,
    )

    force_scale = max(float(np.linalg.norm(expected_force)), 1.0e-30)
    spin_scale = max(float(np.linalg.norm(expected_spin)), 1.0e-30)
    assert np.linalg.norm(result.total_four_force - expected_force) <= (
        2.0e-15 * force_scale
    )
    assert np.linalg.norm(result.spin_rhs - expected_spin) <= 3.0e-15 * spin_scale
    np.testing.assert_allclose(
        result.charge_four_force + result.dipole_four_force,
        result.total_four_force,
        rtol=0.0,
        atol=0.0,
    )


def test_diagonal_first_derivatives_do_not_change_mechanical_rfs_response() -> None:
    beta = np.asarray((0.2, -0.3, 0.1))
    gamma = 1.0 / np.sqrt(1.0 - float(beta @ beta))
    velocity = C_MMNS * gamma * np.concatenate(([1.0], beta))
    spin = boost_rest_polarization((0.3, 0.4, np.sqrt(0.75)), beta)
    partial_a = np.arange(16, dtype=float).reshape(4, 4) * 1.0e-6
    partial2_a = np.zeros((4, 4, 4), dtype=float)
    kwargs = dict(
        four_velocity_mm_ns=velocity,
        spin_four_vector=spin,
        partial2_a=partial2_a,
        charge_native=-1.0,
        mass_amu=1.0,
        magnetic_moment_native=2.0e-3,
        invariant_spin_native=0.5,
    )
    baseline = potential_derivative_rfs_response_native(
        partial_a=partial_a,
        **kwargs,
    )
    changed = partial_a.copy()
    changed[np.diag_indices(4)] += (1.0, 2.0, 3.0, 4.0)
    candidate = potential_derivative_rfs_response_native(
        partial_a=changed,
        **kwargs,
    )
    np.testing.assert_array_equal(candidate.total_four_force, baseline.total_four_force)
    np.testing.assert_array_equal(candidate.spin_rhs, baseline.spin_rhs)


def test_validation_rejects_noncommuting_potential_hessian() -> None:
    partial2_a = np.zeros((4, 4, 4), dtype=float)
    partial2_a[0, 1, 2] = 1.0
    with pytest.raises(ValueError, match="derivative indices must commute"):
        potential_derivative_rfs_response_native(
            four_velocity_mm_ns=(C_MMNS, 0.0, 0.0, 0.0),
            spin_four_vector=(0.0, 0.0, 0.0, 1.0),
            partial_a=np.zeros((4, 4)),
            partial2_a=partial2_a,
            charge_native=1.0,
            mass_amu=1.0,
            magnetic_moment_native=1.0,
            invariant_spin_native=1.0,
        )
