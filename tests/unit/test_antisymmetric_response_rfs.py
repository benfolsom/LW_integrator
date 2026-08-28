"""Equivalence of packed analytical response contractions and RFS tensors."""

from __future__ import annotations

import numpy as np
import pytest

from core.antisymmetric_response_rfs import (
    antisymmetric_response_charge_force_native,
    antisymmetric_response_rfs_native,
    materialize_antisymmetric_response_native,
    materialize_partial_antisymmetric_response_native,
    pack_antisymmetric_response_native,
    pack_partial_antisymmetric_response_native,
)
from core.constants import C_MMNS
from core.magnetic_dipole import boost_rest_polarization
from core.rfs import rfs_four_force_native, rfs_spin_rhs_native

_PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def _unpack(packed: np.ndarray) -> np.ndarray:
    result = np.zeros((4, 4))
    for index, (first, second) in enumerate(_PAIRS):
        result[first, second] = packed[index]
        result[second, first] = -packed[index]
    return result


@pytest.mark.parametrize("seed", range(32))
def test_packed_response_matches_tensor_rfs(seed: int) -> None:
    rng = np.random.default_rng(seed)
    beta = rng.normal(size=3)
    beta *= rng.uniform(0.0, 0.97) / np.linalg.norm(beta)
    gamma = 1.0 / np.sqrt(1.0 - float(beta @ beta))
    velocity = C_MMNS * gamma * np.concatenate(([1.0], beta))
    rest_spin = rng.normal(size=3)
    rest_spin /= np.linalg.norm(rest_spin)
    spin = boost_rest_polarization(rest_spin, beta)
    packed = rng.normal(scale=2.0e-3, size=6)
    partial_packed = rng.normal(scale=4.0e-4, size=(4, 6))
    field = _unpack(packed)
    partial_f = np.stack([_unpack(row) for row in partial_packed])
    charge = float(rng.uniform(-2.0, 2.0))
    moment = float(rng.uniform(-2.0e-3, 2.0e-3))
    mass = float(rng.uniform(0.2, 3.0))
    invariant_spin = float(rng.uniform(0.1, 2.0))
    expected_force = rfs_four_force_native(
        four_velocity_mm_ns=velocity,
        spin_four_vector=spin,
        field_tensor=field,
        partial_f=partial_f,
        charge_native=charge,
        magnetic_moment_native=moment,
    )
    expected_spin = rfs_spin_rhs_native(
        four_velocity_mm_ns=velocity,
        spin_four_vector=spin,
        field_tensor=field,
        partial_f=partial_f,
        charge_native=charge,
        mass_amu=mass,
        magnetic_moment_native=moment,
        invariant_spin_native=invariant_spin,
    )
    result = antisymmetric_response_rfs_native(
        four_velocity_mm_ns=velocity,
        spin_four_vector=spin,
        antisymmetric_response=packed,
        partial_antisymmetric_response=partial_packed,
        charge_native=charge,
        mass_amu=mass,
        magnetic_moment_native=moment,
        invariant_spin_native=invariant_spin,
    )
    np.testing.assert_allclose(
        result.total_four_force, expected_force, rtol=3e-15, atol=1e-18
    )
    np.testing.assert_allclose(result.spin_rhs, expected_spin, rtol=4e-15, atol=1e-18)


def test_response_pack_materialize_and_charge_force_are_exact() -> None:
    packed = np.asarray((0.2, -0.3, 0.4, -0.5, 0.6, -0.7))
    partial_packed = np.arange(24, dtype=float).reshape(4, 6) / 17.0
    field = materialize_antisymmetric_response_native(packed)
    partial_f = materialize_partial_antisymmetric_response_native(partial_packed)
    np.testing.assert_array_equal(pack_antisymmetric_response_native(field), packed)
    np.testing.assert_array_equal(
        pack_partial_antisymmetric_response_native(partial_f), partial_packed
    )

    beta = np.asarray((0.2, -0.1, 0.05))
    gamma = 1.0 / np.sqrt(1.0 - float(beta @ beta))
    velocity = gamma * C_MMNS * np.concatenate(((1.0,), beta))
    expected = rfs_four_force_native(
        four_velocity_mm_ns=velocity,
        spin_four_vector=(0.0, 0.0, 0.0, 1.0),
        field_tensor=field,
        partial_f=np.zeros((4, 4, 4)),
        charge_native=-0.8,
        magnetic_moment_native=0.0,
    )
    actual = antisymmetric_response_charge_force_native(
        four_velocity_mm_ns=velocity,
        antisymmetric_response=packed,
        charge_native=-0.8,
    )
    np.testing.assert_allclose(actual, expected, rtol=3.0e-15, atol=1.0e-18)
