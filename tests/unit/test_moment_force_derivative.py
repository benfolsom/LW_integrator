"""Independent checks for the missing magnetic-moment Taylor coefficient."""

from itertools import permutations

import numpy as np
import pytest

from core.antisymmetric_response_rfs import (
    antisymmetric_response_moment_force_derivative_native as derivative,
    materialize_partial_antisymmetric_response_native as unpack,
    pack_partial_antisymmetric_response_native as pack,
)
from core.constants import C_MMNS
from core.magnetic_dipole import boost_rest_polarization
from core.potential_jet_rfs import potential_directional_rfs_reduction_jet_native
from core.rfs import rfs_four_force_native


def inputs(gamma=2.0):
    rng = np.random.default_rng(582)
    beta = np.array([0.6, -0.8, 0.0]) * np.sqrt(1 - 1 / gamma**2)
    return dict(
        four_velocity_mm_ns=gamma * C_MMNS * np.r_[1.0, beta],
        four_acceleration_mm_ns2=rng.normal(size=4),
        spin_four_vector=boost_rest_polarization(np.array([0.0, 0.0, 1.0]), beta),
        spin_four_vector_derivative_per_ns=rng.normal(size=4),
        partial_antisymmetric_response=rng.normal(scale=1e-3, size=(4, 6)),
        partial_antisymmetric_response_along_velocity=rng.normal(
            scale=1e-3, size=(4, 6)
        ),
        magnetic_moment_native=-0.7,
    )


def force(t, args):
    return rfs_four_force_native(
        four_velocity_mm_ns=args["four_velocity_mm_ns"]
        + t * args["four_acceleration_mm_ns2"],
        spin_four_vector=args["spin_four_vector"]
        + t * args["spin_four_vector_derivative_per_ns"],
        field_tensor=np.zeros((4, 4)),
        partial_f=unpack(
            args["partial_antisymmetric_response"]
            + t * args["partial_antisymmetric_response_along_velocity"]
        ),
        charge_native=0.0,
        magnetic_moment_native=args["magnetic_moment_native"],
    )


@pytest.mark.parametrize("gamma", [1.0, 2.0, 10.0, 100.0, 1000.0])
def test_full_product_rule_against_independent_dense_force(gamma):
    args = inputs(gamma)
    epsilon = 1e-3
    expected = (
        -force(2 * epsilon, args)
        + 8 * force(epsilon, args)
        - 8 * force(-epsilon, args)
        + force(-2 * epsilon, args)
    ) / (12 * epsilon)
    np.testing.assert_allclose(derivative(**args), expected, rtol=2e-10, atol=1e-11)


@pytest.mark.parametrize("gamma", [1.0, 2.0, 10.0, 1000.0])
def test_differentiated_force_orthogonality(gamma):
    args = inputs(gamma)
    u = args["four_velocity_mm_ns"]
    a = args["four_acceleration_mm_ns2"]
    signs = np.array([1, -1, -1, -1])
    kp = derivative(**args)
    k = force(0, args)
    residual = (signs * u) @ kp + (signs * a) @ k
    scale = np.sum(np.abs(u * kp)) + np.sum(np.abs(a * k))
    assert abs(residual) <= 2e-14 * scale


def field_gradient(hessian):
    signs = np.array([1, -1, -1, -1])
    return np.array(
        [
            [
                [
                    signs[i] * hessian[k, i, j] - signs[j] * hessian[k, j, i]
                    for j in range(4)
                ]
                for i in range(4)
            ]
            for k in range(4)
        ]
    )


def test_matches_existing_potential_based_derivative_reference():
    rng = np.random.default_rng(148)
    raw = rng.normal(scale=1e-4, size=(4, 4, 4))
    hessian = (raw + raw.swapaxes(0, 1)) / 2
    raw = rng.normal(scale=1e-6, size=(4, 4, 4, 4))
    third = sum(raw.transpose((*p, 3)) for p in permutations(range(3))) / 6
    u = inputs()["four_velocity_mm_ns"]
    s = inputs()["spin_four_vector"]
    along = np.einsum("k,klnm->lnm", u, third)
    # The reference requires bitwise commuting derivative indices. Remove
    # summation-order roundoff in this manufactured symmetric derivative.
    along = 0.5 * (along + along.swapaxes(0, 1))
    oracle = potential_directional_rfs_reduction_jet_native(
        four_velocity_mm_ns=u,
        spin_four_vector=s,
        partial_a=rng.normal(scale=1e-3, size=(4, 4)),
        partial2_a=hessian,
        partial3_a_along_velocity=along,
        partial3_a_along_acceleration=np.zeros((4, 4, 4)),
        partial4_a_along_velocity_twice=np.zeros((4, 4, 4)),
        charge_native=-0.4,
        mass_amu=2.0,
        magnetic_moment_native=-0.7,
        invariant_spin_native=1.0,
    )
    actual = derivative(
        four_velocity_mm_ns=u,
        four_acceleration_mm_ns2=oracle.four_acceleration,
        spin_four_vector=s,
        spin_four_vector_derivative_per_ns=oracle.normalized_spin_first_derivative,
        partial_antisymmetric_response=pack(field_gradient(hessian)),
        partial_antisymmetric_response_along_velocity=pack(field_gradient(along)),
        magnetic_moment_native=-0.7,
    )
    np.testing.assert_allclose(
        actual, oracle.dipole_four_force_first_derivative, rtol=2e-13, atol=1e-13
    )


@pytest.mark.parametrize("gamma", [1.0, 10.0, 1000.0])
def test_taylor_impulse_has_cubic_local_error(gamma):
    args = inputs(gamma)
    k = force(0, args)
    kp = derivative(**args)
    first = []
    second = []
    for h in (0.04, 0.02, 0.01, 0.005):
        # The manufactured force is cubic in proper time, so two-point
        # Gauss integration is an independent exact integral of this path.
        exact = (
            h
            / 2
            * (
                force(h / 2 * (1 - 1 / np.sqrt(3)), args)
                + force(h / 2 * (1 + 1 / np.sqrt(3)), args)
            )
        )
        first.append(np.linalg.norm(h * k - exact))
        second.append(np.linalg.norm(h * k + 0.5 * h * h * kp - exact))
    assert np.log2(first[-2] / first[-1]) > 1.95
    assert np.log2(second[-2] / second[-1]) > 2.95


@pytest.mark.parametrize("key", list(inputs()))
def test_nonfinite_inputs_rejected(key):
    args = inputs()
    args[key] = (
        np.full_like(args[key], np.nan) if key != "magnetic_moment_native" else np.nan
    )
    with pytest.raises(ValueError):
        derivative(**args)


@pytest.mark.parametrize("key", [k for k in inputs() if k != "magnetic_moment_native"])
def test_wrong_shapes_rejected(key):
    args = inputs()
    args[key] = np.zeros(3)
    with pytest.raises(ValueError):
        derivative(**args)


def test_zero_moment_gives_zero_correction():
    args = inputs()
    args["magnetic_moment_native"] = 0.0
    np.testing.assert_array_equal(derivative(**args), np.zeros(4))
