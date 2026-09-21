from __future__ import annotations

import numpy as np
import pytest

from core.constants import C_MMNS
from core.magnetic_dipole import HBAR_NATIVE
from core.rfs import (
    electromagnetic_field_tensor_native,
    minkowski_dot,
    rfs_charge_radiation_reaction_terms_native,
    rfs_four_force_native,
    rfs_spin_rhs_native,
)


def _four_velocity(beta: np.ndarray) -> np.ndarray:
    gamma = 1.0 / np.sqrt(1.0 - float(beta @ beta))
    return np.concatenate(([gamma * C_MMNS], gamma * C_MMNS * beta))


def _boost_rest_spin(rest_spin: np.ndarray, beta: np.ndarray) -> np.ndarray:
    gamma = 1.0 / np.sqrt(1.0 - float(beta @ beta))
    projection = float(beta @ rest_spin)
    spatial = rest_spin + gamma**2 / (gamma + 1.0) * projection * beta
    return np.concatenate(([gamma * projection], spatial))


def _assert_zero_relative(
    value: float, scale: float, *, relative_tolerance: float = 5.0e-13
) -> None:
    assert abs(value) <= relative_tolerance * max(scale, np.finfo(float).tiny)


def test_rest_frame_force_fixes_fermi_walker_sign() -> None:
    velocity = np.array((C_MMNS, 0.0, 0.0, 0.0))
    spin = np.array((0.0, 0.0, 0.0, 1.0))

    terms = rfs_charge_radiation_reaction_terms_native(
        four_velocity_mm_ns=velocity,
        spin_four_vector=spin,
        applied_radiation_reaction_force_native=(2.0, -3.0, 5.0),
        mass_amu=2.0,
    )

    np.testing.assert_allclose(
        terms.four_acceleration,
        (0.0, 1.0, -1.5, 2.5),
        rtol=0.0,
        atol=0.0,
    )
    # A_RR.a = -2.5 here, so the required minus sign makes delta da^0/dtau
    # positive and cancels the change of u.a caused by the acceleration.
    np.testing.assert_allclose(
        terms.spin_rhs_correction,
        (2.5 / C_MMNS, 0.0, 0.0, 0.0),
        rtol=2.0e-16,
        atol=0.0,
    )
    assert minkowski_dot(terms.four_acceleration, spin) + minkowski_dot(
        velocity, terms.spin_rhs_correction
    ) == pytest.approx(0.0, abs=2.0e-15)


def test_applied_charge_rr_terms_preserve_spin_constraints_instantaneously() -> None:
    rng = np.random.default_rng(20260824)
    for _ in range(32):
        beta = rng.uniform(-0.3, 0.3, size=3)
        velocity = _four_velocity(beta)
        rest_spin = rng.normal(size=3)
        rest_spin /= np.linalg.norm(rest_spin)
        spin = _boost_rest_spin(rest_spin, beta)
        applied_force = rng.uniform(-4.0, 4.0, size=3)
        mass = float(rng.uniform(0.2, 4.0))

        terms = rfs_charge_radiation_reaction_terms_native(
            four_velocity_mm_ns=velocity,
            spin_four_vector=spin,
            applied_radiation_reaction_force_native=applied_force,
            mass_amu=mass,
        )

        acceleration = terms.four_acceleration
        spin_rhs = terms.spin_rhs_correction
        _assert_zero_relative(
            minkowski_dot(velocity, acceleration),
            np.linalg.norm(velocity) * np.linalg.norm(acceleration),
        )
        _assert_zero_relative(
            minkowski_dot(acceleration, spin) + minkowski_dot(velocity, spin_rhs),
            np.linalg.norm(acceleration) * np.linalg.norm(spin)
            + np.linalg.norm(velocity) * np.linalg.norm(spin_rhs),
        )
        _assert_zero_relative(
            2.0 * minkowski_dot(spin, spin_rhs),
            2.0 * np.linalg.norm(spin) * np.linalg.norm(spin_rhs),
        )


def test_capped_applied_force_scales_both_returned_terms() -> None:
    velocity = _four_velocity(np.array((0.2, -0.1, 0.15)))
    spin = _boost_rest_spin(np.array((0.0, 1.0, 0.0)), velocity[1:] / velocity[0])
    uncapped_force = np.array((4.0, -5.0, 6.0))

    uncapped = rfs_charge_radiation_reaction_terms_native(
        four_velocity_mm_ns=velocity,
        spin_four_vector=spin,
        applied_radiation_reaction_force_native=uncapped_force,
        mass_amu=1.5,
    )
    cap_scale = 0.125
    applied = rfs_charge_radiation_reaction_terms_native(
        four_velocity_mm_ns=velocity,
        spin_four_vector=spin,
        applied_radiation_reaction_force_native=cap_scale * uncapped_force,
        mass_amu=1.5,
    )

    np.testing.assert_allclose(
        applied.four_acceleration, cap_scale * uncapped.four_acceleration
    )
    np.testing.assert_allclose(
        applied.spin_rhs_correction, cap_scale * uncapped.spin_rhs_correction
    )


def test_charge_rr_correction_preserves_constraints_with_full_rfs_rhs() -> None:
    beta = np.array((0.18, 0.11, -0.07))
    velocity = _four_velocity(beta)
    rest_spin = np.array((0.557, -0.371, 0.743))
    rest_spin /= np.linalg.norm(rest_spin)
    spin = _boost_rest_spin(rest_spin, beta)
    mass = 1.67
    field = electromagnetic_field_tensor_native((-3.0, 5.0, 2.0), (0.6, -0.1, 0.25))
    partial_f = np.stack(
        [
            electromagnetic_field_tensor_native((0.1, -0.2, 0.3), (0.5, 0.2, -0.3)),
            electromagnetic_field_tensor_native((0.4, 0.15, -0.25), (-0.4, 0.7, 0.1)),
            electromagnetic_field_tensor_native((-0.05, 0.07, 0.12), (0.6, -0.2, 0.9)),
            electromagnetic_field_tensor_native((0.2, -0.1, 0.08), (-0.3, 0.4, 0.2)),
        ]
    )

    rfs_acceleration = (
        rfs_four_force_native(
            four_velocity_mm_ns=velocity,
            spin_four_vector=spin,
            field_tensor=field,
            partial_f=partial_f,
            charge_native=1.2e-5,
            magnetic_moment_native=2.4e-15,
        )
        / mass
    )
    rfs_spin_rhs = rfs_spin_rhs_native(
        four_velocity_mm_ns=velocity,
        spin_four_vector=spin,
        field_tensor=field,
        partial_f=partial_f,
        charge_native=1.2e-5,
        mass_amu=mass,
        magnetic_moment_native=2.4e-15,
        invariant_spin_native=0.5 * HBAR_NATIVE,
    )
    rr_terms = rfs_charge_radiation_reaction_terms_native(
        four_velocity_mm_ns=velocity,
        spin_four_vector=spin,
        applied_radiation_reaction_force_native=(0.002, -0.003, 0.001),
        mass_amu=mass,
    )
    total_acceleration = rfs_acceleration + rr_terms.four_acceleration
    total_spin_rhs = rfs_spin_rhs + rr_terms.spin_rhs_correction

    _assert_zero_relative(
        2.0 * minkowski_dot(velocity, total_acceleration),
        2.0 * np.linalg.norm(velocity) * np.linalg.norm(total_acceleration),
    )
    _assert_zero_relative(
        2.0 * minkowski_dot(spin, total_spin_rhs),
        2.0 * np.linalg.norm(spin) * np.linalg.norm(total_spin_rhs),
    )
    _assert_zero_relative(
        minkowski_dot(total_acceleration, spin)
        + minkowski_dot(velocity, total_spin_rhs),
        np.linalg.norm(total_acceleration) * np.linalg.norm(spin)
        + np.linalg.norm(velocity) * np.linalg.norm(total_spin_rhs),
    )


def test_zero_applied_charge_rr_force_is_an_exact_no_op() -> None:
    terms = rfs_charge_radiation_reaction_terms_native(
        four_velocity_mm_ns=_four_velocity(np.array((0.1, -0.2, 0.05))),
        spin_four_vector=(0.2, 0.5, -0.1, 0.3),
        applied_radiation_reaction_force_native=(0.0, 0.0, 0.0),
        mass_amu=1.0,
    )

    np.testing.assert_array_equal(terms.four_acceleration, 0.0)
    np.testing.assert_array_equal(terms.spin_rhs_correction, 0.0)


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("mass_amu", 0.0, "mass_amu must be finite and positive"),
        (
            "applied_radiation_reaction_force_native",
            (1.0, 2.0),
            "applied_radiation_reaction_force_native must have shape",
        ),
        (
            "applied_radiation_reaction_force_native",
            (1.0, np.nan, 2.0),
            "must contain only finite values",
        ),
        (
            "four_velocity_mm_ns",
            (-C_MMNS, 0.0, 0.0, 0.0),
            "four_velocity_mm_ns must be future-directed",
        ),
    ],
)
def test_charge_rr_term_inputs_are_validated(
    keyword: str, value: object, message: str
) -> None:
    arguments: dict[str, object] = {
        "four_velocity_mm_ns": (C_MMNS, 0.0, 0.0, 0.0),
        "spin_four_vector": (0.0, 0.0, 0.0, 1.0),
        "applied_radiation_reaction_force_native": (1.0, 2.0, 3.0),
        "mass_amu": 1.0,
    }
    arguments[keyword] = value

    with pytest.raises(ValueError, match=message):
        rfs_charge_radiation_reaction_terms_native(**arguments)  # type: ignore[arg-type]
