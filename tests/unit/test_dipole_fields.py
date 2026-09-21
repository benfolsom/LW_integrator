from __future__ import annotations

import math

import numpy as np
import pytest

from core.dipole_fields import (
    DipoleFieldDomainError,
    DipoleSelfFieldError,
    evaluate_static_point_dipole_field_native,
    static_point_dipole_field_native,
)
from core.magnetic_dipole import (
    MAGNETIC_FIELD_NATIVE_TO_TESLA,
    NATIVE_ENERGY_UNIT_J,
    magnetic_moment_j_per_t_to_native,
)
from core.rfs import electromagnetic_field_tensor_native, fields_from_tensor_native
from core.species import get_species


def _field(
    separation: tuple[float, float, float],
    *,
    moment: float = 2.5,
    spin: tuple[float, float, float] = (0.0, 0.0, 1.0),
):
    return static_point_dipole_field_native(
        separation_vector_mm=separation,
        magnetic_moment_native=moment,
        rest_spin_direction=spin,
    )


def _curl_from_gradient(gradient: np.ndarray) -> np.ndarray:
    return np.array(
        (
            gradient[2, 1] - gradient[1, 2],
            gradient[0, 2] - gradient[2, 0],
            gradient[1, 0] - gradient[0, 1],
        )
    )


def _analytic_dipole_force(
    source_moment: np.ndarray,
    observer_moment: np.ndarray,
    separation: np.ndarray,
) -> np.ndarray:
    radius = float(np.linalg.norm(separation))
    direction = separation / radius
    source_on_direction = float(source_moment @ direction)
    observer_on_direction = float(observer_moment @ direction)
    return (
        3.0
        / radius**4
        * (
            source_on_direction * observer_moment
            + observer_on_direction * source_moment
            + float(source_moment @ observer_moment) * direction
            - 5.0 * source_on_direction * observer_on_direction * direction
        )
    )


def test_axial_and_equatorial_field_and_potential_signs() -> None:
    moment = 2.5
    radius = 4.0
    axial = _field((0.0, 0.0, radius), moment=moment)
    equatorial = _field((radius, 0.0, 0.0), moment=moment)

    np.testing.assert_allclose(axial.vector_potential_native, 0.0, atol=0.0)
    np.testing.assert_allclose(
        axial.magnetic_field_native,
        (0.0, 0.0, 2.0 * moment / radius**3),
        atol=0.0,
    )
    np.testing.assert_allclose(
        equatorial.vector_potential_native,
        (0.0, moment / radius**2, 0.0),
        atol=0.0,
    )
    np.testing.assert_allclose(
        equatorial.magnetic_field_native,
        (0.0, 0.0, -moment / radius**3),
        atol=0.0,
    )
    np.testing.assert_array_equal(axial.electric_field_native, np.zeros(3))
    np.testing.assert_array_equal(axial.four_potential_native[0], 0.0)


def test_potential_field_and_gradient_have_exact_radial_scaling() -> None:
    near = _field((2.0, -3.0, 4.0), spin=(0.0, 1.0, 0.0))
    far = _field((4.0, -6.0, 8.0), spin=(0.0, 1.0, 0.0))

    np.testing.assert_allclose(
        far.vector_potential_native, near.vector_potential_native / 4.0
    )
    np.testing.assert_allclose(
        far.magnetic_field_native, near.magnetic_field_native / 8.0
    )
    np.testing.assert_allclose(
        far.magnetic_gradient_native_per_mm,
        near.magnetic_gradient_native_per_mm / 16.0,
    )


def test_analytic_gradients_match_complete_centered_finite_differences() -> None:
    separation = np.array((1.3, -0.7, 2.1))
    spin = np.array((0.2, -0.4, math.sqrt(0.8)))
    center = static_point_dipole_field_native(
        separation_vector_mm=separation,
        magnetic_moment_native=-1.7,
        rest_spin_direction=spin,
    )
    step = 2.0e-6 * float(np.linalg.norm(separation))
    potential_gradient_fd = np.empty((3, 3))
    magnetic_gradient_fd = np.empty((3, 3))
    for coordinate in range(3):
        offset = np.zeros(3)
        offset[coordinate] = step
        lower = static_point_dipole_field_native(
            separation_vector_mm=separation - offset,
            magnetic_moment_native=-1.7,
            rest_spin_direction=spin,
        )
        upper = static_point_dipole_field_native(
            separation_vector_mm=separation + offset,
            magnetic_moment_native=-1.7,
            rest_spin_direction=spin,
        )
        potential_gradient_fd[:, coordinate] = (
            upper.vector_potential_native - lower.vector_potential_native
        ) / (2.0 * step)
        magnetic_gradient_fd[:, coordinate] = (
            upper.magnetic_field_native - lower.magnetic_field_native
        ) / (2.0 * step)

    np.testing.assert_allclose(
        center.vector_potential_gradient_native,
        potential_gradient_fd,
        rtol=3.0e-10,
        atol=5.0e-13,
    )
    np.testing.assert_allclose(
        center.magnetic_gradient_native_per_mm,
        magnetic_gradient_fd,
        rtol=2.0e-9,
        atol=2.0e-11,
    )


@pytest.mark.parametrize(
    "separation,spin",
    [
        ((1.0, 2.0, -0.5), (0.0, 0.0, 1.0)),
        ((-0.4, 0.8, 1.7), (1.0, 0.0, 0.0)),
        ((3.1, -2.2, 0.7), (0.0, 1.0, 0.0)),
    ],
)
def test_static_exterior_solution_has_curl_a_b_and_vacuum_div_curl_b(
    separation: tuple[float, float, float],
    spin: tuple[float, float, float],
) -> None:
    result = _field(separation, spin=spin)
    magnetic_from_potential = _curl_from_gradient(
        result.vector_potential_gradient_native
    )
    curl_magnetic = _curl_from_gradient(result.magnetic_gradient_native_per_mm)

    np.testing.assert_allclose(
        magnetic_from_potential,
        result.magnetic_field_native,
        rtol=3.0e-15,
        atol=2.0e-16,
    )
    assert np.trace(result.magnetic_gradient_native_per_mm) == pytest.approx(
        0.0, abs=2.0e-15
    )
    np.testing.assert_allclose(curl_magnetic, 0.0, atol=2.0e-15)


def test_field_tensor_and_partial_f_match_returned_fields_and_gradient() -> None:
    result = _field((0.8, -1.2, 2.3), spin=(1.0, 0.0, 0.0))
    electric, magnetic = fields_from_tensor_native(result.field_tensor)

    np.testing.assert_array_equal(electric, result.electric_field_native)
    np.testing.assert_array_equal(magnetic, result.magnetic_field_native)
    np.testing.assert_array_equal(result.partial_f[0], np.zeros((4, 4)))
    for coordinate in range(3):
        expected = electromagnetic_field_tensor_native(
            (0.0, 0.0, 0.0),
            result.magnetic_gradient_native_per_mm[:, coordinate],
        )
        np.testing.assert_array_equal(result.partial_f[coordinate + 1], expected)
    np.testing.assert_array_equal(
        result.partial_f + np.swapaxes(result.partial_f, 1, 2),
        np.zeros((4, 4, 4)),
    )


def test_native_result_matches_independent_si_point_dipole_oracle() -> None:
    proton = get_species("proton")
    assert proton.magnetic_moment_j_t is not None
    moment_si = proton.magnetic_moment_j_t
    moment_native = magnetic_moment_j_per_t_to_native(moment_si)
    spin = np.array((0.0, 0.6, 0.8))
    separation_mm = np.array((0.7, -1.1, 1.9))
    result = static_point_dipole_field_native(
        separation_vector_mm=separation_mm,
        magnetic_moment_native=moment_native,
        rest_spin_direction=spin,
    )

    # Independent SI vacuum solution.  The tiny relative tolerance accounts
    # for the measured post-2019 mu_0 versus the historical exact Gaussian-SI
    # conversion encoded by this solver's native field scale.
    mu_0_si = 1.25663706127e-6
    coefficient = mu_0_si / (4.0 * math.pi)
    separation_m = separation_mm * 1.0e-3
    radius_m = float(np.linalg.norm(separation_m))
    direction = separation_m / radius_m
    moment_vector_si = moment_si * spin
    expected_a_si = coefficient * np.cross(moment_vector_si, separation_m) / radius_m**3
    expected_b_si = (
        coefficient
        * (3.0 * direction * float(moment_vector_si @ direction) - moment_vector_si)
        / radius_m**3
    )
    expected_gradient_si = (
        3.0
        * coefficient
        / radius_m**4
        * (
            np.eye(3) * float(moment_vector_si @ direction)
            + np.outer(direction, moment_vector_si)
            + np.outer(moment_vector_si, direction)
            - 5.0 * float(moment_vector_si @ direction) * np.outer(direction, direction)
        )
    )
    native_a_si = (
        result.vector_potential_native * MAGNETIC_FIELD_NATIVE_TO_TESLA * 1.0e-3
    )
    native_b_si = result.magnetic_field_native * MAGNETIC_FIELD_NATIVE_TO_TESLA
    native_gradient_si = (
        result.magnetic_gradient_native_per_mm * MAGNETIC_FIELD_NATIVE_TO_TESLA * 1.0e3
    )

    np.testing.assert_allclose(native_a_si, expected_a_si, rtol=2.0e-10)
    np.testing.assert_allclose(native_b_si, expected_b_si, rtol=2.0e-10)
    np.testing.assert_allclose(native_gradient_si, expected_gradient_si, rtol=2.0e-10)
    assert MAGNETIC_FIELD_NATIVE_TO_TESLA**2 / NATIVE_ENERGY_UNIT_J == pytest.approx(
        100.0, rel=2.0e-15
    )


def test_dipole_field_reproduces_pair_energy_force_torque_and_action_reaction() -> None:
    source_moment = 2.3
    observer_moment = -1.4
    source_spin = np.array((0.0, 0.0, 1.0))
    observer_spin = np.array((0.6, 0.0, 0.8))
    separation = np.array((1.1, -0.7, 2.0))
    source_vector = source_moment * source_spin
    observer_vector = observer_moment * observer_spin

    at_observer = static_point_dipole_field_native(
        separation_vector_mm=separation,
        magnetic_moment_native=source_moment,
        rest_spin_direction=source_spin,
    )
    at_source = static_point_dipole_field_native(
        separation_vector_mm=-separation,
        magnetic_moment_native=observer_moment,
        rest_spin_direction=observer_spin,
    )
    interaction_energy = -float(observer_vector @ at_observer.magnetic_field_native)
    symmetric_energy = -float(source_vector @ at_source.magnetic_field_native)
    force_on_observer = at_observer.magnetic_gradient_native_per_mm.T @ observer_vector
    force_on_source = at_source.magnetic_gradient_native_per_mm.T @ source_vector
    torque_on_observer = np.cross(observer_vector, at_observer.magnetic_field_native)
    radius = float(np.linalg.norm(separation))
    direction = separation / radius
    analytic_field = (
        3.0 * direction * float(source_vector @ direction) - source_vector
    ) / radius**3
    analytic_torque = np.cross(observer_vector, analytic_field)

    assert interaction_energy == pytest.approx(symmetric_energy, rel=3.0e-15)
    np.testing.assert_allclose(
        force_on_observer,
        _analytic_dipole_force(source_vector, observer_vector, separation),
        rtol=3.0e-15,
        atol=2.0e-16,
    )
    np.testing.assert_allclose(
        force_on_source, -force_on_observer, rtol=3.0e-15, atol=2.0e-16
    )
    assert np.linalg.norm(torque_on_observer) > 0.0
    np.testing.assert_allclose(
        torque_on_observer, analytic_torque, rtol=3.0e-15, atol=2.0e-16
    )


def test_signed_electron_and_proton_moments_give_expected_orientations() -> None:
    electron = get_species("electron")
    proton = get_species("proton")
    assert electron.magnetic_moment_j_t is not None
    assert proton.magnetic_moment_j_t is not None
    electron_moment = magnetic_moment_j_per_t_to_native(electron.magnetic_moment_j_t)
    proton_moment = magnetic_moment_j_per_t_to_native(proton.magnetic_moment_j_t)
    proton_field = static_point_dipole_field_native(
        separation_vector_mm=(1.0, 0.0, 0.0),
        magnetic_moment_native=proton_moment,
        rest_spin_direction=(0.0, 0.0, 1.0),
    )

    electron_vector_parallel_spin = electron_moment * np.array((0.0, 0.0, 1.0))
    electron_vector_flipped_spin = electron_moment * np.array((0.0, 0.0, -1.0))
    energy_parallel_spin = -float(
        electron_vector_parallel_spin @ proton_field.magnetic_field_native
    )
    energy_flipped_spin = -float(
        electron_vector_flipped_spin @ proton_field.magnetic_field_native
    )

    assert electron_moment < 0.0
    assert proton_moment > 0.0
    assert proton_field.magnetic_field_native[2] < 0.0
    assert energy_parallel_spin < 0.0
    assert energy_flipped_spin > 0.0


def test_domain_errors_are_strict_and_no_softening_is_applied() -> None:
    with pytest.raises(DipoleFieldDomainError, match="coincides"):
        _field((0.0, 0.0, 0.0))
    with pytest.raises(DipoleFieldDomainError, match="at or inside"):
        static_point_dipole_field_native(
            separation_vector_mm=(0.25, 0.0, 0.0),
            magnetic_moment_native=1.0,
            rest_spin_direction=(0.0, 0.0, 1.0),
            minimum_separation_mm=0.25,
        )
    with pytest.raises(ValueError, match="unit vector"):
        _field((1.0, 0.0, 0.0), spin=(0.0, 0.0, 2.0))
    with pytest.raises(ValueError, match="finite and non-negative"):
        static_point_dipole_field_native(
            separation_vector_mm=(1.0, 0.0, 0.0),
            magnetic_moment_native=1.0,
            rest_spin_direction=(0.0, 0.0, 1.0),
            minimum_separation_mm=-1.0,
        )


def test_identity_self_exclusion_is_explicit_and_other_coincidence_still_errors() -> (
    None
):
    arguments = {
        "source_position_mm": (0.0, 0.0, 0.0),
        "observer_position_mm": (0.0, 0.0, 0.0),
        "magnetic_moment_native": -2.0,
        "rest_spin_direction": (0.0, 0.0, 1.0),
        "source_particle_id": 17,
        "observer_particle_id": 17,
    }
    with pytest.raises(DipoleSelfFieldError, match="intrinsic dipole self-field"):
        evaluate_static_point_dipole_field_native(**arguments)

    excluded = evaluate_static_point_dipole_field_native(**arguments, exclude_self=True)
    assert excluded.excluded
    assert excluded.separation_mm == 0.0
    assert excluded.magnetic_moment_vector_native[2] == -2.0
    np.testing.assert_array_equal(excluded.field_tensor, np.zeros((4, 4)))
    np.testing.assert_array_equal(excluded.partial_f, np.zeros((4, 4, 4)))

    with pytest.raises(DipoleFieldDomainError, match="coincides"):
        evaluate_static_point_dipole_field_native(
            **{
                **arguments,
                "observer_particle_id": 18,
                "exclude_self": True,
            }
        )
