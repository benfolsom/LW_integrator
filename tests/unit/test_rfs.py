from __future__ import annotations

import numpy as np
import pytest

from core.rfs import (
    MINKOWSKI_METRIC,
    SPEED_OF_LIGHT_M_S,
    dipole_charge_from_moment_si,
    electromagnetic_field_tensor_si,
    fields_from_tensor_si,
    hodge_dual,
    magnetic_four_potential_covariant,
    minkowski_dot,
    rfs_four_force_si,
    rfs_g_tensor,
    rfs_spin_rhs_si,
)


def _four_velocity(beta: np.ndarray) -> np.ndarray:
    gamma = 1.0 / np.sqrt(1.0 - float(beta @ beta))
    return np.concatenate(
        ([gamma * SPEED_OF_LIGHT_M_S], gamma * SPEED_OF_LIGHT_M_S * beta)
    )


def _boost_rest_spin(rest_spin: np.ndarray, beta: np.ndarray) -> np.ndarray:
    gamma = 1.0 / np.sqrt(1.0 - float(beta @ beta))
    projection = float(beta @ rest_spin)
    spatial = rest_spin + gamma**2 / (gamma + 1.0) * projection * beta
    return np.concatenate(([gamma * projection], spatial))


def _tensor_gradient_from_field_gradients(
    electric_gradient_v_m2: np.ndarray,
    magnetic_gradient_t_m: np.ndarray,
) -> np.ndarray:
    """Return spatial partials with ``gradient[field component, coordinate]``."""

    partial_f = np.zeros((4, 4, 4), dtype=float)
    for coordinate in range(3):
        partial_f[coordinate + 1] = electromagnetic_field_tensor_si(
            electric_gradient_v_m2[:, coordinate],
            magnetic_gradient_t_m[:, coordinate],
        )
    return partial_f


def _uniform_velocity_charge_field_and_gradient(
    *,
    source_charge_coulomb: float,
    source_beta: np.ndarray,
    instantaneous_separation_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact Heaviside field and ``partial_lambda F^(mu nu)``.

    The source moves uniformly, so the Lienard--Wiechert field is a rigidly
    translated Heaviside field.  The temporal entry uses
    ``partial_0 = (1/c) partial_t = -(beta dot grad)``.
    """

    coulomb_constant = 8.987_551_792_3e9
    beta_squared = float(source_beta @ source_beta)
    shape_matrix = (1.0 - beta_squared) * np.eye(3) + np.outer(source_beta, source_beta)
    denominator_squared = float(
        instantaneous_separation_m @ shape_matrix @ instantaneous_separation_m
    )
    prefactor = coulomb_constant * source_charge_coulomb * (1.0 - beta_squared)
    electric = prefactor * instantaneous_separation_m / denominator_squared**1.5
    isotropic_gradient = np.eye(3) / denominator_squared**1.5
    directional_scale = 3.0 / denominator_squared**2.5
    directional_gradient = directional_scale * np.outer(
        instantaneous_separation_m,
        shape_matrix @ instantaneous_separation_m,
    )
    electric_gradient = prefactor * (isotropic_gradient - directional_gradient)
    magnetic = np.cross(source_beta, electric) / SPEED_OF_LIGHT_M_S
    magnetic_gradient = np.column_stack(
        [
            np.cross(source_beta, electric_gradient[:, coordinate]) / SPEED_OF_LIGHT_M_S
            for coordinate in range(3)
        ]
    )

    partial_f = _tensor_gradient_from_field_gradients(
        electric_gradient, magnetic_gradient
    )
    partial_f[0] = electromagnetic_field_tensor_si(
        -(electric_gradient @ source_beta),
        -(magnetic_gradient @ source_beta),
    )
    return electromagnetic_field_tensor_si(electric, magnetic), partial_f


def _assert_zero_relative(value: float, scale: float, *, rtol: float = 2.0e-13) -> None:
    assert abs(value) <= rtol * max(scale, np.finfo(float).tiny)


def test_field_tensor_and_dual_conventions() -> None:
    electric = np.array((2.0e6, -3.0e6, 5.0e6))
    magnetic = np.array((0.7, -1.1, 0.4))
    field = electromagnetic_field_tensor_si(electric, magnetic)
    dual = hodge_dual(field)

    recovered_electric, recovered_magnetic = fields_from_tensor_si(field)
    np.testing.assert_allclose(recovered_electric, electric)
    np.testing.assert_allclose(recovered_magnetic, magnetic)
    np.testing.assert_allclose(field + field.T, 0.0)

    # epsilon_0123=+1 and (+---) give F*^(0i)=B_i.
    np.testing.assert_allclose(dual[0, 1:4], magnetic)
    assert dual[1, 2] == pytest.approx(-electric[2] / SPEED_OF_LIGHT_M_S)
    assert dual[1, 3] == pytest.approx(electric[1] / SPEED_OF_LIGHT_M_S)
    np.testing.assert_allclose(hodge_dual(dual), -field)


def test_tensor_convention_reproduces_lorentz_force() -> None:
    charge = -1.6e-19
    electric = np.array((3.0e5, -2.0e5, 4.0e5))
    magnetic = np.array((0.2, 0.5, -0.4))
    beta = np.array((0.2, -0.1, 0.3))
    velocity = _four_velocity(beta)
    gamma = velocity[0] / SPEED_OF_LIGHT_M_S
    field = electromagnetic_field_tensor_si(electric, magnetic)

    four_force = charge * field @ (MINKOWSKI_METRIC @ velocity)
    expected_spatial = (
        charge * gamma * (electric + np.cross(beta * SPEED_OF_LIGHT_M_S, magnetic))
    )

    np.testing.assert_allclose(four_force[1:], expected_spatial)


def test_magnetic_potential_has_negative_rest_interaction_energy() -> None:
    magnetic = np.array((0.3, -0.4, 0.8))
    spin = np.array((0.0, 2.0e-35, -3.0e-35, 5.0e-35))
    moment = -9.0e-27
    spin_magnitude = np.linalg.norm(spin[1:])
    coupling = dipole_charge_from_moment_si(moment, spin_magnitude)
    potential_covariant = magnetic_four_potential_covariant(
        electromagnetic_field_tensor_si(np.zeros(3), magnetic), spin
    )
    rest_velocity = np.array((SPEED_OF_LIGHT_M_S, 0.0, 0.0, 0.0))

    interaction_energy = coupling * float(potential_covariant @ rest_velocity)
    moment_vector = moment * spin[1:] / spin_magnitude

    assert interaction_energy == pytest.approx(-float(moment_vector @ magnetic))


def test_neutral_signed_moment_precesses_in_uniform_magnetic_field() -> None:
    spin_magnitude = 0.5 * 1.054_571_817e-34
    magnetic_moment = -9.662_365_3e-27
    magnetic_field = 2.0
    spin = np.array((0.0, spin_magnitude, 0.0, 0.0))
    velocity = np.array((SPEED_OF_LIGHT_M_S, 0.0, 0.0, 0.0))
    field = electromagnetic_field_tensor_si((0.0, 0.0, 0.0), (0.0, 0.0, magnetic_field))
    partial_f = np.zeros((4, 4, 4))
    coupling = dipole_charge_from_moment_si(magnetic_moment, spin_magnitude)

    spin_rhs = rfs_spin_rhs_si(
        four_velocity_m_s=velocity,
        spin_four_vector_j_s=spin,
        field_tensor=field,
        partial_f=partial_f,
        charge_coulomb=0.0,
        mass_kg=1.674_927_500_56e-27,
        dipole_charge=coupling,
    )

    np.testing.assert_allclose(spin_rhs[[0, 1, 3]], 0.0, atol=0.0)
    assert spin_rhs[2] == pytest.approx(-magnetic_moment * magnetic_field)
    assert spin_rhs[2] > 0.0


def test_full_g_current_region_static_rest_force_is_gradient_of_mu_dot_b() -> None:
    spin_magnitude = 0.5 * 1.054_571_817e-34
    magnetic_moment = -9.662_365_3e-27
    dbz_dx = 7.5
    spin = np.array((0.0, 0.0, 0.0, spin_magnitude))
    velocity = np.array((SPEED_OF_LIGHT_M_S, 0.0, 0.0, 0.0))

    # B_z = (dB_z/dx) x has nonzero curl and therefore represents a
    # current-containing region.  It deliberately certifies the full G tensor,
    # not the compact source-free Gilbert form.
    magnetic_gradient = np.zeros((3, 3))
    magnetic_gradient[2, 0] = dbz_dx
    partial_f = _tensor_gradient_from_field_gradients(
        np.zeros((3, 3)), magnetic_gradient
    )
    field = electromagnetic_field_tensor_si(np.zeros(3), np.zeros(3))
    coupling = dipole_charge_from_moment_si(magnetic_moment, spin_magnitude)

    g_tensor = rfs_g_tensor(partial_f, spin)
    four_force = rfs_four_force_si(
        four_velocity_m_s=velocity,
        spin_four_vector_j_s=spin,
        field_tensor=field,
        partial_f=partial_f,
        charge_coulomb=0.0,
        dipole_charge=coupling,
    )

    np.testing.assert_allclose(g_tensor + g_tensor.T, 0.0)
    np.testing.assert_allclose(four_force[[0, 2, 3]], 0.0, atol=0.0)
    assert four_force[1] == pytest.approx(
        magnetic_moment * dbz_dx, rel=2.0e-13, abs=0.0
    )


def test_full_g_source_free_axial_gradient_preserves_spin_velocity_constraint() -> None:
    spin_magnitude = 0.5 * 1.054_571_817e-34
    magnetic_moment = -9.662_365_3e-27
    mass = 1.674_927_500_56e-27
    axial_gradient = 7.5
    spin = np.array((0.0, 0.0, 0.0, spin_magnitude))
    velocity = np.array((SPEED_OF_LIGHT_M_S, 0.0, 0.0, 0.0))

    # B=(-g x/2, -g y/2, g z) is both divergence-free and curl-free:
    # its gradient is symmetric and traceless, as required in vacuum.
    magnetic_gradient = np.diag(
        (-0.5 * axial_gradient, -0.5 * axial_gradient, axial_gradient)
    )
    assert np.trace(magnetic_gradient) == pytest.approx(0.0, abs=0.0)
    np.testing.assert_allclose(magnetic_gradient, magnetic_gradient.T, atol=0.0)
    partial_f = _tensor_gradient_from_field_gradients(
        np.zeros((3, 3)), magnetic_gradient
    )
    field = electromagnetic_field_tensor_si(np.zeros(3), np.zeros(3))
    coupling = dipole_charge_from_moment_si(magnetic_moment, spin_magnitude)

    g_tensor = rfs_g_tensor(partial_f, spin)
    four_force = rfs_four_force_si(
        four_velocity_m_s=velocity,
        spin_four_vector_j_s=spin,
        field_tensor=field,
        partial_f=partial_f,
        charge_coulomb=0.0,
        dipole_charge=coupling,
    )
    spin_rhs = rfs_spin_rhs_si(
        four_velocity_m_s=velocity,
        spin_four_vector_j_s=spin,
        field_tensor=field,
        partial_f=partial_f,
        charge_coulomb=0.0,
        mass_kg=mass,
        dipole_charge=coupling,
    )

    expected_force_z = magnetic_moment * axial_gradient
    expected_spin_time_rhs = (
        expected_force_z * spin_magnitude / (mass * SPEED_OF_LIGHT_M_S)
    )
    assert g_tensor[3, 0] == pytest.approx(
        spin_magnitude * axial_gradient, rel=2.0e-13, abs=0.0
    )
    np.testing.assert_allclose(four_force[:3], 0.0, atol=0.0)
    assert four_force[3] == pytest.approx(expected_force_z, rel=2.0e-13, abs=0.0)
    np.testing.assert_allclose(spin_rhs[1:], 0.0, atol=0.0)
    assert spin_rhs[0] == pytest.approx(expected_spin_time_rhs, rel=2.0e-13, abs=0.0)

    acceleration = four_force / mass
    constraint_rhs = minkowski_dot(acceleration, spin) + minkowski_dot(
        velocity, spin_rhs
    )
    constraint_scale = np.linalg.norm(acceleration) * np.linalg.norm(
        spin
    ) + np.linalg.norm(velocity) * np.linalg.norm(spin_rhs)
    _assert_zero_relative(constraint_rhs, constraint_scale)


def test_uniformly_moving_charge_full_g_includes_temporal_field_gradient() -> None:
    source_charge = 1.0e-9
    source_beta_x = 0.3
    impact_parameter = 0.4
    source_beta = np.array((source_beta_x, 0.0, 0.0))
    separation = np.array((0.0, impact_parameter, 0.0))
    field, partial_f = _uniform_velocity_charge_field_and_gradient(
        source_charge_coulomb=source_charge,
        source_beta=source_beta,
        instantaneous_separation_m=separation,
    )

    spin_magnitude = 0.5 * 1.054_571_817e-34
    magnetic_moment = 2.0e-26
    spin = np.array((0.0, 0.0, 0.0, spin_magnitude))
    velocity = np.array((SPEED_OF_LIGHT_M_S, 0.0, 0.0, 0.0))
    coupling = dipole_charge_from_moment_si(magnetic_moment, spin_magnitude)
    four_force = rfs_four_force_si(
        four_velocity_m_s=velocity,
        spin_four_vector_j_s=spin,
        field_tensor=field,
        partial_f=partial_f,
        charge_coulomb=0.0,
        dipole_charge=coupling,
    )

    coulomb_constant = 8.987_551_792_3e9
    source_gamma = 1.0 / np.sqrt(1.0 - source_beta_x**2)
    force_numerator = magnetic_moment * coulomb_constant * source_charge * source_gamma
    expected_force_y = -(force_numerator * source_beta_x) / (
        SPEED_OF_LIGHT_M_S * impact_parameter**3
    )
    np.testing.assert_allclose(four_force[[0, 1, 3]], 0.0, atol=0.0)
    assert four_force[2] == pytest.approx(expected_force_y, rel=2.0e-13, abs=0.0)

    # At closest approach, grad(mu.B) alone is twice the covariant result.
    # The temporal electric-field derivative supplies the other half.
    spatial_derivatives_only = partial_f.copy()
    spatial_derivatives_only[0] = 0.0
    force_without_temporal_gradient = rfs_four_force_si(
        four_velocity_m_s=velocity,
        spin_four_vector_j_s=spin,
        field_tensor=field,
        partial_f=spatial_derivatives_only,
        charge_coulomb=0.0,
        dipole_charge=coupling,
    )
    assert force_without_temporal_gradient[2] == pytest.approx(
        2.0 * expected_force_y, rel=2.0e-13, abs=0.0
    )

    # This event lies outside the source worldline, so the full G response
    # equals the compact source-free form -d (s.partial) F* . u.
    partial_dual = np.stack(
        [hodge_dual(partial_f[coordinate]) for coordinate in range(4)]
    )
    spin_directional_dual_gradient = np.einsum("l,lmn->mn", spin, partial_dual)
    compact_force = -coupling * (
        spin_directional_dual_gradient @ (MINKOWSKI_METRIC @ velocity)
    )
    np.testing.assert_allclose(compact_force, four_force, rtol=2.0e-13, atol=0.0)


def test_rfs_four_force_is_orthogonal_to_four_velocity() -> None:
    beta = np.array((0.22, -0.14, 0.09))
    velocity = _four_velocity(beta)
    spin = _boost_rest_spin(np.array((2.0e-35, -4.0e-35, 3.0e-35)), beta)
    field = electromagnetic_field_tensor_si((4.0e5, -2.0e5, 3.0e5), (0.3, -0.2, 0.5))
    partial_f = _tensor_gradient_from_field_gradients(
        np.array(
            (
                (2.0e5, -1.0e5, 4.0e4),
                (3.0e4, -7.0e4, 2.0e4),
                (-5.0e4, 8.0e4, 6.0e4),
            )
        ),
        np.array(
            (
                (0.4, -0.2, 0.1),
                (0.3, 0.5, -0.4),
                (-0.6, 0.2, 0.7),
            )
        ),
    )
    coupling = dipole_charge_from_moment_si(-9.0e-27, 5.5e-35)

    force = rfs_four_force_si(
        four_velocity_m_s=velocity,
        spin_four_vector_j_s=spin,
        field_tensor=field,
        partial_f=partial_f,
        charge_coulomb=-1.6e-19,
        dipole_charge=coupling,
    )

    contraction = minkowski_dot(velocity, force)
    scale = np.linalg.norm(velocity) * np.linalg.norm(force)
    _assert_zero_relative(contraction, scale)


def test_instantaneous_rhs_preserves_all_three_covariant_constraints() -> None:
    beta = np.array((0.18, 0.11, -0.07))
    velocity = _four_velocity(beta)
    rest_spin = np.array((3.0e-35, -2.0e-35, 4.0e-35))
    spin = _boost_rest_spin(rest_spin, beta)
    mass = 1.67e-27
    charge = 1.2e-19
    coupling = dipole_charge_from_moment_si(7.0e-27, np.linalg.norm(rest_spin))
    field = electromagnetic_field_tensor_si((-3.0e5, 5.0e5, 2.0e5), (0.6, -0.1, 0.25))
    partial_f = _tensor_gradient_from_field_gradients(
        np.array(
            (
                (1.0e5, -2.0e5, 3.0e5),
                (4.0e5, 1.5e5, -2.5e5),
                (-0.5e5, 0.7e5, 1.2e5),
            )
        ),
        np.array(
            (
                (0.5, 0.2, -0.3),
                (-0.4, 0.7, 0.1),
                (0.6, -0.2, 0.9),
            )
        ),
    )

    force = rfs_four_force_si(
        four_velocity_m_s=velocity,
        spin_four_vector_j_s=spin,
        field_tensor=field,
        partial_f=partial_f,
        charge_coulomb=charge,
        dipole_charge=coupling,
    )
    acceleration = force / mass
    spin_rhs = rfs_spin_rhs_si(
        four_velocity_m_s=velocity,
        spin_four_vector_j_s=spin,
        field_tensor=field,
        partial_f=partial_f,
        charge_coulomb=charge,
        mass_kg=mass,
        dipole_charge=coupling,
    )

    du_squared = 2.0 * minkowski_dot(velocity, acceleration)
    ds_squared = 2.0 * minkowski_dot(spin, spin_rhs)
    d_u_dot_s = minkowski_dot(acceleration, spin) + minkowski_dot(velocity, spin_rhs)

    _assert_zero_relative(
        du_squared,
        2.0 * np.linalg.norm(velocity) * np.linalg.norm(acceleration),
    )
    _assert_zero_relative(
        ds_squared,
        2.0 * np.linalg.norm(spin) * np.linalg.norm(spin_rhs),
    )
    orthogonality_scale = sum(
        (
            np.linalg.norm(acceleration) * np.linalg.norm(spin),
            np.linalg.norm(velocity) * np.linalg.norm(spin_rhs),
        )
    )
    _assert_zero_relative(
        d_u_dot_s,
        orthogonality_scale,
    )


def test_invalid_field_gradient_rejects_nonantisymmetric_field_indices() -> None:
    partial_f = np.zeros((4, 4, 4))
    partial_f[1, 0, 2] = 1.0

    with pytest.raises(ValueError, match="antisymmetric"):
        rfs_g_tensor(partial_f, (0.0, 1.0, 0.0, 0.0))
