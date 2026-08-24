from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from core.constants import C_MMNS
from core.retarded_dipole_fields import (
    DipoleSourceSingularityError,
    evaluate_retarded_dipole_field_gradient_native,
    evaluate_retarded_dipole_hertz_tensor_native,
)
from core.retarded_fields import ObserverEvent
from core.rfs import electromagnetic_field_tensor_native


def _dipole_history(
    *,
    beta: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 0.0),
    moment_native: float = 2.0,
    spin_function: Callable[[float], np.ndarray] | None = None,
    charge_native: float = 0.0,
) -> list[dict[str, np.ndarray]]:
    times_ns = np.linspace(-0.05, 0.01, 121)
    beta_vector = np.asarray(beta, dtype=float)
    if spin_function is None:
        spin_function = lambda _time_ns: np.array((0.0, 0.0, 1.0))
    result = []
    for time_ns in times_ns:
        position = beta_vector * C_MMNS * time_ns
        spin = np.asarray(spin_function(float(time_ns)), dtype=float)
        result.append(
            {
                "t": np.array([time_ns]),
                "x": np.array([position[0]]),
                "y": np.array([position[1]]),
                "z": np.array([position[2]]),
                "bx": np.array([beta_vector[0]]),
                "by": np.array([beta_vector[1]]),
                "bz": np.array([beta_vector[2]]),
                "bdotx": np.array([0.0]),
                "bdoty": np.array([0.0]),
                "bdotz": np.array([0.0]),
                # Deliberately zero by default: neutral magnetic dipoles must
                # remain sources independently of the charge-field provider.
                "q": np.array([charge_native]),
                "q_source": np.array([charge_native]),
                "spin_x": np.array([spin[0]]),
                "spin_y": np.array([spin[1]]),
                "spin_z": np.array([spin[2]]),
                "magnetic_moment_native": np.array([moment_native]),
                "magnetic_dipole_active": np.array([1.0]),
                "_dead_particles": np.array([False]),
            }
        )
    return result


def _static_dipole_field(moment: np.ndarray, position: np.ndarray) -> np.ndarray:
    radius = float(np.linalg.norm(position))
    direction = position / radius
    return (3.0 * direction * float(direction @ moment) - moment) / radius**3


def test_static_gaussian_potential_field_and_gradient_converge_quadratically() -> None:
    moment_z = 2.0
    history = _dipole_history(moment_native=moment_z)
    position = np.array((0.0, 0.0, 1.3))
    radius = float(position[2])
    expected_bz = 2.0 * moment_z / radius**3
    expected_dbz_dz = -6.0 * moment_z / radius**4

    errors = []
    results = []
    for relative_step in (4.0e-3, 2.0e-3, 1.0e-3):
        result = evaluate_retarded_dipole_field_gradient_native(
            history,
            ObserverEvent(0.0, tuple(position)),
            stencil_step_mm=relative_step * radius,
        )
        results.append(result)
        errors.append(abs(result.magnetic_field_native[2] - expected_bz))

    assert errors[0] / errors[1] == pytest.approx(4.0, rel=2.0e-3)
    assert errors[1] / errors[2] == pytest.approx(4.0, rel=2.0e-3)
    finest = results[-1]
    np.testing.assert_allclose(finest.electric_field_native, 0.0, atol=0.0)
    assert finest.magnetic_field_native[2] == pytest.approx(expected_bz, rel=3.0e-6)
    # F^(12)=-Bz in the native (+---) convention.
    assert -finest.partial_f[3, 1, 2] == pytest.approx(expected_dbz_dz, rel=2.0e-6)

    off_axis = np.array((1.2, -0.7, 0.9))
    off_axis_radius = float(np.linalg.norm(off_axis))
    off_axis_result = evaluate_retarded_dipole_field_gradient_native(
        history,
        ObserverEvent(0.0, tuple(off_axis)),
        stencil_step_mm=1.0e-3 * off_axis_radius,
    )
    moment = np.array((0.0, 0.0, moment_z))
    expected_potential = np.cross(moment, off_axis) / off_axis_radius**3
    expected_magnetic = _static_dipole_field(moment, off_axis)
    assert off_axis_result.four_potential[0] == 0.0
    np.testing.assert_allclose(
        off_axis_result.four_potential[1:], expected_potential, rtol=2.0e-6
    )
    np.testing.assert_allclose(
        off_axis_result.magnetic_field_native, expected_magnetic, rtol=4.0e-5
    )


def test_neutral_source_survives_and_stable_identity_excludes_only_self() -> None:
    history = _dipole_history(charge_native=0.0)
    event = ObserverEvent(0.0, (1.0, 0.0, 0.0))
    included = evaluate_retarded_dipole_field_gradient_native(
        history,
        event,
        source_identities=("physical-particle-17",),
        stencil_step_mm=5.0e-4,
    )
    assert included.hertz.valid_sources.tolist() == [True]
    assert included.magnetic_field_native[2] == pytest.approx(-2.0, rel=4.0e-6)

    excluded = evaluate_retarded_dipole_field_gradient_native(
        history,
        event,
        source_identities=("physical-particle-17",),
        observer_source_identity="physical-particle-17",
        stencil_step_mm=1.0e-3,
    )
    assert excluded.hertz.valid_sources.tolist() == [False]
    np.testing.assert_array_equal(excluded.four_potential, 0.0)
    np.testing.assert_array_equal(excluded.partial_a, 0.0)
    np.testing.assert_array_equal(excluded.field_tensor, 0.0)
    np.testing.assert_array_equal(excluded.partial_f, 0.0)


def test_uniform_motion_is_lorentz_transform_of_static_rest_field() -> None:
    beta = np.array((0.23, 0.0, 0.0))
    gamma = 1.0 / np.sqrt(1.0 - float(beta @ beta))
    event_position = np.array((0.6, 1.1, -0.8))
    rest_position = np.array(
        (gamma * event_position[0], event_position[1], event_position[2])
    )
    moment = np.array((0.0, 0.0, 2.0))
    rest_magnetic = _static_dipole_field(moment, rest_position)
    rest_potential = (
        np.cross(moment, rest_position) / float(np.linalg.norm(rest_position)) ** 3
    )

    expected_electric = -gamma * np.cross(beta, rest_magnetic)
    expected_magnetic = rest_magnetic.copy()
    expected_magnetic[1:] *= gamma
    expected_four_potential = np.array(
        (
            gamma * beta[0] * rest_potential[0],
            gamma * rest_potential[0],
            rest_potential[1],
            rest_potential[2],
        )
    )
    result = evaluate_retarded_dipole_field_gradient_native(
        _dipole_history(beta=beta, moment_native=2.0),
        ObserverEvent(0.0, tuple(event_position)),
        stencil_step_mm=8.0e-4 * float(np.linalg.norm(rest_position)),
    )

    np.testing.assert_allclose(
        result.electric_field_native, expected_electric, rtol=2.0e-5, atol=2.0e-7
    )
    np.testing.assert_allclose(
        result.magnetic_field_native, expected_magnetic, rtol=2.0e-5, atol=2.0e-7
    )
    np.testing.assert_allclose(
        result.four_potential,
        expected_four_potential,
        rtol=3.0e-6,
        atol=2.0e-7,
    )
    assert float(result.electric_field_native @ result.magnetic_field_native) == (
        pytest.approx(0.0, abs=2.0e-8)
    )
    lab_invariant = float(
        result.magnetic_field_native @ result.magnetic_field_native
        - result.electric_field_native @ result.electric_field_native
    )
    assert lab_invariant == pytest.approx(
        float(rest_magnetic @ rest_magnetic), rel=2.0e-6
    )


def test_linearly_varying_rest_dipole_matches_sautbekov_heras_limit() -> None:
    moment_at_observer = np.array((0.2, -0.4, 1.3))
    moment_derivative_per_ns = np.array((1.0, 3.0, -2.0))

    def moment_history(time_ns: float) -> np.ndarray:
        return moment_at_observer + moment_derivative_per_ns * time_ns

    position = np.array((1.2, 0.8, -0.4))
    radius = float(np.linalg.norm(position))
    direction = position / radius
    retarded_time_ns = -radius / C_MMNS
    retarded_moment = moment_history(retarded_time_ns)
    effective_moment = retarded_moment + moment_derivative_per_ns * radius / C_MMNS
    expected_potential = np.cross(effective_moment, direction) / radius**2
    expected_electric = np.cross(direction, moment_derivative_per_ns) / (
        C_MMNS * radius**2
    )
    expected_magnetic = _static_dipole_field(effective_moment, position)

    result = evaluate_retarded_dipole_field_gradient_native(
        _dipole_history(
            moment_native=1.0,
            spin_function=moment_history,
        ),
        ObserverEvent(0.0, tuple(position)),
        stencil_step_mm=8.0e-4 * radius,
    )
    assert result.hertz.retarded_time_ns[0] == pytest.approx(
        retarded_time_ns, abs=2.0e-16
    )
    np.testing.assert_allclose(
        result.four_potential[1:], expected_potential, rtol=6.0e-7
    )
    np.testing.assert_allclose(
        result.electric_field_native, expected_electric, rtol=3.0e-6
    )
    np.testing.assert_allclose(
        result.magnetic_field_native, expected_magnetic, rtol=2.0e-5
    )


def test_derivative_diagnostics_preserve_antisymmetry_gauge_and_bianchi_identity() -> (
    None
):
    result = evaluate_retarded_dipole_field_gradient_native(
        _dipole_history(beta=(0.11, -0.07, 0.03)),
        ObserverEvent(0.0, (1.1, -0.6, 0.7)),
        stencil_step_mm=1.0e-3,
    )
    signs = np.array((1.0, -1.0, -1.0, -1.0))
    reconstructed = np.zeros((4, 4), dtype=float)
    for mu in range(4):
        for nu in range(4):
            reconstructed[mu, nu] = (
                signs[mu] * result.partial_a[mu, nu]
                - signs[nu] * result.partial_a[nu, mu]
            )
    np.testing.assert_array_equal(result.field_tensor, reconstructed)
    np.testing.assert_array_equal(
        result.field_tensor,
        electromagnetic_field_tensor_native(
            result.electric_field_native, result.magnetic_field_native
        ),
    )
    np.testing.assert_array_equal(result.field_tensor + result.field_tensor.T, 0.0)
    np.testing.assert_array_equal(
        result.partial_f + np.swapaxes(result.partial_f, 1, 2), 0.0
    )
    assert result.lorenz_gauge_residual_per_mm == pytest.approx(0.0, abs=2.0e-12)

    lowered_gradient = np.einsum("m,n,lmn->lmn", signs, signs, result.partial_f)
    bianchi_scale = max(1.0, float(np.max(np.abs(lowered_gradient))))
    for first in range(4):
        for second in range(4):
            for third in range(4):
                cyclic_sum = (
                    lowered_gradient[first, second, third]
                    + lowered_gradient[second, third, first]
                    + lowered_gradient[third, first, second]
                )
                assert abs(cyclic_sum) <= 2.0e-11 * bianchi_scale

    assert result.stencil_offsets.shape[1] == 4
    assert result.stencil_retarded_time_ns.shape == (
        result.stencil_offsets.shape[0],
        1,
    )
    assert result.stencil_light_cone_residual_mm.shape == (
        result.stencil_offsets.shape[0],
        1,
    )
    assert np.nanmax(np.abs(result.stencil_light_cone_residual_mm)) < 1.0e-14


def test_point_source_minimum_separation_is_a_strict_abort_not_softening() -> None:
    with pytest.raises(DipoleSourceSingularityError, match="minimum_separation_mm"):
        evaluate_retarded_dipole_hertz_tensor_native(
            _dipole_history(),
            ObserverEvent(0.0, (1.0e-6, 0.0, 0.0)),
            minimum_separation_mm=2.0e-6,
        )


def test_duplicate_source_identities_are_rejected() -> None:
    history = _dipole_history()
    # Duplicate the only particle at every history row.
    for state in history:
        for key, value in tuple(state.items()):
            if isinstance(value, np.ndarray) and value.shape == (1,):
                state[key] = np.repeat(value, 2)
    with pytest.raises(ValueError, match="source identities must be unique"):
        evaluate_retarded_dipole_hertz_tensor_native(
            history,
            ObserverEvent(0.0, (1.0, 0.0, 0.0)),
            source_identities=("same", "same"),
        )
