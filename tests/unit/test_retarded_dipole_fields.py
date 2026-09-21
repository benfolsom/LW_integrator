from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from core.constants import C_MMNS
from core.retarded_dipole_fields import (
    DipoleSourceSingularityError,
    RetardedDipoleBackendUnavailableError,
    _interpolate_rest_spin_c1,
    _prepare_dipole_history,
    evaluate_retarded_dipole_field_gradient_native,
    evaluate_retarded_dipole_hertz_tensor_native,
    evaluate_retarded_dipole_potential_native,
)
from core.retarded_fields import ObserverEvent, RetardedHistoryError
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

        def spin_function(_time_ns: float) -> np.ndarray:
            return np.array((0.0, 0.0, 1.0))

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


def test_endpoint_potential_matches_full_oracle_for_representative_histories() -> None:
    angular_frequency_per_ns = 19.0

    def rotating_spin(time_ns: float) -> np.ndarray:
        angle = angular_frequency_per_ns * time_ns
        return np.array((0.8 * np.cos(angle), 0.8 * np.sin(angle), 0.6))

    cases = (
        (_dipole_history(), ObserverEvent(0.0, (1.2, -0.7, 0.9))),
        (
            _dipole_history(beta=(0.17, -0.08, 0.04)),
            ObserverEvent(0.0, (0.8, 1.1, -0.6)),
        ),
        (
            _dipole_history(beta=(-0.06, 0.09, 0.03), spin_function=rotating_spin),
            ObserverEvent(0.0, (1.0, 0.5, 0.8)),
        ),
    )

    for history, event in cases:
        endpoint = evaluate_retarded_dipole_potential_native(
            history,
            event,
            source_identities=("source",),
            stencil_step_mm=7.0e-4,
        )
        oracle = evaluate_retarded_dipole_field_gradient_native(
            history,
            event,
            source_identities=("source",),
            stencil_step_mm=7.0e-4,
        )

        np.testing.assert_array_equal(endpoint.four_potential, oracle.four_potential)
        np.testing.assert_array_equal(
            endpoint.hertz.retarded_time_ns, oracle.hertz.retarded_time_ns
        )
        np.testing.assert_array_equal(
            endpoint.hertz.light_cone_residual_mm,
            oracle.hertz.light_cone_residual_mm,
        )
        assert endpoint.stencil_offsets.shape == (9, 4)
        assert {tuple(offset) for offset in endpoint.stencil_offsets} == {
            (0, 0, 0, 0),
            (-1, 0, 0, 0),
            (1, 0, 0, 0),
            (0, -1, 0, 0),
            (0, 1, 0, 0),
            (0, 0, -1, 0),
            (0, 0, 1, 0),
            (0, 0, 0, -1),
            (0, 0, 0, 1),
        }
        assert endpoint.stencil_retarded_time_ns.shape == (9, 1)
        assert endpoint.stencil_light_cone_residual_mm.shape == (9, 1)
        assert np.nanmax(np.abs(endpoint.stencil_light_cone_residual_mm)) < 1.0e-14


@pytest.mark.parametrize(
    "backend",
    (
        "numba_roots_exact_serial",
        "numba_full_strict_serial",
        "numba_analytic_charge_response_serial",
        "numba_analytic_charge_dipole_response_serial",
    ),
)
def test_endpoint_potential_numba_backends_preserve_reference_contract(
    backend: str,
) -> None:
    pytest.importorskip("numba")

    def rotating_spin(time_ns: float) -> np.ndarray:
        angle = 29.0 * time_ns
        return np.array((0.8 * np.cos(angle), 0.8 * np.sin(angle), 0.6))

    history = _dipole_history(
        beta=(0.13, -0.06, 0.04),
        moment_native=-1.7,
        spin_function=rotating_spin,
    )
    event = ObserverEvent(-1.0e-5, (0.9, 1.1, -0.5))
    kwargs = {
        "source_identities": ("rotating-source",),
        "stencil_step_mm": 7.0e-4,
    }
    reference = evaluate_retarded_dipole_potential_native(history, event, **kwargs)
    candidate = evaluate_retarded_dipole_potential_native(
        history, event, backend=backend, **kwargs
    )

    np.testing.assert_array_equal(candidate.stencil_offsets, reference.stencil_offsets)
    for name in (
        "hertz_tensor",
        "retarded_time_ns",
        "light_cone_residual_mm",
        "separation_mm",
        "valid_sources",
    ):
        np.testing.assert_array_equal(
            getattr(candidate.hertz, name), getattr(reference.hertz, name)
        )
    if backend == "numba_roots_exact_serial":
        np.testing.assert_array_equal(
            candidate.four_potential, reference.four_potential
        )
        np.testing.assert_array_equal(
            candidate.stencil_retarded_time_ns,
            reference.stencil_retarded_time_ns,
        )
        np.testing.assert_array_equal(
            candidate.stencil_light_cone_residual_mm,
            reference.stencil_light_cone_residual_mm,
        )
    else:
        np.testing.assert_allclose(
            candidate.four_potential,
            reference.four_potential,
            rtol=3.0e-12,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            candidate.stencil_retarded_time_ns,
            reference.stencil_retarded_time_ns,
            rtol=0.0,
            atol=2.0e-18,
        )
        np.testing.assert_allclose(
            candidate.stencil_light_cone_residual_mm,
            reference.stencil_light_cone_residual_mm,
            rtol=0.0,
            atol=2.0e-15,
        )


def test_endpoint_potential_has_static_gaussian_sign_and_native_units() -> None:
    moment_native = 2.3
    position = np.array((1.2, -0.7, 0.9))
    radius = float(np.linalg.norm(position))
    expected = np.cross(np.array((0.0, 0.0, moment_native)), position) / radius**3
    result = evaluate_retarded_dipole_potential_native(
        _dipole_history(moment_native=moment_native),
        ObserverEvent(0.0, tuple(position)),
        stencil_step_mm=6.0e-4 * radius,
    )

    # In native Gaussian units [mu]=[q] mm, hence [A]=[q]/mm.
    assert result.four_potential[0] == 0.0
    np.testing.assert_allclose(result.four_potential[1:], expected, rtol=8.0e-7)

    scaled = evaluate_retarded_dipole_potential_native(
        _dipole_history(moment_native=3.0 * moment_native),
        ObserverEvent(0.0, tuple(2.0 * position)),
        stencil_step_mm=1.2e-3 * radius,
    )
    # A scales as mu/r^2, so tripling mu and doubling r gives 3/4 A.
    np.testing.assert_allclose(
        scaled.four_potential, 0.75 * result.four_potential, rtol=1.0e-10
    )


def test_endpoint_potential_is_causal_under_future_history_changes() -> None:
    event = ObserverEvent(0.0, (1.1, -0.4, 0.7))
    baseline = _dipole_history(beta=(0.08, -0.03, 0.02))
    changed_future = _dipole_history(beta=(0.08, -0.03, 0.02))
    for state in changed_future:
        if float(state["t"][0]) >= 0.004:
            state["x"] += 4.0
            state["y"] -= 2.0
            state["spin_x"][:] = -0.7
            state["spin_y"][:] = 0.5
            state["spin_z"][:] = -0.2

    first = evaluate_retarded_dipole_potential_native(
        baseline, event, stencil_step_mm=5.0e-4
    )
    second = evaluate_retarded_dipole_potential_native(
        changed_future, event, stencil_step_mm=5.0e-4
    )

    np.testing.assert_array_equal(first.four_potential, second.four_potential)
    np.testing.assert_array_equal(
        first.stencil_retarded_time_ns, second.stencil_retarded_time_ns
    )
    observer_stencil_times = (
        event.time_ns + first.stencil_offsets[:, 0] * first.stencil_step_mm / C_MMNS
    )
    assert np.all(first.stencil_retarded_time_ns[:, 0] < observer_stencil_times)


def test_endpoint_potential_preserves_exclusion_separation_and_history_guards() -> None:
    history = _dipole_history()
    event = ObserverEvent(0.0, (1.0, 0.0, 0.0))
    excluded = evaluate_retarded_dipole_potential_native(
        history,
        event,
        source_identities=("self",),
        observer_source_identity="self",
        stencil_step_mm=1.0e-3,
    )
    np.testing.assert_array_equal(excluded.four_potential, 0.0)
    assert excluded.hertz.valid_sources.tolist() == [False]

    with pytest.raises(DipoleSourceSingularityError, match="minimum_separation_mm"):
        evaluate_retarded_dipole_potential_native(
            history,
            ObserverEvent(0.0, (1.0e-6, 0.0, 0.0)),
            minimum_separation_mm=2.0e-6,
        )

    incomplete = [state for state in history if float(state["t"][0]) >= -0.001]
    with pytest.raises(RetardedHistoryError, match="does not bracket"):
        evaluate_retarded_dipole_potential_native(incomplete, event)


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


def test_c1_spin_interpolation_preserves_a_constant_physical_moment() -> None:
    angular_frequency_per_ns = 35.0

    def rotating_unit_spin(time_ns: float) -> np.ndarray:
        angle = angular_frequency_per_ns * time_ns
        return np.array((np.cos(angle), np.sin(angle), 0.0))

    prepared = _prepare_dipole_history(
        _dipole_history(spin_function=rotating_unit_spin),
        source_identities=("rotating-dipole",),
        observer_source_identity=None,
        excluded_source_identities=(),
    )
    source = prepared.sources[0]
    knot_times = source.worldline.time_ns
    midpoint_times = 0.5 * (knot_times[:-1] + knot_times[1:])
    interpolated = np.stack(
        [_interpolate_rest_spin_c1(source, float(time)) for time in midpoint_times]
    )

    np.testing.assert_allclose(np.linalg.norm(interpolated, axis=1), 1.0, atol=2e-16)


def _assert_complete_gradient_result_bitwise_equal(reference, candidate) -> None:
    for name in (
        "four_potential",
        "partial_a",
        "electric_field_native",
        "magnetic_field_native",
        "field_tensor",
        "partial_f",
        "stencil_offsets",
        "stencil_retarded_time_ns",
        "stencil_light_cone_residual_mm",
    ):
        np.testing.assert_array_equal(
            getattr(candidate, name), getattr(reference, name)
        )
    assert candidate.stencil_step_mm == reference.stencil_step_mm
    assert (
        candidate.lorenz_gauge_residual_per_mm == reference.lorenz_gauge_residual_per_mm
    )
    assert candidate.hertz.source_identities == reference.hertz.source_identities
    for name in (
        "hertz_tensor",
        "retarded_time_ns",
        "light_cone_residual_mm",
        "separation_mm",
        "valid_sources",
    ):
        np.testing.assert_array_equal(
            getattr(candidate.hertz, name), getattr(reference.hertz, name)
        )


def test_python_backend_is_default_and_never_dispatches_numba(monkeypatch) -> None:
    import core.retarded_dipole_fields as fields

    def unexpected_dispatch(*args, **kwargs):
        del args, kwargs
        raise AssertionError("default Python backend dispatched the Numba batch")

    monkeypatch.setattr(
        fields,
        "_evaluate_prepared_hertz_batch_numba_roots_exact_serial",
        unexpected_dispatch,
    )
    result = fields.evaluate_retarded_dipole_field_gradient_native(
        _dipole_history(beta=(0.07, -0.02, 0.03)),
        ObserverEvent(0.0, (1.0, 0.4, -0.7)),
        stencil_step_mm=8.0e-4,
    )
    endpoint = fields.evaluate_retarded_dipole_potential_native(
        _dipole_history(beta=(0.07, -0.02, 0.03)),
        ObserverEvent(0.0, (1.0, 0.4, -0.7)),
        stencil_step_mm=8.0e-4,
    )

    assert result.stencil_offsets.shape == (129, 4)
    assert endpoint.stencil_offsets.shape == (9, 4)


@pytest.mark.parametrize(
    "provider",
    (
        evaluate_retarded_dipole_potential_native,
        evaluate_retarded_dipole_field_gradient_native,
    ),
)
def test_numba_backend_preserves_first_displaced_history_failure(provider) -> None:
    pytest.importorskip("numba")
    history = []
    for time_ns in np.linspace(-0.01, 0.002, 121):
        zeros = np.zeros(2)
        history.append(
            {
                "t": np.array((time_ns, time_ns)),
                "x": np.array((C_MMNS * 0.0092, C_MMNS * 0.0098)),
                "y": zeros.copy(),
                "z": zeros.copy(),
                "bx": zeros.copy(),
                "by": zeros.copy(),
                "bz": zeros.copy(),
                "bdotx": zeros.copy(),
                "bdoty": zeros.copy(),
                "bdotz": zeros.copy(),
                "q": zeros.copy(),
                "q_source": zeros.copy(),
                "spin_x": np.array((1.0, 0.0)),
                "spin_y": np.array((0.0, 1.0)),
                "spin_z": zeros.copy(),
                "magnetic_moment_native": np.ones(2),
                "magnetic_dipole_active": np.ones(2),
                "_dead_particles": np.zeros(2, dtype=bool),
            }
        )

    def capture_failure(backend: str) -> tuple[type[Exception], str]:
        try:
            provider(
                history,
                ObserverEvent(0.0, (0.0, 0.0, 0.0)),
                source_identities=("a", "b"),
                stencil_step_mm=0.1,
                backend=backend,
            )
        except Exception as exc:
            return type(exc), str(exc)
        raise AssertionError("incomplete displaced history unexpectedly succeeded")

    expected = (
        RetardedHistoryError,
        "source history does not bracket the observer light cone for dipole "
        "source identities ['b']",
    )
    assert capture_failure("python") == expected
    assert capture_failure("numba_roots_exact_serial") == expected
    assert capture_failure("numba_full_strict_serial") == expected


def test_numba_roots_backend_is_bitwise_invariant_to_numba_thread_count() -> None:
    numba = pytest.importorskip("numba")

    angular_frequency_per_ns = 23.0

    def rotating_spin(time_ns: float) -> np.ndarray:
        angle = angular_frequency_per_ns * time_ns
        return np.array((0.8 * np.cos(angle), 0.8 * np.sin(angle), 0.6))

    history = _dipole_history(
        beta=(0.13, -0.06, 0.04),
        moment_native=-1.7,
        spin_function=rotating_spin,
    )
    event = ObserverEvent(0.0, (0.9, 1.1, -0.5))
    kwargs = {
        "source_identities": ("rotating-source",),
        "stencil_step_mm": 7.0e-4,
    }
    reference = evaluate_retarded_dipole_field_gradient_native(history, event, **kwargs)

    original_threads = numba.get_num_threads()
    maximum_threads = int(numba.config.NUMBA_NUM_THREADS)
    try:
        for thread_count in (1, 4, 8, 10, 15):
            if thread_count > maximum_threads:
                continue
            numba.set_num_threads(thread_count)
            candidate = evaluate_retarded_dipole_field_gradient_native(
                history,
                event,
                backend="numba_roots_exact_serial",
                **kwargs,
            )
            _assert_complete_gradient_result_bitwise_equal(reference, candidate)
    finally:
        numba.set_num_threads(original_threads)


def test_numba_backend_recomputes_final_worldline_sample_in_python(
    monkeypatch,
) -> None:
    pytest.importorskip("numba")
    import core.exact_retarded_numba as compiled_roots

    history = _dipole_history(beta=(0.11, -0.04, 0.02))
    event = ObserverEvent(0.0, (0.8, 1.0, -0.6))
    reference = evaluate_retarded_dipole_field_gradient_native(history, event)
    original = compiled_roots.evaluate_source_roots_exact_serial

    def roots_with_corrupted_discarded_samples(*args, **kwargs):
        batch = original(*args, **kwargs)
        corrupted = tuple(np.array(value, copy=True) for value in batch)
        corrupted[1][:] = 123.0
        corrupted[3][:] = 456.0
        corrupted[4][:] = 789.0
        return corrupted

    monkeypatch.setattr(
        compiled_roots,
        "evaluate_source_roots_exact_serial",
        roots_with_corrupted_discarded_samples,
    )
    candidate = evaluate_retarded_dipole_field_gradient_native(
        history,
        event,
        backend="numba_roots_exact_serial",
    )

    _assert_complete_gradient_result_bitwise_equal(reference, candidate)


def test_explicit_numba_backend_fails_when_capability_is_unavailable(
    monkeypatch,
) -> None:
    import core.exact_retarded_numba as compiled_roots

    monkeypatch.setattr(compiled_roots, "NUMBA_AVAILABLE", False)

    for backend in (
        "numba_roots_exact_serial",
        "numba_full_strict_serial",
        "numba_analytic_charge_response_serial",
        "numba_analytic_charge_dipole_response_serial",
    ):
        with pytest.raises(
            RetardedDipoleBackendUnavailableError,
            match="explicitly selected, but Numba is not available",
        ):
            evaluate_retarded_dipole_field_gradient_native(
                _dipole_history(),
                ObserverEvent(0.0, (1.0, 0.0, 0.0)),
                backend=backend,
            )


@pytest.mark.parametrize(
    ("kernel_name", "backend"),
    (
        ("evaluate_source_roots_exact_serial", "numba_roots_exact_serial"),
        (
            "evaluate_source_events_full_strict_serial",
            "numba_full_strict_serial",
        ),
    ),
)
def test_initial_numba_compilation_failure_has_named_capability_error(
    monkeypatch,
    kernel_name: str,
    backend: str,
) -> None:
    numba = pytest.importorskip("numba")
    import core.exact_retarded_numba as compiled_roots

    def failed_compilation(*args, **kwargs):
        del args, kwargs
        raise numba.core.errors.TypingError("synthetic compilation failure")

    failed_compilation.signatures = ()
    monkeypatch.setattr(
        compiled_roots,
        kernel_name,
        failed_compilation,
    )

    with pytest.raises(
        RetardedDipoleBackendUnavailableError,
        match="failed during initial JIT compilation",
    ):
        evaluate_retarded_dipole_field_gradient_native(
            _dipole_history(),
            ObserverEvent(0.0, (1.0, 0.0, 0.0)),
            backend=backend,
        )


@pytest.mark.parametrize(
    ("kernel_name", "backend"),
    (
        ("evaluate_source_roots_exact_serial", "numba_roots_exact_serial"),
        (
            "evaluate_source_events_full_strict_serial",
            "numba_full_strict_serial",
        ),
    ),
)
def test_numba_dispatch_does_not_wrap_non_compilation_failures(
    monkeypatch,
    kernel_name: str,
    backend: str,
) -> None:
    pytest.importorskip("numba")
    import core.exact_retarded_numba as compiled_roots

    def runtime_failure(*args, **kwargs):
        del args, kwargs
        raise ValueError("synthetic runtime failure")

    runtime_failure.signatures = (object(),)
    monkeypatch.setattr(
        compiled_roots,
        kernel_name,
        runtime_failure,
    )

    with pytest.raises(ValueError, match="synthetic runtime failure"):
        evaluate_retarded_dipole_field_gradient_native(
            _dipole_history(),
            ObserverEvent(0.0, (1.0, 0.0, 0.0)),
            backend=backend,
        )


def test_unknown_retarded_dipole_backend_fails_explicitly() -> None:
    with pytest.raises(ValueError, match="backend must be one of"):
        evaluate_retarded_dipole_field_gradient_native(
            _dipole_history(),
            ObserverEvent(0.0, (1.0, 0.0, 0.0)),
            backend="auto",
        )
