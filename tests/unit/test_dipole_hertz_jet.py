from __future__ import annotations

import numpy as np
import pytest

from core.constants import C_MMNS
from core.dipole_fields import static_point_dipole_field_native
from core.dipole_hertz_jet import (
    _spin_coefficients,
    evaluate_retarded_dipole_field_gradient_hertz_jet_native,
    quintic_dipole_hertz_response_jet_native,
    quintic_dipole_hertz_response_jet_numba_native,
)
from core.antisymmetric_response_rfs import (
    pack_antisymmetric_response_native,
    pack_partial_antisymmetric_response_native,
)
from core.retarded_dipole_fields import (
    _prepare_dipole_history,
    evaluate_retarded_dipole_field_gradient_native,
    evaluate_retarded_dipole_hertz_tensor_native,
)
from core.retarded_fields import ObserverEvent
from optimization.analyze_dipole_hertz_coefficients import analyze_coefficients


def _accelerating_rotating_history() -> list[dict[str, np.ndarray]]:
    times_ns = np.linspace(-0.05, 0.01, 121)
    beta0 = np.array((0.12, -0.04, 0.03))
    beta_prime0 = np.array((1.4e-3, -7.0e-4, 4.0e-4))
    beta_jerk = np.array((-2.0e-5, 1.0e-5, 0.5e-5))
    result: list[dict[str, np.ndarray]] = []
    for time_ns in times_ns:
        source_coordinate = C_MMNS * float(time_ns)
        position = (
            beta0 * source_coordinate
            + 0.5 * beta_prime0 * source_coordinate**2
            + (1.0 / 6.0) * beta_jerk * source_coordinate**3
        )
        beta = (
            beta0
            + beta_prime0 * source_coordinate
            + 0.5 * beta_jerk * source_coordinate**2
        )
        beta_prime = beta_prime0 + beta_jerk * source_coordinate
        angle = 31.0 * float(time_ns) + 0.13
        spin = np.array((0.8 * np.cos(angle), 0.8 * np.sin(angle), 0.6))
        result.append(
            {
                "t": np.array([time_ns]),
                "x": np.array([position[0]]),
                "y": np.array([position[1]]),
                "z": np.array([position[2]]),
                "bx": np.array([beta[0]]),
                "by": np.array([beta[1]]),
                "bz": np.array([beta[2]]),
                "bdotx": np.array([beta_prime[0]]),
                "bdoty": np.array([beta_prime[1]]),
                "bdotz": np.array([beta_prime[2]]),
                "q": np.array([0.0]),
                "q_source": np.array([0.0]),
                "spin_x": np.array([spin[0]]),
                "spin_y": np.array([spin[1]]),
                "spin_z": np.array([spin[2]]),
                "magnetic_moment_native": np.array([-1.7]),
                "magnetic_dipole_active": np.array([1.0]),
                "_dead_particles": np.array([False]),
            }
        )
    return result


def _static_history(times_ns: np.ndarray) -> list[dict[str, np.ndarray]]:
    result: list[dict[str, np.ndarray]] = []
    for time_ns in times_ns:
        angle = 17.0 * float(time_ns)
        spin = np.array((0.8 * np.cos(angle), 0.8 * np.sin(angle), 0.6))
        result.append(
            {
                "t": np.array([time_ns]),
                "x": np.array([0.0]),
                "y": np.array([0.0]),
                "z": np.array([0.0]),
                "bx": np.array([0.0]),
                "by": np.array([0.0]),
                "bz": np.array([0.0]),
                "bdotx": np.array([0.0]),
                "bdoty": np.array([0.0]),
                "bdotz": np.array([0.0]),
                "q": np.array([0.0]),
                "q_source": np.array([0.0]),
                "spin_x": np.array([spin[0]]),
                "spin_y": np.array([spin[1]]),
                "spin_z": np.array([spin[2]]),
                "magnetic_moment_native": np.array([1.0]),
                "magnetic_dipole_active": np.array([1.0]),
                "_dead_particles": np.array([False]),
            }
        )
    return result


def _uniform_history(
    times_ns: np.ndarray,
    beta: np.ndarray,
    *,
    moment_native: float,
    phase: float,
) -> list[dict[str, np.ndarray]]:
    result: list[dict[str, np.ndarray]] = []
    for time_ns in times_ns:
        position = beta * C_MMNS * float(time_ns)
        angle = 23.0 * float(time_ns) + phase
        spin = np.array((0.6 * np.cos(angle), 0.6 * np.sin(angle), 0.8))
        result.append(
            {
                "t": np.array([time_ns]),
                "x": np.array([position[0]]),
                "y": np.array([position[1]]),
                "z": np.array([position[2]]),
                "bx": np.array([beta[0]]),
                "by": np.array([beta[1]]),
                "bz": np.array([beta[2]]),
                "bdotx": np.array([0.0]),
                "bdoty": np.array([0.0]),
                "bdotz": np.array([0.0]),
                "q": np.array([0.0]),
                "q_source": np.array([0.0]),
                "spin_x": np.array([spin[0]]),
                "spin_y": np.array([spin[1]]),
                "spin_z": np.array([spin[2]]),
                "magnetic_moment_native": np.array([moment_native]),
                "magnetic_dipole_active": np.array([1.0]),
                "_dead_particles": np.array([False]),
            }
        )
    return result


def _combine_histories(
    left: list[dict[str, np.ndarray]], right: list[dict[str, np.ndarray]]
) -> list[dict[str, np.ndarray]]:
    assert len(left) == len(right)
    return [
        {key: np.concatenate((left_row[key], right_row[key])) for key in left_row}
        for left_row, right_row in zip(left, right)
    ]


def _jet_for_history(
    history: list[dict[str, np.ndarray]],
    event: ObserverEvent,
    *,
    compiled: bool = False,
):
    prepared = _prepare_dipole_history(
        history,
        source_identities=("source",),
        observer_source_identity=None,
        excluded_source_identities=(),
    )
    center = evaluate_retarded_dipole_hertz_tensor_native(
        history,
        event,
        source_identities=("source",),
    )
    root_time = float(center.retarded_time_ns[0])
    source = prepared.sources[0]
    segment = int(
        np.searchsorted(source.worldline.time_ns, root_time, side="right") - 1
    )
    segment = min(max(segment, 0), source.worldline.time_ns.size - 2)
    evaluator = (
        quintic_dipole_hertz_response_jet_numba_native
        if compiled
        else quintic_dipole_hertz_response_jet_native
    )
    return evaluator(
        observer_time_ns=event.time_ns,
        observer_position_mm=event.position_mm,
        magnetic_moment_native=source.magnetic_moment_native,
        segment_start_time_ns=float(source.worldline.time_ns[segment]),
        segment_duration_ns=float(source.worldline.segment_duration_ns[segment]),
        position_coefficients_mm=source.worldline.position_coefficients_mm[segment],
        rest_spin_start=source.rest_spin[segment],
        rest_spin_end=source.rest_spin[segment + 1],
        rest_spin_start_derivative_per_ns=source.rest_spin_derivative_per_ns[segment],
        rest_spin_end_derivative_per_ns=source.rest_spin_derivative_per_ns[segment + 1],
        preserved_rest_spin_magnitude=source.preserved_rest_spin_magnitude,
        retarded_time_ns=root_time,
    )


def test_static_hertz_jet_matches_exact_exterior_dipole_response() -> None:
    separation = np.array((1.2, -0.7, 0.9))
    moment_native = 2.3
    radius = float(np.linalg.norm(separation))
    result = quintic_dipole_hertz_response_jet_native(
        observer_time_ns=radius / C_MMNS,
        observer_position_mm=separation,
        magnetic_moment_native=moment_native,
        segment_start_time_ns=-1.0,
        segment_duration_ns=2.0,
        position_coefficients_mm=np.zeros((6, 3)),
        rest_spin_start=(0.0, 0.0, 1.0),
        rest_spin_end=(0.0, 0.0, 1.0),
        rest_spin_start_derivative_per_ns=(0.0, 0.0, 0.0),
        rest_spin_end_derivative_per_ns=(0.0, 0.0, 0.0),
        preserved_rest_spin_magnitude=1.0,
        retarded_time_ns=0.0,
    )
    expected = static_point_dipole_field_native(
        separation_vector_mm=separation,
        magnetic_moment_native=moment_native,
        rest_spin_direction=(0.0, 0.0, 1.0),
    )

    np.testing.assert_allclose(
        result.four_potential,
        expected.four_potential_native,
        rtol=2.0e-14,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        result.field_tensor,
        expected.field_tensor,
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        result.partial_f,
        expected.partial_f,
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(result.partial_a[0], 0.0, atol=2.0e-14)
    np.testing.assert_allclose(result.partial_a[:, 0], 0.0, atol=2.0e-14)
    np.testing.assert_allclose(
        result.partial_a[1:, 1:].T,
        expected.vector_potential_gradient_native,
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(result.electric_field_native, 0.0, atol=2.0e-13)
    np.testing.assert_allclose(
        result.magnetic_field_native,
        expected.magnetic_field_native,
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    assert np.max(np.abs(result.hertz_tensor + result.hertz_tensor.T)) == 0.0
    assert result.light_cone_jet_residual < 1.0e-12
    assert result.segment_fraction == 0.5


def test_hertz_jet_rejects_a_nonsmooth_segment_boundary() -> None:
    arguments = {
        "observer_time_ns": 0.1,
        "observer_position_mm": (1.0, 0.0, 0.0),
        "magnetic_moment_native": 1.0,
        "segment_start_time_ns": 0.0,
        "segment_duration_ns": 1.0,
        "position_coefficients_mm": np.zeros((6, 3)),
        "rest_spin_start": (0.0, 0.0, 1.0),
        "rest_spin_end": (0.0, 0.0, 1.0),
        "rest_spin_start_derivative_per_ns": (0.0, 0.0, 0.0),
        "rest_spin_end_derivative_per_ns": (0.0, 0.0, 0.0),
        "preserved_rest_spin_magnitude": 1.0,
    }
    for root_time in (0.0, 1.0):
        try:
            quintic_dipole_hertz_response_jet_native(
                **arguments,
                retarded_time_ns=root_time,
            )
        except ValueError as error:
            assert "strictly inside" in str(error)
        else:
            raise AssertionError("a segment-boundary root must be rejected")


def test_accelerating_rotating_hertz_jet_is_the_stencil_limit() -> None:
    history = _accelerating_rotating_history()
    event = ObserverEvent(-1.7e-4, (1.1, -0.6, 0.8))
    analytical = _jet_for_history(history, event)
    coarse = evaluate_retarded_dipole_field_gradient_native(
        history,
        event,
        source_identities=("source",),
        stencil_step_mm=1.6e-3,
        backend="numba_full_strict_serial",
    )
    fine = evaluate_retarded_dipole_field_gradient_native(
        history,
        event,
        source_identities=("source",),
        stencil_step_mm=8.0e-4,
        backend="numba_full_strict_serial",
    )

    np.testing.assert_allclose(
        analytical.hertz_tensor,
        coarse.hertz.hertz_tensor,
        rtol=3.0e-14,
        atol=3.0e-14,
    )
    for name in ("four_potential", "partial_a", "field_tensor", "partial_f"):
        expected = getattr(analytical, name)
        coarse_error = np.linalg.norm(getattr(coarse, name) - expected)
        fine_error = np.linalg.norm(getattr(fine, name) - expected)
        assert fine_error < 0.27 * coarse_error, (name, coarse_error, fine_error)
        scale = max(float(np.linalg.norm(expected)), 1.0)
        assert fine_error / scale < 3.0e-5, (name, fine_error, scale)
    assert analytical.light_cone_jet_residual < 2.0e-10


def test_strict_numba_hertz_jet_matches_the_python_oracle() -> None:
    history = _accelerating_rotating_history()
    event = ObserverEvent(-1.7e-4, (1.1, -0.6, 0.8))
    reference = _jet_for_history(history, event)
    compiled = _jet_for_history(history, event, compiled=True)

    for name in (
        "hertz_tensor",
        "four_potential",
        "partial_a",
        "field_tensor",
        "partial_f",
        "retarded_coordinate_gradient",
        "retarded_coordinate_hessian",
        "retarded_coordinate_third_derivative",
    ):
        np.testing.assert_allclose(
            getattr(compiled, name),
            getattr(reference, name),
            rtol=2.0e-12,
            atol=2.0e-12,
            err_msg=name,
        )
    assert compiled.light_cone_jet_residual < 2.0e-10


def test_sparse_strict_kernel_matches_the_dense_response_outputs() -> None:
    from core.dipole_hertz_jet_numba import (
        _HERTZ_RESPONSE_USED,
        _SPARSE_HERTZ_SIZE,
        quintic_dipole_hertz_response_coefficients_strict_serial,
        quintic_dipole_hertz_sparse_response_strict_serial,
    )

    random = np.random.default_rng(112358)
    stress_beta = (0.0, 0.5, 0.9, 0.99, 0.9999)
    for case in range(20):
        duration_ns = float(random.uniform(5.0e-4, 1.4e-3))
        duration_coordinate = C_MMNS * duration_ns
        start_time = float(random.uniform(-0.02, -0.004))
        fraction = float(random.uniform(0.12, 0.88))
        root_time = start_time + fraction * duration_ns
        coefficients = np.zeros((6, 3))
        coefficients[0] = random.normal(scale=0.15, size=3)
        beta = random.normal(size=3)
        beta_magnitude = (
            stress_beta[case]
            if case < len(stress_beta)
            else float(random.uniform(0.0, 0.92))
        )
        beta *= beta_magnitude / np.linalg.norm(beta)
        coefficients[1] = beta * duration_coordinate
        for order in range(2, 6):
            coefficients[order] = random.normal(scale=1.0e-4 / order**2, size=3)
        source_position = (
            np.asarray([fraction**order for order in range(6)]) @ coefficients
        )
        direction = random.normal(size=3)
        direction /= np.linalg.norm(direction)
        radius = float(random.uniform(0.5, 1.8))
        observer_position = source_position + radius * direction
        observer_time = root_time + radius / C_MMNS
        spin_start = random.normal(size=3)
        spin_start /= np.linalg.norm(spin_start)
        spin_end = random.normal(size=3)
        spin_end /= np.linalg.norm(spin_end)
        start_slope = random.normal(scale=12.0, size=3)
        end_slope = random.normal(scale=12.0, size=3)
        spin_coefficients = _spin_coefficients(
            spin_start,
            spin_end,
            start_slope,
            end_slope,
            duration_ns,
        )
        common = (
            observer_time,
            observer_position,
            float(random.uniform(-2.0, 2.0)),
            start_time,
            duration_ns,
            coefficients,
            spin_coefficients,
            bool(case % 2 == 0),
            1.0,
            root_time,
        )
        dense = quintic_dipole_hertz_response_coefficients_strict_serial(*common)
        sparse = quintic_dipole_hertz_sparse_response_strict_serial(*common)

        assert dense[0] == sparse[0] == 0
        np.testing.assert_allclose(sparse[1], dense[2], rtol=2.0e-12, atol=2.0e-12)
        np.testing.assert_allclose(
            sparse[2],
            pack_antisymmetric_response_native(dense[4]),
            rtol=2.0e-12,
            atol=2.0e-12,
        )
        np.testing.assert_allclose(
            sparse[3],
            pack_partial_antisymmetric_response_native(dense[5]),
            rtol=2.0e-12,
            atol=2.0e-12,
        )
        assert abs(sparse[4] - dense[9]) <= 2.0e-10

    assert int(np.count_nonzero(_HERTZ_RESPONSE_USED)) == 144
    assert _SPARSE_HERTZ_SIZE == 144


def test_random_smooth_segments_match_the_python_third_order_oracle() -> None:
    random = np.random.default_rng(20260828)
    for case in range(20):
        duration_ns = float(random.uniform(4.0e-4, 1.5e-3))
        duration_coordinate = C_MMNS * duration_ns
        start_time = float(random.uniform(-0.02, -0.005))
        fraction = float(random.uniform(0.08, 0.92))
        root_time = start_time + fraction * duration_ns
        coefficients = np.zeros((6, 3))
        coefficients[0] = random.normal(scale=0.2, size=3)
        beta = random.normal(size=3)
        beta *= float(random.uniform(0.0, 0.88)) / np.linalg.norm(beta)
        coefficients[1] = beta * duration_coordinate
        for order in range(2, 6):
            coefficients[order] = random.normal(
                scale=2.0e-4 / order**2,
                size=3,
            )
        powers = np.asarray([fraction**order for order in range(6)])
        source_position = powers @ coefficients
        direction = random.normal(size=3)
        direction /= np.linalg.norm(direction)
        radius = float(random.uniform(0.4, 2.0))
        observer_position = source_position + radius * direction
        observer_time = root_time + radius / C_MMNS
        spin_start = random.normal(size=3)
        spin_start /= np.linalg.norm(spin_start)
        spin_end = random.normal(size=3)
        spin_end /= np.linalg.norm(spin_end)
        slope_start = random.normal(scale=20.0, size=3)
        slope_end = random.normal(scale=20.0, size=3)
        preserved_magnitude = 1.0 if case % 2 == 0 else None
        arguments = {
            "observer_time_ns": observer_time,
            "observer_position_mm": observer_position,
            "magnetic_moment_native": float(random.uniform(-2.0, 2.0)),
            "segment_start_time_ns": start_time,
            "segment_duration_ns": duration_ns,
            "position_coefficients_mm": coefficients,
            "rest_spin_start": spin_start,
            "rest_spin_end": spin_end,
            "rest_spin_start_derivative_per_ns": slope_start,
            "rest_spin_end_derivative_per_ns": slope_end,
            "preserved_rest_spin_magnitude": preserved_magnitude,
            "retarded_time_ns": root_time,
        }
        reference = quintic_dipole_hertz_response_jet_native(**arguments)
        compiled = quintic_dipole_hertz_response_jet_numba_native(**arguments)
        for name in (
            "hertz_tensor",
            "four_potential",
            "partial_a",
            "field_tensor",
            "partial_f",
            "retarded_coordinate_gradient",
            "retarded_coordinate_hessian",
            "retarded_coordinate_third_derivative",
        ):
            np.testing.assert_allclose(
                getattr(compiled, name),
                getattr(reference, name),
                rtol=3.0e-11,
                atol=3.0e-11,
                err_msg=f"case={case}, field={name}",
            )


def test_history_provider_uses_one_event_inside_a_smooth_segment() -> None:
    history = _accelerating_rotating_history()
    event = ObserverEvent(-1.7e-4, (1.1, -0.6, 0.8))
    expected = _jet_for_history(history, event)
    provider = evaluate_retarded_dipole_field_gradient_hertz_jet_native(
        history,
        event,
        source_identities=("source",),
    )

    assert provider.used_analytic_response
    assert provider.fallback_reason is None
    assert provider.response.stencil_offsets.shape == (1, 4)
    assert provider.response.stencil_step_mm == 0.0
    assert 0.0 < provider.source_segment_fraction[0] < 1.0
    np.testing.assert_allclose(provider.source_jet_residual[0], 0.0, atol=2.0e-10)
    for name in ("four_potential", "partial_a", "field_tensor", "partial_f"):
        np.testing.assert_array_equal(
            getattr(provider.response, name),
            getattr(expected, name),
        )

    compiled = evaluate_retarded_dipole_field_gradient_hertz_jet_native(
        history,
        event,
        source_identities=("source",),
        response_kernel="numba_strict_serial",
    )
    assert compiled.used_analytic_response
    for name in ("four_potential", "partial_a", "field_tensor", "partial_f"):
        np.testing.assert_allclose(
            getattr(compiled.response, name),
            getattr(provider.response, name),
            rtol=2.0e-12,
            atol=2.0e-12,
            err_msg=name,
        )

    sparse = evaluate_retarded_dipole_field_gradient_hertz_jet_native(
        history,
        event,
        source_identities=("source",),
        response_kernel="numba_sparse_strict_serial",
    )
    assert sparse.used_analytic_response
    np.testing.assert_allclose(
        sparse.response.four_potential,
        compiled.response.four_potential,
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        sparse.response.antisymmetric_response,
        pack_antisymmetric_response_native(compiled.response.field_tensor),
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        sparse.response.partial_antisymmetric_response,
        pack_partial_antisymmetric_response_native(compiled.response.partial_f),
        rtol=2.0e-12,
        atol=2.0e-12,
    )

    production = evaluate_retarded_dipole_field_gradient_native(
        history,
        event,
        source_identities=("source",),
        backend="numba_analytic_charge_dipole_response_serial",
    )
    assert production.stencil_offsets.shape == (1, 4)
    for name in ("four_potential", "partial_a", "field_tensor", "partial_f"):
        np.testing.assert_array_equal(
            getattr(production, name),
            getattr(compiled.response, name),
        )


def test_sparse_provider_does_not_construct_the_center_hertz_tensor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core import retarded_dipole_fields

    def fail_if_called(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("sparse provider constructed a center Hertz tensor")

    monkeypatch.setattr(
        retarded_dipole_fields,
        "_evaluate_prepared_hertz_tensor_native",
        fail_if_called,
    )
    result = evaluate_retarded_dipole_field_gradient_hertz_jet_native(
        _accelerating_rotating_history(),
        ObserverEvent(-1.7e-4, (1.1, -0.6, 0.8)),
        source_identities=("source",),
        response_kernel="numba_sparse_strict_serial",
    )

    assert result.used_analytic_response
    assert result.response.antisymmetric_response.shape == (6,)
    assert result.response.partial_antisymmetric_response.shape == (4, 6)


def test_history_provider_falls_back_at_a_spin_segment_knot() -> None:
    history = _accelerating_rotating_history()
    knot = 80
    source_time = float(history[knot]["t"][0])
    source_position = np.array(
        (
            history[knot]["x"][0],
            history[knot]["y"][0],
            history[knot]["z"][0],
        )
    )
    separation = np.array((1.0, 0.0, 0.0))
    event = ObserverEvent(
        source_time + float(np.linalg.norm(separation)) / C_MMNS,
        tuple(source_position + separation),
    )
    provider = evaluate_retarded_dipole_field_gradient_hertz_jet_native(
        history,
        event,
        source_identities=("source",),
        boundary_guard_fraction=1.0e-5,
        fallback_stencil_step_mm=2.0e-4,
    )

    assert not provider.used_analytic_response
    assert provider.fallback_reason is not None
    assert "segment-boundary guard" in provider.fallback_reason
    assert (
        min(
            provider.source_segment_fraction[0],
            1.0 - provider.source_segment_fraction[0],
        )
        < 1.0e-10
    )
    assert provider.response.stencil_offsets.shape == (129, 4)

    sparse = evaluate_retarded_dipole_field_gradient_hertz_jet_native(
        history,
        event,
        source_identities=("source",),
        boundary_guard_fraction=1.0e-5,
        fallback_stencil_step_mm=2.0e-4,
        response_kernel="numba_sparse_strict_serial",
    )
    assert not sparse.used_analytic_response
    assert sparse.fallback_reason == provider.fallback_reason
    np.testing.assert_array_equal(
        sparse.response.four_potential,
        provider.response.four_potential,
    )
    np.testing.assert_array_equal(
        sparse.response.antisymmetric_response,
        pack_antisymmetric_response_native(provider.response.field_tensor),
    )
    np.testing.assert_array_equal(
        sparse.response.partial_antisymmetric_response,
        pack_partial_antisymmetric_response_native(provider.response.partial_f),
    )


def test_history_provider_falls_back_in_the_mutable_final_spin_segment() -> None:
    history = _static_history(np.array((-0.002, -0.001, 0.0)))
    root_time = -5.0e-4
    event = ObserverEvent(root_time + 1.0 / C_MMNS, (1.0, 0.0, 0.0))
    provider = evaluate_retarded_dipole_field_gradient_hertz_jet_native(
        history,
        event,
        source_identities=("source",),
        fallback_stencil_step_mm=2.0e-4,
    )

    assert not provider.used_analytic_response
    assert provider.fallback_reason is not None
    assert "mutable final" in provider.fallback_reason


def test_frozen_segment_response_is_unchanged_by_a_future_spin_knot() -> None:
    complete = _static_history(np.linspace(-0.004, 0.001, 6))
    root_time = -2.5e-3
    event = ObserverEvent(root_time + 1.0 / C_MMNS, (1.0, 0.0, 0.0))
    earlier = evaluate_retarded_dipole_field_gradient_hertz_jet_native(
        complete[:-1],
        event,
        source_identities=("source",),
        response_kernel="numba_strict_serial",
    )
    later = evaluate_retarded_dipole_field_gradient_hertz_jet_native(
        complete,
        event,
        source_identities=("source",),
        response_kernel="numba_strict_serial",
    )

    assert earlier.used_analytic_response
    assert later.used_analytic_response
    np.testing.assert_array_equal(
        earlier.source_segment_index,
        later.source_segment_index,
    )
    for name in ("four_potential", "partial_a", "field_tensor", "partial_f"):
        np.testing.assert_array_equal(
            getattr(earlier.response, name),
            getattr(later.response, name),
        )


def test_coefficient_audit_flags_dead_raw_hertz_slots() -> None:
    report = analyze_coefficients()
    blocks = report["blocks"]

    assert report["raw_hertz_jet_coefficient_count"] == 210
    assert report["influential_hertz_jet_coefficient_count"] == 144
    assert report["structurally_unused_hertz_jet_coefficient_count"] == 66
    assert report["compact_response_component_count"] == 34
    assert blocks["H_value"]["structurally_unused_coefficient_count"] == 6
    assert blocks["A"]["influential_coefficient_count"] == 12
    assert blocks["A"]["structurally_unused_coefficient_count"] == 12
    assert blocks["F"]["influential_coefficient_count"] == 36
    assert blocks["F"]["structurally_unused_coefficient_count"] == 24
    assert blocks["partial_F"]["influential_coefficient_count"] == 96
    assert blocks["partial_F"]["structurally_unused_coefficient_count"] == 24
    assert blocks["partial_F"]["output_rank"] == 20


def test_relativistic_uniform_source_converges_to_the_analytical_response() -> None:
    history = _uniform_history(
        np.linspace(-0.05, 0.01, 121),
        np.array((0.82, -0.21, 0.11)),
        moment_native=-1.4,
        phase=0.2,
    )
    event = ObserverEvent(-2.1e-4, (1.2, 0.7, -0.9))
    analytical = evaluate_retarded_dipole_field_gradient_hertz_jet_native(
        history,
        event,
        source_identities=("relativistic",),
        response_kernel="numba_strict_serial",
    )
    assert analytical.used_analytic_response
    coarse = evaluate_retarded_dipole_field_gradient_native(
        history,
        event,
        source_identities=("relativistic",),
        stencil_step_mm=2.4e-3,
        backend="numba_full_strict_serial",
    )
    fine = evaluate_retarded_dipole_field_gradient_native(
        history,
        event,
        source_identities=("relativistic",),
        stencil_step_mm=1.2e-3,
        backend="numba_full_strict_serial",
    )
    for name in ("four_potential", "partial_a", "field_tensor", "partial_f"):
        expected = getattr(analytical.response, name)
        coarse_error = np.linalg.norm(getattr(coarse, name) - expected)
        fine_error = np.linalg.norm(getattr(fine, name) - expected)
        assert fine_error < 0.28 * coarse_error, (name, coarse_error, fine_error)


def test_multiple_sources_sum_in_source_order() -> None:
    times = np.linspace(-0.05, 0.01, 121)
    left = _uniform_history(
        times,
        np.array((0.17, -0.06, 0.03)),
        moment_native=-1.7,
        phase=0.1,
    )
    right = _uniform_history(
        times,
        np.array((-0.09, 0.12, -0.04)),
        moment_native=2.1,
        phase=-0.3,
    )
    event = ObserverEvent(-1.7e-4, (1.1, -0.6, 0.8))
    combined = evaluate_retarded_dipole_field_gradient_hertz_jet_native(
        _combine_histories(left, right),
        event,
        source_identities=("left", "right"),
        response_kernel="numba_strict_serial",
    )
    left_result = evaluate_retarded_dipole_field_gradient_hertz_jet_native(
        left,
        event,
        source_identities=("left",),
        response_kernel="numba_strict_serial",
    )
    right_result = evaluate_retarded_dipole_field_gradient_hertz_jet_native(
        right,
        event,
        source_identities=("right",),
        response_kernel="numba_strict_serial",
    )

    assert combined.used_analytic_response
    for name in ("four_potential", "partial_a", "field_tensor", "partial_f"):
        np.testing.assert_array_equal(
            getattr(combined.response, name),
            getattr(left_result.response, name) + getattr(right_result.response, name),
        )


def test_lost_source_is_skipped_after_its_retarded_wavefront() -> None:
    history = _static_history(np.linspace(-0.004, 0.001, 6))
    for row in history[4:]:
        row["_dead_particles"][0] = True
    event = ObserverEvent(0.02, (1.0, 0.0, 0.0))
    result = evaluate_retarded_dipole_field_gradient_hertz_jet_native(
        history,
        event,
        source_identities=("lost",),
        response_kernel="numba_strict_serial",
    )

    assert result.used_analytic_response
    assert not result.response.hertz.valid_sources[0]
    np.testing.assert_array_equal(result.response.four_potential, np.zeros(4))
    np.testing.assert_array_equal(result.response.field_tensor, np.zeros((4, 4)))
    np.testing.assert_array_equal(result.response.partial_f, np.zeros((4, 4, 4)))


def test_lost_source_wavefront_uses_the_displaced_stencil_fallback() -> None:
    history = _static_history(np.linspace(-0.004, 0.001, 6))
    for row in history[4:]:
        row["_dead_particles"][0] = True
    final_alive_time = float(history[3]["t"][0])
    event = ObserverEvent(
        final_alive_time + (1.0 + 1.0e-4) / C_MMNS,
        (1.0, 0.0, 0.0),
    )
    result = evaluate_retarded_dipole_field_gradient_hertz_jet_native(
        history,
        event,
        source_identities=("lost",),
        response_kernel="numba_strict_serial",
        fallback_stencil_step_mm=2.0e-4,
    )

    assert not result.used_analytic_response
    assert result.fallback_reason is not None
    assert "termination wavefront" in result.fallback_reason
    assert result.response.stencil_offsets.shape == (129, 4)
