"""Quintic coordinate conditioning and resolved mechanical drift regressions."""

from decimal import Decimal, localcontext

import numpy as np
import pytest

from core.constants import C_MMNS
from core.light_cone_history import build_light_cone_segment
from core.resolved_knot import (
    gamma_parts,
    initialize_resolved_result,
    preserve_drift_remainders,
)
from core.retarded_fields import _quintic_position_coefficients_mm

pytestmark = pytest.mark.unit


def test_moderate_speed_transformation_preserves_quintic():
    times = np.array([0.3, 0.302])
    positions = np.array([[1.2, -0.7, 2.1], [1.4, -0.5, 2.3]])
    betas = np.array([[0.31, 0.27, 0.34], [0.34, 0.24, 0.32]])
    primes = np.array([[0.02, -0.01, 0.01], [-0.01, 0.02, -0.01]])
    duration, old = _quintic_position_coefficients_mm(times, positions, betas, primes)
    segment = build_light_cone_segment(times, positions, betas, primes)
    for fraction in np.linspace(0.0, 1.0, 33):
        powers = fraction ** np.arange(6)
        position = powers @ old[0]
        beta = (
            (np.arange(1, 6) * fraction ** np.arange(5))
            @ old[0, 1:]
            / (C_MMNS * duration[0])
        )
        prime = (
            (np.arange(2, 6) * np.arange(1, 5) * fraction ** np.arange(4))
            @ old[0, 2:]
            / (C_MMNS * duration[0]) ** 2
        )
        actual_position, actual_beta, actual_prime, _ = segment.sample(fraction)
        np.testing.assert_allclose(actual_position, position, rtol=0.0, atol=3e-15)
        np.testing.assert_allclose(actual_beta, beta, rtol=0.0, atol=1e-14)
        np.testing.assert_allclose(actual_prime, prime, rtol=0.0, atol=4e-14)


@pytest.mark.parametrize("gamma", [1e3, 1e6, 1e9, 1e12])
def test_axis_ballistic_quintic_keeps_deficit(gamma):
    u = np.array([np.sqrt(gamma**2 - 1), 0.0, 0.0])
    beta = u / np.hypot(1.0, np.linalg.norm(u))
    times = np.array([0.0, 1 / C_MMNS])
    positions = np.array([np.zeros(3), beta])
    segment = build_light_cone_segment(
        times,
        positions,
        np.array([beta, beta]),
        np.zeros((2, 3)),
        proper_velocity=np.array([u, u]),
    )
    with localcontext() as context:
        context.prec = 80
        magnitude = Decimal.from_float(float(u[0]))
        g = (1 + magnitude * magnitude).sqrt()
        expected = float(1 / (g * (g + magnitude)))
    assert segment.coefficients_mm[1, 0] == pytest.approx(expected, rel=3e-16)
    np.testing.assert_array_equal(segment.coefficients_mm[2:], 0.0)
    for fraction in [0.0, 0.5, 1.0]:
        assert segment.sample(fraction)[3] == pytest.approx(1 / gamma**2, rel=1e-15)


def test_drift_carries_sub_ulp_time_increment():
    u = np.array([1e12, 0.0, 0.0])
    high, low = gamma_parts(u)
    assert high == 1e12
    assert low == pytest.approx(5e-13, rel=1e-15)
    initial = {name: np.array([0.0]) for name in ("x", "y", "z", "t", "bx", "by", "bz")}
    initial["gamma"] = np.array([1e12])
    initial["bx"][:] = 1.0
    result = {key: value.copy() for key, value in initial.items()}
    result["t"][:] = 1 / C_MMNS
    result["x"][:] = 1.0
    initialize_resolved_result(result, initial)
    preserve_drift_remainders(result, initial, 0, 1 / (1e12 * C_MMNS), u)
    # Convert the preserved endpoint pair to Decimal rather than rounding it.
    with localcontext() as context:
        context.prec = 80
        ct = Decimal.from_float(C_MMNS) * (
            Decimal.from_float(float(result["t"][0]))
            + Decimal.from_float(float(result["source_time_low_ns"][0]))
        )
        x = Decimal(1) + Decimal.from_float(float(result["source_position_low_x"][0]))
        assert float(ct - x) == pytest.approx(5e-25, rel=1e-8)


def test_accelerated_stored_directional_response_includes_snap():
    from core.retarded_fields import (
        ObserverEvent,
        evaluate_retarded_charge_response_gradient_native,
    )
    from core.retarded_potential_directional_jet import (
        quintic_charge_response_directional_gradient_native,
    )

    coefficients = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.31, -0.12, 0.07],
            [0.02, -0.01, 0.03],
            [0.03, -0.01, 0.02],
            [0.012, 0.025, -0.003],
            [-0.009, 0.012, -0.015],
        ]
    )
    times = np.array([0.0, 1 / C_MMNS])
    positions = np.array([coefficients[0], np.sum(coefficients, axis=0)])
    betas = np.array([coefficients[1], np.arange(1, 6) @ coefficients[1:]])
    primes = np.array(
        [2 * coefficients[2], (np.arange(2, 6) * np.arange(1, 5)) @ coefficients[2:]]
    )
    history = [
        {
            "t": np.array([time]),
            **{axis: np.array([position[index]]) for index, axis in enumerate("xyz")},
            **{f"b{axis}": np.array([beta[index]]) for index, axis in enumerate("xyz")},
            **{
                f"bdot{axis}": np.array([prime[index]])
                for index, axis in enumerate("xyz")
            },
            "q": np.array([1.0]),
            "_dead_particles": np.array([False]),
        }
        for time, position, beta, prime in zip(times, positions, betas, primes)
    ]
    fraction = 0.43
    point = fraction ** np.arange(6) @ coefficients
    separation = np.array([0.4, 0.6, -0.8])
    event = ObserverEvent(
        (fraction + np.linalg.norm(separation)) / C_MMNS, tuple(point + separation)
    )
    velocity = np.array([400.0, 10.0, 25.0, 30.0])
    actual = evaluate_retarded_charge_response_gradient_native(
        history,
        event,
        observer_four_velocity_mm_ns=velocity,
        source_acceleration_semantics="instantaneous",
    )
    reference = quintic_charge_response_directional_gradient_native(
        observer_time_ns=event.time_ns,
        observer_position_mm=event.position_mm,
        charge_native=1.0,
        segment_start_time_ns=0.0,
        segment_duration_ns=1 / C_MMNS,
        position_coefficients_mm=coefficients,
        retarded_time_ns=fraction / C_MMNS,
        four_velocity_mm_ns=velocity,
    )
    assert not actual.fallback_used
    np.testing.assert_allclose(
        actual.partial_antisymmetric_response_along_velocity,
        reference.partial_antisymmetric_response_along_velocity,
        rtol=2e-12,
        atol=1e-11,
    )


def test_resolved_scan_uses_observer_tail_and_rejects_rounded_proof():
    from core.compute_backends import (
        KnotScanBatch,
        latest_light_cone_segments_float64,
        strictly_timelike_source_chords_float64,
        certify_candidate_segments_float64,
    )

    # All three lab residuals round to zero. The observer tail places the
    # true root strictly between knots 0 and 1, rather than at the last knot.
    times = np.array([[0.0], [1 / C_MMNS], [2 / C_MMNS]])
    source = np.array([[[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]]])
    from decimal import Decimal as D

    with localcontext() as context:
        context.prec = 80
        low = np.array(
            [
                [float(D.from_float(C_MMNS) * D.from_float(t[0]) - D(i))]
                for i, t in enumerate(times)
            ]
        )
    position_low = np.zeros_like(source)
    position_low[:, 0, 0] = low[:, 0]
    position_tail = np.zeros_like(source)
    position_tail[:, 0, 0] = np.array([0.0, -1e-25, -2e-25])
    batch = KnotScanBatch(
        observer_time_ns=np.array([3 / C_MMNS]),
        observer_position_mm=np.array([[3.0, 0.0, 0.0]]),
        source_time_ns=times,
        source_position_mm=source,
        alive_counts=np.array([3]),
        source_position_low_mm=position_low,
        source_position_tail_mm=position_tail,
        observer_position_low_mm=np.array(
            [[float(D.from_float(C_MMNS) * D.from_float(3 / C_MMNS) - D(3)), 0.0, 0.0]]
        ),
        observer_position_tail_mm=np.array([[-0.5e-25, 0.0, 0.0]]),
    )
    assert latest_light_cone_segments_float64(batch).tolist() == [[0]]
    assert strictly_timelike_source_chords_float64(batch).tolist() == [False]
    certified = certify_candidate_segments_float64(batch, np.array([[1]]))
    assert certified.segment_indices.tolist() == [[0]]
    assert certified.cpu_fallbacks.tolist() == [[True]]


@pytest.mark.parametrize("gamma", [1e9, 1e12])
def test_initialization_and_prehistory_preserve_massive_knots(gamma):
    from core.particle_initialization import create_particle_state_3d
    from core.integration_runner import (
        _build_inertial_coasting_history,
        _estimate_inertial_prehistory_duration_ns,
    )
    from core.types import MagneticDipoleConfig

    state, _ = create_particle_state_3d(
        kinetic_energy_mev=(gamma - 1) * 931.494,
        particle_mass_amu=1.0,
        charge_sign=1.0,
        stripped_ions=1.0,
        particle_count=1,
        starting_position_mm=(0.0, 0.0, 0.0),
        momentum_axis=(1.0, 0.0, 0.0),
    )
    assert state["source_kinematics_ready"][0] == 1.0
    assert state["source_speed_deficit"][0] == pytest.approx(
        1 / (2 * gamma**2), rel=2e-15
    )
    other = {
        name: value.copy() if isinstance(value, np.ndarray) else value
        for name, value in state.items()
    }
    other["x"][:] = 1000.0
    duration = _estimate_inertial_prehistory_duration_ns(
        state, other, MagneticDipoleConfig(enabled=True)
    )
    assert duration > 1000.0 / (C_MMNS * state["source_speed_deficit"][0])
    history = _build_inertial_coasting_history(state, 1.0 / C_MMNS, knot_count=2)
    with localcontext() as context:
        context.prec = 80
        first = history[0]
        ct = Decimal.from_float(C_MMNS) * sum(
            (
                Decimal.from_float(float(first[name][0]))
                for name in ("t", "source_time_low_ns", "source_time_tail_ns")
            ),
            Decimal(0),
        )
        x = sum(
            (
                Decimal.from_float(float(first[name][0]))
                for name in ("x", "source_position_low_x", "source_position_tail_x")
            ),
            Decimal(0),
        )
        assert float(ct - x) == pytest.approx(-1 / (2 * gamma**2), rel=2e-15)
