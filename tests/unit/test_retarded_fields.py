from __future__ import annotations

import math

import numpy as np
import pytest

import core.retarded_fields as retarded_fields
from core.constants import C_MMNS, ELEMENTARY_CHARGE
from core.magnetic_dipole import (
    ELECTRIC_FIELD_NATIVE_TO_V_PER_M,
    MAGNETIC_FIELD_NATIVE_TO_TESLA,
)
from core.retarded_fields import (
    ObserverEvent,
    RetardedHistoryError,
    evaluate_retarded_charge_field_gradient_native,
    evaluate_retarded_charge_field_native,
    lienard_wiechert_charge_field_native,
)
from core.rfs import electromagnetic_field_tensor_native
from core.vectorized_interactions import (
    ExternalSampleBatch,
    compute_vectorized_contributions,
)

_C_SI = C_MMNS * 1.0e6
_LENGTH_UNIT_M = 1.0e-3
_COULOMB_CONSTANT_SI = 8.987_551_792_3e9
# This is the SI charge represented by one native source-charge unit when the
# native Coulomb law q/R^2 is mapped through the current field/length scales.
_SOURCE_CHARGE_UNIT_C = (
    ELECTRIC_FIELD_NATIVE_TO_V_PER_M * _LENGTH_UNIT_M**2 / _COULOMB_CONSTANT_SI
)


def _source_history(
    *,
    times_ns: np.ndarray,
    position_mm: np.ndarray,
    beta: np.ndarray,
    beta_prime_per_mm: np.ndarray | None = None,
    charge_native: float = ELEMENTARY_CHARGE,
) -> list[dict[str, np.ndarray]]:
    count = int(times_ns.size)
    if position_mm.shape != (count, 3) or beta.shape != (count, 3):
        raise ValueError("test worldline arrays have inconsistent shapes")
    if beta_prime_per_mm is None:
        beta_prime_per_mm = np.zeros_like(beta)
    result = []
    for step in range(count):
        result.append(
            {
                "t": np.array([times_ns[step]], dtype=float),
                "x": np.array([position_mm[step, 0]], dtype=float),
                "y": np.array([position_mm[step, 1]], dtype=float),
                "z": np.array([position_mm[step, 2]], dtype=float),
                "bx": np.array([beta[step, 0]], dtype=float),
                "by": np.array([beta[step, 1]], dtype=float),
                "bz": np.array([beta[step, 2]], dtype=float),
                "bdotx": np.array([beta_prime_per_mm[step, 0]], dtype=float),
                "bdoty": np.array([beta_prime_per_mm[step, 1]], dtype=float),
                "bdotz": np.array([beta_prime_per_mm[step, 2]], dtype=float),
                "q": np.array([charge_native], dtype=float),
                "q_source": np.array([charge_native], dtype=float),
                "_dead_particles": np.array([False]),
            }
        )
    return result


def _stationary_history(*, charge_native: float = ELEMENTARY_CHARGE):
    times_ns = np.linspace(-0.02, 0.002, 23)
    return _source_history(
        times_ns=times_ns,
        position_mm=np.zeros((times_ns.size, 3)),
        beta=np.zeros((times_ns.size, 3)),
        charge_native=charge_native,
    )


def _prepared_single_source(
    history: list[dict[str, np.ndarray]],
) -> retarded_fields._PreparedSourceHistory:
    arrays = retarded_fields._extract_history(history)
    return retarded_fields._prepare_source_history(arrays, 0)


def _prepared_single_source_arrays(
    *,
    times_ns: np.ndarray,
    position_mm: np.ndarray,
    beta: np.ndarray,
    beta_prime_per_mm: np.ndarray | None = None,
    dead: np.ndarray | None = None,
) -> retarded_fields._PreparedSourceHistory:
    if beta_prime_per_mm is None:
        beta_prime_per_mm = np.zeros_like(beta)
    if dead is None:
        dead = np.zeros(times_ns.size, dtype=bool)
    arrays = retarded_fields._HistoryArrays(
        time_ns=times_ns[:, np.newaxis],
        position_mm=position_mm[:, np.newaxis, :],
        beta=beta[:, np.newaxis, :],
        beta_prime_per_mm=beta_prime_per_mm[:, np.newaxis, :],
        charge_native=np.array((ELEMENTARY_CHARGE,)),
        dead=dead[:, np.newaxis],
    )
    return retarded_fields._prepare_source_history(arrays, 0)


def test_prepared_charge_history_reconstructs_endpoint_acceleration_from_beta() -> None:
    times_ns = np.asarray((-0.004, -0.001, 0.002, 0.006), dtype=float)
    coordinate_mm = C_MMNS * times_ns
    beta = np.zeros((times_ns.size, 3), dtype=float)
    beta[:, 0] = 0.2 + 2.0e-5 * coordinate_mm + 3.0e-8 * coordinate_mm**2
    deliberately_mistimed_bdot = np.full_like(beta, 9.0e-3)
    expected_x = 2.0e-5 + 6.0e-8 * coordinate_mm

    prepared = _prepared_single_source_arrays(
        times_ns=times_ns,
        position_mm=np.zeros_like(beta),
        beta=beta,
        beta_prime_per_mm=deliberately_mistimed_bdot,
    )

    np.testing.assert_allclose(
        prepared.beta_prime_per_mm[:, 0],
        expected_x,
        rtol=2.0e-12,
        atol=3.0e-17,
    )
    np.testing.assert_array_equal(prepared.beta_prime_per_mm[:, 1:], 0.0)


def test_prepared_analytic_history_can_declare_instantaneous_acceleration() -> None:
    times_ns = np.asarray((-0.004, -0.001, 0.002, 0.006), dtype=float)
    beta = np.zeros((times_ns.size, 3), dtype=float)
    supplied_endpoint_derivative = np.full_like(beta, 9.0e-3)
    arrays = retarded_fields._extract_history(
        _source_history(
            times_ns=times_ns,
            position_mm=np.zeros_like(beta),
            beta=beta,
            beta_prime_per_mm=supplied_endpoint_derivative,
        )
    )

    prepared = retarded_fields._prepare_source_history(
        arrays,
        0,
        source_acceleration_semantics="instantaneous",
    )

    np.testing.assert_array_equal(
        prepared.beta_prime_per_mm,
        supplied_endpoint_derivative,
    )


def _reference_full_scan_knot_bracket(
    source: retarded_fields._PreparedSourceHistory,
    *,
    observer_time_ns: float,
    observer_position_mm: np.ndarray,
) -> int | None:
    """Historical full-history bracket scan retained as a test oracle."""

    if source.time_ns.size < 2:
        return None
    knot_separations = np.linalg.norm(
        observer_position_mm[np.newaxis, :] - source.position_mm,
        axis=1,
    )
    knot_residuals = C_MMNS * (observer_time_ns - source.time_ns) - knot_separations
    brackets = np.flatnonzero(
        (knot_residuals[:-1] >= 0.0) & (knot_residuals[1:] <= 0.0)
    )
    if brackets.size == 0:
        return None
    return int(brackets[-1])


def _solve_with_reference_full_scan(
    monkeypatch: pytest.MonkeyPatch,
    source: retarded_fields._PreparedSourceHistory,
    *,
    observer_time_ns: float,
    observer_position_mm: np.ndarray,
) -> retarded_fields._RetardedSample | None:
    with monkeypatch.context() as patch:
        patch.setattr(
            retarded_fields,
            "_find_retarded_knot_bracket",
            lambda candidate, *, observer_time_ns, observer_position_mm: (
                _reference_full_scan_knot_bracket(
                    candidate,
                    observer_time_ns=observer_time_ns,
                    observer_position_mm=observer_position_mm,
                )
            ),
        )
        return retarded_fields._solve_retarded_sample(
            source,
            observer_time_ns=observer_time_ns,
            observer_position_mm=observer_position_mm,
            root_tolerance_mm=1.0e-21,
            max_root_iterations=96,
        )


def _si_lw_oracle(
    *,
    charge_native: float,
    separation_vector_mm: np.ndarray,
    source_beta: np.ndarray,
    source_beta_prime_per_mm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    separation_m = separation_vector_mm * _LENGTH_UNIT_M
    separation = float(np.linalg.norm(separation_m))
    direction = separation_m / separation
    beta_squared = float(source_beta @ source_beta)
    kappa = 1.0 - float(direction @ source_beta)
    beta_dot_s = source_beta_prime_per_mm * _C_SI / _LENGTH_UNIT_M
    velocity = (
        (1.0 - beta_squared) * (direction - source_beta) / (kappa**3 * separation**2)
    )
    radiation = np.cross(direction, np.cross(direction - source_beta, beta_dot_s)) / (
        _C_SI * kappa**3 * separation
    )
    electric_si = (
        _COULOMB_CONSTANT_SI
        * charge_native
        * _SOURCE_CHARGE_UNIT_C
        * (velocity + radiation)
    )
    magnetic_si = np.cross(direction, electric_si) / _C_SI
    return (
        electric_si / ELECTRIC_FIELD_NATIVE_TO_V_PER_M,
        magnetic_si / MAGNETIC_FIELD_NATIVE_TO_TESLA,
    )


@pytest.mark.parametrize(
    ("beta", "beta_prime"),
    [
        (np.zeros(3), np.zeros(3)),
        (np.array((0.21, -0.08, 0.04)), np.zeros(3)),
        (np.array((-0.12, 0.07, 0.19)), np.array((0.013, -0.021, 0.009))),
    ],
    ids=("static", "moving", "accelerated"),
)
def test_native_lw_kernel_matches_si_oracle(
    beta: np.ndarray, beta_prime: np.ndarray
) -> None:
    charge = -1.7 * ELEMENTARY_CHARGE
    separation = np.array((0.8, 1.1, -0.4))
    electric, magnetic = lienard_wiechert_charge_field_native(
        charge_native=charge,
        separation_vector_mm=separation,
        source_beta=beta,
        source_beta_prime_per_mm=beta_prime,
    )
    oracle_electric, oracle_magnetic = _si_lw_oracle(
        charge_native=charge,
        separation_vector_mm=separation,
        source_beta=beta,
        source_beta_prime_per_mm=beta_prime,
    )
    np.testing.assert_allclose(electric, oracle_electric, rtol=4.0e-15)
    np.testing.assert_allclose(magnetic, oracle_magnetic, rtol=4.0e-15, atol=2.0e-20)


def test_point_charge_kernel_reduces_to_native_coulomb_field() -> None:
    radius_mm = 2.5
    charge = 3.7 * ELEMENTARY_CHARGE
    electric, magnetic = lienard_wiechert_charge_field_native(
        charge_native=charge,
        separation_vector_mm=(radius_mm, 0.0, 0.0),
        source_beta=(0.0, 0.0, 0.0),
        source_beta_prime_per_mm=(0.0, 0.0, 0.0),
    )
    np.testing.assert_allclose(
        electric, (charge / radius_mm**2, 0.0, 0.0), rtol=1.0e-15
    )
    np.testing.assert_allclose(magnetic, 0.0, atol=0.0)


def test_stationary_source_uses_q_native_directly_without_renormalization() -> None:
    radius_mm = 1.0
    charge = 2.25 * ELEMENTARY_CHARGE
    field = evaluate_retarded_charge_field_native(
        _stationary_history(charge_native=charge),
        ObserverEvent(time_ns=0.0, position_mm=(radius_mm, 0.0, 0.0)),
    )

    assert field.retarded_time_ns[0] == pytest.approx(-radius_mm / C_MMNS, abs=1.0e-16)
    assert abs(field.light_cone_residual_mm[0]) <= 1.0e-15
    np.testing.assert_allclose(
        field.electric_field_native,
        (charge / radius_mm**2, 0.0, 0.0),
        rtol=2.0e-14,
    )
    np.testing.assert_allclose(field.magnetic_field_native, 0.0, atol=0.0)


def test_native_static_field_matches_legacy_charge_force_normalization() -> None:
    radius_mm = 1.7
    source_charge = 1.25 * ELEMENTARY_CHARGE
    observer_charge = -0.8 * ELEMENTARY_CHARGE
    field = evaluate_retarded_charge_field_native(
        _stationary_history(charge_native=source_charge),
        ObserverEvent(time_ns=0.0, position_mm=(radius_mm, 0.0, 0.0)),
    )
    h_step = 3.0e-4
    samples = ExternalSampleBatch(
        charge=np.array([source_charge]),
        gamma=np.array([1.0]),
        bx=np.zeros(1),
        by=np.zeros(1),
        bz=np.zeros(1),
        bdotx=np.zeros(1),
        bdoty=np.zeros(1),
        bdotz=np.zeros(1),
        valid_mask=np.array([True]),
    )
    legacy = compute_vectorized_contributions(
        h=h_step,
        charge_i=observer_charge,
        mass_i=1.0,
        gamma_i=1.0,
        beta_vec=(0.0, 0.0, 0.0),
        nhat_nx=np.array([1.0]),
        nhat_ny=np.array([0.0]),
        nhat_nz=np.array([0.0]),
        R_separation=np.array([radius_mm]),
        samples=samples,
        apply_external=True,
    )
    expected_impulse_x = h_step * observer_charge * field.electric_field_native[0]
    assert legacy[0] == pytest.approx(expected_impulse_x, rel=3.0e-15)


def test_uniform_motion_light_cone_root_matches_analytic_solution() -> None:
    beta_x = 0.2
    times_ns = np.linspace(-0.02, 0.002, 45)
    positions = np.zeros((times_ns.size, 3))
    positions[:, 0] = beta_x * C_MMNS * times_ns
    betas = np.zeros_like(positions)
    betas[:, 0] = beta_x
    history = _source_history(times_ns=times_ns, position_mm=positions, beta=betas)

    field = evaluate_retarded_charge_field_native(
        history, ObserverEvent(time_ns=0.0, position_mm=(1.0, 0.0, 0.0))
    )
    expected_time_ns = -1.0 / (C_MMNS * (1.0 - beta_x))
    assert field.retarded_time_ns[0] == pytest.approx(expected_time_ns, abs=2.0e-16)
    assert abs(field.light_cone_residual_mm[0]) <= 1.0e-15


def test_binary_bracket_and_root_match_full_scan_on_random_timelike_histories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(20260824)

    for _ in range(40):
        knot_count = int(rng.integers(8, 240))
        durations = rng.uniform(2.0e-5, 8.0e-4, size=knot_count - 1)
        times_ns = np.concatenate(([0.0], np.cumsum(durations)))
        times_ns -= 0.8 * times_ns[-1]

        base_beta = rng.normal(0.0, 0.08, size=3)
        amplitudes = rng.normal(0.0, 0.025, size=(2, 3))
        frequencies = rng.uniform(4.0, 18.0, size=2)
        phases = rng.uniform(-np.pi, np.pi, size=2)
        phase_at_knots = (
            times_ns[:, np.newaxis] * frequencies[np.newaxis, :] + phases[np.newaxis, :]
        )
        beta = base_beta[np.newaxis, :] + np.sum(
            amplitudes[np.newaxis, :, :] * np.sin(phase_at_knots)[:, :, np.newaxis],
            axis=1,
        )
        beta_prime_per_mm = (
            np.sum(
                amplitudes[np.newaxis, :, :]
                * frequencies[np.newaxis, :, np.newaxis]
                * np.cos(phase_at_knots)[:, :, np.newaxis],
                axis=1,
            )
            / C_MMNS
        )
        position_mm = C_MMNS * (
            times_ns[:, np.newaxis] * base_beta[np.newaxis, :]
            - np.sum(
                amplitudes[np.newaxis, :, :]
                * np.cos(phase_at_knots)[:, :, np.newaxis]
                / frequencies[np.newaxis, :, np.newaxis],
                axis=1,
            )
        )
        assert np.max(np.linalg.norm(beta, axis=1)) < 0.5

        source = _prepared_single_source(
            _source_history(
                times_ns=times_ns,
                position_mm=position_mm,
                beta=beta,
                beta_prime_per_mm=beta_prime_per_mm,
            )
        )
        target_segment = int(rng.integers(1, knot_count - 2))
        fraction = float(rng.uniform(0.1, 0.9))
        source_time_ns = float(
            times_ns[target_segment]
            + fraction * (times_ns[target_segment + 1] - times_ns[target_segment])
        )
        source_position, _, _ = retarded_fields._quintic_worldline_sample(
            source, target_segment, source_time_ns
        )
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        separation_mm = float(rng.uniform(0.05, 2.0))
        observer_position = source_position + separation_mm * direction
        observer_time_ns = source_time_ns + separation_mm / C_MMNS

        expected_segment = _reference_full_scan_knot_bracket(
            source,
            observer_time_ns=observer_time_ns,
            observer_position_mm=observer_position,
        )
        actual_segment = retarded_fields._find_retarded_knot_bracket(
            source,
            observer_time_ns=observer_time_ns,
            observer_position_mm=observer_position,
        )
        assert actual_segment == expected_segment

        optimized = retarded_fields._solve_retarded_sample(
            source,
            observer_time_ns=observer_time_ns,
            observer_position_mm=observer_position,
            root_tolerance_mm=1.0e-21,
            max_root_iterations=96,
        )
        reference = _solve_with_reference_full_scan(
            monkeypatch,
            source,
            observer_time_ns=observer_time_ns,
            observer_position_mm=observer_position,
        )
        assert optimized is not None
        assert reference is not None
        assert optimized.time_ns == reference.time_ns
        assert optimized.residual_mm == reference.residual_mm
        assert optimized.separation_mm == reference.separation_mm
        np.testing.assert_array_equal(optimized.position_mm, reference.position_mm)
        np.testing.assert_array_equal(optimized.beta, reference.beta)
        np.testing.assert_array_equal(
            optimized.beta_prime_per_mm,
            reference.beta_prime_per_mm,
        )


def test_binary_bracket_preserves_latest_segment_at_exact_internal_knot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact_root_time_ns = -1.0 / C_MMNS
    times_ns = np.array((-0.02, -0.01, exact_root_time_ns, -0.002, 0.001))
    source = _prepared_single_source(
        _source_history(
            times_ns=times_ns,
            position_mm=np.zeros((times_ns.size, 3)),
            beta=np.zeros((times_ns.size, 3)),
        )
    )
    observer_position = np.array((1.0, 0.0, 0.0))

    assert (
        retarded_fields._knot_light_cone_residual_mm(
            source,
            2,
            observer_time_ns=0.0,
            observer_position_mm=observer_position,
        )
        == 0.0
    )
    assert (
        _reference_full_scan_knot_bracket(
            source,
            observer_time_ns=0.0,
            observer_position_mm=observer_position,
        )
        == 2
    )
    assert (
        retarded_fields._find_retarded_knot_bracket(
            source,
            observer_time_ns=0.0,
            observer_position_mm=observer_position,
        )
        == 2
    )

    optimized = retarded_fields._solve_retarded_sample(
        source,
        observer_time_ns=0.0,
        observer_position_mm=observer_position,
        root_tolerance_mm=1.0e-21,
        max_root_iterations=96,
    )
    reference = _solve_with_reference_full_scan(
        monkeypatch,
        source,
        observer_time_ns=0.0,
        observer_position_mm=observer_position,
    )
    assert optimized is not None
    assert reference is not None
    assert optimized.time_ns == reference.time_ns == exact_root_time_ns


@pytest.mark.parametrize(
    "times_ns",
    (
        np.array((-0.001, 0.0)),
        np.array((-0.1, -0.05)),
    ),
    ids=("history_starts_too_late", "history_ends_too_early"),
)
def test_binary_bracket_matches_full_scan_when_no_root_is_bracketed(
    times_ns: np.ndarray,
) -> None:
    source = _prepared_single_source(
        _source_history(
            times_ns=times_ns,
            position_mm=np.zeros((times_ns.size, 3)),
            beta=np.zeros((times_ns.size, 3)),
        )
    )
    observer_position = np.array((10.0, 0.0, 0.0))

    expected = _reference_full_scan_knot_bracket(
        source,
        observer_time_ns=0.0,
        observer_position_mm=observer_position,
    )
    actual = retarded_fields._find_retarded_knot_bracket(
        source,
        observer_time_ns=0.0,
        observer_position_mm=observer_position,
    )
    assert actual == expected is None


def test_binary_bracket_uses_only_the_alive_prefix_of_a_lost_source() -> None:
    times_ns = np.array((-0.02, -0.01, -0.005, -0.005, -0.005))
    history = _source_history(
        times_ns=times_ns,
        position_mm=np.zeros((times_ns.size, 3)),
        beta=np.zeros((times_ns.size, 3)),
    )
    history[-2]["_dead_particles"][0] = True
    history[-1]["_dead_particles"][0] = True
    source = _prepared_single_source(history)
    observer_position = np.array((1.0, 0.0, 0.0))

    assert source.ended_by_loss
    assert source.time_ns.size == 3
    assert retarded_fields._find_retarded_knot_bracket(
        source,
        observer_time_ns=0.0,
        observer_position_mm=observer_position,
    ) == _reference_full_scan_knot_bracket(
        source,
        observer_time_ns=0.0,
        observer_position_mm=observer_position,
    )


def test_binary_bracket_matches_full_scan_for_near_lightlike_timelike_chords(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    beta_x = 1.0 - 1.0e-10
    normalized_knots = np.linspace(0.0, 1.0, 257) ** 1.7
    times_ns = -0.02 + 0.04 * normalized_knots
    positions = np.zeros((times_ns.size, 3))
    positions[:, 0] = beta_x * C_MMNS * times_ns
    betas = np.zeros_like(positions)
    betas[:, 0] = beta_x
    source = _prepared_single_source(
        _source_history(
            times_ns=times_ns,
            position_mm=positions,
            beta=betas,
        )
    )
    target_segment = 127
    source_time_ns = float(
        0.37 * times_ns[target_segment] + 0.63 * times_ns[target_segment + 1]
    )
    source_position, _, _ = retarded_fields._quintic_worldline_sample(
        source, target_segment, source_time_ns
    )
    separation_mm = 0.25
    observer_position = source_position + np.array((separation_mm, 0.0, 0.0))
    observer_time_ns = source_time_ns + separation_mm / C_MMNS

    assert retarded_fields._find_retarded_knot_bracket(
        source,
        observer_time_ns=observer_time_ns,
        observer_position_mm=observer_position,
    ) == _reference_full_scan_knot_bracket(
        source,
        observer_time_ns=observer_time_ns,
        observer_position_mm=observer_position,
    )
    optimized = retarded_fields._solve_retarded_sample(
        source,
        observer_time_ns=observer_time_ns,
        observer_position_mm=observer_position,
        root_tolerance_mm=1.0e-21,
        max_root_iterations=96,
    )
    reference = _solve_with_reference_full_scan(
        monkeypatch,
        source,
        observer_time_ns=observer_time_ns,
        observer_position_mm=observer_position,
    )
    assert optimized is not None
    assert reference is not None
    assert optimized.time_ns == reference.time_ns
    assert optimized.residual_mm == reference.residual_mm
    np.testing.assert_array_equal(optimized.position_mm, reference.position_mm)


def test_binary_bracket_residual_work_is_logarithmic_at_100k_knots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    knot_count = 100_000
    times_ns = np.linspace(-0.1, 0.01, knot_count)
    source = _prepared_single_source_arrays(
        times_ns=times_ns,
        position_mm=np.zeros((knot_count, 3)),
        beta=np.zeros((knot_count, 3)),
    )
    original = retarded_fields._knot_light_cone_residual_mm
    residual_evaluations = 0

    def counted_residual(*args, **kwargs):
        nonlocal residual_evaluations
        residual_evaluations += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        retarded_fields,
        "_knot_light_cone_residual_mm",
        counted_residual,
    )
    sample = retarded_fields._solve_retarded_sample(
        source,
        observer_time_ns=0.0,
        observer_position_mm=np.array((1.0, 0.0, 0.0)),
        root_tolerance_mm=1.0e-21,
        max_root_iterations=96,
    )

    assert sample is not None
    maximum_binary_work = 2 + math.ceil(math.log2(knot_count - 1))
    assert residual_evaluations <= maximum_binary_work
    assert residual_evaluations < 25


def test_uniform_motion_newton_root_needs_few_interpolated_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    beta_x = 0.2
    times_ns = np.linspace(-0.02, 0.002, 45)
    positions = np.zeros((times_ns.size, 3))
    positions[:, 0] = beta_x * C_MMNS * times_ns
    betas = np.zeros_like(positions)
    betas[:, 0] = beta_x
    history = _source_history(
        times_ns=times_ns,
        position_mm=positions,
        beta=betas,
    )
    original = retarded_fields._quintic_worldline_sample
    original_residual = retarded_fields._light_cone_residual_mm
    sample_count = 0
    scalar_residual_count = 0

    def counted_sample(*args, **kwargs):
        nonlocal sample_count
        sample_count += 1
        return original(*args, **kwargs)

    def counted_residual(*args, **kwargs):
        nonlocal scalar_residual_count
        scalar_residual_count += 1
        return original_residual(*args, **kwargs)

    monkeypatch.setattr(retarded_fields, "_quintic_worldline_sample", counted_sample)
    monkeypatch.setattr(retarded_fields, "_light_cone_residual_mm", counted_residual)

    field = evaluate_retarded_charge_field_native(
        history,
        ObserverEvent(time_ns=0.0, position_mm=(1.0, 0.0, 0.0)),
    )

    assert sample_count <= 6
    assert scalar_residual_count <= 6
    assert abs(field.light_cone_residual_mm[0]) <= 1.0e-15


@pytest.mark.parametrize(
    "derivative",
    (float("nan"), 0.0, 1.0, -1.0e-300),
)
def test_safeguarded_newton_bisects_for_unusable_or_near_kappa_derivative(
    derivative: float,
) -> None:
    decision = retarded_fields._next_safeguarded_root_trial(
        trial_time_ns=0.5,
        residual_mm=0.25,
        derivative_mm_per_ns=derivative,
        lower_time_ns=0.5,
        upper_time_ns=1.0,
    )

    assert decision.time_ns == 0.75
    assert decision.used_bisection
    assert not decision.used_nextafter


def test_safeguarded_newton_retains_valid_step_and_breaks_rounding_stall() -> None:
    newton = retarded_fields._next_safeguarded_root_trial(
        trial_time_ns=0.5,
        residual_mm=0.25,
        derivative_mm_per_ns=-1.0,
        lower_time_ns=0.5,
        upper_time_ns=1.0,
    )
    stalled = retarded_fields._next_safeguarded_root_trial(
        trial_time_ns=0.5,
        residual_mm=float(np.nextafter(0.0, 1.0)),
        derivative_mm_per_ns=-1.0,
        lower_time_ns=0.5,
        upper_time_ns=1.0,
    )

    assert newton.time_ns == 0.75
    assert not newton.used_bisection
    assert not newton.used_nextafter
    assert stalled.time_ns == np.nextafter(0.5, 1.0)
    assert not stalled.used_bisection
    assert stalled.used_nextafter


def test_complete_gradient_matches_static_coulomb_jacobian() -> None:
    radius_mm = 1.0
    result = evaluate_retarded_charge_field_gradient_native(
        _stationary_history(),
        ObserverEvent(time_ns=0.0, position_mm=(radius_mm, 0.0, 0.0)),
        relative_step=2.0e-5,
    )

    coefficient = ELEMENTARY_CHARGE / radius_mm**3
    expected_electric_gradient = coefficient * np.diag((-2.0, 1.0, 1.0))
    recovered_electric_gradient = np.empty((3, 3), dtype=float)
    for coordinate in range(3):
        recovered_electric_gradient[:, coordinate] = -result.partial_f[
            coordinate + 1, 0, 1:4
        ]
    np.testing.assert_allclose(
        recovered_electric_gradient,
        expected_electric_gradient,
        rtol=2.0e-8,
        atol=1.0e-18,
    )
    np.testing.assert_allclose(result.partial_f[0], 0.0, atol=1.0e-18)
    np.testing.assert_allclose(
        result.partial_f + np.swapaxes(result.partial_f, 1, 2),
        0.0,
        atol=1.0e-18,
    )


def test_gradient_stencil_resolves_a_new_retarded_event_at_every_point() -> None:
    result = evaluate_retarded_charge_field_gradient_native(
        _stationary_history(),
        ObserverEvent(time_ns=0.0, position_mm=(1.0, 0.0, 0.0)),
    )
    assert (
        result.stencil_retarded_time_ns[0, 0, 0]
        != result.stencil_retarded_time_ns[0, 1, 0]
    )
    assert (
        result.stencil_retarded_time_ns[1, 0, 0]
        != result.stencil_retarded_time_ns[1, 1, 0]
    )


def test_gradient_extracts_and_prepares_shared_history_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = retarded_fields._extract_history
    extraction_count = 0

    def counted_extract(history):
        nonlocal extraction_count
        extraction_count += 1
        return original(history)

    monkeypatch.setattr(retarded_fields, "_extract_history", counted_extract)

    evaluate_retarded_charge_field_gradient_native(
        _stationary_history(),
        ObserverEvent(time_ns=0.0, position_mm=(1.0, 0.0, 0.0)),
    )

    assert extraction_count == 1


def test_uniform_motion_complete_gradient_matches_heaviside_field() -> None:
    beta = np.array((0.21, -0.08, 0.04))
    times_ns = np.linspace(-0.03, 0.003, 67)
    positions_mm = times_ns[:, None] * C_MMNS * beta[None, :]
    history = _source_history(
        times_ns=times_ns,
        position_mm=positions_mm,
        beta=np.broadcast_to(beta, positions_mm.shape).copy(),
    )
    separation = np.array((0.8, 1.1, -0.4))
    result = evaluate_retarded_charge_field_gradient_native(
        history,
        ObserverEvent(time_ns=0.0, position_mm=tuple(separation)),
        relative_step=1.0e-5,
    )

    beta_squared = float(beta @ beta)
    matrix_a = (1.0 - beta_squared) * np.eye(3) + np.outer(beta, beta)
    denominator = float(separation @ matrix_a @ separation)
    coefficient = ELEMENTARY_CHARGE * (1.0 - beta_squared)
    expected_electric = coefficient * separation * denominator ** (-1.5)
    expected_magnetic = np.cross(beta, expected_electric)
    electric_gradient = np.empty((3, 3), dtype=float)
    magnetic_gradient = np.empty((3, 3), dtype=float)
    a_on_r = matrix_a @ separation
    for coordinate in range(3):
        electric_gradient[:, coordinate] = coefficient * (
            np.eye(3)[:, coordinate] * denominator ** (-1.5)
            - 3.0 * separation * a_on_r[coordinate] * denominator ** (-2.5)
        )
        magnetic_gradient[:, coordinate] = np.cross(
            beta, electric_gradient[:, coordinate]
        )
    expected_partial = np.zeros((4, 4, 4), dtype=float)
    expected_partial[0] = electromagnetic_field_tensor_native(
        -(electric_gradient @ beta), -(magnetic_gradient @ beta)
    )
    for coordinate in range(3):
        expected_partial[coordinate + 1] = electromagnetic_field_tensor_native(
            electric_gradient[:, coordinate], magnetic_gradient[:, coordinate]
        )

    np.testing.assert_allclose(
        result.field.electric_field_native, expected_electric, rtol=3.0e-12
    )
    np.testing.assert_allclose(
        result.field.magnetic_field_native, expected_magnetic, rtol=3.0e-12
    )
    np.testing.assert_allclose(
        result.partial_f, expected_partial, rtol=2.0e-7, atol=1.0e-18
    )
    medium = evaluate_retarded_charge_field_gradient_native(
        history,
        ObserverEvent(time_ns=0.0, position_mm=tuple(separation)),
        relative_step=2.0e-5,
    )
    coarse = evaluate_retarded_charge_field_gradient_native(
        history,
        ObserverEvent(time_ns=0.0, position_mm=tuple(separation)),
        relative_step=4.0e-5,
    )
    fine_error = float(np.linalg.norm(result.partial_f - expected_partial))
    medium_error = float(np.linalg.norm(medium.partial_f - expected_partial))
    coarse_error = float(np.linalg.norm(coarse.partial_f - expected_partial))
    assert coarse_error / medium_error > 3.5
    assert medium_error / fine_error > 3.5


def test_missing_retarded_history_is_explicit() -> None:
    times_ns = np.array((-0.001, 0.0))
    history = _source_history(
        times_ns=times_ns,
        position_mm=np.zeros((2, 3)),
        beta=np.zeros((2, 3)),
    )
    with pytest.raises(RetardedHistoryError, match="does not bracket"):
        evaluate_retarded_charge_field_native(
            history, ObserverEvent(time_ns=0.0, position_mm=(10.0, 0.0, 0.0))
        )
    partial = evaluate_retarded_charge_field_native(
        history,
        ObserverEvent(time_ns=0.0, position_mm=(10.0, 0.0, 0.0)),
        require_complete_history=False,
    )
    assert not partial.valid_sources.any()
    np.testing.assert_allclose(partial.field_tensor, 0.0, atol=0.0)


def test_lost_source_uses_alive_prefix_despite_frozen_dead_timestamps() -> None:
    times_ns = np.array((-0.010, -0.005, -0.002, -0.002))
    history = _source_history(
        times_ns=times_ns,
        position_mm=np.zeros((times_ns.size, 3)),
        beta=np.zeros((times_ns.size, 3)),
    )
    history[-1]["_dead_particles"][0] = True
    field = evaluate_retarded_charge_field_native(
        history, ObserverEvent(time_ns=0.0, position_mm=(1.0, 0.0, 0.0))
    )
    assert field.valid_sources[0]
    assert field.retarded_time_ns[0] == pytest.approx(-1.0 / C_MMNS, abs=1.0e-16)
    assert field.electric_field_native[0] > 0.0


def test_lost_source_is_absent_after_its_alive_history_ends() -> None:
    times_ns = np.array((-0.010, -0.008, -0.008))
    history = _source_history(
        times_ns=times_ns,
        position_mm=np.zeros((times_ns.size, 3)),
        beta=np.zeros((times_ns.size, 3)),
    )
    history[-1]["_dead_particles"][0] = True
    field = evaluate_retarded_charge_field_native(
        history, ObserverEvent(time_ns=0.0, position_mm=(1.0, 0.0, 0.0))
    )
    assert not field.valid_sources[0]
    np.testing.assert_array_equal(field.field_tensor, 0.0)
