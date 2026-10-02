"""Experimental diagnostic floor: isolation, quantization, and default parity."""

from dataclasses import replace

import numpy as np
import pytest

from core.exact_pair_integration import _scaled_tolerances
from core.step_doubling import (
    ErrorScale,
    StepControllerConfig,
    StepDoublingState,
    assess_step_doubling,
    build_pair_step_doubling_state,
    propose_next_step_ns,
)
from core.types import AdaptivePairReturnConfig


def _state(projection=0.0, radiation=0.0, reference=1929.7068553):
    return StepDoublingState(
        position_mm=np.zeros((2, 3)),
        mechanical_momentum_native=np.zeros((2, 3)),
        rest_spin=np.zeros((2, 3)),
        diagnostics_native=np.array([[radiation, 0, 0, projection], [0, 0, 0, 0]]),
        projection_reference_energy_native=np.array([reference, reference]),
    )


def _assess(full, refined, **options):
    return assess_step_doubling(
        full, refined, method_order=1, tolerances=_scaled_tolerances(1, **options)
    )


def test_floor_changes_only_projection_and_records_the_changed_decision():
    full, refined = _state(), _state(projection=4.3293e-13)
    ordinary = _assess(full, refined)
    experimental = _assess(full, refined, experimental_projection_ulp_floor=True)
    assert not ordinary.accepted
    assert experimental.accepted
    assert experimental.projection_floor_applied
    assert experimental.projection_floor_changed_acceptance
    assert experimental.maximum_projection_floor_native == 4 * np.spacing(1929.7068553)
    assert experimental.diagnostics_error_without_floor == ordinary.diagnostics_error
    assert not _assess(
        full, _state(radiation=4.3293e-13), experimental_projection_ulp_floor=True
    ).accepted


def test_floor_preserves_float64_scaling_for_integer_diagnostic_inputs():
    full = replace(_state(), diagnostics_native=np.zeros((2, 4), dtype=int))
    changed = full.diagnostics_native.copy()
    changed[0, 0] = 1  # Radiation must retain its original absolute tolerance.
    refined = replace(full, diagnostics_native=changed)
    tolerances = replace(
        _scaled_tolerances(1, experimental_projection_ulp_floor=True),
        diagnostics_native=ErrorScale(1e-13, 0),
    )
    result = assess_step_doubling(full, refined, method_order=1, tolerances=tolerances)
    assert not result.accepted
    assert result.diagnostics_error == 1e13


def test_atol_stays_authoritative_for_small_reference_and_k_is_configurable():
    full, refined = _state(reference=0), _state(projection=2e-13, reference=0)
    result = _assess(full, refined, experimental_projection_ulp_floor=True)
    assert not result.accepted
    assert not result.projection_floor_applied
    assert not result.projection_floor_changed_acceptance
    assert not _assess(
        _state(),
        _state(projection=4.3293e-13),
        experimental_projection_ulp_floor=True,
        projection_ulp_multiplier=1,
    ).accepted


def test_floor_cannot_hide_position_or_momentum_failure():
    for group in ("position_mm", "mechanical_momentum_native"):
        refined = replace(_state(projection=4.3293e-13), **{group: np.ones((2, 3))})
        result = _assess(_state(), refined, experimental_projection_ulp_floor=True)
        assert not result.accepted
        assert not result.projection_floor_changed_acceptance


def test_accepted_rounding_floor_can_still_shrink_the_existing_controller():
    # Two proton momentum ulps expressed as energy are about 0.952 of the
    # four-kinetic-energy-ulp floor. A fixed representational residual does
    # not decrease with slab width; this tests the existing policy, not a fix.
    result = _assess(
        _state(),
        _state(projection=2 * 4.3293088652e-13),
        experimental_projection_ulp_floor=True,
    )
    assert result.accepted
    assert result.normalized_error > 0.9**2
    h = 1e-3
    minimum = h / 64
    for _ in range(100):
        proposed = propose_next_step_ns(
            h,
            result.normalized_error,
            accepted=True,
            config=StepControllerConfig(method_order=1),
            minimum_step_ns=minimum,
            maximum_step_ns=0.064,
        )
        assert proposed <= h
        h = proposed
    assert h == minimum


def test_default_off_is_bit_identical_to_original_error_formula():
    rng = np.random.default_rng(4774)
    for _ in range(50):
        full = replace(_state(), diagnostics_native=rng.normal(0, 1e-12, (2, 4)))
        refined = replace(_state(), diagnostics_native=rng.normal(0, 1e-12, (2, 4)))
        scale = _scaled_tolerances(1).diagnostics_native
        # Frozen pre-option diagnostic formula, including operation order.
        error = np.abs(refined.diagnostics_native - full.diagnostics_native) / 1.0
        denominator = scale.absolute + scale.relative * np.maximum(
            np.abs(full.diagnostics_native), np.abs(refined.diagnostics_native)
        )
        normalized = np.divide(
            error, denominator, out=np.zeros_like(error), where=denominator > 0
        )
        result = _assess(full, refined, projection_ulp_multiplier=1000)
        assert (
            np.float64(result.diagnostics_error).tobytes()
            == np.max(normalized).tobytes()
        )
        assert result.accepted == (np.max(normalized) <= 1)
        assert not result.projection_floor_applied
        assert result.diagnostics_error_without_floor is None


def test_tighter_control_changes_only_position_and_momentum_scales():
    baseline = _scaled_tolerances(1, experimental_projection_ulp_floor=True)
    tight = _scaled_tolerances(
        1, position_momentum_scale=0.1, experimental_projection_ulp_floor=True
    )
    for name in ("position_mm", "mechanical_momentum_native"):
        assert getattr(tight, name).absolute == getattr(baseline, name).absolute * 0.1
        assert getattr(tight, name).relative == getattr(baseline, name).relative * 0.1
    assert tight.rest_spin == baseline.rest_spin
    assert tight.diagnostics_native == baseline.diagnostics_native
    assert tight.projection_ulp_multiplier == baseline.projection_ulp_multiplier


def test_pair_reference_uses_physical_mass_and_default_reducer_skips_it():
    state = {
        **{name: np.zeros(1) for name in ("x", "y", "z", "bx", "by")},
        "bz": np.array([0.2]),
        "gamma": np.array([1 / np.sqrt(1 - 0.2**2)]),
        "m": np.array([100.0]),
        "m_species": np.array([1.0]),
    }
    ordinary = build_pair_step_doubling_state(
        rider_states=(state,), driver_states=(state,)
    )
    floored = build_pair_step_doubling_state(
        rider_states=(state,),
        driver_states=(state,),
        experimental_projection_ulp_floor=True,
    )
    assert ordinary.projection_reference_energy_native is None
    np.testing.assert_allclose(
        floored.projection_reference_energy_native,
        (state["gamma"][0] - 1) * 299.792458**2,
        rtol=1e-14,
    )
    for field in (
        "position_mm",
        "mechanical_momentum_native",
        "rest_spin",
        "diagnostics_native",
    ):
        assert getattr(ordinary, field).tobytes() == getattr(floored, field).tobytes()


@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf])
def test_invalid_multiplier_and_control_scale_fail_early(value):
    with pytest.raises(ValueError):
        AdaptivePairReturnConfig(projection_ulp_multiplier=value)
    with pytest.raises(ValueError):
        AdaptivePairReturnConfig(position_momentum_tolerance_scale=value)
    with pytest.raises(ValueError):
        _scaled_tolerances(1, projection_ulp_multiplier=value)


def test_floor_fails_closed_without_a_reference_or_with_an_invalid_reference():
    for reference in (None, np.array([np.nan, 1]), np.array([-1, 1]), np.ones(3)):
        with pytest.raises(ValueError, match="reference|finite|kinetic energies"):
            _assess(
                replace(_state(), projection_reference_energy_native=reference),
                _state(),
                experimental_projection_ulp_floor=True,
            )
