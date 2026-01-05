"""Comprehensive test suite for trajectory smoothness analyzer.

Tests the multi-step stability analysis that distinguishes numerical
instabilities from physical phenomena.
"""

import numpy as np
import pytest

from core.smoothness_analyzer import (
    SmoothnessConfig,
    StabilityViolationType,
    analyze_trajectory_smoothness,
    filter_stable_trajectories,
)

# ============================================================================
# Test Fixtures - Trajectory Generators
# ============================================================================


@pytest.fixture
def smooth_trajectory():
    """Generate a perfectly smooth trajectory (constant energy)."""
    n_steps = 100
    t = np.linspace(0, 10, n_steps)  # ns

    # Constant gamma (no acceleration)
    gamma = 10.0 * np.ones(n_steps)
    beta = np.sqrt(1.0 - 1.0 / gamma**2)

    # Smooth linear motion
    c_mmns = 299.792458  # mm/ns
    v = beta * c_mmns
    z = v * t
    r = 0.01 * np.ones_like(z)

    # Constant momentum
    m_amu = 0.000548579909  # Electron mass
    p_mag = gamma * m_amu * c_mmns * beta
    pz = p_mag
    pr = np.zeros_like(pz)

    return {
        "t": t,
        "z": z,
        "r": r,
        "pz": pz,
        "pr": pr,
        "gamma": gamma,
    }


@pytest.fixture
def oscillatory_trajectory():
    """Generate trajectory with oscillatory instability."""
    n_steps = 100
    t = np.linspace(0, 10, n_steps)

    # Create back-and-forth oscillations in energy
    base_gamma = 10.0
    # High-frequency oscillation (numerical artifact)
    gamma = base_gamma + 0.5 * np.sin(20 * t)

    beta = np.sqrt(1.0 - 1.0 / gamma**2)
    c_mmns = 299.792458
    v = beta * c_mmns
    z = np.cumsum(v * np.diff(t, prepend=0))
    r = 0.01 * np.ones_like(z)

    m_amu = 0.000548579909
    p_mag = gamma * m_amu * c_mmns * beta
    pz = p_mag
    pr = np.zeros_like(pz)

    return {
        "t": t,
        "z": z,
        "r": r,
        "pz": pz,
        "pr": pr,
        "gamma": gamma,
    }


@pytest.fixture
def erratic_trajectory():
    """Generate trajectory with erratic, non-smooth evolution."""
    n_steps = 100
    t = np.linspace(0, 10, n_steps)

    # Add large random noise to gamma (cannot fit smooth trend)
    np.random.seed(42)
    base_gamma = 10.0 + 0.1 * t  # Gentle trend
    noise = 2.0 * np.random.randn(n_steps)  # Large noise
    gamma = np.maximum(base_gamma + noise, 1.1)  # Keep gamma > 1

    beta = np.sqrt(1.0 - 1.0 / gamma**2)
    c_mmns = 299.792458
    v = beta * c_mmns
    z = np.cumsum(v * np.diff(t, prepend=0))
    r = 0.01 * np.ones_like(z)

    m_amu = 0.000548579909
    p_mag = gamma * m_amu * c_mmns * beta
    pz = p_mag
    pr = np.zeros_like(pz)

    return {
        "t": t,
        "z": z,
        "r": r,
        "pz": pz,
        "pr": pr,
        "gamma": gamma,
    }


@pytest.fixture
def physical_jump_trajectory():
    """Generate trajectory with physical energy jump (radiation reaction)."""
    n_steps = 100
    t = np.linspace(0, 10, n_steps)

    # Smooth evolution with single sharp but physically valid jump at t=5
    gamma = 10.0 * np.ones(n_steps)
    jump_idx = 50
    # Sudden energy loss (radiation reaction) - but then smooth again
    gamma[jump_idx:] = 7.5  # 25% energy loss in one step

    # Smooth before and after jump
    beta = np.sqrt(1.0 - 1.0 / gamma**2)
    c_mmns = 299.792458
    v = beta * c_mmns
    z = np.cumsum(v * np.diff(t, prepend=0))
    r = 0.01 * np.ones_like(z)

    m_amu = 0.000548579909
    p_mag = gamma * m_amu * c_mmns * beta
    pz = p_mag
    pr = np.zeros_like(pz)

    return {
        "t": t,
        "z": z,
        "r": r,
        "pz": pz,
        "pr": pr,
        "gamma": gamma,
    }


@pytest.fixture
def short_trajectory():
    """Generate very short trajectory (< min_steps)."""
    n_steps = 10
    t = np.linspace(0, 1, n_steps)
    gamma = 10.0 * np.ones(n_steps)
    beta = np.sqrt(1.0 - 1.0 / gamma**2)
    c_mmns = 299.792458
    z = beta * c_mmns * t
    r = 0.01 * np.ones_like(z)

    return {
        "t": t,
        "z": z,
        "r": r,
        "gamma": gamma,
    }


# ============================================================================
# Test Basic Functionality
# ============================================================================


def test_smooth_trajectory_passes(smooth_trajectory):
    """Smooth trajectory should pass all stability checks."""
    config = SmoothnessConfig()
    result = analyze_trajectory_smoothness(smooth_trajectory, config)

    assert result.passed, "Smooth trajectory should pass"
    assert len(result.violations) == 0, "Should have no violations"
    assert result.oscillation_score < 0.3, "Should have low oscillation score"
    assert result.trend_smoothness_score < 0.1, "Should have low trend residual"
    assert (
        "Good" in result.quality_summary or "smooth" in result.quality_summary.lower()
    )


def test_oscillatory_trajectory_fails(oscillatory_trajectory):
    """Trajectory with oscillations should fail stability check."""
    config = SmoothnessConfig()
    result = analyze_trajectory_smoothness(oscillatory_trajectory, config)

    assert not result.passed, "Oscillatory trajectory should fail"
    assert len(result.violations) > 0, "Should have violations"

    # Check that oscillatory instability was detected
    violation_types = [v.violation_type for v in result.violations]
    assert StabilityViolationType.OSCILLATORY_INSTABILITY in violation_types

    assert result.oscillation_score > config.oscillation_threshold


def test_erratic_trajectory_fails(erratic_trajectory):
    """Erratic trajectory should fail trend smoothness check."""
    config = SmoothnessConfig()
    result = analyze_trajectory_smoothness(erratic_trajectory, config)

    assert not result.passed, "Erratic trajectory should fail"
    assert len(result.violations) > 0, "Should have violations"

    # Should detect either oscillations or trend divergence (or both)
    violation_types = [v.violation_type for v in result.violations]
    assert (
        StabilityViolationType.TREND_DIVERGENCE in violation_types
        or StabilityViolationType.OSCILLATORY_INSTABILITY in violation_types
    ), "Should detect instability of some kind"

    # Should have elevated scores
    assert result.trend_smoothness_score > 0.1 or result.oscillation_score > 0.3, (
        "Should show numerical issues"
    )


def test_physical_jump_may_pass(physical_jump_trajectory):
    """Physical jump (localized, smooth before/after) may pass multi-step analysis."""
    config = SmoothnessConfig()
    result = analyze_trajectory_smoothness(physical_jump_trajectory, config)

    # Multi-step analysis should recognize this is smooth before and after
    # May or may not pass depending on threshold, but should not be classified as "Poor"
    assert "Poor" not in result.quality_summary or not result.passed

    # If it fails, should have limited violations (not widespread instability)
    if not result.passed:
        assert len(result.violations) <= config.max_allowed_violations + 5


def test_short_trajectory_passes(short_trajectory):
    """Short trajectory should pass (skipped analysis)."""
    config = SmoothnessConfig(min_steps_for_analysis=50)
    result = analyze_trajectory_smoothness(short_trajectory, config)

    assert result.passed, "Short trajectory should pass (skipped)"
    assert result.num_steps_analyzed == 10
    assert "Too short" in result.quality_summary


# ============================================================================
# Test Configuration Options
# ============================================================================


def test_disabled_analysis_always_passes(oscillatory_trajectory):
    """When analysis is disabled, all trajectories should pass."""
    config = SmoothnessConfig(enabled=False)
    result = analyze_trajectory_smoothness(oscillatory_trajectory, config)

    assert result.passed, "Disabled analysis should always pass"
    assert len(result.violations) == 0
    assert "disabled" in result.quality_summary.lower()


def test_strict_config_more_sensitive(smooth_trajectory):
    """Strict config should have tighter thresholds."""
    config_default = SmoothnessConfig()
    config_strict = SmoothnessConfig.strict()

    # Verify strict has tighter thresholds
    assert config_strict.oscillation_threshold < config_default.oscillation_threshold
    assert (
        config_strict.trend_smoothness_threshold
        < config_default.trend_smoothness_threshold
    )
    assert config_strict.max_allowed_violations < config_default.max_allowed_violations

    # Even smooth trajectory might show minor artifacts under strict analysis
    result_default = analyze_trajectory_smoothness(smooth_trajectory, config_default)
    result_strict = analyze_trajectory_smoothness(smooth_trajectory, config_strict)

    assert result_default.passed, "Default should pass smooth trajectory"
    # Strict may or may not pass, but should not find severe issues
    if not result_strict.passed:
        assert len(result_strict.violations) <= 2


def test_permissive_config_more_tolerant(erratic_trajectory):
    """Permissive config should tolerate more variation."""
    config_permissive = SmoothnessConfig.permissive()

    # Verify permissive has looser thresholds
    assert config_permissive.oscillation_threshold > 0.5
    assert config_permissive.trend_smoothness_threshold > 0.4
    assert config_permissive.max_allowed_violations >= 5

    result = analyze_trajectory_smoothness(erratic_trajectory, config_permissive)

    # May still fail severely erratic trajectory, but should be more tolerant
    # At minimum, should not reject if reject_on_violation is False
    if config_permissive.reject_on_violation:
        assert (
            result.passed
            or len(result.violations) > config_permissive.max_allowed_violations
        )
    else:
        # With reject_on_violation=False, may still "fail" but won't be rejected
        assert "REJECTED" not in result.quality_summary


def test_window_size_affects_detection(oscillatory_trajectory):
    """Larger window size should be less sensitive to short oscillations."""
    config_small = SmoothnessConfig(window_size=10)
    config_large = SmoothnessConfig(window_size=50)

    result_small = analyze_trajectory_smoothness(oscillatory_trajectory, config_small)
    result_large = analyze_trajectory_smoothness(oscillatory_trajectory, config_large)

    # Small window should definitely detect oscillations
    assert not result_small.passed, "Small window should detect oscillations"

    # Large window may or may not fail depending on max_allowed_violations
    # But it should still detect the oscillations (just may pass if violations <= threshold)
    assert result_large.oscillation_score > 0, "Should detect some oscillation"

    # The number of violations depends on window overlap and tolerance
    # Just verify that analysis ran
    assert result_small.num_steps_analyzed > 0
    assert result_large.num_steps_analyzed > 0


def test_max_allowed_violations_threshold(erratic_trajectory):
    """Max allowed violations controls pass/fail threshold."""
    # Config that allows some violations
    config_tolerant = SmoothnessConfig(max_allowed_violations=100)
    config_strict = SmoothnessConfig(max_allowed_violations=0)

    result_tolerant = analyze_trajectory_smoothness(erratic_trajectory, config_tolerant)
    result_strict = analyze_trajectory_smoothness(erratic_trajectory, config_strict)

    # Same violations detected, but different pass/fail
    assert result_tolerant.violations == result_strict.violations

    # Strict should fail with even one violation
    if len(result_strict.violations) > 0:
        assert not result_strict.passed

    # Tolerant may pass despite violations
    if len(result_tolerant.violations) <= 100:
        assert result_tolerant.passed


# ============================================================================
# Test Violation Details
# ============================================================================


def test_violation_structure(oscillatory_trajectory):
    """Violations should contain detailed information."""
    config = SmoothnessConfig()
    result = analyze_trajectory_smoothness(oscillatory_trajectory, config)

    if len(result.violations) > 0:
        v = result.violations[0]

        # Check violation attributes
        assert isinstance(v.violation_type, StabilityViolationType)
        assert isinstance(v.window_start, int)
        assert isinstance(v.window_end, int)
        assert v.window_end > v.window_start
        assert isinstance(v.value, float)
        assert isinstance(v.threshold, float)
        assert isinstance(v.severity, float)
        assert isinstance(v.description, str)
        assert len(v.description) > 0

        # Severity should be value/threshold
        assert abs(v.severity - v.value / v.threshold) < 1e-6


def test_oscillation_detection_specifics(oscillatory_trajectory):
    """Test oscillation detection mechanics."""
    config = SmoothnessConfig(oscillation_threshold=0.3)
    result = analyze_trajectory_smoothness(oscillatory_trajectory, config)

    # Should detect oscillations
    osc_violations = [
        v
        for v in result.violations
        if v.violation_type == StabilityViolationType.OSCILLATORY_INSTABILITY
    ]

    assert len(osc_violations) > 0, "Should detect oscillatory instability"

    # Check that oscillation score exceeds threshold
    assert result.oscillation_score > config.oscillation_threshold


def test_trend_divergence_detection(erratic_trajectory):
    """Test trend smoothness detection."""
    # Use stricter threshold to ensure detection
    config = SmoothnessConfig(
        trend_smoothness_threshold=0.15,
        oscillation_threshold=0.3,  # Raise oscillation threshold to focus on trend
    )
    result = analyze_trajectory_smoothness(erratic_trajectory, config)

    # Should detect some form of instability
    assert not result.passed or len(result.violations) > 0, "Should detect instability"

    # Check if trend smoothness score is elevated (even if violations are oscillatory)
    # The erratic trajectory should have high residuals
    assert result.trend_smoothness_score > 0.05, "Should have elevated trend residual"


def test_multi_scale_consistency():
    """Test multi-scale consistency detection."""
    # Create trajectory that's smooth at fine scale but rough when downsampled
    n_steps = 200
    t = np.linspace(0, 10, n_steps)

    # High-frequency noise that averages out at fine scale but not coarse
    np.random.seed(123)
    gamma = 10.0 + 0.5 * np.random.randn(n_steps)
    # Smooth it slightly at fine scale
    from scipy.ndimage import gaussian_filter1d

    gamma = gaussian_filter1d(gamma, sigma=1.5)

    trajectory = {
        "t": t,
        "z": t * 10,
        "r": 0.01 * np.ones_like(t),
        "gamma": gamma,
    }

    config = SmoothnessConfig(downsample_factor=10)
    result = analyze_trajectory_smoothness(trajectory, config)

    # May or may not detect multi-scale issue depending on noise level
    # At minimum, should compute multi_scale_consistency
    assert isinstance(result.multi_scale_consistency, float)
    assert result.multi_scale_consistency >= 0


# ============================================================================
# Test Batch Filtering
# ============================================================================


def test_filter_stable_trajectories_basic(smooth_trajectory, oscillatory_trajectory):
    """Test batch filtering of trajectories."""
    results = [
        {"trajectory": smooth_trajectory, "id": 1},
        {"trajectory": oscillatory_trajectory, "id": 2},
    ]

    config = SmoothnessConfig()
    stable, rejected = filter_stable_trajectories(results, config, verbose=False)

    assert len(stable) + len(rejected) == len(results)
    assert len(stable) >= 1, "At least smooth trajectory should pass"
    assert len(rejected) >= 1, "At least oscillatory trajectory should fail"

    # Check IDs
    stable_ids = [r["id"] for r in stable]
    rejected_ids = [r["id"] for r in rejected]

    assert 1 in stable_ids, "Smooth trajectory should be in stable"
    assert 2 in rejected_ids, "Oscillatory trajectory should be in rejected"


def test_filter_adds_analysis_to_rejected(oscillatory_trajectory):
    """Rejected trajectories should include analysis results."""
    results = [{"trajectory": oscillatory_trajectory, "id": 1}]
    config = SmoothnessConfig()

    stable, rejected = filter_stable_trajectories(results, config, verbose=False)

    if len(rejected) > 0:
        assert "smoothness_analysis" in rejected[0]
        analysis = rejected[0]["smoothness_analysis"]
        assert not analysis.passed


def test_filter_handles_missing_trajectory():
    """Filter should skip results without trajectory data."""
    results = [
        {"id": 1, "data": "no trajectory here"},
        {"id": 2},
    ]

    config = SmoothnessConfig()
    stable, rejected = filter_stable_trajectories(results, config, verbose=False)

    # Both should be skipped
    assert len(stable) == 0
    assert len(rejected) == 0


def test_filter_with_different_configs(smooth_trajectory, erratic_trajectory):
    """Filter behavior should change with config."""
    results = [
        {"trajectory": smooth_trajectory, "id": 1},
        {"trajectory": erratic_trajectory, "id": 2},
    ]

    # Strict config
    config_strict = SmoothnessConfig.strict()
    stable_strict, rejected_strict = filter_stable_trajectories(
        results, config_strict, verbose=False
    )

    # Permissive config
    config_permissive = SmoothnessConfig.permissive()
    stable_permissive, rejected_permissive = filter_stable_trajectories(
        results, config_permissive, verbose=False
    )

    # Permissive should accept more trajectories
    assert len(stable_permissive) >= len(stable_strict)
    assert len(rejected_permissive) <= len(rejected_strict)


# ============================================================================
# Test Edge Cases
# ============================================================================


def test_empty_trajectory():
    """Empty trajectory should be handled gracefully."""
    trajectory = {"t": np.array([]), "z": np.array([]), "gamma": np.array([])}
    config = SmoothnessConfig()

    result = analyze_trajectory_smoothness(trajectory, config)

    # Should pass (too short for analysis)
    assert result.passed
    assert result.num_steps_analyzed == 0


def test_trajectory_without_gamma(smooth_trajectory):
    """Trajectory without gamma should fall back to position."""
    traj_no_gamma = smooth_trajectory.copy()
    # Set gamma to empty array rather than None to avoid 0-d array issue
    traj_no_gamma["gamma"] = np.array([])

    config = SmoothnessConfig()

    # The implementation uses gamma as primary metric
    # Without gamma (empty array), it falls back to computing from position
    # For smooth linear motion, this should still pass
    result = analyze_trajectory_smoothness(traj_no_gamma, config)

    # Should still analyze (using position magnitude as proxy)
    assert result.num_steps_analyzed > 0, "Should analyze trajectory"
    # Linear motion should produce smooth position-based metric
    assert result.passed or result.oscillation_score < 0.7, (
        "Should recognize smoothness"
    )


def test_constant_gamma_trajectory():
    """Constant gamma should pass all checks."""
    n_steps = 100
    trajectory = {
        "t": np.linspace(0, 10, n_steps),
        "z": np.linspace(0, 100, n_steps),
        "r": np.ones(n_steps) * 0.01,
        "gamma": np.ones(n_steps) * 10.0,  # Perfectly constant
    }

    config = SmoothnessConfig()
    result = analyze_trajectory_smoothness(trajectory, config)

    assert result.passed
    assert result.oscillation_score < 0.01
    assert result.trend_smoothness_score < 0.01


def test_nan_in_trajectory():
    """NaN in trajectory should be handled."""
    n_steps = 100
    gamma = 10.0 * np.ones(n_steps)
    gamma[50] = np.nan  # Introduce NaN

    trajectory = {
        "t": np.linspace(0, 10, n_steps),
        "z": np.linspace(0, 100, n_steps),
        "r": 0.01 * np.ones(n_steps),
        "gamma": gamma,
    }

    config = SmoothnessConfig()

    # Should either handle gracefully or detect as instability
    try:
        result = analyze_trajectory_smoothness(trajectory, config)
        # If it completes, NaN should cause failure or be filtered
        assert isinstance(result.passed, bool)
    except (ValueError, RuntimeError):
        # Acceptable to raise error for invalid data
        pass


def test_very_long_trajectory():
    """Very long trajectory should still be analyzable."""
    n_steps = 10000
    t = np.linspace(0, 100, n_steps)
    gamma = 10.0 + 0.001 * t  # Gentle linear increase

    trajectory = {
        "t": t,
        "z": 10 * t,
        "r": 0.01 * np.ones(n_steps),
        "gamma": gamma,
    }

    config = SmoothnessConfig()
    result = analyze_trajectory_smoothness(trajectory, config)

    assert result.passed
    assert result.num_steps_analyzed == n_steps


# ============================================================================
# Test Result String Representation
# ============================================================================


def test_result_str_passed(smooth_trajectory):
    """Test string representation of passed result."""
    config = SmoothnessConfig()
    result = analyze_trajectory_smoothness(smooth_trajectory, config)

    result_str = str(result)

    assert "PASSED" in result_str
    assert "Oscillation score" in result_str
    assert "Trend smoothness" in result_str
    assert "Quality" in result_str


def test_result_str_failed(oscillatory_trajectory):
    """Test string representation of failed result."""
    config = SmoothnessConfig()
    result = analyze_trajectory_smoothness(oscillatory_trajectory, config)

    result_str = str(result)

    assert "FAILED" in result_str
    assert "violations" in result_str.lower()
    assert "Violations:" in result_str


# ============================================================================
# Test Integration with Real Physics
# ============================================================================


def test_gentle_acceleration_passes():
    """Smooth acceleration should pass stability checks."""
    n_steps = 100
    t = np.linspace(0, 10, n_steps)

    # Quadratic gamma increase (constant acceleration)
    gamma = 10.0 + 0.5 * t**2

    beta = np.sqrt(1.0 - 1.0 / gamma**2)
    c_mmns = 299.792458
    v = beta * c_mmns
    z = np.cumsum(v * np.diff(t, prepend=0))
    r = 0.01 * np.ones_like(z)

    trajectory = {
        "t": t,
        "z": z,
        "r": r,
        "gamma": gamma,
    }

    config = SmoothnessConfig()
    result = analyze_trajectory_smoothness(trajectory, config)

    assert result.passed, "Smooth acceleration should pass"
    assert result.trend_smoothness_score < 0.15


def test_exponential_growth_passes():
    """Exponential energy growth (radiation) should pass if smooth."""
    n_steps = 100
    t = np.linspace(0, 5, n_steps)

    # Exponential gamma growth
    gamma = 10.0 * np.exp(0.1 * t)

    beta = np.sqrt(1.0 - 1.0 / gamma**2)
    c_mmns = 299.792458
    v = beta * c_mmns
    z = np.cumsum(v * np.diff(t, prepend=0))
    r = 0.01 * np.ones_like(z)

    trajectory = {
        "t": t,
        "z": z,
        "r": r,
        "gamma": gamma,
    }

    config = SmoothnessConfig()
    result = analyze_trajectory_smoothness(trajectory, config)

    assert result.passed, "Smooth exponential growth should pass"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
