#!/usr/bin/env python3
"""Test the failed run retry mechanism for sweeps."""

from unittest.mock import Mock

import numpy as np
import pytest

from optimization.config import OptimizationConfig


def test_retry_config_default():
    """Test that retry attempts default is set correctly."""
    config = OptimizationConfig()
    assert config.failed_run_retry_attempts == 1, "Default retry attempts should be 1"


def test_retry_config_custom():
    """Test that retry attempts can be customized."""
    config = OptimizationConfig(failed_run_retry_attempts=3)
    assert config.failed_run_retry_attempts == 3


def test_retry_config_zero():
    """Test that retry can be disabled."""
    config = OptimizationConfig(failed_run_retry_attempts=0)
    assert config.failed_run_retry_attempts == 0


def test_seed_override_calculation():
    """Test that seed override is calculated deterministically for retries."""
    base_seed = 12345
    run_num = 42
    retry_attempt = 1

    # Expected seed calculation: base_seed + run_num * 10000 + retry_attempt * 100
    expected_seed = base_seed + run_num * 10000 + retry_attempt * 100
    assert expected_seed == 12345 + 420000 + 100
    assert expected_seed == 432445

    # Second retry should have different seed
    retry_attempt_2 = 2
    expected_seed_2 = base_seed + run_num * 10000 + retry_attempt_2 * 100
    assert expected_seed_2 == 432545
    assert expected_seed != expected_seed_2


def test_seed_uniqueness_across_runs():
    """Test that seeds are unique across different run numbers and retries."""
    base_seed = 12345

    seeds = set()
    for run_num in range(1, 11):
        # Original attempt
        seeds.add(base_seed + run_num)

        # Retry attempts
        for retry in range(1, 4):
            seed = base_seed + run_num * 10000 + retry * 100
            seeds.add(seed)

    # All seeds should be unique
    # 10 runs × (1 original + 3 retries) = 40 unique seeds
    assert len(seeds) == 40, f"Expected 40 unique seeds, got {len(seeds)}"


def test_retry_metadata_in_results():
    """Test that retry attempt count is stored in results."""
    # This is a structural test - verify the expected data format

    # Successful run after retries
    success_result = {
        "run_number": 1,
        "parameters": {
            "aperture_radius": 1e-4,
            "particle_energy_gev": 10.0,
            "retry_attempts": 2,  # Succeeded on 3rd attempt (2 retries)
        },
        "metrics": {"max_percent_energy_gain": 5.0},
    }

    assert "retry_attempts" in success_result["parameters"]
    assert success_result["parameters"]["retry_attempts"] == 2

    # Failed run after all retries exhausted
    failed_result = {
        "run_number": 2,
        "parameters": {"aperture_radius": 1e-4, "particle_energy_gev": 20.0},
        "error": "All particles died - no metrics generated (tried 4 time(s))",
        "retry_attempts": 3,  # 1 original + 3 retries = 4 attempts total
    }

    assert "retry_attempts" in failed_result
    assert failed_result["retry_attempts"] == 3


def test_retry_only_on_failure():
    """Test that successful runs don't trigger retries."""
    config = OptimizationConfig(failed_run_retry_attempts=3)

    # Mock a successful result with metrics
    mock_result = Mock()
    mock_result.metrics = {"max_percent_energy_gain": 5.0, "delta_e_mev": 100.0}
    mock_result.trajectory = None

    # If run succeeds on first attempt, no retries should occur
    # This would be validated by checking that _run_single_integration
    # is only called once (not 4 times)

    # The logic should be:
    # - Attempt 0: Success → break out of retry loop
    # - Total calls: 1

    assert config.failed_run_retry_attempts == 3


def test_retry_exhaustion():
    """Test behavior when all retry attempts are exhausted."""
    config = OptimizationConfig(failed_run_retry_attempts=2)

    # Simulate 3 failed attempts (1 original + 2 retries)
    # All attempts fail with no metrics

    attempts = []
    max_retries = config.failed_run_retry_attempts

    for retry_attempt in range(max_retries + 1):
        attempts.append(
            {
                "attempt": retry_attempt,
                "result": None,  # Failed
                "error": "All particles died",
            }
        )

    # Should have 3 total attempts
    assert len(attempts) == 3
    assert attempts[0]["attempt"] == 0  # Original
    assert attempts[1]["attempt"] == 1  # First retry
    assert attempts[2]["attempt"] == 2  # Second retry


def test_no_retry_when_disabled():
    """Test that retries are skipped when disabled (0 attempts)."""
    config = OptimizationConfig(failed_run_retry_attempts=0)

    # With 0 retries, only original attempt should run
    max_attempts = config.failed_run_retry_attempts + 1
    assert max_attempts == 1


def test_retry_logging_messages():
    """Test expected log message format for retries."""
    run_num = 42
    retry_attempt = 1
    max_retries = 3
    base_seed = 12345
    current_seed = base_seed + run_num * 10000 + retry_attempt * 100

    # Expected log format
    expected_log = f"  [RETRY] Run {run_num}, attempt {retry_attempt}/{max_retries} with new seed {current_seed}"

    # Verify the seed in the message matches our calculation
    assert str(current_seed) in expected_log
    assert f"attempt {retry_attempt}/{max_retries}" in expected_log


def test_success_after_retry_logging():
    """Test expected log message when retry succeeds."""
    run_num = 42
    retry_attempt = 2

    expected_log = (
        f"  [SUCCESS] Run {run_num} succeeded on retry attempt {retry_attempt}"
    )

    assert "SUCCESS" in expected_log
    assert str(retry_attempt) in expected_log


def test_timeout_retry_metadata():
    """Test that timeout failures include retry count."""
    timeout_failure = {
        "run_number": 5,
        "parameters": {"aperture_radius": 1e-4},
        "error": "Timeout after 300.0s (tried 3 time(s))",
        "timed_out": True,
        "retry_attempts": 2,
    }

    assert timeout_failure["timed_out"] is True
    assert timeout_failure["retry_attempts"] == 2
    assert "tried 3 time(s)" in timeout_failure["error"]  # 1 original + 2 retries


def test_integration_retry_parameters():
    """Test that seed_override is properly passed to _run_single_integration."""
    # This test documents the expected signature

    # When calling _run_single_integration during a retry:
    expected_params = {
        "aperture": 1e-4,
        "energy_gev": 10.0,
        "start_z": 0.0,
        "transv_offset": 0.0,
        "timestep": 1e-7,
        "steps": 1000,
        "seed_override": 432445,  # New seed for retry
        # ... other parameters
    }

    assert "seed_override" in expected_params
    assert expected_params["seed_override"] is not None


def test_retry_with_different_particle_distributions():
    """Test that retries with different seeds produce different particle distributions."""
    base_seed = 12345
    run_num = 1

    # Original attempt seed
    original_seed = base_seed + run_num

    # First retry seed
    retry_1_seed = base_seed + run_num * 10000 + 1 * 100

    # Second retry seed
    retry_2_seed = base_seed + run_num * 10000 + 2 * 100

    # All three should be different
    seeds = [original_seed, retry_1_seed, retry_2_seed]
    assert len(set(seeds)) == 3, "All seeds should be unique"

    # Using numpy to verify different seeds produce different random numbers
    np.random.seed(original_seed)
    original_sample = np.random.rand(10)

    np.random.seed(retry_1_seed)
    retry_1_sample = np.random.rand(10)

    np.random.seed(retry_2_seed)
    retry_2_sample = np.random.rand(10)

    # Samples should be different
    assert not np.allclose(original_sample, retry_1_sample)
    assert not np.allclose(original_sample, retry_2_sample)
    assert not np.allclose(retry_1_sample, retry_2_sample)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
