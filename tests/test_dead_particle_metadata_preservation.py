"""Test that dead particle metadata is preserved when initializing result states.

This test verifies that when _initialize_result_state is called on a state
that has dead particles marked, the resulting state preserves the _dead_particles
and _particle_failure_info metadata. This prevents redundant logging when
particles are re-processed in retry loops.
"""

import numpy as np
import pytest

from core.equations import _initialize_result_state
from core.particle_status import mark_particle_dead


def test_dead_particle_metadata_preserved_in_result_state():
    """Test that _initialize_result_state preserves dead particle metadata."""
    # Create a basic particle state
    num_particles = 5
    state = {
        "x": np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        "y": np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        "z": np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        "t": np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        "Px": np.zeros(num_particles),
        "Py": np.zeros(num_particles),
        "Pz": np.zeros(num_particles),
        "Pt": np.ones(num_particles) * 1e4,
        "gamma": np.ones(num_particles),
        "bx": np.zeros(num_particles),
        "by": np.zeros(num_particles),
        "bz": np.zeros(num_particles),
        "bdotx": np.zeros(num_particles),
        "bdoty": np.zeros(num_particles),
        "bdotz": np.zeros(num_particles),
        "q": np.ones(num_particles),
        "char_time": np.zeros(num_particles),
        "m": np.ones(num_particles),
        "origin_x": np.zeros(num_particles),
        "origin_y": np.zeros(num_particles),
        "origin_z": np.zeros(num_particles),
        "beta_avg_x": np.zeros(num_particles),
        "beta_avg_y": np.zeros(num_particles),
        "beta_avg_z": np.zeros(num_particles),
        "beta_samples": np.zeros(num_particles, dtype=int),
    }

    # Mark particles 1 and 3 as dead
    mark_particle_dead(state, 1, step=10, reason="gamma_blowup_hard", gamma_value=1e25)
    mark_particle_dead(state, 3, step=15, reason="aperture_loss", gamma_value=2.5)

    # Verify the state has dead particle metadata
    assert "_dead_particles" in state
    assert state["_dead_particles"][1] == True
    assert state["_dead_particles"][3] == True
    assert state["_dead_particles"][0] == False
    assert "_particle_failure_info" in state
    assert 1 in state["_particle_failure_info"]
    assert 3 in state["_particle_failure_info"]

    # Initialize result state
    result = _initialize_result_state(state)

    # Verify dead particle metadata is preserved
    assert "_dead_particles" in result, "Dead particles array should be copied"
    assert "_particle_failure_info" in result, "Failure info should be copied"

    # Check that dead flags are preserved
    assert result["_dead_particles"][1] == True, "Particle 1 should still be dead"
    assert result["_dead_particles"][3] == True, "Particle 3 should still be dead"
    assert result["_dead_particles"][0] == False, "Particle 0 should still be alive"
    assert result["_dead_particles"][2] == False, "Particle 2 should still be alive"
    assert result["_dead_particles"][4] == False, "Particle 4 should still be alive"

    # Check that failure info is preserved
    assert 1 in result["_particle_failure_info"], (
        "Particle 1 failure info should be copied"
    )
    assert 3 in result["_particle_failure_info"], (
        "Particle 3 failure info should be copied"
    )
    assert result["_particle_failure_info"][1]["reason"] == "gamma_blowup_hard"
    assert result["_particle_failure_info"][3]["reason"] == "aperture_loss"
    assert result["_particle_failure_info"][1]["step"] == 10
    assert result["_particle_failure_info"][3]["step"] == 15

    # Verify that arrays are independent copies (not shared references)
    result["_dead_particles"][2] = True
    assert state["_dead_particles"][2] == False, "Original state should not be affected"

    # Verify that failure info is independent
    result["_particle_failure_info"][1]["step"] = 999
    assert state["_particle_failure_info"][1]["step"] == 10, (
        "Original failure info should not be affected"
    )


def test_initialize_result_without_dead_particles():
    """Test that _initialize_result_state works when no particles are dead."""
    num_particles = 3
    state = {
        "x": np.array([0.0, 1.0, 2.0]),
        "y": np.zeros(num_particles),
        "z": np.zeros(num_particles),
        "t": np.zeros(num_particles),
        "Px": np.zeros(num_particles),
        "Py": np.zeros(num_particles),
        "Pz": np.zeros(num_particles),
        "Pt": np.ones(num_particles) * 1e4,
        "gamma": np.ones(num_particles),
        "bx": np.zeros(num_particles),
        "by": np.zeros(num_particles),
        "bz": np.zeros(num_particles),
        "bdotx": np.zeros(num_particles),
        "bdoty": np.zeros(num_particles),
        "bdotz": np.zeros(num_particles),
        "q": np.ones(num_particles),
        "char_time": np.zeros(num_particles),
        "m": np.ones(num_particles),
        "origin_x": np.zeros(num_particles),
        "origin_y": np.zeros(num_particles),
        "origin_z": np.zeros(num_particles),
        "beta_avg_x": np.zeros(num_particles),
        "beta_avg_y": np.zeros(num_particles),
        "beta_avg_z": np.zeros(num_particles),
        "beta_samples": np.zeros(num_particles, dtype=int),
    }

    # Don't add any dead particle metadata - simulate a fresh state
    assert "_dead_particles" not in state
    assert "_particle_failure_info" not in state

    # Initialize result state
    result = _initialize_result_state(state)

    # Verify that result also doesn't have dead particle metadata (it's optional)
    # This ensures we don't create empty metadata when none exists
    assert "_dead_particles" not in result
    assert "_particle_failure_info" not in result

    # All other fields should be copied
    assert "x" in result
    assert "gamma" in result
    assert np.array_equal(result["x"], state["x"])
