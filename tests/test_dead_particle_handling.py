"""Test dead particle handling implementation.

This module provides basic tests to verify that the dead particle exclusion
strategy works correctly, including marking particles as dead, propagating
status, and computing metrics with dead particles excluded.
"""

import numpy as np

from core.particle_status import (
    all_particles_dead,
    compute_alive_particle_average,
    format_failure_summary,
    get_alive_particle_indices,
    get_alive_particle_values,
    get_particle_failure_summary,
    mark_particle_dead,
    propagate_dead_particle_status,
    validate_particle_status_consistency,
)
from core.types import ParticleState


def create_test_state(num_particles: int = 5) -> ParticleState:
    """Create a test particle state with specified number of particles."""
    return {
        "x": np.zeros(num_particles),
        "y": np.zeros(num_particles),
        "z": np.linspace(0, 10, num_particles),
        "t": np.zeros(num_particles),
        "bx": np.zeros(num_particles),
        "by": np.zeros(num_particles),
        "bz": np.ones(num_particles) * 0.9,
        "gamma": np.ones(num_particles) * 100.0,
        "Px": np.zeros(num_particles),
        "Py": np.zeros(num_particles),
        "Pz": np.ones(num_particles) * 50.0,
        "Pt": np.ones(num_particles) * 51.0,
        "stripped_ions": np.ones(num_particles) * 5.0,
        "m": np.ones(num_particles),
        "bdotx": np.zeros(num_particles),
        "bdoty": np.zeros(num_particles),
        "bdotz": np.zeros(num_particles),
        "origin_x": np.zeros(num_particles),
        "origin_y": np.zeros(num_particles),
        "origin_z": np.zeros(num_particles),
        "beta_avg_x": np.zeros(num_particles),
        "beta_avg_y": np.zeros(num_particles),
        "beta_avg_z": np.ones(num_particles) * 0.9,
        "beta_samples": np.ones(num_particles),
    }


class TestMarkParticleDead:
    """Test marking particles as dead."""

    def test_mark_single_particle(self):
        """Test marking a single particle as dead."""
        state = create_test_state(5)
        original_charge = state["stripped_ions"][2]

        mark_particle_dead(state, 2, 100, "gamma_blowup", gamma_value=1e9)

        # Check dead flag set
        assert "_dead_particles" in state
        assert state["_dead_particles"][2]
        assert np.sum(state["_dead_particles"]) == 1

        # Check charge zeroed
        assert state["stripped_ions"][2] == 0.0
        assert state["stripped_ions"][0] == original_charge  # Others unchanged

        # Check failure info recorded
        assert "_particle_failure_info" in state
        assert 2 in state["_particle_failure_info"]
        assert state["_particle_failure_info"][2]["step"] == 100
        assert state["_particle_failure_info"][2]["reason"] == "gamma_blowup"
        assert state["_particle_failure_info"][2]["gamma_value"] == 1e9

    def test_mark_multiple_particles(self):
        """Test marking multiple particles as dead."""
        state = create_test_state(5)

        mark_particle_dead(state, 1, 50, "gamma_blowup", gamma_value=5e8)
        mark_particle_dead(state, 3, 75, "energy_jump")

        assert np.sum(state["_dead_particles"]) == 2
        assert state["_dead_particles"][1]
        assert state["_dead_particles"][3]
        assert state["stripped_ions"][1] == 0.0
        assert state["stripped_ions"][3] == 0.0

        # Check both recorded in failure info
        assert len(state["_particle_failure_info"]) == 2


class TestPropagateDeadStatus:
    """Test propagating dead particle status."""

    def test_propagate_to_next_step(self):
        """Test propagating dead status from one step to next."""
        prev_state = create_test_state(5)
        mark_particle_dead(prev_state, 2, 100, "gamma_blowup")

        current_state = create_test_state(5)
        propagate_dead_particle_status(current_state, prev_state)

        # Check status copied
        assert "_dead_particles" in current_state
        assert current_state["_dead_particles"][2]
        assert np.sum(current_state["_dead_particles"]) == 1

        # Check charge zeroed
        assert current_state["stripped_ions"][2] == 0.0

        # Check failure info copied
        assert 2 in current_state["_particle_failure_info"]
        assert current_state["_particle_failure_info"][2]["step"] == 100

    def test_propagate_no_failures(self):
        """Test propagating when no particles are dead."""
        prev_state = create_test_state(5)
        current_state = create_test_state(5)

        propagate_dead_particle_status(current_state, prev_state)

        # Should not add metadata if no failures
        assert "_dead_particles" not in current_state


class TestAliveParticleQueries:
    """Test queries for alive particles."""

    def test_get_alive_indices_no_failures(self):
        """Test getting alive indices when no particles are dead."""
        state = create_test_state(5)
        alive = get_alive_particle_indices(state)

        assert len(alive) == 5
        assert np.array_equal(alive, np.arange(5))

    def test_get_alive_indices_with_failures(self):
        """Test getting alive indices with some dead particles."""
        state = create_test_state(5)
        mark_particle_dead(state, 1, 100, "gamma_blowup")
        mark_particle_dead(state, 3, 100, "gamma_blowup")

        alive = get_alive_particle_indices(state)

        assert len(alive) == 3
        assert np.array_equal(alive, np.array([0, 2, 4]))

    def test_all_particles_dead_false(self):
        """Test all_particles_dead returns False when some alive."""
        state = create_test_state(5)
        mark_particle_dead(state, 1, 100, "gamma_blowup")

        assert not all_particles_dead(state)

    def test_all_particles_dead_true(self):
        """Test all_particles_dead returns True when all dead."""
        state = create_test_state(3)
        mark_particle_dead(state, 0, 100, "gamma_blowup")
        mark_particle_dead(state, 1, 100, "gamma_blowup")
        mark_particle_dead(state, 2, 100, "gamma_blowup")

        assert all_particles_dead(state)

    def test_get_alive_values(self):
        """Test extracting values from alive particles only."""
        state = create_test_state(5)
        state["gamma"] = np.array([100, 200, 300, 400, 500], dtype=float)

        mark_particle_dead(state, 1, 100, "gamma_blowup")
        mark_particle_dead(state, 3, 100, "gamma_blowup")

        alive_gammas = get_alive_particle_values(state, "gamma")

        assert alive_gammas is not None
        assert len(alive_gammas) == 3
        assert np.array_equal(alive_gammas, np.array([100, 300, 500]))


class TestComputeAliveAverage:
    """Test computing averages excluding dead particles."""

    def test_average_no_failures(self):
        """Test average when no particles are dead."""
        state = create_test_state(5)
        state["gamma"] = np.array([100, 200, 300, 400, 500], dtype=float)

        avg = compute_alive_particle_average(state, "gamma")

        assert avg == 300.0

    def test_average_with_failures(self):
        """Test average excluding dead particles."""
        state = create_test_state(5)
        state["gamma"] = np.array([100, 200, 300, 400, 500], dtype=float)

        mark_particle_dead(state, 0, 100, "gamma_blowup")  # Remove 100
        mark_particle_dead(state, 4, 100, "gamma_blowup")  # Remove 500

        avg = compute_alive_particle_average(state, "gamma")

        # Average of [200, 300, 400] = 300
        assert avg == 300.0

    def test_average_all_dead(self):
        """Test average returns None when all particles dead."""
        state = create_test_state(3)
        mark_particle_dead(state, 0, 100, "gamma_blowup")
        mark_particle_dead(state, 1, 100, "gamma_blowup")
        mark_particle_dead(state, 2, 100, "gamma_blowup")

        avg = compute_alive_particle_average(state, "gamma")

        assert avg is None


class TestFailureSummary:
    """Test failure summary formatting."""

    def test_get_failure_summary(self):
        """Test extracting failure summary from trajectory."""
        trajectory = [create_test_state(5) for _ in range(3)]

        # Mark failures in final state
        mark_particle_dead(trajectory[-1], 1, 100, "gamma_blowup", gamma_value=1e9)
        mark_particle_dead(trajectory[-1], 3, 150, "energy_jump")

        summary = get_particle_failure_summary(trajectory)

        assert len(summary) == 2
        assert 1 in summary
        assert 3 in summary
        assert summary[1]["step"] == 100
        assert summary[3]["step"] == 150

    def test_format_failure_summary(self):
        """Test formatting failure summary as string."""
        state = create_test_state(5)
        mark_particle_dead(state, 1, 100, "gamma_blowup", gamma_value=1e9, iteration=5)
        mark_particle_dead(state, 3, 150, "energy_jump")

        failure_info = state["_particle_failure_info"]
        summary_str = format_failure_summary(failure_info)

        assert "Particle failures: 2 total" in summary_str
        assert "Particle 1" in summary_str
        assert "gamma_blowup" in summary_str
        assert "step 100" in summary_str
        assert "Particle 3" in summary_str
        assert "energy_jump" in summary_str

    def test_format_no_failures(self):
        """Test formatting when no failures."""
        summary_str = format_failure_summary({})
        assert summary_str == "No particle failures"


class TestConsistencyValidation:
    """Test validation of particle status consistency."""

    def test_consistent_state(self):
        """Test validation passes for consistent state."""
        state = create_test_state(5)
        mark_particle_dead(state, 2, 100, "gamma_blowup")

        issues = validate_particle_status_consistency(state)

        assert len(issues) == 0

    def test_inconsistent_charge(self):
        """Test detection of dead particle with non-zero charge."""
        state = create_test_state(5)
        mark_particle_dead(state, 2, 100, "gamma_blowup")

        # Manually break consistency by restoring charge
        state["stripped_ions"][2] = 5.0

        issues = validate_particle_status_consistency(state)

        assert len(issues) > 0
        assert any("non-zero charge" in issue for issue in issues)

    def test_no_metadata(self):
        """Test validation passes when no dead particle metadata."""
        state = create_test_state(5)

        issues = validate_particle_status_consistency(state)

        assert len(issues) == 0


if __name__ == "__main__":
    # Run basic smoke tests
    print("Running basic dead particle handling tests...")

    # Test 1: Mark particle dead
    print("\nTest 1: Mark particle as dead")
    state = create_test_state(5)
    mark_particle_dead(state, 2, 100, "gamma_blowup", gamma_value=1e9)
    print(f"  Dead particles: {np.where(state['_dead_particles'])[0]}")
    print(f"  Charges: {state['stripped_ions']}")
    assert state["_dead_particles"][2]
    assert state["stripped_ions"][2] == 0.0
    print("  ✓ Passed")

    # Test 2: Propagate status
    print("\nTest 2: Propagate dead status")
    next_state = create_test_state(5)
    propagate_dead_particle_status(next_state, state)
    assert next_state["_dead_particles"][2]
    assert next_state["stripped_ions"][2] == 0.0
    print("  ✓ Passed")

    # Test 3: Alive particle average
    print("\nTest 3: Compute alive particle average")
    state = create_test_state(5)
    state["gamma"] = np.array([100, 200, 300, 400, 500], dtype=float)
    mark_particle_dead(state, 1, 100, "gamma_blowup")
    mark_particle_dead(state, 3, 100, "gamma_blowup")
    avg = compute_alive_particle_average(state, "gamma")
    expected = (100 + 300 + 500) / 3.0
    print(f"  Average gamma (alive only): {avg}")
    print(f"  Expected: {expected}")
    assert abs(avg - expected) < 1e-6
    print("  ✓ Passed")

    # Test 4: All particles dead
    print("\nTest 4: All particles dead detection")
    state = create_test_state(3)
    mark_particle_dead(state, 0, 100, "gamma_blowup")
    mark_particle_dead(state, 1, 100, "gamma_blowup")
    mark_particle_dead(state, 2, 100, "gamma_blowup")
    assert all_particles_dead(state)
    print("  ✓ Passed")

    # Test 5: Failure summary
    print("\nTest 5: Failure summary formatting")
    state = create_test_state(5)
    mark_particle_dead(state, 1, 100, "gamma_blowup", gamma_value=1e9, iteration=5)
    mark_particle_dead(state, 3, 150, "energy_jump")
    summary = format_failure_summary(state["_particle_failure_info"])
    print(f"  {summary}")
    assert "Particle failures: 2 total" in summary
    print("  ✓ Passed")

    print("\n✓ All basic tests passed!")
    print("\nTo run full pytest suite: pytest test_dead_particle_handling.py -v")
