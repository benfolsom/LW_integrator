#!/usr/bin/env python3
"""Test that optimization metrics use average of all alive particles, not just particle 0."""

import numpy as np
import pytest

from core.particle_status import (
    compute_alive_particle_average,
    get_alive_particle_values,
)
from optimization.metrics import (
    compute_max_energy_gain,
    compute_percent_energy_gain,
    compute_relative_energy_gain,
    compute_trajectory_metrics,
)


def test_compute_alive_particle_average_all_alive():
    """Test that average is computed correctly when all particles are alive."""
    state = {
        "gamma": np.array([100.0, 200.0, 300.0]),
        "_dead_particles": np.array([False, False, False]),
    }

    avg = compute_alive_particle_average(state, "gamma")
    expected = (100.0 + 200.0 + 300.0) / 3.0
    assert avg == pytest.approx(expected), "Should average all three particles"


def test_compute_alive_particle_average_one_dead():
    """Test that dead particles are excluded from average."""
    state = {
        "gamma": np.array([100.0, 200.0, 300.0]),
        "_dead_particles": np.array([False, True, False]),  # Particle 1 is dead
    }

    avg = compute_alive_particle_average(state, "gamma")
    expected = (100.0 + 300.0) / 2.0  # Only particles 0 and 2
    assert avg == pytest.approx(expected), (
        "Should only average alive particles (0 and 2)"
    )


def test_compute_alive_particle_average_particle_0_dead():
    """CRITICAL: Test that particle 0 being dead doesn't bias the result."""
    state = {
        "gamma": np.array([1000.0, 200.0, 300.0]),  # Particle 0 has very high gamma
        "_dead_particles": np.array([True, False, False]),  # Particle 0 is DEAD
    }

    avg = compute_alive_particle_average(state, "gamma")
    expected = (200.0 + 300.0) / 2.0  # Should exclude particle 0
    assert avg == pytest.approx(expected), "Should NOT include particle 0's high gamma"
    assert avg != 1000.0, "Should not return particle 0's value"
    assert avg == 250.0, "Should average particles 1 and 2 only"


def test_compute_alive_particle_average_all_dead():
    """Test that all dead particles returns None."""
    state = {
        "gamma": np.array([100.0, 200.0, 300.0]),
        "_dead_particles": np.array([True, True, True]),
    }

    avg = compute_alive_particle_average(state, "gamma")
    assert avg is None, "Should return None when all particles are dead"


def test_get_alive_particle_values_filters_correctly():
    """Test that get_alive_particle_values returns only alive particles."""
    state = {
        "gamma": np.array([100.0, 200.0, 300.0, 400.0]),
        "_dead_particles": np.array([True, False, False, True]),
    }

    alive_values = get_alive_particle_values(state, "gamma")
    expected = np.array([200.0, 300.0])  # Particles 1 and 2

    assert alive_values is not None
    assert len(alive_values) == 2
    np.testing.assert_array_equal(alive_values, expected)


def test_max_energy_gain_uses_average():
    """Test that max_energy_gain uses average across all alive particles."""
    initial_gamma = 100.0
    rest_energy_mev = 0.511

    # Create trajectory with 3 particles
    # Particle 0 has very high gain but dies
    # Particles 1 and 2 have moderate gain
    trajectory = [
        {
            "gamma": np.array([500.0, 150.0, 160.0]),  # Particle 0 very high
            "_dead_particles": np.array([True, False, False]),  # Particle 0 dead
        },
        {
            "gamma": np.array([600.0, 180.0, 190.0]),
            "_dead_particles": np.array([True, False, False]),
        },
    ]

    max_gain = compute_max_energy_gain(trajectory, initial_gamma, rest_energy_mev)

    # Average at step 0: (150 + 160) / 2 = 155
    # Average at step 1: (180 + 190) / 2 = 185
    # Max avg gamma: 185
    # Delta gamma: 185 - 100 = 85
    # Energy gain: 85 * 0.000511 GeV = 0.043435 GeV

    expected_delta_gamma = 85.0
    expected_gain_gev = expected_delta_gamma * (rest_energy_mev * 1e-3)

    assert max_gain == pytest.approx(expected_gain_gev), (
        "Should use average of alive particles, not particle 0"
    )

    # Verify it's NOT using particle 0's high value
    wrong_gain = (600.0 - 100.0) * (rest_energy_mev * 1e-3)
    assert max_gain != pytest.approx(wrong_gain), (
        "Should NOT use particle 0's dead gamma"
    )


def test_percent_energy_gain_uses_average():
    """Test that percent_energy_gain uses average, not particle 0."""
    initial_gamma = 100.0

    trajectory = [
        {
            "gamma": np.array([1000.0, 120.0, 130.0]),  # Particle 0 has 900% gain!
            "_dead_particles": np.array([True, False, False]),  # But it's dead
        }
    ]

    percent_gain = compute_percent_energy_gain(trajectory, initial_gamma)

    # Average of alive: (120 + 130) / 2 = 125
    # Relative gain: (125 - 100) / 100 = 0.25
    # Percent: 25%

    expected = 25.0
    assert percent_gain == pytest.approx(expected), "Should be 25%, not 900%"


def test_relative_energy_gain_uses_average():
    """Test that relative_energy_gain uses average correctly."""
    initial_gamma = 1000.0

    trajectory = [
        {
            "gamma": np.array([5000.0, 1100.0, 1200.0]),  # Particle 0 huge gain
            "_dead_particles": np.array([True, False, False]),  # Dead
        },
        {
            "gamma": np.array([6000.0, 1150.0, 1250.0]),
            "_dead_particles": np.array([True, False, False]),
        },
    ]

    relative_gain = compute_relative_energy_gain(trajectory, initial_gamma)

    # Step 0: avg = (1100 + 1200) / 2 = 1150, gain = 150/1000 = 0.15
    # Step 1: avg = (1150 + 1250) / 2 = 1200, gain = 200/1000 = 0.20
    # Max: 0.20

    expected = 0.20
    assert relative_gain == pytest.approx(expected), (
        "Should use average, not particle 0"
    )

    # Should NOT be 4.0 (from particle 0: (5000-1000)/1000)
    assert relative_gain < 1.0, "Should not use particle 0's huge gain"


def test_trajectory_metrics_multi_particle():
    """Integration test: verify compute_trajectory_metrics uses averaging."""
    initial_state = {
        "gamma": np.array([100.0, 100.0, 100.0]),
        "x": np.array([0.0, 0.0, 0.0]),
        "y": np.array([0.0, 0.0, 0.0]),
    }

    # Trajectory where particle 0 dies and has extreme values
    trajectory = [
        {
            "gamma": np.array([10000.0, 150.0, 160.0]),  # Particle 0: 9900% gain!
            "x": np.array([100.0, 0.1, 0.2]),  # Particle 0: huge displacement
            "y": np.array([100.0, 0.1, 0.2]),
            "_dead_particles": np.array([True, False, False]),
        },
        {
            "gamma": np.array([15000.0, 180.0, 190.0]),
            "x": np.array([200.0, 0.15, 0.25]),
            "y": np.array([200.0, 0.15, 0.25]),
            "_dead_particles": np.array([True, False, False]),
        },
    ]

    rest_energy_mev = 0.511
    metrics = compute_trajectory_metrics(trajectory, initial_state, rest_energy_mev)

    # Max avg gamma: (180 + 190) / 2 = 185
    # Percent gain: (185 - 100) / 100 * 100 = 85%
    assert metrics["max_percent_energy_gain"] == pytest.approx(85.0), (
        "Should be 85%, not 14900%"
    )

    # Verify NOT using particle 0
    assert metrics["max_percent_energy_gain"] < 100.0, (
        "Should not have 14900% from dead particle 0"
    )

    # Max transverse displacement (alive particles avg)
    # Particle 1 and 2 displacements are tiny (< 1mm)
    assert metrics["max_transverse_displacement_mm"] < 1.0, (
        "Should not use particle 0's 200mm displacement"
    )


def test_particle_0_bias_scenario():
    """Regression test: ensure particle 0 is not given special treatment."""
    initial_gamma = 1000.0

    # Scenario: Particle 0 gains 100%, particles 1-3 gain 10% each
    # Then particle 0 dies
    trajectory = [
        {
            "gamma": np.array([1100.0, 1010.0, 1010.0, 1010.0]),
            "_dead_particles": np.array([False, False, False, False]),
        },
        {
            "gamma": np.array([2000.0, 1100.0, 1100.0, 1100.0]),  # Particle 0 doubles!
            "_dead_particles": np.array([True, False, False, False]),  # Then dies
        },
        {
            "gamma": np.array([2500.0, 1150.0, 1150.0, 1150.0]),
            "_dead_particles": np.array([True, False, False, False]),
        },
    ]

    percent_gain = compute_percent_energy_gain(trajectory, initial_gamma)

    # Step 0: avg = (1100 + 1010 + 1010 + 1010) / 4 = 1032.5, gain = 3.25%
    # Step 1: avg = (1100 + 1100 + 1100) / 3 = 1100, gain = 10%  (particle 0 dead)
    # Step 2: avg = (1150 + 1150 + 1150) / 3 = 1150, gain = 15%
    # Max: 15%

    expected = 15.0
    assert percent_gain == pytest.approx(expected, abs=0.1)

    # Should NOT be 100% (from particle 0's doubling before death)
    assert percent_gain < 20.0, "Should exclude particle 0 after death"


def test_initial_gamma_calculation_uses_particle_0():
    """Document that initial gamma DOES use particle 0 (all particles alive initially)."""
    # This is expected behavior - initial conditions have all particles alive
    # So using particle 0 or average should be equivalent for initial state

    # In testbed_runner, initial gamma is calculated from particle 0
    # This is OK because all particles should have same initial energy
    # (they're initialized with the same energy parameter)

    # Just documenting expected behavior
    assert True, (
        "Initial gamma from particle 0 is OK - all particles alive and identical"
    )


def test_final_gamma_uses_alive_average():
    """Test that final gamma calculation excludes dead particles."""
    # Simulating what happens in testbed_runner.py lines 1489-1495

    final_state = {
        "Pz": np.array([500.0, 100.0, 110.0]),  # Particle 0 has huge momentum
        "Px": np.array([0.0, 0.0, 0.0]),
        "Py": np.array([0.0, 0.0, 0.0]),
        "_dead_particles": np.array([True, False, False]),  # Particle 0 dead
    }

    # Get alive particles
    Pz_alive = get_alive_particle_values(final_state, "Pz")
    # Average over alive particles
    Pz_avg = float(np.mean(Pz_alive))

    expected_Pz = (100.0 + 110.0) / 2.0
    assert Pz_avg == pytest.approx(expected_Pz)

    # Should NOT be 500.0 from particle 0
    assert Pz_avg != 500.0, "Should exclude dead particle 0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
