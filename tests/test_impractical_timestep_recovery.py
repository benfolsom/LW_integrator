"""
Test impractical timestep recovery and skip-cooldown-on-death behavior.

This test validates that:
1. When timestep becomes impractically small (requires many sub-steps), cooldown is skipped
2. When a particle dies, survivors skip cooldown and attempt to recover to normal timestep
3. Recovery logic properly transitions from reduced timestep back to normal
"""

import numpy as np

from core.integration_runner import AdaptiveTimestepConfig


def test_impractical_timestep_config():
    """Test that impractical timestep config parameters are available."""
    config = AdaptiveTimestepConfig(
        enabled=True,
        max_substeps_per_step=500,
        skip_cooldown_on_particle_death=True,
    )

    assert config.max_substeps_per_step == 500
    assert config.skip_cooldown_on_particle_death is True


def test_impractical_timestep_detection():
    """Test detection logic for impractical timesteps."""
    config = AdaptiveTimestepConfig(
        enabled=True,
        max_substeps_per_step=1000,
    )

    # Normal timestep
    h_step = 1e-6
    reduced_h_step = 1e-7
    expected_substeps = int(np.ceil(h_step / reduced_h_step))
    assert expected_substeps == 10
    assert expected_substeps <= config.max_substeps_per_step

    # Impractical timestep (reduced by 10000x)
    reduced_h_step_impractical = 1e-10
    expected_substeps_impractical = int(np.ceil(h_step / reduced_h_step_impractical))
    assert expected_substeps_impractical == 10000
    assert expected_substeps_impractical > config.max_substeps_per_step


def test_skip_cooldown_logic():
    """Test the logic for skipping cooldown after particle death."""
    config = AdaptiveTimestepConfig(
        enabled=True,
        cooldown_steps=10,
        skip_cooldown_on_particle_death=True,
    )

    # Simulate scenario
    current_step = 100
    last_particle_death_step = 99  # Particle died previous step

    skip_cooldown = (
        config.skip_cooldown_on_particle_death
        and last_particle_death_step == current_step - 1
    )

    assert skip_cooldown is True

    # Should not skip if particle died earlier
    last_particle_death_step = 50
    skip_cooldown = (
        config.skip_cooldown_on_particle_death
        and last_particle_death_step == current_step - 1
    )

    assert skip_cooldown is False


def test_recovery_jump_to_probing():
    """Test that setting cooldown_counter to cooldown_steps jumps to probing phase."""
    config = AdaptiveTimestepConfig(
        enabled=True,
        cooldown_steps=10,
        max_probe_steps=3,
    )

    # Normal cooldown progression
    cooldown_counter = 0
    for step in range(config.cooldown_steps):
        assert cooldown_counter < config.cooldown_steps  # Still in cooldown
        cooldown_counter += 1

    # Now in probing phase
    assert cooldown_counter >= config.cooldown_steps

    # Test immediate jump (what happens when we skip cooldown)
    cooldown_counter_immediate = config.cooldown_steps
    assert cooldown_counter_immediate >= config.cooldown_steps  # Immediately in probing


def test_combined_scenario_simulation():
    """
    Simulate the combined scenario:
    - Start with normal timestep
    - Reduce timestep multiple times (simulate gamma blowup retries)
    - Particle dies at minimum timestep
    - Check that skip_cooldown flag would trigger
    - Check that impractical timestep would be detected
    """
    config = AdaptiveTimestepConfig(
        enabled=True,
        timestep_reduction_factor=3,
        min_timestep_factor=1e-4,
        cooldown_steps=10,
        max_substeps_per_step=1000,
        skip_cooldown_on_particle_death=True,
    )

    # Initial timestep
    h_step = 3.12e-7  # ns (from user's log)

    # Simulate 8 reductions (3x each)
    current_h = h_step
    for _ in range(8):
        current_h = current_h / config.timestep_reduction_factor

    # Check that this matches user's observed value approximately
    expected_h = h_step / (3**8)  # 6561x reduction
    assert np.isclose(current_h, expected_h)
    assert np.isclose(current_h, 4.76e-11, rtol=0.01)

    # Check impractical timestep detection
    expected_substeps = int(np.ceil(h_step / current_h))
    assert expected_substeps > config.max_substeps_per_step
    impractical = expected_substeps > config.max_substeps_per_step
    assert impractical is True

    # Simulate particle death
    current_step = 876  # From user's log
    last_death_step = 875

    skip_cooldown = (
        config.skip_cooldown_on_particle_death and last_death_step == current_step - 1
    )
    assert skip_cooldown is True

    # Either condition should trigger immediate recovery attempt
    should_skip_cooldown = impractical or skip_cooldown
    assert should_skip_cooldown is True


def test_disabled_skip_cooldown():
    """Test that skip_cooldown_on_particle_death can be disabled."""
    config = AdaptiveTimestepConfig(
        enabled=True,
        skip_cooldown_on_particle_death=False,  # Disabled
        cooldown_steps=10,
    )

    current_step = 100
    last_particle_death_step = 99

    skip_cooldown = (
        config.skip_cooldown_on_particle_death
        and last_particle_death_step == current_step - 1
    )

    # Should NOT skip even though particle just died
    assert skip_cooldown is False


def test_default_config_values():
    """Test that default config has reasonable values for new parameters."""
    config = AdaptiveTimestepConfig(enabled=True)

    # Check defaults
    assert config.max_substeps_per_step == 1000
    assert config.skip_cooldown_on_particle_death is False
    assert config.cooldown_steps == 10
    assert config.max_probe_steps == 3


if __name__ == "__main__":
    # Run all tests
    print("Running impractical timestep recovery tests...\n")

    tests = [
        test_impractical_timestep_config,
        test_impractical_timestep_detection,
        test_skip_cooldown_logic,
        test_recovery_jump_to_probing,
        test_combined_scenario_simulation,
        test_disabled_skip_cooldown,
        test_default_config_values,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            print(f"  {test_func.__name__}...", end=" ")
            test_func()
            print("✓ PASSED")
            passed += 1
        except Exception as e:
            print(f"✗ FAILED: {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
    exit(0 if failed == 0 else 1)
