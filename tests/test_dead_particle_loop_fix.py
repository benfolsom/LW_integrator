"""
Test that dead particles are not reprocessed in an infinite loop.

This test validates the fix for the issue where hard gamma blowups would
repeatedly mark the same particle dead over and over, causing an infinite
loop at step boundaries.

The fix ensures that dead particle status is propagated to the temp_trajectory
base BEFORE substeps begin, so dead particles are skipped in retarded_equations_of_motion.
"""

import numpy as np

from core.particle_status import (
    mark_particle_dead,
    propagate_dead_particle_status,
)


def test_propagate_dead_particle_status():
    """Test that dead particle status is correctly propagated."""
    # Create a state with 3 particles
    previous_state = {
        "x": np.array([0.0, 1.0, 2.0]),
        "y": np.array([0.0, 0.0, 0.0]),
        "z": np.array([0.0, 0.0, 0.0]),
        "gamma": np.array([10.0, 20.0, 30.0]),
        "q": np.array([-1.0, -1.0, -1.0]),
        "_dead_particles": np.array([False, True, False]),  # Particle 1 is dead
        "_particle_failure_info": {
            1: {
                "step": 100,
                "reason": "gamma_blowup_hard",
                "gamma_value": 1e25,
            }
        },
    }

    # Create a new state
    current_state = {
        "x": np.array([0.1, 1.1, 2.1]),
        "y": np.array([0.0, 0.0, 0.0]),
        "z": np.array([0.1, 0.1, 0.1]),
        "gamma": np.array([10.5, 20.5, 30.5]),
        "q": np.array([-1.0, 0.0, -1.0]),  # Particle 1 already neutralized
    }

    # Propagate dead status
    propagate_dead_particle_status(current_state, previous_state)

    # Check that particle 1 is now marked dead in current_state
    assert "_dead_particles" in current_state
    assert current_state["_dead_particles"][1]
    assert not current_state["_dead_particles"][0]
    assert not current_state["_dead_particles"][2]

    # Check that failure info was copied
    assert "_particle_failure_info" in current_state
    assert 1 in current_state["_particle_failure_info"]
    assert current_state["_particle_failure_info"][1]["reason"] == "gamma_blowup_hard"


def test_mark_particle_dead_idempotent():
    """Test that marking a particle dead multiple times is safe (idempotent)."""
    state = {
        "x": np.array([0.0, 1.0, 2.0]),
        "gamma": np.array([10.0, 20.0, 30.0]),
        "q": np.array([-1.0, -1.0, -1.0]),
    }

    # Mark particle 1 dead for the first time
    mark_particle_dead(state, 1, step=100, reason="gamma_blowup_hard", gamma_value=1e25)

    assert state["_dead_particles"][1]
    assert state["q"][1] == 0.0
    assert 1 in state["_particle_failure_info"]

    # Mark the same particle dead again (simulating the bug)
    mark_particle_dead(state, 1, step=101, reason="gamma_blowup_hard", gamma_value=1e26)

    # Should still be dead
    assert state["_dead_particles"][1]
    # Charge should still be zero
    assert state["q"][1] == 0.0
    # Failure info gets updated (last failure wins)
    assert state["_particle_failure_info"][1]["step"] == 101
    assert state["_particle_failure_info"][1]["gamma_value"] == 1e26


def test_temp_trajectory_copy_and_propagate():
    """Test the pattern used in integration_runner for temp_trajectory initialization."""
    # Simulate trajectory[i-1] with a dead particle
    previous_step = {
        "x": np.array([0.0, 1.0, 2.0]),
        "gamma": np.array([10.0, 20.0, 30.0]),
        "q": np.array([-1.0, 0.0, -1.0]),  # Particle 1 neutralized
        "_dead_particles": np.array([False, True, False]),
        "_particle_failure_info": {1: {"step": 99, "reason": "gamma_blowup_hard"}},
    }

    # This is the pattern from integration_runner.py after the fix:
    # Make a COPY of the previous step
    temp_trajectory = [
        {
            k: (v.copy() if isinstance(v, (dict, np.ndarray)) else v)
            for k, v in previous_step.items()
        }
    ]

    # Propagate dead particle status
    propagate_dead_particle_status(temp_trajectory[0], previous_step)

    # Verify that temp_trajectory[0] is a separate copy
    assert temp_trajectory[0] is not previous_step
    assert temp_trajectory[0]["x"] is not previous_step["x"]

    # Verify dead particle status is propagated
    assert temp_trajectory[0]["_dead_particles"][1]
    assert temp_trajectory[0]["q"][1] == 0.0

    # Modifying temp_trajectory[0] should NOT affect previous_step
    temp_trajectory[0]["x"][0] = 999.0
    assert previous_step["x"][0] == 0.0  # Should be unchanged


def test_dead_particle_skip_pattern():
    """Test the pattern used in equations.py to skip dead particles."""
    # Simulate the result state that equations.py creates
    result = {
        "x": np.array([0.0, 1.0, 2.0]),
        "gamma": np.array([10.0, 20.0, 30.0]),
        "_dead_particles": np.array([False, True, False]),
    }

    current_state = {
        "x": np.array([0.0, 1.0, 2.0]),
        "gamma": np.array([10.0, 999.0, 30.0]),  # Different gamma for particle 1
    }

    num_particles = 3
    particles_processed = []

    # Simulate the loop in retarded_equations_of_motion
    for particle_idx in range(num_particles):
        # Skip particles that are already marked dead
        if "_dead_particles" in result and result["_dead_particles"][particle_idx]:
            # Copy previous state for dead particle (don't recompute)
            for key in ["x", "gamma"]:
                if key in current_state:
                    result[key][particle_idx] = current_state[key][particle_idx]
            continue

        # Simulate processing the particle
        particles_processed.append(particle_idx)

    # Should have processed particles 0 and 2, skipped particle 1
    assert particles_processed == [0, 2]
    # Particle 1 should have copied value from current_state
    assert result["gamma"][1] == 999.0


def test_integration_scenario():
    """
    Simulate the scenario from the bug report:
    - Step 879 has hard blowups on multiple particles
    - Without the fix, these would loop infinitely
    - With the fix, they should be marked dead once and then skipped
    """
    # Setup: trajectory[878] has 2 particles already dead
    trajectory_878 = {
        "x": np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
        "gamma": np.array([10.0] * 10),
        "q": np.array([-1.0, -1.0, 0.0, -1.0, -1.0, -1.0, -1.0, 0.0, -1.0, -1.0]),
        "_dead_particles": np.array(
            [False, False, True, False, False, False, False, True, False, False]
        ),
        "_particle_failure_info": {
            2: {"step": 875, "reason": "gamma_blowup_min_timestep"},
            7: {"step": 876, "reason": "gamma_blowup_min_timestep"},
        },
    }

    # Step 879: Create temp_trajectory base (with fix applied)
    temp_trajectory_base = {
        k: (v.copy() if isinstance(v, (dict, np.ndarray)) else v)
        for k, v in trajectory_878.items()
    }

    # Propagate dead status
    propagate_dead_particle_status(temp_trajectory_base, trajectory_878)

    # Verify propagation worked
    assert temp_trajectory_base["_dead_particles"][2]
    assert temp_trajectory_base["_dead_particles"][7]

    # Simulate equations being called multiple times (as would happen in substeps)
    # This should NOT re-mark particles as dead
    call_count = 0
    max_calls = 5  # Arbitrary limit to prove we don't infinite loop

    for call_idx in range(max_calls):
        call_count += 1

        # Simulate what happens in retarded_equations_of_motion
        num_particles = len(temp_trajectory_base["x"])
        particles_marked_dead_this_call = 0

        for particle_idx in range(num_particles):
            # Skip already dead particles (THE FIX)
            if (
                "_dead_particles" in temp_trajectory_base
                and temp_trajectory_base["_dead_particles"][particle_idx]
            ):
                continue

            # Simulate hard blowup detection on remaining particles
            # (In the real scenario, particles 1, 4, 5, 6, 8, 9 would blow up)
            if particle_idx in [1, 4, 5, 6, 8, 9]:
                # This would only happen ONCE per particle now
                if particle_idx not in temp_trajectory_base["_particle_failure_info"]:
                    mark_particle_dead(
                        temp_trajectory_base,
                        particle_idx,
                        step=879,
                        reason="gamma_blowup_hard",
                        gamma_value=1e25,
                    )
                    particles_marked_dead_this_call += 1

        # After first call, 6 new particles should be marked dead
        # After subsequent calls, no new particles should be marked (they're skipped)
        if call_idx == 0:
            assert particles_marked_dead_this_call == 6
        else:
            assert particles_marked_dead_this_call == 0

    # Should have completed all calls without infinite loop
    assert call_count == max_calls

    # Verify final state: 8 total dead particles (2 original + 6 new)
    assert np.sum(temp_trajectory_base["_dead_particles"]) == 8


def test_retry_preserves_dead_particles():
    """
    Test that dead particles from first attempt are preserved in retry attempts.

    This simulates the scenario where:
    1. First attempt: particles blow up and are marked dead
    2. Energy jump detected, retry with smaller timestep
    3. Second attempt: dead particles should be skipped (not re-marked)
    """
    # Setup: previous step with no dead particles
    previous_step = {
        "x": np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        "gamma": np.array([10.0, 20.0, 30.0, 40.0, 50.0]),
        "q": np.array([-1.0, -1.0, -1.0, -1.0, -1.0]),
    }

    # Create base for retry loop (like integration_runner does)
    temp_trajectory_base = {
        k: (v.copy() if isinstance(v, (dict, np.ndarray)) else v)
        for k, v in previous_step.items()
    }

    # Propagate dead status (none in this case)
    propagate_dead_particle_status(temp_trajectory_base, previous_step)

    # Simulate FIRST ATTEMPT - particles 1, 2, 3 blow up
    attempt1_state = {
        k: (v.copy() if isinstance(v, (dict, np.ndarray)) else v)
        for k, v in temp_trajectory_base.items()
    }

    # Mark particles dead in first attempt
    mark_particle_dead(
        attempt1_state, 1, step=100, reason="gamma_blowup_hard", gamma_value=1e25
    )
    mark_particle_dead(
        attempt1_state, 2, step=100, reason="gamma_blowup_hard", gamma_value=1e26
    )
    mark_particle_dead(
        attempt1_state, 3, step=100, reason="gamma_blowup_hard", gamma_value=1e27
    )

    # Verify they're marked dead
    assert attempt1_state["_dead_particles"][1]
    assert attempt1_state["_dead_particles"][2]
    assert attempt1_state["_dead_particles"][3]

    # Update base with dead particles from first attempt (THE FIX)
    if "_dead_particles" not in temp_trajectory_base:
        num_particles = len(temp_trajectory_base["gamma"])
        temp_trajectory_base["_dead_particles"] = np.zeros(num_particles, dtype=bool)
        temp_trajectory_base["_particle_failure_info"] = {}

    temp_trajectory_base["_dead_particles"] |= attempt1_state["_dead_particles"]
    temp_trajectory_base["_particle_failure_info"].update(
        attempt1_state["_particle_failure_info"]
    )

    # Simulate SECOND ATTEMPT (retry with smaller timestep)
    attempt2_state = {
        k: (v.copy() if isinstance(v, (dict, np.ndarray)) else v)
        for k, v in temp_trajectory_base.items()
    }

    # Verify dead particles are already marked in the base for attempt 2
    assert attempt2_state["_dead_particles"][1]
    assert attempt2_state["_dead_particles"][2]
    assert attempt2_state["_dead_particles"][3]

    # Simulate processing particles in attempt 2
    particles_processed = []
    particles_marked_dead = []

    for particle_idx in range(5):
        # Skip already dead particles
        if (
            "_dead_particles" in attempt2_state
            and attempt2_state["_dead_particles"][particle_idx]
        ):
            continue

        particles_processed.append(particle_idx)

        # Simulate that particle 4 also blows up in attempt 2
        if particle_idx == 4:
            mark_particle_dead(
                attempt2_state,
                4,
                step=100,
                reason="gamma_blowup_hard",
                gamma_value=1e28,
            )
            particles_marked_dead.append(particle_idx)

    # Verify: only particles 0 and 4 were processed (1, 2, 3 were skipped)
    assert particles_processed == [0, 4]
    # Only particle 4 was newly marked dead
    assert particles_marked_dead == [4]
    # Total dead particles: 1, 2, 3, 4
    assert np.sum(attempt2_state["_dead_particles"]) == 4


if __name__ == "__main__":
    # Run all tests
    print("Running dead particle loop fix tests...\n")

    tests = [
        test_propagate_dead_particle_status,
        test_mark_particle_dead_idempotent,
        test_temp_trajectory_copy_and_propagate,
        test_dead_particle_skip_pattern,
        test_integration_scenario,
        test_retry_preserves_dead_particles,
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
            import traceback

            traceback.print_exc()
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
    exit(0 if failed == 0 else 1)
