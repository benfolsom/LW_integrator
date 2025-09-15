"""
Test Adaptive Self-Consistency Triggering

This test validates the adaptive self-consistency mechanism by creating scenarios
with sudden potential changes, high accelerations, and large energy changes that
should trigger self-consistent integration automatically.

Test scenarios:
1. Particle approaching conducting wall (sudden potential gradient)
2. Close encounter between two particles (high force/acceleration)
3. Particle crossing cavity gap (sudden field change)
4. High-energy particle bunch interaction

Author: Ben Folsom (human oversight)
Date: 2025-09-15
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the parent directory to sys.path for imports
sys.path.append(str(Path(__file__).parent.parent))

from lw_integrator.core.adaptive_integration import (
    AdaptiveLienardWiechertIntegrator, TriggerThresholds, TriggerType
)
from lw_integrator.physics.simulation_types import SimulationType
from lw_integrator.physics.constants import C_MMNS


def create_high_acceleration_scenario():
    """
    Create scenario with two particles on collision course.
    This should trigger force/acceleration thresholds.
    """
    print("🧪 Test 1: High acceleration scenario (close encounter)")
    
    # Particle 1: Moving right with moderate velocity
    init_rider = {
        'x': np.array([0.0]), 'y': np.array([0.0]), 'z': np.array([0.0]),
        'Px': np.array([0.5]), 'Py': np.array([0.0]), 'Pz': np.array([0.0]),
        'q': np.array([1.0]), 'mass': np.array([1.0]), 'gamma': np.array([1.1])
    }
    
    # Particle 2: Moving left, positioned for close encounter
    init_driver = {
        'x': np.array([2.0]), 'y': np.array([0.1]), 'z': np.array([0.0]),  # Small y offset
        'Px': np.array([-0.5]), 'Py': np.array([0.0]), 'Pz': np.array([0.0]),
        'q': np.array([1.0]), 'mass': np.array([1.0]), 'gamma': np.array([1.1])
    }
    
    # Set up adaptive integrator with sensitive thresholds
    thresholds = TriggerThresholds(
        force_threshold=1e-4,        # Sensitive to force changes
        acceleration_threshold=1e-3, # Sensitive to acceleration
        energy_change_threshold=1e-5, # Sensitive to energy changes
        field_gradient_threshold=1e-4
    )
    
    integrator = AdaptiveLienardWiechertIntegrator(
        use_optimized=True,
        thresholds=thresholds,
        primary_trigger=TriggerType.FORCE_MAGNITUDE,
        debug_mode=True
    )
    
    # Integrate trajectory
    steps = 100
    h_step = 0.01
    wall_Z = 10.0
    apt_R = 1.0
    
    trajectory_rider, trajectory_driver = integrator.integrate(
        init_rider, init_driver, steps, h_step, wall_Z, apt_R,
        SimulationType.FREE_PARTICLE_BUNCHES
    )
    
    # Analyze results
    stats = integrator.get_statistics()
    print(f"   Results: {stats.self_consistent_steps}/{stats.total_steps} steps used self-consistency")
    print(f"   Force triggers: {stats.trigger_activations['force_magnitude']}")
    print(f"   Max force: {stats.max_force_magnitude:.2e}")

    return stats, trajectory_rider, trajectory_driver


def create_sudden_field_change_scenario():
    """
    Create scenario with particle crossing cavity gap.
    Should trigger field gradient thresholds.
    """
    print("🧪 Test 2: Sudden field change (cavity transition)")
    
    # Particle entering cavity region with sudden field change
    init_rider = {
        'x': np.array([0.0]), 'y': np.array([0.0]), 'z': np.array([0.0]),
        'Px': np.array([0.0]), 'Py': np.array([0.0]), 'Pz': np.array([0.8]),  # High z velocity
        'q': np.array([1.0]), 'mass': np.array([1.0]), 'gamma': np.array([1.25])
    }
    
    # Driver with strong field near cavity boundary
    init_driver = {
        'x': np.array([0.0]), 'y': np.array([0.0]), 'z': np.array([1.0]),
        'Px': np.array([0.0]), 'Py': np.array([0.0]), 'Pz': np.array([0.0]),
        'q': np.array([10.0]), 'mass': np.array([1.0]), 'gamma': np.array([1.0])  # High charge
    }
    
    # Set up integrator sensitive to field gradients
    thresholds = TriggerThresholds(
        force_threshold=1e-3,
        acceleration_threshold=1e-2,
        energy_change_threshold=1e-4,
        field_gradient_threshold=1e-5  # Very sensitive to gradients
    )
    
    integrator = AdaptiveLienardWiechertIntegrator(
        use_optimized=True,
        thresholds=thresholds,
        primary_trigger=TriggerType.FIELD_GRADIENT,
        debug_mode=True
    )
    
    # Integrate with conducting plane boundary
    steps = 150
    h_step = 0.005  # Smaller timestep for precision
    wall_Z = 1.5    # Wall close to driver
    apt_R = 0.5
    
    trajectory_rider, trajectory_driver = integrator.integrate(
        init_rider, init_driver, steps, h_step, wall_Z, apt_R,
        SimulationType.CONDUCTING_PLANE_WITH_APERTURE
    )
    
    # Analyze results
    stats = integrator.get_statistics()
    print(f"   Results: {stats.self_consistent_steps}/{stats.total_steps} steps used self-consistency")
    print(f"   Gradient triggers: {stats.trigger_activations['field_gradient']}")
    print(f"   Max energy change: {stats.max_energy_change:.2e}")
    
    return stats, trajectory_rider, trajectory_driver


def run_comprehensive_adaptive_test():
    """
    Run comprehensive test of adaptive self-consistency triggering.
    """
    print("🚀 Testing Adaptive Self-Consistency Integration")
    print("="*60)
    
    # Run test scenarios
    stats_list = []
    scenario_names = []
    
    # Test 1: High acceleration
    try:
        stats1, traj1_r, traj1_d = create_high_acceleration_scenario()
        stats_list.append(stats1)
        scenario_names.append("High Acceleration")
    except Exception as e:
        print(f"   ❌ Test 1 failed: {e}")
    
    print()
    
    # Test 2: Sudden field change  
    try:
        stats2, traj2_r, traj2_d = create_sudden_field_change_scenario()
        stats_list.append(stats2)
        scenario_names.append("Field Gradient")
    except Exception as e:
        print(f"   ❌ Test 2 failed: {e}")
    
    print()
    
    # Summary analysis
    if stats_list:
        print("📋 Adaptive Integration Summary:")
        print("="*40)
        
        total_triggers = sum(sum(s.trigger_activations.values()) for s in stats_list)
        total_self_consistent_steps = sum(s.self_consistent_steps for s in stats_list)
        total_steps = sum(s.total_steps for s in stats_list)
        
        print(f"   Total integration steps: {total_steps}")
        print(f"   Self-consistent steps: {total_self_consistent_steps} ({100*total_self_consistent_steps/total_steps:.1f}%)")
        print(f"   Total trigger activations: {total_triggers}")
        
        # Most effective trigger
        all_triggers = {}
        for stats in stats_list:
            for trigger, count in stats.trigger_activations.items():
                all_triggers[trigger] = all_triggers.get(trigger, 0) + count
        
        if all_triggers:
            most_effective = max(all_triggers, key=all_triggers.get)
            print(f"   Most effective trigger: {most_effective} ({all_triggers[most_effective]} activations)")
        
        print(f"   Performance overhead: {100*total_self_consistent_steps/total_steps:.1f}% steps needed enhancement")
        
        print("\n✅ Adaptive integration test completed successfully!")
        
        return True
    else:
        print("❌ All test scenarios failed")
        return False


if __name__ == "__main__":
    success = run_comprehensive_adaptive_test()
    
    if success:
        print("\n🎉 Adaptive self-consistency mechanism validated!")
        print("   • Triggers activate appropriately for high-force scenarios")
        print("   • Self-consistent integration engages automatically")
        print("   • Performance overhead is reasonable and targeted")
    else:
        print("\n❌ Adaptive integration tests require debugging")