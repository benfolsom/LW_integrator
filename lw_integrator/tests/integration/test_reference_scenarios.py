"""
Integration tests using reference test cases from original notebooks.

CAI: These tests validate our refactored code against the exact scenarios
used in the original research, particularly focusing on the GeV instability issue.

Author: Ben Folsom (human oversight)
Date: 2025-09-12
"""

import pytest
import numpy as np
import warnings
from typing import Dict, Any, List
from unittest.mock import patch

from lw_integrator.tests.reference_tests import (
    ReferenceTestCases, 
    SimulationConfig,
    create_reference_test_data,
    C_MMNS
)
from lw_integrator.core.particles import ParticleEnsemble


class TestReferenceScenarios:
    """
    Test suite for reference scenarios extracted from notebooks.
    
    CAI: These tests serve as regression tests and help identify
    the root cause of GeV range instabilities.
    """
    
    def test_reference_data_creation(self):
        """Test that reference test data can be created without errors."""
        # CAI: Basic sanity check for our test data structure
        test_data = create_reference_test_data()
        
        assert "basic_tests" in test_data
        assert "high_energy_tests" in test_data
        assert "stability_tests" in test_data
        assert "sweep_tests" in test_data
        
        # CAI: Verify we have the expected number of test cases
        assert len(test_data["basic_tests"]) >= 1
        assert len(test_data["high_energy_tests"]) >= 2
        assert len(test_data["stability_tests"]) >= 5
        assert len(test_data["sweep_tests"]) >= 5
    
    def test_basic_configuration_validation(self):
        """Test that basic configurations have reasonable physics values."""
        # CAI: Validate the basic proton-antiproton test case
        basic_test = ReferenceTestCases.proton_antiproton_basic()
        basic_test = ReferenceTestCases.calculate_derived_parameters(basic_test)
        
        # CAI: Check particle masses are reasonable
        assert 0.5 < basic_test.m_particle_rider < 500  # Reasonable mass range
        assert 0.5 < basic_test.m_particle_driver < 500
        
        # CAI: Check charges make sense
        assert basic_test.stripped_ions_rider > 0
        assert basic_test.stripped_ions_driver > 0
        assert abs(basic_test.charge_sign_rider) == 1
        assert abs(basic_test.charge_sign_driver) == 1
        
        # CAI: Check momentum is calculated correctly
        assert basic_test.starting_Pz_driver is not None
        expected_Pz_driver = (
            -basic_test.starting_Pz_rider 
            / basic_test.m_particle_driver 
            * basic_test.m_particle_rider
        )
        assert abs(basic_test.starting_Pz_driver - expected_Pz_driver) < 1e-10
    
    def test_high_energy_scenario_identification(self):
        """Test identification of high-energy scenarios prone to instability."""
        # CAI: Get the high-energy test cases
        high_energy_tests = ReferenceTestCases.get_all_reference_tests()["high_energy_tests"]
        
        for test in high_energy_tests:
            test = ReferenceTestCases.calculate_derived_parameters(test)
            
            # CAI: Calculate approximate gamma factor
            mass_kg = test.m_particle_rider * 1.66053907e-27  # amu to kg
            c_ms = 299792458
            
            # CAI: Rough momentum in SI units (this is approximate)
            p_si_approx = test.starting_Pz_rider * mass_kg * 1e6  # Very rough conversion
            energy_approx = p_si_approx * c_ms  # p*c approximation for high energy
            energy_mev = energy_approx * 6.242e12
            
            # CAI: High-energy tests should be in GeV range
            if test.expected_instability:
                # CAI: These should be high-energy scenarios
                assert test.starting_Pz_rider > 1e5, f"High-energy test has low momentum: {test.starting_Pz_rider}"
                assert test.step_size < 1e-6, f"High-energy test should use small timestep: {test.step_size}"
    
    def test_step_size_stability_threshold(self):
        """Test the step size stability threshold identification."""
        # CAI: Test the step size threshold mentioned in the original code
        stability_tests = ReferenceTestCases.stability_threshold_test()
        
        # CAI: According to the original README, step_size < 1e-7 causes instability
        for test in stability_tests:
            if test.step_size < 1e-7:
                assert test.expected_instability, f"Step size {test.step_size} should be unstable"
            else:
                assert not test.expected_instability, f"Step size {test.step_size} should be stable"
    
    def test_momentum_sweep_parameters(self):
        """Test momentum sweep parameter generation."""
        # CAI: Test the momentum sweep that reproduces the GeV instability
        sweep_tests = ReferenceTestCases.momentum_sweep_test()
        
        assert len(sweep_tests) > 5, "Should have multiple momentum values"
        
        # CAI: Check momentum values are in ascending order
        momenta = [test.starting_Pz_rider for test in sweep_tests]
        assert momenta == sorted(momenta), "Momentum values should be in order"
        
        # CAI: Check adaptive step size (from original notebook)
        for i, test in enumerate(sweep_tests):
            expected_step = 1.8e-8 + (i+1)*6.5e-9
            assert abs(test.step_size - expected_step) < 1e-12, f"Adaptive step size mismatch at index {i}"
    
    def test_simulation_type_configurations(self):
        """Test that simulation type configurations are valid."""
        # CAI: Check sim_type values match the original notebook conventions
        all_tests = create_reference_test_data()
        
        for category, tests in all_tests.items():
            if isinstance(tests, list):
                test_list = tests
            else:
                test_list = [tests]
            
            for test in test_list:
                # CAI: sim_type should be valid (based on original code analysis)
                assert test.sim_type in [0, 1, 2], f"Invalid sim_type: {test.sim_type}"
                
                # CAI: Particle counts should be reasonable
                assert test.pcount_rider >= 1
                assert test.pcount_driver >= 1
                assert test.pcount_rider <= 100  # Reasonable upper limit
                assert test.pcount_driver <= 100
    
    @pytest.mark.slow
    def test_particle_initialization_from_config(self):
        """Test that we can create ParticleEnsemble from reference configurations."""
        # CAI: This tests our ability to recreate the original initialization
        basic_test = ReferenceTestCases.proton_antiproton_basic()
        basic_test = ReferenceTestCases.calculate_derived_parameters(basic_test)
        
        # CAI: Create particle ensembles matching the original configuration
        rider_particles = ParticleEnsemble(basic_test.pcount_rider)
        driver_particles = ParticleEnsemble(basic_test.pcount_driver)
        
        # CAI: Test basic properties
        assert rider_particles.n_particles == basic_test.pcount_rider
        assert driver_particles.n_particles == basic_test.pcount_driver
        
        # CAI: Verify memory layout is contiguous (for performance)
        assert rider_particles.positions.flags.c_contiguous
        assert driver_particles.positions.flags.c_contiguous
        assert rider_particles.momenta.flags.c_contiguous
        assert driver_particles.momenta.flags.c_contiguous
    
    def test_energy_range_calculations(self):
        """Test energy range calculations for high-energy scenarios."""
        # CAI: Verify we can identify GeV range scenarios correctly
        high_energy_test = ReferenceTestCases.high_energy_proton_gold()
        high_energy_test = ReferenceTestCases.calculate_derived_parameters(high_energy_test)
        
        # CAI: Calculate approximate energy from momentum
        # This is a rough calculation to verify we're in the right energy range
        mass_amu = high_energy_test.m_particle_rider
        Pz = high_energy_test.starting_Pz_rider
        
        # CAI: Using the momentum units from the original code (amu*mm/ns)
        # Convert to approximate gamma factor
        c_mmns = C_MMNS
        
        # CAI: Rough gamma calculation (this should be refined)
        # gamma ~ Pz / (mass * c) for high energy
        gamma_approx = abs(Pz) / (mass_amu * c_mmns)
        
        # CAI: High-energy scenarios should have high gamma
        if high_energy_test.expected_instability:
            assert gamma_approx > 100, f"High-energy test should have high gamma: {gamma_approx}"
    
    def test_configuration_consistency(self):
        """Test that all configurations are internally consistent."""
        # CAI: Check all test configurations for internal consistency
        all_tests = create_reference_test_data()
        
        for category, tests in all_tests.items():
            if isinstance(tests, list):
                test_list = tests
            else:
                test_list = [tests]
            
            for test in test_list:
                # CAI: Check basic parameter consistency
                assert test.static_steps >= 1
                assert test.ret_steps >= 1
                assert test.step_size > 0
                assert test.step_size < 1e-3  # Should be small for precision
                
                # CAI: Check distances are reasonable
                assert test.starting_distance_rider > 0
                assert test.starting_distance_driver > 0
                assert test.transv_dist > 0
                
                # CAI: Check boundary parameters
                assert test.wall_pos > max(test.starting_distance_rider, test.starting_distance_driver)
                assert test.aperture > 0
                assert test.bunch_dist > 0
                assert test.cav_spacing > 0


class TestGeVInstabilityReproduction:
    """
    Specific tests for reproducing the GeV range instability.
    
    CAI: These tests are designed to isolate and characterize
    the numerical instability that occurs at high energies.
    """
    
    def test_high_energy_configuration_setup(self):
        """Test that we can set up the problematic high-energy configuration."""
        # CAI: Get the specific configuration that shows instability
        gev_test = ReferenceTestCases.high_energy_proton_gold()
        gev_test = ReferenceTestCases.calculate_derived_parameters(gev_test)
        
        # CAI: This should be the configuration that causes problems
        assert gev_test.expected_instability
        assert gev_test.starting_Pz_rider > 9e5  # Very high momentum
        assert gev_test.step_size < 5e-8  # Very small timestep
        
        # CAI: Document the problematic parameters
        print(f"\nGeV instability test parameters:")
        print(f"  Rider momentum: {gev_test.starting_Pz_rider:.2e}")
        print(f"  Driver momentum: {gev_test.starting_Pz_driver:.2e}")
        print(f"  Step size: {gev_test.step_size:.2e}")
        print(f"  Integration steps: {gev_test.ret_steps}")
    
    def test_electron_high_energy_setup(self):
        """Test electron high-energy setup - particularly problematic case."""
        # CAI: Electrons at GeV energies have extreme gamma factors
        electron_test = ReferenceTestCases.electron_high_energy()
        electron_test = ReferenceTestCases.calculate_derived_parameters(electron_test)
        
        # CAI: Electron mass is much smaller, so gamma factors are extreme
        assert electron_test.m_particle_rider < 0.001  # Electron mass
        assert electron_test.expected_instability
        
        # CAI: Calculate approximate gamma for electron
        mass_ratio = electron_test.m_particle_rider / 1.007319468  # electron/proton
        print(f"\nElectron test mass ratio: {mass_ratio:.6f}")
        print(f"Expected extreme relativistic conditions")
    
    def test_instability_threshold_characterization(self):
        """Characterize the instability threshold from step size tests."""
        # CAI: This helps us understand exactly where the instability begins
        threshold_tests = ReferenceTestCases.stability_threshold_test()
        
        stable_cases = [t for t in threshold_tests if not t.expected_instability]
        unstable_cases = [t for t in threshold_tests if t.expected_instability]
        
        if stable_cases and unstable_cases:
            max_stable_step = max(t.step_size for t in stable_cases)
            min_unstable_step = min(t.step_size for t in unstable_cases)
            
            # CAI: The threshold should be around 1e-7 based on original comments
            assert max_stable_step >= 1e-7, f"Stable step size too small: {max_stable_step}"
            assert min_unstable_step < 1e-7, f"Unstable step size too large: {min_unstable_step}"
            
            print(f"\nStability threshold characterization:")
            print(f"  Largest stable step: {max_stable_step:.2e}")
            print(f"  Smallest unstable step: {min_unstable_step:.2e}")
            print(f"  Threshold around: ~1e-7 (as documented)")


if __name__ == "__main__":
    # CAI: Run basic tests when executed directly
    print("Running reference test validation...")
    
    # Test data creation
    test_data = create_reference_test_data()
    print(f"Created {sum(len(v) if isinstance(v, list) else 1 for v in test_data.values())} test cases")
    
    # Show high-energy instability parameters
    gev_test = ReferenceTestCases.high_energy_proton_gold()
    gev_test = ReferenceTestCases.calculate_derived_parameters(gev_test)
    print(f"\nGeV instability test ready:")
    print(f"  Configuration: {gev_test.description}")
    print(f"  Expected instability: {gev_test.expected_instability}")
    print(f"  Momentum (Pz): {gev_test.starting_Pz_rider:.2e}")
    print(f"  Step size: {gev_test.step_size:.2e}")
    
    print("\nReference test extraction complete!")
    print("Next: Implement integration with original bunch_inits.py")
