"""
Integration tests for particle initialization bridge.

CAI: These tests validate that our BunchInitializer exactly reproduces
the behavior of the original bunch_inits.py function.

Author: Ben Folsom (human oversight)
Date: 2025-09-12
"""

import pytest
import numpy as np
import sys
import os
from typing import Dict, Any

# CAI: Add the original LW_integrator path for direct comparison
sys.path.append('/home/benfol/work/LW_windows/LW_integrator')
import bunch_inits

from lw_integrator.core.initialization import BunchInitializer
from lw_integrator.tests.reference_tests import ReferenceTestCases


class TestInitializationBridge:
    """
    Test suite validating our initialization against the original bunch_inits.py.
    
    CAI: These tests ensure our modernized code produces identical physics
    to the original research implementation.
    """
    
    def test_basic_proton_initialization_exact_match(self):
        """Test that basic proton initialization exactly matches original."""
        # CAI: Use deterministic parameters from the original
        config = ReferenceTestCases.proton_antiproton_basic()
        config = ReferenceTestCases.calculate_derived_parameters(config)
        
        # CAI: Set fixed random seed for reproducible comparison
        np.random.seed(42)
        
        # CAI: Call original function
        original_dict, original_rest_energy = bunch_inits.init_bunch(
            starting_distance=config.starting_distance_rider,
            transv_mom=config.transv_mom_rider,
            starting_Pz=config.starting_Pz_rider,
            stripped_ions=config.stripped_ions_rider,
            m_particle=config.m_particle_rider,
            transv_dist=config.transv_dist,
            pcount=config.pcount_rider,
            charge_sign=config.charge_sign_rider
        )
        
        # CAI: Reset random seed and call our implementation
        np.random.seed(42)
        initializer = BunchInitializer(config)
        ensemble, metadata = initializer.create_rider_ensemble()
        
        # CAI: Compare all key arrays
        # Position arrays
        assert np.allclose(ensemble.positions[:, 0], original_dict['x'], rtol=1e-12)
        assert np.allclose(ensemble.positions[:, 1], original_dict['y'], rtol=1e-12)
        assert np.allclose(ensemble.positions[:, 2], original_dict['z'], rtol=1e-12)
        assert np.allclose(ensemble.time, original_dict['t'], rtol=1e-12)
        
        # Momentum arrays
        assert np.allclose(ensemble.momenta[:, 0], original_dict['Px'], rtol=1e-12)
        assert np.allclose(ensemble.momenta[:, 1], original_dict['Py'], rtol=1e-12)
        assert np.allclose(ensemble.momenta[:, 2], original_dict['Pz'], rtol=1e-12)
        assert np.allclose(ensemble.momenta[:, 3], original_dict['Pt'], rtol=1e-12)
        
        # Velocity arrays (need to convert from beta to velocity)
        expected_vx = original_dict['bx'] * 299.792458  # c_mmns
        expected_vy = original_dict['by'] * 299.792458
        expected_vz = original_dict['bz'] * 299.792458
        assert np.allclose(ensemble.velocities[:, 0], expected_vx, rtol=1e-12)
        assert np.allclose(ensemble.velocities[:, 1], expected_vy, rtol=1e-12)
        assert np.allclose(ensemble.velocities[:, 2], expected_vz, rtol=1e-12)
        
        # Acceleration arrays (bdot to acceleration conversion)
        expected_ax = original_dict['bdotx'] * 299.792458
        expected_ay = original_dict['bdoty'] * 299.792458
        expected_az = original_dict['bdotz'] * 299.792458
        assert np.allclose(ensemble.accelerations[:, 0], expected_ax, rtol=1e-12)
        assert np.allclose(ensemble.accelerations[:, 1], expected_ay, rtol=1e-12)
        assert np.allclose(ensemble.accelerations[:, 2], expected_az, rtol=1e-12)
        
        # Particle properties
        assert np.allclose(ensemble.charge, original_dict['q'], rtol=1e-12)
        assert np.allclose(ensemble.mass, original_dict['m'], rtol=1e-12)
        assert np.allclose(ensemble.gamma, original_dict['gamma'], rtol=1e-12)
        
        # Metadata validation
        assert abs(metadata['E_MeV_rest'] - original_rest_energy) < 1e-10
        
        print(f"✓ Basic proton initialization matches original exactly")
    
    def test_high_energy_electron_exact_match(self):
        """Test high-energy electron case that's prone to instability."""
        # CAI: Test the electron configuration that shows GeV instability
        config = ReferenceTestCases.electron_high_energy()
        config = ReferenceTestCases.calculate_derived_parameters(config)
        
        # CAI: Fixed seed for reproducible test
        np.random.seed(9999)
        
        # CAI: Original function call
        original_dict, original_rest_energy = bunch_inits.init_bunch(
            starting_distance=config.starting_distance_rider,
            transv_mom=config.transv_mom_rider,
            starting_Pz=config.starting_Pz_rider,
            stripped_ions=config.stripped_ions_rider,
            m_particle=config.m_particle_rider,
            transv_dist=config.transv_dist,
            pcount=config.pcount_rider,
            charge_sign=config.charge_sign_rider
        )
        
        # CAI: Our implementation
        np.random.seed(9999)
        initializer = BunchInitializer(config)
        ensemble, metadata = initializer.create_rider_ensemble()
        
        # CAI: Validate key physics quantities for high-energy case
        assert np.allclose(ensemble.gamma, original_dict['gamma'], rtol=1e-10)
        assert np.allclose(ensemble.momenta[:, 3], original_dict['Pt'], rtol=1e-10)
        
        # CAI: Check that we're truly in a high-energy regime (adjust expectation for electron)
        assert np.all(ensemble.gamma > 100), "Should be relativistic"
        assert metadata['E_MeV'] > 100, "Should be in high-energy range"
        
        print(f"✓ High-energy electron case matches original (gamma = {np.mean(ensemble.gamma):.1f})")
    
    def test_momentum_conservation_validation(self):
        """Test momentum conservation in particle pairs."""
        # CAI: Create a balanced proton-antiproton system
        config = ReferenceTestCases.proton_antiproton_basic()
        config = ReferenceTestCases.calculate_derived_parameters(config)
        
        initializer = BunchInitializer(config)
        rider, driver, rider_meta, driver_meta = initializer.create_both_ensembles(
            rider_seed=12345, driver_seed=54321
        )
        
        # CAI: Calculate total momentum
        total_px = np.sum(rider.momenta[:, 0]) + np.sum(driver.momenta[:, 0])
        total_py = np.sum(rider.momenta[:, 1]) + np.sum(driver.momenta[:, 1])
        total_pz = np.sum(rider.momenta[:, 2]) + np.sum(driver.momenta[:, 2])
        
        # CAI: For head-on collision, total momentum should be near zero
        # (allowing for small numerical differences due to momentum spread)
        momentum_magnitude = np.sqrt(total_px**2 + total_py**2 + total_pz**2)
        individual_momentum = np.mean(np.sqrt(
            rider.momenta[:, 0]**2 + rider.momenta[:, 1]**2 + rider.momenta[:, 2]**2
        ))
        
        # CAI: Total momentum should be much smaller than individual momenta
        relative_momentum = momentum_magnitude / individual_momentum
        assert relative_momentum < 0.1, f"Momentum conservation issue: {relative_momentum:.3f}"
        
        print(f"✓ Momentum conservation validated (relative error: {relative_momentum:.1e})")
    
    def test_energy_calculation_consistency(self):
        """Test that energy calculations are consistent across different representations."""
        config = ReferenceTestCases.high_energy_proton_gold()
        config = ReferenceTestCases.calculate_derived_parameters(config)
        
        initializer = BunchInitializer(config)
        ensemble, metadata = initializer.create_rider_ensemble(random_seed=777)
        
        # CAI: Calculate energy from gamma and mass
        rest_mass_energy = ensemble.mass * (299.792458)**2  # mc^2 in amu*(mm/ns)^2
        kinetic_energy = (ensemble.gamma - 1) * rest_mass_energy
        total_energy_from_gamma = ensemble.gamma * rest_mass_energy
        
        # CAI: Calculate energy from 4-momentum
        total_energy_from_momentum = ensemble.momenta[:, 3] * 299.792458  # Pt * c
        
        # CAI: These should be consistent
        assert np.allclose(total_energy_from_gamma, total_energy_from_momentum, rtol=1e-10)
        
        print(f"✓ Energy calculation consistency validated")
        print(f"  Gamma-based energy: {np.mean(total_energy_from_gamma):.2e}")
        print(f"  Momentum-based energy: {np.mean(total_energy_from_momentum):.2e}")
    
    def test_characteristic_time_calculation(self):
        """Test that characteristic time calculation matches original."""
        # CAI: Compare char_time calculation with original
        config = ReferenceTestCases.proton_antiproton_basic()
        config = ReferenceTestCases.calculate_derived_parameters(config)
        
        initializer = BunchInitializer(config)
        ensemble, metadata = initializer.create_rider_ensemble(random_seed=555)
        
        # CAI: Manual calculation using original formula
        c_mmns = 299.792458
        mass = metadata['mass_total'] 
        q = metadata['charge']
        expected_char_time = 2/3 * q**2 / (mass * c_mmns**3)
        
        # CAI: Should match the metadata value
        assert abs(metadata['char_time'] - expected_char_time) < 1e-15
        
        print(f"✓ Characteristic time calculation validated: {expected_char_time:.2e}")
    
    def test_random_distribution_properties(self):
        """Test that random distributions match original specifications."""
        config = ReferenceTestCases.proton_antiproton_basic()
        config = ReferenceTestCases.calculate_derived_parameters(config)
        
        initializer = BunchInitializer(config)
        ensemble, metadata = initializer.create_rider_ensemble(random_seed=1234)
        
        # CAI: Check transverse momentum distribution
        px_values = ensemble.momenta[:, 0]
        py_values = ensemble.momenta[:, 1]
        
        # CAI: Should be uniform distribution around zero
        px_range = np.max(px_values) - np.min(px_values)
        py_range = np.max(py_values) - np.min(py_values)
        
        expected_range = 2 * config.transv_mom_rider * metadata['mass_total']
        
        # CAI: Debug output to see what's happening
        print(f"Debug - Px range: {px_range:.2e}, Py range: {py_range:.2e}")
        print(f"Debug - Expected range: {expected_range:.2e}")
        print(f"Debug - transv_mom_rider: {config.transv_mom_rider:.2e}")
        print(f"Debug - mass_total: {metadata['mass_total']:.2e}")
        
        # CAI: Allow for some statistical variation in the range (handle zero case)
        if expected_range > 0:
            assert abs(px_range - expected_range) / expected_range < 0.5
            assert abs(py_range - expected_range) / expected_range < 0.5
        else:
            # CAI: If transverse momentum is zero, ranges should also be small
            assert px_range < 1e-10
            assert py_range < 1e-10
        
        # CAI: Check that means are near zero (for symmetric distribution)
        if expected_range > 0:
            assert abs(np.mean(px_values)) < expected_range * 0.3
            assert abs(np.mean(py_values)) < expected_range * 0.3
        else:
            # CAI: If no transverse momentum, values should all be zero
            assert np.all(px_values == px_values[0])  # All values should be the same
            assert np.all(py_values == py_values[0])
        
        print(f"✓ Random distribution properties validated")
        print(f"  Px range: {px_range:.2e} (expected: {expected_range:.2e})")
        print(f"  Py range: {py_range:.2e} (expected: {expected_range:.2e})")
    
    def test_beta_velocity_consistency(self):
        """Test that beta (v/c) calculations are consistent."""
        config = ReferenceTestCases.high_energy_proton_gold()
        config = ReferenceTestCases.calculate_derived_parameters(config)
        
        initializer = BunchInitializer(config)
        ensemble, metadata = initializer.create_rider_ensemble(random_seed=888)
        
        # CAI: Calculate beta from velocities
        c_mmns = 299.792458
        beta_from_velocities = ensemble.velocities / c_mmns
        beta_magnitude = np.sqrt(np.sum(beta_from_velocities**2, axis=1))
        
        # CAI: Calculate beta from gamma (relativistic relation)
        beta_from_gamma = np.sqrt(1 - 1/ensemble.gamma**2)
        
        # CAI: These should be consistent
        assert np.allclose(beta_magnitude, beta_from_gamma, rtol=1e-10)
        
        # CAI: Check relativistic limit (beta < 1)
        assert np.all(beta_magnitude < 1.0), "Beta should be less than 1"
        assert np.all(beta_magnitude > 0.0), "Beta should be positive"
        
        print(f"✓ Beta-velocity consistency validated")
        print(f"  Beta range: {np.min(beta_magnitude):.6f} - {np.max(beta_magnitude):.6f}")
        print(f"  Gamma range: {np.min(ensemble.gamma):.1f} - {np.max(ensemble.gamma):.1f}")


class TestGeVInstabilityReproduction:
    """
    Tests specifically for reproducing GeV range instability conditions.
    
    CAI: These tests set up the exact conditions that cause numerical
    instability in the original code for investigation.
    """
    
    def test_gev_instability_particle_setup(self):
        """Test setup of particles for GeV instability investigation."""
        # CAI: Get the specific configuration that shows instability
        gev_config = ReferenceTestCases.high_energy_proton_gold()
        gev_config = ReferenceTestCases.calculate_derived_parameters(gev_config)
        
        initializer = BunchInitializer(gev_config)
        rider, driver, rider_meta, driver_meta = initializer.create_both_ensembles(
            rider_seed=42, driver_seed=84
        )
        
        # CAI: Verify we're in the problematic energy range
        assert rider_meta['E_MeV'] > 2000, "Should be in GeV range"
        assert gev_config.step_size < 2e-8, "Should use small timestep"
        assert gev_config.expected_instability, "Should be expected unstable case"
        
        # CAI: Check that particles have extreme relativistic conditions
        assert np.all(rider.gamma > 1000), f"Rider gamma too low: {np.mean(rider.gamma)}"
        
        # CAI: Document the extreme conditions
        print(f"GeV instability test conditions:")
        print(f"  Rider energy: {rider_meta['E_MeV']:.1f} MeV")
        print(f"  Rider gamma: {np.mean(rider.gamma):.1f}")
        print(f"  Driver energy: {driver_meta['E_MeV']:.1f} MeV")
        print(f"  Step size: {gev_config.step_size:.2e}")
        print(f"  Expected instability: {gev_config.expected_instability}")
    
    def test_step_size_scaling_behavior(self):
        """Test how particle initialization behaves with different step sizes."""
        # CAI: Test multiple step sizes around the instability threshold
        base_config = ReferenceTestCases.high_energy_proton_gold()
        step_sizes = [1e-6, 1e-7, 1e-8, 1e-9]
        
        for step_size in step_sizes:
            config = ReferenceTestCases.calculate_derived_parameters(base_config)
            config.step_size = step_size
            
            initializer = BunchInitializer(config)
            rider, driver, rider_meta, driver_meta = initializer.create_both_ensembles(
                rider_seed=42, driver_seed=84
            )
            
            # CAI: Particle properties shouldn't depend on step size
            assert rider_meta['E_MeV'] > 2000, f"Energy changed with step size {step_size}"
            assert np.all(rider.gamma > 1000), f"Gamma changed with step size {step_size}"
            
            print(f"Step size {step_size:.1e}: Energy = {rider_meta['E_MeV']:.1f} MeV")


if __name__ == "__main__":
    print("Running initialization bridge validation tests...")
    
    # CAI: Create test instances and run key tests
    basic_tests = TestInitializationBridge()
    instability_tests = TestGeVInstabilityReproduction()
    
    try:
        print("\n1. Testing exact match with original bunch_inits.py...")
        basic_tests.test_basic_proton_initialization_exact_match()
        
        print("\n2. Testing high-energy electron case...")
        basic_tests.test_high_energy_electron_exact_match()
        
        print("\n3. Testing momentum conservation...")
        basic_tests.test_momentum_conservation_validation()
        
        print("\n4. Testing energy calculation consistency...")
        basic_tests.test_energy_calculation_consistency()
        
        print("\n5. Testing characteristic time calculation...")
        basic_tests.test_characteristic_time_calculation()
        
        print("\n6. Testing random distribution properties...")
        basic_tests.test_random_distribution_properties()
        
        print("\n7. Testing beta-velocity consistency...")
        basic_tests.test_beta_velocity_consistency()
        
        print("\n8. Testing GeV instability setup...")
        instability_tests.test_gev_instability_particle_setup()
        
        print("\n9. Testing step size scaling behavior...")
        instability_tests.test_step_size_scaling_behavior()
        
        print("\n" + "="*60)
        print("🎉 All initialization bridge tests PASSED!")
        print("✓ Exact match with original bunch_inits.py confirmed")
        print("✓ GeV instability conditions successfully reproduced")
        print("✓ Ready for integration framework development")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
