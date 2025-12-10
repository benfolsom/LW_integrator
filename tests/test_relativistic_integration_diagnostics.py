"""
Diagnostic tests for relativistic charged particle integration issues.

This module provides comprehensive tests to diagnose blowups and instabilities
in the relativistic Lienard-Wiechert integrator, with focus on:
1. Self-consistency between γ_energy and γ_velocity
2. Timestep refinement effectiveness
3. Ultra-relativistic particle behavior
4. Energy conservation and mass-shell constraint
"""

import numpy as np
import pytest
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from core.trajectory_integrator import retarded_integrator
from core.types import SimulationType, ParticleState
from core.self_consistency import SelfConsistencyConfig
from core.integration_runner import EnergyMonitorConfig, AdaptiveTimestepConfig
from legacy.bunch_inits import init_bunch


class IntegrationDiagnostics:
    """Comprehensive diagnostics for integration quality."""
    
    @staticmethod
    def compute_gamma_from_velocity(state: ParticleState, idx: int = 0) -> float:
        """Compute γ from velocity: γ = 1/√(1-β²)"""
        bx = state["bx"][idx]
        by = state["by"][idx]
        bz = state["bz"][idx]
        beta_sq = bx**2 + by**2 + bz**2
        
        if beta_sq >= 1.0:
            return np.inf
        
        return 1.0 / np.sqrt(1.0 - beta_sq)
    
    @staticmethod
    def compute_gamma_from_energy(state: ParticleState, idx: int = 0) -> float:
        """Compute γ from energy: γ = Pt/(mc)"""
        c_mmns = 299.792458  # mm/ns
        mass = state.get("m", np.array([0.00054857990907]))
        if hasattr(mass, "__getitem__"):
            mass = mass[idx]
        
        return state["Pt"][idx] / (mass * c_mmns)
    
    @staticmethod
    def compute_mass_shell_error(state: ParticleState, idx: int = 0) -> float:
        """
        Compute mass-shell constraint error: (P_μ P^μ + m²c²) / m²c²
        
        For a free particle, should satisfy: P_t² - P_x² - P_y² - P_z² = (mc)²
        """
        c_mmns = 299.792458
        mass = state.get("m", np.array([0.00054857990907]))
        if hasattr(mass, "__getitem__"):
            mass = mass[idx]
        
        Pt = state["Pt"][idx]
        Px = state["Px"][idx]
        Py = state["Py"][idx]
        Pz = state["Pz"][idx]
        
        mass_shell_lhs = Pt**2 - Px**2 - Py**2 - Pz**2
        mass_shell_rhs = (mass * c_mmns)**2
        
        if mass_shell_rhs > 0:
            return (mass_shell_lhs - mass_shell_rhs) / mass_shell_rhs
        return np.inf
    
    @staticmethod
    def compute_gamma_discrepancy(state: ParticleState, idx: int = 0) -> Tuple[float, float, float]:
        """
        Compute discrepancy between γ_energy and γ_velocity.
        
        Returns:
            (γ_energy, γ_velocity, relative_error)
        """
        gamma_e = IntegrationDiagnostics.compute_gamma_from_energy(state, idx)
        gamma_v = IntegrationDiagnostics.compute_gamma_from_velocity(state, idx)
        
        if gamma_e > 0:
            rel_error = abs(gamma_e - gamma_v) / gamma_e
        else:
            rel_error = np.inf
        
        return gamma_e, gamma_v, rel_error
    
    @staticmethod
    def compute_energy(state: ParticleState, idx: int = 0) -> float:
        """Compute total energy from Pt."""
        return state["Pt"][idx]
    
    @staticmethod
    def compute_beta_magnitude(state: ParticleState, idx: int = 0) -> float:
        """Compute |β| = √(βx² + βy² + βz²)"""
        bx = state["bx"][idx]
        by = state["by"][idx]
        bz = state["bz"][idx]
        return np.sqrt(bx**2 + by**2 + bz**2)
    
    @staticmethod
    def analyze_trajectory(trajectory: List[ParticleState], 
                          verbose: bool = False) -> Dict:
        """
        Analyze entire trajectory for diagnostic metrics.
        
        Returns dictionary with:
        - gamma_energy: array of γ from energy
        - gamma_velocity: array of γ from velocity
        - gamma_discrepancy: relative errors
        - mass_shell_errors: mass-shell constraint violations
        - energies: total energy at each step
        - beta_magnitudes: |β| at each step
        - max_gamma_discrepancy: worst γ mismatch
        - max_mass_shell_error: worst mass-shell violation
        """
        n_steps = len(trajectory)
        
        gamma_energy = np.zeros(n_steps)
        gamma_velocity = np.zeros(n_steps)
        gamma_discrepancy = np.zeros(n_steps)
        mass_shell_errors = np.zeros(n_steps)
        energies = np.zeros(n_steps)
        beta_magnitudes = np.zeros(n_steps)
        
        for i, state in enumerate(trajectory):
            gamma_e, gamma_v, rel_err = IntegrationDiagnostics.compute_gamma_discrepancy(state)
            gamma_energy[i] = gamma_e
            gamma_velocity[i] = gamma_v
            gamma_discrepancy[i] = rel_err
            mass_shell_errors[i] = IntegrationDiagnostics.compute_mass_shell_error(state)
            energies[i] = IntegrationDiagnostics.compute_energy(state)
            beta_magnitudes[i] = IntegrationDiagnostics.compute_beta_magnitude(state)
            
            if verbose and (i % 100 == 0 or rel_err > 0.01):
                print(f"Step {i}: γ_e={gamma_e:.6f}, γ_v={gamma_v:.6f}, "
                      f"rel_err={rel_err:.6e}, E={energies[i]:.6f}, β={beta_magnitudes[i]:.6f}")
        
        return {
            "gamma_energy": gamma_energy,
            "gamma_velocity": gamma_velocity,
            "gamma_discrepancy": gamma_discrepancy,
            "mass_shell_errors": mass_shell_errors,
            "energies": energies,
            "beta_magnitudes": beta_magnitudes,
            "max_gamma_discrepancy": np.max(gamma_discrepancy),
            "max_mass_shell_error": np.max(np.abs(mass_shell_errors)),
            "final_energy_change": abs(energies[-1] - energies[0]) / energies[0] if energies[0] > 0 else np.inf,
            "mean_gamma_discrepancy": np.mean(gamma_discrepancy),
            "std_gamma_discrepancy": np.std(gamma_discrepancy),
        }


class TestRelativisticIntegrationDiagnostics:
    """Test suite for diagnosing relativistic integration issues."""
    
    @pytest.fixture
    def electron_mass(self):
        """Electron rest mass in MeV."""
        return 0.00054857990907  # MeV
    
    @pytest.fixture
    def c_mmns(self):
        """Speed of light in mm/ns."""
        return 299.792458
    
    def create_ultra_relativistic_electron(self, 
                                          energy_mev: float,
                                          electron_mass: float,
                                          c_mmns: float,
                                          transverse_momentum_mev: float = 1e-5) -> ParticleState:
        """
        Create initial state for ultra-relativistic electron.
        
        Args:
            energy_mev: Total energy in MeV
            electron_mass: Rest mass in MeV
            c_mmns: Speed of light in mm/ns
            transverse_momentum_mev: Small transverse momentum in MeV
        """
        # For ultra-relativistic: E ≈ pc, γ = E/(mc²)
        gamma = energy_mev / (electron_mass * c_mmns**2)
        
        # Compute momentum components
        # Pt = γmc (time component)
        Pt = gamma * electron_mass * c_mmns
        
        # Small transverse momenta
        Px = transverse_momentum_mev / c_mmns
        Py = transverse_momentum_mev / c_mmns
        
        # Longitudinal momentum from energy-momentum relation
        # Pt² = Px² + Py² + Pz² + (mc)²
        Pz_sq = Pt**2 - Px**2 - Py**2 - (electron_mass * c_mmns)**2
        Pz = np.sqrt(max(0, Pz_sq))
        
        # Compute velocities: β = P / (γmc)
        bx = Px / (gamma * electron_mass * c_mmns)
        by = Py / (gamma * electron_mass * c_mmns)
        bz = Pz / (gamma * electron_mass * c_mmns)
        
        return {
            "x": np.array([2e-6]),
            "y": np.array([2e-6]),
            "z": np.array([1e-6]),
            "t": np.array([0.0]),
            "Px": np.array([Px]),
            "Py": np.array([Py]),
            "Pz": np.array([Pz]),
            "Pt": np.array([Pt]),
            "gamma": np.array([gamma]),
            "bx": np.array([bx]),
            "by": np.array([by]),
            "bz": np.array([bz]),
            "bdotx": np.array([0.0]),
            "bdoty": np.array([0.0]),
            "bdotz": np.array([0.0]),
            "q": np.array([-1.0]),
            "m": np.array([electron_mass]),
            "char_time": np.array([1e-20]),
            "beta_samples": np.array([0.0]),
            "beta_avg_x": np.array([0.0]),
            "beta_avg_y": np.array([0.0]),
            "beta_avg_z": np.array([0.0]),
        }
    
    def test_ultra_relativistic_electron_10gev_baseline(self, electron_mass, c_mmns):
        """
        Test 10 GeV electron with baseline settings (no self-consistency).
        
        This test diagnoses behavior at γ ~ 20,000 without iterative refinement.
        """
        energy_mev = 10000.0  # 10 GeV
        gamma_expected = energy_mev / (electron_mass * c_mmns**2)
        
        print(f"\n=== 10 GeV Electron Baseline Test ===")
        print(f"Expected γ: {gamma_expected:.1f}")
        
        init_state = self.create_ultra_relativistic_electron(
            energy_mev, electron_mass, c_mmns
        )
        
        # Verify initial state consistency
        gamma_e0, gamma_v0, err0 = IntegrationDiagnostics.compute_gamma_discrepancy(init_state)
        print(f"Initial: γ_energy={gamma_e0:.1f}, γ_velocity={gamma_v0:.1f}, rel_err={err0:.2e}")
        
        # Run short integration without self-consistency
        trajectory, _ = retarded_integrator(
            steps=100,
            h_step=1e-7,  # 0.1 ns
            wall_z=2200.0,
            aperture_radius=0.05,
            sim_type=SimulationType.CONDUCTING_WALL,
            init_rider=init_state,
            init_driver=None,
            mean=100000.0,
            cav_spacing=0.0,
            z_cutoff=0.0,
            self_consistency=None,  # Disabled for baseline
        )
        
        # Analyze trajectory
        analysis = IntegrationDiagnostics.analyze_trajectory(trajectory, verbose=True)
        
        print(f"\n=== Trajectory Analysis (100 steps) ===")
        print(f"Max γ discrepancy: {analysis['max_gamma_discrepancy']:.6e}")
        print(f"Mean γ discrepancy: {analysis['mean_gamma_discrepancy']:.6e}")
        print(f"Max mass-shell error: {analysis['max_mass_shell_error']:.6e}")
        print(f"Final energy change: {analysis['final_energy_change']:.6e}")
        
        # Diagnostic assertions (not strict failures, but warnings)
        if analysis['max_gamma_discrepancy'] > 0.01:
            print(f"WARNING: Large γ discrepancy detected: {analysis['max_gamma_discrepancy']:.6e}")
        
        if analysis['max_mass_shell_error'] > 1e-6:
            print(f"WARNING: Mass-shell constraint violated: {analysis['max_mass_shell_error']:.6e}")
        
        # Check for blowup
        if np.any(np.isnan(analysis['gamma_energy'])) or np.any(np.isinf(analysis['gamma_energy'])):
            pytest.fail("Blowup detected: NaN or Inf in gamma_energy")
        
        if analysis['max_gamma_discrepancy'] > 10.0:
            pytest.fail(f"Severe γ discrepancy: {analysis['max_gamma_discrepancy']:.6e}")
    
    def test_ultra_relativistic_electron_10gev_with_self_consistency(self, electron_mass, c_mmns):
        """
        Test 10 GeV electron WITH self-consistency enabled.
        
        This test checks if iterative refinement helps maintain γ_energy ≈ γ_velocity.
        """
        energy_mev = 10000.0
        
        print(f"\n=== 10 GeV Electron with Self-Consistency ===")
        
        init_state = self.create_ultra_relativistic_electron(
            energy_mev, electron_mass, c_mmns
        )
        
        sc_config = SelfConsistencyConfig(
            enabled=True,
            tolerance=1e-9,
            max_iterations=10,
            verbosity=1,
        )
        
        trajectory, _ = retarded_integrator(
            steps=100,
            h_step=1e-7,
            wall_z=2200.0,
            aperture_radius=0.05,
            sim_type=SimulationType.CONDUCTING_WALL,
            init_rider=init_state,
            init_driver=None,
            mean=100000.0,
            cav_spacing=0.0,
            z_cutoff=0.0,
            self_consistency=sc_config,
        )
        
        analysis = IntegrationDiagnostics.analyze_trajectory(trajectory, verbose=True)
        
        print(f"\n=== Analysis with Self-Consistency ===")
        print(f"Max γ discrepancy: {analysis['max_gamma_discrepancy']:.6e}")
        print(f"Mean γ discrepancy: {analysis['mean_gamma_discrepancy']:.6e}")
        
        # Self-consistency should reduce discrepancy
        if analysis['max_gamma_discrepancy'] > sc_config.tolerance * 100:
            print(f"WARNING: Self-consistency not achieving target tolerance")
    
    def test_adaptive_timestep_effectiveness(self, electron_mass, c_mmns):
        """
        Test whether adaptive timestep refinement prevents blowups.
        
        Use aggressive settings to trigger refinement.
        """
        energy_mev = 10000.0
        
        print(f"\n=== Adaptive Timestep Test ===")
        
        init_state = self.create_ultra_relativistic_electron(
            energy_mev, electron_mass, c_mmns
        )
        
        adaptive_config = AdaptiveTimestepConfig(
            enabled=True,
            energy_jump_threshold=0.1,  # 10% energy jump triggers refinement
            timestep_reduction_factor=10,
            max_refinement_attempts=5,
            min_timestep_factor=0.0001,
            cooldown_steps=10,
            max_probe_steps=5,
            probe_threshold=0.01,
            debug=True,
        )
        
        trajectory, _ = retarded_integrator(
            steps=200,
            h_step=1e-7,
            wall_z=2200.0,
            aperture_radius=0.05,
            sim_type=SimulationType.CONDUCTING_WALL,
            init_rider=init_state,
            init_driver=None,
            mean=100000.0,
            cav_spacing=0.0,
            z_cutoff=0.0,
            adaptive_timestep=adaptive_config,
        )
        
        analysis = IntegrationDiagnostics.analyze_trajectory(trajectory, verbose=False)
        
        print(f"\n=== Adaptive Timestep Results ===")
        print(f"Final energy change: {analysis['final_energy_change']:.6e}")
        print(f"Max γ discrepancy: {analysis['max_gamma_discrepancy']:.6e}")
        
        # Check if adaptive timestep maintained stability
        if analysis['final_energy_change'] > 0.5:
            print(f"WARNING: Large energy drift despite adaptive timestep")
    
    def test_gamma_sweep(self, electron_mass, c_mmns):
        """
        Sweep through different γ values to identify problematic regimes.
        
        Tests: γ = 2, 10, 100, 1000, 10000
        """
        gamma_values = [2, 10, 100, 1000, 10000]
        
        print(f"\n=== Gamma Sweep Test ===")
        
        results = []
        
        for gamma in gamma_values:
            energy_mev = gamma * electron_mass * c_mmns**2
            
            init_state = self.create_ultra_relativistic_electron(
                energy_mev, electron_mass, c_mmns
            )
            
            trajectory, _ = retarded_integrator(
                steps=50,
                h_step=1e-7,
                wall_z=2200.0,
                aperture_radius=0.05,
                sim_type=SimulationType.CONDUCTING_WALL,
                init_rider=init_state,
                init_driver=None,
                mean=100000.0,
                cav_spacing=0.0,
                z_cutoff=0.0,
                self_consistency=None,
            )
            
            analysis = IntegrationDiagnostics.analyze_trajectory(trajectory, verbose=False)
            
            results.append({
                "gamma": gamma,
                "max_gamma_discrepancy": analysis['max_gamma_discrepancy'],
                "max_mass_shell_error": analysis['max_mass_shell_error'],
                "final_energy_change": analysis['final_energy_change'],
            })
            
            print(f"γ={gamma:5d}: max_γ_err={analysis['max_gamma_discrepancy']:.6e}, "
                  f"mass_shell_err={analysis['max_mass_shell_error']:.6e}, "
                  f"ΔE/E={analysis['final_energy_change']:.6e}")
        
        # Check for scaling issues with γ
        for res in results:
            if res['gamma'] > 1000 and res['max_gamma_discrepancy'] > 0.1:
                print(f"WARNING: Issues at high γ={res['gamma']}")
    
    def test_timestep_scaling(self, electron_mass, c_mmns):
        """
        Test behavior with different timestep sizes.
        
        For ultra-relativistic particles, need Δt << 1/ωp where ωp is plasma frequency.
        """
        energy_mev = 10000.0
        timesteps = [1e-6, 1e-7, 1e-8, 1e-9]  # 1 ns to 0.001 ns
        
        print(f"\n=== Timestep Scaling Test ===")
        
        for h_step in timesteps:
            init_state = self.create_ultra_relativistic_electron(
                energy_mev, electron_mass, c_mmns
            )
            
            try:
                trajectory, _ = retarded_integrator(
                    steps=min(100, int(1e-5 / h_step)),  # Integrate to ~10 ns total time
                    h_step=h_step,
                    wall_z=2200.0,
                    aperture_radius=0.05,
                    sim_type=SimulationType.CONDUCTING_WALL,
                    init_rider=init_state,
                    init_driver=None,
                    mean=100000.0,
                    cav_spacing=0.0,
                    z_cutoff=0.0,
                )
                
                analysis = IntegrationDiagnostics.analyze_trajectory(trajectory, verbose=False)
                
                print(f"Δτ={h_step:.1e} ns: max_γ_err={analysis['max_gamma_discrepancy']:.6e}, "
                      f"ΔE/E={analysis['final_energy_change']:.6e}")
                
            except Exception as e:
                print(f"Δτ={h_step:.1e} ns: FAILED - {str(e)}")
    
    def test_position_velocity_consistency(self, electron_mass, c_mmns):
        """
        Test that position updates are consistent with velocity.
        
        Check: Δx = v·Δt = β·c·Δt where Δt is coordinate time
        """
        energy_mev = 10000.0
        
        print(f"\n=== Position-Velocity Consistency Test ===")
        
        init_state = self.create_ultra_relativistic_electron(
            energy_mev, electron_mass, c_mmns
        )
        
        trajectory, _ = retarded_integrator(
            steps=10,
            h_step=1e-7,
            wall_z=2200.0,
            aperture_radius=0.05,
            sim_type=SimulationType.CONDUCTING_WALL,
            init_rider=init_state,
            init_driver=None,
            mean=100000.0,
            cav_spacing=0.0,
            z_cutoff=0.0,
        )
        
        for i in range(1, len(trajectory)):
            prev = trajectory[i-1]
            curr = trajectory[i]
            
            # Compute position change
            dx = curr["x"][0] - prev["x"][0]
            dy = curr["y"][0] - prev["y"][0]
            dz = curr["z"][0] - prev["z"][0]
            
            # Compute time change (coordinate time)
            dt = curr["t"][0] - prev["t"][0]
            
            # Compute expected position change from velocity
            dx_expected = prev["bx"][0] * c_mmns * dt
            dy_expected = prev["by"][0] * c_mmns * dt
            dz_expected = prev["bz"][0] * c_mmns * dt
            
            # Compute relative errors
            rel_err_x = abs(dx - dx_expected) / (abs(dx) + 1e-20)
            rel_err_y = abs(dy - dy_expected) / (abs(dy) + 1e-20)
            rel_err_z = abs(dz - dz_expected) / (abs(dz) + 1e-20)
            
            if i <= 3 or rel_err_z > 0.01:
                print(f"Step {i}: Δz={dz:.6e}, Δz_expected={dz_expected:.6e}, "
                      f"rel_err={rel_err_z:.6e}, Δt={dt:.6e}")
            
            if rel_err_z > 0.1:
                print(f"WARNING: Large position-velocity inconsistency at step {i}")
    
    def test_beta_evolution_detailed(self, electron_mass, c_mmns):
        """
        Detailed test tracking beta evolution to understand when β ≥ 1.
        
        This is a simplified version to track the exact step where β becomes problematic.
        """
        print(f"\n=== Beta Evolution Detailed Test ===")
        
        # Use same config as gamma_error test but with more diagnostics
        starting_Pz = 6123000.0
        transv_mom = 1.2e-05
        
        init_state, _ = init_bunch(
            starting_distance=1e-6,
            transv_mom=transv_mom,
            starting_Pz=starting_Pz,
            stripped_ions=1.0,
            m_particle=electron_mass,
            transv_dist=2e-6,
            pcount=1,
            charge_sign=-1.0,
        )
        
        trajectory, _ = retarded_integrator(
            steps=1300,  # Run up to where problem occurs
            h_step=3e-7,
            wall_z=2200.0,
            aperture_radius=0.06,
            sim_type=SimulationType.CONDUCTING_WALL,
            init_rider=init_state,
            init_driver=None,
            mean=100000.0,
            cav_spacing=100000.0,
            z_cutoff=0.0,
        )
        
        # Track when β approaches 1
        print(f"\nTracking β evolution (showing steps where β² > 0.9999):")
        problem_step = None
        for i, state in enumerate(trajectory):
            beta_mag = IntegrationDiagnostics.compute_beta_magnitude(state)
            beta_sq = beta_mag**2
            
            if beta_sq > 0.9999 or i < 5 or (i > 1195 and i < 1210):
                gamma_e, gamma_v, rel_err = IntegrationDiagnostics.compute_gamma_discrepancy(state)
                print(f"Step {i:4d}: β²={beta_sq:.10f}, β={beta_mag:.10f}, "
                      f"γ_e={gamma_e:.1f}, γ_v={gamma_v:.1f}, err={rel_err:.6e}")
                
                if beta_sq >= 1.0 and problem_step is None:
                    problem_step = i
                    print(f"*** PROBLEM: β² ≥ 1 detected at step {i} ***")
        
        if problem_step is not None:
            print(f"\n*** β reached or exceeded c at step {problem_step} ***")
            print("This is the root cause of the γ discrepancy blowup.")
            pytest.fail(f"Beta velocity exceeded speed of light at step {problem_step}")
    
    def test_gamma_error_config_reproduction(self, electron_mass, c_mmns):
        """
        Test using the exact config that was labeled as having gamma errors.
        
        This reproduces the electronwall10.3_0.06mm10_gev_gammaerror.json config
        to see if we can trigger the reported blowup issues.
        
        Note: The config is mislabeled - it's actually ~1 TeV, not 10 GeV.
        """
        print(f"\n=== Gamma Error Config Reproduction Test ===")
        
        # Exact parameters from electronwall10.3_0.06mm10_gev_gammaerror.json
        # These match the config file exactly
        starting_Pz = 6123000.0  # This is velocity in mm/ns, not momentum!
        transv_mom = 1.2e-05  # This gets multiplied by mass in legacy code
        
        # Use the legacy init_bunch function to get proper unit conversions
        init_state, E_MeV_rest = init_bunch(
            starting_distance=1e-6,
            transv_mom=transv_mom,
            starting_Pz=starting_Pz,
            stripped_ions=1.0,
            m_particle=electron_mass,
            transv_dist=2e-6,
            pcount=1,
            charge_sign=-1.0,
            verbose=True,
        )
        
        # Extract gamma and energy from initialized state
        gamma_init = init_state["gamma"][0]
        Pt_init = init_state["Pt"][0]
        E_total = Pt_init * c_mmns
        
        print(f"\nConfig parameters (after legacy init):")
        print(f"  starting_Pz (velocity): {starting_Pz:.1f} mm/ns")
        print(f"  Transverse momentum param: {transv_mom:.2e}")
        print(f"  Initial γ = {gamma_init:.1f}")
        print(f"  Initial E = {E_total:.1f} MeV")
        
        # Verify initial consistency
        gamma_e0, gamma_v0, err0 = IntegrationDiagnostics.compute_gamma_discrepancy(init_state)
        print(f"\nInitial state check: γ_energy={gamma_e0:.1f}, γ_velocity={gamma_v0:.1f}, rel_err={err0:.2e}")
        
        if err0 > 1e-6:
            print(f"WARNING: Initial state has significant γ mismatch: {err0:.2e}")
            print("This indicates the initial state construction has issues.")
        
        # Run with config parameters (1700 steps, 0.3 ns timestep)
        # Use adaptive timestep as in the config
        adaptive_config = AdaptiveTimestepConfig(
            enabled=True,
            energy_jump_threshold=0.1,
            timestep_reduction_factor=10,
            max_refinement_attempts=5,
            min_timestep_factor=0.05,
            cooldown_steps=10,
            max_probe_steps=3,
            probe_threshold=0.01,
            debug=True,
        )
        
        try:
            trajectory, _ = retarded_integrator(
                steps=1700,  # Full run as in config
                h_step=3e-7,  # 0.3 ns as in config
                wall_z=2200.0,
                aperture_radius=0.06,  # 0.06 mm as in config
                sim_type=SimulationType.CONDUCTING_WALL,
                init_rider=init_state,
                init_driver=None,
                mean=100000.0,
                cav_spacing=100000.0,
                z_cutoff=0.0,
                adaptive_timestep=adaptive_config,
            )
            
            # Analyze trajectory
            analysis = IntegrationDiagnostics.analyze_trajectory(trajectory, verbose=False)
            
            print(f"\n=== Full 1700-step Analysis ===")
            print(f"Max γ discrepancy: {analysis['max_gamma_discrepancy']:.6e}")
            print(f"Mean γ discrepancy: {analysis['mean_gamma_discrepancy']:.6e}")
            print(f"Std γ discrepancy: {analysis['std_gamma_discrepancy']:.6e}")
            print(f"Max mass-shell error: {analysis['max_mass_shell_error']:.6e}")
            print(f"Final energy change: {analysis['final_energy_change']:.6e}")
            
            # Check for blowup indicators
            # Find worst steps first (before failing)
            worst_indices = np.argsort(analysis['gamma_discrepancy'])[-5:]
            print(f"\nWorst 5 steps by γ discrepancy:")
            for idx in worst_indices:
                print(f"  Step {idx}: γ_err={analysis['gamma_discrepancy'][idx]:.6e}, "
                      f"E={analysis['energies'][idx]:.6f}, β={analysis['beta_magnitudes'][idx]:.6f}")
            
            # Now check for failures
            if np.any(np.isnan(analysis['gamma_energy'])) or np.any(np.isinf(analysis['gamma_energy'])):
                pytest.fail("Blowup detected: NaN or Inf in gamma_energy")
            
            if analysis['max_gamma_discrepancy'] > 1.0:
                print(f"\n*** FAILED: Large γ discrepancy detected: {analysis['max_gamma_discrepancy']:.6e}")
                pytest.fail(f"Severe γ discrepancy exceeds threshold: {analysis['max_gamma_discrepancy']:.6e}")
            
            if analysis['final_energy_change'] > 0.1:
                print(f"WARNING: Significant energy drift: {analysis['final_energy_change']:.6e}")
                      
        except Exception as e:
            print(f"ERROR: Integration failed with: {str(e)}")
            pytest.fail(f"Integration crashed: {str(e)}")

    def test_missing_gamma_factor_in_position_update(self, electron_mass, c_mmns):
        """
        Test to demonstrate the missing 1/γ factor bug in position updates.
        
        The position update equation should be:
            Δx = (P_kinetic / (γ·m))·h
        
        But the current implementation is:
            Δx = (P_kinetic / m)·h    [MISSING 1/γ!]
        
        This causes Δx to be γ times too large at ultra-relativistic speeds,
        making β approach 1.0 even though momentum-based velocity is correct.
        """
        print(f"\n=== Missing 1/γ Factor Demonstration ===")
        
        # Use ultra-relativistic electron
        starting_Pz = 6123000.0  # velocity in mm/ns
        transv_mom = 1.2e-05
        
        init_state, _ = init_bunch(
            starting_distance=1e-6,
            transv_mom=transv_mom,
            starting_Pz=starting_Pz,
            stripped_ions=1.0,
            m_particle=electron_mass,
            transv_dist=2e-6,
            pcount=1,
            charge_sign=-1.0,
        )
        
        gamma_init = init_state["gamma"][0]
        print(f"Initial γ = {gamma_init:.1f}")
        print(f"Initial β = {np.sqrt(init_state['bx'][0]**2 + init_state['by'][0]**2 + init_state['bz'][0]**2):.10f}")
        
        # Run a short integration to check position updates
        trajectory, _ = retarded_integrator(
            steps=5,
            h_step=3e-7,
            wall_z=2200.0,
            aperture_radius=0.06,
            sim_type=SimulationType.CONDUCTING_WALL,
            init_rider=init_state,
            init_driver=None,
            mean=100000.0,
            cav_spacing=100000.0,
            z_cutoff=0.0,
        )
        
        # Analyze position updates
        print(f"\nAnalyzing position updates (should have 1/γ factor):")
        for i in range(1, min(5, len(trajectory))):
            prev = trajectory[i-1]
            curr = trajectory[i]
            
            h = 3e-7  # timestep in proper time
            gamma_prev = prev["gamma"][0]
            
            # Kinetic momentum (P - qA)
            Px_kin = prev["Px"][0]  # Assuming no vector potential
            Pz_kin = prev["Pz"][0]
            
            # Actual position change
            dx_actual = curr["x"][0] - prev["x"][0]
            dz_actual = curr["z"][0] - prev["z"][0]
            
            # What position change SHOULD be (with 1/γ factor)
            dx_correct = h / (gamma_prev * electron_mass) * Px_kin
            dz_correct = h / (gamma_prev * electron_mass) * Pz_kin
            
            # What it would be WITHOUT 1/γ factor (current bug)
            dx_buggy = h / electron_mass * Px_kin
            dz_buggy = h / electron_mass * Pz_kin
            
            # Check which one matches
            error_if_correct = abs(dz_actual - dz_correct)
            error_if_buggy = abs(dz_actual - dz_buggy)
            
            print(f"\nStep {i-1} → {i}:")
            print(f"  γ = {gamma_prev:.1f}")
            print(f"  Δz_actual = {dz_actual:.10e} mm")
            print(f"  Δz_correct (with 1/γ) = {dz_correct:.10e} mm")
            print(f"  Δz_buggy (no 1/γ) = {dz_buggy:.10e} mm")
            print(f"  Error if correct: {error_if_correct:.3e}")
            print(f"  Error if buggy:   {error_if_buggy:.3e}")
            
            # Derived β from position change
            dt_coord = curr["t"][0] - prev["t"][0]
            beta_from_position = dz_actual / (c_mmns * dt_coord)
            
            # Correct β from momentum
            beta_from_momentum = Pz_kin / (gamma_prev * electron_mass * c_mmns)
            
            print(f"  β from position (Δz/cΔt) = {beta_from_position:.10f}")
            print(f"  β from momentum (Pz/γmc) = {beta_from_momentum:.10f}")
            print(f"  β ratio (position/momentum) = {beta_from_position/beta_from_momentum:.6f}")
            
            if error_if_buggy < error_if_correct:
                print(f"  *** BUG CONFIRMED: Position update is MISSING 1/γ factor! ***")
                print(f"  *** This makes Δx be {gamma_prev:.1f}× too large ***")
                print(f"  *** Therefore β = Δx/(cΔt) is {gamma_prev:.1f}× too large ***")
            else:
                print(f"  ✓ Position update correctly includes 1/γ factor")
        
        # Calculate the ratio that proves the bug
        final_state = trajectory[-1]
        beta_pos = np.sqrt(final_state["bx"][0]**2 + final_state["by"][0]**2 + final_state["bz"][0]**2)
        
        # What β should be from momentum
        Px = final_state["Px"][0]
        Py = final_state["Py"][0]
        Pz = final_state["Pz"][0]
        gamma_final = final_state["gamma"][0]
        P_mag = np.sqrt(Px**2 + Py**2 + Pz**2)
        beta_momentum = P_mag / (gamma_final * electron_mass * c_mmns)
        
        print(f"\n=== Summary ===")
        print(f"β from position updates: {beta_pos:.10f}")
        print(f"β from momentum (correct): {beta_momentum:.10f}")
        print(f"Ratio β_pos/β_momentum = {beta_pos/beta_momentum:.6f}")
        print(f"Expected ratio if bug present: ~{gamma_final:.1f}")
        
        if abs(beta_pos / beta_momentum - gamma_final) < gamma_final * 0.01:
            print(f"\n*** BUG CONFIRMED: β is approximately γ times too large! ***")
            print(f"*** This is because position update is missing 1/γ factor ***")
            pytest.fail(
                f"Position update missing 1/γ factor: β is {beta_pos/beta_momentum:.1f}× too large (expected ~{gamma_final:.1f}×)"
            )


def save_diagnostic_report(analysis: Dict, filename: str):
    """Save diagnostic analysis to JSON file."""
    # Convert numpy arrays to lists for JSON serialization
    report = {}
    for key, value in analysis.items():
        if isinstance(value, np.ndarray):
            report[key] = value.tolist()
        else:
            report[key] = value
    
    with open(filename, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Diagnostic report saved to {filename}")


if __name__ == "__main__":
    # Run diagnostics standalone
    print("Running Relativistic Integration Diagnostics...")
    pytest.main([__file__, "-v", "-s"])