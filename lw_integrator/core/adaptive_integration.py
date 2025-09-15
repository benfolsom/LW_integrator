"""
Adaptive Self-Consistent Electromagnetic Field Integration

CAI: Hybrid integrator that automatically switches to self-consistent mode when
particles experience high forces, accelerations, or energy changes that could
lead to numerical instabilities or physics violations.

Key Features:
- Automatic trigger detection for self-consistency activation
- Configurable thresholds for force, acceleration, and energy change
- Seamless switching between standard and self-consistent modes
- Performance optimization by using self-consistency only when needed
- Comprehensive trigger monitoring and statistics

Trigger Parameters:
- Force magnitude: |F| = sqrt(dPx² + dPy² + dPz²)
- Acceleration magnitude: |a| = |F|/(γmc) 
- Relative energy change: |dE/E| = |dPt|/E_total
- Field gradient: rate of change of force magnitude

Author: Ben Folsom (human oversight)
Date: 2025-09-15
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

from ..physics.constants import (
    C_MMNS, NUMERICAL_EPSILON, CONVERGENCE_TOLERANCE,
    gamma_to_beta, beta_to_gamma
)
from ..physics.simulation_types import SimulationType, SimulationConfig
from .trajectory_integrator import LienardWiechertIntegrator
from .performance import OptimizedLienardWiechertIntegrator
from .self_consistent_fields import SelfConsistentLienardWiechertIntegrator


class TriggerType(Enum):
    """Types of triggers for self-consistency activation."""
    FORCE_MAGNITUDE = "force_magnitude"
    ACCELERATION_MAGNITUDE = "acceleration_magnitude" 
    RELATIVE_ENERGY_CHANGE = "relative_energy_change"
    FIELD_GRADIENT = "field_gradient"
    COMBINED = "combined"


@dataclass
class TriggerThresholds:
    """Thresholds for adaptive self-consistency triggers."""
    # Force magnitude threshold (relative to rest mass energy)
    force_threshold: float = 1e-3  # Trigger when |F|/(mc²) > threshold
    
    # Acceleration magnitude threshold (in units of c/characteristic_time)
    acceleration_threshold: float = 1e-2  # Trigger when |a|c/t_char > threshold
    
    # Relative energy change threshold per timestep
    energy_change_threshold: float = 1e-4  # Trigger when |dE/E| > threshold
    
    # Field gradient threshold (change in force per characteristic length)
    field_gradient_threshold: float = 1e-3  # Trigger when d|F|/dx > threshold


@dataclass 
class TriggerStatistics:
    """Statistics for trigger activations."""
    total_steps: int = 0
    self_consistent_steps: int = 0
    trigger_activations: Dict[str, int] = None
    max_force_magnitude: float = 0.0
    max_acceleration_magnitude: float = 0.0
    max_energy_change: float = 0.0
    
    def __post_init__(self):
        if self.trigger_activations is None:
            self.trigger_activations = {trigger.value: 0 for trigger in TriggerType}


class AdaptiveLienardWiechertIntegrator:
    """
    Adaptive Lienard-Wiechert integrator with automatic self-consistency.
    
    This integrator monitors physical quantities during integration and automatically
    switches to self-consistent mode when particles experience sudden changes that
    could lead to numerical instabilities or physics violations.
    """
    
    def __init__(self, use_optimized: bool = True, 
                 thresholds: Optional[TriggerThresholds] = None,
                 primary_trigger: TriggerType = TriggerType.FORCE_MAGNITUDE,
                 debug_mode: bool = False):
        """
        Initialize the adaptive integrator.
        
        Args:
            use_optimized: Use optimized integrator for standard steps
            thresholds: Trigger thresholds for self-consistency activation
            primary_trigger: Primary trigger type to monitor
            debug_mode: Enable detailed trigger diagnostics
        """
        self.use_optimized = use_optimized
        self.thresholds = thresholds or TriggerThresholds()
        self.primary_trigger = primary_trigger
        self.debug_mode = debug_mode
        
        # Initialize integrators
        if use_optimized:
            self.standard_integrator = OptimizedLienardWiechertIntegrator()
        else:
            self.standard_integrator = LienardWiechertIntegrator()
            
        # Self-consistent integrator (initialized when first needed)
        self.self_consistent_integrator = None
        
        # Statistics and monitoring
        self.statistics = TriggerStatistics()
        self.previous_force_magnitudes = []
        self.previous_positions = []
        
    def _initialize_self_consistent_integrator(self) -> None:
        """Initialize self-consistent integrator when first needed."""
        if self.self_consistent_integrator is None:
            config = SimulationConfig(
                simulation_type=SimulationType.FREE_PARTICLE_BUNCHES,
                debug_mode=self.debug_mode,
                convergence_tolerance=CONVERGENCE_TOLERANCE,
                max_iterations=5
            )
            self.self_consistent_integrator = SelfConsistentLienardWiechertIntegrator(config)
    
    def _calculate_force_magnitude(self, dPx: float, dPy: float, dPz: float) -> float:
        """Calculate magnitude of 3-force vector."""
        return np.sqrt(dPx**2 + dPy**2 + dPz**2)
    
    def _calculate_acceleration_magnitude(self, force_magnitude: float, 
                                        gamma: float, mass: float) -> float:
        """Calculate magnitude of 3-acceleration."""
        return force_magnitude / (gamma * mass * C_MMNS**2)
    
    def _calculate_relative_energy_change(self, dPt: float, total_energy: float) -> float:
        """Calculate relative energy change."""
        return abs(dPt) / total_energy if total_energy > 0 else 0.0
    
    def _calculate_field_gradient(self, current_force_mag: float, 
                                current_position: np.ndarray) -> float:
        """Calculate field gradient from force magnitude history."""
        if len(self.previous_force_magnitudes) < 2 or len(self.previous_positions) < 2:
            return 0.0
        
        # Simple finite difference estimate
        prev_force = self.previous_force_magnitudes[-1]
        prev_position = self.previous_positions[-1]
        
        force_change = abs(current_force_mag - prev_force)
        position_change = np.linalg.norm(current_position - prev_position)
        
        return force_change / position_change if position_change > 0 else 0.0
    
    def _check_triggers(self, dPx: float, dPy: float, dPz: float, dPt: float,
                       current_particle: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check if self-consistency should be triggered.
        
        Returns:
            (should_trigger, trigger_reason)
        """
        # Extract particle state
        gamma = current_particle.get('gamma', 1.0)
        if hasattr(gamma, '__iter__'):
            gamma = gamma[0]
        
        mass = current_particle.get('mass', 1.0)  # Assume normalized units
        position = np.array([current_particle.get('x', 0), 
                           current_particle.get('y', 0),
                           current_particle.get('z', 0)])
        total_energy = gamma * mass * C_MMNS**2
        
        # Calculate trigger parameters
        force_magnitude = self._calculate_force_magnitude(dPx, dPy, dPz)
        acceleration_magnitude = self._calculate_acceleration_magnitude(
            force_magnitude, gamma, mass)
        relative_energy_change = self._calculate_relative_energy_change(dPt, total_energy)
        field_gradient = self._calculate_field_gradient(force_magnitude, position)
        
        # Update statistics
        self.statistics.max_force_magnitude = max(
            self.statistics.max_force_magnitude, force_magnitude)
        self.statistics.max_acceleration_magnitude = max(
            self.statistics.max_acceleration_magnitude, acceleration_magnitude)
        self.statistics.max_energy_change = max(
            self.statistics.max_energy_change, relative_energy_change)
        
        # Store for gradient calculation
        self.previous_force_magnitudes.append(force_magnitude)
        self.previous_positions.append(position)
        
        # Keep only recent history
        if len(self.previous_force_magnitudes) > 10:
            self.previous_force_magnitudes.pop(0)
            self.previous_positions.pop(0)
        
        # Check triggers
        trigger_reasons = []
        
        # Normalize force by rest mass energy for threshold comparison
        normalized_force = force_magnitude / (mass * C_MMNS**2)
        if normalized_force > self.thresholds.force_threshold:
            trigger_reasons.append(f"Force: {normalized_force:.2e}")
            self.statistics.trigger_activations[TriggerType.FORCE_MAGNITUDE.value] += 1
        
        if acceleration_magnitude > self.thresholds.acceleration_threshold:
            trigger_reasons.append(f"Acceleration: {acceleration_magnitude:.2e}")
            self.statistics.trigger_activations[TriggerType.ACCELERATION_MAGNITUDE.value] += 1
        
        if relative_energy_change > self.thresholds.energy_change_threshold:
            trigger_reasons.append(f"Energy change: {relative_energy_change:.2e}")
            self.statistics.trigger_activations[TriggerType.RELATIVE_ENERGY_CHANGE.value] += 1
        
        if field_gradient > self.thresholds.field_gradient_threshold:
            trigger_reasons.append(f"Field gradient: {field_gradient:.2e}")
            self.statistics.trigger_activations[TriggerType.FIELD_GRADIENT.value] += 1
        
        should_trigger = len(trigger_reasons) > 0
        trigger_reason = "; ".join(trigger_reasons) if trigger_reasons else "None"
        
        if should_trigger and self.debug_mode:
            print(f"🔥 Self-consistency triggered: {trigger_reason}")
            print(f"   Force: {normalized_force:.2e}, Accel: {acceleration_magnitude:.2e}")
            print(f"   Energy: {relative_energy_change:.2e}, Gradient: {field_gradient:.2e}")
        
        return should_trigger, trigger_reason
    
    def adaptive_integration_step(self, h_step: float, trajectory: List[Dict], 
                                trajectory_drv: List[Dict], i_traj: int, 
                                apt_R: float, sim_type: SimulationType) -> Dict[str, Any]:
        """
        Adaptive integration step with automatic self-consistency triggering.
        
        Args:
            h_step: Integration time step
            trajectory: Rider trajectory array
            trajectory_drv: Driver trajectory array  
            i_traj: Current trajectory index
            apt_R: Aperture radius
            sim_type: Simulation type enum
            
        Returns:
            Next trajectory point with adaptive integration
        """
        self.statistics.total_steps += 1
        
        # First, try standard integration
        if hasattr(self.standard_integrator, 'retarded_rk4_rela_step'):
            next_point_standard = self.standard_integrator.retarded_rk4_rela_step(
                h_step, trajectory, trajectory_drv, i_traj, apt_R, sim_type)
        else:
            # Fallback for interface compatibility
            next_point_standard = trajectory[i_traj].copy()
        
        # Extract force components from the standard step
        # This is a simplified approach - in practice, you'd calculate forces
        # within the step and check triggers during force calculation
        
        # For demonstration, we'll calculate approximate force from position change
        if i_traj > 0:
            current = trajectory[i_traj]
            previous = trajectory[i_traj-1]
            
            # Approximate force from acceleration (simplified)
            gamma = current.get('gamma', 1.0)
            if hasattr(gamma, '__iter__'):
                gamma = gamma[0]
            mass = 1.0  # Normalized units
            
            # Simple finite difference approximation
            dPx = (current.get('Px', 0) - previous.get('Px', 0)) / h_step
            dPy = (current.get('Py', 0) - previous.get('Py', 0)) / h_step  
            dPz = (current.get('Pz', 0) - previous.get('Pz', 0)) / h_step
            dPt = (current.get('Pt', gamma*mass*C_MMNS**2) - 
                   previous.get('Pt', gamma*mass*C_MMNS**2)) / h_step
            
            # Check if self-consistency should be triggered
            should_trigger, trigger_reason = self._check_triggers(
                dPx, dPy, dPz, dPt, current)
            
            if should_trigger:
                # Use self-consistent integration
                self._initialize_self_consistent_integrator()
                self.statistics.self_consistent_steps += 1
                
                if self.debug_mode:
                    print(f"   Using self-consistent step at i={i_traj}")
                
                return self.self_consistent_integrator.self_consistent_enhanced_step(
                    h_step, trajectory, trajectory_drv, i_traj, apt_R, sim_type)
        
        # Use standard integration
        return next_point_standard
    
    def integrate(self, init_rider: Dict[str, Any], init_driver: Dict[str, Any],
                 steps_tot: int, h_step: float, wall_Z: float, apt_R: float,
                 sim_type: SimulationType = SimulationType.FREE_PARTICLE_BUNCHES) -> Tuple[List[Dict], List[Dict]]:
        """
        Complete adaptive integration with automatic self-consistency.
        
        Args:
            init_rider: Initial rider particle state
            init_driver: Initial driver particle state
            steps_tot: Total integration steps
            h_step: Time step size
            wall_Z: Wall position
            apt_R: Aperture radius
            sim_type: Simulation type
            
        Returns:
            (rider_trajectory, driver_trajectory) tuple
        """
        # Initialize trajectories
        trajectory_rider = [init_rider.copy()]
        trajectory_driver = [init_driver.copy()]
        
        if self.debug_mode:
            print(f"🚀 Starting adaptive integration: {steps_tot} steps")
            print(f"   Thresholds - Force: {self.thresholds.force_threshold:.1e}, "
                  f"Accel: {self.thresholds.acceleration_threshold:.1e}")
        
        # Integration loop
        for i in range(1, steps_tot):
            # Adaptive step for rider
            next_rider = self.adaptive_integration_step(
                h_step, trajectory_rider, trajectory_driver, i-1, apt_R, sim_type)
            trajectory_rider.append(next_rider)
            
            # For driver, use standard integration (could also be adaptive)
            if hasattr(self.standard_integrator, 'retarded_rk4_rela_step'):
                next_driver = self.standard_integrator.retarded_rk4_rela_step(
                    h_step, trajectory_driver, trajectory_rider, i-1, apt_R, sim_type)
                trajectory_driver.append(next_driver)
            else:
                trajectory_driver.append(trajectory_driver[i-1].copy())
        
        # Final statistics
        self_consistent_percentage = (100.0 * self.statistics.self_consistent_steps / 
                                    self.statistics.total_steps)
        
        if self.debug_mode:
            print(f"✅ Adaptive integration complete!")
            print(f"   Self-consistent steps: {self.statistics.self_consistent_steps}/{self.statistics.total_steps} "
                  f"({self_consistent_percentage:.1f}%)")
            print(f"   Max force: {self.statistics.max_force_magnitude:.2e}")
            print(f"   Max acceleration: {self.statistics.max_acceleration_magnitude:.2e}")
            print(f"   Max energy change: {self.statistics.max_energy_change:.2e}")
            
            for trigger_type, count in self.statistics.trigger_activations.items():
                if count > 0:
                    print(f"   {trigger_type}: {count} activations")
        
        return trajectory_rider, trajectory_driver
    
    def get_statistics(self) -> TriggerStatistics:
        """Get trigger statistics for analysis."""
        return self.statistics
    
    def reset_statistics(self) -> None:
        """Reset trigger statistics."""
        self.statistics = TriggerStatistics()
        self.previous_force_magnitudes = []
        self.previous_positions = []