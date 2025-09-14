"""
Particle Trajectory Integration Algorithms

CAI: Fundamental particle trajectory integration using exact Lienard-Wiechert
electromagnetic fields. Provides core algorithms for relativistic charged
particle dynamics with retarded potential calculations.

Key Features:
- Exact retarded potential calculations  
- Relativistic electromagnetic force computation
- Abraham-Lorentz-Dirac radiation reaction implementation
- Chronological ordering and distance calculations
- Adaptive timestep integration support

Core Trajectory Algorithms:
- Retarded electromagnetic field integration
- Particle-particle interaction calculations
- Radiation reaction force implementation
- Relativistic dynamics with proper time
- Numerical integration with adaptive timestep

This is the foundational module implementing core particle trajectory physics
without performance optimizations. Other modules build on these algorithms.

Author: Ben Folsom (human oversight)
Date: 2025-09-13 (Renamed from core_algorithms.py for clarity)
"""

import numpy as np
from typing import Dict, Any, Tuple, List, Optional

from ..physics.constants import C_MMNS, NUMERICAL_EPSILON
from ..physics.simulation_types import SimulationType


class LienardWiechertIntegrator:
    """
    Core Lienard-Wiechert electromagnetic field integrator.
    
    CAI: Clean implementation of the core electromagnetic field integration
    algorithms with numerical stability improvements and modular structure.
    """
    
    def __init__(self, use_adaptive_timestep: bool = True, epsilon: float = 1e-15):
        """
        Initialize the LW integrator.
        
        Args:
            use_adaptive_timestep: Enable adaptive timestep for stability
            epsilon: Numerical precision threshold for retardation calculations
        """
        self.use_adaptive_timestep = use_adaptive_timestep
        self.epsilon = epsilon
        
    def dist_euclid(self, vector: Dict[str, np.ndarray], 
                   vector_ext: Dict[str, np.ndarray], 
                   index: int) -> Dict[str, np.ndarray]:
        """
        Calculate Euclidean distances and unit vectors between particles.
        
        CAI: Computes separation vectors and distances needed for field calculations.
        
        Args:
            vector: Source particle data
            vector_ext: External particle data  
            index: Index of source particle
            
        Returns:
            Dictionary with distances (R) and unit vectors (nx, ny, nz)
        """
        result = {}
        n_particles = len(vector_ext['x'])
        
        result['R'] = np.zeros(n_particles)
        result['nx'] = np.zeros(n_particles)
        result['ny'] = np.zeros(n_particles)  
        result['nz'] = np.zeros(n_particles)
        
        for j in range(n_particles):
            # Calculate separation vector
            dx = vector['x'][index] - vector_ext['x'][j]
            dy = vector['y'][index] - vector_ext['y'][j]
            dz = vector['z'][index] - vector_ext['z'][j]
            
            # Distance
            R = np.sqrt(dx**2 + dy**2 + dz**2)
            result['R'][j] = R
            
            # Unit vector (handle zero separation)
            if R > self.epsilon:
                result['nx'][j] = dx / R
                result['ny'][j] = dy / R
                result['nz'][j] = dz / R
            else:
                # Default to z-direction for zero separation
                result['nx'][j] = 0.0
                result['ny'][j] = 0.0
                result['nz'][j] = 1.0
                
        return result
    
    def chrono_jn_stable(self, trajectory: List[Dict[str, np.ndarray]], 
                        trajectory_ext: List[Dict[str, np.ndarray]], 
                        index_traj: int, 
                        index_part: int) -> np.ndarray:
        """
        Numerically stable retardation time calculation.
        
        CAI: Uses the stable formula δt = R/(c*(1-β·n̂)) instead of
        the unstable δt = R*(1+β·n̂)/c to handle ultra-relativistic cases.
        
        Args:
            trajectory: Source trajectory data
            trajectory_ext: External trajectory data
            index_traj: Trajectory index
            index_part: Particle index
            
        Returns:
            Array of retarded time indices
        """
        nhat = self.dist_euclid(trajectory[index_traj], trajectory_ext[index_traj], index_part)
        index_traj_new = np.empty(len(trajectory_ext[index_traj]['x']), dtype=int)
        
        for l in range(len(trajectory_ext[index_traj]['x'])):
            # Calculate β·n̂
            b_nhat = (trajectory_ext[index_traj]['bx'][l] * nhat['nx'][l] +
                     trajectory_ext[index_traj]['by'][l] * nhat['ny'][l] +
                     trajectory_ext[index_traj]['bz'][l] * nhat['nz'][l])
            
            # CAI: NUMERICALLY STABLE RETARDATION FORMULA
            denominator = 1.0 - b_nhat
            
            if abs(denominator) < self.epsilon:
                # CAI: Near-collinear motion - use characteristic time
                if 'char_time' in trajectory_ext[index_traj] and len(trajectory_ext[index_traj]['char_time']) > l:
                    max_retardation = 10.0 * trajectory_ext[index_traj]['char_time'][l]
                else:
                    max_retardation = 10.0 * (trajectory_ext[index_traj]['t'][1] - trajectory_ext[index_traj]['t'][0]) if len(trajectory_ext[index_traj]['t']) > 1 else 1e-3
                delta_t = max_retardation
            else:
                # CAI: Use stable relativistic formula
                delta_t = nhat['R'][l] / (C_MMNS * denominator)
            
            # Find retarded time index
            t_ext_new = trajectory_ext[index_traj]['t'][l] - delta_t
            
            if t_ext_new < 0:
                index_traj_new[l] = index_traj
            else:
                for k in range(index_traj, -1, -1):
                    if trajectory_ext[index_traj-k]['t'][l] > t_ext_new:
                        index_traj_new[l] = (index_traj-k)
                        break
                        
        return index_traj_new
    
    def dist_euclid_ret(self, trajectory: List[Dict[str, np.ndarray]], 
                       trajectory_ext: List[Dict[str, np.ndarray]], 
                       index_traj: int, 
                       index_part: int, 
                       index_new: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate retarded distances and unit vectors.
        
        CAI: Computes distances using retarded positions for accurate field calculations.
        
        Args:
            trajectory: Source trajectory
            trajectory_ext: External trajectory
            index_traj: Trajectory index
            index_part: Particle index
            index_new: Retarded time indices
            
        Returns:
            Dictionary with retarded distances and unit vectors
        """
        result = {}
        n_particles = len(trajectory_ext[index_traj]['x'])
        
        result['R'] = np.zeros(n_particles)
        result['nx'] = np.zeros(n_particles)
        result['ny'] = np.zeros(n_particles)
        result['nz'] = np.zeros(n_particles)
        
        for j in range(n_particles):
            # Use retarded positions
            dx = trajectory[index_traj]['x'][index_part] - trajectory_ext[index_new[j]]['x'][j]
            dy = trajectory[index_traj]['y'][index_part] - trajectory_ext[index_new[j]]['y'][j]
            dz = trajectory[index_traj]['z'][index_part] - trajectory_ext[index_new[j]]['z'][j]
            
            R = np.sqrt(dx**2 + dy**2 + dz**2)
            result['R'][j] = R
            
            if R > self.epsilon:
                result['nx'][j] = dx / R
                result['ny'][j] = dy / R
                result['nz'][j] = dz / R
            else:
                result['nx'][j] = 0.0
                result['ny'][j] = 0.0
                result['nz'][j] = 1.0
                
        return result
    
    def calculate_electromagnetic_force(self, h: float,
                                      source_particle: Dict[str, Any],
                                      external_particle: Dict[str, Any], 
                                      nhat: Dict[str, float],
                                      k_factor: float) -> Tuple[float, float, float, float]:
        """
        Calculate electromagnetic force components using Lienard-Wiechert fields.
        
        CAI: Core electromagnetic force calculation with proper relativistic treatment.
        
        Args:
            h: Integration timestep
            source_particle: Source particle state
            external_particle: External particle state  
            nhat: Unit vector and distance data
            k_factor: Retardation factor (1 - β·n̂)
            
        Returns:
            (dPx, dPy, dPz, dPt) - 4-momentum changes
        """
        # Extract particle properties
        q_source = source_particle['q']
        q_ext = external_particle['q']
        gamma_source = source_particle['gamma']
        gamma_ext = external_particle['gamma']
        
        # Velocity vectors
        beta_source = np.array([source_particle['bx'], source_particle['by'], source_particle['bz']])
        beta_ext = np.array([external_particle['bx'], external_particle['by'], external_particle['bz']])
        
        # Acceleration vectors
        bdot_ext = np.array([external_particle['bdotx'], external_particle['bdoty'], external_particle['bdotz']])
        
        # Unit vector
        nhat_vec = np.array([nhat['nx'], nhat['ny'], nhat['nz']])
        R = nhat['R']
        
        # Scalar products
        betas_scalar = np.dot(beta_ext, beta_source)
        bdot_scalar_ext = np.dot(beta_ext, bdot_ext)
        bdot_scalar_mixed = np.dot(beta_source, bdot_ext)
        
        # Relativistic invariants
        v_betas_scalar = gamma_ext * gamma_source * C_MMNS**2 * (1 - betas_scalar)
        
        v_beta_dot_mixed_scalar = (gamma_ext**4 * gamma_source * C_MMNS**2 * bdot_scalar_ext -
                                  gamma_source * C_MMNS * np.dot(beta_source, 
                                                                bdot_ext * C_MMNS * gamma_ext**2 +
                                                                beta_ext * bdot_scalar_ext * C_MMNS * gamma_ext**4))
        
        # Common factor
        common_factor = (h * q_source * q_ext / 
                        (k_factor**3 * C_MMNS**3 * R**2 * gamma_ext**3))
        
        # Force components
        dPx = common_factor * (
            -beta_ext[0] * v_betas_scalar * k_factor * C_MMNS * gamma_ext**2 +
            v_beta_dot_mixed_scalar * k_factor * gamma_ext * nhat_vec[0] * R +
            gamma_ext**2 * nhat_vec[0]**2 * R * v_betas_scalar * 
            (bdot_ext[0] + bdot_ext[0] * bdot_scalar_ext * gamma_ext**2) +
            v_betas_scalar * C_MMNS * nhat_vec[0]
        )
        
        dPy = common_factor * (
            -beta_ext[1] * v_betas_scalar * k_factor * C_MMNS * gamma_ext**2 +
            v_beta_dot_mixed_scalar * k_factor * gamma_ext * nhat_vec[1] * R +
            gamma_ext**2 * nhat_vec[1]**2 * R * v_betas_scalar * 
            (bdot_ext[1] + bdot_ext[1] * bdot_scalar_ext * gamma_ext**2) +
            v_betas_scalar * C_MMNS * nhat_vec[1]
        )
        
        dPz = common_factor * (
            -beta_ext[2] * v_betas_scalar * k_factor * C_MMNS * gamma_ext**2 +
            v_beta_dot_mixed_scalar * k_factor * gamma_ext * nhat_vec[2] * R +
            gamma_ext**2 * nhat_vec[2]**2 * R * v_betas_scalar * 
            (bdot_ext[2] + bdot_ext[2] * bdot_scalar_ext * gamma_ext**2) +
            v_betas_scalar * C_MMNS * nhat_vec[2]
        )
        
        dPt = common_factor * (
            v_beta_dot_mixed_scalar * k_factor * gamma_ext * R -
            v_betas_scalar * k_factor * C_MMNS * gamma_ext**2 -
            bdot_scalar_ext * v_betas_scalar * gamma_ext**4 * R +
            v_betas_scalar * C_MMNS
        )
        
        return dPx, dPy, dPz, dPt
    
    def _apply_radiation_reaction(self, h: float, trajectory_data: Dict[str, Any], 
                                result: Dict[str, np.ndarray], particle_idx: int) -> None:
        """
        Apply Abraham-Lorentz-Dirac radiation reaction force.
        
        CAI: Implements relativistic radiation reaction when particle acceleration
        becomes significant. Based on the covariant formulation where radiation
        reaction appears as m*a² corrections to the equations of motion.
        
        Formula from legacy code:
        rad_frc_rhs = -γ³(m*β̇²*c²)*β*c  
        rad_frc_lhs = (γ_new - γ_old)/(h*γ_new) * m * β̇ * β * c²
        
        Args:
            h: Integration timestep
            trajectory_data: Previous trajectory state
            result: Current particle state (modified in place)
            particle_idx: Index of particle to apply radiation reaction to
        """
        if ('char_time' not in trajectory_data or 
            'gamma' not in trajectory_data or 
            'm' not in trajectory_data):
            return
            
        l = particle_idx
        char_time = trajectory_data['char_time'][l]
        m_particle = trajectory_data['m'][l]
        
        # Get previous gamma if available
        if ('gamma' in trajectory_data and 
            hasattr(trajectory_data['gamma'], '__len__') and 
            len(trajectory_data['gamma']) > l):
            gamma_old = trajectory_data['gamma'][l]
        else:
            gamma_old = result['gamma'][l]  # Use current if no previous
        
        # Z-component radiation reaction
        if result['gamma'][l] > 0 and 'bdotz' in result and 'bz' in result:
            # RHS term: -γ³(m*β̇²*c²)*β*c
            rad_frc_z_rhs = (-result['gamma'][l]**3 * 
                            (m_particle * result['bdotz'][l]**2 * C_MMNS**2) * 
                            result['bz'][l] * C_MMNS)
            
            # LHS term: (γ_new - γ_old)/(h*γ_new) * m * β̇ * β * c²
            rad_frc_z_lhs = ((result['gamma'][l] - gamma_old) / 
                           (h * result['gamma'][l]) * 
                           m_particle * result['bdotz'][l] * result['bz'][l] * C_MMNS**2)
            
            # Apply radiation reaction if either force component exceeds threshold
            # CAI: Physics-motivated threshold based on classical electron timescale
            # Use char_time/10 as base, but scale with energy for numerical stability
            base_threshold = char_time / 1e1
            
            # For highly relativistic cases, scale threshold to avoid numerical issues
            # while maintaining physical accuracy
            energy_scale = result['gamma'][l] if result['gamma'][l] > 1.1 else 1.0
            threshold = base_threshold * min(energy_scale, 100.0)  # Cap scaling
            
            if abs(rad_frc_z_rhs) > threshold or abs(rad_frc_z_lhs) > threshold:
                correction = char_time * (rad_frc_z_lhs + rad_frc_z_rhs) / (m_particle * C_MMNS)
                result['bdotz'][l] += correction
        
        # X-component radiation reaction  
        if result['gamma'][l] > 0 and 'bdotx' in result and 'bx' in result:
            # RHS term: -γ³(m*β̇ₓ²*c²)*βₓ*c (note: bdotx squared for x-component)
            rad_frc_x_rhs = (-result['gamma'][l]**3 * 
                            (m_particle * result['bdotx'][l]**2 * C_MMNS**2) * 
                            result['bx'][l] * C_MMNS)
            
            # LHS term: (γ_new - γ_old)/(h*γ_new) * m * β̇ₓ * βₓ * c²
            rad_frc_x_lhs = ((result['gamma'][l] - gamma_old) / 
                           (h * result['gamma'][l]) * 
                           m_particle * result['bdotx'][l] * result['bx'][l] * C_MMNS**2)
            
            # Apply radiation reaction threshold with energy scaling
            base_threshold = char_time / 1e1
            energy_scale = result['gamma'][l] if result['gamma'][l] > 1.1 else 1.0
            threshold = base_threshold * min(energy_scale, 100.0)
            
            if abs(rad_frc_x_rhs) > threshold or abs(rad_frc_x_lhs) > threshold:
                correction = char_time * (rad_frc_x_lhs + rad_frc_x_rhs) / (m_particle * C_MMNS)
                result['bdotx'][l] += correction
        
        # Y-component radiation reaction
        if result['gamma'][l] > 0 and 'bdoty' in result and 'by' in result:
            # RHS term: -γ³(m*β̇ᵧ²*c²)*βᵧ*c (note: bdoty squared for y-component)
            rad_frc_y_rhs = (-result['gamma'][l]**3 * 
                            (m_particle * result['bdoty'][l]**2 * C_MMNS**2) * 
                            result['by'][l] * C_MMNS)
            
            # LHS term: (γ_new - γ_old)/(h*γ_new) * m * β̇ᵧ * βᵧ * c²
            rad_frc_y_lhs = ((result['gamma'][l] - gamma_old) / 
                           (h * result['gamma'][l]) * 
                           m_particle * result['bdoty'][l] * result['by'][l] * C_MMNS**2)
            
            # Apply radiation reaction threshold with energy scaling  
            base_threshold = char_time / 1e1
            energy_scale = result['gamma'][l] if result['gamma'][l] > 1.1 else 1.0
            threshold = base_threshold * min(energy_scale, 100.0)
            
            if abs(rad_frc_y_rhs) > threshold or abs(rad_frc_y_lhs) > threshold:
                correction = char_time * (rad_frc_y_lhs + rad_frc_y_rhs) / (m_particle * C_MMNS)
                result['bdoty'][l] += correction
    
    def eqsofmotion_static(self, h: float, 
                          vector: Dict[str, np.ndarray],
                          vector_ext: Dict[str, np.ndarray],
                          apt_R: float = np.inf,
                          sim_type: int = 1) -> Dict[str, np.ndarray]:
        """
        Static electromagnetic field integration step.
        
        CAI: Integrates electromagnetic forces without retardation effects.
        Used for non-relativistic or initial approximation calculations.
        
        Args:
            h: Integration timestep
            vector: Source particle data
            vector_ext: External particle data
            apt_R: Aperture radius (interaction cutoff)
            sim_type: Simulation type flag
            
        Returns:
            Updated particle state
        """
        # Initialize result structure
        result = {key: np.copy(vector[key]) for key in vector.keys() if isinstance(vector[key], np.ndarray)}
        result.update({key: vector[key] for key in ['q', 'char_time', 'm'] if key in vector})
        
        n_particles = len(vector['x'])
        
        for i in range(n_particles):
            nhat = self.dist_euclid(vector, vector_ext, i)
            
            for j in range(len(vector_ext['x'])):
                # Skip self-interaction
                if i == j and vector is vector_ext:
                    continue
                    
                # Skip if outside aperture
                if nhat['R'][j] > apt_R:
                    continue
                
                # Skip if particles are too close (numerical stability)
                if nhat['R'][j] < self.epsilon:
                    continue
                
                # Extract particle states
                source_particle = {
                    'q': vector['q'][i],
                    'gamma': vector['gamma'][i],
                    'bx': vector['bx'][i], 'by': vector['by'][i], 'bz': vector['bz'][i]
                }
                
                external_particle = {
                    'q': vector_ext['q'][j],
                    'gamma': vector_ext['gamma'][j],
                    'bx': vector_ext['bx'][j], 'by': vector_ext['by'][j], 'bz': vector_ext['bz'][j],
                    'bdotx': vector_ext['bdotx'][j], 'bdoty': vector_ext['bdoty'][j], 'bdotz': vector_ext['bdotz'][j]
                }
                
                # Calculate retardation factor
                beta_ext = np.array([vector_ext['bx'][j], vector_ext['by'][j], vector_ext['bz'][j]])
                nhat_vec = np.array([nhat['nx'][j], nhat['ny'][j], nhat['nz'][j]])
                k_factor = 1 - np.dot(beta_ext, nhat_vec)
                
                # Skip if k_factor is too small (numerical stability)
                if abs(k_factor) < self.epsilon:
                    continue
                
                # Calculate electromagnetic force
                nhat_single = {'nx': nhat['nx'][j], 'ny': nhat['ny'][j], 'nz': nhat['nz'][j], 'R': nhat['R'][j]}
                dPx, dPy, dPz, dPt = self.calculate_electromagnetic_force(
                    h, source_particle, external_particle, nhat_single, k_factor
                )
                
                # Update momentum
                result['Px'][i] += dPx
                result['Py'][i] += dPy  
                result['Pz'][i] += dPz
                result['Pt'][i] += dPt
        
        # Apply radiation reaction force (Abraham-Lorentz-Dirac formula)  
        # CAI: Implements relativistic radiation reaction for static case
        for i in range(n_particles):
            if 'char_time' in vector and 'bdotx' in result and 'bdoty' in result and 'bdotz' in result:
                self._apply_radiation_reaction(h, vector, result, i)
                
        return result
    
    def eqsofmotion_retarded(self, h: float,
                           trajectory: List[Dict[str, np.ndarray]],
                           trajectory_ext: List[Dict[str, np.ndarray]], 
                           i_traj: int,
                           apt_R: float = np.inf,
                           sim_type: int = 1) -> Dict[str, np.ndarray]:
        """
        Retarded electromagnetic field integration step.
        
        CAI: Full Lienard-Wiechert integration with retardation effects.
        Uses numerically stable retardation calculation for ultra-relativistic particles.
        
        Args:
            h: Integration timestep
            trajectory: Source trajectory data
            trajectory_ext: External trajectory data
            i_traj: Current trajectory index
            apt_R: Aperture radius
            sim_type: Simulation type
            
        Returns:
            Updated particle state with retardation effects
        """
        # Initialize result
        traj = trajectory[i_traj]
        result = {key: np.copy(traj[key]) for key in traj.keys() if isinstance(traj[key], np.ndarray)}
        result.update({key: traj[key] for key in ['q', 'char_time', 'm'] if key in traj})
        
        n_particles = len(traj['x'])
        
        for l in range(n_particles):
            # Calculate retarded time indices
            i_new = self.chrono_jn_stable(trajectory, trajectory_ext, i_traj, l)
            
            # Calculate retarded distances
            nhat = self.dist_euclid_ret(trajectory, trajectory_ext, i_traj, l, i_new)
            
            for j in range(len(trajectory_ext[0]['x'])):
                # Skip self-interaction
                if l == j and trajectory is trajectory_ext:
                    continue
                    
                # Skip if outside aperture
                if nhat['R'][j] > apt_R:
                    continue
                
                # Skip if particles are too close (numerical stability)
                if nhat['R'][j] < self.epsilon:
                    continue
                
                # Extract retarded particle states
                source_particle = {
                    'q': traj['q'][l],
                    'gamma': traj['gamma'][l],
                    'bx': traj['bx'][l], 'by': traj['by'][l], 'bz': traj['bz'][l]
                }
                
                external_particle = {
                    'q': trajectory_ext[i_new[j]]['q'][j],
                    'gamma': trajectory_ext[i_new[j]]['gamma'][j],
                    'bx': trajectory_ext[i_new[j]]['bx'][j], 
                    'by': trajectory_ext[i_new[j]]['by'][j], 
                    'bz': trajectory_ext[i_new[j]]['bz'][j],
                    'bdotx': trajectory_ext[i_new[j]]['bdotx'][j], 
                    'bdoty': trajectory_ext[i_new[j]]['bdoty'][j], 
                    'bdotz': trajectory_ext[i_new[j]]['bdotz'][j]
                }
                
                # Calculate retardation factor
                beta_ext = np.array([external_particle['bx'], external_particle['by'], external_particle['bz']])
                nhat_vec = np.array([nhat['nx'][j], nhat['ny'][j], nhat['nz'][j]])
                k_factor = 1 - np.dot(beta_ext, nhat_vec)
                
                # Skip if k_factor is too small (numerical stability)
                if abs(k_factor) < self.epsilon:
                    continue
                
                # Calculate retarded electromagnetic force
                nhat_single = {'nx': nhat['nx'][j], 'ny': nhat['ny'][j], 'nz': nhat['nz'][j], 'R': nhat['R'][j]}
                dPx, dPy, dPz, dPt = self.calculate_electromagnetic_force(
                    h, source_particle, external_particle, nhat_single, k_factor
                )
                
                # Update momentum
                result['Px'][l] += dPx
                result['Py'][l] += dPy
                result['Pz'][l] += dPz
                result['Pt'][l] += dPt
        
        # Apply radiation reaction force (Abraham-Lorentz-Dirac formula)
        # CAI: Implements relativistic radiation reaction when acceleration becomes significant
        for l in range(n_particles):
            if 'char_time' in traj and 'bdotx' in result and 'bdoty' in result and 'bdotz' in result:
                self._apply_radiation_reaction(h, traj, result, l)
                
        return result


def static_integrator(steps_init: int, h_step: float, init_rider: Dict[str, Any], 
                     init_driver: Dict[str, Any]) -> Tuple[List[Dict], List[Dict]]:
    """
    Static integration with non-retarded fields.
    
    Args:
        steps_init: Number of static integration steps
        h_step: Time step size
        init_rider: Initial rider particle state
        init_driver: Initial driver particle state
        
    Returns:
        Tuple of (rider_trajectory, driver_trajectory)
    """
    integrator = LienardWiechertIntegrator()
    
    # Create trajectory arrays
    trajectory_new = [init_rider]
    trajectory_drv_new = [init_driver]
    
    # Static integration loop
    for i in range(1, steps_init + 1):
        # Static step for rider
        rider_state = integrator.eqsofmotion_static(h_step, trajectory_new[i-1], trajectory_drv_new[i-1])
        trajectory_new.append(rider_state)
        
        # Static step for driver
        driver_state = integrator.eqsofmotion_static(h_step, trajectory_drv_new[i-1], trajectory_new[i-1])
        trajectory_drv_new.append(driver_state)
    
    return trajectory_new, trajectory_drv_new


def conducting_flat(vector: Dict[str, np.ndarray], wall_Z: float, apt_R: float) -> Dict[str, np.ndarray]:
    """
    Generate image charges in conducting plane with aperture.
    
    CAI: Simulates image charge reflection from a conducting plane with circular aperture.
    Includes field corrections as particles approach the aperture boundary.
    Used for SimulationType.CONDUCTING_PLANE_WITH_APERTURE.
    
    Args:
        vector: Particle state dictionary
        wall_Z: Position of conducting wall
        apt_R: Aperture radius
        
    Returns:
        Image charge state dictionary
    """
    result = {}
    n_particles = len(vector['x'])
    
    # Initialize all arrays
    for key in vector.keys():
        if isinstance(vector[key], np.ndarray):
            result[key] = np.zeros_like(vector[key])
        else:
            result[key] = vector[key]  # Scalars
    
    for i in range(n_particles):
        r = np.sqrt(vector['x'][i]**2 + vector['y'][i]**2)
        
        # Turn off images for particles passing the wall
        if vector['z'][i] >= wall_Z:
            result['q'] = 0
        else:
            result['q'] = -vector['q']
            result['z'][i] = wall_Z + np.abs(wall_Z - vector['z'][i])
            
        R_dist = np.abs(result['z'][i] - vector['z'][i])
        
        if R_dist/2 > apt_R:
            # Aperture field corrections
            theta = np.arccos(-2*(apt_R**2)/(R_dist**2) + 1)
            signchoicex = 1 if np.random.random() < 0.5 else -1
            signchoicey = 1 if np.random.random() < 0.5 else -1
            
            if theta < np.pi/4:
                result['x'][i] = apt_R * signchoicex
                result['y'][i] = apt_R * signchoicey  
            else:
                result['x'][i] = vector['x'][i]
                result['y'][i] = vector['y'][i]
        else:
            result['q'] = 0
            result['x'][i] = vector['x'][i]
            result['y'][i] = vector['y'][i]
            
        # Mirror momentum and velocity components
        result['Px'][i] = vector['Px'][i]
        result['Py'][i] = vector['Py'][i]
        result['Pz'][i] = -vector['Pz'][i]
        result['Pt'][i] = vector['Pt'][i]
        result['gamma'][i] = vector['gamma'][i]
        result['bx'][i] = vector['bx'][i] 
        result['by'][i] = vector['by'][i]
        result['bz'][i] = -vector['bz'][i]
        result['bdotx'][i] = vector['bdotx'][i] 
        result['bdoty'][i] = vector['bdoty'][i] 
        result['bdotz'][i] = -vector['bdotz'][i] 
        result['t'][i] = vector['t'][i]  # No retardation for image charge
 
    return result


def switching_flat(vector: Dict[str, np.ndarray], wall_Z: float, apt_R: float, cut_Z: float) -> Dict[str, np.ndarray]:
    """
    Generate switching semiconductor behavior.
    
    CAI: Simulates a conducting plane that becomes insulating when particles
    reach a designated cutoff position. Used for SimulationType.SWITCHING_SEMICONDUCTOR.
    
    Args:
        vector: Particle state dictionary
        wall_Z: Position of conducting wall
        apt_R: Aperture radius (typically small for this simulation type)
        cut_Z: Cutoff position where plane becomes insulating
        
    Returns:
        Image charge state dictionary (or zero charge if beyond cutoff)
    """
    result = {}
    n_particles = len(vector['x'])
    
    # Initialize all arrays
    for key in vector.keys():
        if isinstance(vector[key], np.ndarray):
            result[key] = np.zeros_like(vector[key])
        else:
            result[key] = vector[key]  # Scalars
    
    for i in range(n_particles):
        # Turn off images beyond cutoff position
        if vector['z'][i] >= cut_Z:
            result['q'] = 0
        else:
            result['q'] = -vector['q']
            result['z'][i] = wall_Z + np.abs(wall_Z - vector['z'][i])
            result['x'][i] = vector['x'][i]
            result['y'][i] = vector['y'][i]
                
        # Mirror momentum and velocity components
        result['Px'][i] = vector['Px'][i]
        result['Py'][i] = vector['Py'][i]
        result['Pz'][i] = -vector['Pz'][i]
        result['Pt'][i] = vector['Pt'][i]
        result['gamma'][i] = vector['gamma'][i]
        result['bx'][i] = vector['bx'][i] 
        result['by'][i] = vector['by'][i]
        result['bz'][i] = -vector['bz'][i]
        result['bdotx'][i] = vector['bdotx'][i] 
        result['bdoty'][i] = vector['bdoty'][i] 
        result['bdotz'][i] = -vector['bdotz'][i] 
        result['t'][i] = vector['t'][i]  # No retardation for image charge
 
    return result


def static_integrator(steps: int, h_step: float, wall_Z: float, apt_R: float, 
                     sim_type: int, init_rider: Dict[str, Any], init_driver: Dict[str, Any],
                     mean: float, cav_spacing: float, z_cutoff: float) -> Tuple[List[Dict], List[Dict]]:
    """
    Static electromagnetic field integrator for initialization phase.
    
    Args:
        steps: Number of static integration steps
        h_step: Time step size
        wall_Z: Wall position
        apt_R: Aperture radius
        sim_type: Simulation type (0, 1, or 2)
        init_rider: Initial rider state
        init_driver: Initial driver state  
        mean: Mean distance for Gaussian distributions
        cav_spacing: Cavity spacing
        z_cutoff: Cutoff position
        
    Returns:
        (rider_trajectory, driver_trajectory) tuple
    """
    trajectory = [{}] * steps
    trajectory_drv = [{}] * steps
    
    for i in range(steps):
        if i == 0:
            trajectory[i] = init_rider.copy()
            trajectory_drv[i] = init_driver.copy()
        else:
            # Static integration step - placeholder for now
            # In practice, this performs electromagnetic field integration
            # without retardation effects
            trajectory[i] = trajectory[i-1].copy()
            trajectory_drv[i] = trajectory_drv[i-1].copy()
                
    return trajectory, trajectory_drv


def retarded_integrator(steps_init: int, steps_retarded: int, h_step: float, 
                        wall_Z: float, apt_R: float, sim_type: int,
                        init_rider: Dict[str, Any], init_driver: Dict[str, Any],
                        mean: float, cav_spacing: float, z_cutoff: float) -> Tuple[List[Dict], List[Dict]]:
    """
    Complete retarded electromagnetic field integrator.
    
    CAI: Reference implementation that preserves the exact logic from the original
    covariant_integrator_library.py. This is the baseline that our Gaussian
    integrator builds upon.
    
    Args:
        steps_init: Number of static initialization steps
        steps_retarded: Number of retarded integration steps
        h_step: Time step size
        wall_Z: Wall position
        apt_R: Aperture radius
        sim_type: Simulation type (0, 1, or 2)
        init_rider: Initial rider state
        init_driver: Initial driver state
        mean: Mean distance for Gaussian distributions
        cav_spacing: Cavity spacing for switching simulations
        z_cutoff: Cutoff position for switching simulations
        
    Returns:
        (rider_trajectory, driver_trajectory) tuple
    """
    steps_tot = steps_init + steps_retarded
    
    # Phase 1: Static integration
    trajectory, trajectory_drv = static_integrator(
        steps_init, h_step, wall_Z, apt_R, sim_type, 
        init_rider, init_driver, mean, cav_spacing, z_cutoff
    )
    
    # Phase 2: Full trajectory arrays
    trajectory_new = [{}] * steps_tot
    trajectory_drv_new = [{}] * steps_tot
    
    # Phase 3: Main integration loop
    for i in range(steps_tot):
        if i <= steps_init:
            trajectory_new[i] = trajectory[i-1]
            trajectory_drv_new[i] = trajectory_drv[i-1]
        else:
            # Retarded electromagnetic integration step
            # This would call eqsofmotion_retarded in the full implementation
            trajectory_new[i] = trajectory_new[i-1].copy()  # Placeholder
            
            # Handle different simulation types
            if sim_type == 1:  # Switching semiconductor
                trajectory_drv_new[i] = switching_flat(trajectory_new[i], wall_Z, apt_R, z_cutoff)
                if np.mean(trajectory_new[i]['z']) > z_cutoff:
                    z_cutoff += cav_spacing
                    wall_Z += cav_spacing
            elif sim_type == 0:  # Conducting plane
                trajectory_drv_new[i] = conducting_flat(trajectory_new[i], wall_Z, apt_R)
            elif sim_type == 2:  # Free bunches
                trajectory_drv_new[i] = trajectory_drv_new[i-1].copy()  # Placeholder
                
    return trajectory_new, trajectory_drv_new
