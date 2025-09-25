"""
Updated Bunch Initialization Module

This module provides particle bunch initialization compatible with the updated
integrator while maintaining consistency with the legacy bunch_inits approach.

Key principles:
- Macroparticles define TOTAL CHARGE of a bunch
- Individual particle mass used (not macroparticle mass)
- Screening effects assumed negligible
- Compatible with amu-mm-ns unit system

Author: Generated for LW_integrator harmonization
Date: 2025-09-25
"""

import numpy as np
from typing import Dict, Tuple
from physics.constants import C_MMNS, ELEMENTARY_CHARGE_GAUSSIAN
from legacy.bunch_inits import init_bunch


def create_updated_bunch_from_energy(
    kinetic_energy_mev: float,
    mass_amu: float,
    charge_sign: int,
    position_z: float,
    transv_momentum_fraction: float = 1e-6,
    transv_position_spread: float = 1e-6,
    particle_count: int = 1,
    stripped_ions: int = 1
) -> Tuple[Dict[str, np.ndarray], float]:
    """
    Create particle bunch from kinetic energy using legacy init_bunch.
    
    This function bridges the energy-based interface with the legacy momentum-based
    initialization while preserving all the original physics.
    
    Args:
        kinetic_energy_mev: Kinetic energy in MeV
        mass_amu: Particle mass in amu
        charge_sign: +1 for proton, -1 for antiproton/electron
        position_z: Initial z position in mm
        transv_momentum_fraction: Transverse momentum as fraction of total
        transv_position_spread: Transverse position spread in mm
        particle_count: Number of particles in bunch
        stripped_ions: Ionization state
        
    Returns:
        Tuple of (bunch_dict, rest_energy_mev)
    """
    
    # Convert kinetic energy to momentum parameter for init_bunch
    rest_energy_mev = mass_amu * 931.494  # MeV
    gamma_target = (kinetic_energy_mev / rest_energy_mev) + 1.0
    beta_target = np.sqrt(1.0 - 1.0/gamma_target**2)
    
    # Calculate starting_Pz parameter (momentum-like quantity)
    starting_Pz = gamma_target * beta_target * C_MMNS
    
    # Set transverse momentum based on fraction
    transv_mom = transv_momentum_fraction * abs(starting_Pz)
    
    # Call legacy init_bunch directly
    bunch_dict, rest_energy = init_bunch(
        starting_distance=position_z,
        transv_mom=transv_mom,
        starting_Pz=starting_Pz,
        stripped_ions=stripped_ions,
        m_particle=mass_amu,
        transv_dist=transv_position_spread,
        pcount=particle_count,
        charge_sign=charge_sign
    )
    
    # Convert scalar q and m to arrays for updated integrator compatibility
    if not hasattr(bunch_dict['q'], '__len__'):
        bunch_dict['q'] = np.array([bunch_dict['q']])
    if not hasattr(bunch_dict['m'], '__len__'):
        bunch_dict['m'] = np.array([bunch_dict['m']])
        
    return bunch_dict, rest_energy


def extract_bunch_properties(bunch: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Extract key properties from a bunch for comparison and analysis.
    
    Args:
        bunch: Bunch dictionary from init_bunch or create_updated_bunch_from_energy
        
    Returns:
        Dictionary with extracted properties
    """
    
    # Handle both array and scalar formats
    def extract_value(field):
        value = bunch[field]
        return value[0] if hasattr(value, '__len__') else value
    
    properties = {
        'position_x': extract_value('x'),
        'position_y': extract_value('y'), 
        'position_z': extract_value('z'),
        'momentum_px': extract_value('Px'),
        'momentum_py': extract_value('Py'),
        'momentum_pz': extract_value('Pz'),
        'momentum_pt': extract_value('Pt'),
        'gamma': extract_value('gamma'),
        'beta_x': extract_value('bx'),
        'beta_y': extract_value('by'),
        'beta_z': extract_value('bz'),
        'charge': extract_value('q'),
        'mass': extract_value('m'),
        'char_time': bunch['char_time'] if 'char_time' in bunch else 0.0,
        'energy_total': extract_value('Pt') * C_MMNS
    }
    
    # Calculate derived quantities
    properties['beta_magnitude'] = np.sqrt(
        properties['beta_x']**2 + 
        properties['beta_y']**2 + 
        properties['beta_z']**2
    )
    
    properties['kinetic_energy_mev'] = (properties['gamma'] - 1.0) * properties['mass'] * 931.494
    
    return properties


def compare_bunch_initialization(
    legacy_bunch: Dict[str, np.ndarray],
    updated_bunch: Dict[str, np.ndarray],
    tolerance: float = 1e-10
) -> Dict[str, float]:
    """
    Compare legacy and updated bunch initialization results.
    
    Args:
        legacy_bunch: Bunch from legacy init_bunch
        updated_bunch: Bunch from create_updated_bunch_from_energy
        tolerance: Numerical tolerance for comparison
        
    Returns:
        Dictionary with comparison results
    """
    
    legacy_props = extract_bunch_properties(legacy_bunch)
    updated_props = extract_bunch_properties(updated_bunch)
    
    comparison = {}
    
    for key in legacy_props:
        if key in updated_props:
            diff = abs(legacy_props[key] - updated_props[key])
            relative_diff = diff / abs(legacy_props[key]) if legacy_props[key] != 0 else diff
            
            comparison[f'{key}_difference'] = diff
            comparison[f'{key}_relative_difference'] = relative_diff
            comparison[f'{key}_matches'] = diff < tolerance
    
    # Overall match status
    comparison['all_match'] = all(
        comparison.get(f'{key}_matches', False) 
        for key in ['gamma', 'momentum_pt', 'char_time', 'charge', 'mass']
    )
    
    return comparison