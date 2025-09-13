"""
Physical Constants for Lienard-Wiechert Simulations

CAI: Physical constants in Gaussian CGS units for electromagnetic field 
calculations and relativistic dynamics. This provides the most natural
unit system for electromagnetic calculations.

Author: Ben Folsom (human oversight)
Date: 2025-09-13
"""

import numpy as np

# ==============================================================================
# GAUSSIAN CGS UNIT SYSTEM (Primary)
# ==============================================================================

# Speed of light
C_CGS = 2.998e10  # cm/s
C_MMNS = 299.792458  # mm/ns (legacy compatibility)

# Fundamental constants in Gaussian CGS
ELEMENTARY_CHARGE_ESU = 4.803e-10  # esu (statcoulomb)
ELECTRON_MASS_G = 9.109e-28  # g
PROTON_MASS_G = 1.673e-24   # g

# Standard aliases for backward compatibility and convenience
ELEMENTARY_CHARGE = ELEMENTARY_CHARGE_ESU
ELECTRON_MASS = ELECTRON_MASS_G 
PROTON_MASS = PROTON_MASS_G

# Particle masses in energy units (MeV) - for relativistic calculations
ELECTRON_MASS_MEV = 0.511    # MeV/c²
PROTON_MASS_MEV = 938.3      # MeV/c²
NEUTRON_MASS_MEV = 939.6     # MeV/c²
LEAD_ION_MASS_AMU = 207.2    # amu

# Energy conversion
MEV_TO_ERG = 1.602e-6        # erg/MeV
ERG_TO_MEV = 1.0 / MEV_TO_ERG

# Length scales
CM_TO_MM = 10.0              # mm/cm
CLASSICAL_ELECTRON_RADIUS = 2.818e-13  # cm
COMPTON_WAVELENGTH = 2.426e-10         # cm (electron)
BOHR_RADIUS = 5.292e-9                 # cm

# Time scales  
S_TO_NS = 1e9                # ns/s
CHARACTERISTIC_TIME_SCALE = 1e-18  # s (typical for high-energy interactions)

# ==============================================================================
# MM⋅NS⋅AMU LEGACY UNIT SYSTEM (For backward compatibility)
# ==============================================================================

# Particle masses in amu
ELECTRON_MASS_AMU = 0.0005485  # amu
PROTON_MASS_AMU = 1.007319468  # amu

# Charge conversion factors (for legacy code)
ELEMENTARY_CHARGE_LEGACY = 1.0  # Elementary charge units

# Coulomb constant in legacy units
COULOMB_CONSTANT_LEGACY = 1.44e-3  # MeV⋅mm/e²

# ==============================================================================
# DERIVED QUANTITIES AND TYPICAL SCALES
# ==============================================================================

# Fine structure constant (dimensionless, same in all units)
FINE_STRUCTURE_CONSTANT = 1.0/137.036

# Typical simulation scales
TYPICAL_TIMESTEP_NS = 1e-6     # ns
TYPICAL_DISTANCE_MM = 1e-6     # mm (1 nm)
TYPICAL_ENERGY_MEV = 1000.0    # MeV (1 GeV)
TYPICAL_GAMMA = 1000.0         # For ultra-relativistic particles

# Standard aliases for legacy compatibility
TYPICAL_TIMESTEP = TYPICAL_TIMESTEP_NS * 1e-9  # Convert to seconds
TYPICAL_DISTANCE = TYPICAL_DISTANCE_MM * 0.1   # Convert to cm

# Coulomb constant in Gaussian CGS units (k=1 in Gaussian units)
COULOMB_CONSTANT = 1.0  # dimensionless in Gaussian CGS

# Numerical precision
NUMERICAL_EPSILON = 1e-15      # For avoiding division by zero
CONVERGENCE_TOLERANCE = 1e-10  # For iterative algorithms

# ==============================================================================
# UNIT CONVERSION FUNCTIONS
# ==============================================================================

def gamma_to_beta(gamma: float) -> float:
    """Convert Lorentz factor to velocity in units of c."""
    return np.sqrt(1 - 1/gamma**2)

def beta_to_gamma(beta: float) -> float:
    """Convert velocity (in units of c) to Lorentz factor."""
    return 1.0 / np.sqrt(1 - beta**2)

def kinetic_to_betagamma(kinetic_energy_mev: float, rest_energy_mev: float) -> tuple[float, float]:
    """
    Convert kinetic energy to beta and gamma.
    
    Args:
        kinetic_energy_mev: Kinetic energy in MeV
        rest_energy_mev: Rest mass energy in MeV
    
    Returns:
        (beta, gamma) tuple
    """
    gamma = kinetic_energy_mev / rest_energy_mev + 1
    beta = gamma_to_beta(gamma)
    return beta, gamma

def mev_to_gamma(total_energy_mev: float, rest_mass_mev: float) -> float:
    """Convert total energy to Lorentz factor."""
    return total_energy_mev / rest_mass_mev
