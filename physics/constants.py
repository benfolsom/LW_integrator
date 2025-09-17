"""
Physics Constants and Unit Conversions

This module provides fundamental physics constants and unit conversion utilities
for the Lienard-Wiechert integrator system using Benjamin Folsom's 
amu-mm-ns unit system with Gaussian electromagnetic units.

Author: Ben Folsom (original design)
Date: 2025-09-17
"""

import numpy as np

# Speed of light in mm/ns (exactly)
C_MMNS = 299.792458  # mm/ns

# Numerical precision constants
NUMERICAL_EPSILON = 1e-12
CONVERGENCE_TOLERANCE = 1e-10

# Elementary charge in Gaussian units (legacy Benjamin Folsom factor)
# This factor converts from SI to Gaussian to amu-mm-ns units
ELEMENTARY_CHARGE_GAUSSIAN = 1.178734e-5  # amu*mm/ns

# Particle masses in amu
ELECTRON_MASS_AMU = 5.485799e-4  # amu
PROTON_MASS_AMU = 1.007276466812  # amu

def gamma_to_beta(gamma):
    """Convert relativistic gamma factor to beta (v/c)."""
    return np.sqrt(1.0 - 1.0/(gamma**2))

def beta_to_gamma(beta):
    """Convert beta (v/c) to relativistic gamma factor."""
    return 1.0 / np.sqrt(1.0 - beta**2)

def energy_to_gamma(energy_mev, mass_amu):
    """Convert kinetic energy (MeV) to gamma factor."""
    # Convert MeV to amu*c^2 units
    rest_energy_mev = mass_amu * 931.494  # MeV/c^2 * c^2
    total_energy_mev = energy_mev + rest_energy_mev
    return total_energy_mev / rest_energy_mev

def momentum_magnitude(gamma, mass_amu):
    """Calculate momentum magnitude in amu*mm/ns units."""
    beta = gamma_to_beta(gamma)
    return gamma * mass_amu * beta * C_MMNS