"""
Physical Constants for Lienard-Wiechert Simulations

CAI: Fundamental physical constants in mm⋅ns⋅amu units for
electromagnetic field calculations and relativistic dynamics.

Author: Ben Folsom (human oversight)
Date: 2025-09-12
"""

import numpy as np

# Speed of light
C_MMNS = 299.792458  # mm/ns

# Particle masses (MeV/c²)
ELECTRON_MASS = 0.511  # MeV/c²
PROTON_MASS = 938.3    # MeV/c²
NEUTRON_MASS = 939.6   # MeV/c²

# Elementary charge and electromagnetic constants
ELEMENTARY_CHARGE = 1.0  # Elementary charge units
FINE_STRUCTURE_CONSTANT = 1.0/137.036  # Dimensionless

# Coulomb constant in natural units
# k = 1.44 MeV⋅mm / (elementary charge)²
COULOMB_CONSTANT = 1.44e-3  # MeV⋅mm/e²

# Conversion factors
MEV_TO_JOULE = 1.602176634e-13  # J/MeV
METER_TO_MM = 1000.0  # mm/m
SECOND_TO_NS = 1e9    # ns/s

# Physical scales
CLASSICAL_ELECTRON_RADIUS = 2.818e-12  # mm
COMPTON_WAVELENGTH = 3.862e-10  # mm (electron)
BOHR_RADIUS = 5.292e-8  # mm

# Energy scales
ELECTRON_REST_ENERGY = ELECTRON_MASS  # MeV
PROTON_REST_ENERGY = PROTON_MASS      # MeV

# Typical scales for simulations
TYPICAL_TIMESTEP = 1e-6  # ns
TYPICAL_DISTANCE = 1e-6  # mm (1 nm)
TYPICAL_ENERGY = 1000.0  # MeV (1 GeV)
