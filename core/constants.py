"""Shared physical constants for the LW integrator core."""

from __future__ import annotations

C_MMNS: float = 299.792458  # Speed of light in mm / ns (exact legacy value)
ELEMENTARY_CHARGE: float = 1.178734e-5  # Gaussian units in amu·mm/ns
ELECTRON_MASS_AMU: float = 5.485799e-4
PROTON_MASS_AMU: float = 1.007276466812
NUMERICAL_EPSILON: float = 1e-12
CONVERGENCE_TOLERANCE: float = 1e-10

__all__ = [
	"C_MMNS",
	"ELEMENTARY_CHARGE",
	"ELECTRON_MASS_AMU",
	"PROTON_MASS_AMU",
	"NUMERICAL_EPSILON",
	"CONVERGENCE_TOLERANCE",
]
