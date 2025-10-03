"""Shared physical constants for the LW integrator core.

All values use the historical ``mm/ns`` system adopted by the legacy
codebase, which keeps compatibility with archived benchmark data and notebook
derivations.  Constants are surfaced here so that tests, examples, and
downstream tools can rely on a single source of truth when performing unit
conversions or asserting physics invariants.
"""

from __future__ import annotations

C_MMNS: float = 299.792458
"""Speed of light in mm/ns (exact value used by the legacy solver)."""

ELEMENTARY_CHARGE: float = 1.178734e-5
"""Elementary charge in Gaussian units (amu·mm/ns)."""

ELECTRON_MASS_AMU: float = 5.485799e-4
"""Electron rest mass in atomic mass units (amu)."""

PROTON_MASS_AMU: float = 1.007276466812
"""Proton rest mass in atomic mass units (amu)."""

NUMERICAL_EPSILON: float = 1e-12
"""General-purpose tolerance for floating-point comparisons."""

CONVERGENCE_TOLERANCE: float = 1e-10
"""Default convergence target for iterative self-consistency loops."""

__all__ = [
    "C_MMNS",
    "ELEMENTARY_CHARGE",
    "ELECTRON_MASS_AMU",
    "PROTON_MASS_AMU",
    "NUMERICAL_EPSILON",
    "CONVERGENCE_TOLERANCE",
]
