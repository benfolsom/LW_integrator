"""Shared physical constants for the LW integrator core.

All values use the historical ``mm/ns`` system retained by the modern solver
so archived benchmark data and notebook derivations remain comparable.
Constants are surfaced here so tests, examples, and downstream tools can rely
on a single source of truth when performing unit conversions or asserting
physics invariants.
"""

from __future__ import annotations

C_MMNS: float = 299.792458
"""Speed of light in mm/ns (exact value used by the maintained solver)."""

STATCOULOMB_TO_NATIVE_CHARGE: float = 24540.05045243616
"""Convert statcoulombs (cgs esu) to the solver's native charge units."""

ELEMENTARY_CHARGE_STATC: float = 4.803204712570263e-10
"""Elementary charge in statcoulombs (cgs esu), for analysis/conversion only."""

ELEMENTARY_CHARGE: float = 1.178734e-5
"""Elementary charge in native solver units.

Particle states passed to the integrator must use this value, not raw
statcoulombs. It is the historical charge value used by the solver in its
``amu``, ``mm``, ``ns`` unit system.
"""

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
    "ELEMENTARY_CHARGE_STATC",
    "ELECTRON_MASS_AMU",
    "STATCOULOMB_TO_NATIVE_CHARGE",
    "PROTON_MASS_AMU",
    "NUMERICAL_EPSILON",
    "CONVERGENCE_TOLERANCE",
]
