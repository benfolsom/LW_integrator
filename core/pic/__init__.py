"""Native backend-selectable, free-space relativistic electrostatic PIC (SI units).

This is a separate lab-time solver. It does not dispatch the LW integrator,
radiation reaction, boundary fields, or the exact-path compute backends.
"""

from .backend import NumpyBackend, PICBackend, select_backend
from .grid import Grid, Species, PICFields, ElectrostaticPIC
from .simulation import run_pic

__all__ = [
    "NumpyBackend",
    "PICBackend",
    "select_backend",
    "Grid",
    "Species",
    "PICFields",
    "ElectrostaticPIC",
    "run_pic",
]
