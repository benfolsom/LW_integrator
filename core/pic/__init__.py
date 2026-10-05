"""Stage 1, float64, free-space relativistic electrostatic PIC (SI units).

This is a separate lab-time solver. It does not dispatch the LW integrator,
radiation reaction, boundary fields, or a GPU backend.
"""

from .backend import NumpyBackend, PICBackend
from .grid import Grid, Species, PICFields, ElectrostaticPIC
from .simulation import run_pic

__all__ = [
    "NumpyBackend",
    "PICBackend",
    "Grid",
    "Species",
    "PICFields",
    "ElectrostaticPIC",
    "run_pic",
]
