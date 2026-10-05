"""Float64 free-space electrostatic PIC with optional exact-cloud LW correction.

This is a separate lab-time SI solver. Radiation reaction, boundary fields,
near pairs, and GPU execution remain separate stages.
"""

from .backend import NumpyBackend, PICBackend
from .grid import Grid, Species, PICFields, ElectrostaticPIC
from .correction import CloudCorrection, CorrectionConfig
from .simulation import run_pic

__all__ = [
    "NumpyBackend",
    "PICBackend",
    "Grid",
    "Species",
    "PICFields",
    "ElectrostaticPIC",
    "run_pic",
    "CloudCorrection",
    "CorrectionConfig",
]
