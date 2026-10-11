"""Native backend-selectable, free-space relativistic electrostatic PIC (SI units).

This is a separate lab-time solver, with an optional exact-cloud LW correction
(CPU backend only). It does not dispatch the LW integrator, radiation reaction,
boundary fields, or the exact-path compute backends.
"""

from .backend import NumpyBackend, PICBackend, select_backend
from .grid import Grid, Species, PICFields, ElectrostaticPIC
from .correction import CloudCorrection, CorrectionConfig
from .nearfield import NearFieldConfig, NearFieldCorrection
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
    "CloudCorrection",
    "CorrectionConfig",
    "NearFieldConfig",
    "NearFieldCorrection",
]
