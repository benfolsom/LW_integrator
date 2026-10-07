"""Experimental one-way, axisymmetric boundary fields (CPU float64).

Prescribed axial LW fields drive grid-aligned Drude walls. Observers consume
only the scattered field. No coupled particle feedback or default PIC changes.
All solver quantities use normalized Heaviside–Lorentz units with c=1; the
optional PIC consumer requires explicit BoundaryUnits for SI conversion.
"""

from .mesh import AxisymmetricGrid
from .materials import AlignedWall, DrudeMedium, DrudeWall
from .incident import BallisticDrive, PrescribedDrive
from .solver import ScatteredFieldSolver
from .history import MaterialHistory
from .observers import BoundarySnapshot, BoundaryUnits, add_boundary_fields
from .diagnostics import GridLedger, PhysicalLedger, Surface, continuous_storage
from .particle import replay_particle

__all__ = [
    "AxisymmetricGrid",
    "AlignedWall",
    "DrudeMedium",
    "DrudeWall",
    "BallisticDrive",
    "PrescribedDrive",
    "ScatteredFieldSolver",
    "MaterialHistory",
    "BoundarySnapshot",
    "BoundaryUnits",
    "add_boundary_fields",
    "GridLedger",
    "PhysicalLedger",
    "Surface",
    "continuous_storage",
    "replay_particle",
]
