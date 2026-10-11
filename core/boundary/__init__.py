"""Experimental axisymmetric boundary fields (CPU float64).

Prescribed axial LW fields drive grid-aligned Drude walls. Observers consume
only the scattered field. Smooth mapped PEC and finite-window axial two-way
coupling require explicit opt-in; no production integrator/PIC defaults change.
All solver quantities use normalized Heaviside–Lorentz units with c=1; the
optional PIC consumer requires explicit BoundaryUnits for SI conversion.
"""

from .axial_particle import integrate_axial_particle
from .conformal import ConformalPEC, select_pec
from .coupling import (
    AxialBoundaryForce,
    AxialParticle,
    AxialTrajectory,
    TwoWayBoundaryCoupling,
)
from .diagnostics import GridLedger, PhysicalLedger, Surface, continuous_storage
from .history import MaterialHistory
from .incident import BallisticDrive, PrescribedDrive
from .materials import AlignedWall, DrudeMedium, DrudeWall
from .mesh import AxisymmetricGrid
from .observers import BoundarySnapshot, BoundaryUnits, add_boundary_fields
from .particle import replay_particle
from .solver import ScatteredFieldSolver

__all__ = [
    "AlignedWall",
    "AxialBoundaryForce",
    "AxialParticle",
    "AxialTrajectory",
    "AxisymmetricGrid",
    "BallisticDrive",
    "BoundarySnapshot",
    "BoundaryUnits",
    "ConformalPEC",
    "DrudeMedium",
    "DrudeWall",
    "GridLedger",
    "MaterialHistory",
    "PhysicalLedger",
    "PrescribedDrive",
    "ScatteredFieldSolver",
    "Surface",
    "TwoWayBoundaryCoupling",
    "add_boundary_fields",
    "continuous_storage",
    "integrate_axial_particle",
    "replay_particle",
    "select_pec",
]
