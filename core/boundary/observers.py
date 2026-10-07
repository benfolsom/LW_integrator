"""Plain observers and an optional SI native-PIC consumer; no particle feedback."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.pic.grid import Grid, Species
from core.pic.kernels import C, EPSILON_0

from .mesh import cpu_backend


@dataclass(frozen=True)
class BoundaryUnits:
    """Explicit SI scales for normalized lengths and charges (c=eps0=1)."""

    length_m: float
    charge_c: float

    def __post_init__(self):
        if (
            not np.all(np.isfinite([self.length_m, self.charge_c]))
            or min(self.length_m, self.charge_c) <= 0
        ):
            raise ValueError("SI length and charge scales must be finite and positive")

    @property
    def time_s(self):
        return self.length_m / C

    @property
    def electric_v_m(self):
        return self.charge_c / (EPSILON_0 * self.length_m**2)

    @property
    def energy_j(self):
        return self.charge_c**2 / (EPSILON_0 * self.length_m)

    @property
    def magnetic_t(self):
        return self.electric_v_m / C

    @property
    def mass_kg(self):
        return self.energy_j / C**2

    @property
    def momentum_kg_m_s(self):
        return self.energy_j / C


class BoundarySnapshot:
    def __init__(self, grid, time, arrays):
        shapes = ((grid.nr, grid.nz + 1), (grid.nr + 1, grid.nz), (grid.nr, grid.nz))
        if (
            not np.isfinite(time)
            or len(arrays) != 3
            or any(
                np.shape(a) != shape or not np.all(np.isfinite(a))
                for a, shape in zip(arrays, shapes)
            )
        ):
            raise ValueError(
                "snapshot needs a finite time and three matching staggered fields"
            )
        self.grid, self.time = grid, float(time)
        self.arrays = tuple(np.array(a, dtype=np.float64, copy=True) for a in arrays)
        for a in self.arrays:
            a.setflags(write=False)

    def sample(self, positions, times=None):
        """Normalized Cartesian E and cB at positions and this snapshot's time.

        Keep separate snapshots for different times. No implicit extrapolation
        outside the mesh or in time is allowed. At the axis Er and Btheta=0.
        """
        from .diagnostics import interpolate

        p = np.asarray(positions, dtype=float)
        if p.ndim != 2 or p.shape[1] != 3 or not np.all(np.isfinite(p)):
            raise ValueError("observer positions need finite shape (N,3)")
        if times is not None:
            ts = np.broadcast_to(np.asarray(times, dtype=float), (len(p),))
            if not np.all(ts == self.time):
                raise ValueError("observer times must equal the snapshot time")
        r = np.hypot(p[:, 0], p[:, 1])
        g = self.grid
        # Require all component interpolation stencils within the retained grid.
        if (
            np.any(r > g.r_half[-1])
            or np.any(p[:, 2] < g.z_half[0])
            or np.any(p[:, 2] > g.z_half[-1])
        ):
            raise ValueError("observer outside boundary interpolation domain")
        er, ez, bt = interpolate(g, self.arrays, r, p[:, 2])
        axis = np.divide(
            p[:, :2], r[:, None], out=np.zeros((len(p), 2)), where=r[:, None] != 0
        )
        e, b = np.zeros_like(p), np.zeros_like(p)
        e[:, :2], e[:, 2] = er[:, None] * axis, ez
        b[:, 0], b[:, 1] = -bt * axis[:, 1], bt * axis[:, 0]
        return e, b

    def pic_consumer(self, grid: Grid, units: BoundaryUnits, backend=None):
        """Resample once onto a lab-frame PIC grid; reuse CIC or TSC gather.

        The boundary mesh is a fixed lab r–z grid. Bunch-local PIC rest grids
        are not directly interchangeable; this explicit SI lab-grid bridge is
        the consumer seam. Source populations never enter the observer gather.
        """
        backend = cpu_backend(backend)
        indices = backend.xp.indices(grid.shape).reshape(3, -1).T
        positions = grid.origin_m + indices * grid.spacing_m
        e, b = self.sample(positions / units.length_m)
        return PICBoundaryFields(
            grid,
            e.reshape((*grid.shape, 3)) * units.electric_v_m,
            b.reshape((*grid.shape, 3)) * units.electric_v_m / C,
            self.time * units.time_s,
            backend,
        )


@dataclass
class PICBoundaryFields:
    grid: Grid
    electric: np.ndarray
    magnetic: np.ndarray
    time_s: float
    backend: object

    def sample(self, species: Species, time_s):
        if time_s != self.time_s:
            raise ValueError("PIC observer time must match the boundary snapshot")
        return (
            self.grid.gather(species.position_m, self.electric, self.backend),
            self.grid.gather(species.position_m, self.magnetic, self.backend),
        )


def add_boundary_fields(electric, magnetic, species, time_s, boundary=None):
    """Opt-in field addition before a caller's push; None returns original arrays.

    Neither the PIC simulation nor its default radiation setting is changed.
    Coupled feedback would need to rebuild the prescribed incident history and
    solve material again; that iteration is deliberately outside this API.
    """
    if boundary is None:
        return electric, magnetic
    e, b = boundary.sample(species, time_s)
    return electric + e, magnetic + b
