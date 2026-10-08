"""Physical field gathering on an EXPERIMENTAL wall-following mesh."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .observers import BoundarySnapshot

if TYPE_CHECKING:
    from .conformal import ConformalPEC


class ConformalSnapshot(BoundarySnapshot):
    """Copy physical components and their physical staggered radii together.

    Interpolate radially on each bracketing axial plane, then axially. Stored
    covariant components must never enter an ordinary cylindrical gather.
    These are total fields; this adapter does not remove the incident field.
    """

    def __init__(self, solver: ConformalPEC) -> None:
        super().__init__(solver.g, solver.t, solver.physical_fields())
        self.radii = tuple(
            a.copy() for a in (solver.radius_r, solver.radius_z, solver.radius_b)
        )
        for a in self.radii:
            a.setflags(write=False)

    def sample(
        self, positions: Any, times: Any = None
    ) -> tuple[np.ndarray, np.ndarray]:
        p = np.asarray(positions, dtype=float)
        if p.ndim != 2 or p.shape[1] != 3 or not np.all(np.isfinite(p)):
            raise ValueError("observer positions need finite shape (N,3)")
        if times is not None and not np.all(
            np.broadcast_to(np.asarray(times), (len(p),)) == self.time
        ):
            raise ValueError("observer times must equal the snapshot time")
        r, z = np.hypot(p[:, 0], p[:, 1]), p[:, 2]
        axial = (self.grid.z_node, self.grid.z_half, self.grid.z_half)
        values = []
        for k, (array, radii, nodes) in enumerate(zip(self.arrays, self.radii, axial)):
            if np.any(z < nodes[0]) or np.any(z > nodes[-1]):
                raise ValueError("observer outside mapped interpolation domain")
            low = np.clip(
                np.searchsorted(nodes, z, side="right") - 1, 0, len(nodes) - 2
            )
            fraction = (z - nodes[low]) / (nodes[low + 1] - nodes[low])
            ends = []
            for columns in (low, low + 1):
                if np.any(r > radii[-1, columns]):
                    raise ValueError("observer outside mapped interpolation domain")
                end = np.empty(len(p))
                for n, column in enumerate(columns):
                    rr, ff = radii[:, column], array[:, column]
                    if k != 1:
                        # Cylindrical symmetry fixes Er=Btheta=0 at r=0.
                        rr, ff = np.r_[0.0, rr], np.r_[0.0, ff]
                    end[n] = np.interp(r[n], rr, ff)
                ends.append(end)
            values.append((1 - fraction) * ends[0] + fraction * ends[1])
        er, ez, bt = values
        axis = np.divide(
            p[:, :2], r[:, None], out=np.zeros((len(p), 2)), where=r[:, None] != 0
        )
        e, b = np.zeros_like(p), np.zeros_like(p)
        e[:, :2], e[:, 2] = er[:, None] * axis, ez
        b[:, 0], b[:, 1] = -bt * axis[:, 1], bt * axis[:, 0]
        return e, b
