"""Causal, recorded material polarization and delayed-work completion.

Volume quadrature follows retarded_step2i.py and completion_step2k.py.
The force uses field_step2m.py's smooth endpoint reconstruction. Ballistic
completion uses the independently derived cubic endpoint primitive, with an
explicit abrupt stop; it is a diagnostic, not an ordinary radiation model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ._retarded import field, replay
from .mesh import volumes


class MaterialHistory:
    def __init__(self, solver, capacity, order=4, directory=None):
        if solver.wall is None:
            masks = tuple(np.zeros_like(a, dtype=bool) for a in solver.x)
        else:
            masks = solver.wall.masks
        if (
            int(capacity) != capacity
            or capacity < 2
            or int(order) != order
            or order < 1
        ):
            raise ValueError(
                "history needs capacity>=2 and a positive quadrature order"
            )
        self.solver, self.g = solver, solver.g
        self.smooth_endpoints = (
            solver.wall is None or solver.wall.ramp_duration is not None
        )
        self.indices = [np.nonzero(m) for m in masks]
        self.offsets = np.r_[0, np.cumsum([len(i[0]) for i in self.indices])]
        shape = (int(capacity), int(self.offsets[-1]))
        self.times = np.empty(int(capacity))
        if directory is None or shape[1] == 0:
            self.polarization, self.current = np.empty(shape), np.empty(shape)
        else:
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=True)
            self.polarization = np.lib.format.open_memmap(
                directory / "polarization.npy", mode="w+", dtype="float64", shape=shape
            )
            self.current = np.lib.format.open_memmap(
                directory / "current.npy", mode="w+", dtype="float64", shape=shape
            )
        self.count = 0
        self.sealed = False
        nodes, qw = np.polynomial.legendre.leggauss(order)
        fullvol = volumes(self.g)
        r, z, w, cells, components, inactive = [], [], [], [], [], []
        for k, (ri, zi) in enumerate(self.indices):
            rl, rh, zl, zh = self.g.dual_bounds(k)
            rn = self.g.r_half if k == 0 else self.g.r_node
            stop = np.full(len(ri), np.inf)
            wall = solver.wall
            if wall is not None and wall.aperture_radius is not None:
                stop[rn[ri] < wall.aperture_radius] = wall.opening_time
            for x, wx in zip(nodes, qw):
                rr = (rl[ri] + rh[ri]) / 2 + (rh[ri] - rl[ri]) * x / 2
                for y, wy in zip(nodes, qw):
                    r.extend(rr)
                    z.extend((zl[zi] + zh[zi]) / 2 + (zh[zi] - zl[zi]) * y / 2)
                    w.extend(fullvol[k][ri, zi] * wx * wy / 2 * rr / (rl[ri] + rh[ri]))
                    cells.extend(self.offsets[k] + np.arange(len(ri)))
                    components.extend(np.full(len(ri), k))
                    inactive.extend(stop)
        self.geometry = (
            np.asarray(r),
            np.asarray(z),
            np.asarray(w),
            np.asarray(cells, dtype=np.int64),
            np.asarray(components, dtype=np.int64),
            np.asarray(inactive),
        )

    @classmethod
    def from_arrays(cls, solver, times, polarization, current, order=4):
        """Replay a caller's saved record with the same explicit wall geometry.

        Arrays may be read-only memory maps. This does not restore the evolving
        solver state; its grid and wall define the source-cell ordering and
        physical endpoint model. The record is sealed and cannot be appended.
        """
        result = cls(solver, 2, order)
        ts = np.asarray(times, dtype=np.float64)
        p, j = np.asarray(polarization), np.asarray(current)
        shape = (len(ts), int(result.offsets[-1]))
        if (
            ts.ndim != 1
            or len(ts) < 2
            or not np.all(np.isfinite(ts))
            or np.any(np.diff(ts) <= 0)
            or p.shape != shape
            or j.shape != shape
            or p.dtype != np.float64
            or j.dtype != np.float64
            or not np.all(np.isfinite(p))
            or not np.all(np.isfinite(j))
        ):
            raise ValueError(
                "record needs increasing finite times and matching float64 P/J"
            )
        result.times, result.polarization, result.current = ts, p, j
        result.count, result.sealed = len(ts), True
        return result

    def append(self):
        if self.sealed:
            raise ValueError("material history is sealed")
        s, k = self.solver, self.count
        if k >= len(self.times):
            raise ValueError("material history capacity exceeded")
        if k and s.t <= self.times[k - 1]:
            raise ValueError("material times must increase strictly")
        self.times[k] = s.t
        coupling = s.coupling(s.t)
        gd = (
            s.wall.derivative(s.t)
            if s.wall is not None
            else tuple(np.zeros_like(a) for a in s.x)
        )
        for c, index in enumerate(self.indices):
            sl = slice(self.offsets[c], self.offsets[c + 1])
            self.polarization[k, sl] = (coupling[c] * s.x[c])[index]
            self.current[k, sl] = (coupling[c] * s.v[c] + gd[c] * s.x[c])[index]
        self.count += 1

    def seal(self):
        """Declare the finite record before using any terminal continuation."""
        if self.count < 2:
            raise ValueError("retain at least two material endpoints")
        self.sealed = True
        for a in (self.polarization, self.current):
            if isinstance(a, np.memmap):
                a.flush()
        return self

    @property
    def arrays(self):
        return (
            self.times[: self.count],
            self.polarization[: self.count],
            self.current[: self.count],
        )

    def sample_axis(
        self, t, *, beta, z_at_zero=-1.0, delta_z=0.0, velocity=None, stop_duration=1.0
    ):
        """Return Ez and its complete derivative along an axial observer.

        Coordinates use z=z_at_zero+beta*t+delta_z to preserve tiny departures.
        Off-axis consumers use BoundarySnapshot. Smooth preparation/opening is
        required for a finite ordinary radiation ledger; abrupt joins remain
        diagnostics with missing distributional impulses.
        """
        values = [t, beta, z_at_zero, delta_z, stop_duration]
        v = beta if velocity is None else velocity
        if not np.all(np.isfinite(values + [v])) or abs(beta) >= 1 or abs(v) >= 1:
            raise ValueError("observer requires finite times and subluminal velocities")
        if self.count < 2 or stop_duration <= 0:
            raise ValueError(
                "sampling needs two endpoints and a positive stop duration"
            )
        if not self.sealed and len(self.geometry[0]):
            r, z, _, _, _, inactive = self.geometry
            centre = z - z_at_zero - delta_z
            distance = np.hypot(r, beta * t - centre)
            retarded = (
                (1 - beta) * (1 + beta) * t * t
                + 2 * beta * centre * t
                - centre * centre
                - r * r
            ) / (t + distance)
            if np.any((retarded >= self.times[self.count - 1]) & (retarded < inactive)):
                raise ValueError("seal the record to enable terminal continuation")
        return field(
            float(t),
            float(delta_z),
            float(v),
            float(beta),
            *self.arrays,
            *self.geometry,
            float(stop_duration),
            float(z_at_zero),
            self.smooth_endpoints,
        )

    def complete_ballistic_work(
        self, gamma, charge, observation_times, *, z_at_zero=-1.0, temporal_order=4
    ):
        """Work curve, completed work, and explicit terminal-stop contribution.

        The continuation freezes P and stops J with its arrival impulse and
        static tail. Source-window convergence must be checked independently;
        this declaration alone does not certify an indefinitely evolving wall.
        """
        queries = np.asarray(observation_times, dtype=float)
        if (
            not self.sealed
            or queries.ndim != 1
            or len(queries) == 0
            or not np.all(np.isfinite(queries))
            or np.any(queries < 0)
            or np.any(np.diff(queries) <= 0)
            or not np.isfinite(gamma)
            or gamma <= 1
            or not np.isfinite(charge)
            or not np.isfinite(z_at_zero)
            or temporal_order < 1
        ):
            raise ValueError(
                "completion needs a sealed record, gamma>1, and increasing times"
            )
        nodes, qw = np.polynomial.legendre.leggauss(temporal_order)
        curve, drive, terminal = replay(
            *self.arrays, *self.geometry, gamma, charge, queries, nodes, qw, z_at_zero
        )
        return dict(
            time=queries.copy(),
            work=curve,
            completed_work=-drive,
            terminal_continuation_work=terminal,
            continuation="frozen polarization, stopped current, impulse, and static tail",
        )
