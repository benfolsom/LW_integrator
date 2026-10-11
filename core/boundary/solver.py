"""Material-only midpoint Drude update with an exact axial vacuum rotation.

Ported from direct_step2h.py and material_step2j.py. Only scattered fields
are evolved; no source current or auxiliary vacuum grid is deposited.
"""

from __future__ import annotations

import numpy as np
from scipy.fft import dst, dct, idst, idct
from .mesh import Fields, cpu_backend
from .materials import DrudeWall


class ScatteredFieldSolver:
    def __init__(
        self, g, medium, dt, history, coupling=None, average=False, backend=None
    ):
        self.backend = cpu_backend(backend)
        if not np.isfinite(dt) or dt <= 0 or dt > min(g.dr, g.dz) / 2:
            raise ValueError("require 0 < dt <= min(dr,dz)/2")
        if coupling is not None and not isinstance(coupling, DrudeWall):
            raise ValueError(
                "material geometry must use an explicit grid-aligned DrudeWall"
            )
        if coupling is not None and coupling.grid != g:
            raise ValueError("wall and solver grids must match")
        if coupling is not None and history is None:
            raise ValueError("a material wall needs an explicit incident provider")
        if coupling is not None and coupling.aperture_radius is not None:
            ticks = coupling.opening_time / dt
            if not np.isclose(ticks, round(ticks), rtol=0, atol=1e-10):
                raise ValueError(
                    "aperture removal must coincide with a material time endpoint"
                )
        self.g, self.medium, self.dt, self.history = g, medium, dt, history
        self.wall = coupling
        self.f = Fields(g, self.backend)
        xp = self.backend.xp
        self.x = [xp.zeros_like(self.f.er), xp.zeros_like(self.f.ez)]
        self.v = [xp.zeros_like(self.f.er), xp.zeros_like(self.f.ez)]
        self.coupling = coupling or (lambda t: tuple(xp.zeros_like(x) for x in self.x))
        self.projection = "edge" if average else "volume"
        self.t = 0.0
        self.iterations = 0
        self.wave_numbers = np.pi * np.arange(1, g.nz) / (g.nz * g.dz)
        self.symbol = 2 / dt * np.tan(dt * self.wave_numbers / 2)
        self.fraction = np.arange(1, g.nz) / g.nz
        for rate, shape in zip(
            (medium.absorb_r, medium.absorb_z, medium.absorb_b),
            (self.f.er.shape, self.f.ez.shape, self.f.bt.shape),
        ):
            if np.broadcast_shapes(np.shape(rate), shape) != shape:
                raise ValueError("absorber arrays must match staggered grid shapes")
        if history is not None and hasattr(history, "g") and history.g is not g:
            raise ValueError("incident provider and solver must use the same grid")

    def axial_forward(self, er):
        left, right = er[:, :1], er[:, -1:]
        lift = left + (right - left) * self.fraction[None, :]
        spectrum = np.zeros((er.shape[0], self.g.nz))
        spectrum[:, 1:] = (
            dst(er[:, 1:-1] - lift, type=1, norm="ortho", workers=1) * self.symbol
        )
        return idct(spectrum, type=2, norm="ortho", workers=1) + (right - left) / (
            self.g.nz * self.g.dz
        )

    def axial_reverse(self, bt):
        spectrum = dct(bt, type=2, norm="ortho", workers=1)
        result = np.zeros((bt.shape[0], self.g.nz + 1))
        result[:, 1:-1] = idst(
            spectrum[:, 1:] * self.symbol, type=1, norm="ortho", workers=1
        )
        return result

    def ce(self, g, er, ez):
        return (ez[1:] - ez[:-1]) / g.dr - self.axial_forward(er)

    def curl_b(self, f):
        g = self.g
        rb = g.r_half[:, None] * f.bt
        cz = np.zeros_like(f.ez)
        cz[1:-1] = (rb[1:] - rb[:-1]) / (g.r_node[1:-1, None] * g.dr)
        cz[0] = 4 * f.bt[0] / g.dr
        return self.axial_reverse(f.bt), cz

    def state(self):
        return (
            self.t,
            self.f.er.copy(),
            self.f.ez.copy(),
            self.f.bt.copy(),
            [a.copy() for a in self.x],
            [a.copy() for a in self.v],
        )

    def restore(self, state):
        t, er, ez, bt, x, v = state
        self.t, self.f.er, self.f.ez, self.f.bt = t, er.copy(), ez.copy(), bt.copy()
        self.x, self.v = [a.copy() for a in x], [a.copy() for a in v]

    def step(self):
        g, dt, m = self.g, self.dt, self.medium
        t0, t1 = self.t, self.t + self.dt
        e0, b0 = (self.f.er.copy(), self.f.ez.copy()), self.f.bt.copy()
        g0, g1 = self.coupling(t0), self.coupling(t1)
        gh = tuple((a + b) / 2 for a, b in zip(g0, g1))
        gd = tuple((b - a) / dt for a, b in zip(g0, g1))
        masks = tuple((a != 0) | (b != 0) for a, b in zip(g0, g1))
        if (
            self.history is not None
            and hasattr(self.history, "position")
            and getattr(self.history, "q", 0) != 0
        ):
            positions = [self.history.position(t) for t in (t0, (t0 + t1) / 2, t1)]
            crossing = (g.z_node[:-1] <= max(positions)) & (
                g.z_node[1:] >= min(positions)
            )
            if np.any(masks[1][0] & crossing):
                raise ValueError(
                    "point source crossing local Drude material is undefined; "
                    "provide an aperture or a spatial material cutoff"
                )
        if not any(np.any(mask) for mask in masks):
            incident = tuple(np.zeros_like(a) for a in e0)
        elif self.projection == "volume":
            a = self.history.material_volume(g, t0, masks)
            b = self.history.material_volume(g, t1, masks)
            incident = tuple((x + y) / 2 for x, y in zip(a, b))
        else:
            incident = self.history.edges_mid(g, t0, t1, masks)
        den = 1 + m.nu * dt / 2
        coefficients = []
        for a, i, v, x, gg, gm, dg, absorb in zip(
            e0, incident, self.v, self.x, g1, gh, gd, (m.absorb_r, m.absorb_z)
        ):
            physical = gg * gm * dt / (2 * den)
            effective = physical + absorb
            coefficients.append(
                (
                    (1 - dt * effective / 2) * a
                    - dt * physical * i
                    - dt * (gg * v / den + dg * x),
                    1 + dt * effective / 2,
                )
            )
        electric = tuple(a.copy() for a in e0)
        f = Fields(g)
        for iteration in range(50):
            es = tuple((a + b) / 2 for a, b in zip(e0, electric))
            bm = (b0 + dt / 2 * self.ce(g, *es)) / (1 + dt / 2 * np.asarray(m.absorb_b))
            f.bt = bm
            new = tuple(
                (base + dt * curl) / coeff
                for (base, coeff), curl in zip(coefficients, self.curl_b(f))
            )
            new[0][:, [0, -1]] = 0.0
            new[1][-1] = 0.0
            error = max(np.max(abs(a - b)) for a, b in zip(new, electric))
            electric = new
            if error < 2e-13 * max(1e-12, *(np.max(abs(a)) for a in electric)):
                break
        else:
            raise RuntimeError("Scattered midpoint iteration failed")
        es = tuple((a + b) / 2 for a, b in zip(e0, electric))
        bm = (b0 + dt / 2 * self.ce(g, *es)) / (1 + dt / 2 * np.asarray(m.absorb_b))
        details, current = [], []
        for k, (e, i, gg, gm, dg) in enumerate(zip(es, incident, g1, gh, gd)):
            et = e + i
            vm = (self.v[k] + dt / 2 * gm * et) / den
            xm = self.x[k] + dt / 2 * vm
            self.v[k] = 2 * vm - self.v[k]
            self.x[k] += dt * vm
            details.append(dict(e_total=et, v_mid=vm, x_mid=xm, gdot=dg))
            current.append(gm * vm + dg * xm)
        self.f.er, self.f.ez, self.f.bt = *electric, 2 * bm - b0
        self.t = t1
        self.iterations = max(self.iterations, iteration + 1)
        return dict(
            e0=e0,
            e1=electric,
            b0=b0,
            b1=self.f.bt,
            es=es,
            bs=bm,
            incident=incident,
            material=details,
            current=current,
            t0=t0,
            t1=t1,
        )

    def snapshot(self):
        """Read-only field copy at the current normalized lab time."""
        from .observers import BoundarySnapshot

        return BoundarySnapshot(self.g, self.t, (self.f.er, self.f.ez, self.f.bt))
