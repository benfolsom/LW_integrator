"""Restricted PEC Gaussian-current wake reference (wake_step2f.py).

This reference evolves total fields, unlike the material-only LW solver.
It is retained to reproduce the independent WarpX-qualified wake family.
"""

from __future__ import annotations

import numpy as np
import time
from scipy.fft import dst, idct
from scipy.special import erf
from .mesh import AxisymmetricGrid, Fields, volumes
from .materials import DrudeMedium
from .solver import ScatteredFieldSolver


class PECWakeSolver(ScatteredFieldSolver):
    """Project both curls onto the same vacuum magnetic cells and PEC edges.

    This defines the geometry as part of the operator, rather than deleting
    magnetic energy after the update. It preserves the closed-grid energy
    adjoint. Its discontinuous steps still need numerical convergence checks.
    """

    def __init__(self, g, dt: float, segments) -> None:
        super().__init__(g, DrudeMedium(), dt, None, None, False)
        radius = np.empty(g.nz)
        last = g.z0
        for end, pipe_radius in segments:
            g.require_aligned(end, "z")
            if not last < end <= g.z_node[-1]:
                raise ValueError("pipe segments must be ordered axial intervals")
            radius[(g.z_half >= last) & (g.z_half < end)] = pipe_radius
            last = end
        if not np.isclose(last, g.z_node[-1]):
            raise ValueError("pipe segments must cover the axial mesh")
        if radius.shape != (g.nz,):
            raise ValueError("one pipe radius per axial cell is required")
        g.require_aligned(radius, "r")
        if np.any(radius <= 0) or np.any(radius > g.nr * g.dr):
            raise ValueError("pipe radii must lie inside the grid")
        if g.dr != g.dz:
            raise ValueError("PEC wake reference currently requires an isotropic mesh")
        self.bmask = g.r_half[:, None] < radius[None, :] - 1e-12
        self.rmask = np.zeros((g.nr, g.nz + 1), bool)
        self.rmask[:, 1:-1] = self.bmask[:, :-1] & self.bmask[:, 1:]
        self.zmask = np.zeros((g.nr + 1, g.nz), bool)
        self.zmask[0] = self.bmask[0]
        self.zmask[1:-1] = self.bmask[:-1] & self.bmask[1:]
        self.volumes = volumes(g)

    def ce(self, g: AxisymmetricGrid, er: np.ndarray, ez: np.ndarray) -> np.ndarray:
        return super().ce(g, er * self.rmask, ez * self.zmask) * self.bmask

    def curl_b(self, f: Fields) -> tuple[np.ndarray, np.ndarray]:
        ff = Fields(self.g)
        ff.bt = f.bt * self.bmask
        cr, cz = super().curl_b(ff)
        return cr * self.rmask, cz * self.zmask

    def advance(self, j: np.ndarray) -> tuple[np.ndarray, float]:
        e0 = (self.f.er, self.f.ez)
        b0 = self.f.bt
        er, ez = (e.copy() for e in e0)
        ff = Fields(self.g)
        for n in range(60):
            bm = b0 + self.dt / 2 * self.ce(self.g, (e0[0] + er) / 2, (e0[1] + ez) / 2)
            ff.bt = bm
            cr, cz = self.curl_b(ff)
            nr = (e0[0] + self.dt * cr) * self.rmask
            nz = (e0[1] + self.dt * (cz - j)) * self.zmask
            error = max(np.max(abs(nr - er)), np.max(abs(nz - ez)))
            er, ez = nr, nz
            if error < 1e-12 * max(1.0, np.max(abs(er)), np.max(abs(ez))):
                break
        else:
            raise RuntimeError("PEC midpoint solve failed")
        self.f.er, self.f.ez, self.f.bt = er, ez, 2 * bm - b0
        self.t += self.dt
        em = (e0[1] + ez) / 2
        return em, float(self.dt * np.sum(j * em * self.volumes[1]))

    def energy(self) -> float:
        return 0.5 * sum(
            float(np.sum(a * a * v))
            for a, v in zip((self.f.er, self.f.ez, self.f.bt), self.volumes)
        )


def gaussian(z: float | np.ndarray, center: float, sigma: float) -> float | np.ndarray:
    return np.exp(-0.5 * ((z - center) / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma)


def node_shape(g: AxisymmetricGrid, center: float, sigma: float) -> np.ndarray:
    return (
        erf((g.z_node + g.dz / 2 - center) / (np.sqrt(2) * sigma))
        - erf((g.z_node - g.dz / 2 - center) / (np.sqrt(2) * sigma))
    ) / (2 * g.dz)


def gaussian_wake(
    case, *, h=0.02, sigma=0.03, extent=12.0, end=8.0, dt_fraction=0.15, core=0.06
):
    """Saved PEC reference family with an outgoing-pipe completion.

    A prescribed continuous beta=1 disk/Gaussian current drives total fields.
    This reference path uses spectral continuity, not the production LW drive.
    No particle push, radiation model, reference amplitude, or fitted offset.
    """
    if (
        case not in ("pillbox", "step_out", "step_in", "pipe")
        or not np.all(np.isfinite([h, sigma, extent, end, dt_fraction, core]))
        or min(h, sigma, extent, core, dt_fraction) <= 0
    ):
        raise ValueError(
            "wake reference needs a supported geometry and positive parameters"
        )
    a, b, gap = 0.4, 1.2, 0.4
    grid = AxisymmetricGrid(h, h, round(b / h), round(2 * extent / h), -extent)
    if not np.isclose(grid.z_node[-1], extent):
        raise ValueError("wake extent must align with the mesh")
    if case == "pillbox":
        segments = [(-gap / 2, a), (gap / 2, b), (extent, a)]
    elif case == "step_out":
        segments = [(0.0, a), (extent, 0.8)]
    elif case == "step_in":
        segments = [(0.0, 0.8), (extent, a)]
    else:
        segments = [(extent, a)]
    solver = PECWakeSolver(grid, h * dt_fraction, segments)
    # A continuous finite radial disk; its exact enclosed charge fixes the
    # radial grid Gauss density, including the axis disk. No point singularity.
    enclosed = np.minimum((grid.r_half / core) ** 2, 1.0)
    coeff = enclosed / (2 * np.pi * grid.r_half)
    rho_r = np.zeros(grid.nr + 1)
    rho_r[0] = 4 * coeff[0] / h
    rho_r[1:-1] = (grid.r_half[1:] * coeff[1:] - grid.r_half[:-1] * coeff[:-1]) / (
        grid.r_node[1:-1] * h
    )
    start = -2.0
    lam0 = node_shape(grid, start, sigma)
    bh = (
        erf((grid.z_node[1:] - start) / (np.sqrt(2) * sigma))
        - erf((grid.z_node[:-1] - start) / (np.sqrt(2) * sigma))
    ) / (2 * h)
    solver.f.er = coeff[:, None] * lam0[None, :] * solver.rmask
    solver.f.bt = coeff[:, None] * bh[None, :] * solver.bmask
    solver.t = start
    initial = solver.energy()
    work = 0.0
    separations = np.linspace(-5 * sigma, max(0.6, 8 * sigma), 401)
    wake = np.zeros_like(separations)
    history = []
    begin = time.perf_counter()
    steps = round((end - start) / solver.dt)
    for n in range(steps):
        t0 = solver.t
        t1 = t0 + solver.dt
        lam1 = node_shape(grid, t1, sigma)
        # Exact spectral continuity: d rho/dt - reverse(Jz)=0. The zero
        # mode carries the physical integrated current along the finite pipe.
        spectrum = np.zeros(grid.nz)
        spectrum[1:] = (
            dst((lam1 - lam0)[1:-1] / solver.dt, type=1, norm="ortho") / solver.symbol
        )
        spectrum[0] = 1 / (2 * extent) * np.sqrt(grid.nz)
        j = rho_r[:, None] * idct(spectrum, type=2, norm="ortho")[None, :]
        ez, dwork = solver.advance(j)
        work += dwork
        witness = (t0 + t1) / 2 - separations
        wake -= solver.dt * np.interp(witness, grid.z_half, ez[0], left=0, right=0)
        if n % 10 == 0 or n == steps - 1:
            energy = solver.energy()
            history.append([solver.t, work, energy - initial, energy - initial + work])
        lam0 = lam1
    lam = gaussian(separations, 0, sigma)
    loss = float(np.trapezoid(wake * lam, separations))
    radius = 0.8 if case == "step_out" else 0.4
    if np.min(solver.t - separations) <= gap / 2:
        raise ValueError("all witnesses must have entered the uniform output pipe")
    keep = grid.r_half < radius - 1e-12
    base_er = coeff[:, None] * node_shape(grid, solver.t, sigma)[None, :]
    shape_b = (
        erf((grid.z_node[1:] - solver.t) / (np.sqrt(2) * sigma))
        - erf((grid.z_node[:-1] - solver.t) / (np.sqrt(2) * sigma))
    ) / (2 * h)
    base_bt = coeff[:, None] * shape_b[None, :]
    witness = solver.t - separations
    radial = np.array(
        [
            np.interp(witness, grid.z_node, e)
            - np.interp(witness, grid.z_node, e0)
            + np.interp(witness, grid.z_half, b)
            - np.interp(witness, grid.z_half, b0)
            for e, e0, b, b0 in zip(
                solver.f.er[keep], base_er[keep], solver.f.bt[keep], base_bt[keep]
            )
        ]
    )
    tail = -h * np.sum(radial, axis=0)
    complete = wake + tail
    result = dict(
        case=case,
        h=h,
        sigma=sigma,
        core=core,
        end=solver.t,
        direct_loss=loss,
        complete_loss=float(np.trapezoid(complete * lam, separations)),
        tail_loss=float(np.trapezoid(tail * lam, separations)),
        source_work=work,
        field_energy_change=history[-1][2],
        energy_residual=history[-1][3],
        elapsed_s=time.perf_counter() - begin,
        scope="restricted PEC prescribed-current wake reference; normalized c=1",
    )
    return result, dict(
        s=separations,
        direct_wake=wake,
        tail_wake=tail,
        complete_wake=complete,
        lambda_bunch=lam,
        history=np.array(history),
    )
