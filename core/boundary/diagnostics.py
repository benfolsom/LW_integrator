"""Discrete grid identity and independent physical flux/storage diagnostics.

The physical Surface and continuous_storage use the same linear scattered
reconstruction and exact incident primitives (grid_step2l.py, cut_step2l.py).
GridLedger preserves the discrete identity from direct_step2h.py.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from typing import Any

from .mesh import Fields, volumes


def interpolate(g, arrays, r, z):
    result = []
    for k, a in enumerate(arrays):
        rn = g.r_node if k == 1 else g.r_half
        zn = g.z_node if k == 0 else g.z_half
        if k != 1:
            rn = np.r_[0.0, rn]
            a = np.vstack((np.zeros((1, a.shape[1])), a))
        f = RegularGridInterpolator((rn, zn), a, bounds_error=False, fill_value=None)
        result.append(f(np.column_stack((r, z))))
    return result


def strips(low, high, spacing, order):
    # Composite Gauss in r; exact axial incident averages use the strip bounds.
    bounds = np.linspace(low, high, round((high - low) / spacing) * order + 1)
    return bounds[:-1], bounds[1:]


class Surface:
    def __init__(self, g, control, order):
        volumes(g, control)  # validate the diagnostic cylinder
        self.control, self.order = control, order
        radius, zlo, zhi = control
        nodes, w = np.polynomial.legendre.leggauss(order)
        bounds = np.linspace(0, radius, round(radius / g.dr) + 1)
        lo, hi = bounds[:-1, None], bounds[1:, None]
        self.r = ((lo + hi) / 2 + (hi - lo) * nodes / 2).ravel()
        self.rw = (
            2 * np.pi * ((lo + hi) / 2 + (hi - lo) * nodes / 2) * (hi - lo) * w / 2
        ).ravel()
        self.zl, self.zh = strips(zlo, zhi, g.dz, order)
        self.z = (self.zl + self.zh) / 2
        self.zw = 2 * np.pi * radius * (self.zh - self.zl)
        self.cumulative = np.zeros(4)

    def increment(self, g, arrays, t0, t1, gamma, q, z_at_zero=-1.0):
        radius, zlo, zhi = self.control
        beta = np.sqrt(1 - gamma**-2)
        t = (t0 + t1) / 2
        dt = t1 - t0
        # Side: integrate the contracted incident pulse analytically in z.
        r = np.full_like(self.z, radius)
        er, ez, bt = interpolate(g, arrays, r, self.z)
        xl = gamma * (self.zl - z_at_zero - beta * t)
        xh = gamma * (self.zh - z_at_zero - beta * t)
        ir = (
            q
            / (4 * np.pi * radius * (self.zh - self.zl))
            * (xh / np.hypot(radius, xh) - xl / np.hypot(radius, xl))
        )
        iz = (
            q
            / (4 * np.pi * gamma * (self.zh - self.zl))
            * (1 / np.hypot(radius, xl) - 1 / np.hypot(radius, xh))
        )
        ib = beta * ir
        values = np.array(
            [
                np.dot(-ez * bt, self.zw),
                np.dot(-iz * bt - ez * ib, self.zw),
                np.dot(-ez * er, self.zw),
                np.dot(-iz * er - ez * ir, self.zw),
            ]
        )
        # Caps: the exact temporal primitive resolves sub-timestep crossings.
        for z, sign in ((zlo, -1), (zhi, 1)):
            rr = self.r
            er, ez, bt = interpolate(g, arrays, rr, np.full_like(rr, z))
            x0 = gamma * (z - z_at_zero - beta * t0)
            x1 = gamma * (z - z_at_zero - beta * t1)
            ir = (
                q
                / (4 * np.pi * beta * rr * dt)
                * (x0 / np.hypot(rr, x0) - x1 / np.hypot(rr, x1))
            )
            iz = (
                q
                / (4 * np.pi * gamma * beta * dt)
                * (1 / np.hypot(rr, x1) - 1 / np.hypot(rr, x0))
            )
            ib = beta * ir
            values += sign * np.array(
                [
                    np.dot(er * bt, self.rw),
                    np.dot(ir * bt + er * ib, self.rw),
                    np.dot((er * er + bt * bt - ez * ez) / 2, self.rw),
                    np.dot(ir * er + ib * bt - iz * ez, self.rw),
                ]
            )
        self.cumulative += dt * values
        return values


def continuous_storage(
    g: Any,
    drive: Any,
    t: float,
    arrays: tuple[np.ndarray, ...],
    control: tuple[float, float, float],
    radial_order: int = 4,
) -> tuple[float, float]:
    """Products of bilinear scattered fields and analytic incident fields.

    The z integral is analytic, including its first moment for linear scattered
    interpolation. Pure scattered storage is integrated exactly on each linear
    segment. This matches the reconstruction used by the physical surfaces.
    """
    radius, low, high = control
    nodes, weights = np.polynomial.legendre.leggauss(radial_order)
    source = drive.z0 + drive.beta * t
    scattered = cross = 0.0
    for k, array in enumerate(arrays):
        rn = g.r_node if k == 1 else g.r_half
        zn = g.z_node if k == 0 else g.z_half
        if k != 1:
            rn = np.r_[0.0, rn]
            array = np.vstack((np.zeros((1, array.shape[1])), array))
        radial = np.unique(np.r_[0.0, rn[(rn > 0) & (rn < radius)], radius])
        axial = np.unique(np.r_[low, zn[(zn > low) & (zn < high)], high])
        zl, zh = axial[:-1], axial[1:]
        length = zh - zl
        for left, right in zip(radial[:-1], radial[1:]):
            for x, w in zip(nodes, weights):
                r = (left + right) / 2 + (right - left) * x / 2
                ir = min(np.searchsorted(rn, r) - 1, len(rn) - 2)
                fraction = (r - rn[ir]) / (rn[ir + 1] - rn[ir])
                profile = (1 - fraction) * array[ir] + fraction * array[ir + 1]
                values = np.interp(axial, zn, profile)
                a, b = values[:-1], values[1:]
                measure = 2 * np.pi * r * (right - left) * w / 2
                scattered += measure * np.sum(length * (a * a + a * b + b * b) / 6)
                xl = drive.gamma * (zl - source)
                xh = drive.gamma * (zh - source)
                dl = np.hypot(r, xl)
                dh = np.hypot(r, xh)
                if k == 1:
                    i0 = drive.q / (4 * np.pi * drive.gamma) * (1 / dl - 1 / dh)
                    i1 = (source - zl) * i0 + drive.q / (
                        4 * np.pi * drive.gamma**2
                    ) * (np.arcsinh(xh / r) - xh / dh - np.arcsinh(xl / r) + xl / dl)
                else:
                    i0 = drive.q / (4 * np.pi * r) * (xh / dh - xl / dl)
                    i1 = (source - zl) * i0 - drive.q * r / (
                        4 * np.pi * drive.gamma
                    ) * (1 / dh - 1 / dl)
                    if k == 2:
                        i0 *= drive.beta
                        i1 *= drive.beta
                cross += measure * np.sum(a * i0 + (b - a) / length * i1)
    return float(scattered), float(cross)


class PhysicalLedger:
    """Consistent physical surface/storage balance for a ballistic incident field.

    Call advance for every material step, then measure at selected times. The
    independently measured particle work enters with its own sign; a ledger
    residual is never assigned to work, heat, radiation, or momentum. Exterior
    radiation and whole-system particle feedback remain separate qualifications.
    """

    def __init__(self, solver, drive, control, order=4):
        from .incident import BallisticDrive, PrescribedDrive

        if not isinstance(drive, BallisticDrive) or isinstance(drive, PrescribedDrive):
            raise ValueError(
                "physical surface primitives currently require ballistic sources"
            )
        if drive.gamma <= 1:
            raise ValueError("physical cap time primitives require a moving source")
        self.solver, self.drive, self.control = solver, drive, control
        self.order = order
        self.grid = GridLedger(solver, control, drive)
        self.surface = Surface(solver.g, control, order)
        self.initial = continuous_storage(
            solver.g,
            drive,
            solver.t,
            (solver.f.er, solver.f.ez, solver.f.bt),
            control,
            order,
        )
        self.last = None

    def advance(self, step, particle_work=0.0):
        self.last = self.grid.measure_direct(step, particle_work)
        self.surface.increment(
            self.solver.g,
            (*step["es"], step["bs"]),
            step["t0"],
            step["t1"],
            self.drive.gamma,
            self.drive.q,
            self.drive.z0,
        )

    def measure(self):
        if self.last is None:
            raise ValueError("advance the physical ledger before measuring")
        s = self.solver
        us, ux = continuous_storage(
            s.g, self.drive, s.t, (s.f.er, s.f.ez, s.f.bt), self.control, self.order
        )
        us, ux = us - self.initial[0], ux - self.initial[1]
        self_flux, cross_flux, self_momentum, cross_momentum = self.surface.cumulative
        d = self.last
        residual = (
            us
            + ux
            + d["material_storage"]
            + d["heat"]
            + self_flux
            + cross_flux
            + d["absorber_difference_work"]
            - d["actuator_work"]
            + d["wall_work"]
        )
        return dict(
            time=s.t,
            scattered_energy=us,
            cross_energy=ux,
            material_storage=d["material_storage"],
            heat=d["heat"],
            actuator_work=d["actuator_work"],
            particle_work=d["wall_work"],
            absorber_difference_work=d["absorber_difference_work"],
            scattered_flux=float(self_flux),
            cross_flux=float(cross_flux),
            scattered_momentum_flux=float(self_momentum),
            cross_momentum_flux=float(cross_momentum),
            residual=float(residual),
            discrete_grid_residual=d["grid_residual"],
        )


class GridLedger:
    """Separate material-driven grid identity and uncorrected physical ledger."""

    def __init__(self, solver, control, history):
        self.solver, self.control = solver, control
        self.volumes = volumes(solver.g, control)
        self.work = self.heat = self.actuator = self.defect = 0.0
        self.history_provider = history
        self.previous = history.volume(solver.g, solver.t)
        self.initial_scattered = 0.5 * (
            self.inner(solver.f.er, solver.f.ez, solver.f.er, solver.f.ez)
            + float(np.sum(solver.f.bt**2 * self.volumes[2]))
        )
        self.initial_cross = self.inner(
            solver.f.er, solver.f.ez, *self.previous[:2]
        ) + float(np.sum(solver.f.bt * self.previous[2] * self.volumes[2]))
        self.initial_material = 0.5 * sum(
            float(np.sum(a * a * v)) for a, v in zip(solver.v, self.volumes[:2])
        )
        self.drive_work = self.self_flux = self.cross_flux = self.mismatch = 0.0
        self.self_absorb = self.cross_absorb = 0.0

    def measure_direct(self, data, particle_work: float):
        s, dt = self.solver, self.solver.dt
        diag = self.history_provider.volume(s.g, s.t)
        es, bs = data["es"], data["bs"]
        ei = tuple((a + b) / 2 for a, b in zip(self.previous[:2], diag[:2]))
        bi = (self.previous[2] + diag[2]) / 2
        f = Fields(s.g)
        f.bt = bs
        cs = s.curl_b(f)
        f.bt = bi
        ci = s.curl_b(f)
        ces = s.ce(s.g, *es)
        cei = s.ce(s.g, *ei)
        self.self_flux -= dt * (
            self.inner(*es, *cs) + float(np.sum(bs * ces * self.volumes[2]))
        )
        self.cross_flux -= dt * (
            self.inner(*es, *ci)
            + self.inner(*ei, *cs)
            + float(np.sum((bs * cei + bi * ces) * self.volumes[2]))
        )
        self.self_absorb += dt * (
            self.inner(*es, s.medium.absorb_r * es[0], s.medium.absorb_z * es[1])
            + float(np.sum(s.medium.absorb_b * bs * bs * self.volumes[2]))
        )
        self.cross_absorb += dt * (
            self.inner(*ei, s.medium.absorb_r * es[0], s.medium.absorb_z * es[1])
            + float(np.sum(s.medium.absorb_b * bi * bs * self.volumes[2]))
        )
        self.drive_work += dt * self.inner(*data["incident"], *data["current"])
        self.mismatch += dt * self.inner(
            ei[0] - data["incident"][0], ei[1] - data["incident"][1], *data["current"]
        )
        ji = tuple(c - (b - a) / dt for c, a, b in zip(ci, self.previous[:2], diag[:2]))
        ki = cei - (diag[2] - self.previous[2]) / dt
        self.defect += (
            dt * (self.inner(*es, *ji) + float(np.sum(bs * ki * self.volumes[2])))
            - particle_work
        )
        self.work += particle_work
        for k, d in enumerate(data["material"]):
            self.heat += dt * float(
                np.sum(s.medium.nu * d["v_mid"] ** 2 * self.volumes[k])
            )
            self.actuator -= dt * float(
                np.sum(d["gdot"] * d["x_mid"] * d["e_total"] * self.volumes[k])
            )
        stored = 0.5 * sum(
            float(np.sum(a * a * v)) for a, v in zip(s.v, self.volumes[:2])
        )
        scattered = 0.5 * (
            self.inner(*data["e1"], *data["e1"])
            + float(np.sum(data["b1"] ** 2 * self.volumes[2]))
        )
        cross = self.inner(*data["e1"], *diag[:2]) + float(
            np.sum(data["b1"] * diag[2] * self.volumes[2])
        )
        scattered -= self.initial_scattered
        cross -= self.initial_cross
        stored -= self.initial_material
        grid_residual = (
            scattered
            + stored
            + self.heat
            + self.self_flux
            + self.self_absorb
            - self.actuator
            - self.drive_work
        )
        residual = (
            scattered
            + cross
            + stored
            + self.heat
            + self.self_flux
            + self.cross_flux
            + self.self_absorb
            + self.cross_absorb
            - self.actuator
            + self.work
        )
        self.previous = diag
        return dict(
            scattered_energy=scattered,
            cross_energy=cross,
            material_storage=stored,
            heat=self.heat,
            actuator_work=self.actuator,
            drive_work=self.drive_work,
            wall_work=self.work,
            grid_flux=self.self_flux + self.cross_flux,
            window_scattered_flux=self.self_flux,
            window_cross_flux=self.cross_flux,
            absorber_difference_work=self.self_absorb + self.cross_absorb,
            grid_residual=grid_residual,
            boundary_residual=residual,
            incident_defect=self.defect,
            material_projection_work=self.mismatch,
            identity_residual=residual + self.defect + self.mismatch,
        )

    def inner(self, er, ez, br, bz):
        return sum(
            float(np.sum(a * b * v))
            for a, b, v in zip((er, ez), (br, bz), self.volumes[:2])
        )
