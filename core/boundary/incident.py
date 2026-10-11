"""Analytic, deposition-free uniform-motion LW drive. Normalized c=1 units.

Ported from material_step2j.py and hybrid.py; no study runtime imports.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicHermiteSpline


def uniform_axial_average(rad, lo, hi, position, q, beta, core, component):
    gamma = 1 / np.sqrt(1 - beta**2)

    def primitive(x):
        s = rad * rad + x * x
        if component == 1:
            v = -1 / np.sqrt(s)
            if core > 0:
                v = np.where(
                    s < core**2,
                    -(15 / 8 - 5 / 4 * s / core**2 + 3 / 8 * s * s / core**4) / core,
                    v,
                )
            return v
        v = x / (rad * rad * np.sqrt(s))
        if core > 0:
            cut = np.sqrt(np.maximum(core**2 - rad * rad, 0))
            xc = np.clip(x, -cut, cut)
            interior = (
                (2.5 - 1.5 * rad * rad / core**2) * xc - 0.5 * xc**3 / core**2
            ) / core**3
            v += np.where(
                rad < core,
                interior - xc / (rad * rad * np.sqrt(rad * rad + xc * xc)),
                0,
            )
        return v

    value = (
        primitive(gamma * (hi - position)) - primitive(gamma * (lo - position))
    ) / (hi - lo)
    if component == 1:
        return q / (4 * np.pi * gamma) * value
    return q * rad / (4 * np.pi) * value * (beta if component == 2 else 1.0)


class BallisticDrive:
    """One prescribed point charge moving along +z, including inertial prehistory.

    q is the source charge: include macro population here when applicable.
    Observer charge, observer mass, and diagnostic population remain separate.
    Axial integration is analytic; radial integration uses fixed Gauss weights.
    """

    def __init__(self, g, gamma, q, order=8, z_at_zero=-1.0):
        if not (
            np.isfinite(gamma)
            and gamma >= 1
            and np.isfinite(q)
            and np.isfinite(z_at_zero)
            and int(order) == order
            and order > 0
        ):
            raise ValueError(
                "incident drive needs gamma>=1, finite charge/origin, and positive order"
            )
        self.g, self.gamma, self.q = g, gamma, q
        self.beta = np.sqrt(1 - gamma**-2)
        if self.beta >= 1:
            raise ValueError(
                "source gamma is too large for a subluminal float64 velocity"
            )
        self.z0 = z_at_zero
        self.nodes, self.weights = np.polynomial.legendre.leggauss(order)
        self.cache_time = None
        self.cache = None
        self.geometry = []
        for k in range(3):
            r = g.r_node if k == 1 else g.r_half
            z = g.z_node if k == 0 else g.z_half
            rl = np.maximum(r - g.dr / 2, 0)
            rh = np.minimum(r + g.dr / 2, g.nr * g.dr)
            zl = np.maximum(z - g.dz / 2, g.z0)
            zh = np.minimum(z + g.dz / 2, g.z_node[-1])
            self.geometry.append((rl, rh, zl, zh))

    def position(self, t):
        return self.z0 + self.beta * t

    def component(self, t, k, t1=None):
        rl, rh, zl, zh = self.geometry[k]
        out = np.zeros((len(rl), len(zl)))
        for n, w in zip(self.nodes, self.weights):
            r = ((rl + rh) / 2 + (rh - rl) * n / 2)[:, None]
            rw = (w * r[:, 0] / (rl + rh))[:, None]
            if t1 is None:
                val = uniform_axial_average(
                    r,
                    zl[None, :],
                    zh[None, :],
                    self.z0 + self.beta * t,
                    self.q,
                    self.beta,
                    0.0,
                    k,
                )
            elif self.beta == 0:
                val = uniform_axial_average(
                    r, zl[None, :], zh[None, :], self.z0, self.q, self.beta, 0.0, k
                )
            else:
                xlo0 = self.gamma * (zl[None, :] - self.z0 - self.beta * t)
                xhi0 = self.gamma * (zh[None, :] - self.z0 - self.beta * t)
                xlo1 = self.gamma * (zl[None, :] - self.z0 - self.beta * t1)
                xhi1 = self.gamma * (zh[None, :] - self.z0 - self.beta * t1)
                if k == 1:
                    v = (
                        np.arcsinh(xhi1 / r)
                        - np.arcsinh(xhi0 / r)
                        - np.arcsinh(xlo1 / r)
                        + np.arcsinh(xlo0 / r)
                    )
                    val = (
                        self.q
                        * v
                        / (
                            4
                            * np.pi
                            * self.gamma**2
                            * self.beta
                            * (zh - zl)[None, :]
                            * (t1 - t)
                        )
                    )
                else:
                    v = (
                        np.hypot(r, xhi0)
                        - np.hypot(r, xlo0)
                        - np.hypot(r, xhi1)
                        + np.hypot(r, xlo1)
                    )
                    val = (
                        self.q
                        * v
                        / (
                            4
                            * np.pi
                            * self.gamma
                            * self.beta
                            * r
                            * (zh - zl)[None, :]
                            * (t1 - t)
                        )
                    )
                    if k == 2:
                        val *= self.beta
            out += rw * val
        return out

    def volume(self, g, t):
        if self.cache_time != t:
            self.cache = tuple(self.component(t, k) for k in range(3))
            self.cache_time = t
        return self.cache

    def material_volume(self, g, t, masks):
        return tuple(
            np.where(mask, x, 0.0) for mask, x in zip(masks, self.volume(g, t)[:2])
        )

    def edges_mid(self, g, t0, t1, masks):
        return tuple(
            np.where(mask, self.component(t0, k, t1), 0.0)
            for k, mask in enumerate(masks)
        )


def point_fields(
    path, q: float, r: np.ndarray, z: np.ndarray, t: np.ndarray, mirror: bool = False
) -> tuple:
    r, z, t = np.broadcast_arrays(r, z, t)
    sign = -1.0 if mirror else 1.0

    def evaluate(u, order=0):
        # The saved prehistory is ballistic. Evaluate it directly, avoiding
        # subtraction of the two endpoints of a 20,000-unit spline interval.
        value = path(u, order)
        ballistic = (
            path(0.0) + u * path(0.0, 1)
            if order == 0
            else (path(0.0, 1) if order == 1 else 0.0)
        )
        return np.where(u < 0, ballistic, value)

    instant = np.minimum(t, path.x[-1])
    gap = z - sign * evaluate(instant)
    beta = sign * evaluate(instant, 1)
    travel = (r * r + gap * gap) / (
        np.sqrt(gap * gap + (1 - beta * beta) * r * r) - beta * gap
    )
    u = t - travel
    for _ in range(16):
        zs = sign * evaluate(u)
        v = sign * evaluate(u, 1)
        dz = z - zs
        distance = np.sqrt(r * r + dz * dz)
        change = (t - u - distance) / (1 - dz / distance * v)
        u += change
        if (
            np.max(
                abs(t - u - np.sqrt(r * r + (z - sign * evaluate(u)) ** 2))
                / (1 + distance)
            )
            < 3e-14
            or np.max(abs(change) / (1 + abs(u))) < 2e-12
        ):
            break
    else:
        worst = np.argmax(abs(change) / (1 + abs(u)))
        raise RuntimeError(
            f"Point LW root did not converge: relative_change={np.max(abs(change)/(1+abs(u))):.3g}, u={u.flat[worst]:.9g}, t={t.flat[worst]:.9g}, beta={v.flat[worst]:.9g}"
        )
    if np.max(u) > path.x[-1] + 1e-8:
        raise RuntimeError("Observation requires unsaved future source history")
    dz = z - sign * evaluate(u)
    v = sign * evaluate(u, 1)
    a = sign * evaluate(u, 2)
    distance = np.sqrt(r * r + dz * dz)
    nr, nz = r / distance, dz / distance
    kappa = 1 - nz * v
    prefactor = sign * q / (4 * np.pi)
    er_v = prefactor * (1 - v * v) * nr / (kappa**3 * distance**2)
    ez_v = prefactor * (1 - v * v) * (nz - v) / (kappa**3 * distance**2)
    er_a = prefactor * a * nr * nz / (kappa**3 * distance)
    ez_a = -prefactor * a * nr * nr / (kappa**3 * distance)
    return ((er_v, ez_v, nz * er_v - nr * ez_v), (er_a, ez_a, nz * er_a - nr * ez_a))


class ResidualPath:
    """Interpolate small departures from a ballistic path without cancellation."""

    def __init__(self, ts, delta_z, delta_beta, z0, beta):
        self.residual = CubicHermiteSpline(ts, delta_z, delta_beta)
        self.x = self.residual.x
        self.z0, self.beta = z0, beta

    def __call__(self, t, order=0):
        value = np.where(np.asarray(t) < 0, 0.0, self.residual(t, order))
        if order == 0:
            value = value + self.z0 + self.beta * np.asarray(t)
        elif order == 1:
            value = value + self.beta
        return value


class PrescribedDrive(BallisticDrive):
    """Read-only prescribed axial history, with ballistic prehistory at t<0.

    Retarded velocity and acceleration fields follow direct_step2i.py and
    radiation_step2g.py. Store small position/velocity departures from a
    ballistic reference; no material or particle feedback edits this source.
    Converge source sampling and spatial/time quadrature for accelerated paths.
    """

    def __init__(
        self,
        g,
        gamma,
        q,
        times,
        delta_z,
        delta_beta,
        *,
        z_at_zero=-1.0,
        order=8,
        correction_order=3,
    ):
        super().__init__(g, gamma, q, order, z_at_zero=z_at_zero)
        ts, dz, dv = (
            np.array(a, dtype=float, copy=True) for a in (times, delta_z, delta_beta)
        )
        if (
            ts.ndim != 1
            or len(ts) < 2
            or dz.shape != ts.shape
            or dv.shape != ts.shape
            or not np.all(np.isfinite([ts, dz, dv]))
            or ts[0] != 0
            or np.any(np.diff(ts) <= 0)
            or dz[0] != 0
            or dv[0] != 0
        ):
            raise ValueError(
                "prescribed history needs increasing times from zero and zero initial departures"
            )
        self.path = ResidualPath(ts, dz, dv, self.z0, self.beta)
        critical = self.path.residual.derivative(2).roots(extrapolate=False)
        critical = critical[np.isfinite(critical)]
        if np.any(abs(self.path(np.r_[ts, critical], 1)) >= 1):
            raise ValueError("prescribed source history must remain subluminal")
        self.correction_nodes, self.correction_weights = (
            np.polynomial.legendre.leggauss(correction_order)
        )

    def position(self, t):
        if t > self.path.x[-1]:
            raise ValueError("prescribed position requires unsaved future history")
        return self.path(t)

    def correction(self, r, z, t, k):
        r, z = np.broadcast_arrays(r, z)
        out = np.zeros_like(r)
        active = t > np.hypot(r, z - self.z0)
        if np.any(active):
            vel, acc = point_fields(
                self.path, self.q, r[active], z[active], np.full(np.sum(active), t)
            )
            gap = z[active] - self.z0 - self.beta * t
            pref = (
                self.q
                * self.gamma
                / (4 * np.pi * (r[active] ** 2 + (self.gamma * gap) ** 2) ** 1.5)
            )
            baseline = (
                pref * (gap if k == 1 else r[active]) * (self.beta if k == 2 else 1.0)
            )
            out[active] = vel[k] + acc[k] - baseline
        return out

    def component(self, t, k, t1=None):
        out = super().component(t, k, t1)
        rl, rh, zl, zh = self.geometry[k]
        temporal = (
            [(t, 1.0)]
            if t1 is None
            else [
                ((t + t1) / 2 + (t1 - t) * x / 2, w / 2)
                for x, w in zip(self.correction_nodes, self.correction_weights)
            ]
        )
        for tm, tw in temporal:
            if tm <= 0:
                continue
            for x, wx in zip(self.correction_nodes, self.correction_weights):
                rr = (rl + rh) / 2 + (rh - rl) * x / 2
                rw = wx * rr / (rl + rh)
                for y, wy in zip(self.correction_nodes, self.correction_weights):
                    zz = (zl + zh) / 2 + (zh - zl) * y / 2
                    out += (
                        tw
                        * rw[:, None]
                        * wy
                        / 2
                        * self.correction(rr[:, None], zz[None, :], tm, k)
                    )
        return out
