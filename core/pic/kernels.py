"""Serial float64 reference kernels; no atomics, parallelism, or fastmath.

Grid values are collocated at cell centres. Current components live on the
faces of their nodal control volumes. Kernel arguments are plain arrays so a
future backend can replace them without changing the physical model.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numba import njit

C = 299792458.0
EPSILON_0 = 8.8541878128e-12
COULOMB = 1.0 / (4.0 * np.pi * EPSILON_0)


@njit(cache=True)
def stencil(x: float, order: int) -> tuple[int, np.ndarray]:
    if order == 1:
        base = int(np.floor(x))
        f = x - base
        return base, np.array([1.0 - f, f])
    mid = int(np.floor(x + 0.5))
    d = x - mid
    return mid - 1, np.array([0.5 * (0.5 - d) ** 2, 0.75 - d * d, 0.5 * (0.5 + d) ** 2])


@njit(cache=True)
def deposit_kernel(
    coordinates: Any, charge: Any, shape: tuple[int, ...], order: int
) -> np.ndarray:
    rho = np.zeros(shape)
    for p in range(len(charge)):
        i, wx = stencil(coordinates[p, 0], order)
        j, wy = stencil(coordinates[p, 1], order)
        k, wz = stencil(coordinates[p, 2], order)
        for a in range(order + 1):
            for b in range(order + 1):
                for d in range(order + 1):
                    rho[i + a, j + b, k + d] += charge[p] * wx[a] * wy[b] * wz[d]
    return rho


@njit(cache=True)
def gather_kernel(coordinates: Any, field: Any, order: int) -> np.ndarray:
    result = np.zeros((len(coordinates), field.shape[3]))
    for p in range(len(coordinates)):
        i, wx = stencil(coordinates[p, 0], order)
        j, wy = stencil(coordinates[p, 1], order)
        k, wz = stencil(coordinates[p, 2], order)
        for a in range(order + 1):
            for b in range(order + 1):
                for d in range(order + 1):
                    for c in range(field.shape[3]):
                        result[p, c] += (
                            field[i + a, j + b, k + d, c] * wx[a] * wy[b] * wz[d]
                        )
    return result


@njit(cache=True)
def aligned_shapes(
    old: float, new: float, order: int
) -> tuple[int, np.ndarray, np.ndarray]:
    a, wa = stencil(old, order)
    b, wb = stencil(new, order)
    start = min(a, b)
    n = max(a, b) + order + 1 - start
    s = np.zeros(n)
    delta = np.zeros(n)
    s[a - start : a - start + order + 1] = wa
    delta[b - start : b - start + order + 1] = wb
    delta -= s
    return start, s, delta


@njit(cache=True)
def current_kernel(
    old: Any,
    new: Any,
    charge: Any,
    shape: tuple[int, ...],
    spacing: Any,
    dt: float,
    order: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Esirkepov tensor-product split, with long paths split into cell steps.

    For x, the transverse factor is Sy*Sz + dSy*Sz/2 + Sy*dSz/2
    + dSy*dSz/3. Summing its three cyclic contributions gives exactly
    Sx_new*Sy_new*Sz_new - Sx_old*Sy_old*Sz_old. Prefix sums along the
    current direction then satisfy the finite-volume continuity equation.
    """
    nx, ny, nz = shape
    currents = (
        np.zeros((nx + 1, ny, nz)),
        np.zeros((nx, ny + 1, nz)),
        np.zeros((nx, ny, nz + 1)),
    )
    volume = spacing[0] * spacing[1] * spacing[2]
    for p in range(len(charge)):
        displacement = new[p] - old[p]
        segments = max(1, int(np.ceil(np.max(np.abs(displacement)))))
        for segment in range(segments):
            lo = old[p] + displacement * (segment / segments)
            hi = old[p] + displacement * ((segment + 1) / segments)
            starts = np.zeros(3, dtype=np.int64)
            s0, x, dx = aligned_shapes(lo[0], hi[0], order)
            s1, y, dy = aligned_shapes(lo[1], hi[1], order)
            s2, z, dz = aligned_shapes(lo[2], hi[2], order)
            starts[:] = s0, s1, s2
            for axis in range(3):
                # Cycle axes to reuse exactly the same formula.
                if axis == 0:
                    sa, da, sb, db, sc, dc = x, dx, y, dy, z, dz
                elif axis == 1:
                    sa, da, sb, db, sc, dc = y, dy, z, dz, x, dx
                else:
                    sa, da, sb, db, sc, dc = z, dz, x, dx, y, dy
                cumulative = 0.0
                for a in range(len(sa)):
                    cumulative += da[a]
                    for b in range(len(sb)):
                        for c in range(len(sc)):
                            transverse = (
                                sb[b] * sc[c]
                                + 0.5 * db[b] * sc[c]
                                + 0.5 * sb[b] * dc[c]
                                + db[b] * dc[c] / 3.0
                            )
                            value = (
                                -charge[p]
                                * spacing[axis]
                                / (volume * dt)
                                * (cumulative * transverse)
                            )
                            if axis == 0:
                                currents[0][s0 + a + 1, s1 + b, s2 + c] += value
                            elif axis == 1:
                                currents[1][s0 + c, s1 + a + 1, s2 + b] += value
                            else:
                                currents[2][s0 + b, s1 + c, s2 + a + 1] += value
    return currents


@njit(cache=True)
def boris_kernel(
    u: Any, electric: Any, magnetic: Any, charge_mass: Any, dt: float
) -> Any:
    """Relativistic Boris rotation of u=p/(mc), using physical q/m."""
    result = np.empty_like(u)
    for p in range(len(u)):
        half_step = charge_mass[p] * dt / 2.0
        minus = u[p] + half_step * electric[p] / C
        gamma = np.sqrt(1.0 + np.dot(minus, minus))
        t = half_step * magnetic[p] / gamma
        s = 2.0 * t / (1.0 + np.dot(t, t))
        prime = minus + np.cross(minus, t)
        result[p] = minus + np.cross(prime, s) + half_step * electric[p] / C
    return result


def _surface(x: Any, y: Any, z: Any) -> Any:
    r = np.sqrt(x * x + y * y + z * z)
    return y * np.log(z + r) + z * np.log(y + r) - x * np.arctan(y * z / (x * r))


def _primitive(x: Any, y: Any, z: Any) -> Any:
    r = np.sqrt(x * x + y * y + z * z)
    return (
        x * y * np.log(z + r)
        + y * z * np.log(x + r)
        + z * x * np.log(y + r)
        - 0.5 * x * x * np.arctan(y * z / (x * r))
        - 0.5 * y * y * np.arctan(z * x / (y * r))
        - 0.5 * z * z * np.arctan(x * y / (z * r))
    )


def integrated_green(displacement: Any, spacing: Any, component: int | None) -> Any:
    """Exact rectangular-cell average of 1/r or r_component/r**3.

    Half-cell corner offsets avoid singular corners on the nodal convolution
    lattice. Arbitrary off-grid evaluation uses the same lattice at virtual
    gather nodes, so this routine never evaluates a source-cell face.
    """
    x, y, z = displacement
    hx, hy, hz = np.asarray(spacing) / 2.0
    total = np.zeros(np.broadcast_shapes(np.shape(x), np.shape(y), np.shape(z)))
    for a in (-1, 1):
        for b in (-1, 1):
            for c in (-1, 1):
                xx, yy, zz = x + a * hx, y + b * hy, z + c * hz
                if component is None:
                    value = _primitive(xx, yy, zz)
                elif component == 0:
                    value = -_surface(xx, yy, zz)
                elif component == 1:
                    value = -_surface(yy, zz, xx)
                else:
                    value = -_surface(zz, xx, yy)
                total += a * b * c * value
    return COULOMB * total / np.prod(spacing)


@njit(cache=True)
def node_field_kernel(
    nodes: Any, source_indices: Any, source_charge: Any, spacing: Any
) -> np.ndarray:
    """Integrated-cell fields at virtual integer nodes outside the solve box."""
    result = np.zeros((len(nodes), 3))
    for p in range(len(nodes)):
        for j in range(len(source_charge)):
            r = (nodes[p] - source_indices[j]) * spacing
            for axis in range(3):
                value = 0.0
                for a in (-1, 1):
                    for b in (-1, 1):
                        for d in (-1, 1):
                            corner = r + np.array([a, b, d]) * spacing / 2.0
                            x = corner[axis]
                            y = corner[(axis + 1) % 3]
                            z = corner[(axis + 2) % 3]
                            radius = np.sqrt(x * x + y * y + z * z)
                            surface = (
                                y * np.log(z + radius)
                                + z * np.log(y + radius)
                                - x * np.arctan(y * z / (x * radius))
                            )
                            value -= a * b * d * surface
                result[p, axis] += COULOMB * source_charge[j] * value / np.prod(spacing)
    return result
