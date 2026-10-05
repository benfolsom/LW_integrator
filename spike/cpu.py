"""Authoritative float64 NumPy/Numba kernels (strict math, one CPU thread)."""

import numpy as np
from numba import njit

from .common import array_poisson, inverse_laplacian


@njit
def stencil(x, n, dx, order):
    indices = np.empty((3, 3), np.int64)
    weights = np.zeros((3, 3), np.float64)
    for axis in range(3):
        s = (x[axis] / dx) % n
        if order == 1:
            base = int(np.floor(s))
            f = s - base
            weights[axis, 0] = 1 - f
            weights[axis, 1] = f
        else:
            center = int(np.floor(s + 0.5))
            d = s - center
            base = center - 1
            weights[axis, 0] = 0.5 * (0.5 - d) ** 2
            weights[axis, 1] = 0.75 - d * d
            weights[axis, 2] = 0.5 * (0.5 + d) ** 2
        for j in range(order + 1):
            indices[axis, j] = (base + j) % n
    return indices, weights


@njit
def deposit(x, q, n, dx, order):
    rho = np.zeros((n, n, n), np.float64)
    correction = np.zeros_like(rho)
    for p in range(len(x)):
        ids, w = stencil(x[p], n, dx, order)
        for i in range(order + 1):
            for j in range(order + 1):
                for k in range(order + 1):
                    a, b, c = ids[0, i], ids[1, j], ids[2, k]
                    value = q[p] * w[0, i] * w[1, j] * w[2, k] / dx**3
                    # Neumaier summation handles mixed signs and magnitudes.
                    old = rho[a, b, c]
                    new = old + value
                    if abs(old) >= abs(value):
                        correction[a, b, c] += (old - new) + value
                    else:
                        correction[a, b, c] += (value - new) + old
                    rho[a, b, c] = new
    return rho + correction


@njit
def gather(x, field, n, dx, order):
    result = np.zeros((len(x), 3), np.float64)
    for p in range(len(x)):
        ids, w = stencil(x[p], n, dx, order)
        correction = np.zeros(3)
        for i in range(order + 1):
            for j in range(order + 1):
                for k in range(order + 1):
                    weight = w[0, i] * w[1, j] * w[2, k]
                    for axis in range(3):
                        value = weight * field[ids[0, i], ids[1, j], ids[2, k], axis]
                        y = value - correction[axis]
                        new = result[p, axis] + y
                        correction[axis] = (new - result[p, axis]) - y
                        result[p, axis] = new
    return result


@njit
def push(x, u, electric, magnetic, qm, dt, length):
    """Relativistic Boris rotation of u=p/(mc), then periodic position drift."""
    xn, un = np.empty_like(x), np.empty_like(u)
    for p in range(len(x)):
        kick = qm[p] * dt / 2
        minus = u[p] + kick * electric[p]
        t = kick * magnetic[p] / np.sqrt(1 + np.sum(minus * minus))
        s = 2 * t / (1 + np.sum(t * t))
        prime = minus + np.cross(minus, t)
        plus = minus + np.cross(prime, s)
        un[p] = plus + kick * electric[p]
        xn[p] = (x[p] + dt * un[p] / np.sqrt(1 + np.sum(un[p] * un[p]))) % length
    return xn, un


class CPU:
    dtype = "float64"
    name = "cpu"

    def __init__(self, grid):
        self.grid = grid
        self.inverse = inverse_laplacian(grid)

    def array(self, value):
        return np.ascontiguousarray(value, dtype=np.float64)

    def host(self, value):
        return np.asarray(value)

    def sync(self, *values):
        pass

    def deposit(self, x, q, order):
        return deposit(x, q, self.grid.n, self.grid.dx, order)

    def gather(self, x, field, order):
        return gather(x, field, self.grid.n, self.grid.dx, order)

    def push(self, x, u, electric, magnetic, qm, dt):
        return push(x, u, electric, magnetic, qm, dt, self.grid.length)

    def poisson(self, rho):
        return array_poisson(np, rho, self.inverse, self.grid.dx)
