"""Shared normalized periodic problem and backend selection."""

import importlib
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Grid:
    n: int
    dx: float = 1.0

    def __post_init__(self):
        if (
            self.n < 4
            or self.n & (self.n - 1)
            or not np.isfinite(self.dx)
            or self.dx <= 0
        ):
            raise ValueError("Use a power-of-two cubic grid >= 4 and positive dx")

    @property
    def length(self):
        return self.n * self.dx


def inverse_laplacian(grid):
    """Periodic discrete Laplacian; zero mode fixes mean(phi) = 0."""
    k = 2 * np.pi * np.fft.fftfreq(grid.n)
    a = 4 * np.sin(k / 2) ** 2 / grid.dx**2
    denominator = a[:, None, None] + a[None, :, None] + a[None, None, :]
    denominator[0, 0, 0] = 1
    inverse = 1 / denominator
    inverse[0, 0, 0] = 0
    return inverse


def array_poisson(xp, rho, inverse, dx):
    """Solve -laplacian(phi)=rho with a uniform neutralizing background.

    c = epsilon_0 = 1. E is the centered, skew-symmetric gradient, so using
    matching deposit/gather shapes gives zero net periodic electrostatic force.
    """
    phi = xp.fft.ifftn(xp.fft.fftn(rho) * inverse).real
    field = xp.stack(
        [(xp.roll(phi, 1, a) - xp.roll(phi, -1, a)) / (2 * dx) for a in range(3)],
        axis=-1,
    )
    return phi, field


def make_backend(name, grid, dtype=None, compensated=True):
    if name == "cpu":
        from .cpu import CPU

        return CPU(grid)
    if name in ("mlx", "mlx-cuda"):
        if dtype not in (None, "float32"):
            raise ValueError("The MLX spike adapter supports float32 only")
        from .mlx_backend import MLX

        return MLX(grid, cuda=name == "mlx-cuda", compensated=compensated)
    if name == "cupy":
        from .cupy_backend import CuPy

        return CuPy(grid, dtype=dtype or "float64")
    if name in ("taichi-metal", "taichi-cuda", "taichi-cpu"):
        module = importlib.import_module("spike.taichi_backend")
        return module.Taichi(
            grid, name.split("-")[1], dtype=dtype, compensated=compensated
        )
    raise ValueError(f"Unknown backend: {name}")


def synthetic(count, grid, seed=20261005):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, grid.length, (count, 3))
    u = rng.normal(0, 0.7, (count, 3))
    population = rng.uniform(0.5, 1.5, count)
    species_charge = np.where(np.arange(count) % 2, -1.0, 1.0)
    return x, u, population * species_charge, species_charge, population
