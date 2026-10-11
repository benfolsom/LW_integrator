"""Grid-aligned walls and passive Drude response, including a smooth iris.

Finite-film frequency-domain response has no qualified time-domain adapter.
The Drude switching model and ramps come from material_step2j.py and
ramps_step2m.py. Wall geometry is explicit; arbitrary vacuum masks are rejected.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BPoly


@dataclass
class DrudeMedium:
    nu: float = 0.0
    absorb_r: object = 0.0
    absorb_z: object = 0.0
    absorb_b: object = 0.0

    def __post_init__(self):
        if not np.isfinite(self.nu) or self.nu < 0:
            raise ValueError("Drude collision rate must be finite and nonnegative")
        for rate in (self.absorb_r, self.absorb_z, self.absorb_b):
            if not np.all(np.isfinite(rate)) or np.any(np.asarray(rate) < 0):
                raise ValueError("absorber rates must be finite and nonnegative")

    @classmethod
    def with_sponge(cls, g, nu=2.0, width=2.0, strength=3.0):
        """Matched electric/magnetic sponge; its heat is a separate diagnostic."""
        if not (0 < width < min(g.nr * g.dr, g.nz * g.dz / 2)):
            raise ValueError("sponge width must fit the mesh")

        def profile(r, z):
            edge = np.maximum(
                np.maximum(r - (g.nr * g.dr - width), g.z0 + width - z),
                z - (g.z_node[-1] - width),
            )
            return strength * np.clip(edge / width, 0, 1) ** 3

        return cls(
            nu,
            profile(g.r_half[:, None], g.z_node[None, :]),
            profile(g.r_node[:, None], g.z_half[None, :]),
            profile(g.r_half[:, None], g.z_half[None, :]),
        )


@dataclass(frozen=True)
class AlignedWall:
    """Constant radial slabs with axial faces; circular pipes align in r–z.

    A planar sheet is a finite slab with inner_radius=0. Point crossing of
    local Drude material is undefined; provide an aperture or spatial cutoff.
    Tilted sheets, tapers, and profiles sampled from smooth curves are excluded.
    """

    inner_radius: float
    outer_radius: float
    z_min: float
    z_max: float

    def fractions(self, g):
        for v in (self.inner_radius, self.outer_radius):
            g.require_aligned(v, "r")
        for v in (self.z_min, self.z_max):
            g.require_aligned(v, "z")
        if not (
            0 <= self.inner_radius < self.outer_radius <= g.r_node[-1]
            and g.z0 <= self.z_min < self.z_max <= g.z_node[-1]
        ):
            raise ValueError("wall must have positive extent inside the mesh")
        sr = (
            (
                (g.r_half[:, None] >= self.inner_radius)
                & (g.r_half[:, None] < self.outer_radius)
            )
            * np.clip(0.5 + (g.z_node[None, :] - self.z_min) / g.dz, 0, 1)
            * np.clip(0.5 + (self.z_max - g.z_node[None, :]) / g.dz, 0, 1)
        )
        sz = (
            (g.r_node[:, None] >= self.inner_radius)
            & (g.r_node[:, None] < self.outer_radius)
            & (g.z_half[None, :] > self.z_min)
            & (g.z_half[None, :] < self.z_max)
        ).astype(float)
        return sr, sz


class DrudeWall:
    """Constant wall coupling, or the qualified prepared/opening iris model."""

    def __init__(
        self,
        grid,
        wall,
        plasma_frequency=10.0,
        *,
        aperture_radius=None,
        opening_time=0.2,
        ramp_duration=None,
    ):
        if not isinstance(wall, AlignedWall):
            raise ValueError(
                "only explicit grid-aligned walls are supported; "
                "sloped or curved geometry requires cut cells"
            )
        if not np.isfinite(plasma_frequency) or plasma_frequency <= 0:
            raise ValueError("plasma frequency must be finite and positive")
        self.base = tuple(plasma_frequency * np.sqrt(a) for a in wall.fractions(grid))
        self.grid = grid
        self.masks = tuple(a != 0 for a in self.base)
        self.aperture_radius, self.opening_time = aperture_radius, opening_time
        self.ramp_duration = ramp_duration
        self.inner = (
            np.zeros_like(self.base[0], dtype=bool),
            np.zeros_like(self.base[1], dtype=bool),
        )
        if aperture_radius is not None:
            grid.require_aligned(aperture_radius, "r")
            if not (
                wall.inner_radius <= aperture_radius < wall.outer_radius
                and np.isfinite(opening_time)
                and opening_time > 0
            ):
                raise ValueError("opening needs an interior aperture and positive time")
            self.inner = (
                grid.r_half[:, None] < aperture_radius,
                grid.r_node[:, None] < aperture_radius,
            )
        if ramp_duration is not None:
            if not (
                np.isfinite(ramp_duration)
                and ramp_duration > 0
                and (aperture_radius is None or ramp_duration < opening_time)
            ):
                raise ValueError(
                    "preparation duration must be positive and fit opening"
                )
            if aperture_radius is not None:
                start = opening_time - ramp_duration
                a = np.pi / opening_time
                jets = [
                    0.5 * (1 + np.cos(a * start)),
                    -0.5 * a * np.sin(a * start),
                    -0.5 * a * a * np.cos(a * start),
                    0.5 * a**3 * np.sin(a * start),
                ]
                self.bridge = BPoly.from_derivatives(
                    [start, opening_time], [jets, [0.0, 0.0, 0.0, 0.0]]
                )

    def factors(self, t):
        prep, pd = 1.0, 0.0
        if self.ramp_duration is not None:
            prep, pd = preparation(t, self.ramp_duration)
        op, od = 1.0, 0.0
        if self.aperture_radius is not None:
            if t >= self.opening_time:
                op = 0.0
            elif self.ramp_duration is not None and t >= (
                self.opening_time - self.ramp_duration
            ):
                op, od = float(self.bridge(t)), float(self.bridge(t, nu=1))
            else:
                phase = np.pi * np.clip(t / self.opening_time, 0, 1)
                op = 0.5 * (1 + np.cos(phase))
                od = -np.pi / (2 * self.opening_time) * np.sin(phase) if t > 0 else 0.0
        return prep, pd, op, od

    def __call__(self, t):
        prep, _, op, _ = self.factors(t)
        return tuple(
            a * prep * np.where(i, op, 1.0) for a, i in zip(self.base, self.inner)
        )

    def derivative(self, t):
        prep, pd, op, od = self.factors(t)
        return tuple(
            a * (pd * np.where(i, op, 1.0) + prep * np.where(i, od, 0.0))
            for a, i in zip(self.base, self.inner)
        )


def preparation(t, duration):
    """Seventh-degree smoothstep with three zero endpoint derivatives."""
    if t <= 0:
        return 0.0, 0.0
    if t >= duration:
        return 1.0, 0.0
    x = t / duration
    return (
        x**4 * (35 - 84 * x + 70 * x * x - 20 * x**3),
        140 * x**3 * (1 - x) ** 3 / duration,
    )
