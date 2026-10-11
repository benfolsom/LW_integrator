"""Axisymmetric staggered mesh in Heaviside–Lorentz units with c=1.

The mesh and volume metric come from fdtd_rz.py and material_step2j.py.
This is an r–z TM solver, not a general Cartesian Maxwell solver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from core.pic.backend import NumpyBackend, PICBackend


def cpu_backend(backend: PICBackend | None = None) -> PICBackend:
    backend = backend or NumpyBackend()
    if backend.xp is not np or backend.dtype != "float64":
        raise ValueError("boundary fields require the CPU NumPy float64 backend")
    return backend


@dataclass(frozen=True)
class AxisymmetricGrid:
    dr: float
    dz: float
    nr: int
    nz: int
    z0: float

    # Installed dynamically in __post_init__; keep the runtime dataclass
    # fields (constructor, equality, and serialization) unchanged.
    if TYPE_CHECKING:
        r_node: np.ndarray = field(init=False)
        r_half: np.ndarray = field(init=False)
        z_node: np.ndarray = field(init=False)
        z_half: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if (
            not np.all(np.isfinite([self.dr, self.dz, self.z0]))
            or min(self.dr, self.dz) <= 0
            or any(
                isinstance(n, bool) or int(n) != n or n < 4 for n in (self.nr, self.nz)
            )
        ):
            raise ValueError("grid needs positive spacings and integer sizes >=4")
        object.__setattr__(self, "nr", int(self.nr))
        object.__setattr__(self, "nz", int(self.nz))
        for name, value in (
            ("r_node", np.arange(self.nr + 1) * self.dr),
            ("r_half", (np.arange(self.nr) + 0.5) * self.dr),
            ("z_node", self.z0 + np.arange(self.nz + 1) * self.dz),
            ("z_half", self.z0 + (np.arange(self.nz) + 0.5) * self.dz),
        ):
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    def dual_bounds(
        self, component: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        r = self.r_node if component == 1 else self.r_half
        z = self.z_node if component == 0 else self.z_half
        return (
            np.maximum(r - self.dr / 2, 0),
            np.minimum(r + self.dr / 2, self.nr * self.dr),
            np.maximum(z - self.dz / 2, self.z0),
            np.minimum(z + self.dz / 2, self.z_node[-1]),
        )

    def require_aligned(self, value: Any, axis: str) -> None:
        if axis not in ("r", "z"):
            raise ValueError("alignment axis must be r or z")
        spacing, origin = (self.dr, 0) if axis == "r" else (self.dz, self.z0)
        coordinate = (np.asarray(value) - origin) / spacing
        if not np.all(np.isfinite(coordinate)) or not np.allclose(
            coordinate, np.rint(coordinate), atol=1e-10, rtol=0
        ):
            raise ValueError(
                "wall must be grid-aligned; sloped and curved walls "
                "require a qualified conformal operator"
            )


class Fields:
    er: np.ndarray
    ez: np.ndarray
    bt: np.ndarray

    def __init__(self, g: AxisymmetricGrid, backend: PICBackend | None = None) -> None:
        xp = cpu_backend(backend).xp
        self.er = xp.zeros((g.nr, g.nz + 1), dtype=xp.float64)
        self.ez = xp.zeros((g.nr + 1, g.nz), dtype=xp.float64)
        self.bt = xp.zeros((g.nr, g.nz), dtype=xp.float64)


def volumes(
    g: AxisymmetricGrid, control: tuple[float, float, float] | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Actual dual-volume intersections, including half cells at end caps."""
    out = []
    for k in range(3):
        rl, rh, zl, zh = g.dual_bounds(k)
        if control is not None:
            radius, low, high = control
            if not (0 < radius <= g.r_node[-1] and g.z0 <= low < high <= g.z_node[-1]):
                raise ValueError("diagnostic cylinder must lie inside the mesh")
            rh = np.minimum(rh, radius)
            zl, zh = np.maximum(zl, low), np.minimum(zh, high)
        out.append(
            np.pi
            * np.maximum(rh**2 - rl**2, 0)[:, None]
            * np.maximum(zh - zl, 0)[None, :]
        )
    return cast(tuple[np.ndarray, np.ndarray, np.ndarray], tuple(out))
