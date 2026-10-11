"""EXPERIMENTAL wall-following PEC TM operator with an energy-adjoint metric.

The reference radial coordinate u covers 0..2 mm. A fixed inner source disk
is left unmapped; the exterior annulus follows r=R(u,z). Stored electric
fields are covariant components eu=R_u Er and ev=Ez+R_z Er; b=R_u Btheta.
A positive discrete metric supplies the physical energy and its exact inverse.
No partial cells, deleted cells, fitted work correction, or damping is used.
The map is continuous at u=core, but its radial derivative jumps. The shear
is zero at that node, although its dual ring straddles the mapped annulus;
source weights there use the reference volume rather than the mapped volume.
This local first-order inconsistency did not spoil global second-order
convergence in the reviewed taper refinement; other geometries need checks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, cast

import numpy as np
from scipy.fft import dct, dst, idct, idst  # type: ignore[import-untyped]
from scipy.linalg.lapack import dpttrf, dpttrs  # type: ignore[import-untyped]

from .mesh import AxisymmetricGrid as Grid
from .mesh import Fields, cpu_backend, volumes

if TYPE_CHECKING:
    from .conformal_observers import ConformalSnapshot

Wall = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]


class ConformalPEC:
    def __init__(
        self,
        g: Grid,
        dt: float,
        wall: Wall | None,
        core: float = 0.15,
        *,
        solver: str = "fixed_point",
        backend: Any = None,
    ) -> None:
        self.backend = cpu_backend(backend)
        if not np.all(np.isfinite([dt, core])) or dt <= 0 or not 0 < core < g.nr * g.dr:
            raise ValueError(
                "Require positive dt and a fixed source disk inside the mesh"
            )
        if wall is None:
            raise ValueError("Conformal PEC needs an explicit smooth wall function")
        if not callable(wall):
            raise TypeError("Wall must be callable")
        self.g, self.dt, self.core = g, dt, core
        self.f = Fields(g)
        self.t = 0.0
        self.last_iterations = 0
        if solver not in ("fixed_point", "cg"):
            raise ValueError("Unknown midpoint solver")
        self.solver = solver
        outer = g.nr * g.dr
        if abs(core / g.dr - round(core / g.dr)) > 1e-10:
            raise ValueError("The fixed source disk must end on a radial node")

        def metric(
            u: np.ndarray, z: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            a, slope = (np.asarray(v, dtype=float) for v in wall(z))
            if (
                a.shape != z.shape
                or slope.shape != z.shape
                or not np.all(np.isfinite([a, slope]))
                or np.any(a <= core)
            ):
                raise ValueError(
                    "Smooth wall needs finite radius/slope arrays and aperture > source disk"
                )
            # A small central difference checks the supplied tangent independently
            # of the grid spacing. Scale h with coordinate magnitude to limit
            # subtraction error on meshes far from the coordinate origin.
            h = np.cbrt(np.finfo(float).eps) * np.maximum(1.0, abs(z))
            left = np.asarray(wall(z - h)[0], dtype=float)
            right = np.asarray(wall(z + h)[0], dtype=float)
            if (
                left.shape != z.shape
                or right.shape != z.shape
                or not np.all(np.isfinite([left, right]))
            ):
                raise ValueError("Wall finite-difference radii must be finite arrays")
            finite_slope = (right - left) / (2 * h)
            slope_tolerance = 1e-6 * (np.max(abs(slope)) + g.dr / g.dz)
            if np.any(abs(slope - finite_slope) > slope_tolerance):
                raise ValueError(
                    "Wall slope is inconsistent with its radius derivative"
                )
            scale = (a - core) / (outer - core)
            outside = u[:, None] > core
            s = np.maximum(u - core, 0)[:, None] / (outer - core)
            r = np.where(outside, core + s * (a - core), u[:, None] + np.zeros_like(a))
            ru = np.where(outside, scale, 1.0)
            rz = s * slope
            return r, ru, rz

        self.radius_r, self.ru_r, _ = metric(g.r_half, g.z_node)
        self.radius_b, self.ru_b, _ = metric(g.r_half, g.z_half)
        self.radius_z, self.ru_z, rz = metric(g.r_node, g.z_half)
        v = volumes(g)
        # A current given on the fixed source mesh keeps its population
        # weights. The dual ring at the disk's boundary partly stretches,
        # so multiplying that current by the mapped volume changes charge.
        self.source_hz = v[1]
        self.hr = v[0] * self.radius_r / g.r_half[:, None] / self.ru_r
        self.hb = v[2] * self.radius_b / g.r_half[:, None] / self.ru_b
        # Exact mapped radial dual-ring area, including the finite axis disk.
        low = np.maximum(g.r_node - g.dr / 2, 0)
        high = np.minimum(g.r_node + g.dr / 2, outer)
        rl = metric(low, g.z_half)[0]
        rh = metric(high, g.z_half)[0]
        self.hz = np.pi * (rh**2 - rl**2) * g.dz
        self.shear = rz / self.ru_z
        self.rmask = np.ones_like(self.f.er, dtype=bool)
        self.rmask[:, [0, -1]] = False
        self.zmask = np.ones_like(self.f.ez, dtype=bool)
        self.zmask[-1] = False  # covariant tangential field vanishes on PEC
        # Eliminating the wall tangential unknown leaves its normal-field
        # energy in the last radial row. Its Schur complement is tridiagonal
        # in z because P averages the two adjacent radial electric edges.
        wall_metric = self.hz[-1] * self.shear[-1] ** 2 / 4
        diagonal = self.hr[-1, 1:-1] + wall_metric[:-1] + wall_metric[1:]
        offdiagonal = wall_metric[1:-1].copy()
        self.wall_diagonal, self.wall_offdiagonal, info = dpttrf(diagonal, offdiagonal)
        if info != 0:
            raise ValueError("Wall energy metric must be positive")
        self.bmask = np.ones_like(self.f.bt, dtype=bool)
        self.wave_numbers = np.pi * np.arange(1, g.nz) / (g.nz * g.dz)
        self.symbol = 2 / dt * np.tan(dt * self.wave_numbers / 2)
        # Induced-norm bound for the exact energy-scaled curl. The radial
        # incidence bound uses absolute row sums of its symmetric Gram
        # matrix; the transform derivative uses its largest singular value.
        diagonal = self.hb / g.dr**2 * (1 / self.hz[:-1] + 1 / self.hz[1:])
        off = np.sqrt(self.hb[:-1] * self.hb[1:]) / g.dr**2 / self.hz[1:-1]
        row_sum = diagonal.copy()
        row_sum[:-1] += off
        row_sum[1:] += off
        radial_bound = float(np.sqrt(np.max(row_sum)))
        axial_bound = float(
            self.symbol[-1]
            * np.max(
                np.sqrt(np.max(self.hb, axis=1) / np.min(self.hr[:, 1:-1], axis=1))
            )
        )
        scaled_shear = np.sqrt(self.hz) * abs(self.shear)
        rows = scaled_shear * self.interpolate(self.rmask / np.sqrt(self.hr))
        cols = self.transpose(scaled_shear) / np.sqrt(self.hr) * self.rmask
        shear_bound = float(np.sqrt(np.max(rows) * np.max(cols)))
        self.frequency_bound = float(
            np.hypot(radial_bound, axial_bound + radial_bound * shear_bound)
        )
        self.iteration_contraction_bound = (dt * self.frequency_bound / 2) ** 2
        # Positive zero-shear radial-block preconditioner for the magnetic
        # midpoint system. Vectorized independent Thomas factorizations.
        self.sqrt_hb = np.sqrt(self.hb)
        self.cg_d = 1 + dt**2 / 4 * diagonal
        self.cg_d[-1] -= dt**2 / 4 * self.hb[-1] / g.dr**2 / self.hz[-1]
        self.cg_l = -(dt**2) / 4 * off
        for i in range(1, g.nr):
            raw = self.cg_l[i - 1].copy()
            self.cg_l[i - 1] /= self.cg_d[i - 1]
            self.cg_d[i] -= self.cg_l[i - 1] * raw
        # Conservative contraction bound for the midpoint fixed-point solve.
        # The Cayley update itself is energy stable for any dt, but this
        # inexpensive solver requires dt*||H_b^.5 C H_e^-.5|| < 2.
        shortest = min(float(np.min(self.ru_b)) * g.dr, g.dz)
        if dt * self.wave_numbers[-1] >= np.pi / 2:
            raise ValueError("Require dt*k_max < pi/2 for the axial symbol")
        if solver == "fixed_point" and dt > 0.4 * shortest:
            raise ValueError(f"Require dt <= 0.4*min(mapped dr,dz) = {.4*shortest:g}")
        if solver == "fixed_point" and self.iteration_contraction_bound >= 0.9:
            raise ValueError(
                "Midpoint iteration norm bound requires a smaller timestep"
            )
        self.volumes = (self.hr, self.hz, self.hb)

    @staticmethod
    def interpolate(er: np.ndarray) -> np.ndarray:
        z = (er[:, :-1] + er[:, 1:]) / 2
        result = np.zeros((er.shape[0] + 1, z.shape[1]))
        result[1:-1] = (z[:-1] + z[1:]) / 2
        result[0], result[-1] = z[0], z[-1]
        return cast(np.ndarray, result)

    @staticmethod
    def transpose(ez: np.ndarray) -> np.ndarray:
        z = np.zeros((ez.shape[0] - 1, ez.shape[1]))
        z[0] += ez[0]
        z[-1] += ez[-1]
        z[:-1] += ez[1:-1] / 2
        z[1:] += ez[1:-1] / 2
        result = np.zeros((z.shape[0], z.shape[1] + 1))
        result[:, :-1] += z / 2
        result[:, 1:] += z / 2
        return cast(np.ndarray, result)

    def axial_forward(self, er: np.ndarray) -> np.ndarray:
        spectrum = np.zeros((er.shape[0], self.g.nz))
        spectrum[:, 1:] = (
            dst(er[:, 1:-1], type=1, norm="ortho", workers=1) * self.symbol
        )
        return cast(np.ndarray, idct(spectrum, type=2, norm="ortho", workers=1))

    def axial_reverse(self, bt: np.ndarray) -> np.ndarray:
        spectrum = dct(bt, type=2, norm="ortho", workers=1)
        result = np.zeros((bt.shape[0], self.g.nz + 1))
        result[:, 1:-1] = idst(
            spectrum[:, 1:] * self.symbol, type=1, norm="ortho", workers=1
        )
        return cast(np.ndarray, result)

    def circulation(self, er: np.ndarray, ez: np.ndarray) -> np.ndarray:
        return cast(np.ndarray, (ez[1:] - ez[:-1]) / self.g.dr - self.axial_forward(er))

    def metric_inverse(
        self, fr: np.ndarray, fz: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        fz = fz * self.zmask
        rhs = fr + self.transpose(self.shear * fz)
        er = rhs / self.hr * self.rmask
        edge, info = dpttrs(
            self.wall_diagonal, self.wall_offdiagonal, rhs[-1, 1:-1, None]
        )
        if info != 0:
            raise RuntimeError("Wall metric solve failed")
        er[-1, 1:-1] = edge[:, 0]
        ez = (fz / self.hz + self.shear * self.interpolate(er)) * self.zmask
        return er, ez

    def curl_b(self, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        weighted = self.hb * b
        fr = self.axial_reverse(weighted)
        fz = np.zeros_like(self.f.ez)
        fz[:-1] += weighted / self.g.dr
        fz[1:] -= weighted / self.g.dr
        return fr, fz

    def physical_ez(self, er: np.ndarray, ez: np.ndarray) -> np.ndarray:
        return cast(np.ndarray, ez - self.shear * self.interpolate(er))

    def physical_fields(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return physical staggered Er, Ez, and Btheta for force gathering."""
        return (
            self.f.er / self.ru_r,
            self.physical_ez(self.f.er, self.f.ez),
            self.f.bt / self.ru_b,
        )

    def energy(self) -> float:
        ez = self.physical_ez(self.f.er, self.f.ez)
        return 0.5 * float(
            np.sum(self.hr * self.f.er**2)
            + np.sum(self.hz * ez**2)
            + np.sum(self.hb * self.f.bt**2)
        )

    def snapshot(self) -> ConformalSnapshot:
        """Physical total fields on their mapped staggered positions.

        This is a prescribed-current backend. Its total fields are not a
        scattered-field provider for the coupled-particle seam.
        """
        from .conformal_observers import ConformalSnapshot

        return ConformalSnapshot(self)

    def _source_current(self, j: np.ndarray) -> np.ndarray:
        j = np.asarray(j, dtype=float)
        if j.shape != self.f.ez.shape or not np.all(np.isfinite(j)):
            raise ValueError("Current needs a finite matching axial staggered array")
        if np.any(
            abs(j[self.g.r_node > self.core + 1e-12])
            > 1e-13 * max(1e-30, float(np.max(abs(j))))
        ):
            raise ValueError("Current support must stay inside the fixed source disk")
        return cast(np.ndarray, j)

    def advance(self, j: np.ndarray) -> tuple[np.ndarray, float]:
        if self.solver == "cg":
            return self.advance_cg(j)
        j = self._source_current(j)
        e0r, e0z, b0 = self.f.er, self.f.ez, self.f.bt
        er, ez = e0r.copy(), e0z.copy()
        fzj = self.source_hz * j * self.zmask
        frj = -self.transpose(self.shear * fzj)
        for n in range(60):
            bm = b0 + self.dt / 4 * self.circulation(e0r + er, e0z + ez)
            fr, fz = self.curl_b(bm)
            dr, dz = self.metric_inverse(fr - frj, fz - fzj)
            nr, nz = e0r + self.dt * dr, e0z + self.dt * dz
            error = max(float(np.max(abs(nr - er))), float(np.max(abs(nz - ez))))
            er, ez = nr, nz
            # Normalized fields have no absolute amplitude floor. Zero fields
            # converge exactly; nonzero updates require relative accuracy.
            if error <= 2e-14 * max(float(np.max(abs(er))), float(np.max(abs(ez)))):
                break
        else:
            raise RuntimeError("Conformal midpoint solve failed")
        self.last_iterations = n + 1
        self.f.er, self.f.ez, self.f.bt = er, ez, 2 * bm - b0
        self.t += self.dt
        physical = self.physical_ez((e0r + er) / 2, (e0z + ez) / 2)
        return physical, float(self.dt * np.sum(fzj * physical))

    def precondition(self, rhs: np.ndarray) -> np.ndarray:
        result = rhs.copy()
        for i in range(1, self.g.nr):
            result[i] -= self.cg_l[i - 1] * result[i - 1]
        result /= self.cg_d
        for i in range(self.g.nr - 2, -1, -1):
            result[i] -= self.cg_l[i] * result[i + 1]
        return result

    def advance_cg(self, j: np.ndarray) -> tuple[np.ndarray, float]:
        """Solve the same Cayley update through its positive magnetic system.

        A = I + dt^2/4 H_b^.5 C H_e^-1 C^T H_b^.5.
        Radial block preconditioning removes the radial stiffness. There is
        no CFL stability limit; dt controls temporal accuracy and symbol
        regularity. A true, recomputed residual checks the iterative solve.
        """
        j = self._source_current(j)
        dt = self.dt
        e0r, e0z, b0 = self.f.er, self.f.ez, self.f.bt
        fzj = self.source_hz * j * self.zmask
        frj = -self.transpose(self.shear * fzj)
        jr, jz = self.metric_inverse(frj, fzj)
        predictor = b0 + dt / 2 * self.circulation(e0r, e0z)
        rhs = self.sqrt_hb * (predictor - dt**2 / 4 * self.circulation(jr, jz))

        def apply(x: np.ndarray) -> np.ndarray:
            fr, fz = self.curl_b(x / self.sqrt_hb)
            er, ez = self.metric_inverse(fr, fz)
            return cast(
                np.ndarray, x - dt**2 / 4 * self.sqrt_hb * self.circulation(er, ez)
            )

        x = self.sqrt_hb * predictor
        residual = rhs - apply(x)
        tolerance = 2e-14 * float(np.linalg.norm(rhs))
        z = self.precondition(residual)
        direction = z.copy()
        rz = float(np.sum(residual * z))
        for iteration in range(100):
            if float(np.linalg.norm(residual)) <= tolerance:
                break
            ad = apply(direction)
            denominator = float(np.sum(direction * ad))
            if denominator <= 0:
                raise RuntimeError("Magnetic midpoint system must be positive")
            alpha = rz / denominator
            x += alpha * direction
            residual -= alpha * ad
            z = self.precondition(residual)
            next_rz = float(np.sum(residual * z))
            direction = z + (next_rz / rz) * direction
            rz = next_rz
        else:
            raise RuntimeError("Conformal conjugate-gradient solve failed")
        true_residual = float(np.linalg.norm(rhs - apply(x)))
        if true_residual > 5 * tolerance:
            raise RuntimeError("Conformal true midpoint residual too large")
        self.last_iterations = iteration
        self.last_linear_residual = true_residual
        bm = x / self.sqrt_hb
        fr, fz = self.curl_b(bm)
        er, ez = self.metric_inverse(fr - frj, fz - fzj)
        erm, ezm = e0r + dt / 2 * er, e0z + dt / 2 * ez
        self.f.er, self.f.ez, self.f.bt = 2 * erm - e0r, 2 * ezm - e0z, 2 * bm - b0
        self.t += dt
        physical = self.physical_ez(erm, ezm)
        return physical, float(dt * np.sum(fzj * physical))


def select_pec(
    g: Grid,
    dt: float,
    segments: Any = None,
    *,
    boundary: str = "original",
    wall: Wall | None = None,
    core: float = 0.15,
    solver: str = "fixed_point",
    backend: Any = None,
) -> Any:
    """Opt in explicitly; the default constructs the unchanged aligned solver.

    A callable wall returns radius and its axial derivative at each requested
    position. Smoothness is the caller's physical geometry contract. Mapped
    material, incident/scattered adapters, and off-axis sources are unqualified.
    """
    if boundary == "original":
        if wall is not None:
            raise ValueError('Select boundary="conformal" explicitly for smooth walls')
        cpu_backend(backend)
        from .wake import PECWakeSolver

        return PECWakeSolver(g, dt, segments)
    if boundary == "conformal":
        if segments is not None:
            raise ValueError("Mapped walls use a wall function, not aligned segments")
        return ConformalPEC(g, dt, wall, core, solver=solver, backend=backend)
    raise ValueError("Unknown PEC boundary backend")
