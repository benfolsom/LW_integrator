"""EXPERIMENTAL finite-window, axial two-way material/particle coupling.

Ported from thread C's smooth step-2m gate. Each material pass starts cold,
driven by the preceding actual C2 particle trajectory and its inertial
prehistory. The sealed material record supplies retarded Lorentz force and
its complete trajectory derivative to the axial particle integrator. No
production RK/PIC defaults, native Medina kernel, or mapped fields are changed.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np
from scipy.interpolate import BPoly, PPoly  # type: ignore[import-untyped]

from ._retarded import field
from .axial_particle import integrate_axial_particle
from .history import MaterialHistory
from .incident import BallisticDrive, point_fields, uniform_axial_average
from .materials import DrudeWall
from .solver import ScatteredFieldSolver


@dataclass(frozen=True)
class AxialParticle:
    """Explicit observer inertia/charge, source charge, and diagnostic weight.

    Source charge defaults to population*charge. Radiation is calculated for
    one particle; population-weighted radiation uses the incoherent convention.
    A different coherence model requires an independently defined diagnostic.
    All values use normalized Heaviside–Lorentz units with c=1.
    """

    gamma: float
    charge: float
    mass: float
    population: float = 1.0
    source_charge: float | None = None
    z_at_zero: float = -1.0

    def __post_init__(self) -> None:
        source = (
            self.population * self.charge
            if self.source_charge is None
            else self.source_charge
        )
        if (
            not np.all(
                np.isfinite(
                    [
                        self.gamma,
                        self.charge,
                        self.mass,
                        self.population,
                        source,
                        self.z_at_zero,
                    ]
                )
            )
            or self.gamma <= 1
            or min(self.mass, self.population) <= 0
            or self.beta >= 1
        ):
            raise ValueError(
                "axial particle needs gamma>1, finite charges, and positive mass/population"
            )
        object.__setattr__(self, "source_charge", float(source))

    @property
    def beta(self) -> float:
        return cast(float, np.sqrt(1 - self.gamma**-2))


class AxialTrajectory:
    """Owned C2 residual path, with inertial prehistory and no future extrapolation."""

    def __init__(
        self,
        times: Any,
        delta_z: Any,
        delta_beta: Any,
        acceleration: Any,
        *,
        beta: float,
        z_at_zero: float,
    ) -> None:
        ts, dz, dv, acc = (
            np.array(a, dtype=float, copy=True)
            for a in (times, delta_z, delta_beta, acceleration)
        )
        if (
            ts.ndim != 1
            or len(ts) < 2
            or any(a.shape != ts.shape for a in (dz, dv, acc))
            or not np.all(np.isfinite([ts, dz, dv, acc]))
            or ts[0] != 0
            or np.any(np.diff(ts) <= 0)
            or any(a[0] != 0 for a in (dz, dv, acc))
            or not np.all(np.isfinite([beta, z_at_zero]))
            or not 0 < beta < 1
        ):
            raise ValueError(
                "C2 trajectory needs increasing times from zero and zero initial departures"
            )
        self.beta, self.z0 = beta, z_at_zero
        self.spline = BPoly.from_derivatives(ts, np.column_stack((dz, dv, acc)))
        self.x = self.spline.x
        velocity = PPoly.from_bernstein_basis(self.spline).derivative()
        critical = velocity.derivative().roots(extrapolate=False)
        critical = critical[np.isfinite(critical)]
        if np.any(abs(beta + velocity(np.r_[ts, critical])) >= 1):
            raise ValueError("C2 trajectory must remain subluminal between knots")
        self.x.setflags(write=False)
        self.spline.c.setflags(write=False)

    def __call__(self, t: Any, order: int = 0) -> Any:
        t = np.asarray(t)
        if np.any(t > self.x[-1] + 1e-8):
            raise RuntimeError("Observation requires unsaved future source history")
        if not np.all(np.isfinite(t)):
            raise ValueError(
                "trajectory requires finite times within the saved source window"
            )
        value = np.where(t < 0, 0.0, self.spline(np.minimum(t, self.x[-1]), nu=order))
        return value + (
            self.z0 + self.beta * t if order == 0 else self.beta if order == 1 else 0.0
        )


class RespondingDrive(BallisticDrive):
    """Material-only volume drive including responding velocity and acceleration.

    This optimized support-only projection preserves step 2m's eight-point
    ballistic radial quadrature and independently chosen correction quadrature.
    No grid deposition of a singular point charge or incident-field feedback.
    """

    def __init__(
        self,
        grid: Any,
        particle: AxialParticle,
        trajectory: AxialTrajectory | None = None,
        *,
        order: int = 8,
        correction_order: int = 2,
    ) -> None:
        super().__init__(
            grid, particle.gamma, particle.source_charge, order, particle.z_at_zero
        )
        if int(correction_order) != correction_order or correction_order < 1:
            raise ValueError("correction quadrature needs a positive integer order")
        if trajectory is not None and (
            not np.isclose(trajectory.beta, self.beta, rtol=0, atol=1e-15)
            or trajectory.z0 != self.z0
        ):
            raise ValueError("responding trajectory and incident reference must match")
        self.path = trajectory
        self.correction_nodes, self.correction_weights = (
            np.polynomial.legendre.leggauss(correction_order)
        )
        self.material_cache: dict[float, list[np.ndarray]] = {}
        self.correction_max = 0.0

    def position(self, t: float) -> Any:
        return super().position(t) if self.path is None else self.path(t)

    def corrections(self, r: Any, z: Any, t: float) -> list[np.ndarray]:
        r, z = np.broadcast_arrays(
            np.asarray(r, dtype=float), np.asarray(z, dtype=float)
        )
        out = [np.zeros_like(r) for _ in range(3)]
        active = t > np.hypot(r, z - self.z0)
        if self.path is not None and np.any(active):
            vel, acc = point_fields(
                self.path, self.q, r[active], z[active], np.full(np.sum(active), t)
            )
            gap = z[active] - self.z0 - self.beta * t
            kernel = (
                self.q
                * self.gamma
                / (4 * np.pi * (r[active] ** 2 + (self.gamma * gap) ** 2) ** 1.5)
            )
            baseline = (
                kernel * r[active],
                kernel * gap,
                self.beta * kernel * r[active],
            )
            for k in range(3):
                out[k][active] = vel[k] + acc[k] - baseline[k]
        return out

    def material_volume(
        self, g: Any, t: float, masks: tuple[np.ndarray, ...]
    ) -> list[np.ndarray]:
        if g is not self.g:
            raise ValueError("incident provider and material grid must match")
        if t in self.material_cache:
            return self.material_cache[t]
        base = [np.zeros((g.nr, g.nz + 1)), np.zeros((g.nr + 1, g.nz))]
        for k in (0, 1):
            ri, zi = np.nonzero(masks[k])
            rl, rh, zl, zh = self.geometry[k]
            for x, wx in zip(self.nodes, self.weights):
                r = (rl[ri] + rh[ri]) / 2 + (rh[ri] - rl[ri]) * x / 2
                base[k][ri, zi] += (
                    wx
                    * r
                    / (rl[ri] + rh[ri])
                    * uniform_axial_average(
                        r,
                        zl[zi],
                        zh[zi],
                        self.z0 + self.beta * t,
                        self.q,
                        self.beta,
                        0.0,
                        k,
                    )
                )
            if self.path is not None:
                value = np.zeros(len(ri))
                for x, wx in zip(self.correction_nodes, self.correction_weights):
                    r = (rl[ri] + rh[ri]) / 2 + (rh[ri] - rl[ri]) * x / 2
                    for y, wy in zip(self.correction_nodes, self.correction_weights):
                        z = (zl[zi] + zh[zi]) / 2 + (zh[zi] - zl[zi]) * y / 2
                        value += (
                            wx
                            * wy
                            / 2
                            * r
                            / (rl[ri] + rh[ri])
                            * self.corrections(r, z, t)[k]
                        )
                self.correction_max = max(
                    self.correction_max, float(np.max(abs(value), initial=0))
                )
                base[k][ri, zi] += value
        self.material_cache = {t: base}
        return base

    def volume(self, g: Any, t: float) -> Any:
        if self.path is not None:
            raise ValueError("responding drive supports material projection only")
        return super().volume(g, t)

    def edges_mid(self, *args: Any) -> None:
        raise ValueError("coupled drive requires endpoint volume projection")


class AxialBoundaryForce:
    """Boundary-to-integrator seam: force and complete trajectory derivative.

    MaterialHistory gathers only the retarded material field, avoiding double
    counting of the particle's incident/self field. Source macro charge never
    enters the observer Lorentz multiplier, observer mass, or native RR charge.
    """

    def __init__(
        self,
        history: MaterialHistory,
        particle: AxialParticle,
        stop_duration: float = 1.0,
    ) -> None:
        if (
            not isinstance(history, MaterialHistory)
            or not history.sealed
            or not history.smooth_endpoints
        ):
            raise ValueError("coupled force requires a sealed smooth material history")
        if not np.isfinite(stop_duration) or stop_duration <= 0:
            raise ValueError("current-stop duration must be positive")
        self.history, self.particle, self.stop_duration = (
            history,
            particle,
            stop_duration,
        )

    def __call__(
        self, t: float, delta_z: float, velocity: float
    ) -> tuple[float, float]:
        h, p = self.history, self.particle
        e, derivative = field(  # type: ignore[call-arg]
            t,
            delta_z,
            velocity,
            p.beta,
            *h.arrays,
            *h.geometry,
            self.stop_duration,
            p.z_at_zero,
            True,
        )
        return p.charge * e, p.charge * derivative


class TwoWayBoundaryCoupling:
    """Serial finite-record feedback, disabled unless enabled=True.

    solver_factory(drive) must construct a fresh, cold ScatteredFieldSolver
    on the declared grid and smooth DrudeWall. No convergence means an error,
    rather than an implicitly accepted final iterate. The record ends at a
    declared source_end, then current stops smoothly and polarization freezes.
    Convergence qualifies only pass-to-pass feedback, not window or spatial
    accuracy. Results expose post-source-window work and impulse, and flag
    unmeasured source_end/stop_duration sensitivity and force reconstruction
    uncertainty. Repeat with longer source windows, varied stop durations, and
    refined meshes before making accuracy claims.
    This is an axial experimental coupling entry point, not unrestricted 3D
    responding-source deposition, mapped coupling, or whole-system closure.
    """

    def __init__(
        self,
        grid: Any,
        particle: AxialParticle,
        solver_factory: Callable[[RespondingDrive], ScatteredFieldSolver],
        *,
        enabled: bool = False,
    ) -> None:
        self.grid, self.particle, self.solver_factory = grid, particle, solver_factory
        self.enabled = enabled

    def run(
        self,
        *,
        source_end: float = 8.0,
        observation_end: float = 1e6,
        max_feedback_updates: int = 8,
        tolerance: float = 1e-4,
        order: int = 4,
        correction_order: int = 2,
        stop_duration: float = 1.0,
        directory: str | Path | None = None,
        progress: Callable[[int, str, float], None] | None = None,
        rtol: float = 2e-10,
        atol: float = 1e-17,
        radiation_rtol: float = 2e-6,
        radiation_atol: float = 1e-22,
    ) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        if (
            not np.all(np.isfinite([source_end, observation_end, tolerance]))
            or source_end <= 0
            or observation_end < source_end
            or tolerance <= 0
            or isinstance(max_feedback_updates, bool)
            or int(max_feedback_updates) != max_feedback_updates
            or max_feedback_updates < 1
        ):
            raise ValueError(
                "feedback needs a positive finite window, tolerance, and update count"
            )
        rows: list[dict[str, Any]] = []
        trajectory: AxialTrajectory | None = None
        for index in range(max_feedback_updates + 1):
            drive = RespondingDrive(
                self.grid, self.particle, trajectory, correction_order=correction_order
            )
            solver = self.solver_factory(drive)
            if (
                type(solver) is not ScatteredFieldSolver
                or solver.g is not self.grid
                or solver.history is not drive
                or not isinstance(solver.wall, DrudeWall)
                or solver.wall.ramp_duration is None
                or solver.wall.aperture_radius is None
                or solver.t != 0
                or solver.projection != "volume"
                or any(
                    np.any(a)
                    for a in (
                        solver.f.er,
                        solver.f.ez,
                        solver.f.bt,
                        *solver.x,
                        *solver.v,
                    )
                )
            ):
                raise ValueError(
                    "feedback factory must return a fresh cold smooth aligned material solver using the supplied drive"
                )
            ticks = source_end / solver.dt
            if not np.isclose(ticks, round(ticks), rtol=0, atol=1e-10):
                raise ValueError("source_end must coincide with a material endpoint")
            record_dir = (
                None if directory is None else Path(directory) / f"pass_{index}"
            )
            if record_dir is not None and record_dir.exists():
                raise ValueError(
                    "feedback output directory already exists; preserve prior records"
                )
            record = MaterialHistory(solver, round(ticks) + 1, order, record_dir)
            record.append()
            for n in range(round(ticks)):
                solver.step()
                record.append()
                if progress is not None and (
                    n % max(1, round(ticks) // 16) == 0 or n + 1 == round(ticks)
                ):
                    progress(index, "material", solver.t)
            record.seal()
            if progress is not None:
                progress(index, "particle", observation_end)
            result = integrate_axial_particle(
                AxialBoundaryForce(record, self.particle, stop_duration),
                gamma=self.particle.gamma,
                charge=self.particle.charge,
                mass=self.particle.mass,
                source_end=source_end,
                end=observation_end,
                rtol=rtol,
                atol=atol,
                radiation_rtol=radiation_rtol,
                radiation_atol=radiation_atol,
                progress=(
                    None if progress is None else partial(progress, index, "particle")
                ),
            )
            new_path = AxialTrajectory(
                **result["trajectory"],
                beta=self.particle.beta,
                z_at_zero=self.particle.z_at_zero,
            )
            history = result["history"]
            source_index = int(np.searchsorted(history["time"], source_end))
            continuation_terms = {
                f"post_source_window_{term}": float(
                    result["final"][term] - history[term][source_index]
                )
                for term in ("work", "impulse")
            }
            row = dict(
                iteration=index,
                **result["final"],
                peak_energy_residual_relative=result["peak_energy_residual_relative"],
                peak_momentum_residual_relative=result[
                    "peak_momentum_residual_relative"
                ],
                max_relative_gamma_change=result["max_relative_gamma_change"],
                medina_over_lienard=result["medina_over_lienard"],
                peak_incident_correction=drive.correction_max,
                **continuation_terms,
            )
            converged = False
            if rows:
                assert trajectory is not None
                for term in ("work", "impulse"):
                    row[f"relative_{term}_change"] = abs(
                        row[term] - rows[-1][term]
                    ) / max(1e-30, abs(row[term]))
                tt = new_path.x
                zz_new, zz_old = new_path.spline(tt), trajectory.spline(tt)
                vv_new, vv_old = new_path.spline(tt, nu=1), trajectory.spline(tt, nu=1)
                aperture = solver.wall.aperture_radius
                if aperture is None:
                    raise ValueError(
                        "axial feedback qualification needs an explicit source aperture"
                    )
                row["source_position_change_on_incident_width"] = float(
                    np.max(abs(zz_new - zz_old)) / (aperture / self.particle.gamma)
                )
                beta = self.particle.beta
                inverse_new = self.particle.gamma**-2 - 2 * beta * vv_new - vv_new**2
                inverse_old = self.particle.gamma**-2 - 2 * beta * vv_old - vv_old**2
                row["source_gamma_change_relative"] = float(
                    np.max(
                        abs((vv_new - vv_old) * (2 * beta + vv_new + vv_old))
                        / (
                            np.sqrt(inverse_old)
                            * (np.sqrt(inverse_old) + np.sqrt(inverse_new))
                        )
                    )
                )
                row["achieved_feedback_change"] = max(
                    row["relative_work_change"],
                    row["relative_impulse_change"],
                    row["source_position_change_on_incident_width"],
                    row["source_gamma_change_relative"],
                )
                converged = row["achieved_feedback_change"] < tolerance
            rows.append(row)
            if record_dir is not None:
                np.save(record_dir / "times.npy", record.arrays[0])
                np.savez_compressed(record_dir / "particle.npz", **result["history"])
                np.savez_compressed(
                    record_dir / "trajectory.npz", **result["trajectory"]
                )
            if converged:
                result.update(
                    converged=True,
                    feedback_updates=index,
                    coupling_tolerance=tolerance,
                    achieved_feedback_change=row["achieved_feedback_change"],
                    convergence_scope="pass-to-pass feedback only; window and spatial accuracy unqualified",
                    continuation_diagnostics=continuation_terms,
                    uncertainties={
                        "finite_window": {
                            "source_end": source_end,
                            "stop_duration": stop_duration,
                            "sensitivity_measured": False,
                            "qualification": "Vary source_end and stop_duration independently; post-window work includes retarded saved sources and terminal continuation, not an isolated frozen-tail estimate.",
                        },
                        "force_reconstruction": {
                            "comparison_measured_for_this_run": False,
                            "review_evidence": "Dipole P/J reconstruction versus FDTD on-axis Ez differed by 3–10% at h=0.1 and 1.7–2.6% at h=0.05 in the reviewed iris, gamma=10, dt=h/10 comparison.",
                            "qualification": "Geometry- and resolution-dependent evidence, not an error bound for this run. Particle force uses dipole reconstruction; field/surface ledgers use FDTD. Require independent mesh and quadrature convergence.",
                        },
                    },
                    iterations=rows,
                    material_history=record,
                    material_feedback_iterated=True,
                    source_end=source_end,
                    observation_end=observation_end,
                    continuation="smooth C2 current stop, then static polarization",
                    source_charge=self.particle.source_charge,
                    observer_charge=self.particle.charge,
                    observer_mass=self.particle.mass,
                    diagnostic_population=self.particle.population,
                    population_weighted_final={
                        k: self.particle.population * v
                        for k, v in result["final"].items()
                        if k
                        in (
                            "work",
                            "impulse",
                            "kinetic",
                            "delta_momentum",
                            "medina_energy",
                            "medina_momentum",
                            "bound_energy",
                            "bound_momentum",
                        )
                    },
                    radiation_scaling="single particle; population totals assume incoherent emission",
                    scope="experimental smooth axial finite-window feedback; whole-system accounting and unrestricted late material evolution remain open",
                )
                return result
            trajectory = new_path
            # Only one material record and one solver remain live per serial pass.
            del record, result, solver
        raise RuntimeError(
            "material/particle feedback did not converge within the update limit"
        )
