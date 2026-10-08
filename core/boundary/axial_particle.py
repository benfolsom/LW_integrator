"""EXPERIMENTAL step-2m axial particle integration through a force-provider seam."""

from __future__ import annotations

from typing import Any, Callable, cast

import numpy as np
from scipy.integrate import solve_ivp  # type: ignore[import-untyped]

from core.constants import C_MMNS

from .particle import native_rr


def integrate_axial_particle(
    force_provider: Callable[[float, float, float], tuple[float, float]],
    *,
    gamma: float,
    charge: float,
    mass: float,
    source_end: float,
    end: float = 1e6,
    rtol: float = 2e-10,
    atol: float = 1e-17,
    radiation_rtol: float = 2e-6,
    radiation_atol: float = 1e-22,
    progress: Callable[[float], None] | None = None,
) -> dict[str, Any]:
    """EXPERIMENTAL step-2m axial Medina/LAD integrator in normalized c=1.

    force_provider(t, delta_z, velocity) returns observer Lorentz force and
    its complete lab-time derivative along the actual axial trajectory.
    The force must be axial electric F(t,z), independent of observer velocity;
    its total trajectory derivative depends on that velocity.
    delta_z is measured from z=z0+beta0*t. The force must start smoothly at
    zero. This exact rapidity change of coordinate qualifies only axial motion;
    it does not change the production RK momentum push or its default path.

    Integrate w=eta-eta0-tau*F/m, with dw/dt=F/(m*gamma). Evaluate the unchanged
    native reaction and emission independently. Work is F*velocity*dt, while
    impulse is F*dt. Return a C2 source-trajectory record for material feedback.
    """
    if (
        not np.all(
            np.isfinite(
                [
                    gamma,
                    charge,
                    mass,
                    source_end,
                    end,
                    rtol,
                    atol,
                    radiation_rtol,
                    radiation_atol,
                ]
            )
        )
        or gamma <= 1
        or min(mass, source_end, rtol, atol, radiation_rtol, radiation_atol) <= 0
        or end < source_end
    ):
        raise ValueError(
            "axial integration needs gamma>1, positive parameters, and end>=source_end"
        )
    beta = np.sqrt(1 - gamma**-2)
    if beta >= 1:
        raise ValueError("particle gamma must give a subluminal float64 velocity")
    if force_provider(0.0, 0.0, beta) != (0.0, 0.0):
        raise ValueError("axial integration requires a smooth zero initial force")
    p0, e0 = mass * np.sqrt(gamma**2 - 1), mass * gamma
    tau = charge * charge / (6 * np.pi * mass)
    eta0 = np.arccosh(gamma)
    calls = 0

    def momentum(t: float, w: float, dz: float) -> Any:
        f = force_provider(t, dz, beta)[0]
        de = w + tau * f / mass
        # Stable difference m*(sinh(eta0+de)-sinh(eta0)). Axial Ez is
        # independent of observer velocity; its derivative is evaluated below.
        return 2 * mass * np.sinh(de / 2) * np.cosh(eta0 + de / 2)

    def velocity(dp: Any) -> tuple[Any, Any, Any]:
        if np.any(p0 + dp <= 0):
            raise RuntimeError("axial reference requires motion along positive z")
        energy = np.hypot(mass, p0 + dp)
        dv = (
            mass
            * mass
            * dp
            * (2 * p0 + dp)
            / (((p0 + dp) * e0 + p0 * energy) * energy * e0)
        )
        return beta + dv, dv, energy / mass

    def force(t: float, dp: float, dz: float) -> tuple[Any, ...]:
        v, dv, gm = velocity(dp)
        f, fd = force_provider(t, dz, v)
        pdot = f + tau * gm * fd
        return f, fd, pdot, v, dv, gm

    def rhs(x: float, y: np.ndarray) -> np.ndarray:
        nonlocal calls
        t = min(float(np.expm1(x)), end)
        calls += 1
        if progress is not None and calls % 10000 == 0:
            progress(t)
        w, dz = y[:2]
        dp = momentum(t, w, dz)
        f, fd, pdot, v, dv, gm = force(t, dp, dz)
        acc = pdot / (mass * gm**3)
        # Use the same on-shell momentum gamma in both formulations. Recovering
        # gamma from rounded v loses precision at high gamma; the kernel accepts
        # gamma directly, and the axial acceleration terms cancel algebraically.
        rr = native_rr(charge, mass, f, fd, v, acc, gamma=gm)
        prad = rr.far_radiated_power / C_MMNS**3
        native = f + rr.radiation_reaction_force[2] / C_MMNS**2
        if abs(native - pdot) > 1e-10 * max(1e-25, abs(pdot), abs(f)):
            raise RuntimeError("native Medina reaction conversion disagrees")
        lienard = charge * charge / (6 * np.pi) * gm**6 * acc * acc
        return cast(
            np.ndarray,
            (1 + t)
            * np.array(
                [
                    f / (mass * gm),
                    dv,
                    f * v,
                    f,
                    prad,
                    prad * v,
                    lienard,
                    lienard * v,
                ]
            ),
        )

    sol = solve_ivp(
        rhs,
        (0.0, np.log1p(end)),
        np.zeros(8),
        method="DOP853",
        rtol=np.array([rtol] * 4 + [radiation_rtol] * 4),
        atol=np.array(
            [atol / (mass * gamma), atol / 10, atol, atol] + [radiation_atol] * 4
        ),
        max_step=0.02,
        dense_output=True,
    )
    if not sol.success:
        raise RuntimeError(sol.message)

    def actual(times: np.ndarray) -> Any:
        y = sol.sol(np.log1p(times)).copy()
        y[0] = [momentum(t, w, dz) for t, w, dz in zip(times, y[0], y[1])]
        return y

    times = np.unique(
        np.r_[
            0.0,
            source_end,
            np.geomspace(min(0.001, end), end, 501),
            np.clip(np.expm1(sol.t), 0.0, end),
            end,
        ]
    )
    dp, dz, work, impulse, rad, radp, lienard, lienardp = actual(times)
    v, _, gm = velocity(dp)
    energy = np.hypot(mass, p0 + dp)
    kinetic = dp * (2 * p0 + dp) / (energy + e0)
    f = np.array([force(t, p, z)[0] for t, p, z in zip(times, dp, dz)])
    boundp, bounde = -tau * gm * f, -tau * gm * f * v
    er, pr = kinetic + rad + bounde - work, dp + radp + boundp - impulse
    path_t = np.unique(
        np.r_[
            np.linspace(0, source_end, round(source_end / 0.02) + 1),
            times[times <= source_end],
        ]
    )
    pp, zz = actual(path_t)[:2]
    _, dd, gg = velocity(pp)
    aa = np.array(
        [force(t, p, z)[2] / (mass * g**3) for t, p, z, g in zip(path_t, pp, zz, gg)]
    )
    arrays = {
        "time": times,
        "delta_momentum": dp,
        "delta_z": dz,
        "work": work,
        "impulse": impulse,
        "kinetic": kinetic,
        "medina_energy": rad,
        "medina_momentum": radp,
        "lienard_energy": lienard,
        "lienard_momentum": lienardp,
        "force": f,
        "bound_energy": bounde,
        "bound_momentum": boundp,
        "energy_residual": er,
        "momentum_residual": pr,
    }
    if not all(np.all(np.isfinite(a)) for a in arrays.values()) or np.any(abs(v) >= 1):
        raise RuntimeError("axial integration produced an invalid particle state")
    final = {k: float(a[-1]) for k, a in arrays.items() if k != "time"}
    return {
        "final": final,
        "history": arrays,
        "trajectory": {
            "times": path_t,
            "delta_z": zz,
            "delta_beta": dd,
            "acceleration": aa,
        },
        "peak_energy_residual_relative": float(
            np.max(abs(er)) / max(abs(work[-1]), 1e-30)
        ),
        "peak_momentum_residual_relative": float(
            np.max(abs(pr)) / max(abs(impulse[-1]), 1e-30)
        ),
        "max_relative_gamma_change": float(np.max(abs(kinetic)) / e0),
        "medina_over_lienard": float(rad[-1] / lienard[-1]) if lienard[-1] else None,
        "solver_coordinate": "rapidity minus tau force / mass",
        "radiation_reaction_mode": "medina_lad",
        "radiation_scaling": "single_particle",
        "accounting": "one representative particle; weight totals by population",
    }
