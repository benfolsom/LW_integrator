"""One-way axial responding probe and separately measured particle ledger.

Ported from particle_step2k.py and field_step2m.py. No material feedback.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from core.constants import C_MMNS
from core.medina_radiation_reaction import compute_medina_radiation_reaction
from ._retarded import field


def native_rr(
    q: float,
    mass: float,
    force: float,
    derivative: float,
    velocity: float,
    acceleration: float,
    *,
    gamma: float | None = None,
):
    """Convert study HL units (c=1) to the maintained kernel's native units.

    Momentum-based callers can supply gamma to avoid the ill-conditioned
    reconstruction from rounded velocity near c. Legacy callers retain it.
    """
    c = C_MMNS
    return compute_medina_radiation_reaction(
        external_force=(0.0, 0.0, force * c * c),
        external_force_time_derivative=(0.0, 0.0, derivative * c**3),
        beta=(0.0, 0.0, velocity),
        acceleration=(0.0, 0.0, acceleration * c * c),
        gamma=gamma if gamma is not None else 1 / np.sqrt(1 - velocity * velocity),
        mass=mass,
        charge=q * c / np.sqrt(4 * np.pi),
        coordinate_dt=0.0,
    )


def worldtube(gamma, order):
    """Independent angular cross-stress integral in the point-tube limit.

    Incident uniformly moving charge field, and a locally constant axial
    scattered electric field. Integrates S_cross-v U_cross and the moving
    Maxwell momentum flux. q=E_s=radius=1 for the coefficients.
    """
    beta = np.sqrt(1 - gamma**-2)
    eta, w = np.polynomial.legendre.leggauss(order)
    eta *= beta
    w *= beta
    mu = eta / (gamma * beta * np.sqrt(1 - eta**2))
    jac = 1 / (gamma * beta * (1 - eta**2) ** 1.5)
    nr = np.sqrt(1 - mu * mu)
    ei = gamma / (4 * np.pi * (1 + (gamma**2 - 1) * mu * mu) ** 1.5)
    er, ez = ei * nr, ei * mu
    bt = beta * er
    # Scattered Ez=1, B=0; area element is 2 pi dmu.
    cross_energy = ez
    sr = -bt
    sz = np.zeros_like(sr)
    outward_energy = sr * nr + sz * mu - beta * cross_energy * mu
    maxwell_zr = -er
    maxwell_zz = -ez
    outward_momentum = maxwell_zr * nr + maxwell_zz * mu
    area = 2 * np.pi * w * jac
    return dict(
        order=order,
        energy_outward_coefficient=float(np.dot(outward_energy, area)),
        momentum_outward_coefficient=float(np.dot(outward_momentum, area)),
        charge_flux=float(np.dot(ei, area)),
        energy_relative_error=float(np.dot(outward_energy, area) / (-beta) - 1),
        momentum_relative_error=float(-np.dot(outward_momentum, area) - 1),
        scope="Local point-worldtube constant-field limit; checks the cross tensor and moving-surface signs, not global scattered propagation or finite-radius convergence",
    )


def replay_particle(
    history,
    *,
    gamma,
    charge,
    mass,
    end=1e8,
    z_at_zero=-1.0,
    stop_duration=1.0,
    rtol=2e-7,
    atol=1e-13,
):
    """Responding axial probe in a sealed prescribed material record.

    The native Medina kernel receives the complete derivative along the actual
    observer trajectory. Small departures from initial momentum and position
    avoid cancellation at high gamma. Totals are for one representative
    particle; callers must weight them by population for ensemble claims.
    The worldtube cross flux is a local particle check, not global wall closure.
    """
    if (
        not history.sealed
        or not np.all(
            np.isfinite(
                [gamma, charge, mass, end, z_at_zero, stop_duration, rtol, atol]
            )
        )
        or gamma <= 1
        or min(mass, end, stop_duration, rtol, atol) <= 0
    ):
        raise ValueError(
            "particle replay needs a sealed history and positive parameters"
        )
    beta = np.sqrt(1 - gamma**-2)
    p0, e0 = mass * np.sqrt(gamma * gamma - 1), mass * gamma
    tau = charge * charge / (6 * np.pi * mass)

    def velocity(dp):
        energy = np.hypot(mass, p0 + dp)
        dv = (
            mass
            * mass
            * dp
            * (2 * p0 + dp)
            / (((p0 + dp) * e0 + p0 * energy) * energy * e0)
        )
        return beta + dv, dv, energy / mass

    def sample(t, dz, v):
        return field(
            t,
            dz,
            v,
            beta,
            *history.arrays,
            *history.geometry,
            stop_duration,
            z_at_zero,
            history.smooth_endpoints,
        )

    initial_force = charge * sample(0.0, 0.0, beta)[0]
    initial_bound_p = -tau * gamma * initial_force
    initial_bound_e = initial_bound_p * beta

    def rhs(x, y):
        t = np.expm1(x)
        dp, dz = y[:2]
        v, dv, gm = velocity(dp)
        ef, df = sample(t, dz, v)
        f, fd = charge * ef, charge * df
        pdot = f + tau * gm * fd
        acc = pdot / (mass * gm**3)
        rr = native_rr(charge, mass, f, fd, v, acc)
        prad = rr.far_radiated_power / C_MMNS**3
        native_pdot = f + rr.radiation_reaction_force[2] / C_MMNS**2
        if abs(native_pdot - pdot) > 1e-10 * max(1e-25, abs(pdot), abs(f)):
            raise RuntimeError("native Medina reaction conversion disagrees")
        lienard = charge * charge / (6 * np.pi) * gm**6 * acc * acc
        return (1 + t) * np.array([pdot, dv, f * v, f, prad, prad * v, lienard])

    sol = solve_ivp(
        rhs,
        (0.0, np.log1p(end)),
        np.zeros(7),
        method="DOP853",
        rtol=rtol,
        atol=np.array([atol, atol / 10, atol, atol, 1e-24, 1e-24, 1e-24]),
        max_step=0.025,
        dense_output=True,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    ts = np.unique(np.r_[0.0, np.geomspace(min(0.01, end), end, 351), np.expm1(sol.t)])
    dp, dz, work, impulse, rad, radp, lienard = sol.sol(np.log1p(ts))
    v, _, gm = velocity(dp)
    energy = np.hypot(mass, p0 + dp)
    kinetic = dp * (2 * p0 + dp) / (energy + e0)
    force = np.array(
        [
            charge * sample(float(t), float(zz), float(vv))[0]
            for t, zz, vv in zip(ts, dz, v)
        ]
    )
    boundp = -tau * gm * force - initial_bound_p
    bounde = -tau * gm * force * v - initial_bound_e
    er = kinetic + rad + bounde - work
    pr = dp + radp + boundp - impulse
    wt = worldtube(gamma, 128)
    # Independently integrate the local cross-stress coefficients, then multiply
    # by measured work/impulse; never assign an energy residual to cross flux.
    cross_e = work * wt["energy_outward_coefficient"] / beta
    cross_p = impulse * wt["momentum_outward_coefficient"]
    arrays = dict(
        time=ts,
        delta_momentum=dp,
        delta_z=dz,
        work=work,
        impulse=impulse,
        kinetic=kinetic,
        medina_energy=rad,
        medina_momentum=radp,
        lienard_energy=lienard,
        bound_energy=bounde,
        bound_momentum=boundp,
        energy_residual=er,
        momentum_residual=pr,
        force=force,
        local_cross_energy_flux=cross_e,
        local_cross_momentum_flux=cross_p,
    )
    final = {name: float(value[-1]) for name, value in arrays.items() if name != "time"}
    return dict(
        final=final,
        history=arrays,
        worldtube=wt,
        peak_energy_residual_relative_to_completed_work=float(
            np.max(abs(er)) / max(abs(work[-1]), 1e-30)
        ),
        peak_momentum_residual_relative_to_completed_impulse=float(
            np.max(abs(pr)) / max(abs(impulse[-1]), 1e-30)
        ),
        medina_over_lienard=float(rad[-1] / lienard[-1]) if lienard[-1] else None,
        radiation_reaction_mode="medina_lad",
        radiation_scaling="single_particle",
        material_feedback_iterated=False,
        continuation="smooth C2 current stop, then static polarization",
        accounting="one representative particle; weight totals by population",
    )
