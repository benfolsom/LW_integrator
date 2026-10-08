"""Thread C's prescribed Gaussian smooth-taper regression, normalized c=1.

Lengths and c*dt are in mm. This total-field reference has no particle push
or radiation-reaction model. It is separate from the default aligned solver.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from scipy.fft import dst, idct, next_fast_len  # type: ignore[import-untyped]
from scipy.special import erf  # type: ignore[import-untyped]

from .conformal import ConformalPEC
from .mesh import AxisymmetricGrid, volumes


def taper_profile(
    z: np.ndarray, case: str, length: float = 300.0
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth single-valued contraction, expansion, pair, or uniform pipe."""
    z = np.asarray(z)
    s = np.clip(z / length, 0, 1)
    f = s**3 * (10 + s * (-15 + 6 * s))
    df = 30 * s * s * (1 - s) ** 2 / length
    if case == "in":
        return 2 - f, -df
    if case == "out":
        return 1 + f, df
    if case == "pair":
        t = np.clip((z - length) / length, 0, 1)
        return (
            2 - f + t**3 * (10 + t * (-15 + 6 * t)),
            -df + 30 * t * t * (1 - t) ** 2 / length,
        )
    if case in ("pipe1", "pipe2"):
        return np.full_like(z, float(case[-1])), np.zeros_like(z)
    raise ValueError("unknown smooth-taper reference case")


def conformal_wake(
    case: str,
    *,
    dr: float = 0.05,
    dz: float = 0.125,
    dt: float = 0.05,
    core: float = 0.2,
    length: float = 300.0,
    pre: float = 12.0,
    post: float = 32.0,
    margin: float = 40.0,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    """Reproduce current work and an independent physical Gaussian gather.

    Source N=10^6, gamma=1000, sigma_z=1 mm, and observer charge=e. Work
    converts to eV per representative electron, not total bunch energy. The
    fixed disk's reference-volume weights and spectral continuity are retained.
    Positive current work denotes energy received by the prescribed source;
    no background subtraction or fitted correction is made.
    """
    if (
        case not in ("in", "out", "pair", "pipe1", "pipe2")
        or not np.all(np.isfinite([dr, dz, dt, core, length, pre, post, margin]))
        or min(dr, dz, dt, core, length, pre, post, margin) <= 0
    ):
        raise ValueError("reference needs a supported case and positive parameters")
    beta = np.sqrt(1 - 1000**-2)
    last = length * (2 if case == "pair" else 1)
    start, end = -pre, last + post
    zlo, zhi = start - margin, end + margin
    nz = next_fast_len(int(np.ceil((zhi - zlo) / dz)), real=True)
    if not np.isclose(round(2 / dr) * dr, 2, rtol=0, atol=1e-12):
        raise ValueError("reference outer radius 2 mm must align with the grid")
    g = AxisymmetricGrid(dr, dz, round(2 / dr), nz, zlo)
    solver = ConformalPEC(
        g, dt, lambda z: taper_profile(z, case, length), core, solver="cg"
    )

    def gaussian(z: np.ndarray, center: float) -> Any:
        return (
            erf((z + dz / 2 - center) / np.sqrt(2))
            - erf((z - dz / 2 - center) / np.sqrt(2))
        ) / (2 * dz)

    enclosed = np.minimum((g.r_half / core) ** 2, 1.0)
    coeff = enclosed / (2 * np.pi * g.r_half)
    rho_r = np.zeros(g.nr + 1)
    rho_r[0] = 4 * coeff[0] / dr
    rho_r[1:-1] = (g.r_half[1:] * coeff[1:] - g.r_half[:-1] * coeff[:-1]) / (
        g.r_node[1:-1] * dr
    )
    radial_weights = rho_r * volumes(g)[1][:, 0] / dz
    lam0 = gaussian(g.z_node, start)
    cr = np.minimum((solver.radius_r / core) ** 2, 1) / (2 * np.pi * solver.radius_r)
    cb = np.minimum((solver.radius_b / core) ** 2, 1) / (2 * np.pi * solver.radius_b)
    solver.f.er = solver.ru_r * cr * lam0[None, :] * solver.rmask
    solver.f.bt = beta * solver.ru_b * cb * gaussian(g.z_half, start)[None, :]
    solver.t = start / beta
    initial = solver.energy()
    work = gathered = 0.0
    xi = np.linspace(-6, 6, 241)
    lamxi = np.exp(-(xi**2) / 2) / np.sqrt(2 * np.pi)
    history = []
    steps = int(np.ceil((end - start) / (beta * dt)))
    for n in range(steps):
        t0, t1 = solver.t, solver.t + dt
        lam1 = gaussian(g.z_node, beta * t1)
        spectrum = np.zeros(g.nz)
        spectrum[1:] = (
            dst((lam1 - lam0)[1:-1] / dt, type=1, norm="ortho", workers=1)
            / solver.symbol
        )
        spectrum[0] = beta / (g.nz * dz) * np.sqrt(g.nz)
        j = rho_r[:, None] * idct(spectrum, type=2, norm="ortho", workers=1)[None, :]
        ez, dw = solver.advance(j)
        work += dw
        # q_observer * beta * E dt; source population enters only EV_SCALE.
        de = (
            beta
            * dt
            * np.interp(
                beta * (t0 + t1) / 2 + xi,
                g.z_half,
                np.sum(ez * radial_weights[:, None], axis=0),
            )
        )
        gathered += float(np.trapezoid(de * lamxi, xi))
        lam0 = lam1
        if n % max(1, steps // 120) == 0 or n == steps - 1:
            energy = solver.energy()
            history.append(
                [solver.t, work, gathered, energy - initial, energy - initial + work]
            )
        if progress is not None and (n % max(1, steps // 16) == 0 or n == steps - 1):
            progress(n + 1, steps)
    # HL source is normalized to a physical bunch of 10^6 electrons. Gathering
    # with one electron's charge gives N e/(epsilon_0 L) in eV, L=1 mm.
    ev_scale = 1e6 * 1.602176634e-19 / (8.8541878128e-12 * 1e-3)
    result = {
        "case": case,
        "dr": dr,
        "dz": dz,
        "dt": dt,
        "core": core,
        "steps": steps,
        "work_eV_per_electron": work * ev_scale,
        "physical_gather_eV_per_electron": gathered * ev_scale,
        "reference_eV_per_electron": 0.563121173
        * (1 if case == "in" else -1 if case == "out" else 0),
        "peak_energy_identity_residual_hl_mm": float(
            np.max(abs(np.asarray(history)[:, -1]))
        ),
        "source_population": 1e6,
        "gamma": 1000.0,
        "radiation_reaction_mode": "prescribed source; not evolved",
    }
    return result, np.asarray(history)
