"""Serial prescribed expanding-Gaussian controls; no trajectory/energy claim.

Independent analytic width response and vectorized light-cone bisection feed
the textbook SI point-charge LW formula. Tensor Gaussian quadrature supplies
the high-resolution reference. No production history or field kernel is used
by the reference. RR is off explicitly: these are prescribed sources.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from core.pic import CloudCorrection, CorrectionConfig, Species
from core.pic.kernels import COULOMB, C

SIGMA = np.array([0.4, 0.5, 0.6]) * 1e-3
RATE = np.array([0.6, 0.8, 1.0]) * 1e6
CHARGE = 1.602176634e-19
POPULATION = 1e7
PROBES = np.array([[5, 1, 2], [-3, 5, 1], [2, -1, 6]]) * 1e-3


def gaussian(order):
    x, w = np.polynomial.hermite.hermgauss(order)
    nodes = np.stack(np.meshgrid(x, x, x, indexing="ij"), -1).reshape(-1, 3)
    weights = (w[:, None, None] * w[None, :, None] * w[None, None, :]).ravel()
    return np.sqrt(2) * nodes, weights / weights.sum()


def response(t, tau):
    """Exact third-stage response to target sigma(t)=SIGMA+RATE*t."""
    t = np.maximum(t, 0)
    r = t / tau
    decay = np.exp(-r)
    width = SIGMA + RATE * (
        t[..., None] - 3 * tau + tau * (decay * (3 + 2 * r + 0.5 * r**2))[..., None]
    )
    speed = RATE * (1 - decay * (1 + r + 0.5 * r**2))[..., None]
    acceleration = RATE / tau * (0.5 * decay * r**2)[..., None]
    return width, speed, acceleration


def reference(order, t, tau):
    nodes, fractions = gaussian(order)
    # All roots bracketed, including the inertial t<0 continuation.
    lower = np.full((len(PROBES), len(nodes)), -1 / C)
    upper = np.full_like(lower, t)
    for _ in range(64):
        middle = (lower + upper) / 2
        widths = response(middle, tau)[0]
        r = PROBES[:, None] - nodes * widths
        residual = C * (t - middle) - np.linalg.norm(r, axis=-1)
        lower = np.where(residual > 0, middle, lower)
        upper = np.where(residual > 0, upper, middle)
    retarded = (lower + upper) / 2
    widths, speed, acceleration = response(retarded, tau)
    r = PROBES[:, None] - nodes * widths
    distance = np.linalg.norm(r, axis=-1)
    n = r / distance[..., None]
    beta = nodes * speed / C
    beta_dot = nodes * acceleration / C
    if np.any(np.sum(beta**2, axis=-1) >= 1):
        raise ValueError("reference quadrature source is not subluminal")
    kappa = 1 - np.sum(n * beta, axis=-1)
    electric_terms = (
        COULOMB
        * CHARGE
        * POPULATION
        * fractions[None, :, None]
        * (
            (1 - np.sum(beta**2, axis=-1))[..., None]
            * (n - beta)
            / (kappa**3 * distance**2)[..., None]
            + np.cross(n, np.cross(n - beta, beta_dot))
            / (C * kappa**3 * distance)[..., None]
        )
    )
    field = np.stack((electric_terms.sum(1), np.cross(n, electric_terms).sum(1)))
    width = response(np.array(t), tau)[0]
    r = PROBES[:, None] - nodes * width
    # The PIC baseline uses centroid velocity (zero here), leaving the
    # internal expansion current in the additive correction.
    terms = (
        COULOMB
        * CHARGE
        * POPULATION
        * fractions[None, :, None]
        * r
        / np.linalg.norm(r, axis=-1)[..., None] ** 3
    )
    quasi = np.stack((terms.sum(1), np.zeros_like(PROBES)))
    return field, quasi


def total(corr):
    qe, qb = corr.quasi_static(PROBES)
    de, db = corr.direct(PROBES)
    return np.stack((qe + de, C * (qb + db)))


def relative(value, truth):
    return float(np.linalg.norm(value - truth) / np.linalg.norm(truth))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--reference-orders", type=int, nargs="+", default=[8, 12])
    parser.add_argument("--duration-ps", type=float, default=200)
    parser.add_argument("--response-ps", type=float, default=10)
    parser.add_argument("--refit-every", type=int, default=1)
    parser.add_argument("--mode", choices=["reference", "compare"], default="reference")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 2 or args.order < 2 or min(args.reference_orders) < 2:
        parser.error("steps and quadrature orders must be at least two")
    duration, tau = args.duration_ps * 1e-12, args.response_ps * 1e-12
    if duration <= 0 or tau <= 0:
        parser.error("duration and response time must be positive")
    tick = time.perf_counter()
    nodes, weights = gaussian(4)
    particles = Species(
        "prescribed",
        nodes * SIGMA,
        np.zeros_like(nodes),
        CHARGE,
        1.67262192369e-27,
        weights * POPULATION,
    )
    cfg = CorrectionConfig(
        clouds_per_species=1,
        cloud_breathing=True,
        subcharge_count=args.order**2,
        cloud_breathing_longitudinal_order=args.order,
        cloud_breathing_response_time_s=tau,
        refit_every=args.refit_every,
    )
    breathing = CloudCorrection([particles], cfg)
    # A frozen 3D control uses identical initial nodes, charges, and geometry.
    # It is held inertial by unchanged cohort moments, never by history edits.
    frozen_particles = Species(
        "frozen",
        particles.position_m.copy(),
        particles.momentum_mc.copy(),
        particles.charge_c,
        particles.mass_kg,
        particles.population.copy(),
    )
    frozen = (
        CloudCorrection([frozen_particles], cfg) if args.mode == "compare" else None
    )
    rows = []
    for step in range(1, args.steps + 1):
        t = duration * step / args.steps
        particles.position_m = nodes * (SIGMA + RATE * t)
        beta = nodes * RATE / C
        particles.momentum_mc = beta / np.sqrt(1 - np.sum(beta**2, axis=1))[:, None]
        breathing.accept([particles], t)
        if frozen is not None:
            frozen.accept([frozen_particles], t)
        if step not in {args.steps // 4, args.steps // 2, args.steps}:
            continue
        field = total(breathing)
        pairs = [reference(order, t, tau) for order in args.reference_orders]
        refs = [p[0] for p in pairs]
        qe, qb = breathing.quasi_static(PROBES)
        correction = field - np.stack((qe, C * qb))
        row = {
            "time_ps": t * 1e12,
            "breathing_relative_error": relative(field, refs[-1]),
            "electric_relative_error": relative(field[0], refs[-1][0]),
            "magnetic_relative_error": relative(field[1], refs[-1][1]),
            "correction_relative_error": relative(
                correction, pairs[-1][0] - pairs[-1][1]
            ),
            "reference_changes": [relative(b, a) for a, b in zip(refs[:-1], refs[1:])],
            "max_expansion_speed_c": float(
                np.max(np.linalg.norm(breathing.offset_velocity_m_s, axis=-1)) / C
            ),
            "breathing_e_v_m": field[0].tolist(),
            "breathing_c_b_v_m": field[1].tolist(),
        }
        if frozen is not None:
            frozen_field = total(frozen)
            row.update(
                frozen_relative_error=relative(frozen_field, refs[-1]),
                frozen_vs_breathing=relative(frozen_field, field),
                frozen_e_v_m=frozen_field[0].tolist(),
            )
        rows.append(row)
    result = {
        "parameters": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "rows": rows,
        "elapsed_s": time.perf_counter() - tick,
        "actual_subcharges": args.order**3,
        "radiation_reaction": "off: prescribed sources",
        "reference": "independent SI LW, analytic filter response, 3D Gaussian quadrature",
        "scope": "source-field validation only; no energy or self-consistent trajectory claim",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
