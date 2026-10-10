"""Serial prescribed expanding-Gaussian and rigid-kick controls; no trajectory/energy claim.

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
from dataclasses import replace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from core.pic import CloudCorrection, CorrectionConfig, Species
from core.pic.kernels import COULOMB, C

SIGMA = np.array([0.4, 0.5, 0.6]) * 1e-3
RIGID_SIGMA = SIGMA * [1, 1, 0.1]
RATE = np.array([0.6, 0.8, 1.0]) * 1e6
CHARGE = 1.602176634e-19
POPULATION = 1e7
PROBES = np.array([[5, 1, 2], [-3, 5, 1], [2, -1, 6]]) * 1e-3
RIGID_PROBES = np.array([[3, 1, 0], [-3, 2, -1], [2, -1, -2]]) * 1e-3


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


def rigid_state(t, tau):
    """Analytic rigid lab translation with a smooth transverse kick, gamma~10.

    beta_z stays fixed; the transverse acceleration rises continuously from
    zero. The initial lab shape is unchanged at every prescribed event.
    """
    t = np.asarray(t)
    positive = np.maximum(t, 0)
    decay = np.exp(-positive / tau)
    rate = 1e8
    position = np.zeros((*t.shape, 3))
    beta = np.zeros_like(position)
    beta_dot = np.zeros_like(position)
    position[..., 0] = (
        C * rate * (positive**2 / 2 - tau * positive + tau**2 * (1 - decay))
    )
    beta[..., 0] = rate * (positive - tau * (1 - decay))
    beta_dot[..., 0] = rate * (1 - decay)
    beta[..., 2] = np.sqrt(0.99)
    position[..., 2] = C * beta[..., 2] * t
    return position, beta, beta_dot


def reference(order, t, tau, case="expanding", probes=PROBES):
    nodes, fractions = gaussian(order)
    # All roots bracketed, including the inertial t<0 continuation.
    lower = np.full(
        (len(probes), len(nodes)), -20e-9 if case == "rigid-kick" else -1 / C
    )
    upper = np.full_like(lower, t)
    for _ in range(64):
        middle = (lower + upper) / 2
        widths = response(middle, tau)[0]
        if case == "rigid-kick":
            r = probes[:, None] - rigid_state(middle, tau)[0] - nodes * RIGID_SIGMA
        else:
            r = probes[:, None] - nodes * widths
        residual = C * (t - middle) - np.linalg.norm(r, axis=-1)
        lower = np.where(residual > 0, middle, lower)
        upper = np.where(residual > 0, upper, middle)
    retarded = (lower + upper) / 2
    widths, speed, acceleration = response(retarded, tau)
    r = probes[:, None] - nodes * widths
    distance = np.linalg.norm(r, axis=-1)
    n = r / distance[..., None]
    beta = nodes * speed / C
    beta_dot = nodes * acceleration / C
    if case == "rigid-kick":
        centroid, beta, beta_dot = rigid_state(retarded, tau)
        r = probes[:, None] - centroid - nodes * RIGID_SIGMA
        distance = np.linalg.norm(r, axis=-1)
        n = r / distance[..., None]
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
    centroid_beta = np.zeros(3)
    r = probes[:, None] - nodes * width
    if case == "rigid-kick":
        centroid, centroid_beta, _ = rigid_state(t, tau)
        r = probes[:, None] - centroid - nodes * RIGID_SIGMA
    # The baseline uses the centroid velocity, leaving internal currents
    # in the additive correction. This is boosted Coulomb for rigid motion.
    deficit = 1 - centroid_beta @ centroid_beta
    d2 = deficit * np.sum(r**2, axis=-1) + (r @ centroid_beta) ** 2
    terms = (
        COULOMB
        * CHARGE
        * POPULATION
        * fractions[None, :, None]
        * deficit
        * r
        / d2[..., None] ** 1.5
    )
    qe = terms.sum(1)
    quasi = np.stack((qe, np.cross(centroid_beta, qe)))
    return field, quasi


def total(corr, probes=PROBES):
    qe, qb = corr.quasi_static(probes)
    de, db = corr.direct(probes)
    return np.stack((qe + de, C * (qb + db)))


def relative(value, truth):
    return float(np.linalg.norm(value - truth) / np.linalg.norm(truth))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", choices=["expanding", "rigid-kick"], default="expanding"
    )
    parser.add_argument("--steps", type=int, default=256)
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
    if args.case == "rigid-kick":
        beta = rigid_state(0, tau)[1]
        particles.momentum_mc[:] = beta / np.sqrt(1 - beta @ beta)
        particles.position_m = nodes * RIGID_SIGMA
    cfg = CorrectionConfig(
        clouds_per_species=1,
        cloud_breathing=True,
        cloud_width_m=1e-6 if args.case == "rigid-kick" else 0.0002,
        subcharge_count=args.order**2,
        cloud_breathing_longitudinal_order=args.order,
        cloud_breathing_response_time_s=tau,
        refit_every=args.refit_every,
        prehistory_duration_s=20e-9 if args.case == "rigid-kick" else None,
    )
    breathing = CloudCorrection([particles], cfg)
    # A frozen 3D control uses identical initial nodes, charges, and geometry.
    # Its shape is held by unchanged moments; the rigid case shares the kick.
    frozen_particles = Species(
        "frozen",
        particles.position_m.copy(),
        particles.momentum_mc.copy(),
        particles.charge_c,
        particles.mass_kg,
        particles.population.copy(),
    )
    frozen = (
        CloudCorrection(
            [frozen_particles], replace(cfg, cloud_breathing_response_time_s=1)
        )
        if args.mode == "compare"
        else None
    )
    initial_map = breathing.breathing_filters[2].copy()
    rows = []
    for step in range(1, args.steps + 1):
        t = duration * step / args.steps
        particles.position_m = nodes * (SIGMA + RATE * t)
        beta = nodes * RATE / C
        particles.momentum_mc = beta / np.sqrt(1 - np.sum(beta**2, axis=1))[:, None]
        probes = PROBES
        if args.case == "rigid-kick":
            centroid, beta, _ = rigid_state(t, tau)
            particles.position_m = centroid + nodes * RIGID_SIGMA
            particles.momentum_mc[:] = beta / np.sqrt(1 - beta @ beta)
            frozen_particles.position_m = particles.position_m.copy()
            frozen_particles.momentum_mc = particles.momentum_mc.copy()
            probes = RIGID_PROBES + centroid
        breathing.accept([particles], t)
        if frozen is not None:
            frozen.accept([frozen_particles], t)
        if step not in {args.steps // 4, args.steps // 2, args.steps}:
            continue
        field = total(breathing, probes)
        pairs = [
            reference(order, t, tau, args.case, probes)
            for order in args.reference_orders
        ]
        refs = [p[0] for p in pairs]
        qe, qb = breathing.quasi_static(probes)
        correction = field - np.stack((qe, C * qb))
        matched = reference(args.order, t, tau, args.case, probes)
        row = {
            "time_ps": t * 1e12,
            "breathing_relative_error": relative(field, refs[-1]),
            "matched_quadrature_field_error": relative(field, matched[0]),
            "matched_quadrature_correction_error": relative(
                correction, matched[0] - matched[1]
            ),
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
        if args.case == "rigid-kick":
            row["rigid_map_change_m"] = float(
                np.max(np.abs(breathing.breathing_filters[2] - initial_map))
            )
            if (
                row["rigid_map_change_m"] > 1e-15
                or row["max_expansion_speed_c"] > 1e-12
            ):
                raise AssertionError("rigid kick changed the cloud shape or current")
        if frozen is not None:
            frozen_field = total(frozen, probes)
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
        "reference": "independent SI LW, analytic source motion, 3D Gaussian quadrature",
        "scope": "source-field validation only; no energy or self-consistent trajectory claim",
        "error_interpretation": "includes sample-and-hold timestep lag and quadrature error; refine both independently",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
