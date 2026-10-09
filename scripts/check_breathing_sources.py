"""Prescribed expanding Gaussian: analytic worldlines and frozen comparison.

Single process; no coupled simulation, force feedback, or energy-gain claim.
Radiation reaction is off because all trajectories are prescribed controls.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.constants import C_MMNS
from core.exact_source_cloud import exact_cloud_history
from core.exact_source_reduction import reduce_exact_initial_state
from core.integration_runner import _build_inertial_coasting_history
from core.retarded_fields import (
    ObserverEvent,
    evaluate_retarded_charge_field_native,
    lienard_wiechert_charge_field_native,
)
from tests.unit.test_exact_same_bunch_fields import _merge
from tests.unit.test_exact_source_cloud import cloud_config
from tests.unit.test_inertial_prehistory import _species_state

SIGMA_MM = 0.2
SLOPE_MM_NS = 2.0
RESPONSE_NS = 0.01


def analytic_width(t):
    """Independent solution of the filter for target sigma0 + slope*t."""
    if t <= 0:
        return SIGMA_MM, 0.0, 0.0
    u = t / RESPONSE_NS
    decay = np.exp(-u)
    return (
        SIGMA_MM + SLOPE_MM_NS * (t - 2 * RESPONSE_NS + (t + 2 * RESPONSE_NS) * decay),
        SLOPE_MM_NS * (1 - (1 + u) * decay),
        SLOPE_MM_NS * t / RESPONSE_NS**2 * decay,
    )


def gaussian_reference(event, order, centre_beta, charge, *, moving=True):
    """Solve each analytic worldline's light cone, then sum its LW field.

    This bypasses cloud preparation, the width filter implementation, stored
    trajectory interpolation, and the integrator's retarded root solver.
    """
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    position = np.asarray(event.position_mm)
    electric, magnetic = np.zeros(3), np.zeros(3)
    for i, j in np.ndindex(order, order):
        node = np.sqrt(2) * np.array([nodes[i], nodes[j], 0.0])

        def sample(t, node=node):
            width, rate, acceleration = (
                analytic_width(t) if moving else (SIGMA_MM, 0, 0)
            )
            return (
                centre_beta * C_MMNS * t + width * node,
                centre_beta + rate * node / C_MMNS,
                acceleration * node / C_MMNS**2,
            )

        def residual(t):
            return C_MMNS * (event.time_ns - t) - np.linalg.norm(
                position - sample(t)[0]
            )

        retarded = brentq(residual, event.time_ns - 2, event.time_ns, xtol=1e-14)
        source_position, beta, prime = sample(retarded)
        if np.dot(beta, beta) >= 1:
            raise ValueError("reference quadrature child is superluminal")
        e, b = lienard_wiechert_charge_field_native(
            charge_native=charge * weights[i] * weights[j] / np.pi,
            separation_vector_mm=position - source_position,
            source_beta=beta,
            source_beta_prime_per_mm=prime,
        )
        electric += e
        magnetic += b
    return np.concatenate((electric, magnetic))


def prescribed_history(reduced, knots, end_ns=0.15):
    history = _build_inertial_coasting_history(reduced, 1.0, knot_count=4)
    beta = np.array([reduced["b" + a][0] for a in "xyz"])
    for t in np.linspace(0, end_ns, knots)[1:]:
        state = copy.deepcopy(history[-1])
        state["t"][:] = t
        for i, a in enumerate("xyz"):
            state[a][:] = beta[i] * C_MMNS * t
        history.append(state)
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", nargs="+", type=int, default=[16, 64, 100])
    parser.add_argument("--knots", nargs="+", type=int, default=[65, 129])
    parser.add_argument("--reference-orders", nargs=2, type=int, default=[20, 24])
    parser.add_argument("--events", type=int, default=9)
    parser.add_argument("--compare-frozen", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.knots) < 3 or args.events < 1 or min(args.reference_orders) < 2:
        parser.error("knots >= 3, events >= 1, and reference orders >= 2 required")
    started = time.monotonic()
    config = replace(
        cloud_config(4, 0.02),
        breathing_enabled=True,
        breathing_response_time_ns=RESPONSE_NS,
    )
    # The four represented members have exactly linear transverse RMS.
    represented = _merge(
        *[
            _species_state(
                "electron",
                position_mm=(x * SIGMA_MM, y * SIGMA_MM, 0),
                beta=(x * SLOPE_MM_NS / C_MMNS, y * SLOPE_MM_NS / C_MMNS, 0.4),
            )
            for x, y in ((-1, -1), (-1, 1), (1, -1), (1, 1))
        ]
    )
    reduced, mapping = reduce_exact_initial_state(represented, 1, config)
    beta = np.array([reduced["b" + a][0] for a in "xyz"])
    charge = reduced["q_source"][0]
    events = [
        ObserverEvent(float(t), (2.0, 0.7, beta[2] * C_MMNS * t + 0.3))
        for t in np.linspace(0.02, 0.12, args.events)
    ]
    references = [
        np.array([gaussian_reference(e, order, beta, charge) for e in events])
        for order in args.reference_orders
    ]
    reference = references[-1]
    scale = np.linalg.norm(reference)
    rows = []
    for knots in args.knots:
        history = prescribed_history(reduced, knots)
        for count in args.counts:
            moving_config = replace(config, subcharge_count=count)
            cloud = exact_cloud_history(history, moving_config)
            fields = [evaluate_retarded_charge_field_native(cloud, e) for e in events]
            if not all(f.valid_sources.all() for f in fields):
                raise RuntimeError("invalid retarded source in validation")
            values = np.array(
                [
                    np.concatenate((f.electric_field_native, f.magnetic_field_native))
                    for f in fields
                ]
            )
            # Same-node analytic reference separates stored-history error from
            # Gaussian quadrature error. Square counts have Hermite rules.
            order = int(np.sqrt(count))
            if order**2 != count:
                parser.error("validation counts must be square Hermite rules")
            same_nodes = np.array(
                [gaussian_reference(e, order, beta, charge) for e in events]
            )
            row = {
                "count": count,
                "knots": knots,
                "relative_gaussian_field_error": float(
                    np.linalg.norm(values - reference) / scale
                ),
                "relative_same_node_worldline_error": float(
                    np.linalg.norm(values - same_nodes) / np.linalg.norm(same_nodes)
                ),
            }
            if args.compare_frozen:
                frozen = exact_cloud_history(
                    history, replace(moving_config, breathing_enabled=False)
                )
                fields = [
                    evaluate_retarded_charge_field_native(frozen, e) for e in events
                ]
                frozen_values = np.array(
                    [
                        np.concatenate(
                            (f.electric_field_native, f.magnetic_field_native)
                        )
                        for f in fields
                    ]
                )
                row.update(
                    relative_frozen_error=float(
                        np.linalg.norm(frozen_values - reference) / scale
                    ),
                    relative_breathing_vs_frozen=float(
                        np.linalg.norm(values - frozen_values) / scale
                    ),
                )
            rows.append(row)
    report = {
        "model": "prescribed isotropic expanding Gaussian with analytic C2 onset",
        "radiation_reaction": "off: prescribed trajectories, diagnostic control",
        "covariance_closure": mapping["breathing_covariance_model"],
        "initial_width_mm": SIGMA_MM,
        "response_time_ns": RESPONSE_NS,
        "final_width_mm": analytic_width(0.15)[0],
        "charge_native": float(charge),
        "reference_orders": args.reference_orders,
        "relative_reference_refinement": float(
            np.linalg.norm(references[0] - reference) / scale
        ),
        "rows": rows,
        "elapsed_seconds": time.monotonic() - started,
    }
    payload = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
