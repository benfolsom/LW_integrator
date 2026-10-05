"""Deterministic kernel and conservation checks, with explicit failure status.

All cases are periodic synthetic controls in c=epsilon_0=m_species=1 units.
Radiation reaction is explicitly off; these are kernel tests, not LW runs.
Source q=N*q_species; observer qm=q_species; momentum weight=N*m_species.
"""

import argparse
import json
import traceback
from pathlib import Path

import numpy as np

from .common import Grid, make_backend, synthetic
from .cpu import CPU
from .provenance import metadata


def error_metrics(actual, reference, atol, rtol, period=None):
    a, r = np.asarray(actual, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    difference = a - r
    if period is not None:
        difference = (difference + period / 2) % period - period / 2
    absolute = np.abs(difference)
    finite = bool(np.all(np.isfinite(a)) and np.all(np.isfinite(r)))
    # Raw relative error is undefined at zero, reported only on nonzero entries.
    nonzero = np.abs(r) > 0
    relative = absolute[nonzero] / np.abs(r[nonzero])
    scaled = absolute / (atol + rtol * np.abs(r))
    return {
        "max_abs": float(absolute.max(initial=0)),
        "max_rel": float(relative.max(initial=0)),
        "max_tolerance_ratio": float(scaled.max(initial=0)),
        "atol": atol,
        "rtol": rtol,
        "finite": finite,
        "passed": bool(finite and np.all(scaled <= 1)),
    }


def run_backend(backend, count=8192):
    grid, cpu = backend.grid, CPU(backend.grid)
    f32 = backend.dtype == "float32"
    atol, rtol = (5e-5, 2e-4) if f32 else (2e-11, 5e-12)
    checks = []

    def compare(name, actual, reference, period=None, tolerances=None):
        aa, rr = tolerances or (atol, rtol)
        row = dict(
            test=name, **error_metrics(backend.host(actual), reference, aa, rr, period)
        )
        checks.append(row)

    def scalar(name, value, tolerance, **extra):
        checks.append(
            dict(
                test=name,
                value=float(value),
                tolerance=tolerance,
                passed=bool(np.isfinite(value) and abs(value) <= tolerance),
                **extra,
            )
        )

    x, u, q, qm, population = synthetic(count, grid)
    # Boundaries, outside-box coordinates, half-cell ties, and full periodic wraps.
    edge = np.array(
        [
            [0, 0, 0],
            [grid.length, grid.length, grid.length],
            [-grid.dx / 4, grid.dx / 2, grid.length + grid.dx / 4],
            [grid.length - 1e-8, 1e-8, grid.dx / 2],
        ]
    )
    x[: len(edge)] = edge
    rng = np.random.default_rng(702)
    field = rng.normal(0, 0.1, (grid.n,) * 3 + (3,))
    xb, qb, ub, qmb = [backend.array(v) for v in (x, q, u, qm)]
    fb = backend.array(field)
    zero = np.zeros_like(u)
    zb = backend.array(zero)
    for order in (1, 2):
        rho_ref = cpu.deposit(x, q, order)
        rho = backend.deposit(xb, qb, order)
        compare(f"deposit_{order}_uniform", rho, rho_ref)
        total = backend.host(rho).astype(np.float64).sum() * grid.dx**3
        normalized = (total - q.sum()) / np.abs(q).sum()
        scalar(
            f"charge_{order}_uniform",
            normalized,
            2e-6 if f32 else 1e-13,
            net_charge_error=float(total - q.sum()),
            expected_net_charge=float(q.sum()),
        )
        compare(
            f"gather_{order}_random",
            backend.gather(xb, fb, order),
            cpu.gather(x, field, order),
        )
        constant = np.broadcast_to([0.25, -0.5, 1.0], field.shape).copy()
        compare(
            f"gather_{order}_constant",
            backend.gather(xb, backend.array(constant), order),
            np.broadcast_to([0.25, -0.5, 1.0], u.shape),
        )
        # Same-cell mixed-magnitude loads expose lost increments in atomics.
        xc = np.full_like(x, grid.dx * 2.25)
        qc = np.ones(count) * 0.25
        third = count // 3
        qc[:third], qc[-third:] = 2**20, -(2**20)
        clustered = backend.deposit(backend.array(xc), backend.array(qc), order)
        compare(f"deposit_{order}_cancellation", clustered, cpu.deposit(xc, qc, order))
        cluster_total = backend.host(clustered).astype(np.float64).sum() * grid.dx**3
        scalar(
            f"charge_{order}_cancellation",
            (cluster_total - qc.sum()) / max(abs(qc.sum()), 1),
            5e-5 if f32 else 1e-12,
            net_charge_error=float(cluster_total - qc.sum()),
        )
        # Closed electrostatic momentum: matched deposit/gather, skew gradient.
        _, internal = backend.poisson(rho)
        force = (
            backend.host(backend.gather(xb, internal, order)).astype(np.float64)
            * q[:, None]
        )
        scale = max(np.linalg.norm(force, axis=1).sum(), 1)
        scalar(
            f"internal_force_{order}",
            np.linalg.norm(force.sum(axis=0)) / scale,
            5e-6 if f32 else 1e-12,
        )
        _, updated = backend.push(
            xb, ub, backend.gather(xb, internal, order), zb, qmb, 0.05
        )
        # Sum N*m*u, not an unweighted macro mean. Compare ORIGINAL input too.
        delta = (backend.host(updated).astype(np.float64) - u) * population[:, None]
        norm = max(np.linalg.norm(u * population[:, None], axis=1).sum(), 1)
        scalar(
            f"weighted_momentum_{order}",
            np.linalg.norm(delta.sum(axis=0)) / norm,
            5e-6 if f32 else 1e-12,
            delta_momentum=delta.sum(axis=0).tolist(),
        )

    electric = rng.normal(0, 0.03, u.shape)
    magnetic = rng.normal(0, 0.1, u.shape)
    eb, bb = backend.array(electric), backend.array(magnetic)
    xr, ur = cpu.push(x, u, electric, magnetic, qm, 0.05)
    xn, un = backend.push(xb, ub, eb, bb, qmb, 0.05)
    compare("boris_position", xn, xr, grid.length)
    compare("boris_momentum", un, ur)
    _, free = backend.push(xb, ub, zb, zb, qmb, 0.05)
    compare("zero_field_momentum", free, u)
    _, impulse = backend.push(xb, ub, eb, zb, qmb, 0.05)
    compare("electric_impulse", impulse, u + qm[:, None] * 0.05 * electric)
    _, rotated = backend.push(xb, ub, zb, bb, qmb, 0.05)
    energy_error = np.max(
        np.abs(
            np.sum(backend.host(rotated).astype(np.float64) ** 2, axis=1)
            - np.sum(u * u, axis=1)
        )
        / (1 + np.sum(u * u, axis=1))
    )
    scalar("magnetic_norm", energy_error, 2e-6 if f32 else 2e-13)

    # Independent manufactured Poisson solution and E sign/normalization.
    coord = np.arange(grid.n) * 2 * np.pi / grid.n
    phi_exact = (
        np.cos(coord)[:, None, None]
        + 0.3 * np.sin(2 * coord)[None, :, None]
        + 0.2 * np.cos(3 * coord)[None, None, :]
    )
    rho_exact = np.zeros_like(phi_exact)
    field_exact = np.empty(phi_exact.shape + (3,))
    for axis in range(3):
        rho_exact += (
            2 * phi_exact - np.roll(phi_exact, 1, axis) - np.roll(phi_exact, -1, axis)
        ) / grid.dx**2
        field_exact[..., axis] = (
            np.roll(phi_exact, 1, axis) - np.roll(phi_exact, -1, axis)
        ) / (2 * grid.dx)
    phi, ep = backend.poisson(backend.array(rho_exact + 0.37))
    compare("poisson_manufactured_phi", phi, phi_exact)
    compare("poisson_manufactured_field", ep, field_exact)
    rho_random = rng.normal(0, 0.05, phi_exact.shape)
    phi, ep = backend.poisson(backend.array(rho_random))
    pr, er = cpu.poisson(rho_random)
    compare("poisson_random_phi", phi, pr)
    compare("poisson_random_field", ep, er)
    scalar(
        "poisson_mean_phi",
        abs(backend.host(phi).astype(np.float64).mean()),
        2e-6 if f32 else 2e-13,
    )
    return {
        "backend": backend.name,
        "dtype": backend.dtype,
        "grid": grid.n,
        "dx": grid.dx,
        "count": count,
        "checks": checks,
        "passed": all(c["passed"] for c in checks),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", nargs="+", default=["cpu", "mlx", "taichi-metal"])
    parser.add_argument("--grids", nargs="+", type=int, default=[16, 64, 128])
    parser.add_argument("--count", type=int, default=8192)
    parser.add_argument("--dtype", choices=["float32", "float64"])
    parser.add_argument("--uncompensated", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.count < 12:
        parser.error("count must be >= 12")
    report = {
        "metadata": metadata(),
        "radiation_reaction": "off (synthetic kernel controls)",
        "results": [],
        "unavailable": [],
    }
    for name in args.backends:
        for n in args.grids:
            try:
                backend = make_backend(
                    name, Grid(n), args.dtype, compensated=not args.uncompensated
                )
                result = run_backend(backend, args.count)
                result["compensated"] = not args.uncompensated
                report["results"].append(result)
                failed = [c["test"] for c in result["checks"] if not c["passed"]]
                print(
                    f"{name} {n}^3: {'PASS' if result['passed'] else 'FAIL'} {failed}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                report["unavailable"].append(
                    {
                        "backend": name,
                        "grid": n,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                print(f"{name} {n}^3: ERROR {exc}", flush=True)
                break
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return (
        0
        if not report["unavailable"] and all(r["passed"] for r in report["results"])
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
