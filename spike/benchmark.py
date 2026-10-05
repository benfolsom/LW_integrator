"""Synchronized resident-array timings; host upload measured separately."""

import argparse
import csv
import gc
import json
import statistics
import time
from pathlib import Path

import numpy as np

from .common import Grid, make_backend, synthetic
from .provenance import metadata


def timed(backend, operation):
    backend.sync()
    start = time.perf_counter()
    result = operation()
    values = result if isinstance(result, tuple) else (result,)
    backend.sync(*values)
    return (time.perf_counter() - start) * 1000, result


def sample(backend, operation, repeats):
    first, result = timed(backend, operation)
    del result
    times = []
    for _ in range(repeats):
        elapsed, result = timed(backend, operation)
        times.append(elapsed)
        del result
    return {
        "first_ms": first,
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "samples_ms": times,
    }


def run_case(backend, count, repeats):
    grid = backend.grid
    x, u, q, qm, _ = synthetic(count, grid)
    expected_charge = float(q.sum())
    absolute_charge = float(np.abs(q).sum())
    start = time.perf_counter()
    xb, ub, qb, qmb = [backend.array(a) for a in (x, u, q, qm)]
    zero = backend.array(np.zeros_like(u))
    backend.sync(xb, ub, qb, qmb, zero)
    upload_ms = (time.perf_counter() - start) * 1000
    # Release large float64 host inputs before resident timing.
    del x, u, q, qm
    rows = []
    for order in (1, 2):
        rho = backend.deposit(xb, qb, order)
        phi, field = backend.poisson(rho)
        electric = backend.gather(xb, field, order)
        backend.sync(rho, phi, field, electric)
        # Validate full-load charge and finite states outside the timed region.
        density = backend.host(rho).astype(np.float64)
        net_charge_error = float(density.sum() * grid.dx**3 - expected_charge)
        charge_error = abs(net_charge_error) / absolute_charge
        tolerance = 5e-6 if backend.dtype == "float32" else 5e-13
        if charge_error > tolerance or not all(
            np.all(np.isfinite(backend.host(value)))
            for value in (rho, phi, field, electric)
        ):
            raise RuntimeError("Full-load charge or finite-state check failed")

        def pipeline(order=order):
            density = backend.deposit(xb, qb, order)
            _, mesh = backend.poisson(density)
            e = backend.gather(xb, mesh, order)
            return backend.push(xb, ub, e, zero, qmb, 0.05)

        operations = {
            "deposit": lambda order=order: backend.deposit(xb, qb, order),
            "gather": lambda field=field, order=order: backend.gather(xb, field, order),
            "push": lambda electric=electric: backend.push(
                xb, ub, electric, zero, qmb, 0.05
            ),
            "poisson": lambda rho=rho: backend.poisson(rho),
            "pipeline": pipeline,
        }
        for kernel, operation in operations.items():
            row = dict(
                backend=backend.name,
                dtype=backend.dtype,
                particles=count,
                grid=grid.n,
                order=order,
                kernel=kernel,
                status="measured",
                normalized_charge_error=charge_error,
                net_charge_error=net_charge_error,
                host_upload_ms=upload_ms,
                repeats=repeats,
                **sample(backend, operation, repeats),
            )
            rows.append(row)
            print(
                f"{backend.name} N={count} {grid.n}^3 order={order} {kernel}: "
                f"{row['median_ms']:.3f} ms",
                flush=True,
            )
        del rho, phi, field, electric, operations
    return rows


def write(report, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    columns = [
        "backend",
        "dtype",
        "particles",
        "grid",
        "order",
        "kernel",
        "status",
        "median_ms",
        "min_ms",
        "max_ms",
        "first_ms",
        "host_upload_ms",
        "repeats",
        "error",
    ]
    with output.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report["rows"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", nargs="+", default=["cpu", "mlx", "taichi-metal"])
    parser.add_argument(
        "--particles", nargs="+", type=int, default=[100_000, 1_000_000, 10_000_000]
    )
    parser.add_argument("--grids", nargs="+", type=int, default=[64, 128])
    parser.add_argument("--dtype", choices=["float32", "float64"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or min(args.particles) < 1:
        parser.error("repeats and particle counts must be positive")
    report = {
        "metadata": metadata(),
        "rows": [],
        "timing_contract": "Warm median, synchronized GPU completion; allocations, "
        "zeroing, neutralization, FFT, gradient, and dispatch included. "
        "Host input upload separate. JIT/compilation excluded from warm medians. "
        "Pipeline is one electrostatic step with radiation reaction off.",
    }
    failed = False
    for name in args.backends:
        creation_error = None
        for n in args.grids:
            backend = None
            # Failed extension imports can leave native registrations behind.
            # Never re-import an unavailable backend in the same process.
            if creation_error is None:
                try:
                    backend = make_backend(name, Grid(n), args.dtype)
                except Exception as exc:  # noqa: BLE001
                    creation_error = exc
            for count in args.particles:
                try:
                    if creation_error is not None:
                        raise creation_error
                    report["rows"].extend(run_case(backend, count, args.repeats))
                    gc.collect()
                except Exception as exc:  # noqa: BLE001
                    failed = True
                    print(f"{name} N={count} {n}^3 ERROR: {exc}", flush=True)
                    for order in (1, 2):
                        for kernel in (
                            "deposit",
                            "gather",
                            "push",
                            "poisson",
                            "pipeline",
                        ):
                            report["rows"].append(
                                {
                                    "backend": name,
                                    "dtype": args.dtype
                                    or (
                                        "float32"
                                        if name in ("mlx", "mlx-cuda", "taichi-metal")
                                        else "float64"
                                    ),
                                    "grid": n,
                                    "particles": count,
                                    "order": order,
                                    "kernel": kernel,
                                    "status": (
                                        "unavailable"
                                        if creation_error is not None
                                        else "failed"
                                    ),
                                    "error": str(exc),
                                }
                            )
                write(report, args.output)
            del backend
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
