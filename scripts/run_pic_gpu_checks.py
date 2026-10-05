#!/usr/bin/env python3
"""Native PIC conformance and synchronized timing; synthetic inputs only.

--bundle writes a self-contained Python runner with the current core/pic
sources embedded. Upload that one file to Colab; no editable install or Git
commit is needed. Dependencies must already exist in the chosen environment.
"""

import argparse
import base64
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile

EMBEDDED_PACKAGE = None


def bootstrap():
    if EMBEDDED_PACKAGE:
        root = Path(tempfile.mkdtemp(prefix="native-pic-package-"))
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(EMBEDDED_PACKAGE))) as archive:
            for member in archive.infolist():
                path = Path(member.filename)
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError("unsafe embedded package member")
            archive.extractall(root)
        sys.path.insert(0, str(root))
        return root
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    return root


def package_files(root):
    return [
        root / "core/__init__.py",
        root / "core/_version.py",
        *sorted((root / "core/pic").glob("*.py")),
        root / "scripts/check_pic_gpu_source.py",
    ]


def write_bundle(root, path):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for source in package_files(root):
            archive.writestr(str(source.relative_to(root)), source.read_bytes())
    source = Path(__file__).read_text()
    source = source.replace(
        "EMBEDDED_PACKAGE = None",
        "EMBEDDED_PACKAGE = " + repr(base64.b64encode(buffer.getvalue()).decode()),
        1,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    print(f"Synthetic runner bundle: {path} ({path.stat().st_size} bytes)")


def metadata(root, backend):
    versions = {}
    for name in (
        "numpy",
        "scipy",
        "numba",
        "mlx",
        "mlx-metal",
        "cupy",
        "cupy-cuda11x",
        "cupy-cuda12x",
        "cupy-cuda13x",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    result = dict(
        python=sys.version,
        platform=platform.platform(),
        machine=platform.machine(),
        versions=versions,
        source_sha256={
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in package_files(root)
        },
        backend=backend.name,
        precision=backend.dtype,
    )
    if backend.name.startswith("cupy"):
        runtime = backend.xp.cuda.runtime
        device = backend.xp.cuda.Device().id
        properties = runtime.getDeviceProperties(device)
        name = properties["name"]
        result["cuda"] = dict(
            device=name.decode() if isinstance(name, bytes) else name,
            device_id=device,
            runtime_version=runtime.runtimeGetVersion(),
            driver_version=runtime.driverGetVersion(),
            compute_capability=[properties["major"], properties["minor"]],
        )
        try:
            result["nvidia_smi"] = subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True, check=True, timeout=15
            ).stdout
        except (OSError, subprocess.SubprocessError) as exc:
            result["nvidia_smi_error"] = str(exc)
    if backend.name.startswith("mlx"):
        result["metal"] = backend.xp.metal.device_info()
    if hasattr(backend, "compile_command"):
        result["host_c_compile_command"] = backend.compile_command
        result["validation_scope"] = (
            "serial host C emulation; not GPU compilation, atomics, or FFT validation"
        )
    return result


def timing_matrix(backend, meshes, particles, orders, repeats):
    import numpy as np

    rows = []
    for mesh in meshes:
        for count in particles:
            for order in orders:
                rng = np.random.default_rng(20261005)
                shape = (mesh,) * 3
                x = backend.array(rng.uniform(4, mesh - 5, (count, 3)))
                new = x + backend.array(rng.uniform(-0.25, 0.25, (count, 3)))
                q = backend.array(np.full(count, 1e-10 / count))
                spacing = np.array([0.012 / mesh] * 3)
                u = backend.array(np.broadcast_to([0, 0, 0.7], (count, 3)))
                qm = backend.array(np.full(count, 9.578833156e7))
                dt = 1e-12
                rho = backend.deposit(x, q, shape, order)
                e, _ = backend.solve(rho, spacing)
                gathered = backend.gather(x, e, order)
                b = backend.xp.zeros_like(gathered)
                backend.synchronize(rho, e, gathered, b)

                def pipeline():
                    density = backend.deposit(x, q, shape, order)
                    field, _ = backend.solve(density, spacing)
                    electric = backend.gather(x, field, order)
                    pushed = backend.push(u, electric, b, qm, dt)
                    current = backend.current(x, new, q, shape, spacing, dt, order)
                    return density, field, electric, pushed, *current

                jobs = dict(
                    deposit=lambda: (backend.deposit(x, q, shape, order),),
                    solve=lambda: backend.solve(rho, spacing)[:1],
                    gather=lambda: (backend.gather(x, e, order),),
                    push=lambda: (backend.push(u, gathered, b, qm, dt),),
                    current=lambda: backend.current(
                        x, new, q, shape, spacing, dt, order
                    ),
                    pipeline=pipeline,
                )
                row = dict(
                    mesh=mesh,
                    particles=count,
                    shape_order=order,
                    timings={},
                    scope=(
                        "warm device seam, uploads excluded; validation included; "
                        "host Green setup cached; current included"
                    ),
                )
                for name, job in jobs.items():
                    backend.synchronize(*job())
                    samples = []
                    for _ in range(repeats):
                        backend.synchronize()
                        start = time.perf_counter()
                        values = job()
                        backend.synchronize(*values)
                        samples.append(time.perf_counter() - start)
                    row["timings"][name] = dict(
                        samples_s=samples, median_s=float(np.median(samples))
                    )
                # Full-load validation outside the timed region. No timings
                # are accepted merely because a launch returned successfully.
                values = pipeline()
                backend.synchronize(*values)
                host = [backend.to_host(v) for v in values]
                charge_error = abs(np.sum(host[0], dtype=float) / 1e-10 - 1)
                finite = all(np.all(np.isfinite(v)) for v in host)
                limit = 3e-6 if backend.dtype == "float32" else 5e-13
                end = backend.to_host(backend.deposit(new, q, shape, order))
                residual = (end - host[0]) / (np.prod(spacing) * dt)
                residual += sum(
                    np.diff(host[4 + a].astype(float), axis=a) / spacing[a]
                    for a in range(3)
                )
                scale = max(np.max(np.abs(host[0])), np.max(np.abs(end))) / (
                    np.prod(spacing) * dt
                )
                continuity = float(np.max(np.abs(residual)) / scale)
                continuity_limit = 8e-6 if backend.dtype == "float32" else 5e-11
                row["validation"] = dict(
                    relative_charge_error=float(charge_error),
                    finite=bool(finite),
                    relative_continuity_residual=continuity,
                    passed=bool(
                        finite
                        and charge_error <= limit
                        and continuity <= continuity_limit
                    ),
                )
                rows.append(row)
                print(backend.name, mesh, count, order, row["validation"], flush=True)
    return rows


def json_safe(value):
    """Retain failure records without emitting nonstandard NaN JSON."""
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main():
    root = bootstrap()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle", type=Path, help="Write a standalone uploadable script and exit"
    )
    parser.add_argument(
        "--backend", choices=("cpu", "mlx", "cupy", "host-c"), default="cpu"
    )
    parser.add_argument("--precision", choices=("float32", "float64"))
    parser.add_argument("--host-dialect", choices=("cuda", "metal"), default="cuda")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mesh", type=int, default=24, help="Proton conformance grid")
    parser.add_argument("--analytic-mesh", type=int)
    parser.add_argument(
        "--power", type=int, default=13, help="Sources per bunch: 2**power"
    )
    parser.add_argument("--events", type=int, default=3)
    parser.add_argument(
        "--stage1",
        action="store_true",
        help="128³ proton, 192³ analytic, 2**22 sources, 33 events",
    )
    parser.add_argument("--timings", action="store_true")
    parser.add_argument("--meshes", nargs="+", type=int, default=[64, 128])
    parser.add_argument("--particles", nargs="+", type=int, default=[100000, 1000000])
    parser.add_argument("--orders", nargs="+", type=int, choices=(1, 2), default=[1, 2])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.bundle:
        write_bundle(root, args.bundle)
        return 0
    if args.output is None:
        parser.error("--output is required when running checks")
    if min(args.meshes) < 12 or min(args.particles) < 1 or args.repeats < 1:
        parser.error("timing meshes must be >=12, particles and repeats positive")
    # Keep interpreter/JIT/kernel caches outside synchronized source trees.
    os.environ.setdefault(
        "NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "native-pic-numba")
    )
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    if args.stage1:
        args.mesh, args.analytic_mesh, args.power, args.events = 128, 192, 22, 33
    report = dict(request=vars(args).copy(), passed=False, synthetic_only=True)
    report["request"]["output"] = str(args.output)
    report["request"]["bundle"] = None
    try:
        from core.pic import select_backend
        from core.pic.conformance import run_conformance

        if args.backend == "host-c":
            from scripts.check_pic_gpu_source import HostCBackend

            backend = HostCBackend(
                args.precision or "float64",
                tempfile.mkdtemp(prefix="pic-host-c-"),
                args.host_dialect,
            )
        else:
            backend = select_backend(args.backend, args.precision)
        report["metadata"] = metadata(root, backend)
        report["conformance"] = run_conformance(
            backend, args.mesh, args.power, args.events, args.analytic_mesh
        )
        report["passed"] = report["conformance"]["passed"]
        if args.timings and report["passed"]:
            report["timings"] = timing_matrix(
                backend, args.meshes, args.particles, args.orders, args.repeats
            )
            report["passed"] = all(r["validation"]["passed"] for r in report["timings"])
        elif args.timings:
            report["timings_status"] = "not run because conformance failed"
    except Exception as exc:
        report["error"] = dict(
            type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n"
    )
    print(f"passed={report['passed']}: {args.output}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
