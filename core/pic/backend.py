"""Array namespace plus explicit PIC kernels; CPU is the reference backend."""

from __future__ import annotations

import platform
from typing import Any, Protocol, cast

import numpy as np
from scipy import fft

from . import kernels


class PICBackend(Protocol):
    """Backends own arrays, kernels, and FFT plans; model code owns physics.

    An accelerator must implement the entire seam, including bounded
    host diagnostics. Changing only ``xp`` does not create a GPU backend.
    CPU boundaries validate inputs in float64 before invoking these kernels.
    """

    xp: Any
    name: str
    dtype: str

    def array(self, value: Any, dtype: Any = None) -> Any:
        """Explicit host-to-device transfer (float64 on CPU)."""
        ...

    def synchronize(self, *values: Any) -> None:
        """Complete queued operations, including lazy device arrays."""
        ...

    def deposit(
        self, coordinates: Any, charge: Any, shape: tuple[int, ...], order: int
    ) -> Any:
        """Scatter source charges in C to cell-centred charge arrays."""
        ...

    def current(
        self,
        old: Any,
        new: Any,
        charge: Any,
        shape: tuple[int, ...],
        spacing: Any,
        dt: float,
        order: int,
    ) -> Any:
        """Conservative flux on three face arrays, in C/(m**2 s)."""
        ...

    def gather(self, coordinates: Any, field: Any, order: int) -> Any:
        """Gather a collocated vector field with the deposition shape."""
        ...

    def gather_open(
        self, coordinates: Any, field: Any, charge: Any, spacing: Any, order: int
    ) -> Any:
        """Same-shape gather, including free-space virtual nodes off the mesh."""
        ...

    def push(
        self, u: Any, electric: Any, magnetic: Any, charge_mass: Any, dt: float
    ) -> Any:
        """Return new p/(mc), using physical q/m and a lab timestep."""
        ...

    def solve(
        self, charge: Any, spacing: Any, potential: bool = False
    ) -> tuple[Any, Any]:
        """Return rest-frame E in V/m and optional potential in V."""
        ...

    def nodes(
        self, nodes: Any, source_indices: Any, source_charge: Any, spacing: Any
    ) -> Any:
        """Evaluate free-space E at virtual integer grid nodes."""
        ...

    def to_host(self, array: Any) -> Any:
        """Explicit transfer for geometry checks and serialized diagnostics."""
        ...


class NumpyBackend:
    """NumPy arrays, serial Numba kernels, and single-worker SciPy FFTs.

    Cache one mesh's Green spectra, bounding memory even for many groups.
    Integrated Green functions give second-order spatial accuracy, including
    TSC; a higher shape order alone does not make the Poisson solve third order.
    """

    xp: Any = np
    name: str = "cpu_numpy_numba_float64"
    dtype: str = "float64"

    def __init__(self) -> None:
        self._key: tuple | None = None
        self._spectra: dict[int | None, np.ndarray] = {}

    deposit = staticmethod(kernels.deposit_kernel)
    current = staticmethod(kernels.current_kernel)
    gather = staticmethod(kernels.gather_kernel)
    push = staticmethod(kernels.boris_kernel)
    nodes = staticmethod(kernels.node_field_kernel)

    def array(self, value: Any, dtype: Any = None) -> Any:
        return np.ascontiguousarray(value, dtype=dtype or self.dtype)

    def synchronize(self, *values: Any) -> None:
        pass

    def to_host(self, array: Any) -> Any:
        return np.asarray(array)

    def gather_open(
        self, coordinates: Any, field: Any, charge: Any, spacing: Any, order: int
    ) -> Any:
        """Keep irregular stencil construction and scatter inside the backend.

        Virtual nodes use the same source-cell integral as the FFT. This
        reference fallback is O(virtual nodes * occupied source cells).
        An accelerator or a future fast node evaluator can replace it here.
        """
        base = np.floor(coordinates) if order == 1 else np.floor(coordinates + 0.5) - 1
        inside = np.all((base >= 0) & (base + order < charge.shape), axis=1)
        result = np.zeros((len(coordinates), 3))
        result[inside] = self.gather(coordinates[inside], field, order)
        if not np.all(inside):
            indices = np.argwhere(charge != 0)
            weights = charge[tuple(indices.T)]
            nodes, factors, owners = [], [], []
            for owner in np.flatnonzero(~inside):
                shapes = [
                    kernels.stencil(coordinates[owner, a], order) for a in range(3)
                ]
                for a in range(order + 1):
                    for b in range(order + 1):
                        for c in range(order + 1):
                            nodes.append(
                                [shapes[0][0] + a, shapes[1][0] + b, shapes[2][0] + c]
                            )
                            factors.append(
                                shapes[0][1][a] * shapes[1][1][b] * shapes[2][1][c]
                            )
                            owners.append(owner)
            nodes, inverse = np.unique(nodes, axis=0, return_inverse=True)
            fields = self.nodes(nodes, indices, weights, spacing)
            np.add.at(
                result,
                np.asarray(owners),
                fields[inverse] * np.asarray(factors)[:, None],
            )
        return result

    def _spectrum(
        self, shape: tuple[int, ...], spacing: Any, component: int | None
    ) -> np.ndarray:
        key = (tuple(shape), tuple(spacing))
        if key != self._key:
            self._spectra.clear()
            self._key = key
        if component not in self._spectra:
            padded = green_mesh(shape, spacing, component)
            self._spectra[component] = fft.rfftn(padded, workers=1)
        return self._spectra[component]

    def solve(
        self, charge: Any, spacing: Any, potential: bool = False
    ) -> tuple[Any, Any]:
        """Linear convolution of cell charges with free-space Green functions."""
        shape = charge.shape
        extended = tuple(2 * n for n in shape)
        crop = tuple(slice(0, n) for n in shape)
        spectrum = fft.rfftn(charge, s=extended, workers=1)
        field = np.empty((*shape, 3), dtype=np.float64)
        for axis in range(3):
            green = self._spectrum(shape, spacing, axis)
            field[..., axis] = fft.irfftn(spectrum * green, s=extended, workers=1)[crop]
        phi = None
        if potential:
            green = self._spectrum(shape, spacing, None)
            phi = fft.irfftn(spectrum * green, s=extended, workers=1)[crop].copy()
        return field, phi


def green_mesh(
    shape: tuple[int, ...], spacing: Any, component: int | None
) -> np.ndarray:
    """Host float64 geometry setup; no charges or particle operations."""
    axes = [np.arange(n) * h for n, h in zip(shape, spacing)]
    positive = kernels.integrated_green(
        (
            axes[0][:, None, None],
            axes[1][None, :, None],
            axes[2][None, None, :],
        ),
        spacing,
        component,
    )
    if component is not None:
        zero: list[slice | int] = [slice(None)] * 3
        zero[component] = 0
        positive[tuple(zero)] = 0.0
    # Reflect the octant explicitly: odd force and even potential to
    # roundoff. Displacement -n is never used by the physical crop.
    indices = [
        np.minimum(np.abs(np.r_[np.arange(n), np.arange(-n, 0)]), n - 1) for n in shape
    ]
    padded = positive[np.ix_(*indices)]
    if component is not None:
        signs = np.r_[np.ones(shape[component]), -np.ones(shape[component])]
        reshape = [1, 1, 1]
        reshape[component] = len(signs)
        padded *= signs.reshape(reshape)
    return cast(np.ndarray, padded)


def require_host(selection: str) -> None:
    """Reject unsupported hosts before importing any optional GPU framework."""
    system, machine = platform.system(), platform.machine().lower()
    if selection == "mlx":
        if system != "Darwin" or machine not in ("arm64", "aarch64"):
            raise RuntimeError(
                "MLX Metal requires Apple-silicon macOS; MLX was not imported"
            )
    elif selection == "cupy":
        if system not in ("Linux", "Windows") or machine not in (
            "x86_64",
            "amd64",
            "aarch64",
            "arm64",
        ):
            raise RuntimeError(
                "CuPy CUDA requires a supported Linux/Windows host; "
                "CuPy was not imported"
            )
    else:
        raise ValueError(f"unknown PIC GPU backend: {selection}")


def select_backend(selection: str = "auto", precision: str | None = None) -> PICBackend:
    """Explicit GPU opt-in. Auto remains the authoritative CPU reference."""
    if selection not in ("auto", "cpu", "mlx", "cupy"):
        raise ValueError("PIC backend must be auto, cpu, mlx, or cupy")
    if precision not in (None, "float32", "float64"):
        raise ValueError("PIC precision must be float32 or float64")
    if selection in ("auto", "cpu"):
        if precision == "float32":
            raise ValueError("CPU reference requires float64")
        return NumpyBackend()
    if selection == "mlx" and precision == "float64":
        raise ValueError("MLX Metal requires float32 precision")
    require_host(selection)
    try:
        if selection == "mlx":
            from .mlx_backend import MLXBackend

            return MLXBackend()
        from .cupy_backend import CuPyBackend

        return CuPyBackend(precision or "float64")
    except (ImportError, OSError, RuntimeError) as exc:
        raise RuntimeError(
            f"PIC {selection} unavailable: {exc}. No CPU fallback was selected."
        ) from exc


def cuda_build_hint() -> str | None:
    """Find the host toolkit major without importing CuPy or installing it.

    nvcc reports the installed toolkit; nvidia-smi reports driver capability,
    which can be newer and must not select a wheel. Source/Conda builds have
    their own runtime resolution. Unknown toolkits are checked by CuPy itself.
    """
    import os
    import re
    import subprocess
    from pathlib import Path

    candidates = []
    if os.environ.get("CUDA_PATH"):
        candidates.append(str(Path(os.environ["CUDA_PATH"]) / "bin" / "nvcc"))
    candidates.extend(["nvcc", "/usr/local/cuda/bin/nvcc"])
    for executable in candidates:
        try:
            result = subprocess.run(
                [executable, "--version"],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        match = re.search(r"release\s+(\d+)\.", result.stdout)
        if match:
            return f"cupy-cuda{match.group(1)}x"
    return None


def check_cupy_build() -> str | None:
    """Reject conflicting or toolkit-mismatched wheel installations."""
    from importlib.metadata import PackageNotFoundError, version

    wheels = []
    for name in ("cupy-cuda11x", "cupy-cuda12x", "cupy-cuda13x"):
        try:
            version(name)
            wheels.append(name)
        except PackageNotFoundError:
            pass
    expected = cuda_build_hint()
    if len(wheels) > 1:
        raise RuntimeError(f"multiple CuPy CUDA builds installed: {', '.join(wheels)}")
    if expected and wheels and wheels[0] != expected:
        raise RuntimeError(
            f"host CUDA toolkit needs {expected}, but {wheels[0]} is installed; "
            "select the matching existing environment/build "
            "(Colab CUDA 13: cupy-cuda13x)"
        )
    return expected
