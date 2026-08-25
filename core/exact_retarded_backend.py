"""Shared backend contract for exact retarded charge and dipole providers."""

from __future__ import annotations

EXACT_RETARDED_BACKENDS = (
    "python",
    "numba_roots_exact_serial",
    "numba_full_strict_serial",
    "metal_certified_full_strict",
)


class ExactRetardedBackendUnavailableError(RuntimeError):
    """Raised when an explicitly selected exact backend cannot run."""


def validate_exact_retarded_backend(backend: str) -> str:
    """Return one canonical backend name without importing Numba."""

    selected = str(backend).strip().lower()
    if selected not in EXACT_RETARDED_BACKENDS:
        choices = ", ".join(EXACT_RETARDED_BACKENDS)
        raise ValueError(f"exact retarded backend must be one of: {choices}")
    return selected


def require_exact_retarded_backend(backend: str) -> str:
    """Validate a backend and fail clearly when explicit Numba is unavailable."""

    selected = validate_exact_retarded_backend(backend)
    if selected == "python":
        return selected

    from .exact_retarded_numba import NUMBA_AVAILABLE

    if not NUMBA_AVAILABLE:
        raise ExactRetardedBackendUnavailableError(
            f"exact retarded backend {selected!r} was explicitly selected, but "
            "Numba is not available; install the configured Numba runtime or select "
            "backend 'python'"
        )
    if selected == "metal_certified_full_strict":
        from .compute_backends import (
            ComputeBackendUnavailableError,
            resolve_knot_scan_backend,
        )

        try:
            resolve_knot_scan_backend("metal")
        except ComputeBackendUnavailableError as exc:
            raise ExactRetardedBackendUnavailableError(
                "exact retarded backend 'metal_certified_full_strict' was "
                "explicitly selected, but the certified Metal proposal backend "
                f"is unavailable: {exc}"
            ) from exc
    return selected


__all__ = [
    "EXACT_RETARDED_BACKENDS",
    "ExactRetardedBackendUnavailableError",
    "require_exact_retarded_backend",
    "validate_exact_retarded_backend",
]
