"""Shared backend contract for exact retarded charge and dipole providers."""

from __future__ import annotations

EXACT_RETARDED_BACKENDS = (
    "python",
    "numba_roots_exact_serial",
    "numba_full_strict_serial",
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
    return selected


__all__ = [
    "EXACT_RETARDED_BACKENDS",
    "ExactRetardedBackendUnavailableError",
    "require_exact_retarded_backend",
    "validate_exact_retarded_backend",
]
