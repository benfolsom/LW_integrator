"""Module entry point for ``python -m lw_integrator``."""

from __future__ import annotations

from .cli import main


if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(main())
