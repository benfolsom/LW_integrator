#!/usr/bin/env python3
"""Compatibility wrapper for the packaged optimization monitor entry point."""

from lw_integrator.optimization_monitor import main


if __name__ == "__main__":
    raise SystemExit(main())
