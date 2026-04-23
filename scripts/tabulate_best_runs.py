#!/usr/bin/env python3
"""Compatibility wrapper for the packaged optimization-results CLI."""

from lw_integrator.optimization_results import main


if __name__ == "__main__":
    raise SystemExit(main())
