#!/usr/bin/env python3
"""Compatibility wrapper for the packaged live logcache plotter."""

from lw_integrator.logcache_plotter import (
    create_1d_curves_plot,
    create_combined_gains_plot,
    create_contour_plot,
    find_latest_log,
    live_monitor,
    main,
    parse_sweep_log,
)

__all__ = [
    "create_1d_curves_plot",
    "create_combined_gains_plot",
    "create_contour_plot",
    "find_latest_log",
    "live_monitor",
    "main",
    "parse_sweep_log",
]


if __name__ == "__main__":
    raise SystemExit(main())
