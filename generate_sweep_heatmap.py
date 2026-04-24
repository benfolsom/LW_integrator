#!/usr/bin/env python3
"""Compatibility wrapper for the packaged sweep heatmap tool."""

from lw_integrator.sweep_heatmap import (
    build_grey_zero_cmap,
    create_smooth_heatmap,
    detect_swept_parameters,
    extract_data,
    generate_heatmap,
    load_sweep_results,
    main,
)

__all__ = [
    "build_grey_zero_cmap",
    "create_smooth_heatmap",
    "detect_swept_parameters",
    "extract_data",
    "generate_heatmap",
    "load_sweep_results",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
