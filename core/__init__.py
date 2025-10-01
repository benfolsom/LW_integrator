"""
Core module for the Lienard-Wiechert integrator system.

This module contains the core integration algorithms and utilities.
"""

# Expose the validated trajectory integrator and version metadata
from . import trajectory_integrator
from ._version import __version__, VERSION

__all__ = ["trajectory_integrator", "__version__", "VERSION"]
