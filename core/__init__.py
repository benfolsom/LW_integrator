"""
Core module for the Lienard-Wiechert integrator system.

This module contains the core integration algorithms and utilities.
"""

# Expose the validated trajectory integrator
from . import trajectory_integrator

__all__ = ["trajectory_integrator"]
