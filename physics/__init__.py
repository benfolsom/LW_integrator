"""
Physics module for the Lienard-Wiechert integrator system.

This module contains fundamental physics constants, simulation types,
and other physics-related utilities.
"""

# Make core physics modules available at package level
from . import constants
from . import simulation_types

__all__ = ["constants", "simulation_types"]
