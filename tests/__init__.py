"""
Tests module for the Lienard-Wiechert integrator system.

This module contains unit tests, integration tests, and benchmarks.
"""

from core._version import __version__

# Test categories available
TEST_CATEGORIES = ["unit", "integration", "benchmarks", "performance", "physics"]

__all__ = ["TEST_CATEGORIES", "__version__"]
