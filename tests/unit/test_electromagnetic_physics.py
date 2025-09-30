"""Physics regression tests now live under :mod:`tests.physics`."""

import pytest


pytest.skip(
    "Physics-heavy Coulomb and EM regression checks moved to"
    " tests/physics/test_electromagnetic_physics.py",
    allow_module_level=True,
)
