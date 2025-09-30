"""Legacy integrator regression tests relocated to :mod:`tests.physics`."""

import pytest


pytest.skip(
    "Legacy-style integrator unit coverage now resides in"
    " tests/physics/test_integrator_units.py",
    allow_module_level=True,
)
