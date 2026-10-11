"""Test configuration helpers shared across the suite."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import tkinter  # noqa: F401
except ImportError:
    # These modules import tkinter at module level; skip them instead of
    # failing collection on Pythons built without Tk.
    collect_ignore = [
        "test_energy_aware_timestep.py",
        "test_gui.py",
        "test_optimization_plugin.py",
        "unit/test_nonlinear_pair_gui.py",
    ]
