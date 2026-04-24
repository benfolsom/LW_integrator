"""Public package entry points for the LW Integrator.

The command-line interface lives in :mod:`lw_integrator.cli` and can be invoked
via ``python -m lw_integrator`` or the ``lw-simulate`` console script.
Implementation APIs remain in the ``core`` package and in the concrete
``lw_integrator`` submodules.
"""

from __future__ import annotations

from core._version import VERSION, __version__
from .cli import main as cli_main

__all__ = [
    "cli_main",
    "__version__",
    "VERSION",
]
