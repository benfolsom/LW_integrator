"""Types for the existing dynamically initialized, frozen staggered mesh.

This stub adds no runtime changes to the default boundary backend.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from core.pic.backend import PICBackend

def cpu_backend(backend: PICBackend | None = ...) -> PICBackend: ...
@dataclass(frozen=True)
class AxisymmetricGrid:
    dr: float
    dz: float
    nr: int
    nz: int
    z0: float
    @property
    def r_node(self) -> np.ndarray: ...
    @property
    def r_half(self) -> np.ndarray: ...
    @property
    def z_node(self) -> np.ndarray: ...
    @property
    def z_half(self) -> np.ndarray: ...
    def dual_bounds(
        self, component: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...
    def require_aligned(self, value: Any, axis: str) -> None: ...

class Fields:
    er: np.ndarray
    ez: np.ndarray
    bt: np.ndarray
    def __init__(
        self, g: AxisymmetricGrid, backend: PICBackend | None = ...
    ) -> None: ...

def volumes(
    g: AxisymmetricGrid, control: tuple[float, float, float] | None = ...
) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
