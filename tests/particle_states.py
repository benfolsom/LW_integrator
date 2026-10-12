"""Particle-state builders shared only by tests with identical conventions."""

from __future__ import annotations

import numpy as np

from core.constants import C_MMNS, ELEMENTARY_CHARGE

def distance_state(
    *,
    x: list[float] | None = None,
    y: list[float] | None = None,
    z: list[float] | None = None,
    t: list[float] | None = None,
    bx: list[float] | None = None,
    by: list[float] | None = None,
    bz: list[float] | None = None,
    char_time: list[float] | None = None,
) -> dict[str, np.ndarray]:
    x_vals = [0.0] if x is None else x
    size = len(x_vals)

    def _arr(values: list[float] | None, default: float = 0.0) -> np.ndarray:
        if values is None:
            values = [default] * size
        return np.array(values, dtype=float)

    return {
        "x": _arr(x),
        "y": _arr(y),
        "z": _arr(z),
        "t": _arr(t),
        "bx": _arr(bx),
        "by": _arr(by),
        "bz": _arr(bz),
        "char_time": _arr(char_time, 1e-3),
    }


def empty_driver_state() -> dict:
    return {
        key: np.array([], dtype=float)
        for key in (
            "x",
            "y",
            "z",
            "t",
            "Px",
            "Py",
            "Pz",
            "Pt",
            "gamma",
            "bx",
            "by",
            "bz",
            "bdotx",
            "bdoty",
            "bdotz",
            "q",
            "m",
            "char_time",
        )
    }


def single_particle_state(
    *,
    x: float = 0.0,
    y: float = 0.0,
    z: float = -1.0,
    t: float = 0.0,
    px: float = 0.0,
    py: float = 0.0,
    pz: float = 0.0,
    charge: float = ELEMENTARY_CHARGE,
    mass: float = 1.0,
    gamma: float = 1.0,
    bx: float = 0.0,
    by: float = 0.0,
    bz: float = 0.0,
    char_time: float = 1e-3,
) -> dict[str, np.ndarray]:
    arr = np.array
    return {
        "x": arr([x], dtype=float),
        "y": arr([y], dtype=float),
        "z": arr([z], dtype=float),
        "t": arr([t], dtype=float),
        "Px": arr([px], dtype=float),
        "Py": arr([py], dtype=float),
        "Pz": arr([pz], dtype=float),
        "Pt": arr([gamma * mass * C_MMNS], dtype=float),
        "gamma": arr([gamma], dtype=float),
        "bx": arr([bx], dtype=float),
        "by": arr([by], dtype=float),
        "bz": arr([bz], dtype=float),
        "bdotx": arr([0.0], dtype=float),
        "bdoty": arr([0.0], dtype=float),
        "bdotz": arr([0.0], dtype=float),
        "q": arr([charge], dtype=float),
        "m": arr([mass], dtype=float),
        "char_time": arr([char_time], dtype=float),
    }
