"""Order-four, factorial-scaled real Taylor arithmetic for local ODE derivatives.

No trajectory fitting. This small scalar type also supports object-array
arithmetic in the shared momentum-center force algebra.
"""

from __future__ import annotations

from typing import overload
import numpy as np
from numpy.typing import ArrayLike

ORDER = 4


class Series:
    def __init__(self, coefficients: ArrayLike) -> None:
        self.c = np.asarray(coefficients, dtype=float)
        if self.c.shape != (ORDER + 1,):
            raise ValueError("Five Taylor coefficients required")

    @classmethod
    def constant(cls, value: float) -> Series:
        result = np.zeros(ORDER + 1)
        result[0] = value
        return cls(result)

    @staticmethod
    def coerce(value: Series | float) -> Series:
        return value if isinstance(value, Series) else Series.constant(value)

    def __add__(self, other: Series | float) -> Series:
        return Series(self.c + self.coerce(other).c)

    __radd__ = __add__

    def __neg__(self) -> Series:
        return Series(-self.c)

    def __sub__(self, other: Series | float) -> Series:
        return self + (-self.coerce(other))

    def __rsub__(self, other: Series | float) -> Series:
        return self.coerce(other) - self

    @overload
    def __mul__(self, other: Series | float) -> Series: ...

    @overload
    def __mul__(self, other: np.ndarray) -> Series | np.ndarray: ...

    def __mul__(self, other: Series | float | np.ndarray) -> Series | np.ndarray:
        if isinstance(other, np.ndarray) and other.ndim:
            return np.array([self * v for v in other.flat], dtype=object).reshape(
                other.shape
            )
        scalar = float(other) if isinstance(other, np.ndarray) else other
        return Series(np.convolve(self.c, self.coerce(scalar).c)[: ORDER + 1])

    __rmul__ = __mul__

    def __pow__(self, power: int) -> Series:
        if not isinstance(power, int) or power < 0:
            raise ValueError("Nonnegative integer power required")
        result = self.constant(1)
        for _ in range(power):
            result = result * self
        return result

    def reciprocal(self) -> Series:
        if self.c[0] == 0:
            raise ZeroDivisionError("Zero Taylor constant term")
        result = np.zeros(ORDER + 1)
        result[0] = 1 / self.c[0]
        for n in range(1, ORDER + 1):
            result[n] = (
                -sum(self.c[j] * result[n - j] for j in range(1, n + 1)) / self.c[0]
            )
        return Series(result)

    def __truediv__(self, other: Series | float) -> Series:
        return self * self.coerce(other).reciprocal()

    def __rtruediv__(self, other: Series | float) -> Series:
        return self.coerce(other) * self.reciprocal()

    def sqrt(self) -> Series:
        if not np.isfinite(self.c[0]) or self.c[0] <= 0:
            raise ValueError("Positive finite Taylor constant term required")
        result = np.zeros(ORDER + 1)
        result[0] = np.sqrt(self.c[0])
        for n in range(1, ORDER + 1):
            result[n] = (
                self.c[n] - sum(result[j] * result[n - j] for j in range(1, n))
            ) / (2 * result[0])
        return Series(result)


def to_series(coefficients: ArrayLike) -> np.ndarray:
    values = np.asarray(coefficients)
    return np.array(
        [Series(v) for v in values.reshape(ORDER + 1, -1).T], dtype=object
    ).reshape(values.shape[1:])


def from_series(values: ArrayLike) -> np.ndarray:
    values = np.asarray(values)
    return np.array([Series.coerce(v).c for v in values.flat]).T.reshape(
        (ORDER + 1,) + values.shape
    )


def solve(matrix: ArrayLike, right: ArrayLike) -> np.ndarray:
    """Match M[0] v[n] = b[n] - sum M[j] v[n-j]; no frozen matrix."""
    matrices, rhs = from_series(matrix), from_series(right)
    values = np.zeros_like(rhs)
    for n in range(ORDER + 1):
        adjusted = rhs[n].copy()
        for j in range(1, n + 1):
            adjusted -= matrices[j] @ values[n - j]
        values[n] = np.linalg.solve(matrices[0], adjusted)
    return to_series(values)
