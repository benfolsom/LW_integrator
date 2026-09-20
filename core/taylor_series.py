"""Order-four, factorial-scaled real Taylor arithmetic for local ODE derivatives.

No trajectory fitting. This small scalar type also supports object-array
arithmetic in the shared momentum-center force algebra.
"""

import numpy as np

ORDER = 4


class Series:
    def __init__(self, coefficients):
        self.c = np.asarray(coefficients, dtype=float)
        if self.c.shape != (ORDER + 1,):
            raise ValueError("Five Taylor coefficients required")

    @classmethod
    def constant(cls, value):
        result = np.zeros(ORDER + 1)
        result[0] = value
        return cls(result)

    @staticmethod
    def coerce(value):
        return value if isinstance(value, Series) else Series.constant(value)

    def __add__(self, other):
        return Series(self.c + self.coerce(other).c)

    __radd__ = __add__

    def __neg__(self):
        return Series(-self.c)

    def __sub__(self, other):
        return self + (-self.coerce(other))

    def __rsub__(self, other):
        return self.coerce(other) - self

    def __mul__(self, other):
        if isinstance(other, np.ndarray) and other.ndim:
            return np.array([self * v for v in other.flat], dtype=object).reshape(
                other.shape
            )
        return Series(np.convolve(self.c, self.coerce(other).c)[: ORDER + 1])

    __rmul__ = __mul__

    def __pow__(self, power):
        if not isinstance(power, int) or power < 0:
            raise ValueError("Nonnegative integer power required")
        result = self.constant(1)
        for _ in range(power):
            result = result * self
        return result

    def reciprocal(self):
        if self.c[0] == 0:
            raise ZeroDivisionError("Zero Taylor constant term")
        result = np.zeros(ORDER + 1)
        result[0] = 1 / self.c[0]
        for n in range(1, ORDER + 1):
            result[n] = (
                -sum(self.c[j] * result[n - j] for j in range(1, n + 1)) / self.c[0]
            )
        return Series(result)

    def __truediv__(self, other):
        return self * self.coerce(other).reciprocal()

    def __rtruediv__(self, other):
        return self.coerce(other) * self.reciprocal()

    def sqrt(self):
        if not np.isfinite(self.c[0]) or self.c[0] <= 0:
            raise ValueError("Positive finite Taylor constant term required")
        result = np.zeros(ORDER + 1)
        result[0] = np.sqrt(self.c[0])
        for n in range(1, ORDER + 1):
            result[n] = (
                self.c[n] - sum(result[j] * result[n - j] for j in range(1, n))
            ) / (2 * result[0])
        return Series(result)


def to_series(coefficients):
    values = np.asarray(coefficients)
    return np.array(
        [Series(v) for v in values.reshape(ORDER + 1, -1).T], dtype=object
    ).reshape(values.shape[1:])


def from_series(values):
    values = np.asarray(values)
    return np.array([Series.coerce(v).c for v in values.flat]).T.reshape(
        (ORDER + 1,) + values.shape
    )


def solve(matrix, right):
    """Match M[0] v[n] = b[n] - sum M[j] v[n-j]; no frozen matrix."""
    matrices, rhs = from_series(matrix), from_series(right)
    values = np.zeros_like(rhs)
    for n in range(ORDER + 1):
        adjusted = rhs[n].copy()
        for j in range(1, n + 1):
            adjusted -= matrices[j] @ values[n - j]
        values[n] = np.linalg.solve(matrices[0], adjusted)
    return to_series(values)
