"""Observer derivatives through order three, carried along order-four motion.

The 5 by 35 coefficients retain the existing observer derivative ordering.
Strict compiled arithmetic reuses the original order-three multiplication
kernel; no fastmath or finite-difference sampling is used.
"""

from math import factorial
from functools import lru_cache
import numpy as np

from .dipole_hertz_jet import _INDEX, _MULTIINDICES
from .dipole_hertz_jet import _multiply_coefficients, _reciprocal_coefficients


@lru_cache(maxsize=128)
def _derivative_location(indices):
    """Cache index bookkeeping only; coefficient arithmetic stays unchanged."""
    alpha = [0] * 4
    for index in indices:
        alpha[index] += 1
    factor = 1
    for count in alpha:
        factor *= factorial(count)
    return _INDEX[tuple(alpha)], factor


def _product(a, b):
    result = np.zeros_like(a)
    for n in range(5):
        for j in range(n + 1):
            result[n] += _multiply_coefficients(a[j], b[n - j])
    return result


def _inverse(a):
    result = np.zeros_like(a)
    result[0] = _reciprocal_coefficients(a[0])
    for n in range(1, 5):
        total = np.zeros(a.shape[1])
        for j in range(1, n + 1):
            total += _multiply_coefficients(a[j], result[n - j])
        result[n] = -_multiply_coefficients(result[0], total)
    return result


try:
    from numba import njit

    # Optional Numba; the imported kernels use exactly the same split ordering.
    if hasattr(_multiply_coefficients, "py_func"):
        _product = njit(cache=True)(_product)
        _inverse = njit(cache=True)(_inverse)
except ImportError:
    pass


class ResponseTaylor:
    def __init__(self, coefficients):
        self.coefficients = np.asarray(coefficients, dtype=float)
        if self.coefficients.shape != (5, len(_MULTIINDICES)):
            raise ValueError(
                "Order-four motion and order-three observer coefficients required"
            )

    @classmethod
    def constant(cls, value):
        data = np.zeros((5, len(_MULTIINDICES)))
        data[0, 0] = value
        return cls(data)

    @classmethod
    def variable(cls, coefficients, index):
        data = np.zeros((5, len(_MULTIINDICES)))
        data[:, 0] = coefficients
        alpha = [0] * 4
        alpha[index] = 1
        data[0, _INDEX[tuple(alpha)]] = 1
        return cls(data)

    @property
    def value(self):
        return self.coefficients[0, 0]

    def with_value(self, value):
        data = self.coefficients.copy()
        data[0, 0] = value
        return type(self)(data)

    def derivative(self, *indices):
        column, factor = _derivative_location(indices)
        return self.coefficients[:, column] * factor

    @classmethod
    def coerce(cls, value):
        return value if isinstance(value, cls) else cls.constant(value)

    def __add__(self, other):
        return type(self)(self.coefficients + self.coerce(other).coefficients)

    __radd__ = __add__

    def __neg__(self):
        return type(self)(-self.coefficients)

    def __sub__(self, other):
        return self + (-self.coerce(other))

    def __rsub__(self, other):
        return self.coerce(other) - self

    def __mul__(self, other):
        return type(self)(_product(self.coefficients, self.coerce(other).coefficients))

    __rmul__ = __mul__

    def reciprocal(self):
        if self.value == 0:
            raise ZeroDivisionError("Zero Taylor constant term")
        return type(self)(_inverse(self.coefficients))

    def __truediv__(self, other):
        return self * self.coerce(other).reciprocal()

    def __rtruediv__(self, other):
        return self.coerce(other) * self.reciprocal()

    def sqrt(self):
        if not np.isfinite(self.value) or self.value <= 0:
            raise ValueError("Positive Taylor constant term required")
        root = type(self).constant(np.sqrt(self.value))
        # Highest combined degree is 4+3=7; four Newton lifts suffice.
        for _ in range(4):
            root = 0.5 * (root + self / root)
        return root
