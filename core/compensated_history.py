"""Preserve small source-history variations as two float64 components.

These numerical primitives do not select a history model or change a force law.
Callers must preserve precision while generating samples, not split previously
rounded absolute values and expect lost information to return.
"""

from __future__ import annotations

from fractions import Fraction
from math import frexp, fsum, isfinite, ldexp
import sys

import numpy as np


def _sum_parts(terms):
    terms = tuple(terms)
    high = fsum(terms)
    low = fsum((*terms, -high))
    if not isfinite(high) or not isfinite(low):
        raise ValueError("Finite compensated sum required")
    return high, low


def add_parts(first, second):
    """Add two (high, low) scalar pairs without first rounding their sums."""
    if len(first) != 2 or len(second) != 2:
        raise ValueError("Two components per compensated value required")
    if not all(isfinite(v) for v in (*first, *second)):
        raise ValueError("Finite compensated inputs required")
    return _sum_parts((*first, *second))


def _product_parts(a, b):
    product = a * b
    if not isfinite(product):
        raise ValueError("Compensated product overflow")
    if not a or not b:
        return product, 0.0
    if abs(product) < sys.float_info.min:
        raise ValueError("Compensated products must remain normal float64 values")
    # Split normalized mantissas, avoiding overflow in the splitter itself.
    ma, ea = frexp(a)
    mb, eb = frexp(b)
    splitter = 134217729.0
    ca, cb = splitter * ma, splitter * mb
    ah, bh = ca - (ca - ma), cb - (cb - mb)
    al, bl = ma - ah, mb - bh
    p = ma * mb
    error = ((ah * bh - p) + ah * bl + al * bh) + al * bl
    return product, ldexp(error, ea + eb)


def multiply_parts(first, second):
    """Multiply scalar pairs, retaining the low part of all four products."""
    if len(first) != 2 or len(second) != 2:
        raise ValueError("Two components per compensated value required")
    if not all(isfinite(v) for v in (*first, *second)):
        raise ValueError("Finite compensated inputs required")
    return _sum_parts(
        part for a in first for b in second for part in _product_parts(a, b)
    )


class CompensatedHistoryMap:
    """Apply a known linear reconstruction map to preserved sample differences.

    Rational weights are split once. Input arrays have samples on the first axis;
    the two arrays must be generated with preserved precision. An optional sample
    reference is subtracted in compensated arithmetic before multiplication.
    Returned high/low arrays retain the output coefficient remainders.
    This helper supplies no implicit extrapolation or derivative-order choice.
    """

    def __init__(self, weights):
        weights = np.asarray(weights, dtype=object)
        if weights.ndim != 2 or not all(weights.shape):
            raise ValueError("A nonempty two-dimensional weight matrix is required")
        high, low = np.empty(weights.shape), np.empty(weights.shape)
        for index in np.ndindex(weights.shape):
            exact = Fraction(weights[index])
            high[index] = float(exact)
            low[index] = float(exact - Fraction(high[index]))
        if not np.isfinite(high).all() or not np.isfinite(low).all():
            raise ValueError("Finite reconstruction weights required")
        high.flags.writeable = low.flags.writeable = False
        self.high, self.low = high, low

    def apply(self, high, low, *, reference_index=None):
        high, low = np.asarray(high, dtype=float), np.asarray(low, dtype=float)
        if (
            high.shape != low.shape
            or high.ndim < 1
            or high.shape[0] != self.high.shape[1]
            or not np.isfinite(high).all()
            or not np.isfinite(low).all()
        ):
            raise ValueError(
                "Finite equally shaped sample arrays matching the map are required"
            )
        if reference_index is not None and (
            type(reference_index) is not int or not 0 <= reference_index < len(high)
        ):
            raise ValueError("Reference sample index is out of range")
        h, tail = high.reshape(len(high), -1), low.reshape(len(low), -1)
        if reference_index is not None:
            h, tail = h.copy(), tail.copy()
            refh, refl = h[reference_index].copy(), tail[reference_index].copy()
            for i, j in np.ndindex(h.shape):
                h[i, j], tail[i, j] = add_parts(
                    (h[i, j], tail[i, j]), (-refh[j], -refl[j])
                )
        result_h = np.empty((len(self.high), h.shape[1]))
        result_l = np.empty_like(result_h)
        for row, column in np.ndindex(result_h.shape):
            terms = []
            for sample in range(len(high)):
                terms.extend(
                    multiply_parts(
                        (self.high[row, sample], self.low[row, sample]),
                        (h[sample, column], tail[sample, column]),
                    )
                )
            result_h[row, column], result_l[row, column] = _sum_parts(terms)
        shape = (len(self.high),) + high.shape[1:]
        return result_h.reshape(shape), result_l.reshape(shape)
