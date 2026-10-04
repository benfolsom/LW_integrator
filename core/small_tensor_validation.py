"""Strict compiled checks for the small, repeatedly validated observer tensors."""

import math

import numpy as np
from numba import njit


@njit(cache=True, fastmath=False)
def antisymmetric_tensor_status(values: np.ndarray) -> int:
    """Match finite checks and allclose(A, -A.T, rtol=0, atol=1e-15)."""
    for value in values.flat:
        if not math.isfinite(value):
            return 1
    matrices = values.reshape((-1, 4, 4))
    for matrix in matrices:
        for row in range(4):
            for column in range(4):
                if abs(matrix[row, column] - (-matrix[column, row])) > 1e-15:
                    return 2
    return 0
