"""CuPy float64 CUDA kernels and FFT; optional dependency imported on request."""

import numpy as np

from .common import array_poisson, inverse_laplacian
from .kernel_source import cuda_source


class CuPy:
    name = "cupy"

    def __init__(self, grid, dtype="float64"):
        import cupy as cp

        if dtype not in ("float32", "float64"):
            raise ValueError("CuPy dtype must be float32 or float64")
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("CuPy requires a CUDA GPU")
        self.cp, self.grid, self.dtype = cp, grid, dtype
        self.inverse = self.array(inverse_laplacian(grid))
        self.module = cp.RawModule(
            code=cuda_source(grid, dtype), options=("--std=c++11", "--fmad=false")
        )
        # Compile every specialization upfront, including push.
        self.kernels = {
            name: self.module.get_function(name)
            for name in ("deposit1", "deposit2", "gather1", "gather2", "push")
        }

    def array(self, value):
        return self.cp.ascontiguousarray(self.cp.asarray(value, dtype=self.dtype))

    def host(self, value):
        return self.cp.asnumpy(value)

    def sync(self, *values):
        self.cp.cuda.get_current_stream().synchronize()

    def _run(self, name, count, args):
        self.kernels[name](((count + 255) // 256,), (256,), (*args, np.int32(count)))

    def deposit(self, x, q, order):
        shape = (self.grid.n,) * 3
        high, low = (self.cp.zeros(shape, self.dtype) for _ in range(2))
        self._run(f"deposit{order}", len(x), (x, q, high, low))
        return high + low

    def gather(self, x, field, order):
        result = self.cp.empty(x.shape, self.dtype)
        self._run(f"gather{order}", len(x), (x, field, result))
        return result

    def push(self, x, u, electric, magnetic, qm, dt):
        xn, un = self.cp.empty_like(x), self.cp.empty_like(u)
        self._run(
            "push", len(x), (x, u, electric, magnetic, qm, self.array([dt]), xn, un)
        )
        return xn, un

    def poisson(self, rho):
        return array_poisson(self.cp, rho, self.inverse, self.grid.dx)
