"""MLX float32 arrays, compensated Metal deposition, and custom push/gather.

Import is safe without MLX. The optional CUDA mode tests the same float32
body through fast.cuda_kernel; it is not a float64 replacement for CuPy.
"""

import numpy as np

from .common import array_poisson, inverse_laplacian
from .kernel_source import deposit_body, gather_body, push_body


class MLX:
    dtype = "float32"

    def __init__(self, grid, cuda=False, compensated=True):
        import mlx.core as mx

        self.mx, self.grid = mx, grid
        self.name = "mlx-cuda" if cuda else "mlx"
        mx.set_default_device(mx.gpu)
        if not mx.is_available(mx.gpu):
            raise RuntimeError("MLX GPU unavailable; CPU fallback forbidden")
        self.cuda, self.compensated = cuda, compensated
        self.kernels = {}
        self.inverse = self.array(inverse_laplacian(grid))
        for order in (1, 2):
            body = deposit_body(grid, order, metal=not cuda, compensated=compensated)
            self.kernels[f"deposit{order}"] = self._kernel(
                f"deposit{order}",
                ["x", "q"],
                ["high", "low"],
                body,
                atomic=True,
            )
            self.kernels[f"gather{order}"] = self._kernel(
                f"gather{order}",
                ["x", "field"],
                ["result"],
                gather_body(grid, order),
            )
        self.kernels["push"] = self._kernel(
            "push",
            ["x", "u", "electric", "magnetic", "qm", "dt"],
            ["xn", "un"],
            push_body(grid),
        )

    def _kernel(self, name, inputs, outputs, body, atomic=False):
        mx = self.mx
        if self.cuda:
            # API checked at runtime, with the same conformance gate in Colab.
            factory = mx.fast.cuda_kernel
            prefix = "uint p=blockIdx.x*blockDim.x+threadIdx.x;"
            header = "typedef float T;"
            kwargs = {}
        else:
            factory = mx.fast.metal_kernel
            prefix = "uint p=thread_position_in_grid.x;"
            header = "typedef float T;"
            kwargs = {
                "atomic_outputs": atomic,
                "compile_options": {"math_mode": "safe"},
            }
        return factory(
            name=f"pic_{name}",
            input_names=inputs,
            output_names=outputs,
            header=header,
            source=prefix + "if(p >= x_shape[0]) return;\n" + body,
            **kwargs,
        )

    def _run(self, name, inputs, shapes, zero=False):
        count = inputs[0].shape[0]
        kwargs = {
            "inputs": inputs,
            "output_shapes": shapes,
            "output_dtypes": [self.mx.float32] * len(shapes),
            "grid": (count, 1, 1),
            "threadgroup": (256, 1, 1),
        }
        if zero:
            kwargs["init_value"] = 0
        return self.kernels[name](**kwargs)

    def array(self, value):
        return self.mx.array(np.asarray(value, dtype=np.float32))

    def host(self, value):
        self.sync(value)
        return np.asarray(value)

    def sync(self, *values):
        if values:
            self.mx.eval(*values)
        self.mx.synchronize()

    def deposit(self, x, q, order):
        shape = (self.grid.n,) * 3
        high, low = self._run(f"deposit{order}", [x, q], [shape, shape], True)
        return high + low if self.compensated else high

    def gather(self, x, field, order):
        return self._run(f"gather{order}", [x, field], [x.shape])[0]

    def push(self, x, u, electric, magnetic, qm, dt):
        return tuple(
            self._run(
                "push",
                [x, u, electric, magnetic, qm, self.array([dt])],
                [x.shape, u.shape],
            )
        )

    def poisson(self, rho):
        return array_poisson(self.mx, rho, self.inverse, self.grid.dx)
