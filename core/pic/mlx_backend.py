"""MLX Metal float32 adapter; imported only after Apple-silicon host checks."""

from typing import Any

import numpy as np

from .gpu_backend import GPUBackend
from .gpu_source import header, operations


class MLXBackend(GPUBackend):
    dtype = "float32"
    name = "mlx_metal_float32_compensated"

    def __init__(self) -> None:
        from .backend import require_host

        require_host("mlx")
        import mlx.core as mx

        if not mx.is_available(mx.gpu):
            raise RuntimeError("MLX Metal device unavailable; CPU fallback forbidden")
        self.xp = mx
        mx.set_default_device(mx.gpu)
        super().__init__()
        self._kernels = {}
        for name, (inputs, outputs, body, atomic) in operations(True).items():
            first = inputs[0]
            self._kernels[name] = mx.fast.metal_kernel(
                name=f"native_pic_{name}",
                input_names=inputs,
                output_names=outputs,
                header=header(True),
                source=(
                    "uint p=thread_position_in_grid.x;"
                    f"if(p>={first}_shape[0]) return;\n" + body
                ),
                atomic_outputs=atomic,
                compile_options={"math_mode": "safe"},
            )

    def array(self, value: Any, dtype: str | None = None) -> Any:
        mx = self.xp
        kind = getattr(mx, dtype or self.dtype)
        if isinstance(value, mx.array):
            return mx.contiguous(value.astype(kind))
        return mx.array(np.asarray(value, dtype=dtype or self.dtype), dtype=kind)

    def to_host(self, value: Any) -> np.ndarray:
        if isinstance(value, self.xp.array):
            self.synchronize(value)
        return np.asarray(value)

    def synchronize(self, *values: Any) -> None:
        if values:
            self.xp.eval(*values)
        self.xp.synchronize()

    def _run(
        self,
        name: str,
        inputs: list[Any],
        shapes: list[Any],
        count: int,
        zero: bool = False,
    ) -> list[Any]:
        if count == 0:
            return [self.xp.zeros(s, dtype=self.xp.float32) for s in shapes]
        kwargs: dict[str, Any] = dict(
            inputs=inputs,
            output_shapes=shapes,
            output_dtypes=[self.xp.float32] * len(shapes),
            grid=(count, 1, 1),
            threadgroup=(256, 1, 1),
        )
        if zero:
            kwargs["init_value"] = 0
        return list(self._kernels[name](**kwargs))
