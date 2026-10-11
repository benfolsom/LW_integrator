"""CuPy CUDA adapter with float64 default and explicit float32 control."""

from typing import Any

import numpy as np

from .gpu_backend import GPUBackend
from .gpu_source import cuda_source, operations


class CuPyBackend(GPUBackend):
    def __init__(self, dtype: str = "float64") -> None:
        from .backend import check_cupy_build, require_host

        require_host("cupy")
        if dtype not in ("float32", "float64"):
            raise ValueError("CuPy precision must be float32 or float64")
        self.expected_cuda_build = check_cupy_build()
        import cupy as cp

        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("CuPy requires a CUDA device; CPU fallback forbidden")
        self.xp, self.dtype = cp, dtype
        self.name = f"cupy_cuda_{dtype}_compensated"
        super().__init__()
        self._module = cp.RawModule(
            code=cuda_source(dtype), options=("--std=c++11", "--fmad=false")
        )
        self._kernels = {name: self._module.get_function(name) for name in operations()}

    def array(self, value: Any, dtype: str | None = None) -> Any:
        return self.xp.ascontiguousarray(
            self.xp.asarray(value, dtype=dtype or self.dtype)
        )

    def to_host(self, value: Any) -> np.ndarray:
        return (
            self.xp.asnumpy(value)
            if isinstance(value, self.xp.ndarray)
            else np.asarray(value)
        )

    def synchronize(self, *values: Any) -> None:
        self.xp.cuda.get_current_stream().synchronize()

    def _run(
        self,
        name: str,
        inputs: list[Any],
        shapes: list[Any],
        count: int,
        zero: bool = False,
    ) -> list[Any]:
        alloc = self.xp.zeros if zero else self.xp.empty
        outputs = [alloc(shape, self.dtype) for shape in shapes]
        if count:
            self._kernels[name](
                ((count + 255) // 256,), (256,), (*inputs, *outputs, np.int32(count))
            )
        return outputs
