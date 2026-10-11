"""CPU emulation of shared C bodies; this does not validate GPU execution."""

import ctypes
import shutil
import subprocess
from pathlib import Path

import numpy as np

from core.pic.gpu_backend import GPUBackend
from core.pic.gpu_source import cuda_source, header, operations


class HostCBackend(GPUBackend):
    """Strict serial C, NumPy FFT, and the actual GPU orchestration."""

    xp = np

    def __init__(self, dtype, build_dir, dialect="cuda"):
        if dialect not in ("cuda", "metal") or (
            dialect == "metal" and dtype != "float32"
        ):
            raise ValueError(
                "host source check supports CUDA float32/float64 or Metal float32"
            )
        self.dialect = dialect
        self.dtype = dtype
        self.name = f"host_{dialect}_c_emulation_{dtype}"
        super().__init__()
        build_dir = Path(build_dir)
        build_dir.mkdir(parents=True, exist_ok=True)
        compiler = shutil.which("clang++") or shutil.which("g++")
        if compiler is None:
            raise RuntimeError("C++ compiler required for CPU source check")
        preamble = """
#define __device__
#define __global__
struct Dim { int x; };
Dim blockIdx{0},blockDim{1},threadIdx{0};
template<class V> V atomicAdd(V* ptr,V v) { V old=*ptr;*ptr+=v;return old; }
"""
        device_source = cuda_source(dtype)
        if dialect == "metal":
            preamble += """
#include <math.h>
#define thread
const int memory_order_relaxed=0;
template<class V> V atomic_fetch_add_explicit(V* ptr,V v,int) {
    V old=*ptr;*ptr+=v;return old;
}
"""
            device_source = header(True)
            for name, (inputs, outputs, body, _) in operations(True).items():
                args = [f"const {'int' if a == 'meta' else 'T'}* {a}" for a in inputs]
                args += [f"T* {a}" for a in outputs] + ["int count"]
                device_source += (
                    f'void {name}({",".join(args)}) {{'
                    "int p=threadIdx.x;if(p>=count) return;" + body + "}\n"
                )
        wrappers = []
        for name, (inputs, outputs, _, _) in operations().items():
            args = [f"const {'int' if a == 'meta' else 'T'}* {a}" for a in inputs]
            args += [f"T* {a}" for a in outputs] + ["int count"]
            wrappers.append(
                f'extern "C" void host_{name}({",".join(args)}) {{'
                f"for(int p=0;p<count;++p) {{threadIdx.x=p;"
                f'{name}({",".join(inputs + outputs)},count);}}}}'
            )
        source = build_dir / f"pic_gpu_{dialect}_{dtype}.cpp"
        library = build_dir / f"pic_gpu_{dialect}_{dtype}.so"
        source.write_text(preamble + device_source + "\n".join(wrappers))
        self.compile_command = [
            compiler,
            "-shared",
            "-fPIC",
            "-std=c++11",
            "-O2",
            "-ffp-contract=off",
            "-fno-fast-math",
            str(source),
            "-o",
            str(library),
        ]
        subprocess.run(self.compile_command, check=True, capture_output=True, text=True)
        self.library = ctypes.CDLL(str(library))
        self.functions = {}
        for name, (inputs, outputs, _, _) in operations().items():
            function = getattr(self.library, "host_" + name)
            kinds = ["int32" if a == "meta" else dtype for a in inputs + outputs]
            function.argtypes = [
                np.ctypeslib.ndpointer(dtype=k, flags="C_CONTIGUOUS") for k in kinds
            ] + [ctypes.c_int]
            function.restype = None
            self.functions[name] = function

    def array(self, value, dtype=None):
        return np.ascontiguousarray(value, dtype=dtype or self.dtype)

    def to_host(self, value):
        return np.asarray(value)

    def synchronize(self, *values):
        pass

    def _run(self, name, inputs, shapes, count, zero=False):
        alloc = np.zeros if zero else np.empty
        outputs = [alloc(s, dtype=self.dtype) for s in shapes]
        inputs = [
            self.array(v, "int32" if a == "meta" else self.dtype)
            for a, v in zip(operations()[name][0], inputs)
        ]
        if count:
            self.functions[name](*inputs, *outputs, count)
        return outputs
