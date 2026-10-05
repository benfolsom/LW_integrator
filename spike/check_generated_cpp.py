"""Compile and execute shared CUDA/C kernel bodies on the Mac CPU.

This catches C syntax, stencil, and push errors without CuPy or CUDA installed.
It does NOT compile a Metal shader or a CUDA device binary, test atomics under
concurrency, or validate either GPU backend. FFT here is NumPy, not GPU FFT.
"""

import argparse
import ctypes
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .common import Grid
from .conformance import run_backend
from .cpu import CPU
from .kernel_source import cuda_source
from .provenance import metadata


class HostC(CPU):
    def __init__(self, grid, dtype, build_dir):
        super().__init__(grid)
        self.dtype = dtype
        self.name = f"host-c-emulation-{dtype}"
        compiler = shutil.which("clang++") or shutil.which("g++")
        if not compiler:
            raise RuntimeError("C++ compiler unavailable")
        preamble = """
        #define __global__
        struct Dim { int x; };
        Dim blockIdx{0}, blockDim{1}, threadIdx{0};
        template<class V> V atomicAdd(V* ptr, V v) {
            V old = *ptr; *ptr += v; return old;
        }
        """
        wrappers = []
        for order in (1, 2):
            wrappers.append(f"""extern "C" void host_deposit{order}(
                const T* x,const T* q,T* high,T* low,int count) {{
                for(int p=0;p<count;++p) {{
                    threadIdx.x=p; deposit{order}(x,q,high,low,count);
                }}
            }}""")
            wrappers.append(f"""extern "C" void host_gather{order}(
                const T* x,const T* field,T* result,int count) {{
                for(int p=0;p<count;++p) {{
                    threadIdx.x=p; gather{order}(x,field,result,count);
                }}
            }}""")
        wrappers.append("""extern "C" void host_push(const T* x,const T* u,
            const T* electric,const T* magnetic,const T* qm,const T* dt,
            T* xn,T* un,int count) {
            for(int p=0;p<count;++p) {
                threadIdx.x=p; push(x,u,electric,magnetic,qm,dt,xn,un,count);
            }
        }""")
        build_dir.mkdir(parents=True, exist_ok=True)
        source = build_dir / f"shared-{dtype}.cpp"
        library = build_dir / f"shared-{dtype}.so"
        source.write_text(preamble + cuda_source(grid, dtype) + "\n".join(wrappers))
        self.command = [
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
        completed = subprocess.run(
            self.command, capture_output=True, text=True, check=False
        )
        if completed.returncode:
            raise RuntimeError(completed.stderr)
        self.library = ctypes.CDLL(str(library))
        self.pointer = np.ctypeslib.ndpointer(dtype=dtype, flags="C_CONTIGUOUS")
        for name, count in (
            ("deposit1", 4),
            ("deposit2", 4),
            ("gather1", 3),
            ("gather2", 3),
            ("push", 8),
        ):
            kernel = getattr(self.library, f"host_{name}")
            kernel.argtypes = [self.pointer] * count + [ctypes.c_int]
            kernel.restype = None

    def array(self, value):
        return np.ascontiguousarray(value, dtype=self.dtype)

    def deposit(self, x, q, order):
        high, low = [np.zeros((self.grid.n,) * 3, dtype=self.dtype) for _ in range(2)]
        getattr(self.library, f"host_deposit{order}")(x, q, high, low, len(x))
        return high + low

    def gather(self, x, field, order):
        result = np.empty_like(x)
        getattr(self.library, f"host_gather{order}")(
            x, self.array(field), result, len(x)
        )
        return result

    def push(self, x, u, electric, magnetic, qm, dt):
        xn, un = np.empty_like(x), np.empty_like(u)
        self.library.host_push(
            x, u, self.array(electric), magnetic, qm, self.array([dt]), xn, un, len(x)
        )
        return xn, un


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"metadata": metadata(), "validation_scope": __doc__, "results": []}
    for dtype in ("float32", "float64"):
        backend = HostC(Grid(16), dtype, args.build_dir)
        result = run_backend(backend, count=4096)
        result["compile_command"] = backend.command
        report["results"].append(result)
        print(dtype, result["passed"], [c for c in result["checks"] if not c["passed"]])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return int(not all(r["passed"] for r in report["results"]))


if __name__ == "__main__":
    raise SystemExit(main())
