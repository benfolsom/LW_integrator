"""Single-source Taichi particle kernels and radix-2 3D FFT.

The FFT is deliberately native Taichi, rather than a hidden NumPy/MLX/CuPy
bridge. It is a simple baseline with a dispatch per axis/stage, not an
optimized vendor FFT. Native arrays stay resident between operations.
"""

import math
import os
from pathlib import Path

import numpy as np
import taichi as ti

from .common import inverse_laplacian

_INITIALIZED_ARCH = None


@ti.data_oriented
class Taichi:
    def __init__(self, grid, arch="metal", dtype=None, compensated=True):
        global _INITIALIZED_ARCH
        self.name = f"taichi-{arch}"
        self.grid = grid
        self.dtype = dtype or ("float32" if arch == "metal" else "float64")
        if self.dtype not in ("float32", "float64"):
            raise ValueError("Taichi dtype must be float32 or float64")
        if arch == "metal" and self.dtype != "float32":
            raise ValueError("Metal has no float64 shader arithmetic")
        self.real = ti.f64 if self.dtype == "float64" else ti.f32
        self.n, self.dx, self.length = grid.n, grid.dx, grid.length
        self.logn = int(math.log2(self.n))
        self.compensated = compensated
        if _INITIALIZED_ARCH is None:
            cache = Path(
                os.environ.get(
                    "LW_SPIKE_CACHE",
                    str(Path.home() / ".local/share/lw-gpu-spike/taichi-cache"),
                )
            )
            cache.mkdir(parents=True, exist_ok=True)
            ti.init(
                arch=getattr(ti, arch),
                default_fp=self.real,
                enable_fallback=False,
                fast_math=False,
                cpu_max_num_threads=1,
                offline_cache_file_path=str(cache),
            )
            _INITIALIZED_ARCH = arch
        elif _INITIALIZED_ARCH != arch:
            raise RuntimeError("Use a separate process for each Taichi architecture")
        if ti.lang.impl.current_cfg().arch != getattr(ti, arch):
            raise RuntimeError("Unexpected Taichi architecture; fallback forbidden")
        self.inverse = self.array(inverse_laplacian(grid))

    def array(self, value):
        data = np.ascontiguousarray(value, dtype=self.dtype)
        result = ti.ndarray(dtype=self.real, shape=data.shape)
        result.from_numpy(data)
        return result

    def host(self, value):
        return value.to_numpy()

    def sync(self, *values):
        ti.sync()

    def empty(self, shape):
        return ti.ndarray(dtype=self.real, shape=shape)

    @ti.func
    def _stencil(self, x, order: ti.template()):
        ids = ti.Matrix.zero(ti.i32, 3, 3)
        w = ti.Matrix.zero(self.real, 3, 3)
        for a in ti.static(range(3)):
            s = x[a] / self.dx
            s -= ti.floor(s / self.n) * self.n
            base = ti.cast(ti.floor(s), ti.i32)
            if ti.static(order == 1):
                f = s - base
                w[a, 0], w[a, 1] = 1 - f, f
            else:
                center = ti.cast(ti.floor(s + 0.5), ti.i32)
                d = s - center
                base = center - 1
                w[a, 0] = 0.5 * (0.5 - d) ** 2
                w[a, 1] = 0.75 - d * d
                w[a, 2] = 0.5 * (0.5 + d) ** 2
            for j in ti.static(range(order + 1)):
                ids[a, j] = (base + j + self.n) % self.n
        return ids, w

    @ti.kernel
    def _deposit(
        self,
        x: ti.types.ndarray(ndim=2),
        q: ti.types.ndarray(ndim=1),
        high: ti.types.ndarray(ndim=3),
        low: ti.types.ndarray(ndim=3),
        order: ti.template(),
    ):
        for p in range(x.shape[0]):
            xp = ti.Vector([x[p, 0], x[p, 1], x[p, 2]])
            ids, w = self._stencil(xp, order)
            for i, j, k in ti.ndrange(order + 1, order + 1, order + 1):
                a, b, c = ids[0, i], ids[1, j], ids[2, k]
                v = q[p] * w[0, i] * w[1, j] * w[2, k] / self.dx**3
                old = ti.atomic_add(high[a, b, c], v)
                if ti.static(self.compensated):
                    updated = old + v
                    z = updated - old
                    err = (old - (updated - z)) + (v - z)
                    ti.atomic_add(low[a, b, c], err)

    @ti.kernel
    def _combine(self, high: ti.types.ndarray(ndim=3), low: ti.types.ndarray(ndim=3)):
        for i, j, k in ti.ndrange(self.n, self.n, self.n):
            high[i, j, k] += low[i, j, k]

    def deposit(self, x, q, order):
        high, low = self.empty((self.n,) * 3), self.empty((self.n,) * 3)
        high.fill(0)
        low.fill(0)
        self._deposit(x, q, high, low, order)
        if self.compensated:
            self._combine(high, low)
        return high

    @ti.kernel
    def _gather(
        self,
        x: ti.types.ndarray(ndim=2),
        field: ti.types.ndarray(ndim=4),
        result: ti.types.ndarray(ndim=2),
        order: ti.template(),
    ):
        for p in range(x.shape[0]):
            xp = ti.Vector([x[p, 0], x[p, 1], x[p, 2]])
            ids, w = self._stencil(xp, order)
            total, correction = ti.Vector.zero(self.real, 3), ti.Vector.zero(
                self.real, 3
            )
            for i, j, k in ti.ndrange(order + 1, order + 1, order + 1):
                weight = w[0, i] * w[1, j] * w[2, k]
                for a in ti.static(range(3)):
                    y = (
                        weight * field[ids[0, i], ids[1, j], ids[2, k], a]
                        - correction[a]
                    )
                    updated = total[a] + y
                    correction[a] = (updated - total[a]) - y
                    total[a] = updated
            for a in ti.static(range(3)):
                result[p, a] = total[a]

    def gather(self, x, field, order):
        result = self.empty(x.shape)
        self._gather(x, field, result, order)
        return result

    @ti.kernel
    def _push(
        self,
        x: ti.types.ndarray(ndim=2),
        u: ti.types.ndarray(ndim=2),
        electric: ti.types.ndarray(ndim=2),
        magnetic: ti.types.ndarray(ndim=2),
        qm: ti.types.ndarray(ndim=1),
        dt: ti.template(),
        xn: ti.types.ndarray(ndim=2),
        un: ti.types.ndarray(ndim=2),
    ):
        for p in range(x.shape[0]):
            up = ti.Vector([u[p, 0], u[p, 1], u[p, 2]])
            e = ti.Vector([electric[p, 0], electric[p, 1], electric[p, 2]])
            b = ti.Vector([magnetic[p, 0], magnetic[p, 1], magnetic[p, 2]])
            kick = qm[p] * dt * 0.5
            minus = up + kick * e
            t = kick * b / ti.sqrt(1 + minus.dot(minus))
            s = 2 * t / (1 + t.dot(t))
            prime = minus + minus.cross(t)
            out = minus + prime.cross(s) + kick * e
            gamma = ti.sqrt(1 + out.dot(out))
            for a in ti.static(range(3)):
                pos = x[p, a] + dt * out[a] / gamma
                xn[p, a] = pos - ti.floor(pos / self.length) * self.length
                un[p, a] = out[a]

    def push(self, x, u, electric, magnetic, qm, dt):
        xn, un = self.empty(x.shape), self.empty(u.shape)
        self._push(x, u, electric, magnetic, qm, dt, xn, un)
        return xn, un

    @ti.kernel
    def _to_complex(self, rho: ti.types.ndarray(ndim=3), out: ti.types.ndarray(ndim=4)):
        for i, j, k in ti.ndrange(self.n, self.n, self.n):
            out[i, j, k, 0], out[i, j, k, 1] = rho[i, j, k], 0

    @ti.kernel
    def _bit_reverse(
        self, src: ti.types.ndarray(ndim=4), dst: ti.types.ndarray(ndim=4), axis: ti.i32
    ):
        for i, j, k in ti.ndrange(self.n, self.n, self.n):
            idx = ti.Vector([i, j, k])
            v, reverse = idx[axis], 0
            for bit in ti.static(range(self.logn)):
                reverse = (reverse << 1) | ((v >> bit) & 1)
            idx[axis] = reverse
            for a in ti.static(range(2)):
                dst[i, j, k, a] = src[idx[0], idx[1], idx[2], a]

    @ti.kernel
    def _butterfly(
        self,
        src: ti.types.ndarray(ndim=4),
        dst: ti.types.ndarray(ndim=4),
        axis: ti.i32,
        half: ti.i32,
        direction: ti.i32,
    ):
        for i, j, k in ti.ndrange(self.n, self.n, self.n):
            idx = ti.Vector([i, j, k])
            offset = idx[axis] % (2 * half)
            phase = offset % half
            idx[axis] = idx[axis] - offset + phase
            r, im = src[idx[0], idx[1], idx[2], 0], src[idx[0], idx[1], idx[2], 1]
            idx[axis] += half
            vr, vi = src[idx[0], idx[1], idx[2], 0], src[idx[0], idx[1], idx[2], 1]
            angle = (
                ti.cast(direction, self.real)
                * math.pi
                * ti.cast(phase, self.real)
                / half
            )
            co, si = ti.cos(angle), ti.sin(angle)
            sign = ti.cast(1, self.real)
            if offset >= half:
                sign = -1
            dst[i, j, k, 0] = r + sign * (co * vr - si * vi)
            dst[i, j, k, 1] = im + sign * (si * vr + co * vi)

    def _fft(self, src, scratch, direction):
        for axis in range(3):
            self._bit_reverse(src, scratch, axis)
            src, scratch = scratch, src
            for stage in range(self.logn):
                self._butterfly(src, scratch, axis, 1 << stage, direction)
                src, scratch = scratch, src
        return src, scratch

    @ti.kernel
    def _spectral_solve(
        self, spectrum: ti.types.ndarray(ndim=4), inverse: ti.types.ndarray(ndim=3)
    ):
        for i, j, k in ti.ndrange(self.n, self.n, self.n):
            for a in ti.static(range(2)):
                spectrum[i, j, k, a] *= inverse[i, j, k]

    @ti.kernel
    def _from_complex(
        self,
        src: ti.types.ndarray(ndim=4),
        phi: ti.types.ndarray(ndim=3),
        field: ti.types.ndarray(ndim=4),
    ):
        for i, j, k in ti.ndrange(self.n, self.n, self.n):
            phi[i, j, k] = src[i, j, k, 0] / self.n**3
            idx = ti.Vector([i, j, k])
            for a in ti.static(range(3)):
                original = idx[a]
                idx[a] = (original + 1) % self.n
                plus = src[idx[0], idx[1], idx[2], 0]
                idx[a] = (original - 1 + self.n) % self.n
                minus = src[idx[0], idx[1], idx[2], 0]
                idx[a] = original
                field[i, j, k, a] = (minus - plus) / (2 * self.dx * self.n**3)

    def poisson(self, rho):
        spectrum = self.empty((self.n,) * 3 + (2,))
        scratch = self.empty((self.n,) * 3 + (2,))
        self._to_complex(rho, spectrum)
        spectrum, scratch = self._fft(spectrum, scratch, -1)
        self._spectral_solve(spectrum, self.inverse)
        spectrum, scratch = self._fft(spectrum, scratch, 1)
        phi, field = self.empty((self.n,) * 3), self.empty((self.n,) * 3 + (3,))
        self._from_complex(spectrum, phi, field)
        return phi, field
