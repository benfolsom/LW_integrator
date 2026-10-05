"""Complete GPU seam with explicit host geometry and bounded Green setup.

Mesh Green coefficients and virtual-node coefficient tiles are constructed in
host float64, then uploaded. Particle deposition, current, gather, push, FFTs,
and charge-weighted node reductions execute on the selected device. This
avoids float32 cancellation in the integrated-cell Green primitive. No force
or particle-kernel CPU fallback is used.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import kernels
from .backend import green_mesh


class GPUBackend:
    """Shared array orchestration; subclasses supply allocation and dispatch."""

    xp: Any
    dtype: str
    name: str

    def __init__(self) -> None:
        self._key: tuple | None = None
        self._spectra: dict[int | None, Any] = {}

    def array(self, value: Any, dtype: str | None = None) -> Any:
        raise NotImplementedError

    def to_host(self, value: Any) -> np.ndarray:
        raise NotImplementedError

    def synchronize(self, *values: Any) -> None:
        raise NotImplementedError

    def _run(
        self,
        name: str,
        inputs: list[Any],
        shapes: list[Any],
        count: int,
        zero: bool = False,
    ) -> list[Any]:
        raise NotImplementedError

    def _coordinates(self, value: Any, shape: tuple[int, ...], order: int) -> Any:
        if order not in (1, 2):
            raise ValueError("shape order must be CIC(1) or TSC(2)")
        x = self.array(value)
        if x.ndim != 2 or x.shape[1] != 3:
            raise ValueError("coordinates must have shape (particles, 3)")
        base = self.xp.floor(x if order == 1 else x + 0.5) - (order == 2)
        # Validate AFTER float32 conversion as well as Grid's host checks.
        valid = self.xp.all(
            self.xp.isfinite(x) & (base >= 0) & (base + order < self.array(shape))
        )
        if not bool(self.to_host(valid)):
            raise ValueError(
                "particle shape extends outside local grid after precision conversion"
            )
        return x

    def _charges(self, q: Any, count: int) -> Any:
        q = self.array(q)
        if q.shape != (count,) or not bool(
            self.to_host(self.xp.all(self.xp.isfinite(q)))
        ):
            raise ValueError("source charges must be finite and match coordinates")
        return q

    def deposit(
        self, coordinates: Any, charge: Any, shape: tuple[int, ...], order: int
    ) -> Any:
        x = self._coordinates(coordinates, shape, order)
        q = self._charges(charge, len(x))
        meta = self.array([*shape, order], "int32")
        high, low = self._run("deposit", [x, q, meta], [shape, shape], len(x), True)
        return high + low

    def current(
        self,
        old: Any,
        new: Any,
        charge: Any,
        shape: tuple[int, ...],
        spacing: Any,
        dt: float,
        order: int,
    ) -> Any:
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("current deposition needs finite positive dt")
        spacing = np.asarray(self.to_host(spacing), dtype=float)
        if (
            spacing.shape != (3,)
            or not np.all(np.isfinite(spacing))
            or np.any(spacing <= 0)
        ):
            raise ValueError("current needs positive finite spacings")
        a, b = (self._coordinates(v, shape, order) for v in (old, new))
        if a.shape != b.shape:
            raise ValueError("current endpoints must match")
        q = self._charges(charge, len(a))
        shapes = [
            (shape[0] + 1, shape[1], shape[2]),
            (shape[0], shape[1] + 1, shape[2]),
            (shape[0], shape[1], shape[2] + 1),
        ]
        outputs = self._run(
            "current",
            [a, b, q, self.array([*shape, order], "int32"), self.array([*spacing, dt])],
            [s for s in shapes for _ in range(2)],
            len(a),
            True,
        )
        return tuple(outputs[2 * i] + outputs[2 * i + 1] for i in range(3))

    def gather(self, coordinates: Any, field: Any, order: int) -> Any:
        field = self.array(field)
        if field.ndim != 4:
            raise ValueError("field must have three grid axes and a component axis")
        x = self._coordinates(coordinates, field.shape[:3], order)
        meta = self.array([*field.shape[:3], order, field.shape[3]], "int32")
        return self._run(
            "gather", [x, field, meta], [(len(x), field.shape[3])], len(x)
        )[0]

    def push(
        self, u: Any, electric: Any, magnetic: Any, charge_mass: Any, dt: float
    ) -> Any:
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Boris push needs finite positive dt")
        u, e, b, qm = (self.array(v) for v in (u, electric, magnetic, charge_mass))
        if (
            u.ndim != 2
            or u.shape[1] != 3
            or e.shape != u.shape
            or b.shape != u.shape
            or qm.shape != (len(u),)
        ):
            raise ValueError("Boris inputs must have matching particle shapes")
        return self._run("push", [u, e, b, qm, self.array([dt])], [u.shape], len(u))[0]

    def _spectrum(
        self, shape: tuple[int, ...], spacing: Any, component: int | None
    ) -> Any:
        key = (tuple(shape), tuple(spacing))
        if key != self._key:
            self._spectra.clear()
            self._key = key
        if component not in self._spectra:
            mesh = self.array(green_mesh(shape, spacing, component))
            self._spectra[component] = self.xp.fft.rfftn(mesh)
            # Materialize each spectrum and release the large host setup array.
            self.synchronize(self._spectra[component])
        return self._spectra[component]

    def solve(
        self, charge: Any, spacing: Any, potential: bool = False
    ) -> tuple[Any, Any]:
        charge = self.array(charge)
        spacing = np.asarray(self.to_host(spacing), dtype=float)
        if (
            charge.ndim != 3
            or spacing.shape != (3,)
            or not np.all(np.isfinite(spacing))
            or np.any(spacing <= 0)
        ):
            raise ValueError(
                "solve needs a 3D charge mesh and positive finite spacings"
            )
        shape = charge.shape
        extended = tuple(2 * n for n in shape)
        crop = tuple(slice(0, n) for n in shape)
        axes = tuple(range(len(extended)))
        spectrum = self.xp.fft.rfftn(charge, s=extended, axes=axes)
        field = self.xp.stack(
            [
                self.xp.fft.irfftn(
                    spectrum * self._spectrum(shape, spacing, a), s=extended, axes=axes
                )[crop]
                for a in range(3)
            ],
            axis=-1,
        )
        phi = None
        if potential:
            phi = self.xp.fft.irfftn(
                spectrum * self._spectrum(shape, spacing, None), s=extended, axes=axes
            )[crop]
        self.synchronize(field, *([phi] if phi is not None else []))
        return field, phi

    def _reduce_nodes(self, table: Any, q: Any) -> Any:
        n, ns, _ = table.shape
        return self._run(
            "nodes",
            [self.array(table), self.array(q), self.array([ns], "int32")],
            [(n, 3)],
            n,
        )[0]

    def nodes(
        self, nodes: Any, source_indices: Any, source_charge: Any, spacing: Any
    ) -> Any:
        nodes = np.asarray(self.to_host(nodes))
        indices = np.asarray(self.to_host(source_indices))
        spacing = np.asarray(self.to_host(spacing), dtype=float)
        if (
            nodes.ndim != 2
            or nodes.shape[1] != 3
            or indices.ndim != 2
            or indices.shape[1] != 3
        ):
            raise ValueError("nodes and source_indices must have shape (count, 3)")
        if (
            not np.all(np.isfinite(nodes))
            or not np.all(np.isfinite(indices))
            or np.any(nodes != np.floor(nodes))
            or np.any(indices != np.floor(indices))
        ):
            raise ValueError("virtual nodes and source indices must be finite integers")
        if (
            spacing.shape != (3,)
            or not np.all(np.isfinite(spacing))
            or np.any(spacing <= 0)
        ):
            raise ValueError("nodes need positive finite spacings")
        q = self._charges(source_charge, len(indices))
        # At most 131072 pair coefficients in a host/device tile. Serial
        # compensated GPU reduction per node; cost remains O(nodes * sources).
        results = []
        node_batch = max(1, min(16, 131072 // max(1, len(indices))))
        for start in range(0, len(nodes), node_batch):
            part = nodes[start : start + node_batch]
            total = self.xp.zeros((len(part), 3), dtype=getattr(self.xp, self.dtype))
            correction = self.xp.zeros_like(total)
            for j in range(0, len(indices), 131072 // len(part)):
                idx = indices[j : j + 131072 // len(part)]
                displacement = (part[:, None, :] - idx[None, :, :]) * spacing
                coefficients = np.stack(
                    [
                        kernels.integrated_green(
                            tuple(displacement[..., a] for a in range(3)), spacing, c
                        )
                        for c in range(3)
                    ],
                    axis=-1,
                )
                weights = self.xp.broadcast_to(
                    q[j : j + len(idx)], (len(part), len(idx))
                )
                value = self._reduce_nodes(coefficients, weights)
                y = value - correction
                updated = total + y
                correction = (updated - total) - y
                total = updated
                self.synchronize(total, correction)
            results.append(total)
        return (
            self.xp.concatenate(results, axis=0)
            if results
            else self.xp.zeros((0, 3), dtype=getattr(self.xp, self.dtype))
        )

    def gather_open(
        self, coordinates: Any, field: Any, charge: Any, spacing: Any, order: int
    ) -> Any:
        if order not in (1, 2):
            raise ValueError("shape order must be CIC(1) or TSC(2)")
        coordinates = np.asarray(self.to_host(coordinates), dtype=float)
        if (
            coordinates.ndim != 2
            or coordinates.shape[1] != 3
            or not np.all(np.isfinite(coordinates))
        ):
            raise ValueError("coordinates must be finite with shape (particles, 3)")
        # Use the selected precision for the same stencil/inside decision.
        coordinates = coordinates.astype(self.dtype)
        base = np.floor(coordinates) if order == 1 else np.floor(coordinates + 0.5) - 1
        inside = np.all((base >= 0) & (base + order < charge.shape), axis=1)
        pieces: list[Any] = []
        owners: list[int] = []
        if np.any(inside):
            pieces.append(self.gather(coordinates[inside], field, order))
            owners.extend(np.flatnonzero(inside))
        outside = np.flatnonzero(~inside)
        if len(outside):
            host_charge = self.to_host(charge)
            indices = np.argwhere(host_charge != 0)
            weights = self.array(charge)[
                tuple(self.array(indices[:, a], "int32") for a in range(3))
            ]
            # Bound virtual-node work and avoid a per-observer Python dispatch.
            for start in range(0, len(outside), 32):
                batch = outside[start : start + 32]
                virtual, factors = [], []
                for owner in batch:
                    stencils = [
                        kernels.stencil(float(coordinates[owner, a]), order)
                        for a in range(3)
                    ]
                    for a in range(order + 1):
                        for b in range(order + 1):
                            for c in range(order + 1):
                                virtual.append(
                                    [
                                        stencils[0][0] + a,
                                        stencils[1][0] + b,
                                        stencils[2][0] + c,
                                    ]
                                )
                                factors.append(
                                    stencils[0][1][a]
                                    * stencils[1][1][b]
                                    * stencils[2][1][c]
                                )
                evaluated = self.nodes(np.asarray(virtual), indices, weights, spacing)
                support = (order + 1) ** 3
                pieces.append(
                    self._reduce_nodes(
                        evaluated.reshape(len(batch), support, 3),
                        np.asarray(factors).reshape(len(batch), support),
                    )
                )
                owners.extend(batch)
        if not pieces:
            return self.xp.zeros((0, 3), dtype=getattr(self.xp, self.dtype))
        inverse = np.argsort(owners)
        return self.xp.concatenate(pieces, axis=0)[self.array(inverse, "int32")]
