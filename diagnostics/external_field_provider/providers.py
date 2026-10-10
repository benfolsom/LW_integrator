"""Analytic toy channels, not fitted crystal or nanotube material models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.external_field_provider import ExternalFieldSample


def electrostatic_sample(phi, gradient, hessian):
    partial_e = np.zeros((4, 3))
    partial_e[1:] = -np.asarray(hessian).T
    return ExternalFieldSample(
        phi=phi,
        vector_potential=np.zeros(3),
        electric=-np.asarray(gradient),
        magnetic=np.zeros(3),
        partial_e=partial_e,
        partial_b=np.zeros((4, 3)),
    )


@dataclass(frozen=True)
class HarmonicChannel:
    """phi = k (x^2+y^2)/2; k in native field/mm, focusing for q*k>0."""

    curvature: float

    def __call__(self, *, position_mm, time_ns):
        x, y, _ = position_mm
        k = self.curvature
        return electrostatic_sample(
            0.5 * k * (x * x + y * y),
            np.array([k * x, k * y, 0.0]),
            np.diag([k, k, 0.0]),
        )


@dataclass(frozen=True)
class SmoothTube:
    """Smooth radial well with a C2 axial entrance and exit.

    phi = depth * [1-exp(-r^2/(2*radius^2))] * window(z).
    depth is native field*mm; radius, z_min, z_max, and ramp are mm.
    The quintic window has zero first and second derivatives at both joins.
    The axial fringe field and cross Hessian entries are included.
    """

    depth: float
    radius_mm: float
    z_min_mm: float
    z_max_mm: float
    ramp_mm: float

    def __post_init__(self):
        if (
            not np.all(
                np.isfinite(
                    [
                        self.depth,
                        self.radius_mm,
                        self.z_min_mm,
                        self.z_max_mm,
                        self.ramp_mm,
                    ]
                )
            )
            or self.radius_mm <= 0
            or self.ramp_mm <= 0
            or self.z_max_mm - self.z_min_mm < 2 * self.ramp_mm
        ):
            raise ValueError(
                "tube needs finite parameters, positive radius/ramp, and two nonoverlapping ramps"
            )

    @staticmethod
    def _ramp(u):
        if u <= 0:
            return 0.0, 0.0, 0.0
        if u >= 1:
            return 1.0, 0.0, 0.0
        return (
            u**3 * (10 - 15 * u + 6 * u * u),
            30 * u * u * (1 - u) ** 2,
            60 * u * (1 - u) * (1 - 2 * u),
        )

    def __call__(self, *, position_mm, time_ns):
        x, y, z = position_mm
        left = self._ramp((z - self.z_min_mm) / self.ramp_mm)
        right = self._ramp((self.z_max_mm - z) / self.ramp_mm)
        w = left[0] * right[0]
        wp = (left[1] * right[0] - left[0] * right[1]) / self.ramp_mm
        wpp = (
            left[2] * right[0] - 2 * left[1] * right[1] + left[0] * right[2]
        ) / self.ramp_mm**2
        xy = np.array([x, y])
        a2 = self.radius_mm**2
        exponential = np.exp(-np.dot(xy, xy) / (2 * a2))
        radial = self.depth * (-np.expm1(-np.dot(xy, xy) / (2 * a2)))
        grad = self.depth * exponential * xy / a2
        hess = self.depth * exponential * (np.eye(2) / a2 - np.outer(xy, xy) / a2**2)
        gradient = np.array([w * grad[0], w * grad[1], wp * radial])
        hessian = np.zeros((3, 3))
        hessian[:2, :2] = w * hess
        hessian[:2, 2] = wp * grad
        hessian[2, :2] = wp * grad
        hessian[2, 2] = wpp * radial
        return electrostatic_sample(w * radial, gradient, hessian)
