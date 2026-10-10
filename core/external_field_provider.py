"""Study-supplied smooth electromagnetic fields in the solver's native units.

Coordinates are (ct, x, y, z), all in mm. Fields E and B have native
force/charge units (B enters beta cross B); phi and A have field*mm units.
``partial_e[lambda, i]`` and ``partial_b[lambda, i]`` are first spacetime
partials per mm. For a static electrostatic potential E=-grad(phi), so
partial_e[1:].T is minus the complete spatial Hessian of phi.

A provider must be pure, finite, and at least C2 in its potentials, including
region entrances/exits. It may be called repeatedly on rejected trials. No
hard config windows or implicit numerical differentiation are applied.

The default non-exact route samples the step-start force and is first order in
dt for provider fields, including net energy errors after region exit. For
channeling and energy-loss claims, prefer the exact
``second_order_start_taylor_endpoint`` route with ``INERTIAL_PREHISTORY`` and
verify timestep convergence. The analytic Medina/LAD derivative does not
upgrade the underlying trajectory order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from .constants import C_MMNS
from .external_fields import (
    C_M_PER_S,
    electric_field_v_per_m_to_native,
    magnetic_field_tesla_to_native,
)


@dataclass(frozen=True)
class ExternalFieldSample:
    """One event's native potentials, fields, and first field derivatives.

    Potential values are for diagnostics; the existing external-field hook
    advances mechanical momentum without adding a new canonical gauge offset.
    Arrays are copied and made read-only to keep a stage's sample immutable.
    """

    phi: float
    vector_potential: np.ndarray
    electric: np.ndarray
    magnetic: np.ndarray
    partial_e: np.ndarray
    partial_b: np.ndarray

    def __post_init__(self) -> None:
        phi = np.asarray(self.phi, dtype=float)
        if phi.shape != () or not np.isfinite(phi):
            raise ValueError("phi must be a finite scalar")
        object.__setattr__(self, "phi", float(phi))
        for name, shape in (
            ("vector_potential", (3,)),
            ("electric", (3,)),
            ("magnetic", (3,)),
            ("partial_e", (4, 3)),
            ("partial_b", (4, 3)),
        ):
            value = np.array(getattr(self, name), dtype=float, copy=True)
            if value.shape != shape or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite with shape {shape}")
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    @classmethod
    def from_si(
        cls,
        *,
        phi_volts: float,
        vector_potential_t_m: Sequence[float],
        electric_v_m: Sequence[float],
        magnetic_t: Sequence[float],
        partial_e_si: np.ndarray,
        partial_b_si: np.ndarray,
    ) -> ExternalFieldSample:
        """Convert SI values; derivative rows are (t[s], x[m], y[m], z[m]).

        Electric derivative units are V/m/s and V/m^2; magnetic derivatives
        are T/s and T/m. Events passed to the callable still use mm and ns.
        """
        for name, value in (
            ("partial_e_si", partial_e_si),
            ("partial_b_si", partial_b_si),
        ):
            if np.shape(value) != (4, 3):
                raise ValueError(f"{name} must have shape (4, 3)")
        e_scale = electric_field_v_per_m_to_native(1.0)
        b_scale = magnetic_field_tesla_to_native(1.0)
        coordinate_scale = np.array([1 / C_M_PER_S, 1.0, 1.0, 1.0]) / 1000
        return cls(
            phi=phi_volts * e_scale * 1000,
            vector_potential=np.asarray(vector_potential_t_m) * b_scale * 1000,
            electric=np.asarray(electric_v_m) * e_scale,
            magnetic=np.asarray(magnetic_t) * b_scale,
            partial_e=np.asarray(partial_e_si) * coordinate_scale[:, None] * e_scale,
            partial_b=np.asarray(partial_b_si) * coordinate_scale[:, None] * b_scale,
        )


class ExternalFieldProvider(Protocol):
    """Callable receiving a native lab event, independent of particle species."""

    def __call__(
        self, *, position_mm: tuple[float, float, float], time_ns: float
    ) -> ExternalFieldSample: ...


def provider_force_derivative(
    sample: ExternalFieldSample,
    *,
    charge: float,
    beta: Sequence[float],
    beta_dot: Sequence[float],
) -> np.ndarray:
    """Complete lab-time d[q(E+beta cross B)]/dt, in native force/ns."""
    velocity = C_MMNS * np.asarray(beta)
    event_rate = np.concatenate(([C_MMNS], velocity))
    e_dot = event_rate @ sample.partial_e
    b_dot = event_rate @ sample.partial_b
    return np.asarray(
        charge * (e_dot + np.cross(beta_dot, sample.magnetic) + np.cross(beta, b_dot)),
        dtype=float,
    )
