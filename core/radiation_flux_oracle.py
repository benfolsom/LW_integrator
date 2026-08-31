"""Independent electromagnetic flux accounting on a spherical surface.

The routines in this module are diagnostics.  They do not apply a force or
torque to a particle.  Instead they evaluate how much electromagnetic energy,
linear momentum, and angular momentum crosses a sphere.  This provides an
independent conservation check for present and future radiation-reaction
models.

All formulas use the solver's native scaled-Gaussian units.  The sampled
electric and magnetic fields therefore enter the Gaussian Poynting vector and
Maxwell stress without an SI conversion or an ``epsilon_0`` factor.

At finite radius the result includes transport associated with the bound near
field as well as radiation.  A radiation claim consequently requires radius
convergence while comparing the same retarded source-time interval.  The
reported angular-momentum quantity is specifically the symmetric
stress-tensor flux; it is not silently identified with a purely radiative
Noether flux.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Sequence, cast

import numpy as np

from .constants import C_MMNS
from .retarded_dipole_fields import evaluate_retarded_dipole_field_gradient_native
from .retarded_fields import (
    ObserverEvent,
    TrajectoryHistory,
    evaluate_retarded_charge_field_native,
)

_FOUR_PI = 4.0 * np.pi


def _readonly_array(
    value: np.ndarray, *, shape: tuple[int, ...], name: str
) -> np.ndarray:
    array = np.array(value, dtype=float, copy=True)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    array.flags.writeable = False
    return array


@dataclass(frozen=True)
class RadiationSphereQuadrature:
    """Directions and solid-angle weights for one spherical surface."""

    directions: np.ndarray
    solid_angle_weights: np.ndarray

    def __post_init__(self) -> None:
        directions = np.array(self.directions, dtype=float, copy=True)
        weights = np.array(self.solid_angle_weights, dtype=float, copy=True)
        if directions.ndim != 2 or directions.shape[1:] != (3,):
            raise ValueError("directions must have shape (sample_count, 3)")
        if weights.shape != (directions.shape[0],):
            raise ValueError("solid_angle_weights must have shape (sample_count,)")
        if directions.shape[0] == 0:
            raise ValueError("sphere quadrature must contain at least one sample")
        if not np.all(np.isfinite(directions)) or not np.all(np.isfinite(weights)):
            raise ValueError("sphere quadrature must contain only finite values")
        if np.any(weights <= 0.0):
            raise ValueError("solid-angle weights must be positive")
        norms = np.linalg.norm(directions, axis=1)
        if not np.allclose(norms, 1.0, rtol=0.0, atol=2.0e-14):
            raise ValueError("every sphere direction must have unit length")
        if not np.isclose(np.sum(weights), _FOUR_PI, rtol=0.0, atol=2.0e-13):
            raise ValueError("solid-angle weights must sum to 4 pi")
        directions.flags.writeable = False
        weights.flags.writeable = False
        object.__setattr__(self, "directions", directions)
        object.__setattr__(self, "solid_angle_weights", weights)

    @property
    def sample_count(self) -> int:
        """Return the number of angular samples."""

        return int(self.solid_angle_weights.size)


def gauss_legendre_sphere_quadrature(
    *, polar_order: int, azimuthal_order: int
) -> RadiationSphereQuadrature:
    """Return a tensor-product quadrature over solid angle.

    Gauss--Legendre nodes integrate in ``cos(theta)`` and equally spaced
    azimuthal nodes integrate around each latitude.  The poles are not sampled,
    which avoids an arbitrary coordinate singularity for transverse test
    fields.
    """

    polar = int(polar_order)
    azimuthal = int(azimuthal_order)
    if polar < 2:
        raise ValueError("polar_order must be at least 2")
    if azimuthal < 4:
        raise ValueError("azimuthal_order must be at least 4")
    cos_theta, polar_weights = np.polynomial.legendre.leggauss(polar)
    phi = _FOUR_PI * 0.5 * np.arange(azimuthal, dtype=float) / azimuthal
    sin_theta = np.sqrt(np.maximum(0.0, 1.0 - cos_theta**2))
    directions = np.stack(
        (
            np.repeat(sin_theta, azimuthal) * np.tile(np.cos(phi), polar),
            np.repeat(sin_theta, azimuthal) * np.tile(np.sin(phi), polar),
            np.repeat(cos_theta, azimuthal),
        ),
        axis=1,
    )
    weights = np.repeat(polar_weights * (_FOUR_PI * 0.5 / azimuthal), azimuthal)
    return RadiationSphereQuadrature(
        directions=directions,
        solid_angle_weights=weights,
    )


@dataclass(frozen=True)
class ElectromagneticFluxSector:
    """Integrated outward flux in one quadratic field sector.

    ``energy_rate_native`` has units of native energy per nanosecond,
    ``momentum_rate_native`` has units of native force, and
    ``angular_momentum_rate_native`` has units of native torque (equivalently
    native energy).
    """

    energy_rate_native: float
    momentum_rate_native: np.ndarray
    angular_momentum_rate_native: np.ndarray

    def __post_init__(self) -> None:
        energy_rate = float(self.energy_rate_native)
        if not np.isfinite(energy_rate):
            raise ValueError("energy_rate_native must be finite")
        object.__setattr__(self, "energy_rate_native", energy_rate)
        object.__setattr__(
            self,
            "momentum_rate_native",
            _readonly_array(
                self.momentum_rate_native,
                shape=(3,),
                name="momentum_rate_native",
            ),
        )
        object.__setattr__(
            self,
            "angular_momentum_rate_native",
            _readonly_array(
                self.angular_momentum_rate_native,
                shape=(3,),
                name="angular_momentum_rate_native",
            ),
        )


@dataclass(frozen=True)
class RadiationSphereFluxResult:
    """Charge, interference, dipole, and total flux through one sphere."""

    q_squared: ElectromagneticFluxSector
    q_mu_interference: ElectromagneticFluxSector
    mu_squared: ElectromagneticFluxSector
    total: ElectromagneticFluxSector
    observation_time_ns: float
    sphere_center_mm: np.ndarray
    angular_momentum_origin_mm: np.ndarray
    radius_mm: float
    quadrature_sample_count: int
    maximum_charge_light_cone_residual_mm: float | None = None
    maximum_dipole_light_cone_residual_mm: float | None = None
    charge_retarded_time_range_ns: tuple[float, float] | None = None
    dipole_retarded_time_range_ns: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        time = float(self.observation_time_ns)
        radius = float(self.radius_mm)
        if not np.isfinite(time):
            raise ValueError("observation_time_ns must be finite")
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError("radius_mm must be finite and positive")
        if int(self.quadrature_sample_count) <= 0:
            raise ValueError("quadrature_sample_count must be positive")
        for name in (
            "maximum_charge_light_cone_residual_mm",
            "maximum_dipole_light_cone_residual_mm",
        ):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or value < 0.0):
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in (
            "charge_retarded_time_range_ns",
            "dipole_retarded_time_range_ns",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            if len(value) != 2 or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must contain two finite values")
            if value[0] > value[1]:
                raise ValueError(f"{name} must be ordered from minimum to maximum")
        object.__setattr__(self, "observation_time_ns", time)
        object.__setattr__(self, "radius_mm", radius)
        object.__setattr__(
            self,
            "sphere_center_mm",
            _readonly_array(self.sphere_center_mm, shape=(3,), name="sphere_center_mm"),
        )
        object.__setattr__(
            self,
            "angular_momentum_origin_mm",
            _readonly_array(
                self.angular_momentum_origin_mm,
                shape=(3,),
                name="angular_momentum_origin_mm",
            ),
        )


@dataclass(frozen=True)
class IntegratedElectromagneticFluxSector:
    """Time-integrated transport in one quadratic field sector.

    ``energy_native`` is the energy crossing the sphere,
    ``momentum_native`` is the transported linear momentum, and
    ``angular_momentum_native`` is the transported angular momentum over the
    observation interval.
    """

    energy_native: float
    momentum_native: np.ndarray
    angular_momentum_native: np.ndarray

    def __post_init__(self) -> None:
        energy = float(self.energy_native)
        if not np.isfinite(energy):
            raise ValueError("energy_native must be finite")
        object.__setattr__(self, "energy_native", energy)
        object.__setattr__(
            self,
            "momentum_native",
            _readonly_array(
                self.momentum_native,
                shape=(3,),
                name="momentum_native",
            ),
        )
        object.__setattr__(
            self,
            "angular_momentum_native",
            _readonly_array(
                self.angular_momentum_native,
                shape=(3,),
                name="angular_momentum_native",
            ),
        )


@dataclass(frozen=True)
class RadiationSphereFluxIntegralResult:
    """Trapezoidal time integral of flux through one fixed sphere.

    This is transported field energy and momentum, not a recoil impulse.  A
    comparison with particle work or impulse must separately include any
    change in bound (Schott-like) field energy and momentum.
    """

    q_squared: IntegratedElectromagneticFluxSector
    q_mu_interference: IntegratedElectromagneticFluxSector
    mu_squared: IntegratedElectromagneticFluxSector
    total: IntegratedElectromagneticFluxSector
    observation_time_interval_ns: tuple[float, float]
    sphere_center_mm: np.ndarray
    angular_momentum_origin_mm: np.ndarray
    radius_mm: float
    sample_count: int
    quadrature_sample_count: int
    maximum_charge_light_cone_residual_mm: float | None = None
    maximum_dipole_light_cone_residual_mm: float | None = None
    charge_retarded_time_envelope_ns: tuple[float, float] | None = None
    dipole_retarded_time_envelope_ns: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        interval = tuple(float(value) for value in self.observation_time_interval_ns)
        if len(interval) != 2 or not np.all(np.isfinite(interval)):
            raise ValueError(
                "observation_time_interval_ns must contain two finite values"
            )
        if interval[1] <= interval[0]:
            raise ValueError("observation_time_interval_ns must have positive duration")
        radius = float(self.radius_mm)
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError("radius_mm must be finite and positive")
        if int(self.sample_count) < 2:
            raise ValueError("sample_count must be at least two")
        if int(self.quadrature_sample_count) <= 0:
            raise ValueError("quadrature_sample_count must be positive")
        for name in (
            "maximum_charge_light_cone_residual_mm",
            "maximum_dipole_light_cone_residual_mm",
        ):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or value < 0.0):
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in (
            "charge_retarded_time_envelope_ns",
            "dipole_retarded_time_envelope_ns",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            if len(value) != 2 or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must contain two finite values")
            if value[0] > value[1]:
                raise ValueError(f"{name} must be ordered from minimum to maximum")
        object.__setattr__(self, "observation_time_interval_ns", interval)
        object.__setattr__(self, "radius_mm", radius)
        object.__setattr__(
            self,
            "sphere_center_mm",
            _readonly_array(self.sphere_center_mm, shape=(3,), name="sphere_center_mm"),
        )
        object.__setattr__(
            self,
            "angular_momentum_origin_mm",
            _readonly_array(
                self.angular_momentum_origin_mm,
                shape=(3,),
                name="angular_momentum_origin_mm",
            ),
        )


@dataclass(frozen=True)
class RadiationReactionBalanceResult:
    """Conservation residual for a proposed local self-reaction model.

    The sign convention is

    ``mechanical change + outward flux + bound-field change = residual``.

    A consistent model approaches zero residual under time, sphere-radius,
    angular-quadrature, and source-model refinement.  Angular momentum is
    optional because the current charge-only Medina model supplies bound
    energy and momentum but not a complete intrinsic-dipole self-torque.
    """

    outward_flux: IntegratedElectromagneticFluxSector
    mechanical_work_native: float
    mechanical_impulse_native: np.ndarray
    bound_field_energy_change_native: float
    bound_field_momentum_change_native: np.ndarray
    energy_residual_native: float
    momentum_residual_native: np.ndarray
    mechanical_angular_momentum_change_native: np.ndarray | None = None
    bound_field_angular_momentum_change_native: np.ndarray | None = None
    angular_momentum_residual_native: np.ndarray | None = None

    def __post_init__(self) -> None:
        for name in (
            "mechanical_work_native",
            "bound_field_energy_change_native",
            "energy_residual_native",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        for name in (
            "mechanical_impulse_native",
            "bound_field_momentum_change_native",
            "momentum_residual_native",
        ):
            object.__setattr__(
                self,
                name,
                _readonly_array(getattr(self, name), shape=(3,), name=name),
            )
        angular_names = (
            "mechanical_angular_momentum_change_native",
            "bound_field_angular_momentum_change_native",
            "angular_momentum_residual_native",
        )
        angular_values = [getattr(self, name) for name in angular_names]
        if any(value is None for value in angular_values) and not all(
            value is None for value in angular_values
        ):
            raise ValueError(
                "angular-momentum balance values must be all set or all None"
            )
        for name, value in zip(angular_names, angular_values):
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    _readonly_array(value, shape=(3,), name=name),
                )


def _field_samples(value: np.ndarray, *, sample_count: int, name: str) -> np.ndarray:
    return _readonly_array(value, shape=(sample_count, 3), name=name)


def _integrate_sector_native(
    *,
    first_electric: np.ndarray,
    first_magnetic: np.ndarray,
    second_electric: np.ndarray,
    second_magnetic: np.ndarray,
    directions: np.ndarray,
    surface_weights_mm2: np.ndarray,
    lever_arms_mm: np.ndarray,
    diagonal: bool,
) -> ElectromagneticFluxSector:
    if diagonal:
        poynting = C_MMNS * np.cross(first_electric, first_magnetic) / _FOUR_PI
        field_inner = np.sum(first_electric**2 + first_magnetic**2, axis=1)
        momentum_flux = (
            0.5 * field_inner[:, np.newaxis] * directions
            - first_electric
            * np.sum(first_electric * directions, axis=1)[:, np.newaxis]
            - first_magnetic
            * np.sum(first_magnetic * directions, axis=1)[:, np.newaxis]
        ) / _FOUR_PI
    else:
        poynting = (
            C_MMNS
            * (
                np.cross(first_electric, second_magnetic)
                + np.cross(second_electric, first_magnetic)
            )
            / _FOUR_PI
        )
        field_inner = np.sum(
            first_electric * second_electric + first_magnetic * second_magnetic,
            axis=1,
        )
        momentum_flux = (
            field_inner[:, np.newaxis] * directions
            - first_electric
            * np.sum(second_electric * directions, axis=1)[:, np.newaxis]
            - second_electric
            * np.sum(first_electric * directions, axis=1)[:, np.newaxis]
            - first_magnetic
            * np.sum(second_magnetic * directions, axis=1)[:, np.newaxis]
            - second_magnetic
            * np.sum(first_magnetic * directions, axis=1)[:, np.newaxis]
        ) / _FOUR_PI

    energy_rate = float(
        np.sum(surface_weights_mm2 * np.sum(poynting * directions, axis=1))
    )
    momentum_rate = np.sum(
        surface_weights_mm2[:, np.newaxis] * momentum_flux,
        axis=0,
    )
    angular_momentum_rate = np.sum(
        surface_weights_mm2[:, np.newaxis] * np.cross(lever_arms_mm, momentum_flux),
        axis=0,
    )
    return ElectromagneticFluxSector(
        energy_rate_native=energy_rate,
        momentum_rate_native=momentum_rate,
        angular_momentum_rate_native=angular_momentum_rate,
    )


def _sum_sectors(*sectors: ElectromagneticFluxSector) -> ElectromagneticFluxSector:
    return ElectromagneticFluxSector(
        energy_rate_native=sum(sector.energy_rate_native for sector in sectors),
        momentum_rate_native=np.sum(
            [sector.momentum_rate_native for sector in sectors], axis=0
        ),
        angular_momentum_rate_native=np.sum(
            [sector.angular_momentum_rate_native for sector in sectors], axis=0
        ),
    )


def _sum_integrated_sectors(
    *sectors: IntegratedElectromagneticFluxSector,
) -> IntegratedElectromagneticFluxSector:
    return IntegratedElectromagneticFluxSector(
        energy_native=sum(sector.energy_native for sector in sectors),
        momentum_native=np.sum([sector.momentum_native for sector in sectors], axis=0),
        angular_momentum_native=np.sum(
            [sector.angular_momentum_native for sector in sectors], axis=0
        ),
    )


def _trapezoidal_integral(values: np.ndarray, times_ns: np.ndarray) -> np.ndarray:
    intervals = np.diff(times_ns)
    reshape = (intervals.size,) + (1,) * (values.ndim - 1)
    return np.asarray(
        np.sum(
            0.5 * (values[:-1] + values[1:]) * intervals.reshape(reshape),
            axis=0,
        ),
        dtype=float,
    )


def _integrate_sector_history(
    sectors: Sequence[ElectromagneticFluxSector],
    times_ns: np.ndarray,
) -> IntegratedElectromagneticFluxSector:
    energy_rates = np.asarray(
        [sector.energy_rate_native for sector in sectors], dtype=float
    )
    momentum_rates = np.asarray(
        [sector.momentum_rate_native for sector in sectors], dtype=float
    )
    angular_rates = np.asarray(
        [sector.angular_momentum_rate_native for sector in sectors], dtype=float
    )
    return IntegratedElectromagneticFluxSector(
        energy_native=float(_trapezoidal_integral(energy_rates, times_ns)),
        momentum_native=_trapezoidal_integral(momentum_rates, times_ns),
        angular_momentum_native=_trapezoidal_integral(angular_rates, times_ns),
    )


def _retarded_time_envelope(
    samples: Sequence[RadiationSphereFluxResult],
    attribute: str,
) -> tuple[float, float] | None:
    ranges = [getattr(sample, attribute) for sample in samples]
    if any(value is None for value in ranges):
        return None
    complete_ranges = cast(list[tuple[float, float]], ranges)
    return (
        min(value[0] for value in complete_ranges),
        max(value[1] for value in complete_ranges),
    )


def integrate_radiation_sphere_flux_history_native(
    samples: Sequence[RadiationSphereFluxResult],
) -> RadiationSphereFluxIntegralResult:
    """Integrate an ordered flux history over observation time.

    Samples must describe the same fixed sphere, angular quadrature size, and
    angular-momentum origin.  Observation times may be irregular but must be
    strictly increasing.  Radius-comparison studies must shift observation
    windows so that they cover the same retarded source-time interval.
    """

    history = tuple(samples)
    if len(history) < 2:
        raise ValueError("at least two flux samples are required")
    first = history[0]
    times = np.asarray([sample.observation_time_ns for sample in history], dtype=float)
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0.0):
        raise ValueError("flux sample observation times must be finite and increasing")
    for sample in history[1:]:
        if sample.radius_mm != first.radius_mm:
            raise ValueError("all flux samples must use the same sphere radius")
        if not np.array_equal(sample.sphere_center_mm, first.sphere_center_mm):
            raise ValueError("all flux samples must use the same sphere center")
        if not np.array_equal(
            sample.angular_momentum_origin_mm,
            first.angular_momentum_origin_mm,
        ):
            raise ValueError(
                "all flux samples must use the same angular-momentum origin"
            )
        if sample.quadrature_sample_count != first.quadrature_sample_count:
            raise ValueError("all flux samples must use the same quadrature size")

    q_squared = _integrate_sector_history(
        [sample.q_squared for sample in history], times
    )
    q_mu = _integrate_sector_history(
        [sample.q_mu_interference for sample in history], times
    )
    mu_squared = _integrate_sector_history(
        [sample.mu_squared for sample in history], times
    )
    charge_residual = _maximum_absolute_finite(
        np.asarray(
            [
                sample.maximum_charge_light_cone_residual_mm
                for sample in history
                if sample.maximum_charge_light_cone_residual_mm is not None
            ],
            dtype=float,
        )
    )
    dipole_residual = _maximum_absolute_finite(
        np.asarray(
            [
                sample.maximum_dipole_light_cone_residual_mm
                for sample in history
                if sample.maximum_dipole_light_cone_residual_mm is not None
            ],
            dtype=float,
        )
    )
    return RadiationSphereFluxIntegralResult(
        q_squared=q_squared,
        q_mu_interference=q_mu,
        mu_squared=mu_squared,
        total=_sum_integrated_sectors(q_squared, q_mu, mu_squared),
        observation_time_interval_ns=(float(times[0]), float(times[-1])),
        sphere_center_mm=first.sphere_center_mm,
        angular_momentum_origin_mm=first.angular_momentum_origin_mm,
        radius_mm=first.radius_mm,
        sample_count=len(history),
        quadrature_sample_count=first.quadrature_sample_count,
        maximum_charge_light_cone_residual_mm=charge_residual,
        maximum_dipole_light_cone_residual_mm=dipole_residual,
        charge_retarded_time_envelope_ns=_retarded_time_envelope(
            history, "charge_retarded_time_range_ns"
        ),
        dipole_retarded_time_envelope_ns=_retarded_time_envelope(
            history, "dipole_retarded_time_range_ns"
        ),
    )


def evaluate_radiation_reaction_balance_native(
    *,
    outward_flux: IntegratedElectromagneticFluxSector,
    mechanical_work_native: float,
    mechanical_impulse_native: Sequence[float],
    bound_field_energy_change_native: float,
    bound_field_momentum_change_native: Sequence[float],
    mechanical_angular_momentum_change_native: Sequence[float] | None = None,
    bound_field_angular_momentum_change_native: Sequence[float] | None = None,
) -> RadiationReactionBalanceResult:
    """Compare a local reaction ledger with independently measured flux.

    This function performs accounting only.  It does not infer a missing
    bound-field term, manufacture a recoil force, or decide that a residual is
    sufficiently small.  Those are model- and convergence-study decisions.
    """

    mechanical_work = float(mechanical_work_native)
    bound_energy_change = float(bound_field_energy_change_native)
    if not np.isfinite(mechanical_work) or not np.isfinite(bound_energy_change):
        raise ValueError("energy balance inputs must be finite")
    mechanical_impulse = _readonly_array(
        np.asarray(mechanical_impulse_native, dtype=float),
        shape=(3,),
        name="mechanical_impulse_native",
    )
    bound_momentum_change = _readonly_array(
        np.asarray(bound_field_momentum_change_native, dtype=float),
        shape=(3,),
        name="bound_field_momentum_change_native",
    )

    if (mechanical_angular_momentum_change_native is None) != (
        bound_field_angular_momentum_change_native is None
    ):
        raise ValueError(
            "mechanical and bound angular-momentum changes must be supplied together"
        )
    mechanical_angular: np.ndarray | None = None
    bound_angular: np.ndarray | None = None
    angular_residual: np.ndarray | None = None
    if mechanical_angular_momentum_change_native is not None:
        mechanical_angular = _readonly_array(
            np.asarray(mechanical_angular_momentum_change_native, dtype=float),
            shape=(3,),
            name="mechanical_angular_momentum_change_native",
        )
        bound_angular = _readonly_array(
            np.asarray(bound_field_angular_momentum_change_native, dtype=float),
            shape=(3,),
            name="bound_field_angular_momentum_change_native",
        )
        angular_residual = (
            mechanical_angular + outward_flux.angular_momentum_native + bound_angular
        )

    return RadiationReactionBalanceResult(
        outward_flux=outward_flux,
        mechanical_work_native=mechanical_work,
        mechanical_impulse_native=mechanical_impulse,
        bound_field_energy_change_native=bound_energy_change,
        bound_field_momentum_change_native=bound_momentum_change,
        energy_residual_native=(
            mechanical_work + outward_flux.energy_native + bound_energy_change
        ),
        momentum_residual_native=(
            mechanical_impulse + outward_flux.momentum_native + bound_momentum_change
        ),
        mechanical_angular_momentum_change_native=mechanical_angular,
        bound_field_angular_momentum_change_native=bound_angular,
        angular_momentum_residual_native=angular_residual,
    )


def integrate_radiation_sphere_flux_native(
    *,
    quadrature: RadiationSphereQuadrature,
    radius_mm: float,
    charge_electric_field_native: np.ndarray,
    charge_magnetic_field_native: np.ndarray,
    dipole_electric_field_native: np.ndarray,
    dipole_magnetic_field_native: np.ndarray,
    observation_time_ns: float = 0.0,
    sphere_center_mm: Sequence[float] = (0.0, 0.0, 0.0),
    angular_momentum_origin_mm: Sequence[float] | None = None,
) -> RadiationSphereFluxResult:
    """Integrate Maxwell energy, momentum, and angular-momentum flux.

    The result is split into the charge-only ``q_squared`` sector, the signed
    charge--dipole interference sector, and the dipole-only ``mu_squared``
    sector.  This split is algebraic: their sum is the flux of the total field.
    """

    radius = float(radius_mm)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius_mm must be finite and positive")
    center = np.asarray(sphere_center_mm, dtype=float)
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError("sphere_center_mm must contain three finite values")
    if angular_momentum_origin_mm is None:
        origin = center
    else:
        origin = np.asarray(angular_momentum_origin_mm, dtype=float)
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError("angular_momentum_origin_mm must contain three finite values")

    count = quadrature.sample_count
    charge_electric = _field_samples(
        charge_electric_field_native,
        sample_count=count,
        name="charge_electric_field_native",
    )
    charge_magnetic = _field_samples(
        charge_magnetic_field_native,
        sample_count=count,
        name="charge_magnetic_field_native",
    )
    dipole_electric = _field_samples(
        dipole_electric_field_native,
        sample_count=count,
        name="dipole_electric_field_native",
    )
    dipole_magnetic = _field_samples(
        dipole_magnetic_field_native,
        sample_count=count,
        name="dipole_magnetic_field_native",
    )
    surface_weights = radius**2 * quadrature.solid_angle_weights
    sample_positions = center[np.newaxis, :] + radius * quadrature.directions
    lever_arms = sample_positions - origin[np.newaxis, :]

    common = {
        "directions": quadrature.directions,
        "surface_weights_mm2": surface_weights,
        "lever_arms_mm": lever_arms,
    }
    q_squared = _integrate_sector_native(
        first_electric=charge_electric,
        first_magnetic=charge_magnetic,
        second_electric=charge_electric,
        second_magnetic=charge_magnetic,
        diagonal=True,
        **common,
    )
    q_mu = _integrate_sector_native(
        first_electric=charge_electric,
        first_magnetic=charge_magnetic,
        second_electric=dipole_electric,
        second_magnetic=dipole_magnetic,
        diagonal=False,
        **common,
    )
    mu_squared = _integrate_sector_native(
        first_electric=dipole_electric,
        first_magnetic=dipole_magnetic,
        second_electric=dipole_electric,
        second_magnetic=dipole_magnetic,
        diagonal=True,
        **common,
    )
    return RadiationSphereFluxResult(
        q_squared=q_squared,
        q_mu_interference=q_mu,
        mu_squared=mu_squared,
        total=_sum_sectors(q_squared, q_mu, mu_squared),
        observation_time_ns=float(observation_time_ns),
        sphere_center_mm=center,
        angular_momentum_origin_mm=origin,
        radius_mm=radius,
        quadrature_sample_count=count,
    )


def _maximum_absolute_finite(values: np.ndarray) -> float | None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    return float(np.max(np.abs(finite)))


def _finite_range(values: np.ndarray) -> tuple[float, float] | None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    return float(np.min(finite)), float(np.max(finite))


def evaluate_retarded_radiation_sphere_native(
    *,
    quadrature: RadiationSphereQuadrature,
    observation_time_ns: float,
    sphere_center_mm: Sequence[float],
    radius_mm: float,
    charge_history: TrajectoryHistory | None = None,
    dipole_history: TrajectoryHistory | None = None,
    angular_momentum_origin_mm: Sequence[float] | None = None,
    source_identities: Sequence[Hashable] | None = None,
    require_complete_history: bool = True,
    dipole_relative_step: float = 1.0e-3,
    dipole_minimum_step_mm: float = 1.0e-15,
    dipole_stencil_step_mm: float | None = None,
    dipole_minimum_separation_mm: float = 1.0e-15,
    root_tolerance_mm: float = 1.0e-21,
    max_root_iterations: int = 96,
    backend: str = "python",
) -> RadiationSphereFluxResult:
    """Sample retarded charge/dipole fields and integrate their sphere flux.

    Passing the same trajectory as ``charge_history`` and ``dipole_history``
    evaluates both source sectors.  Either sector may be omitted by passing
    ``None``.  The default Python provider is intentionally slow and
    transparent; faster exact-retarded backends are explicit opt-ins.
    """

    if charge_history is None and dipole_history is None:
        raise ValueError("at least one source history must be provided")
    time = float(observation_time_ns)
    center = np.asarray(sphere_center_mm, dtype=float)
    radius = float(radius_mm)
    if not np.isfinite(time):
        raise ValueError("observation_time_ns must be finite")
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError("sphere_center_mm must contain three finite values")
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius_mm must be finite and positive")

    count = quadrature.sample_count
    charge_electric = np.zeros((count, 3), dtype=float)
    charge_magnetic = np.zeros((count, 3), dtype=float)
    dipole_electric = np.zeros((count, 3), dtype=float)
    dipole_magnetic = np.zeros((count, 3), dtype=float)
    charge_residuals: list[np.ndarray] = []
    dipole_residuals: list[np.ndarray] = []
    charge_retarded_times: list[np.ndarray] = []
    dipole_retarded_times: list[np.ndarray] = []

    for sample_index, direction in enumerate(quadrature.directions):
        position = center + radius * direction
        event = ObserverEvent(
            time_ns=time,
            position_mm=cast(
                tuple[float, float, float], tuple(float(value) for value in position)
            ),
        )
        if charge_history is not None:
            charge = evaluate_retarded_charge_field_native(
                charge_history,
                event,
                require_complete_history=require_complete_history,
                root_tolerance_mm=root_tolerance_mm,
                max_root_iterations=max_root_iterations,
                backend=backend,
            )
            charge_electric[sample_index] = charge.electric_field_native
            charge_magnetic[sample_index] = charge.magnetic_field_native
            charge_residuals.append(charge.light_cone_residual_mm)
            charge_retarded_times.append(charge.retarded_time_ns)
        if dipole_history is not None:
            dipole = evaluate_retarded_dipole_field_gradient_native(
                dipole_history,
                event,
                source_identities=source_identities,
                require_complete_history=require_complete_history,
                relative_step=dipole_relative_step,
                minimum_step_mm=dipole_minimum_step_mm,
                stencil_step_mm=dipole_stencil_step_mm,
                minimum_separation_mm=dipole_minimum_separation_mm,
                root_tolerance_mm=root_tolerance_mm,
                max_root_iterations=max_root_iterations,
                backend=backend,
            )
            dipole_electric[sample_index] = dipole.electric_field_native
            dipole_magnetic[sample_index] = dipole.magnetic_field_native
            dipole_residuals.append(dipole.stencil_light_cone_residual_mm)
            dipole_retarded_times.append(dipole.hertz.retarded_time_ns)

    result = integrate_radiation_sphere_flux_native(
        quadrature=quadrature,
        radius_mm=radius,
        charge_electric_field_native=charge_electric,
        charge_magnetic_field_native=charge_magnetic,
        dipole_electric_field_native=dipole_electric,
        dipole_magnetic_field_native=dipole_magnetic,
        observation_time_ns=time,
        sphere_center_mm=cast(
            tuple[float, float, float], tuple(float(value) for value in center)
        ),
        angular_momentum_origin_mm=angular_momentum_origin_mm,
    )
    return RadiationSphereFluxResult(
        q_squared=result.q_squared,
        q_mu_interference=result.q_mu_interference,
        mu_squared=result.mu_squared,
        total=result.total,
        observation_time_ns=result.observation_time_ns,
        sphere_center_mm=result.sphere_center_mm,
        angular_momentum_origin_mm=result.angular_momentum_origin_mm,
        radius_mm=result.radius_mm,
        quadrature_sample_count=result.quadrature_sample_count,
        maximum_charge_light_cone_residual_mm=(
            _maximum_absolute_finite(np.concatenate(charge_residuals))
            if charge_residuals
            else None
        ),
        maximum_dipole_light_cone_residual_mm=(
            _maximum_absolute_finite(np.concatenate(dipole_residuals))
            if dipole_residuals
            else None
        ),
        charge_retarded_time_range_ns=(
            _finite_range(np.concatenate(charge_retarded_times))
            if charge_retarded_times
            else None
        ),
        dipole_retarded_time_range_ns=(
            _finite_range(np.concatenate(dipole_retarded_times))
            if dipole_retarded_times
            else None
        ),
    )


__all__ = [
    "ElectromagneticFluxSector",
    "IntegratedElectromagneticFluxSector",
    "RadiationReactionBalanceResult",
    "RadiationSphereFluxIntegralResult",
    "RadiationSphereFluxResult",
    "RadiationSphereQuadrature",
    "evaluate_radiation_reaction_balance_native",
    "evaluate_retarded_radiation_sphere_native",
    "gauss_legendre_sphere_quadrature",
    "integrate_radiation_sphere_flux_history_native",
    "integrate_radiation_sphere_flux_native",
]
