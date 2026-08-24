"""Full-retarded finite-difference oracle for intrinsic magnetic dipoles.

This module constructs the ordinary Maxwell four-potential sourced by moving
point magnetic dipoles.  It is deliberately an *oracle*, not yet the fast
production path: the field and its spacetime gradient are commuting centered
finite differences of a complete retarded Hertz-tensor evaluation.  Every
stencil event independently solves every source light cone, so acceleration,
spin evolution, and the observer-event dependence of retarded time are not
silently frozen.

The covariant construction is

``M^(mu nu) = *[(u^mu m^nu - m^mu u^nu) / c]``

``H^(mu nu) = M^(mu nu) / ((R.u) / c)``

``A^mu = partial_nu H^(mu nu)``.

Here ``m`` is the boosted rest-frame magnetic-moment four-vector and ``R`` is
the future null displacement from the retarded source event.  Antisymmetry of
``H`` makes ``partial_mu A^mu = 0`` and corresponds to the conserved
magnetization current ``j^mu = c partial_nu M^(mu nu)``.  In the static rest
limit the convention gives ``A = m x r / r^3`` and the usual Gaussian point
dipole field.  The retarded potential and its static, oscillating, and moving
limits follow the Green-function constructions discussed by Sautbekov,
Nuclear Physics B 945, 114665 (2019), https://arxiv.org/abs/1806.07089, and
Heras, Phys. Rev. E 58, 5047 (1998),
https://doi.org/10.1103/PhysRevE.58.5047.

Conventions
-----------

* Native scaled-Gaussian units: amu, millimetres, nanoseconds, and native
  charge.  A magnetic moment has units ``native charge * mm``.
* Coordinates are ``x=(ct,x,y,z)`` with metric ``diag(+1,-1,-1,-1)``.
* ``partial_a[lambda, nu]`` is ``partial_lambda A^nu`` per millimetre.
* ``partial_f[lambda, mu, nu]`` is ``partial_lambda F^(mu nu)`` per
  millimetre.
* Source rest-spin histories are interpolated with a shared-slope cubic
  Hermite interpolant, which is C1 at history knots.  When every stored knot
  has the same spin magnitude, the interpolant is smoothly projected back to
  that magnitude; a fully polarized source therefore cannot acquire a
  step-size-dependent moment merely from Cartesian interpolation.

The point source is singular.  ``minimum_separation_mm`` is a strict numerical
guard, not a finite-size or softening model.  Dipole self-reaction and contact
terms are outside this provider.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from typing import Hashable, Sequence, cast

import numpy as np

from .constants import C_MMNS
from .magnetic_dipole import boost_rest_polarization
from .retarded_fields import (
    ObserverEvent,
    RetardedHistoryError,
    TrajectoryHistory,
    _extract_history,
    _history_constant,
    _history_matrix,
    _HistoryArrays,
    _prepare_source_history,
    _PreparedSourceHistory,
    _solve_retarded_sample,
    _source_terminated_before_light_cone,
    _validated_root_options,
)
from .rfs import (
    electromagnetic_field_tensor_native,
    fields_from_tensor_native,
    hodge_dual,
)

_DEFAULT_ROOT_TOLERANCE_MM = 1.0e-21
_DEFAULT_MAX_ROOT_ITERATIONS = 96
_DEFAULT_STENCIL_RELATIVE_STEP = 1.0e-3
_DEFAULT_STENCIL_MINIMUM_STEP_MM = 1.0e-15
_DEFAULT_MINIMUM_SEPARATION_MM = 1.0e-15
_CONTRAVARIANT_DERIVATIVE_SIGNS = np.array((1.0, -1.0, -1.0, -1.0))


class DipoleSourceSingularityError(ValueError):
    """Raised when an observer/stencil event approaches a point dipole too far."""


@dataclass(frozen=True)
class RetardedDipoleHertzResult:
    """Summed retarded Hertz tensor and center-event source diagnostics."""

    hertz_tensor: np.ndarray
    source_identities: tuple[Hashable, ...]
    retarded_time_ns: np.ndarray
    light_cone_residual_mm: np.ndarray
    separation_mm: np.ndarray
    valid_sources: np.ndarray


@dataclass(frozen=True)
class RetardedDipoleFieldGradientResult:
    """Ordinary dipole potential, field, and full spacetime field gradient.

    ``partial_a[lambda, nu]`` is the covariant coordinate derivative
    ``partial_lambda A^nu``.  It is returned explicitly for the integrator's
    charge-canonical momentum update; callers must not replace that update by
    an undocumented mechanical ``q F`` injection.

    ``stencil_offsets`` records integer multiples of ``stencil_step_mm`` for
    every Hertz evaluation used.  The matching retarded times make
    ``delta/2``, ``delta``, and ``2 delta`` convergence studies auditable.
    """

    four_potential: np.ndarray
    partial_a: np.ndarray
    electric_field_native: np.ndarray
    magnetic_field_native: np.ndarray
    field_tensor: np.ndarray
    partial_f: np.ndarray
    hertz: RetardedDipoleHertzResult
    stencil_step_mm: float
    stencil_offsets: np.ndarray
    stencil_retarded_time_ns: np.ndarray
    stencil_light_cone_residual_mm: np.ndarray
    lorenz_gauge_residual_per_mm: float


@dataclass(frozen=True)
class _PreparedDipoleSource:
    identity: Hashable
    worldline: _PreparedSourceHistory
    rest_spin: np.ndarray
    rest_spin_derivative_per_ns: np.ndarray
    preserved_rest_spin_magnitude: float | None
    magnetic_moment_native: float


@dataclass(frozen=True)
class _PreparedDipoleHistory:
    arrays: _HistoryArrays
    source_identities: tuple[Hashable, ...]
    sources: dict[int, _PreparedDipoleSource]


def _validate_source_identities(
    n_sources: int,
    source_identities: Sequence[Hashable] | None,
) -> tuple[Hashable, ...]:
    if source_identities is None:
        identities: tuple[Hashable, ...] = tuple(range(n_sources))
    else:
        identities = tuple(source_identities)
        if len(identities) != n_sources:
            raise ValueError("source_identities must match the particle count")
    try:
        unique_count = len(set(identities))
    except TypeError as exc:
        raise TypeError("source identities must be hashable") from exc
    if unique_count != len(identities):
        raise ValueError("source identities must be unique")
    return identities


def _source_spin_slopes_per_ns(spin: np.ndarray, time_ns: np.ndarray) -> np.ndarray:
    """Return shared Hermite slopes, giving a C1 interpolant at each knot."""

    count = int(time_ns.size)
    if count < 2:
        return cast(np.ndarray, np.zeros_like(spin))
    secants = np.diff(spin, axis=0) / np.diff(time_ns)[:, np.newaxis]
    if count == 2:
        return cast(np.ndarray, np.stack((secants[0], secants[0]), axis=0))

    slopes = np.empty_like(spin)
    slopes[0] = secants[0]
    slopes[-1] = secants[-1]
    previous_duration = np.diff(time_ns)[:-1]
    next_duration = np.diff(time_ns)[1:]
    denominator = previous_duration + next_duration
    slopes[1:-1] = (
        next_duration[:, np.newaxis] * secants[:-1]
        + previous_duration[:, np.newaxis] * secants[1:]
    ) / denominator[:, np.newaxis]
    return cast(np.ndarray, slopes)


def _prepare_dipole_history(
    history: TrajectoryHistory,
    *,
    source_identities: Sequence[Hashable] | None,
    observer_source_identity: Hashable | None,
    excluded_source_identities: Sequence[Hashable],
) -> _PreparedDipoleHistory:
    arrays = _extract_history(history)
    identities = _validate_source_identities(arrays.n_sources, source_identities)
    excluded = set(excluded_source_identities)
    if observer_source_identity is not None:
        excluded.add(observer_source_identity)

    moments = np.asarray(
        _history_constant(history, "magnetic_moment_native"), dtype=float
    )
    if moments.shape != (arrays.n_sources,) or not np.all(np.isfinite(moments)):
        raise ValueError(
            "magnetic_moment_native must contain one finite value per source"
        )
    try:
        active = np.asarray(
            _history_constant(history, "magnetic_dipole_active"), dtype=bool
        )
    except (AttributeError, KeyError):
        active = moments != 0.0
    if active.shape != (arrays.n_sources,):
        raise ValueError("magnetic_dipole_active must match the particle count")

    spin_components = tuple(
        np.asarray(_history_matrix(history, f"spin_{axis}"), dtype=float)
        for axis in "xyz"
    )
    if any(component.shape != arrays.time_ns.shape for component in spin_components):
        raise ValueError("source spin histories must share shape [steps, particles]")
    if not all(np.all(np.isfinite(component)) for component in spin_components):
        raise ValueError("source spin histories must contain only finite values")
    spin = np.stack(spin_components, axis=-1)

    sources: dict[int, _PreparedDipoleSource] = {}
    for source_index in range(arrays.n_sources):
        identity = identities[source_index]
        if (
            identity in excluded
            or not active[source_index]
            or moments[source_index] == 0.0
        ):
            continue
        worldline = _prepare_source_history(arrays, source_index)
        alive_count = int(worldline.time_ns.size)
        source_spin = spin[:alive_count, source_index]
        source_spin_norm = np.linalg.norm(source_spin, axis=1)
        preserved_magnitude = (
            float(source_spin_norm[0])
            if source_spin_norm.size
            and np.allclose(
                source_spin_norm,
                source_spin_norm[0],
                rtol=1.0e-10,
                atol=1.0e-12,
            )
            else None
        )
        sources[source_index] = _PreparedDipoleSource(
            identity=identity,
            worldline=worldline,
            rest_spin=source_spin,
            rest_spin_derivative_per_ns=_source_spin_slopes_per_ns(
                source_spin, worldline.time_ns
            ),
            preserved_rest_spin_magnitude=preserved_magnitude,
            magnetic_moment_native=float(moments[source_index]),
        )
    return _PreparedDipoleHistory(
        arrays=arrays,
        source_identities=identities,
        sources=sources,
    )


def _interpolate_rest_spin_c1(
    source: _PreparedDipoleSource, time_ns: float
) -> np.ndarray:
    times = source.worldline.time_ns
    if times.size == 0:
        raise RetardedHistoryError("dipole source has no alive history")
    if times.size == 1:
        return cast(np.ndarray, np.asarray(source.rest_spin[0], dtype=float))
    segment = int(np.searchsorted(times, time_ns, side="right") - 1)
    segment = min(max(segment, 0), int(times.size) - 2)
    duration = float(times[segment + 1] - times[segment])
    fraction = float(np.clip((time_ns - times[segment]) / duration, 0.0, 1.0))
    fraction_squared = fraction * fraction
    fraction_cubed = fraction_squared * fraction
    h00 = 2.0 * fraction_cubed - 3.0 * fraction_squared + 1.0
    h10 = fraction_cubed - 2.0 * fraction_squared + fraction
    h01 = -2.0 * fraction_cubed + 3.0 * fraction_squared
    h11 = fraction_cubed - fraction_squared
    interpolated = np.asarray(
        h00 * source.rest_spin[segment]
        + h10 * duration * source.rest_spin_derivative_per_ns[segment]
        + h01 * source.rest_spin[segment + 1]
        + h11 * duration * source.rest_spin_derivative_per_ns[segment + 1],
    )
    target_magnitude = source.preserved_rest_spin_magnitude
    if target_magnitude is None:
        return cast(np.ndarray, interpolated)
    if target_magnitude == 0.0:
        return cast(np.ndarray, np.zeros(3, dtype=float))
    interpolated_magnitude = float(np.linalg.norm(interpolated))
    if interpolated_magnitude <= 1.0e-15:
        raise RetardedHistoryError(
            "constant-magnitude source-spin interpolation crossed zero; "
            "reduce the source-history timestep"
        )
    return cast(
        np.ndarray,
        interpolated * (target_magnitude / interpolated_magnitude),
    )


def _moment_tensor_contravariant(
    *,
    magnetic_moment_native: float,
    rest_spin: np.ndarray,
    source_beta: np.ndarray,
) -> np.ndarray:
    beta_squared = float(source_beta @ source_beta)
    if beta_squared >= 1.0:
        raise ValueError("source beta magnitude must be less than one")
    gamma = 1.0 / np.sqrt(1.0 - beta_squared)
    four_velocity = gamma * C_MMNS * np.concatenate(((1.0,), source_beta))
    moment_four = magnetic_moment_native * boost_rest_polarization(
        rest_spin, source_beta
    )
    velocity_moment_wedge = (
        np.outer(four_velocity, moment_four) - np.outer(moment_four, four_velocity)
    ) / C_MMNS
    return cast(np.ndarray, hodge_dual(velocity_moment_wedge))


def _evaluate_prepared_hertz_tensor_native(
    prepared: _PreparedDipoleHistory,
    observer_event: ObserverEvent,
    *,
    require_complete_history: bool,
    minimum_separation_mm: float,
    root_tolerance_mm: float,
    max_root_iterations: int,
) -> RetardedDipoleHertzResult:
    arrays = prepared.arrays
    observer_time_ns = float(observer_event.time_ns)
    observer_position_mm = np.asarray(observer_event.position_mm, dtype=float)
    hertz_total: np.ndarray = np.zeros((4, 4), dtype=float)
    retarded_time_ns = np.full(arrays.n_sources, np.nan, dtype=float)
    residual_mm = np.full(arrays.n_sources, np.nan, dtype=float)
    separation_mm = np.full(arrays.n_sources, np.nan, dtype=float)
    valid_sources = np.zeros(arrays.n_sources, dtype=bool)
    missing_sources: list[int] = []

    for source_index, source in prepared.sources.items():
        sample = _solve_retarded_sample(
            source.worldline,
            observer_time_ns=observer_time_ns,
            observer_position_mm=observer_position_mm,
            root_tolerance_mm=root_tolerance_mm,
            max_root_iterations=max_root_iterations,
        )
        if sample is None:
            if _source_terminated_before_light_cone(
                source.worldline,
                observer_time_ns=observer_time_ns,
                observer_position_mm=observer_position_mm,
            ):
                continue
            missing_sources.append(source_index)
            continue
        if sample.separation_mm <= minimum_separation_mm:
            raise DipoleSourceSingularityError(
                "observer/stencil event is within minimum_separation_mm of "
                f"dipole source identity {source.identity!r}: "
                f"{sample.separation_mm:.17g} <= {minimum_separation_mm:.17g} mm"
            )

        beta_squared = float(sample.beta @ sample.beta)
        if beta_squared >= 1.0:
            raise ValueError("source beta magnitude must be less than one")
        gamma = 1.0 / np.sqrt(1.0 - beta_squared)
        separation_vector = observer_position_mm - sample.position_mm
        direction = separation_vector / sample.separation_mm
        kappa = 1.0 - float(direction @ sample.beta)
        if kappa <= 1.0e-14:
            raise DipoleSourceSingularityError(
                "retarded dipole field is singular because 1 - n.beta is too small"
            )
        invariant_retarded_distance_mm = gamma * sample.separation_mm * kappa
        rest_spin = _interpolate_rest_spin_c1(source, sample.time_ns)
        moment_tensor = _moment_tensor_contravariant(
            magnetic_moment_native=source.magnetic_moment_native,
            rest_spin=rest_spin,
            source_beta=sample.beta,
        )
        hertz_total += moment_tensor / invariant_retarded_distance_mm
        retarded_time_ns[source_index] = sample.time_ns
        residual_mm[source_index] = sample.residual_mm
        separation_mm[source_index] = sample.separation_mm
        valid_sources[source_index] = True

    if require_complete_history and missing_sources:
        missing_identities = [
            prepared.source_identities[index] for index in missing_sources
        ]
        raise RetardedHistoryError(
            "source history does not bracket the observer light cone for dipole "
            f"source identities {missing_identities!r}"
        )
    return RetardedDipoleHertzResult(
        hertz_tensor=hertz_total,
        source_identities=prepared.source_identities,
        retarded_time_ns=retarded_time_ns,
        light_cone_residual_mm=residual_mm,
        separation_mm=separation_mm,
        valid_sources=valid_sources,
    )


def _validated_oracle_options(
    *,
    relative_step: float,
    minimum_step_mm: float,
    stencil_step_mm: float | None,
    minimum_separation_mm: float,
) -> tuple[float, float, float | None, float]:
    relative = float(relative_step)
    minimum_step = float(minimum_step_mm)
    minimum_separation = float(minimum_separation_mm)
    if not np.isfinite(relative) or relative <= 0.0 or relative >= 0.05:
        raise ValueError("relative_step must be finite and in (0, 0.05)")
    if not np.isfinite(minimum_step) or minimum_step <= 0.0:
        raise ValueError("minimum_step_mm must be finite and positive")
    if not np.isfinite(minimum_separation) or minimum_separation <= 0.0:
        raise ValueError("minimum_separation_mm must be finite and positive")
    explicit_step = None if stencil_step_mm is None else float(stencil_step_mm)
    if explicit_step is not None and (
        not np.isfinite(explicit_step) or explicit_step <= 0.0
    ):
        raise ValueError("stencil_step_mm must be finite and positive when supplied")
    return relative, minimum_step, explicit_step, minimum_separation


def evaluate_retarded_dipole_hertz_tensor_native(
    history: TrajectoryHistory,
    observer_event: ObserverEvent,
    *,
    source_identities: Sequence[Hashable] | None = None,
    observer_source_identity: Hashable | None = None,
    excluded_source_identities: Sequence[Hashable] = (),
    require_complete_history: bool = True,
    minimum_separation_mm: float = _DEFAULT_MINIMUM_SEPARATION_MM,
    root_tolerance_mm: float = _DEFAULT_ROOT_TOLERANCE_MM,
    max_root_iterations: int = _DEFAULT_MAX_ROOT_ITERATIONS,
) -> RetardedDipoleHertzResult:
    """Evaluate the summed covariant retarded Hertz tensor at one event."""

    tolerance, iterations = _validated_root_options(
        root_tolerance_mm, max_root_iterations
    )
    _, _, _, minimum_separation = _validated_oracle_options(
        relative_step=_DEFAULT_STENCIL_RELATIVE_STEP,
        minimum_step_mm=_DEFAULT_STENCIL_MINIMUM_STEP_MM,
        stencil_step_mm=None,
        minimum_separation_mm=minimum_separation_mm,
    )
    prepared = _prepare_dipole_history(
        history,
        source_identities=source_identities,
        observer_source_identity=observer_source_identity,
        excluded_source_identities=excluded_source_identities,
    )
    return _evaluate_prepared_hertz_tensor_native(
        prepared,
        observer_event,
        require_complete_history=require_complete_history,
        minimum_separation_mm=minimum_separation,
        root_tolerance_mm=tolerance,
        max_root_iterations=iterations,
    )


def evaluate_retarded_dipole_field_gradient_native(
    history: TrajectoryHistory,
    observer_event: ObserverEvent,
    *,
    source_identities: Sequence[Hashable] | None = None,
    observer_source_identity: Hashable | None = None,
    excluded_source_identities: Sequence[Hashable] = (),
    require_complete_history: bool = True,
    relative_step: float = _DEFAULT_STENCIL_RELATIVE_STEP,
    minimum_step_mm: float = _DEFAULT_STENCIL_MINIMUM_STEP_MM,
    stencil_step_mm: float | None = None,
    minimum_separation_mm: float = _DEFAULT_MINIMUM_SEPARATION_MM,
    root_tolerance_mm: float = _DEFAULT_ROOT_TOLERANCE_MM,
    max_root_iterations: int = _DEFAULT_MAX_ROOT_ITERATIONS,
) -> RetardedDipoleFieldGradientResult:
    """Return full-retarded ``A``, ``partial A``, ``F``, and ``partial F``.

    This first implementation is intentionally a high-cost finite-difference
    oracle.  ``stencil_step_mm`` selects an absolute step; otherwise the step
    is ``max(minimum_step_mm, relative_step * nearest_retarded_separation)``.
    Re-run with half and twice the returned step before treating a difficult
    near-field result as converged.
    """

    tolerance, iterations = _validated_root_options(
        root_tolerance_mm, max_root_iterations
    )
    relative, minimum_step, explicit_step, minimum_separation = (
        _validated_oracle_options(
            relative_step=relative_step,
            minimum_step_mm=minimum_step_mm,
            stencil_step_mm=stencil_step_mm,
            minimum_separation_mm=minimum_separation_mm,
        )
    )
    prepared = _prepare_dipole_history(
        history,
        source_identities=source_identities,
        observer_source_identity=observer_source_identity,
        excluded_source_identities=excluded_source_identities,
    )
    center = _evaluate_prepared_hertz_tensor_native(
        prepared,
        observer_event,
        require_complete_history=require_complete_history,
        minimum_separation_mm=minimum_separation,
        root_tolerance_mm=tolerance,
        max_root_iterations=iterations,
    )
    finite_separations = center.separation_mm[center.valid_sources]
    if explicit_step is not None:
        step = explicit_step
    elif finite_separations.size == 0:
        step = minimum_step
    else:
        step = max(minimum_step, relative * float(np.min(finite_separations)))
    # Repeated centered first derivatives reach offsets of three steps.  Keep
    # the entire stencil outside the strict singularity guard at the center.
    if finite_separations.size and 3.0 * step >= (
        float(np.min(finite_separations)) - minimum_separation
    ):
        raise ValueError(
            "stencil reaches the minimum-separation guard; reduce stencil_step_mm"
        )

    center_position = np.asarray(observer_event.position_mm, dtype=float)

    used_offsets: set[tuple[int, int, int, int]] = set()

    @lru_cache(maxsize=None)
    def hertz_at_offset(offset: tuple[int, int, int, int]) -> RetardedDipoleHertzResult:
        used_offsets.add(offset)
        if offset == (0, 0, 0, 0):
            return center
        displaced_position = center_position + step * np.asarray(
            offset[1:], dtype=float
        )
        displaced = ObserverEvent(
            time_ns=float(observer_event.time_ns) + offset[0] * step / C_MMNS,
            position_mm=cast(
                tuple[float, float, float],
                tuple(float(value) for value in displaced_position),
            ),
        )
        return _evaluate_prepared_hertz_tensor_native(
            prepared,
            displaced,
            require_complete_history=require_complete_history,
            minimum_separation_mm=minimum_separation,
            root_tolerance_mm=tolerance,
            max_root_iterations=iterations,
        )

    @lru_cache(maxsize=None)
    def _hertz_derivative(derivative_indices: tuple[int, ...]) -> np.ndarray:
        """Apply commuting centered first-difference operators to H."""

        if not derivative_indices:
            return hertz_at_offset((0, 0, 0, 0)).hertz_tensor
        total: np.ndarray = np.zeros((4, 4), dtype=float)
        for signs in product((-1, 1), repeat=len(derivative_indices)):
            offset = [0, 0, 0, 0]
            coefficient = 1
            for coordinate, sign in zip(derivative_indices, signs):
                offset[coordinate] += sign
                coefficient *= sign
            total += (
                coefficient
                * hertz_at_offset(
                    cast(tuple[int, int, int, int], tuple(offset))
                ).hertz_tensor
            )
        return cast(np.ndarray, total / (2.0 * step) ** len(derivative_indices))

    def hertz_derivative(derivative_indices: tuple[int, ...]) -> np.ndarray:
        # Sorting makes mixed derivatives share the exact same arithmetic and
        # cache entry, reflecting the commuting continuum derivatives.
        return cast(np.ndarray, _hertz_derivative(tuple(sorted(derivative_indices))))

    four_potential: np.ndarray = np.zeros(4, dtype=float)
    partial_a: np.ndarray = np.zeros((4, 4), dtype=float)
    partial_f: np.ndarray = np.zeros((4, 4, 4), dtype=float)
    for mu in range(4):
        four_potential[mu] = sum(hertz_derivative((nu,))[mu, nu] for nu in range(4))
    for derivative_index in range(4):
        for mu in range(4):
            partial_a[derivative_index, mu] = sum(
                hertz_derivative((derivative_index, nu))[mu, nu] for nu in range(4)
            )

    field_tensor: np.ndarray = np.zeros((4, 4), dtype=float)
    for mu in range(4):
        for nu in range(4):
            field_tensor[mu, nu] = (
                _CONTRAVARIANT_DERIVATIVE_SIGNS[mu] * partial_a[mu, nu]
                - _CONTRAVARIANT_DERIVATIVE_SIGNS[nu] * partial_a[nu, mu]
            )
            for derivative_index in range(4):
                first = sum(
                    hertz_derivative((derivative_index, mu, rho))[nu, rho]
                    for rho in range(4)
                )
                second = sum(
                    hertz_derivative((derivative_index, nu, rho))[mu, rho]
                    for rho in range(4)
                )
                partial_f[derivative_index, mu, nu] = (
                    _CONTRAVARIANT_DERIVATIVE_SIGNS[mu] * first
                    - _CONTRAVARIANT_DERIVATIVE_SIGNS[nu] * second
                )

    electric, magnetic = fields_from_tensor_native(field_tensor)
    # Reconstruct through the public convention helper as a local sign check.
    reconstructed = electromagnetic_field_tensor_native(
        cast(Sequence[float], electric),
        cast(Sequence[float], magnetic),
    )
    if not np.array_equal(field_tensor, reconstructed):
        raise RuntimeError("internal dipole field-tensor convention mismatch")

    ordered_offsets = tuple(sorted(used_offsets))
    stencil_retarded_times = np.stack(
        [hertz_at_offset(offset).retarded_time_ns for offset in ordered_offsets],
        axis=0,
    )
    stencil_residuals = np.stack(
        [hertz_at_offset(offset).light_cone_residual_mm for offset in ordered_offsets],
        axis=0,
    )
    lorenz_residual = float(np.trace(partial_a))
    return RetardedDipoleFieldGradientResult(
        four_potential=four_potential,
        partial_a=partial_a,
        electric_field_native=electric,
        magnetic_field_native=magnetic,
        field_tensor=field_tensor,
        partial_f=partial_f,
        hertz=center,
        stencil_step_mm=step,
        stencil_offsets=np.asarray(ordered_offsets, dtype=int),
        stencil_retarded_time_ns=stencil_retarded_times,
        stencil_light_cone_residual_mm=stencil_residuals,
        lorenz_gauge_residual_per_mm=lorenz_residual,
    )


__all__ = [
    "DipoleSourceSingularityError",
    "RetardedDipoleFieldGradientResult",
    "RetardedDipoleHertzResult",
    "evaluate_retarded_dipole_field_gradient_native",
    "evaluate_retarded_dipole_hertz_tensor_native",
]
