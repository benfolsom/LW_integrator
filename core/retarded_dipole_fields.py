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
J. Magn. Magn. Mater. 484, 403--407 (2019),
https://doi.org/10.1016/j.jmmm.2019.04.012, and Heras,
Phys. Rev. E 58, 5047 (1998),
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

from dataclasses import dataclass, field as dataclass_field
from functools import lru_cache
from itertools import product
from typing import Hashable, Sequence, cast

import numpy as np

from .constants import C_MMNS
from .exact_retarded_backend import (
    EXACT_RETARDED_BACKENDS,
    ExactRetardedBackendUnavailableError,
    require_exact_retarded_backend,
    validate_exact_retarded_backend,
)
from .magnetic_dipole import boost_rest_polarization
from .prepared_history_cache import (
    AppendAwarePreparedHistoryCache,
    history_prepared_buffer_capacity,
    history_storage_capacity,
)
from .retarded_fields import (
    ObserverEvent,
    RetardedHistoryError,
    TrajectoryHistory,
    _extract_history,
    _history_constant,
    _history_matrix,
    _HistoryArrays,
    _append_history_arrays,
    _append_prepared_source_history,
    _history_matrix_slice,
    _light_cone_residual_mm,
    _PreparedSourceHistory,
    _prepare_source_history,
    _quintic_worldline_sample,
    _reserve_history_arrays,
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
RETARDED_DIPOLE_BACKENDS = EXACT_RETARDED_BACKENDS


class DipoleSourceSingularityError(ValueError):
    """Raised when an observer/stencil event approaches a point dipole too far."""


# Compatibility alias for callers of the alpha-stage dipole-only API.
RetardedDipoleBackendUnavailableError = ExactRetardedBackendUnavailableError


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
class RetardedDipolePotentialResult:
    """Ordinary dipole four-potential from a nine-event Hertz stencil.

    ``four_potential`` is the contravariant native Gaussian potential
    ``A^mu = partial_nu H^(mu nu)``.  The center event is evaluated for source
    diagnostics and the eight signed unit offsets supply the four commuting
    centered first derivatives.  ``stencil_offsets`` therefore always records
    the center plus those eight derivative events, including when every source
    is excluded.
    """

    four_potential: np.ndarray
    hertz: RetardedDipoleHertzResult
    stencil_step_mm: float
    stencil_offsets: np.ndarray
    stencil_retarded_time_ns: np.ndarray
    stencil_light_cone_residual_mm: np.ndarray


@dataclass(frozen=True)
class RetardedDipoleFieldGradientResult:
    """Ordinary dipole potential, field, and full spacetime field gradient.

    ``partial_a[lambda, nu]`` is the covariant coordinate derivative
    ``partial_lambda A^nu``.  It remains an explicit canonical-equation oracle.
    The maintained exact integration path instead evolves the equivalent
    mechanical ``q F`` response and reconstructs
    ``P_end = p_end + (q/c) A_end`` at the accepted endpoint.

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


@dataclass
class _PreparedDipoleSource:
    identity: Hashable
    worldline: _PreparedSourceHistory
    rest_spin: np.ndarray
    rest_spin_derivative_per_ns: np.ndarray
    preserved_rest_spin_magnitude: float | None
    magnetic_moment_native: float
    _rest_spin_buffer: np.ndarray | None = dataclass_field(default=None, repr=False)
    _slope_buffer: np.ndarray | None = dataclass_field(default=None, repr=False)


@dataclass(frozen=True)
class _PreparedDipoleHistory:
    arrays: _HistoryArrays
    source_identities: tuple[Hashable, ...]
    sources: dict[int, _PreparedDipoleSource]


_DIPOLE_PREPARED_HISTORY_CACHE: AppendAwarePreparedHistoryCache[
    TrajectoryHistory, _PreparedDipoleHistory
] = AppendAwarePreparedHistoryCache(max_entries=2)


@lru_cache(maxsize=1)
def _full_gradient_stencil_offsets() -> tuple[tuple[int, int, int, int], ...]:
    """Return the 129 events in the Python oracle's exact first-use order.

    The reference provider evaluates derivatives lazily.  This order is
    observable when a displaced event raises a history or singularity error,
    so the eager compiled-root batch must not sort its events independently.
    """

    derivative_tuples: set[tuple[int, ...]] = set()
    center_offset = (0, 0, 0, 0)
    offsets: list[tuple[int, int, int, int]] = [center_offset]
    seen_offsets: set[tuple[int, int, int, int]] = {center_offset}

    def record_derivative(derivative_indices: tuple[int, ...]) -> None:
        canonical = tuple(sorted(derivative_indices))
        if canonical in derivative_tuples:
            return
        derivative_tuples.add(canonical)
        for signs in product((-1, 1), repeat=len(canonical)):
            offset = [0, 0, 0, 0]
            for coordinate, sign in zip(canonical, signs):
                offset[coordinate] += sign
            event_offset = cast(tuple[int, int, int, int], tuple(offset))
            if event_offset not in seen_offsets:
                seen_offsets.add(event_offset)
                offsets.append(event_offset)

    # Mirror the loop and generator order below.  Repeated requests are
    # skipped just as _hertz_derivative's lru_cache skips them at runtime.
    for _mu in range(4):
        for nu in range(4):
            record_derivative((nu,))
    for derivative_index in range(4):
        for _mu in range(4):
            for nu in range(4):
                record_derivative((derivative_index, nu))
    for mu in range(4):
        for nu in range(4):
            for derivative_index in range(4):
                for rho in range(4):
                    record_derivative((derivative_index, mu, rho))
                for rho in range(4):
                    record_derivative((derivative_index, nu, rho))

    result = tuple(offsets)
    if len(result) != 129:
        raise RuntimeError(
            f"internal full-gradient stencil must contain 129 events, got {len(result)}"
        )
    return result


def _validated_retarded_dipole_backend(backend: str) -> str:
    return validate_exact_retarded_backend(backend)


def _require_retarded_dipole_backend(backend: str) -> str:
    return require_exact_retarded_backend(backend)


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


def _append_source_spin_slopes_per_ns(
    source: _PreparedDipoleSource,
    appended_spin: np.ndarray,
    combined_time_ns: np.ndarray,
) -> None:
    """Append spin knots in place and recompute only the Hermite tail.

    Appending a knot changes the former endpoint slope.  That changes the old
    final segment and creates the new final segment, so both tail segments are
    represented with the same slopes as a full rebuild while the older prefix
    remains bit-for-bit unchanged.
    """

    old_count = int(source.rest_spin.shape[0])
    append_count = int(appended_spin.shape[0])
    count = old_count + append_count
    current_capacity = (
        0
        if source._rest_spin_buffer is None
        else int(source._rest_spin_buffer.shape[0])
    )
    if (
        source._rest_spin_buffer is None
        or source._slope_buffer is None
        or count > current_capacity
    ):
        capacity = max(count, max(8, 2 * current_capacity))
        if source.worldline._maximum_capacity is not None:
            capacity = min(capacity, source.worldline._maximum_capacity)
        if count > capacity:
            raise ValueError("prepared dipole spin append exceeded builder capacity")
        spin_buffer = np.empty((capacity, 3), dtype=float)
        slope_buffer = np.empty((capacity, 3), dtype=float)
        spin_buffer[:old_count] = source.rest_spin
        slope_buffer[:old_count] = source.rest_spin_derivative_per_ns
        source._rest_spin_buffer = spin_buffer
        source._slope_buffer = slope_buffer
    assert source._rest_spin_buffer is not None
    assert source._slope_buffer is not None
    source._rest_spin_buffer[old_count:count] = appended_spin
    spin = source._rest_spin_buffer[:count]
    slopes = source._slope_buffer
    if count < 2:
        slopes[:count] = 0.0
        source.rest_spin = spin
        source.rest_spin_derivative_per_ns = slopes[:count]
        return
    recompute_start = max(0, old_count - 1)
    for index in range(recompute_start, count):
        if index == 0:
            duration = combined_time_ns[1] - combined_time_ns[0]
            slopes[index] = (spin[1] - spin[0]) / duration
        elif index == count - 1:
            duration = combined_time_ns[-1] - combined_time_ns[-2]
            slopes[index] = (spin[-1] - spin[-2]) / duration
        else:
            previous_duration = combined_time_ns[index] - combined_time_ns[index - 1]
            next_duration = combined_time_ns[index + 1] - combined_time_ns[index]
            previous_secant = (spin[index] - spin[index - 1]) / previous_duration
            next_secant = (spin[index + 1] - spin[index]) / next_duration
            slopes[index] = (
                next_duration * previous_secant + previous_duration * next_secant
            ) / (previous_duration + next_duration)
    source.rest_spin = spin
    source.rest_spin_derivative_per_ns = slopes[:count]


def _reserve_dipole_spin(
    source: _PreparedDipoleSource,
    capacity: int | None,
) -> _PreparedDipoleSource:
    """Give a cached dipole source fixed-capacity spin and slope buffers."""

    if capacity is None:
        return source
    count = int(source.rest_spin.shape[0])
    reserved = max(count, int(capacity))
    spin_buffer = np.empty((reserved, 3), dtype=float)
    slope_buffer = np.empty((reserved, 3), dtype=float)
    spin_buffer[:count] = source.rest_spin
    slope_buffer[:count] = source.rest_spin_derivative_per_ns
    source.rest_spin = spin_buffer[:count]
    source.rest_spin_derivative_per_ns = slope_buffer[:count]
    source._rest_spin_buffer = spin_buffer
    source._slope_buffer = slope_buffer
    return source


def _prepare_dipole_history_uncached(
    history: TrajectoryHistory,
    *,
    source_identities: Sequence[Hashable] | None,
    observer_source_identity: Hashable | None,
    excluded_source_identities: Sequence[Hashable],
    reserve_capacity: int | None = None,
    maximum_capacity: int | None = None,
) -> _PreparedDipoleHistory:
    arrays = _reserve_history_arrays(
        _extract_history(history),
        reserve_capacity,
        maximum_capacity=maximum_capacity,
    )
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
        worldline = _prepare_source_history(
            arrays,
            source_index,
            reserve_capacity=reserve_capacity,
        )
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
        sources[source_index] = _reserve_dipole_spin(
            _PreparedDipoleSource(
                identity=identity,
                worldline=worldline,
                rest_spin=source_spin,
                rest_spin_derivative_per_ns=_source_spin_slopes_per_ns(
                    source_spin, worldline.time_ns
                ),
                preserved_rest_spin_magnitude=preserved_magnitude,
                magnetic_moment_native=float(moments[source_index]),
            ),
            reserve_capacity,
        )
    return _PreparedDipoleHistory(
        arrays=arrays,
        source_identities=identities,
        sources=sources,
    )


def _append_prepared_dipole_history(
    previous: _PreparedDipoleHistory,
    history: TrajectoryHistory,
    old_stop: int,
    *,
    source_identities: Sequence[Hashable] | None,
    observer_source_identity: Hashable | None,
    excluded_source_identities: Sequence[Hashable],
) -> _PreparedDipoleHistory:
    """Extend worldline quintics and the C1 source-spin interpolation tail."""

    arrays = _append_history_arrays(previous.arrays, history, old_stop)
    moments = np.asarray(
        _history_constant(history, "magnetic_moment_native"), dtype=float
    )
    try:
        active = np.asarray(
            _history_constant(history, "magnetic_dipole_active"), dtype=bool
        )
    except (AttributeError, KeyError):
        active = moments != 0.0
    if (
        moments.shape != (arrays.n_sources,)
        or not np.all(np.isfinite(moments))
        or active.shape != (arrays.n_sources,)
    ):
        return _prepare_dipole_history_uncached(
            history,
            source_identities=source_identities,
            observer_source_identity=observer_source_identity,
            excluded_source_identities=excluded_source_identities,
            reserve_capacity=history_prepared_buffer_capacity(history),
            maximum_capacity=history_storage_capacity(history),
        )
    previous_moments = np.zeros(arrays.n_sources, dtype=float)
    previous_active = np.zeros(arrays.n_sources, dtype=bool)
    for source_index, source in previous.sources.items():
        previous_moments[source_index] = source.magnetic_moment_native
        previous_active[source_index] = True
    included_identities = {
        previous.source_identities[source_index] for source_index in previous.sources
    }
    excluded = set(excluded_source_identities)
    if observer_source_identity is not None:
        excluded.add(observer_source_identity)
    expected_included = np.array(
        [
            previous.source_identities[index] not in excluded
            and bool(active[index])
            and moments[index] != 0.0
            for index in range(arrays.n_sources)
        ],
        dtype=bool,
    )
    if (
        not np.array_equal(arrays.charge_native, previous.arrays.charge_native)
        or moments.shape != previous_moments.shape
        or active.shape != previous_active.shape
        or {
            previous.source_identities[index]
            for index in np.flatnonzero(expected_included)
        }
        != included_identities
        or any(moments[index] != previous_moments[index] for index in previous.sources)
    ):
        return _prepare_dipole_history_uncached(
            history,
            source_identities=source_identities,
            observer_source_identity=observer_source_identity,
            excluded_source_identities=excluded_source_identities,
            reserve_capacity=history_prepared_buffer_capacity(history),
            maximum_capacity=history_storage_capacity(history),
        )

    spin_components = tuple(
        np.asarray(
            _history_matrix_slice(history, f"spin_{axis}", old_stop), dtype=float
        )
        for axis in "xyz"
    )
    tail_shape = (int(arrays.time_ns.shape[0]) - int(old_stop), arrays.n_sources)
    if any(component.shape != tail_shape for component in spin_components):
        raise ValueError("source spin histories must share shape [steps, particles]")
    if not all(np.all(np.isfinite(component)) for component in spin_components):
        raise ValueError("source spin histories must contain only finite values")
    spin_tail = np.stack(spin_components, axis=-1)

    for source_index, source in previous.sources.items():
        old_alive_count = int(source.worldline.time_ns.size)
        worldline = _append_prepared_source_history(
            source.worldline,
            arrays,
            old_stop,
        )
        new_alive_count = int(worldline.time_ns.size)
        appended_alive_count = new_alive_count - old_alive_count
        if appended_alive_count:
            appended_spin = spin_tail[:appended_alive_count, source_index]
            _append_source_spin_slopes_per_ns(
                source,
                appended_spin,
                worldline.time_ns,
            )
        preserved_magnitude = source.preserved_rest_spin_magnitude
        if preserved_magnitude is not None and appended_alive_count:
            appended_norms = np.linalg.norm(appended_spin, axis=1)
            if not np.allclose(
                appended_norms,
                preserved_magnitude,
                rtol=1.0e-10,
                atol=1.0e-12,
            ):
                preserved_magnitude = None
        source.worldline = worldline
        source.preserved_rest_spin_magnitude = preserved_magnitude
    return previous


def _prepare_dipole_history(
    history: TrajectoryHistory,
    *,
    source_identities: Sequence[Hashable] | None,
    observer_source_identity: Hashable | None,
    excluded_source_identities: Sequence[Hashable],
) -> _PreparedDipoleHistory:
    """Prepare dipole histories, extending a safe builder-backed cache."""

    identities_key = None if source_identities is None else tuple(source_identities)
    effective_excluded = set(excluded_source_identities)
    if observer_source_identity is not None:
        effective_excluded.add(observer_source_identity)
    if identities_key is not None:
        available_identities = frozenset(identities_key)
        effective_excluded.intersection_update(available_identities)
    variant = (
        "dipole",
        identities_key,
        frozenset(effective_excluded),
    )
    return _DIPOLE_PREPARED_HISTORY_CACHE.prepare(
        history,
        variant=variant,
        prepare_full=lambda current: _prepare_dipole_history_uncached(
            current,
            source_identities=source_identities,
            observer_source_identity=observer_source_identity,
            excluded_source_identities=excluded_source_identities,
            reserve_capacity=history_prepared_buffer_capacity(current),
            maximum_capacity=history_storage_capacity(current),
        ),
        append=lambda prepared, current, old_stop: _append_prepared_dipole_history(
            prepared,
            current,
            old_stop,
            source_identities=source_identities,
            observer_source_identity=observer_source_identity,
            excluded_source_identities=excluded_source_identities,
        ),
    ).value


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


def _evaluate_prepared_hertz_batch_numba_roots_exact_serial(
    prepared: _PreparedDipoleHistory,
    observer_events: Sequence[ObserverEvent],
    *,
    require_complete_history: bool,
    minimum_separation_mm: float,
    root_tolerance_mm: float,
    max_root_iterations: int,
) -> tuple[RetardedDipoleHertzResult, ...]:
    """Compile roots while preserving the Python Hertz and addition order."""

    from .exact_retarded_numba import (
        NUMBA_COMPILATION_ERRORS,
        _STATUS_MISSING_HISTORY,
        _STATUS_TERMINATED_SOURCE,
        _STATUS_VALID,
        evaluate_source_roots_exact_serial,
    )

    event_count = len(observer_events)
    event_time_ns = np.asarray(
        [float(event.time_ns) for event in observer_events], dtype=float
    )
    event_position_mm = np.asarray(
        [
            tuple(float(value) for value in event.position_mm)
            for event in observer_events
        ],
        dtype=float,
    )
    if event_position_mm.shape != (event_count, 3):
        raise ValueError("retarded dipole observer events must have shape [events, 3]")

    source_batches: dict[int, tuple[np.ndarray, ...]] = {}
    for source_index, source in prepared.sources.items():
        worldline = source.worldline
        compiling_initial_signature = not bool(
            getattr(evaluate_source_roots_exact_serial, "signatures", ())
        )
        try:
            source_batches[source_index] = cast(
                tuple[np.ndarray, ...],
                evaluate_source_roots_exact_serial(
                    worldline.time_ns,
                    worldline.position_mm,
                    worldline.segment_duration_ns,
                    worldline.position_coefficients_mm,
                    bool(worldline.ended_by_loss),
                    event_time_ns,
                    event_position_mm,
                    float(root_tolerance_mm),
                    int(max_root_iterations),
                ),
            )
        except NUMBA_COMPILATION_ERRORS as exc:
            if compiling_initial_signature:
                raise RetardedDipoleBackendUnavailableError(
                    "exact retarded backend 'numba_roots_exact_serial' failed "
                    "during initial JIT compilation; select backend 'python' or "
                    "inspect the chained Numba error"
                ) from exc
            raise

    results: list[RetardedDipoleHertzResult] = []
    arrays = prepared.arrays
    for event_index in range(event_count):
        hertz_total = np.zeros((4, 4), dtype=float)
        retarded_time_ns = np.full(arrays.n_sources, np.nan, dtype=float)
        residual_mm = np.full(arrays.n_sources, np.nan, dtype=float)
        separation_mm = np.full(arrays.n_sources, np.nan, dtype=float)
        valid_sources = np.zeros(arrays.n_sources, dtype=bool)
        missing_sources: list[int] = []

        for source_index, source in prepared.sources.items():
            batch = source_batches[source_index]
            status = int(batch[0][event_index])
            if status == _STATUS_TERMINATED_SOURCE:
                continue
            if status == _STATUS_MISSING_HISTORY:
                missing_sources.append(source_index)
                continue
            if status != _STATUS_VALID:
                raise RuntimeError(f"unknown retarded dipole root status {status}")

            source_retarded_time_ns = float(batch[2][event_index])
            # Only the compiled root is authoritative. Recompute the final
            # sample and diagnostics with the reference operations so this
            # backend cannot leak Numba expression-order differences into H.
            worldline = source.worldline
            segment_index = int(
                np.searchsorted(
                    worldline.time_ns, source_retarded_time_ns, side="right"
                )
                - 1
            )
            segment_index = min(max(segment_index, 0), int(worldline.time_ns.size) - 2)
            source_position_mm, source_beta, _source_beta_prime = (
                _quintic_worldline_sample(
                    worldline,
                    segment_index,
                    source_retarded_time_ns,
                )
            )
            source_residual_mm, source_separation_mm = _light_cone_residual_mm(
                observer_time_ns=float(event_time_ns[event_index]),
                observer_position_mm=event_position_mm[event_index],
                source_time_ns=source_retarded_time_ns,
                source_position_mm=source_position_mm,
            )
            if source_separation_mm <= minimum_separation_mm:
                raise DipoleSourceSingularityError(
                    "observer/stencil event is within minimum_separation_mm of "
                    f"dipole source identity {source.identity!r}: "
                    f"{source_separation_mm:.17g} <= "
                    f"{minimum_separation_mm:.17g} mm"
                )

            beta_squared = float(source_beta @ source_beta)
            if beta_squared >= 1.0:
                raise ValueError("source beta magnitude must be less than one")
            gamma = 1.0 / np.sqrt(1.0 - beta_squared)
            separation_vector = event_position_mm[event_index] - source_position_mm
            direction = separation_vector / source_separation_mm
            kappa = 1.0 - float(direction @ source_beta)
            if kappa <= 1.0e-14:
                raise DipoleSourceSingularityError(
                    "retarded dipole field is singular because 1 - n.beta is too small"
                )
            invariant_retarded_distance_mm = gamma * source_separation_mm * kappa
            rest_spin = _interpolate_rest_spin_c1(source, source_retarded_time_ns)
            moment_tensor = _moment_tensor_contravariant(
                magnetic_moment_native=source.magnetic_moment_native,
                rest_spin=rest_spin,
                source_beta=source_beta,
            )
            hertz_total += moment_tensor / invariant_retarded_distance_mm
            retarded_time_ns[source_index] = source_retarded_time_ns
            residual_mm[source_index] = source_residual_mm
            separation_mm[source_index] = source_separation_mm
            valid_sources[source_index] = True

        if require_complete_history and missing_sources:
            missing_identities = [
                prepared.source_identities[index] for index in missing_sources
            ]
            raise RetardedHistoryError(
                "source history does not bracket the observer light cone for dipole "
                f"source identities {missing_identities!r}"
            )
        results.append(
            RetardedDipoleHertzResult(
                hertz_tensor=hertz_total,
                source_identities=prepared.source_identities,
                retarded_time_ns=retarded_time_ns,
                light_cone_residual_mm=residual_mm,
                separation_mm=separation_mm,
                valid_sources=valid_sources,
            )
        )
    return tuple(results)


def _evaluate_prepared_hertz_batch_numba_full_strict_serial(
    prepared: _PreparedDipoleHistory,
    observer_events: Sequence[ObserverEvent],
    *,
    require_complete_history: bool,
    minimum_separation_mm: float,
    root_tolerance_mm: float,
    max_root_iterations: int,
    use_metal_certified_segments: bool = False,
) -> tuple[RetardedDipoleHertzResult, ...]:
    """Compile complete source events while retaining reference reductions."""

    from .exact_retarded_numba import (
        NUMBA_COMPILATION_ERRORS,
        _STATUS_MINIMUM_SEPARATION,
        _STATUS_MISSING_HISTORY,
        _STATUS_SINGULAR_KAPPA,
        _STATUS_SPIN_INTERPOLATION_ZERO,
        _STATUS_SUPERLUMINAL_SOURCE,
        _STATUS_TERMINATED_SOURCE,
        _STATUS_VALID,
        evaluate_source_events_full_strict_serial,
        evaluate_source_events_full_strict_from_segments_serial,
    )

    event_count = len(observer_events)
    event_time_ns = np.asarray(
        [float(event.time_ns) for event in observer_events], dtype=float
    )
    event_position_mm = np.asarray(
        [
            tuple(float(value) for value in event.position_mm)
            for event in observer_events
        ],
        dtype=float,
    )
    if event_position_mm.shape != (event_count, 3):
        raise ValueError("retarded dipole observer events must have shape [events, 3]")

    certified_segments: dict[int, np.ndarray] | None = None
    if use_metal_certified_segments:
        from .metal_certified_roots import certified_metal_segments

        certified_segments = certified_metal_segments(
            prepared.sources,
            event_time_ns,
            event_position_mm,
        )

    source_batches: dict[int, tuple[np.ndarray, ...]] = {}
    for source_index, source in prepared.sources.items():
        worldline = source.worldline
        preserved = source.preserved_rest_spin_magnitude
        event_evaluator = (
            evaluate_source_events_full_strict_from_segments_serial
            if certified_segments is not None
            else evaluate_source_events_full_strict_serial
        )
        compiling_initial_signature = not bool(
            getattr(event_evaluator, "signatures", ())
        )
        try:
            arguments = (
                worldline.time_ns,
                worldline.position_mm,
                worldline.segment_duration_ns,
                worldline.position_coefficients_mm,
                source.rest_spin,
                source.rest_spin_derivative_per_ns,
                preserved is not None,
                0.0 if preserved is None else float(preserved),
                float(source.magnetic_moment_native),
                bool(worldline.ended_by_loss),
                event_time_ns,
                event_position_mm,
                float(minimum_separation_mm),
                float(root_tolerance_mm),
                int(max_root_iterations),
            )
            if certified_segments is not None:
                arguments = arguments + (certified_segments[source_index],)
            source_batches[source_index] = cast(
                tuple[np.ndarray, ...],
                event_evaluator(*arguments),
            )
        except NUMBA_COMPILATION_ERRORS as exc:
            if compiling_initial_signature:
                selected_name = (
                    "metal_certified_full_strict"
                    if use_metal_certified_segments
                    else "numba_full_strict_serial"
                )
                raise RetardedDipoleBackendUnavailableError(
                    f"exact retarded backend {selected_name!r} failed "
                    "during initial JIT compilation; select backend 'python' or "
                    "inspect the chained Numba error"
                ) from exc
            raise

    results: list[RetardedDipoleHertzResult] = []
    arrays = prepared.arrays
    for event_index in range(event_count):
        hertz_total = np.zeros((4, 4), dtype=float)
        retarded_time_ns = np.full(arrays.n_sources, np.nan, dtype=float)
        residual_mm = np.full(arrays.n_sources, np.nan, dtype=float)
        separation_mm = np.full(arrays.n_sources, np.nan, dtype=float)
        valid_sources = np.zeros(arrays.n_sources, dtype=bool)
        missing_sources: list[int] = []

        for source_index, source in prepared.sources.items():
            batch = source_batches[source_index]
            status = int(batch[0][event_index])
            if status == _STATUS_TERMINATED_SOURCE:
                continue
            if status == _STATUS_MISSING_HISTORY:
                missing_sources.append(source_index)
                continue

            source_hertz = batch[1][event_index]
            source_retarded_time_ns = float(batch[2][event_index])
            source_residual_mm = float(batch[3][event_index])
            source_separation_mm = float(batch[4][event_index])
            if status == _STATUS_MINIMUM_SEPARATION:
                raise DipoleSourceSingularityError(
                    "observer/stencil event is within minimum_separation_mm of "
                    f"dipole source identity {source.identity!r}: "
                    f"{source_separation_mm:.17g} <= "
                    f"{minimum_separation_mm:.17g} mm"
                )
            if status == _STATUS_SUPERLUMINAL_SOURCE:
                raise ValueError("source beta magnitude must be less than one")
            if status == _STATUS_SINGULAR_KAPPA:
                raise DipoleSourceSingularityError(
                    "retarded dipole field is singular because 1 - n.beta is too small"
                )
            if status == _STATUS_SPIN_INTERPOLATION_ZERO:
                raise RetardedHistoryError(
                    "constant-magnitude source-spin interpolation crossed zero; "
                    "reduce the source-history timestep"
                )
            if status != _STATUS_VALID:
                raise RuntimeError(f"unknown strict Hertz event status {status}")

            hertz_total += source_hertz
            retarded_time_ns[source_index] = source_retarded_time_ns
            residual_mm[source_index] = source_residual_mm
            separation_mm[source_index] = source_separation_mm
            valid_sources[source_index] = True

        if require_complete_history and missing_sources:
            missing_identities = [
                prepared.source_identities[index] for index in missing_sources
            ]
            raise RetardedHistoryError(
                "source history does not bracket the observer light cone for dipole "
                f"source identities {missing_identities!r}"
            )
        results.append(
            RetardedDipoleHertzResult(
                hertz_tensor=hertz_total,
                source_identities=prepared.source_identities,
                retarded_time_ns=retarded_time_ns,
                light_cone_residual_mm=residual_mm,
                separation_mm=separation_mm,
                valid_sources=valid_sources,
            )
        )
    return tuple(results)


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


def evaluate_retarded_dipole_potential_native(
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
    backend: str = "python",
) -> RetardedDipolePotentialResult:
    """Return the full-retarded ordinary dipole four-potential efficiently.

    This is the endpoint-reconstruction path.  It evaluates the Hertz tensor
    at the observer event for diagnostics and at the eight independently
    retarded signed unit offsets needed by
    ``A^mu = partial_nu H^(mu nu)``.  In contrast, the full field-gradient
    oracle needs 129 Hertz evaluations to construct derivatives through third
    order.

    ``stencil_step_mm`` selects an absolute step.  Otherwise the step is
    ``max(minimum_step_mm, relative_step * nearest_retarded_separation)``.
    Every displaced event performs its own light-cone solve; source state and
    retarded time are never frozen across the stencil.

    The center evaluation remains on the Python reference path for every
    backend because it selects the adaptive stencil step and supplies the
    archived center diagnostics. Explicit Numba backends batch only the eight
    displaced events, then retain the Python reference finite-difference and
    source-reduction order.
    """

    selected_backend = _require_retarded_dipole_backend(backend)
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
    # This provider reaches one step from the center.  Keep the complete
    # first-derivative stencil outside the strict singularity guard.  Each
    # displaced evaluation retains the exact guard as the final authority.
    if finite_separations.size and step >= (
        float(np.min(finite_separations)) - minimum_separation
    ):
        raise ValueError(
            "stencil reaches the minimum-separation guard; reduce stencil_step_mm"
        )

    center_position = np.asarray(observer_event.position_mm, dtype=float)
    evaluated: dict[tuple[int, int, int, int], RetardedDipoleHertzResult] = {
        (0, 0, 0, 0): center
    }

    if selected_backend != "python":
        displaced_offsets: list[tuple[int, int, int, int]] = []
        displaced_events: list[ObserverEvent] = []
        for derivative_index in range(4):
            for sign in (-1, 1):
                offset = [0, 0, 0, 0]
                offset[derivative_index] = sign
                event_offset = cast(tuple[int, int, int, int], tuple(offset))
                displaced_position = center_position + step * np.asarray(
                    event_offset[1:], dtype=float
                )
                displaced_offsets.append(event_offset)
                displaced_events.append(
                    ObserverEvent(
                        time_ns=(
                            float(observer_event.time_ns)
                            + event_offset[0] * step / C_MMNS
                        ),
                        position_mm=cast(
                            tuple[float, float, float],
                            tuple(float(value) for value in displaced_position),
                        ),
                    )
                )
        if selected_backend == "numba_roots_exact_serial":
            displaced_results = _evaluate_prepared_hertz_batch_numba_roots_exact_serial(
                prepared,
                displaced_events,
                require_complete_history=require_complete_history,
                minimum_separation_mm=minimum_separation,
                root_tolerance_mm=tolerance,
                max_root_iterations=iterations,
            )
        else:
            displaced_results = _evaluate_prepared_hertz_batch_numba_full_strict_serial(
                prepared,
                displaced_events,
                require_complete_history=require_complete_history,
                minimum_separation_mm=minimum_separation,
                root_tolerance_mm=tolerance,
                max_root_iterations=iterations,
                use_metal_certified_segments=(
                    selected_backend == "metal_certified_full_strict"
                ),
            )
        evaluated.update(zip(displaced_offsets, displaced_results))

    def hertz_at_offset(
        offset: tuple[int, int, int, int],
    ) -> RetardedDipoleHertzResult:
        cached = evaluated.get(offset)
        if cached is not None:
            return cached
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
        result = _evaluate_prepared_hertz_tensor_native(
            prepared,
            displaced,
            require_complete_history=require_complete_history,
            minimum_separation_mm=minimum_separation,
            root_tolerance_mm=tolerance,
            max_root_iterations=iterations,
        )
        evaluated[offset] = result
        return result

    first_derivatives: list[np.ndarray] = []
    for derivative_index in range(4):
        lower_offset = [0, 0, 0, 0]
        upper_offset = [0, 0, 0, 0]
        lower_offset[derivative_index] = -1
        upper_offset[derivative_index] = 1
        lower = hertz_at_offset(
            cast(tuple[int, int, int, int], tuple(lower_offset))
        ).hertz_tensor
        upper = hertz_at_offset(
            cast(tuple[int, int, int, int], tuple(upper_offset))
        ).hertz_tensor
        # Match the full oracle's accumulation order exactly: zero, lower
        # with coefficient -1, then upper with coefficient +1.
        derivative = np.zeros((4, 4), dtype=float)
        derivative += -lower
        derivative += upper
        derivative /= 2.0 * step
        first_derivatives.append(derivative)

    four_potential = np.zeros(4, dtype=float)
    for mu in range(4):
        four_potential[mu] = sum(first_derivatives[nu][mu, nu] for nu in range(4))

    ordered_offsets = tuple(sorted(evaluated))
    stencil_retarded_times = np.stack(
        [evaluated[offset].retarded_time_ns for offset in ordered_offsets], axis=0
    )
    stencil_residuals = np.stack(
        [evaluated[offset].light_cone_residual_mm for offset in ordered_offsets],
        axis=0,
    )
    return RetardedDipolePotentialResult(
        four_potential=four_potential,
        hertz=center,
        stencil_step_mm=step,
        stencil_offsets=np.asarray(ordered_offsets, dtype=int),
        stencil_retarded_time_ns=stencil_retarded_times,
        stencil_light_cone_residual_mm=stencil_residuals,
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
    backend: str = "python",
) -> RetardedDipoleFieldGradientResult:
    """Return full-retarded ``A``, ``partial A``, ``F``, and ``partial F``.

    This first implementation is intentionally a high-cost finite-difference
    oracle.  ``stencil_step_mm`` selects an absolute step; otherwise the step
    is ``max(minimum_step_mm, relative_step * nearest_retarded_separation)``.
    Re-run with half and twice the returned step before treating a difficult
    near-field result as converged.

    ``backend='numba_roots_exact_serial'`` compiles only the independent
    light-cone roots. The Python reference arithmetic remains authoritative for
    every Hertz tensor, source sum, and finite difference.
    ``backend='numba_full_strict_serial'`` additionally compiles worldline,
    spin, moment, Hodge-dual, and per-source Hertz arithmetic with
    ``fastmath=False``; source and finite-difference reductions retain their
    reference order. ``backend='numba_analytic_charge_dipole_response_serial'``
    propagates one third-order observer jet through a smooth source segment and
    falls back to the full-strict oracle at spin/history nonsmoothness.
    ``python`` is the cross-platform default and no automatic backend selection
    is performed.
    """

    selected_backend = _require_retarded_dipole_backend(backend)
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
    if selected_backend == "numba_analytic_charge_dipole_response_serial":
        from .dipole_hertz_jet import (
            evaluate_retarded_dipole_field_gradient_hertz_jet_native,
        )

        return evaluate_retarded_dipole_field_gradient_hertz_jet_native(
            history,
            observer_event,
            source_identities=source_identities,
            observer_source_identity=observer_source_identity,
            excluded_source_identities=excluded_source_identities,
            require_complete_history=require_complete_history,
            fallback_relative_step=relative,
            fallback_minimum_step_mm=minimum_step,
            fallback_stencil_step_mm=explicit_step,
            minimum_separation_mm=minimum_separation,
            root_tolerance_mm=tolerance,
            max_root_iterations=iterations,
            response_kernel="numba_strict_serial",
            fallback_backend="numba_full_strict_serial",
        ).response
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

    precomputed_hertz: dict[tuple[int, int, int, int], RetardedDipoleHertzResult] = {}
    if selected_backend != "python":
        batch_offsets = tuple(
            offset
            for offset in _full_gradient_stencil_offsets()
            if offset != (0, 0, 0, 0)
        )
        batch_events = []
        for offset in batch_offsets:
            displaced_position = center_position + step * np.asarray(
                offset[1:], dtype=float
            )
            batch_events.append(
                ObserverEvent(
                    time_ns=float(observer_event.time_ns) + offset[0] * step / C_MMNS,
                    position_mm=cast(
                        tuple[float, float, float],
                        tuple(float(value) for value in displaced_position),
                    ),
                )
            )
        if selected_backend == "numba_roots_exact_serial":
            batch_results = _evaluate_prepared_hertz_batch_numba_roots_exact_serial(
                prepared,
                batch_events,
                require_complete_history=require_complete_history,
                minimum_separation_mm=minimum_separation,
                root_tolerance_mm=tolerance,
                max_root_iterations=iterations,
            )
        else:
            batch_results = _evaluate_prepared_hertz_batch_numba_full_strict_serial(
                prepared,
                batch_events,
                require_complete_history=require_complete_history,
                minimum_separation_mm=minimum_separation,
                root_tolerance_mm=tolerance,
                max_root_iterations=iterations,
                use_metal_certified_segments=(
                    selected_backend == "metal_certified_full_strict"
                ),
            )
        precomputed_hertz = dict(zip(batch_offsets, batch_results))

    used_offsets: set[tuple[int, int, int, int]] = set()

    @lru_cache(maxsize=None)
    def hertz_at_offset(offset: tuple[int, int, int, int]) -> RetardedDipoleHertzResult:
        used_offsets.add(offset)
        if offset == (0, 0, 0, 0):
            return center
        precomputed = precomputed_hertz.get(offset)
        if precomputed is not None:
            return precomputed
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
    "RETARDED_DIPOLE_BACKENDS",
    "RetardedDipoleBackendUnavailableError",
    "RetardedDipoleFieldGradientResult",
    "RetardedDipoleHertzResult",
    "RetardedDipolePotentialResult",
    "evaluate_retarded_dipole_field_gradient_native",
    "evaluate_retarded_dipole_hertz_tensor_native",
    "evaluate_retarded_dipole_potential_native",
]
