"""Ideal visibility gates on exact source worldlines.

The gate multiplies the ordinary field and potential; its discontinuity is
handled as a canonical potential offset, never differentiated as a field.
Cloud children have independent retarded roots and boundary events.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import fsum
from typing import Any, Iterable

import numpy as np

from .beamline_geometry import compute_directional_visibility_mask
from .constants import C_MMNS
from .types import BeamlineGeometryConfig, Occluder, ParticleState


@dataclass(frozen=True)
class ExactVisibilityHistory:
    """Observer-local selection over an unchanged, cacheable source history."""

    base: Any
    geometry: BeamlineGeometryConfig
    direction: tuple[float, float, float]
    # An explicit set selects a smooth branch during an event-split trial.
    visible_indices: frozenset[int] | None = None


def visibility_history(
    history: Any,
    geometry: BeamlineGeometryConfig | None,
    direction: Iterable[float],
    visible_indices: frozenset[int] | None = None,
) -> Any:
    if geometry is None or not geometry.enabled or not geometry.occluders:
        return history
    return ExactVisibilityHistory(
        history, geometry, tuple(float(value) for value in direction), visible_indices
    )


def boundary_visibility_overrides(
    state: ParticleState, *, dipole: bool = False
) -> dict[int, frozenset[int]]:
    """Recover right-hand boundary masks from checkpointed inclusion metadata."""
    from .potential_inclusion import decode_inclusion_state

    masks = {}
    for particle, encoded in enumerate(state.get("potential_inclusion_state", [])):
        saved = decode_inclusion_state(str(encoded)).get("exact_gate_boundary")
        if saved is not None and saved["time_ns"] == float(state["t"][particle]):
            key = "dipole_visible_indices" if dipole else "visible_indices"
            if key in saved:
                masks[particle] = frozenset(saved[key])
    return masks


def save_boundary_visibility(
    state: ParticleState,
    masks: dict[int, frozenset[int]],
    *,
    dipole_masks: dict[int, frozenset[int]] | None = None,
) -> None:
    from .potential_inclusion import decode_inclusion_state, encode_inclusion_state

    count = len(state["t"])
    metadata = list(state.get("potential_inclusion_state", [""] * count))
    for particle, indices in masks.items():
        saved = decode_inclusion_state(str(metadata[particle]))
        saved["exact_gate_boundary"] = {
            "time_ns": float(state["t"][particle]),
            "visible_indices": sorted(indices),
        }
        if dipole_masks is not None:
            saved["exact_gate_boundary"]["dipole_visible_indices"] = sorted(
                dipole_masks.get(particle, ())
            )
        metadata[particle] = encode_inclusion_state(saved)
    state["potential_inclusion_state"] = np.asarray(metadata, dtype=str)


def selected_occluder(
    geometry: BeamlineGeometryConfig,
    direction: Iterable[float],
) -> Occluder | None:
    vector = np.asarray(tuple(direction), dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm < 1.0e-15 or not geometry.enabled or not geometry.occluders:
        return None
    vector /= norm
    return max(geometry.occluders, key=lambda item: abs(np.dot(item.axis, vector)))


def _relative_position(
    high: np.ndarray,
    low: np.ndarray,
    tail: np.ndarray,
    center: np.ndarray,
    displacement: Iterable[float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Cancel the coordinate origin before rounding away its remainders."""
    return np.array(
        [
            fsum((float(h), -float(c), float(l), float(t), float(dx)))
            for h, l, t, c, dx in zip(high, low, tail, center, displacement)
        ]
    )


def _relative_null_coefficients(segment: Any, center: np.ndarray) -> np.ndarray:
    coefficients = np.array(segment.coefficients_mm, copy=True)
    coefficients[:, 0] *= -1
    coefficients[1, 0] += C_MMNS * segment.duration_ns
    coefficients = coefficients @ segment.frame
    coefficients[0] += _relative_position(
        segment.origin_position_mm,
        segment.origin_position_low_mm,
        (
            np.zeros(3)
            if segment.origin_position_tail_mm is None
            else segment.origin_position_tail_mm
        ),
        center,
    )
    return coefficients


def _sample_relative_position(
    source: Any, sample: Any, event: Any, center: np.ndarray
) -> np.ndarray:
    if sample.source_segment_index is not None and source.light_cone_segments:
        segment = source.light_cone_segments[sample.source_segment_index]
        fraction = sample.source_segment_fraction
        if fraction == 0.0:
            return _relative_position(
                segment.origin_position_mm,
                segment.origin_position_low_mm,
                (
                    np.zeros(3)
                    if segment.origin_position_tail_mm is None
                    else segment.origin_position_tail_mm
                ),
                center,
            )
        if fraction == 1.0 and segment.endpoint_coordinates is not None:
            return _relative_position(*segment.endpoint_coordinates[:3], center)
        values = np.polynomial.polynomial.polyval(fraction, segment.coefficients_mm)
        displacement = (
            np.array(
                [
                    C_MMNS * segment.duration_ns * fraction - values[0],
                    values[1],
                    values[2],
                ]
            )
            @ segment.frame
        )
        return _relative_position(
            segment.origin_position_mm,
            segment.origin_position_low_mm,
            (
                np.zeros(3)
                if segment.origin_position_tail_mm is None
                else segment.origin_position_tail_mm
            ),
            center,
            displacement,
        )
    # Analytic inertial samples carry a resolved separation instead of a
    # segment fraction. Their diagnostic Cartesian position is rounded.
    if sample.precise_separation_mm is not None:
        return np.array(
            [
                fsum((float(h), -float(c), float(l), float(t), -float(r)))
                for h, l, t, r, c in zip(
                    event.position_mm,
                    event.position_low_mm,
                    event.position_tail_mm,
                    sample.precise_separation_mm,
                    center,
                )
            ]
        )
    return sample.position_mm - center


def visible_prepared(
    prepared: Any,
    history: Any,
    event: Any,
    tolerance: float,
    iterations: int,
    extrapolate: float = 0.0,
) -> Any:
    """Select sources at the exact root, retaining the original reduction order."""
    if not isinstance(history, ExactVisibilityHistory):
        return prepared
    if history.visible_indices is not None:
        indices = history.visible_indices
    else:
        from .retarded_fields import _solve_retarded_sample

        indices = set()
        occluder = selected_occluder(history.geometry, history.direction)
        if occluder is None:
            return prepared
        local_geometry = replace(
            history.geometry,
            occluders=[replace(occluder, center_mm=(0.0, 0.0, 0.0))],
        )
        for index, source in prepared.sources.items():
            worldline = getattr(source, "worldline", source)
            sample = _solve_retarded_sample(
                worldline,
                observer_time_ns=event.time_ns,
                observer_position_mm=np.asarray(event.position_mm),
                observer_time_low_ns=event.time_low_ns,
                observer_position_low_mm=np.asarray(event.position_low_mm),
                observer_time_tail_ns=event.time_tail_ns,
                observer_position_tail_mm=np.asarray(event.position_tail_mm),
                root_tolerance_mm=tolerance,
                max_root_iterations=iterations,
                extrapolate_ns=extrapolate,
            )
            # Missing history remains a failure, including for hidden sources.
            if (
                sample is None
                or compute_directional_visibility_mask(
                    _sample_relative_position(
                        worldline, sample, event, np.asarray(occluder.center_mm)
                    )[None, :],
                    local_geometry,
                    history.direction,
                )[0]
            ):
                indices.add(index)
    sources = {i: s for i, s in prepared.sources.items() if i in indices}
    if len(sources) == len(prepared.sources):
        return prepared
    return replace(prepared, sources=sources)


@dataclass(frozen=True)
class VisibilitySwitch:
    source_index: int
    time_ns: float
    position_mm: np.ndarray
    visible_after: bool
    position_low_mm: tuple[float, ...] = (0.0, 0.0, 0.0)
    position_tail_mm: tuple[float, ...] = (0.0, 0.0, 0.0)


def _polynomial_switches(coefficients: np.ndarray, occluder: Occluder) -> list[float]:
    """Find aperture and axial intersections, including enter/exit pairs."""
    from numpy.polynomial import polynomial as poly

    coefficients = np.array(coefficients, copy=True)
    coefficients[0] -= np.asarray(occluder.center_mm)
    axis = np.asarray(occluder.axis)
    axial = coefficients @ axis
    transverse = coefficients - axial[:, None] * axis
    radial = np.zeros(1)
    for column in transverse.T:
        radial = poly.polyadd(radial, poly.polymul(column, column))
    radial[0] -= occluder.radius_mm**2
    boundaries = [radial]
    for sign in (-1, 1):
        end = axial.copy()
        end[0] -= sign * occluder.length_mm * 0.5
        boundaries.append(end)
    roots = []
    for boundary in boundaries:
        # polyroots trims trailing exact zeros and handles linear coasting.
        for value in poly.polyroots(poly.polytrim(boundary)):
            if abs(value.imag) < 1.0e-9:
                roots.append(float(value.real))
    return sorted(set(roots))


def _endpoint_on_boundary(coordinates: np.ndarray, occluder: Occluder) -> bool:
    """Certify an intersection using accepted coordinates, without a tie band."""
    from decimal import Decimal, localcontext

    with localcontext() as context:
        context.prec = 90

        def d(value: float) -> Decimal:
            return Decimal.from_float(float(value))

        relative = [
            d(h) + d(l) + d(t) - d(c)
            for h, l, t, c in zip(*coordinates, occluder.center_mm)
        ]
        axis = [d(value) for value in occluder.axis]
        axial = sum((r * a for r, a in zip(relative, axis)), Decimal(0))
        radial_sq = sum(
            ((r - axial * a) ** 2 for r, a in zip(relative, axis)), Decimal(0)
        )
        return (
            radial_sq == d(occluder.radius_mm) ** 2
            or abs(axial) == d(occluder.length_mm) / 2
        )


def source_visibility_switches(
    source: Any,
    index: int,
    geometry: BeamlineGeometryConfig,
    direction: Iterable[float],
    lower: float,
    upper: float,
) -> list[VisibilitySwitch]:
    """Locate switches on the same polynomial worldline as the exact solver.

    Analytic inertial boundaries use their linear worldline, rather than a
    sampled retarded position. Tangencies and intersections outside the other
    aperture condition do not switch visibility.
    """
    from numpy.polynomial.polynomial import polyval

    occluder = selected_occluder(geometry, direction)
    if occluder is None or upper <= lower:
        return []
    pieces = []
    center = np.asarray(occluder.center_mm)
    local_occluder = replace(occluder, center_mm=(0.0, 0.0, 0.0))
    local_geometry = replace(geometry, occluders=[local_occluder])
    requested_lower = lower
    if source.inertial_boundary is not None:
        time, position, proper_velocity = source.inertial_boundary
        stop = min(upper, time)
        if lower < stop:
            gamma = np.sqrt(1.0 + np.dot(proper_velocity, proper_velocity))
            velocity = C_MMNS * proper_velocity / gamma
            pieces.append(
                (
                    lower,
                    stop,
                    np.stack(
                        (
                            (position - center) + (lower - time) * velocity,
                            (stop - lower) * velocity,
                        )
                    ),
                    (),
                )
            )
        lower = max(lower, time)
    for index_segment, duration in enumerate(source.segment_duration_ns):
        start = float(source.time_ns[index_segment])
        stop = start + float(duration)
        if stop < lower or start > upper:
            continue
        if source.light_cone_segments:
            segment = source.light_cone_segments[index_segment]
            # The solver evaluates displacement in this null frame. Transform
            # that very polynomial, rather than refitting its Cartesian knots.
            coefficients = _relative_null_coefficients(segment, center)
            start = fsum(
                (
                    segment.time_ns,
                    segment.origin_time_low_ns,
                    segment.origin_time_tail_ns,
                )
            )
            stop = start + segment.duration_ns
            endpoints = [
                np.array(
                    [
                        segment.origin_position_mm,
                        segment.origin_position_low_mm,
                        (
                            np.zeros(3)
                            if segment.origin_position_tail_mm is None
                            else segment.origin_position_tail_mm
                        ),
                    ]
                ),
                segment.endpoint_coordinates,
            ]
        else:
            coefficients = np.array(
                source.position_coefficients_mm[index_segment], copy=True
            )
            coefficients[0] -= center
            endpoints = [
                np.array([position, np.zeros(3), np.zeros(3)])
                for position in source.position_mm[index_segment : index_segment + 2]
            ]
        pieces.append((start, stop, coefficients, endpoints))
    result = []
    for start, stop, coefficients, endpoints in pieces:
        roots = _polynomial_switches(coefficients, local_occluder)
        for endpoint, coordinates in enumerate(endpoints):
            if coordinates is not None and _endpoint_on_boundary(
                coordinates[:3], occluder
            ):
                # Snap nearby polynomial roots only to a certified endpoint;
                # the strict [0, 1] filter still rejects genuinely later roots.
                roots = [
                    r for r in roots if abs(r - endpoint) > 128 * np.finfo(float).eps
                ]
                roots.append(float(endpoint))
        roots = sorted(set(roots))
        for root in roots:
            if not 0.0 <= root <= 1.0:
                continue
            time = start + root * (stop - start)
            if not requested_lower <= time <= upper:
                continue
            # Determine branches inside neighbouring root intervals. This
            # avoids assigning a finite tolerance to the aperture itself.
            before = max((r for r in roots if r < root), default=root - 1.0)
            after = min((r for r in roots if r > root), default=root + 1.0)
            probes = np.array(((before + root) * 0.5, (after + root) * 0.5))
            positions = np.stack([polyval(r, coefficients) for r in probes])
            visibility = compute_directional_visibility_mask(
                positions, local_geometry, direction
            )
            if visibility[0] != visibility[1]:
                from .retarded_fields import _translate_resolved_position

                if (
                    root in (0.0, 1.0)
                    and endpoints
                    and endpoints[int(root)] is not None
                ):
                    position, low, tail = endpoints[int(root)][:3]
                else:
                    position, low, tail = _translate_resolved_position(
                        center, np.zeros(3), np.zeros(3), polyval(root, coefficients)
                    )
                result.append(
                    VisibilitySwitch(
                        index,
                        time,
                        position,
                        bool(visibility[1]),
                        tuple(low),
                        tuple(tail),
                    )
                )
    return sorted(result, key=lambda item: item.time_ns)
