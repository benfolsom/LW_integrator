"""Local retarded source jets from accepted acceleration and spin histories.

This module deliberately separates two numerical jobs:

* a cubic Hermite curve, fixed by accepted position and velocity, locates the
  retarded event; and
* local least-squares fits evaluate acceleration derivatives and spin-chart
  derivatives directly at that event.

The fitted derivatives are encoded as a centered Taylor polynomial only as an
input adapter for the already validated Hertz-jet algebra. They are not joined
into a global high-continuity worldline. This avoids amplifying small endpoint
derivative inconsistencies through a degree-eleven Hermite segment.

The route is currently an opt-in provider primitive. It fails closed when the
accepted prefix does not contain a complete fit window around the retarded
event or when exact equations-of-motion start acceleration is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Any, Literal, Sequence

import numpy as np

from .causal_c5_dipole_provider import CausalC5DipoleSourceCollection
from .causal_c5_source_history import (
    CausalC5HistoryUnavailableError,
    _spin_to_stereographic,
)
from .constants import C_MMNS
from .dipole_hertz_jet import (
    DipoleHertzResponseJetResult,
    polynomial_dipole_hertz_response_jet_native,
)
from .rfs import fields_from_tensor_native

if TYPE_CHECKING:
    from .retarded_fields import ObserverEvent


@dataclass(frozen=True)
class LocalSourceJetFitConfig:
    """Numerical differentiation settings for one local source jet."""

    half_width_ns: float
    acceleration_degree: int = 5
    spin_degree: int = 5
    window_weighting: Literal["tricube", "uniform"] = "tricube"
    maximum_condition_number: float = 1.0e5

    def __post_init__(self) -> None:
        width = float(self.half_width_ns)
        acceleration_degree = int(self.acceleration_degree)
        spin_degree = int(self.spin_degree)
        window_weighting = str(self.window_weighting)
        maximum_condition = float(self.maximum_condition_number)
        if not np.isfinite(width) or width <= 0.0:
            raise ValueError("half_width_ns must be finite and positive")
        if acceleration_degree < 3:
            raise ValueError("acceleration_degree must be at least three")
        if spin_degree < 5:
            raise ValueError("spin_degree must be at least five")
        if window_weighting not in {"tricube", "uniform"}:
            raise ValueError("window_weighting must be 'tricube' or 'uniform'")
        if not np.isfinite(maximum_condition) or maximum_condition <= 1.0:
            raise ValueError(
                "maximum_condition_number must be finite and greater than one"
            )
        object.__setattr__(self, "half_width_ns", width)
        object.__setattr__(self, "acceleration_degree", acceleration_degree)
        object.__setattr__(self, "spin_degree", spin_degree)
        object.__setattr__(self, "window_weighting", window_weighting)
        object.__setattr__(self, "maximum_condition_number", maximum_condition)


@dataclass(frozen=True)
class LocalSourceJetModelSpreadConfig:
    """Narrow and wide fits used to test local-model sensitivity."""

    narrow_fit: LocalSourceJetFitConfig
    wide_fit: LocalSourceJetFitConfig
    maximum_relative_spread: float = 1.0e-3

    def __post_init__(self) -> None:
        maximum = float(self.maximum_relative_spread)
        if not np.isfinite(maximum) or maximum <= 0.0:
            raise ValueError("maximum_relative_spread must be finite and positive")
        object.__setattr__(self, "maximum_relative_spread", maximum)


@dataclass(frozen=True)
class LocalSourceJetModelSpread:
    """Largest pairwise response difference across nested local fits."""

    four_potential: float
    partial_a: float
    field_tensor: float
    partial_f: float

    def __post_init__(self) -> None:
        for name in ("four_potential", "partial_a", "field_tensor", "partial_f"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} model spread must be finite and non-negative")
            object.__setattr__(self, name, value)

    @property
    def maximum(self) -> float:
        return max(
            self.four_potential,
            self.partial_a,
            self.field_tensor,
            self.partial_f,
        )


class CausalLocalSourceJetModelSpreadError(CausalC5HistoryUnavailableError):
    """Raised when nested fits do not define a stable local response plateau."""


@dataclass(frozen=True)
class LocalSourceJetDiagnostics:
    """Auditable root and fit information for one source contribution."""

    root_segment_index: int
    acceleration_sample_indices: np.ndarray
    spin_sample_indices: np.ndarray
    acceleration_condition_number: float
    spin_condition_number: float
    light_cone_residual_mm: float
    model_spread: LocalSourceJetModelSpread | None = None

    def __post_init__(self) -> None:
        if int(self.root_segment_index) < 0:
            raise ValueError("root_segment_index must be non-negative")
        for name in ("acceleration_sample_indices", "spin_sample_indices"):
            values = np.asarray(getattr(self, name), dtype=np.int64)
            if values.ndim != 1 or values.size == 0 or np.any(np.diff(values) <= 0):
                raise ValueError(f"{name} must be a nonempty increasing vector")
            values = np.array(values, copy=True)
            values.setflags(write=False)
            object.__setattr__(self, name, values)
        for name in (
            "acceleration_condition_number",
            "spin_condition_number",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        residual = float(self.light_cone_residual_mm)
        if not np.isfinite(residual):
            raise ValueError("light_cone_residual_mm must be finite")
        object.__setattr__(self, "root_segment_index", int(self.root_segment_index))
        object.__setattr__(self, "light_cone_residual_mm", residual)
        if self.model_spread is not None and not isinstance(
            self.model_spread,
            LocalSourceJetModelSpread,
        ):
            raise TypeError("model_spread must be a LocalSourceJetModelSpread")


@dataclass(frozen=True)
class CausalLocalSourceJetEvaluation:
    """One identity-labelled local source-jet contribution."""

    identity: str
    response: DipoleHertzResponseJetResult
    diagnostics: LocalSourceJetDiagnostics


@dataclass(frozen=True)
class CausalLocalSourceJetProviderResult:
    """Stable-order response sum and individual local-fit diagnostics."""

    four_potential: np.ndarray
    partial_a: np.ndarray
    electric_field_native: np.ndarray
    magnetic_field_native: np.ndarray
    field_tensor: np.ndarray
    partial_f: np.ndarray
    source_results: tuple[CausalLocalSourceJetEvaluation, ...]

    def __post_init__(self) -> None:
        shapes = {
            "four_potential": (4,),
            "partial_a": (4, 4),
            "electric_field_native": (3,),
            "magnetic_field_native": (3,),
            "field_tensor": (4, 4),
            "partial_f": (4, 4, 4),
        }
        for name, shape in shapes.items():
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != shape or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be a finite array with shape {shape}")
            detached = np.array(value, copy=True)
            detached.setflags(write=False)
            object.__setattr__(self, name, detached)
        object.__setattr__(self, "source_results", tuple(self.source_results))


def _cubic_position_velocity(
    history: Any,
    segment_index: int,
    source_time_ns: float,
) -> tuple[np.ndarray, np.ndarray]:
    left = int(segment_index)
    right = left + 1
    start = float(history.time_ns[left])
    duration = float(history.time_ns[right] - start)
    fraction = (float(source_time_ns) - start) / duration
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("source time lies outside the cubic root segment")
    position_start = np.asarray(history.position_mm[left], dtype=np.float64)
    position_end = np.asarray(history.position_mm[right], dtype=np.float64)
    velocity_start = C_MMNS * np.asarray(history.beta[left], dtype=np.float64)
    velocity_end = C_MMNS * np.asarray(history.beta[right], dtype=np.float64)
    coefficients = np.asarray(
        (
            position_start,
            duration * velocity_start,
            3.0 * (position_end - position_start)
            - duration * (2.0 * velocity_start + velocity_end),
            2.0 * (position_start - position_end)
            + duration * (velocity_start + velocity_end),
        )
    )
    position = np.zeros(3, dtype=np.float64)
    velocity = np.zeros(3, dtype=np.float64)
    for power, coefficient in enumerate(coefficients):
        position += coefficient * fraction**power
        if power:
            velocity += power * coefficient * fraction ** (power - 1) / duration
    return position, velocity


def _solve_cubic_retarded_root(
    history: Any,
    observer_event: "ObserverEvent",
    *,
    root_tolerance_mm: float,
    max_root_iterations: int,
    minimum_separation_mm: float,
) -> tuple[int, float, np.ndarray, np.ndarray, float]:
    observer_time = float(observer_event.time_ns)
    observer_position = np.asarray(observer_event.position_mm, dtype=np.float64)
    tolerance = float(root_tolerance_mm)
    iterations = int(max_root_iterations)
    minimum_separation = float(minimum_separation_mm)
    if observer_position.shape != (3,) or not np.all(np.isfinite(observer_position)):
        raise ValueError("observer position must contain three finite values")
    if not np.isfinite(observer_time):
        raise ValueError("observer time must be finite")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("root_tolerance_mm must be finite and positive")
    if iterations < 1:
        raise ValueError("max_root_iterations must be positive")
    if not np.isfinite(minimum_separation) or minimum_separation <= 0.0:
        raise ValueError("minimum_separation_mm must be finite and positive")
    count = int(history.sample_count)
    if count < 2:
        raise CausalC5HistoryUnavailableError(
            "local source jet needs at least two accepted source knots"
        )

    def knot_residual(knot: int) -> float:
        separation = float(
            np.linalg.norm(observer_position - history.position_mm[knot])
        )
        return C_MMNS * (observer_time - float(history.time_ns[knot])) - separation

    if knot_residual(0) < 0.0:
        raise CausalC5HistoryUnavailableError(
            "observer light cone predates the accepted source history"
        )
    if knot_residual(count - 1) > 0.0:
        raise CausalC5HistoryUnavailableError(
            "observer light cone reaches the unaccepted source future"
        )
    lower = 0
    upper = count - 1
    while upper - lower > 1:
        middle = (lower + upper) // 2
        if knot_residual(middle) > 0.0:
            lower = middle
        else:
            upper = middle
    lower_time = float(history.time_ns[lower])
    upper_time = float(history.time_ns[upper])
    lower_residual = knot_residual(lower)
    upper_residual = knot_residual(upper)
    if abs(lower_residual) <= tolerance:
        root_time = lower_time
    elif abs(upper_residual) <= tolerance:
        root_time = upper_time
    else:
        root_time = lower_time - lower_residual * (upper_time - lower_time) / (
            upper_residual - lower_residual
        )
        for _ in range(iterations):
            source_position, source_velocity = _cubic_position_velocity(
                history, lower, root_time
            )
            displacement = observer_position - source_position
            separation = float(np.linalg.norm(displacement))
            if separation <= minimum_separation:
                raise ValueError("observer is too close to the local-jet source")
            residual = C_MMNS * (observer_time - root_time) - separation
            if abs(residual) <= tolerance:
                break
            if residual > 0.0:
                lower_time = root_time
                lower_residual = residual
            else:
                upper_time = root_time
                upper_residual = residual
            direction = displacement / separation
            derivative = -C_MMNS + float(direction @ source_velocity)
            candidate = root_time - residual / derivative
            if not lower_time < candidate < upper_time:
                candidate = 0.5 * (lower_time + upper_time)
            if candidate == root_time:
                break
            root_time = candidate
    source_position, source_velocity = _cubic_position_velocity(
        history, lower, root_time
    )
    displacement = observer_position - source_position
    separation = float(np.linalg.norm(displacement))
    if separation <= minimum_separation:
        raise ValueError("observer is too close to the local-jet source")
    residual = C_MMNS * (observer_time - root_time) - separation
    return lower, root_time, source_position, source_velocity, residual


def _local_polynomial_derivatives(
    *,
    sample_times_ns: np.ndarray,
    sample_values: np.ndarray,
    target_time_ns: float,
    half_width_ns: float,
    degree: int,
    maximum_derivative: int,
    window_weighting: Literal["tricube", "uniform"],
    maximum_condition_number: float,
    sample_indices: np.ndarray,
    label: str,
) -> tuple[list[np.ndarray], float, np.ndarray]:
    times = np.asarray(sample_times_ns, dtype=np.float64)
    values = np.asarray(sample_values, dtype=np.float64)
    indices = np.asarray(sample_indices, dtype=np.int64)
    window_start = float(target_time_ns) - float(half_width_ns)
    window_end = float(target_time_ns) + float(half_width_ns)
    if times.size == 0:
        raise CausalC5HistoryUnavailableError(
            f"local {label} fit has no accepted exact samples"
        )
    if times[0] > window_start or times[-1] < window_end:
        raise CausalC5HistoryUnavailableError(
            f"local {label} fit does not have its full physical window in the "
            "accepted source history"
        )
    selected_mask = np.abs(times - float(target_time_ns)) <= float(half_width_ns)
    selected_times = times[selected_mask]
    selected_values = values[selected_mask]
    selected_indices = indices[selected_mask]
    if (
        selected_times.size < degree + 1
        or selected_times[0] >= target_time_ns
        or selected_times[-1] <= target_time_ns
    ):
        raise CausalC5HistoryUnavailableError(
            f"local {label} fit window is not fully accepted around the retarded event"
        )
    scale = float(np.max(np.abs(selected_times - target_time_ns)))
    normalized = (selected_times - target_time_ns) / scale
    design = np.vander(normalized, N=degree + 1, increasing=True)
    reference = selected_values[int(np.argmin(np.abs(normalized)))]
    if window_weighting == "tricube":
        window_coordinate = np.abs(
            (selected_times - float(target_time_ns)) / float(half_width_ns)
        )
        weights = np.maximum(0.0, 1.0 - window_coordinate**3) ** 3
    else:
        weights = np.ones(selected_times.size, dtype=np.float64)
    square_root_weight = np.sqrt(weights)
    weighted_design = design * square_root_weight[:, np.newaxis]
    value_weight_shape = (selected_times.size,) + (1,) * (selected_values.ndim - 1)
    weighted_values = (selected_values - reference) * square_root_weight.reshape(
        value_weight_shape
    )
    coefficients, _, rank, _ = np.linalg.lstsq(
        weighted_design,
        weighted_values,
        rcond=None,
    )
    condition = float(np.linalg.cond(weighted_design))
    if rank != degree + 1:
        raise CausalC5HistoryUnavailableError(f"local {label} fit is rank deficient")
    if condition > maximum_condition_number:
        raise CausalC5HistoryUnavailableError(
            f"local {label} fit exceeds its condition-number limit"
        )
    derivatives = []
    for order in range(maximum_derivative + 1):
        derivative = coefficients[order] * math.factorial(order) / scale**order
        if order == 0:
            derivative = derivative + reference
        derivatives.append(np.asarray(derivative, dtype=np.float64))
    return derivatives, condition, selected_indices


def _centered_taylor_coefficients(
    derivatives: Sequence[np.ndarray],
    *,
    duration_ns: float,
) -> np.ndarray:
    coefficients = np.zeros(
        (len(derivatives), np.asarray(derivatives[0]).size), dtype=np.float64
    )
    for order, derivative in enumerate(derivatives):
        scaled = (
            np.asarray(derivative, dtype=np.float64)
            * duration_ns**order
            / math.factorial(order)
        )
        for power in range(order + 1):
            coefficients[power] += (
                scaled * math.comb(order, power) * (-0.5) ** (order - power)
            )
    return coefficients


def _relative_response_difference(
    left: DipoleHertzResponseJetResult,
    right: DipoleHertzResponseJetResult,
    name: str,
) -> float:
    first = np.asarray(getattr(left, name), dtype=np.float64)
    second = np.asarray(getattr(right, name), dtype=np.float64)
    scale = max(
        float(np.linalg.norm(first)),
        float(np.linalg.norm(second)),
        np.finfo(np.float64).tiny,
    )
    return float(np.linalg.norm(first - second) / scale)


def _response_model_spread(
    responses: Sequence[DipoleHertzResponseJetResult],
) -> LocalSourceJetModelSpread:
    if len(responses) < 2:
        raise ValueError("model spread needs at least two response fits")

    def largest(name: str) -> float:
        return max(
            _relative_response_difference(responses[left], responses[right], name)
            for left in range(len(responses))
            for right in range(left + 1, len(responses))
        )

    return LocalSourceJetModelSpread(
        four_potential=largest("four_potential"),
        partial_a=largest("partial_a"),
        field_tensor=largest("field_tensor"),
        partial_f=largest("partial_f"),
    )


def evaluate_causal_local_source_jet_native(
    history: Any,
    observer_event: "ObserverEvent",
    *,
    magnetic_moment_native: float,
    fit: LocalSourceJetFitConfig,
    model_spread: LocalSourceJetModelSpreadConfig | None = None,
    root_tolerance_mm: float = 1.0e-21,
    max_root_iterations: int = 96,
    minimum_separation_mm: float = 1.0e-15,
) -> tuple[DipoleHertzResponseJetResult, LocalSourceJetDiagnostics]:
    """Evaluate one source using a cubic root and a direct local derivative jet."""

    segment, root_time, position, velocity, residual = _solve_cubic_retarded_root(
        history,
        observer_event,
        root_tolerance_mm=root_tolerance_mm,
        max_root_iterations=max_root_iterations,
        minimum_separation_mm=minimum_separation_mm,
    )
    source_knots = np.arange(history.sample_count - 1, dtype=np.int64)
    step_rows = source_knots + 1
    ready = np.asarray(history.step_start_beta_prime_ready, dtype=bool)[step_rows]
    acceleration_knots = source_knots[ready]
    accelerations = (
        C_MMNS**2
        * np.asarray(history.step_start_beta_prime_per_mm, dtype=np.float64)[
            acceleration_knots + 1
        ]
    )
    acceleration_derivatives, acceleration_condition, acceleration_indices = (
        _local_polynomial_derivatives(
            sample_times_ns=history.time_ns[acceleration_knots],
            sample_values=accelerations,
            target_time_ns=root_time,
            half_width_ns=fit.half_width_ns,
            degree=fit.acceleration_degree,
            maximum_derivative=3,
            window_weighting=fit.window_weighting,
            maximum_condition_number=fit.maximum_condition_number,
            sample_indices=acceleration_knots,
            label="acceleration",
        )
    )
    spin_chart = _spin_to_stereographic(
        np.asarray(history.rest_spin, dtype=np.float64),
        np.asarray(history.stereographic_frame, dtype=np.float64),
    )
    spin_indices = np.arange(history.sample_count, dtype=np.int64)
    spin_derivatives, spin_condition, selected_spin_indices = (
        _local_polynomial_derivatives(
            sample_times_ns=history.time_ns,
            sample_values=spin_chart,
            target_time_ns=root_time,
            half_width_ns=fit.half_width_ns,
            degree=fit.spin_degree,
            maximum_derivative=5,
            window_weighting=fit.window_weighting,
            maximum_condition_number=fit.maximum_condition_number,
            sample_indices=spin_indices,
            label="spin",
        )
    )
    position_derivatives = [position, velocity, *acceleration_derivatives]
    adapter_duration = 2.0 * fit.half_width_ns
    response = polynomial_dipole_hertz_response_jet_native(
        observer_time_ns=float(observer_event.time_ns),
        observer_position_mm=observer_event.position_mm,
        magnetic_moment_native=float(magnetic_moment_native),
        segment_start_time_ns=root_time - 0.5 * adapter_duration,
        segment_duration_ns=adapter_duration,
        position_coefficients_mm=_centered_taylor_coefficients(
            position_derivatives,
            duration_ns=adapter_duration,
        ),
        rest_spin_coefficients=None,
        rest_spin_stereographic_coefficients=_centered_taylor_coefficients(
            spin_derivatives,
            duration_ns=adapter_duration,
        ),
        rest_spin_stereographic_frame=history.stereographic_frame,
        preserved_rest_spin_magnitude=None,
        retarded_time_ns=root_time,
    )
    measured_spread = None
    if model_spread is not None:
        if not (
            model_spread.narrow_fit.half_width_ns
            < fit.half_width_ns
            < model_spread.wide_fit.half_width_ns
        ):
            raise ValueError(
                "model-spread fits must have narrow < primary < wide half-widths"
            )
        comparison_responses = [response]
        for comparison_fit in (model_spread.narrow_fit, model_spread.wide_fit):
            comparison_response, _ = evaluate_causal_local_source_jet_native(
                history,
                observer_event,
                magnetic_moment_native=magnetic_moment_native,
                fit=comparison_fit,
                root_tolerance_mm=root_tolerance_mm,
                max_root_iterations=max_root_iterations,
                minimum_separation_mm=minimum_separation_mm,
            )
            comparison_responses.append(comparison_response)
        measured_spread = _response_model_spread(comparison_responses)
        if measured_spread.maximum > model_spread.maximum_relative_spread:
            raise CausalLocalSourceJetModelSpreadError(
                "local source-jet nested fits exceed their response-spread limit: "
                f"{measured_spread.maximum:.6e} > "
                f"{model_spread.maximum_relative_spread:.6e}"
            )
    return response, LocalSourceJetDiagnostics(
        root_segment_index=segment,
        acceleration_sample_indices=acceleration_indices,
        spin_sample_indices=selected_spin_indices,
        acceleration_condition_number=acceleration_condition,
        spin_condition_number=spin_condition,
        light_cone_residual_mm=residual,
        model_spread=measured_spread,
    )


def evaluate_causal_local_source_jet_collection_native(
    collection: CausalC5DipoleSourceCollection,
    observer_event: "ObserverEvent",
    *,
    fit: LocalSourceJetFitConfig,
    model_spread: LocalSourceJetModelSpreadConfig | None = None,
    excluded_source_identities: Sequence[str] = (),
    root_tolerance_mm: float = 1.0e-21,
    max_root_iterations: int = 96,
    minimum_separation_mm: float = 1.0e-15,
) -> CausalLocalSourceJetProviderResult:
    """Evaluate and sum local source jets in declared source order."""

    excluded = set(str(identity) for identity in excluded_source_identities)
    source_results: list[CausalLocalSourceJetEvaluation] = []
    potential = np.zeros(4, dtype=np.float64)
    partial_a = np.zeros((4, 4), dtype=np.float64)
    field = np.zeros((4, 4), dtype=np.float64)
    partial_f = np.zeros((4, 4, 4), dtype=np.float64)
    for source in collection.sources:
        if source.identity in excluded:
            continue
        try:
            response, diagnostics = evaluate_causal_local_source_jet_native(
                source.history,
                observer_event,
                magnetic_moment_native=source.magnetic_moment_native,
                fit=fit,
                model_spread=model_spread,
                root_tolerance_mm=root_tolerance_mm,
                max_root_iterations=max_root_iterations,
                minimum_separation_mm=minimum_separation_mm,
            )
        except CausalC5HistoryUnavailableError as exc:
            raise CausalC5HistoryUnavailableError(
                f"source identity {source.identity!r}: {exc}"
            ) from exc
        except ValueError as exc:
            raise ValueError(f"source identity {source.identity!r}: {exc}") from exc
        potential += response.four_potential
        partial_a += response.partial_a
        field += response.field_tensor
        partial_f += response.partial_f
        source_results.append(
            CausalLocalSourceJetEvaluation(
                identity=source.identity,
                response=response,
                diagnostics=diagnostics,
            )
        )
    electric, magnetic = fields_from_tensor_native(field)
    return CausalLocalSourceJetProviderResult(
        four_potential=potential,
        partial_a=partial_a,
        electric_field_native=electric,
        magnetic_field_native=magnetic,
        field_tensor=field,
        partial_f=partial_f,
        source_results=tuple(source_results),
    )


__all__ = [
    "CausalLocalSourceJetModelSpreadError",
    "CausalLocalSourceJetEvaluation",
    "CausalLocalSourceJetProviderResult",
    "LocalSourceJetDiagnostics",
    "LocalSourceJetFitConfig",
    "LocalSourceJetModelSpread",
    "LocalSourceJetModelSpreadConfig",
    "evaluate_causal_local_source_jet_collection_native",
    "evaluate_causal_local_source_jet_native",
]
