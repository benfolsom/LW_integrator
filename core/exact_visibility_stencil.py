"""One-sided dipole Hertz stencils on one ideal-gate branch.

Near a gate, axis-aligned spatial stencils can have no safe side (a tangent
light cone is an example). Timelike tilted coordinate directions give a
one-sided retarded-time stencil; the chain rule maps derivatives back to
the original laboratory coordinates. Every stencil event re-solves its root.
"""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from itertools import combinations_with_replacement, product
from math import fsum, prod

import numpy as np

from .beamline_geometry import compute_directional_visibility_mask
from .constants import C_MMNS
from .exact_visibility import ExactVisibilityHistory
from .retarded_fields import ObserverEvent, _solve_retarded_sample, _sum_source_terms


def _displaced(event, offset, step):
    from .retarded_fields import _shift_resolved_coordinate

    time = _shift_resolved_coordinate(
        float(event.time_ns),
        float(event.time_low_ns),
        float(event.time_tail_ns),
        float(offset[0] * step / C_MMNS),
    )
    positions = [
        _shift_resolved_coordinate(
            float(high), float(low), float(tail), float(shift * step)
        )
        for high, low, tail, shift in zip(
            event.position_mm, event.position_low_mm, event.position_tail_mm, offset[1:]
        )
    ]
    return ObserverEvent(
        time[0],
        tuple(p[0] for p in positions),
        time[1],
        tuple(p[1] for p in positions),
        time[2],
        tuple(p[2] for p in positions),
    )


def gate_dipole_stencil(
    prepared,
    history,
    event,
    center,
    step,
    *,
    potential_only,
    require_complete_history,
    minimum_separation_mm,
    root_tolerance_mm,
    max_root_iterations,
):
    """Return a one-sided result only when a centered stencil crosses a gate."""
    if not isinstance(history, ExactVisibilityHistory) or not prepared.sources:
        return None
    if history.visible_indices is not None:
        # The integrator splits at the boundary and differentiates an explicit
        # smooth branch. Trial steps may extend that branch past the event while
        # bracketing it; no gate mask is differentiated at their stencil nodes.
        return None
    from .retarded_dipole_fields import (
        RetardedDipoleFieldGradientResult,
        RetardedDipolePotentialResult,
        _evaluate_prepared_hertz_tensor_native,
        _full_gradient_stencil_offsets,
    )
    from .rfs import fields_from_tensor_native

    def visible(source, offset, width):
        displaced = _displaced(event, offset, width)
        sample = _solve_retarded_sample(
            source.worldline,
            observer_time_ns=displaced.time_ns,
            observer_position_mm=np.asarray(displaced.position_mm),
            observer_time_low_ns=displaced.time_low_ns,
            observer_position_low_mm=np.asarray(displaced.position_low_mm),
            observer_time_tail_ns=displaced.time_tail_ns,
            observer_position_tail_mm=np.asarray(displaced.position_tail_mm),
            root_tolerance_mm=root_tolerance_mm,
            max_root_iterations=max_root_iterations,
        )
        return sample is not None and bool(
            compute_directional_visibility_mask(
                sample.position_mm[None, :], history.geometry, history.direction
            )[0]
        )

    centered_offsets = _full_gradient_stencil_offsets()
    if all(
        visible(source, offset, step)
        for source in prepared.sources.values()
        for offset in centered_offsets
        if any(offset)
    ):
        return None

    # b0 = s e0, bi = ei + 2s e0. Every positive combination is timelike,
    # hence advances retarded source time monotonically in the chosen direction.
    # d/dx0 = s D0; d/dxi = Di - 2 D0.
    lattice = {
        tuple(offset)
        for degree in (1, 2, 3)
        for indices in combinations_with_replacement(range(4), degree)
        for nodes in product((0, 1, 2), repeat=degree)
        for offset in [np.bincount(indices, weights=nodes, minlength=4).astype(int)]
    }

    def physical(offset, sign):
        return (sign * (offset[0] + 2 * sum(offset[1:])), *offset[1:])

    signs = {}
    for _ in range(20):
        signs.clear()
        for index, source in prepared.sources.items():
            for sign in (-1, 1):
                if all(
                    visible(source, physical(offset, sign), step)
                    for offset in lattice
                    if any(offset)
                ):
                    signs[index] = sign
                    break
        if len(signs) == len(prepared.sources):
            break
        step *= 0.5
    else:
        raise ValueError("no one-sided dipole stencil fits the visible gate branch")

    n = prepared.arrays.n_sources
    potential_terms = np.zeros((n, 4))
    partial_terms = np.zeros((n, 4, 4))
    gradient_terms = np.zeros((n, 4, 4, 4))
    diagnostics = {}
    for index, source in prepared.sources.items():
        single = replace(prepared, sources={index: source})
        sign = signs[index]

        @lru_cache(maxsize=None)
        def hertz(offset):
            offset = physical(offset, sign)
            result = _evaluate_prepared_hertz_tensor_native(
                single,
                _displaced(event, offset, step),
                require_complete_history=require_complete_history,
                minimum_separation_mm=minimum_separation_mm,
                root_tolerance_mm=root_tolerance_mm,
                max_root_iterations=max_root_iterations,
            )
            times, residuals = diagnostics.setdefault(
                offset, (np.full(n, np.nan), np.full(n, np.nan))
            )
            times[index] = result.retarded_time_ns[index]
            residuals[index] = result.light_cone_residual_mm[index]
            return result.hertz_tensor

        @lru_cache(maxsize=None)
        def derivative(indices):
            operators = [
                ((0, sign),) if axis == 0 else ((axis, 1), (0, -2)) for axis in indices
            ]
            weights = {}
            for combination in product(*operators):
                operator_weight = prod(weight for _, weight in combination)
                for nodes in product(((0, -3), (1, 4), (2, -1)), repeat=len(indices)):
                    offset = [0] * 4
                    weight = operator_weight
                    for (axis, _), (node, coefficient) in zip(combination, nodes):
                        offset[axis] += node
                        weight *= coefficient
                    key = tuple(offset)
                    weights[key] = weights.get(key, 0) + weight
            # Combine integer chain-rule weights before evaluating any floating
            # differences. Nested differentiated sums amplify third-derivative
            # cancellation unnecessarily near the one-sided branch boundary.
            terms = [
                weight * hertz(offset) for offset, weight in weights.items() if weight
            ]
            return np.array(
                [
                    fsum(term[mu, nu] for term in terms)
                    for mu, nu in product(range(4), repeat=2)
                ]
            ).reshape(4, 4) / (2 * step) ** len(indices)

        for mu in range(4):
            potential_terms[index, mu] = sum(
                derivative((nu,))[mu, nu] for nu in range(4)
            )
        if not potential_only:
            for axis in range(4):
                for mu in range(4):
                    partial_terms[index, axis, mu] = sum(
                        derivative(tuple(sorted((axis, nu))))[mu, nu] for nu in range(4)
                    )
            metric = (1, -1, -1, -1)
            for axis, mu, nu in product(range(4), repeat=3):
                gradient_terms[index, axis, mu, nu] = sum(
                    metric[mu] * derivative(tuple(sorted((axis, mu, rho))))[nu, rho]
                    - metric[nu] * derivative(tuple(sorted((axis, nu, rho))))[mu, rho]
                    for rho in range(4)
                )
    offsets = sorted(diagnostics)
    metadata = dict(
        four_potential=_sum_source_terms(potential_terms),
        hertz=center,
        stencil_step_mm=step,
        stencil_offsets=np.array(offsets, dtype=int),
        stencil_retarded_time_ns=np.stack([diagnostics[o][0] for o in offsets]),
        stencil_light_cone_residual_mm=np.stack([diagnostics[o][1] for o in offsets]),
    )
    if potential_only:
        return RetardedDipolePotentialResult(**metadata)
    partial_a = _sum_source_terms(partial_terms)
    tensor = np.zeros((4, 4))
    metric = (1, -1, -1, -1)
    for mu, nu in product(range(4), repeat=2):
        tensor[mu, nu] = metric[mu] * partial_a[mu, nu] - metric[nu] * partial_a[nu, mu]
    electric, magnetic = fields_from_tensor_native(tensor)
    return RetardedDipoleFieldGradientResult(
        **metadata,
        partial_a=partial_a,
        field_tensor=tensor,
        electric_field_native=electric,
        magnetic_field_native=magnetic,
        partial_f=_sum_source_terms(gradient_terms),
        lorenz_gauge_residual_per_mm=float(np.trace(partial_a)),
    )
