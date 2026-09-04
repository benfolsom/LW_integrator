"""Resolved electromagnetic force on a finite translating shell source.

This diagnostic module is deliberately separate from the live equations of
motion. It evaluates the charge-generated field at every material node,
omits only that same point charge, and integrates the resulting Lorentz
four-force on one central Fermi slice.

The integral includes ``d tau_node / d tau_center``. Without this lapse, node
forces defined per unit node proper time cannot be added as a derivative with
respect to the center's proper time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import C_MMNS
from .retarded_fields import (
    ObserverEvent,
    evaluate_retarded_charge_field_native,
    evaluate_retarded_mutual_charge_field_matrix_native,
)
from .rfs import electromagnetic_field_tensor_native, rfs_four_force_native
from .translating_magnetic_shell_kinematics import TranslatingMagneticShellHistory


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=float).copy()
    if not np.all(np.isfinite(result)):
        raise ValueError("finite-source force result contains a nonfinite value")
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class FiniteMagneticSourceForceSlice:
    """Node fields and integrated force on one central proper-time slice."""

    boundary_condition: str
    center_proper_time_ns: float
    electric_field_native: np.ndarray
    magnetic_field_native: np.ndarray
    node_four_force_native: np.ndarray
    integrated_four_force_native: np.ndarray
    maximum_light_cone_residual_mm: float
    pair_electric_field_native: np.ndarray | None = None
    pair_magnetic_field_native: np.ndarray | None = None

    def __post_init__(self) -> None:
        count = np.asarray(self.electric_field_native).shape[0]
        for name, shape in (
            ("electric_field_native", (count, 3)),
            ("magnetic_field_native", (count, 3)),
            ("node_four_force_native", (count, 4)),
            ("integrated_four_force_native", (4,)),
        ):
            value = np.asarray(getattr(self, name))
            if value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
            object.__setattr__(self, name, _readonly(value))
        if self.boundary_condition not in {"retarded", "advanced"}:
            raise ValueError("boundary_condition must be 'retarded' or 'advanced'")
        for name in ("pair_electric_field_native", "pair_magnetic_field_native"):
            value = getattr(self, name)
            if value is None:
                continue
            array = np.asarray(value)
            if array.shape != (count, count, 3):
                raise ValueError(f"{name} must have shape {(count, count, 3)}")
            object.__setattr__(self, name, _readonly(array))
        scalars = (self.center_proper_time_ns, self.maximum_light_cone_residual_mm)
        if not np.all(np.isfinite(scalars)) or self.maximum_light_cone_residual_mm < 0:
            raise ValueError("force-slice diagnostics must be finite and nonnegative")


@dataclass(frozen=True)
class FiniteMagneticSourceForceSplit:
    """Retarded/advanced force split with an independent pairwise reduction."""

    retarded: FiniteMagneticSourceForceSlice
    advanced: FiniteMagneticSourceForceSlice
    time_symmetric_four_force_native: np.ndarray
    radiation_reaction_four_force_native: np.ndarray
    pairwise_radiation_reaction_four_force_native: np.ndarray
    pairwise_reduction_difference_native: np.ndarray

    def __post_init__(self) -> None:
        for name in (
            "time_symmetric_four_force_native",
            "radiation_reaction_four_force_native",
            "pairwise_radiation_reaction_four_force_native",
            "pairwise_reduction_difference_native",
        ):
            value = np.asarray(getattr(self, name))
            if value.shape != (4,):
                raise ValueError(f"{name} must have shape (4,)")
            object.__setattr__(self, name, _readonly(value))


def _time_reversed_charge_history(
    history: list[dict[str, np.ndarray]],
) -> list[dict[str, np.ndarray]]:
    """Return the time-reversed source used to construct advanced fields."""

    result = []
    for state in reversed(history):
        result.append(
            {
                "t": -state["t"],
                "x": state["x"],
                "y": state["y"],
                "z": state["z"],
                "bx": -state["bx"],
                "by": -state["by"],
                "bz": -state["bz"],
                # beta_T(t)=-beta(-t), so its coordinate-time derivative is
                # even under time reversal.
                "bdotx": state["bdotx"],
                "bdoty": state["bdoty"],
                "bdotz": state["bdotz"],
                "q": state["q"],
                "q_source": state["q_source"],
                "_dead_particles": state["_dead_particles"],
            }
        )
    return result


def evaluate_finite_magnetic_source_force_slice_native(
    history: TranslatingMagneticShellHistory,
    *,
    slice_index: int,
    boundary_condition: str = "retarded",
    backend: str = "python",
) -> FiniteMagneticSourceForceSlice:
    """Integrate resolved charge Lorentz forces on one Fermi slice.

    The advanced field is obtained by evaluating the retarded solver on the
    time-reversed source. Electric fields are even and magnetic fields are odd
    under this transformation. The same-node exclusion removes the point
    singularity; convergence under surface and shell-size refinement remains
    a required physical acceptance test rather than an assumption here.
    """

    boundary = str(boundary_condition).strip().lower()
    if boundary not in {"retarded", "advanced"}:
        raise ValueError("boundary_condition must be 'retarded' or 'advanced'")
    index = int(slice_index)
    step_count, node_count = history.event_time_ns.shape
    if index < 0:
        index += step_count
    if index < 0 or index >= step_count:
        raise IndexError("slice_index is outside the shell history")

    provider_history = history.as_charge_provider_history()
    evaluation_history = (
        provider_history
        if boundary == "retarded"
        else _time_reversed_charge_history(provider_history)
    )
    electric = np.empty((node_count, 3), dtype=float)
    magnetic = np.empty_like(electric)
    maximum_residual = 0.0
    event_time_sign = 1.0 if boundary == "retarded" else -1.0
    events = tuple(
        ObserverEvent(
            time_ns=event_time_sign * float(history.event_time_ns[index, node]),
            position_mm=tuple(history.position_mm[index, node]),
        )
        for node in range(node_count)
    )
    pair_electric = None
    pair_magnetic = None
    if str(backend).strip().lower() == "python":
        matrix = evaluate_retarded_mutual_charge_field_matrix_native(
            evaluation_history,
            events,
            source_acceleration_semantics="instantaneous",
        )
        pair_electric = matrix.electric_field_native
        pair_magnetic = (
            matrix.magnetic_field_native
            if boundary == "retarded"
            else -matrix.magnetic_field_native
        )
        electric.fill(0.0)
        magnetic.fill(0.0)
        for source in range(node_count):
            electric += pair_electric[:, source]
            magnetic += pair_magnetic[:, source]
        valid_sources = matrix.valid_sources
        residuals = matrix.light_cone_residual_mm
    else:
        fields = tuple(
            evaluate_retarded_charge_field_native(
                evaluation_history,
                events[node],
                excluded_source_indices=(node,),
                backend=backend,
                source_acceleration_semantics="instantaneous",
            )
            for node in range(node_count)
        )
        valid_sources = np.stack([field.valid_sources for field in fields])
        residuals = np.stack([field.light_cone_residual_mm for field in fields])
        for node, field in enumerate(fields):
            electric[node] = field.electric_field_native
            magnetic[node] = (
                field.magnetic_field_native
                if boundary == "retarded"
                else -field.magnetic_field_native
            )
    for node in range(node_count):
        expected_valid = np.ones(node_count, dtype=bool)
        expected_valid[node] = False
        if not np.array_equal(valid_sources[node], expected_valid):
            raise ArithmeticError("a non-self source was omitted from a node field")
        maximum_residual = max(
            maximum_residual,
            float(np.max(np.abs(residuals[node, valid_sources[node]]))),
        )

    beta = history.beta[index]
    gamma = 1.0 / np.sqrt(1.0 - np.sum(beta**2, axis=1))
    four_velocity = (
        gamma[:, np.newaxis] * C_MMNS * np.column_stack((np.ones(node_count), beta))
    )
    zero_spin = np.zeros(4)
    zero_gradient = np.zeros((4, 4, 4))
    node_force = np.empty((node_count, 4), dtype=float)
    for node in range(node_count):
        node_force[node] = rfs_four_force_native(
            four_velocity_mm_ns=four_velocity[node],
            spin_four_vector=zero_spin,
            field_tensor=electromagnetic_field_tensor_native(
                electric[node], magnetic[node]
            ),
            partial_f=zero_gradient,
            charge_native=float(history.charge_native[node]),
            magnetic_moment_native=0.0,
        )
    integrated_force = np.einsum(
        "n,nm->m", history.material_proper_time_lapse[index], node_force
    )
    return FiniteMagneticSourceForceSlice(
        boundary_condition=boundary,
        center_proper_time_ns=float(history.center_proper_time_ns[index]),
        electric_field_native=electric,
        magnetic_field_native=magnetic,
        node_four_force_native=node_force,
        integrated_four_force_native=integrated_force,
        maximum_light_cone_residual_mm=maximum_residual,
        pair_electric_field_native=pair_electric,
        pair_magnetic_field_native=pair_magnetic,
    )


def evaluate_finite_magnetic_source_force_split_native(
    history: TranslatingMagneticShellHistory,
    *,
    slice_index: int,
) -> FiniteMagneticSourceForceSplit:
    """Evaluate the force split and reduce its radiative part pair by pair."""

    retarded = evaluate_finite_magnetic_source_force_slice_native(
        history,
        slice_index=slice_index,
        boundary_condition="retarded",
        backend="python",
    )
    advanced = evaluate_finite_magnetic_source_force_slice_native(
        history,
        slice_index=slice_index,
        boundary_condition="advanced",
        backend="python",
    )
    assert retarded.pair_electric_field_native is not None
    assert retarded.pair_magnetic_field_native is not None
    assert advanced.pair_electric_field_native is not None
    assert advanced.pair_magnetic_field_native is not None
    pair_electric = 0.5 * (
        retarded.pair_electric_field_native - advanced.pair_electric_field_native
    )
    pair_magnetic = 0.5 * (
        retarded.pair_magnetic_field_native - advanced.pair_magnetic_field_native
    )
    index = int(slice_index)
    if index < 0:
        index += history.center_proper_time_ns.size
    beta = history.beta[index]
    gamma = 1.0 / np.sqrt(1.0 - np.sum(beta**2, axis=1))
    four_velocity = (
        gamma[:, None] * C_MMNS * np.column_stack((np.ones(beta.shape[0]), beta))
    )
    pairwise_node_force = np.zeros((beta.shape[0], 4))
    zero_spin = np.zeros(4)
    zero_gradient = np.zeros((4, 4, 4))
    for observer in range(beta.shape[0]):
        for source in range(beta.shape[0]):
            if observer == source:
                continue
            pairwise_node_force[observer] += rfs_four_force_native(
                four_velocity_mm_ns=four_velocity[observer],
                spin_four_vector=zero_spin,
                field_tensor=electromagnetic_field_tensor_native(
                    pair_electric[observer, source],
                    pair_magnetic[observer, source],
                ),
                partial_f=zero_gradient,
                charge_native=float(history.charge_native[observer]),
                magnetic_moment_native=0.0,
            )
    pairwise_radiative = np.einsum(
        "n,nm->m", history.material_proper_time_lapse[index], pairwise_node_force
    )
    symmetric = 0.5 * (
        retarded.integrated_four_force_native + advanced.integrated_four_force_native
    )
    radiative = 0.5 * (
        retarded.integrated_four_force_native - advanced.integrated_four_force_native
    )
    return FiniteMagneticSourceForceSplit(
        retarded=retarded,
        advanced=advanced,
        time_symmetric_four_force_native=symmetric,
        radiation_reaction_four_force_native=radiative,
        pairwise_radiation_reaction_four_force_native=pairwise_radiative,
        pairwise_reduction_difference_native=pairwise_radiative - radiative,
    )


__all__ = [
    "FiniteMagneticSourceForceSplit",
    "FiniteMagneticSourceForceSlice",
    "evaluate_finite_magnetic_source_force_split_native",
    "evaluate_finite_magnetic_source_force_slice_native",
]
