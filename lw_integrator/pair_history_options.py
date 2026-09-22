"""Explicit history selection without rewriting an evolved trajectory."""

import math
from typing import Any
import numpy as np

from core.full_dipole_history import FullDipoleHistory

HISTORY_METHODS = ("endpoint", "connected_direct", "connected_single_fit")
DEFAULT_DIPOLE_RELATIVE_BUDGET = 1e-10


def configure_startup_dipole_fit(payload, method=None):
    """Select fresh startup fitting without rewriting any evolved history."""
    from dataclasses import replace

    if method is None:
        return payload
    if method not in ("interpolate", "constrained"):
        raise ValueError("Unknown startup dipole fitting policy")
    recorded = payload["histories"]
    if all(h.get("startup_dipole_fit", "interpolate") == method for h in recorded):
        return payload
    if payload.get("accepted_steps") != 0 or any(
        h.get("inertial_until") is None or h["time"][-1] != h["inertial_until"]
        for h in recorded
    ):
        raise ValueError(
            "Resume preserves startup fitting; changes require fresh inertial-boundary data"
        )
    histories = [
        replace(
            FullDipoleHistory.from_checkpoint_payload(h),
            segments=(),
            startup_dipole_fit=method,
        )
        .completed()
        .to_checkpoint_payload()
        for h in recorded
    ]
    return dict(payload, histories=histories)


def configure_run_history(
    payload: dict[str, Any],
    method: str | None,
    budget: float | None,
    spacing: float | None,
    step_ns: float,
    relative_budget: float | None = None,
) -> dict[str, Any]:
    """Default compatible fresh fixed-grid runs; preserve evolved checkpoints."""
    histories = payload.get("histories", [])
    fresh = payload.get("accepted_steps") == 0
    if relative_budget is not None:
        if (
            isinstance(relative_budget, (bool, str))
            or not math.isfinite(relative_budget)
            or relative_budget <= 0
        ):
            raise ValueError(
                "Relative dipole drift allowance must be finite and positive"
            )
        if budget is not None:
            raise ValueError(
                "Choose either an absolute dipole budget or a relative allowance"
            )
        if not fresh:
            raise ValueError(
                "Resume preserves the recorded absolute dipole budgets; omit the relative allowance"
            )
    compatible = len(histories) >= 2 and all(
        h.get("dipole_reference") is not None
        and h.get("inertial_until") is not None
        and h.get("startup_fit") == "one_sided"
        and h.get("derivative_degree") == 10
        and h.get("integrate_velocity") is True
        and h.get("derivative_sampling", "consecutive") == "consecutive"
        and h["time"][-1] == h["inertial_until"]
        for h in histories
    )
    if method in (None, "auto") and fresh and compatible:
        method = "connected_single_fit"
        # A prepared single-fit setup already has its physical clock selected.
        if spacing is None and not all(
            h.get("fit_sample_spacing") is not None for h in histories
        ):
            spacing = step_ns
    elif method in (None, "auto", "preserve"):
        method = None
    selecting_connected = method in ("connected_single_fit", "connected_direct")
    if relative_budget is not None and not selecting_connected:
        raise ValueError(
            "Relative allowance requires fresh connected history selection"
        )
    derive = (
        fresh
        and selecting_connected
        and budget is None
        and (
            relative_budget is not None
            or any(h.get("dipole_tolerance") is None for h in histories)
        )
    )
    if derive:
        allowance = (
            DEFAULT_DIPOLE_RELATIVE_BUDGET
            if relative_budget is None
            else relative_budget
        )
        scales = [float(np.linalg.norm(h["dipole_reference"])) for h in histories]
        if any(not math.isfinite(s) or s <= 0 for s in scales):
            raise ValueError(
                "Zero or invalid initial dipole scale: supply an explicit absolute --dipole-drift-budget"
            )
        budgets = [allowance * scale for scale in scales]
        result = configure_history(payload, method, None, spacing, _budgets=budgets)
        return dict(
            result,
            dipole_budget_policy=dict(
                kind="initial_tensor_relative",
                relative_allowance=allowance,
                reference_norms_native=scales,
                absolute_budgets_native=budgets,
                scale="Frobenius norm of preserved initial dipole tensor in the simulation frame",
            ),
        )
    result = configure_history(payload, method, budget, spacing)
    if budget is not None and result is not payload:
        result = dict(
            result,
            dipole_budget_policy=dict(
                kind="explicit_absolute",
                absolute_budgets_native=[budget] * len(histories),
            ),
        )
    return result


def validate_recording_spacing(payload: dict[str, Any], step_ns: float) -> None:
    """Require each physical fit sample to lie on the recording grid."""
    for history in payload.get("histories", []):
        if history.get("geometry_reconstruction") != "connected_single_fit":
            continue
        spacing = history["fit_sample_spacing"]
        ratio = spacing / step_ns
        phase = (history["time"][-1] - history["inertial_until"]) / step_ns
        if not math.isclose(ratio, round(ratio), rel_tol=0, abs_tol=1e-8) or ratio < 1:
            raise ValueError(
                "History sample spacing must be an integer multiple of the recording timestep"
            )
        if not math.isclose(phase, round(phase), rel_tol=0, abs_tol=1e-8):
            raise ValueError(
                "Recording timestep does not align with the checkpoint history clock"
            )


def configure_history(
    payload: dict[str, Any],
    method: str | None = None,
    budget: float | None = None,
    spacing: float | None = None,
    *,
    _budgets: list[float] | None = None,
) -> dict[str, Any]:
    """Omitted options preserve checkpoints, including earlier history formats."""
    if method is None and budget is None and spacing is None and _budgets is None:
        return payload
    if spacing is not None and (
        isinstance(spacing, (bool, str)) or not math.isfinite(spacing) or spacing <= 0
    ):
        raise ValueError("History sample spacing must be finite and positive in ns")
    if method is not None and method not in HISTORY_METHODS:
        raise ValueError("Unknown source-history method")
    if budget is not None and (
        isinstance(budget, (bool, str)) or not math.isfinite(budget) or budget <= 0
    ):
        raise ValueError(
            "Dipole drift budget must be finite and positive in native dipole units"
        )
    histories = payload["histories"]
    if len(histories) < 2:
        raise ValueError("At least two source histories required")
    selected = []
    for index, recorded in enumerate(histories):
        current = recorded.get("dipole_reconstruction", "endpoint")
        if recorded.get("geometry_reconstruction") == "connected_single_fit":
            current = "connected_single_fit"
        target = current if method is None else method
        old_budget = recorded.get("dipole_tolerance")
        target_budget = old_budget if budget is None else budget
        if _budgets is not None:
            target_budget = _budgets[index]
            if not math.isfinite(target_budget) or target_budget <= 0:
                raise ValueError("Derived dipole budget must be finite and positive")
        old_spacing = recorded.get("fit_sample_spacing")
        target_spacing = old_spacing if spacing is None else spacing
        if target != "connected_single_fit" and target_spacing is not None:
            raise ValueError("History sample spacing requires connected_single_fit")
        if target == "connected_single_fit" and target_spacing is None:
            raise ValueError(
                "Single-fit history requires an explicit sample spacing in ns"
            )
        if target == "endpoint" and budget is not None:
            raise ValueError("Dipole drift budget requires connected_direct history")
        if payload.get("accepted_steps", 0) > 0 and (
            target != current
            or target_budget != old_budget
            or target_spacing != old_spacing
        ):
            raise ValueError(
                "Resume preserves the history method and drift budget. "
                "A budget-only diagnostic continuation requires a separately verified migration; "
                "no stored history was changed."
            )
        if (
            target == current
            and target_budget == old_budget
            and target_spacing == old_spacing
        ):
            selected.append(recorded)
            continue
        if target == "endpoint":
            raise ValueError(
                "Prepare fresh endpoint initial data; do not rewrite connected history"
            )
        if target_budget is None:
            raise ValueError(
                "Connected history requires an explicit native dipole drift budget"
            )
        history = FullDipoleHistory.from_checkpoint_payload(recorded)
        if target == "connected_single_fit":
            selected.append(
                history.with_single_fit_geometry(
                    fit_sample_spacing=target_spacing, dipole_tolerance=target_budget
                ).to_checkpoint_payload()
            )
            continue
        selected.append(
            history.with_connected_dipoles(target_budget).to_checkpoint_payload()
        )
    return dict(payload, histories=selected)
