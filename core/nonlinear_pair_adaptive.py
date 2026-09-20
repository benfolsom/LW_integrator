"""Shared-lab-time step doubling for the full-spin native pair.

Compare one step with two half steps, and retain the latter only when accepted.
This reuses the maintained error norm/controller, not legacy energy-jump rules.
It cannot certify the accuracy of source intervals already published in the past.
"""

import copy
from dataclasses import asdict, dataclass

import numpy as np

from . import momentum_center as model
from .momentum_center_pair import advance_pair, _restore
from .step_doubling import (
    ErrorScale,
    StepControllerConfig,
    StepDoublingState,
    StepDoublingTolerances,
    assess_step_doubling,
    propose_next_step_ns,
)

FORMAT = "adaptive-full-spin-pair-v1"


class AdaptiveAccuracyError(ValueError):
    """Accuracy refusal with trial evidence; no trial checkpoint is accepted."""

    def __init__(self, message, reports):
        super().__init__(message)
        self.reports = copy.deepcopy(reports)


@dataclass(frozen=True)
class AdaptivePairSettings:
    position_mm: ErrorScale
    momentum_native: ErrorScale
    spin_native: ErrorScale
    impulse_native: ErrorScale
    torque_native: ErrorScale
    mass_exchange: ErrorScale
    minimum_interval_ns: float
    maximum_interval_ns: float
    maximum_attempts: int = 8

    def __post_init__(self):
        for name in (
            "position_mm",
            "momentum_native",
            "spin_native",
            "impulse_native",
            "torque_native",
            "mass_exchange",
        ):
            if not isinstance(getattr(self, name), ErrorScale):
                raise ValueError("Explicit physical error scales required")
        if (
            not np.isfinite([self.minimum_interval_ns, self.maximum_interval_ns]).all()
            or not 0 < self.minimum_interval_ns <= self.maximum_interval_ns
        ):
            raise ValueError("Positive ordered adaptive interval bounds required")
        if (
            type(self.maximum_attempts) is not int
            or not 1 <= self.maximum_attempts <= 32
        ):
            raise ValueError("One to 32 adaptive attempts required")


def _settings(value):
    value = dict(value)
    for key in (
        "position_mm",
        "momentum_native",
        "spin_native",
        "impulse_native",
        "torque_native",
        "mass_exchange",
    ):
        value[key] = ErrorScale(**value[key])
    return AdaptivePairSettings(**value)


def initialize_adaptive(payload, settings, next_interval_ns):
    if any(
        h.get("geometry_reconstruction") == "connected_single_fit"
        for h in payload.get("histories", [])
    ):
        raise ValueError(
            "Single-fit history requires fixed recording intervals; use the fixed runner with internal DOP853 error control"
        )
    if "internal_step_control" in payload:
        raise ValueError(
            "Use the native checkpoint runner for internal error control; stacking both adaptive controllers is not supported"
        )
    _restore(payload)
    if any(
        p["reaction_mode"] not in ("off", "full_dipole_coupled", "full_dipole_rr")
        for p in payload["particles"]
    ):
        raise ValueError(
            "Adaptive full-spin adapter supports off and full-dipole reaction modes"
        )
    if (
        not isinstance(settings, AdaptivePairSettings)
        or not np.isfinite(next_interval_ns)
        or not settings.minimum_interval_ns
        <= next_interval_ns
        <= settings.maximum_interval_ns
    ):
        raise ValueError("Initial adaptive interval must be within declared bounds")
    return dict(
        format=FORMAT,
        native_payload=copy.deepcopy(payload),
        settings=asdict(settings),
        next_interval_ns=float(next_interval_ns),
        accepted_intervals=0,
        rejected_trials=0,
    )


def _physical_state(payload, records):
    states = np.asarray(payload["states"])
    momenta = np.array([p["kinetic_momentum_native"] for p in records[-1]["particles"]])
    spins = []
    for state, p in zip(states, momenta):
        mass2 = model.dot(p, p)
        if mass2 <= 0 or p[0] <= 0:
            raise ValueError(
                "Adaptive spin comparison requires timelike kinetic momentum"
            )
        mass = np.sqrt(mass2)
        boost = np.eye(4)
        boost[0, 0] = p[0] / mass
        boost[0, 1:] = boost[1:, 0] = -p[1:] / mass
        boost[1:, 1:] += np.outer(p[1:], p[1:]) / (mass * (p[0] + mass))
        rest = boost @ model.unpack(state[8:]) @ boost.T
        spins.append([rest[2, 3], rest[3, 1], rest[1, 2]])
    return states[:, 1:4], momenta, np.asarray(spins)


def _increments(payload):
    impulses, torques, exchanges = [], [], []
    for row in payload.get("dipole_reaction_ledger", [None, None]):
        if row is None:
            impulses.append(np.zeros(16))
            torques.append(np.zeros(24))
            exchanges.append(0.0)
            continue
        impulses.append(
            np.concatenate(
                [
                    row["applied_impulse_native"],
                    *[
                        row["sectors"][k]["impulse_native"]
                        for k in ("q_squared", "qD", "D_squared")
                    ],
                ]
            )
        )
        torques.append(
            np.concatenate(
                [
                    row["applied_torque_native"],
                    *[
                        row["sectors"][k]["torque_native"]
                        for k in ("q_squared", "qD", "D_squared")
                    ],
                ]
            )
        )
        exchanges.append(row["external_action_mass_exchange"])
    return (
        np.asarray(impulses, dtype=float),
        np.asarray(torques, dtype=float),
        np.asarray(exchanges, dtype=float),
    )


def _diagnostics(full, fine, settings):
    output = [[], []]
    for a, b, scale in zip(
        _increments(full),
        _increments(fine),
        (settings.impulse_native, settings.torque_native, settings.mass_exchange),
    ):
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("Reaction increments must be finite with matching shapes")
        denominator = scale.absolute + scale.relative * np.maximum(np.abs(a), np.abs(b))
        for target, value in zip(output, (a, b)):
            target.extend(
                np.divide(
                    value, denominator, out=np.zeros_like(value), where=denominator > 0
                ).ravel()
            )
    return np.asarray(output)


def _add_ledgers(old, new):
    if isinstance(new, dict):
        return {
            key: _add_ledgers(
                old.get(key, {} if isinstance(value, dict) else 0.0), value
            )
            for key, value in new.items()
        }
    if isinstance(new, list):
        if isinstance(old, list):
            return [_add_ledgers(a, b) for a, b in zip(old, new)]
        return copy.deepcopy(new)
    return old + new


def _zero_ledger(value):
    """Keep the schema for native validation, rejecting invalid numeric totals."""
    if isinstance(value, dict):
        return {key: _zero_ledger(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_zero_ledger(item) for item in value]
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (float, int, np.floating, np.integer))
        or not np.isfinite(value)
    ):
        raise ValueError("Accumulated reaction ledger must contain finite numbers")
    return 0.0


def advance_adaptive_interval(checkpoint):
    """Return one accepted interval and trial report, without modifying input.

    History/domain/root failures propagate: they are not disguised as local
    truncation errors. Interval bounds apply to the whole comparison; the
    accepted path contains two native half steps. Order one is used for the
    error estimate until whole-history convergence establishes a higher order.
    """
    if checkpoint.get("format") != FORMAT:
        raise ValueError("Wrong adaptive full-spin checkpoint format")
    settings = _settings(checkpoint["settings"])
    original = checkpoint["native_payload"]
    validated = initialize_adaptive(original, settings, checkpoint["next_interval_ns"])
    for key in ("accepted_intervals", "rejected_trials"):
        if type(checkpoint[key]) is not int or checkpoint[key] < 0:
            raise ValueError("Nonnegative adaptive counts required")
    # Compare interval increments without subtracting large accumulated totals.
    base = copy.deepcopy(original)
    for key in ("reaction_ledger", "dipole_reaction_ledger"):
        if key in base:
            base[key] = _zero_ledger(base[key])
    width = validated["next_interval_ns"]
    reports = []
    for attempt in range(settings.maximum_attempts):
        full, full_records = advance_pair(base, width, 1)
        fine, fine_records = advance_pair(base, width / 2, 2)
        diagnostics = _diagnostics(full, fine, settings)
        a = StepDoublingState(*_physical_state(full, full_records), diagnostics[0])
        b = StepDoublingState(*_physical_state(fine, fine_records), diagnostics[1])
        assessment = assess_step_doubling(
            a,
            b,
            method_order=1,
            tolerances=StepDoublingTolerances(
                settings.position_mm,
                settings.momentum_native,
                settings.spin_native,
                ErrorScale(1.0, 0.0),
            ),
        )
        proposed = propose_next_step_ns(
            width,
            assessment.normalized_error,
            accepted=assessment.accepted,
            config=StepControllerConfig(
                method_order=1, minimum_factor=0.5, maximum_growth_factor=1.25
            ),
            minimum_step_ns=settings.minimum_interval_ns,
            maximum_step_ns=settings.maximum_interval_ns,
        )
        reports.append(dict(interval_ns=width, assessment=asdict(assessment)))
        if assessment.accepted:
            for key in ("reaction_ledger", "dipole_reaction_ledger"):
                if key in fine and key in original:
                    fine[key] = _add_ledgers(original[key], fine[key])
                elif key in original:
                    fine[key] = copy.deepcopy(original[key])
            result = dict(
                checkpoint,
                native_payload=fine,
                next_interval_ns=proposed,
                accepted_intervals=checkpoint["accepted_intervals"] + 1,
                rejected_trials=checkpoint["rejected_trials"] + attempt,
            )
            return result, reports
        if proposed >= width:
            raise AdaptiveAccuracyError(
                "Adaptive accuracy unmet at minimum interval", reports
            )
        width = proposed
    raise AdaptiveAccuracyError("Adaptive accuracy unmet within trial budget", reports)
