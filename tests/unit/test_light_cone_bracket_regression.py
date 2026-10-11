"""Cut-knot bracket continuity and off-axis pre-merge field comparisons."""

import json
import math
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest

from core.constants import C_MMNS
from core.light_cone_history import endpoint_geometry
from core.light_cone_roots import null_residual
from core.retarded_fields import (
    ObserverEvent,
    RetardedHistoryError,
    _find_retarded_knot_bracket,
    _prepare_history,
    evaluate_retarded_charge_field_native,
)
from scripts.bracket_regression.reproduce import history_from_event
from scripts.bracket_regression.sweep import analytic_field, setup

pytestmark = pytest.mark.unit
DATA = Path(__file__).parent / "data"
FAILURE = json.loads((DATA / "bracket_failure_event.json").read_text())
PREMERGE = json.loads((DATA / "bracket_premerge_fields.json").read_text())


@pytest.mark.parametrize("backend", ["python", "numba_full_strict_serial"])
@pytest.mark.parametrize("representation", ["light_cone_quintic", "proper_velocity"])
def test_saved_cut_event_evaluates(backend, representation):
    history = history_from_event(FAILURE)
    if representation == "proper_velocity":
        beta = FAILURE["source_beta"]
        proper = beta / np.sqrt((1 - beta) * (1 + beta))
        for row in history:
            row.update(
                source_u_x=np.zeros(1),
                source_u_y=np.zeros(1),
                source_u_z=np.array([proper]),
                source_kinematics_ready=np.ones(1),
                source_history_mode=np.ones(1),
            )
            # The constrained representation honors accepted displacements.
            # Supply the missing sub-ulp coordinates of this ballistic line,
            # rather than making it accelerate to fit rounded short chords.
            with localcontext() as context:
                context.prec = 90
                d = Decimal.from_float
                exact_beta = d(proper) / (1 + d(proper) ** 2).sqrt()
                remainder = (
                    1
                    + exact_beta * d(C_MMNS) * d(float(row["t"][0]))
                    - d(float(row["z"][0]))
                )
                low = float(remainder)
                row["source_position_low_z"] = np.array([low])
                row["source_position_tail_z"] = np.array([float(remainder - d(low))])
    event = ObserverEvent(
        FAILURE["observer_time_ns"], tuple(FAILURE["observer_position_mm"])
    )
    result = evaluate_retarded_charge_field_native(history, event, backend=backend)
    assert result.valid_sources.tolist() == [True]
    # Captured by reproduce.py with the read-only 8d42ba6 source archive.
    old_electric = np.array([-0.000134336679303947, 0.0, 0.0006148495625378983])
    electric, magnetic = analytic_field(event, FAILURE["source_beta"], 1.0)
    # The reconstruction uses one negative elementary source charge.
    charge = history[0]["q_source"][0]
    np.testing.assert_allclose(result.electric_field_native, old_electric, rtol=2e-13)
    np.testing.assert_allclose(
        result.electric_field_native, charge * electric, rtol=2e-13
    )
    np.testing.assert_allclose(
        result.magnetic_field_native, charge * magnetic, rtol=2e-13
    )
    assert -1e-17 < result.retarded_time_ns[0] < 1e-17


def test_cut_knot_has_the_same_sign_from_either_segment():
    history = history_from_event(FAILURE)
    source = _prepare_history(history, ()).sources[0]
    kwargs = dict(
        observer_time_ns=FAILURE["observer_time_ns"],
        observer_position_mm=np.array(FAILURE["observer_position_mm"]),
    )
    index = _find_retarded_knot_bracket(source, **kwargs)
    assert index == 159
    previous, following = source.light_cone_segments[index : index + 2]
    geometry = endpoint_geometry(previous, **kwargs, include_endpoint=True)
    # The old arithmetic rejects the selected segment even though the next
    # segment's accepted knot is already behind the observer's light cone.
    assert (
        null_residual(
            previous.coefficients_mm, C_MMNS * previous.duration_ns, 1.0, geometry[:5]
        )[0]
        > 0.0
    )
    upper = null_residual(
        previous.coefficients_mm, C_MMNS * previous.duration_ns, 1.0, geometry
    )[0]
    lower = null_residual(
        following.coefficients_mm,
        C_MMNS * following.duration_ns,
        0.0,
        endpoint_geometry(following, **kwargs),
    )[0]
    assert upper == lower < 0.0


@pytest.mark.parametrize("backend", ["python", "numba_full_strict_serial"])
def test_rotating_source_shared_knot_uses_scan_residual(backend):
    # Minimal three-knot version of the slow nonperiodic spin-flux failure.
    # Adjacent segments use distinct frames, so even their endpoint-anchored
    # norms assign opposite signs to the same observer ray.
    omega, radius, charge = 50.0, 0.03, 0.8
    period = 2 * math.pi / omega
    times = np.linspace(-0.25 * period, 1.25 * period, 1537)[335:338]
    history = []
    for time in times:
        radial = np.array([math.cos(omega * time), math.sin(omega * time), 0.0])
        tangent = np.array([-radial[1], radial[0], 0.0])
        position = radius * radial
        beta = radius * omega / C_MMNS * tangent
        prime = -radius * omega**2 / C_MMNS**2 * radial
        row = {
            "t": np.array([time]),
            "q": np.array([charge]),
            "q_source": np.array([charge]),
        }
        for index, axis in enumerate("xyz"):
            row[axis] = np.array([position[index]])
            row[f"b{axis}"] = np.array([beta[index]])
            row[f"bdot{axis}"] = np.array([prime[index]])
        history.append(row)
    event = ObserverEvent(
        1.344129676632352,
        (-252.9822128134703, 3.0981385716323095e-14, -309.83866769659335),
    )
    source = _prepare_history(
        history, (), source_acceleration_semantics="instantaneous"
    ).sources[0]
    kwargs = dict(
        observer_time_ns=event.time_ns, observer_position_mm=np.array(event.position_mm)
    )
    assert _find_retarded_knot_bracket(source, **kwargs) == 0
    segment = source.light_cone_segments[0]
    geometry = endpoint_geometry(segment, **kwargs, include_endpoint=True)
    assert (
        null_residual(
            segment.coefficients_mm, C_MMNS * segment.duration_ns, 1.0, geometry
        )[0]
        > 0
    )

    result = evaluate_retarded_charge_field_native(
        history, event, backend=backend, source_acceleration_semantics="instantaneous"
    )
    assert result.valid_sources.tolist() == [True]
    # Ordinary analytic LW field at the circular source's shared knot.
    knot = history[1]
    separation = np.array(event.position_mm) - np.array([knot[a][0] for a in "xyz"])
    distance = np.linalg.norm(separation)
    n = separation / distance
    beta = np.array([knot[f"b{a}"][0] for a in "xyz"])
    prime = np.array([knot[f"bdot{a}"][0] for a in "xyz"])
    kappa = 1 - n @ beta
    electric = charge * (
        (1 - beta @ beta) * (n - beta) / (kappa**3 * distance**2)
        + np.cross(n, np.cross(n - beta, prime)) / (kappa**3 * distance)
    )
    np.testing.assert_allclose(result.electric_field_native, electric, rtol=2e-13)
    np.testing.assert_allclose(
        result.magnetic_field_native, np.cross(n, electric), rtol=2e-13
    )


@pytest.mark.parametrize("backend", ["python", "numba_full_strict_serial"])
@pytest.mark.parametrize("row", PREMERGE, ids=lambda row: str(row["case"]))
def test_off_axis_sweep_against_premerge_and_ballistic(row, backend):
    history, event, beta, anchor = setup(row["case"])
    result = evaluate_retarded_charge_field_native(history, event, backend=backend)
    electric, magnetic = analytic_field(event, beta, anchor)
    # These are historical beta-only histories with rounded macroscopic knots.
    # Their coordinate uncertainty is amplified by gamma squared. The resolved
    # high-gamma histories retain their separate, stricter existing tests.
    input_roundoff = max(2e-13, 8 * np.finfo(float).eps * row["case"]["gamma"] ** 2)
    for actual, reference in (
        (result.electric_field_native, electric),
        (result.magnetic_field_native, magnetic),
    ):
        assert np.linalg.norm(actual - reference) <= input_roundoff * np.linalg.norm(
            reference
        )
    # Allow the archived provider's independently measured oracle error in
    # this comparison; it is less accurate than the new provider at some cuts.
    for actual, reference in (
        (result.electric_field_native, np.array(row["electric"])),
        (result.magnetic_field_native, np.array(row["magnetic"])),
    ):
        bound = input_roundoff + row["analytic_error"]
        assert np.linalg.norm(actual - reference) <= bound * np.linalg.norm(reference)


@pytest.mark.parametrize("backend", ["python", "numba_full_strict_serial"])
def test_reanchoring_does_not_supply_missing_future_history(backend):
    history = history_from_event(FAILURE)
    with pytest.raises(RetardedHistoryError):
        evaluate_retarded_charge_field_native(
            history,
            ObserverEvent(time_ns=1.0, position_mm=(0.0145, 0.0, 0.4667)),
            backend=backend,
        )
