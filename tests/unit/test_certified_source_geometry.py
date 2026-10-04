"""Byte-level differential checks against the retained Decimal oracles."""

from dataclasses import asdict
from decimal import Decimal, localcontext
import json
from pathlib import Path
import pickle

import numpy as np
import pytest

from core.ballistic_charge import (
    _compiled_velocity_constants,
    ballistic_retarded_point,
    ballistic_retarded_point_decimal,
)
from core.certified_ballistic import ballistic_geometry_certified
from core.certified_source_geometry import _add, _float, _mul, _round
from core.light_cone_history import (
    endpoint_geometry,
    endpoint_geometry_decimal,
    separation_in_velocity_frame,
    separation_in_velocity_frame_decimal,
)
from core.retarded_fields import _prepare_history
from scripts.bracket_regression.reproduce import history_from_event
from scripts.bracket_regression.sweep import setup

pytestmark = pytest.mark.unit


def same(left, right):
    if isinstance(left, np.ndarray):
        assert left.dtype == right.dtype and left.shape == right.shape
        assert left.tobytes() == right.tobytes()
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            same(left[key], right[key])
    else:
        assert pickle.dumps(left, protocol=4) == pickle.dumps(right, protocol=4)


def test_random_ballistic_differential():
    rng = np.random.default_rng(301743)
    certified_count = 0
    for index in range(2000):
        scale = 10.0 ** rng.uniform(-6, 40)
        u = rng.normal(size=3)
        if index % 4 == 0:
            u[:2] = 0.0
        u *= 10.0 ** rng.uniform(-2, 11) / np.linalg.norm(u)
        kwargs = dict(
            observer_time_ns=float(rng.normal() * scale / 300),
            anchor_time_ns=float(rng.normal() * scale / 300),
            observer_position_mm=rng.normal(size=3) * scale,
            anchor_position_mm=rng.normal(size=3) * scale,
            source_proper_velocity=u,
            observer_time_low_ns=float(rng.normal() * scale * 1e-18),
            observer_time_tail_ns=float(rng.normal() * scale * 1e-35),
            observer_position_low_mm=rng.normal(size=3) * scale * 1e-16,
            observer_position_tail_mm=rng.normal(size=3) * scale * 1e-33,
        )
        reference = ballistic_retarded_point_decimal(**kwargs)
        same(asdict(ballistic_retarded_point(**kwargs)), asdict(reference))
        _, certified = ballistic_geometry_certified(
            kwargs["observer_time_ns"],
            kwargs["observer_position_mm"],
            kwargs["anchor_time_ns"],
            kwargs["anchor_position_mm"],
            u,
            kwargs["observer_time_low_ns"],
            kwargs["observer_position_low_mm"],
            kwargs["observer_time_tail_ns"],
            kwargs["observer_position_tail_mm"],
            _compiled_velocity_constants(tuple(u)),
        )
        certified_count += certified
    assert certified_count > 1500


def test_random_endpoint_and_rotation_differential():
    rng = np.random.default_rng(73801)
    for index in range(400):
        case = dict(
            gamma=float(10.0 ** rng.uniform(0.01, 7)),
            side=1 if index % 2 else -1,
            u=float(10.0 ** rng.uniform(-3, 2)),
            kind="cut" if index % 3 else "present",
        )
        # The regression setup supplies valid resolved frames and high-gamma
        # source lines, rather than testing arbitrary nonphysical frame arrays.
        history, event, _, _ = setup(case)
        segment = (
            _prepare_history(history, ()).sources[0].light_cone_segments[index % 2]
        )
        kwargs = dict(
            observer_time_ns=event.time_ns,
            observer_position_mm=np.asarray(event.position_mm),
            observer_time_low_ns=float(rng.normal() * 1e-17),
            observer_time_tail_ns=float(rng.normal() * 1e-34),
            observer_position_low_mm=rng.normal(size=3) * 1e-16,
            observer_position_tail_mm=rng.normal(size=3) * 1e-33,
            include_endpoint=True,
        )
        same(
            endpoint_geometry(segment, **kwargs),
            endpoint_geometry_decimal(segment, **kwargs),
        )
        u = rng.normal(size=3) * 10.0 ** rng.uniform(-2, 12)
        separation = rng.normal(size=3) * 10.0 ** rng.uniform(-10, 10)
        same(
            separation_in_velocity_frame(segment, separation, u),
            separation_in_velocity_frame_decimal(segment, separation, u),
        )


def test_saved_cut_geometry_by_bytes():
    event = json.loads(
        (Path(__file__).parent / "data/bracket_failure_event.json").read_text()
    )
    source = _prepare_history(history_from_event(event), ()).sources[0]
    kwargs = dict(
        observer_time_ns=event["observer_time_ns"],
        observer_position_mm=np.asarray(event["observer_position_mm"]),
        include_endpoint=True,
    )
    for segment in source.light_cone_segments[158:162]:
        same(
            endpoint_geometry(segment, **kwargs),
            endpoint_geometry_decimal(segment, **kwargs),
        )


def test_error_enclosures_and_midpoint_fallback():
    rng = np.random.default_rng(8243)
    for _ in range(1000):
        a, b = rng.normal(size=2) * 10.0 ** rng.uniform(-90, 90)
        for operation, decimal_operation in (
            (_add, lambda x, y: x + y),
            (_mul, lambda x, y: x * y),
        ):
            h, l, error = operation(_float(a), _float(b))
            with localcontext() as context:
                context.prec = 80
                reference = decimal_operation(
                    Decimal.from_float(a), Decimal.from_float(b)
                )
                context.prec = 400
                assert abs(
                    reference - Decimal.from_float(h) - Decimal.from_float(l)
                ) <= Decimal.from_float(error)
    assert not _round((1.0, 2.0**-53, 0.0))[1]
    assert not _round((0.0, 0.0, 1e-280))[1]


@pytest.mark.parametrize("u", [np.zeros(3), np.array([0.0, 0.0, -1e12])])
def test_zero_and_axis_velocity(u):
    kwargs = dict(
        observer_time_ns=0.1,
        observer_position_mm=np.array([1.0, 0.0, 3.0]),
        anchor_time_ns=-1.0,
        anchor_position_mm=np.zeros(3),
        source_proper_velocity=u,
    )
    same(
        asdict(ballistic_retarded_point(**kwargs)),
        asdict(ballistic_retarded_point_decimal(**kwargs)),
    )


def test_integer_time_inputs_use_the_same_binary64_arithmetic():
    kwargs = dict(
        observer_time_ns=0,
        observer_position_mm=np.array([1.0, 2.0, 3.0]),
        anchor_time_ns=0,
        anchor_position_mm=np.zeros(3),
        source_proper_velocity=np.array([3.7, 5.1, 7.3]),
    )
    same(
        asdict(ballistic_retarded_point(**kwargs)),
        asdict(ballistic_retarded_point_decimal(**kwargs)),
    )


@pytest.mark.parametrize("sign", [-1.0, 1.0])
@pytest.mark.parametrize("observer_sign", [-1.0, 1.0])
def test_signed_zero_transverse_differential(sign, observer_sign):
    kwargs = dict(
        observer_time_ns=0.1,
        observer_position_mm=np.array([np.copysign(0.0, observer_sign), 0.0, 3.0]),
        anchor_time_ns=-1.0,
        anchor_position_mm=np.zeros(3),
        source_proper_velocity=np.array([np.copysign(0.0, sign), 0.0, sign * 10.0]),
    )
    same(
        asdict(ballistic_retarded_point(**kwargs)),
        asdict(ballistic_retarded_point_decimal(**kwargs)),
    )


@pytest.mark.parametrize("replacement", [np.nan, np.inf, 1e101])
def test_invalid_ballistic_inputs_keep_reference_error(replacement):
    kwargs = dict(
        observer_time_ns=0.1,
        observer_position_mm=np.array([replacement, 0.0, 3.0]),
        anchor_time_ns=-1.0,
        anchor_position_mm=np.zeros(3),
        source_proper_velocity=np.array([0.0, 0.0, 10.0]),
    )
    with pytest.raises(ValueError) as reference:
        ballistic_retarded_point_decimal(**kwargs)
    with pytest.raises(ValueError) as actual:
        ballistic_retarded_point(**kwargs)
    assert str(reference.value) == str(actual.value)


@pytest.mark.parametrize(
    "centered,on_shell", [(False, False), (False, True), (True, False), (True, True)]
)
def test_random_drift_remainders_differential(centered, on_shell):
    import copy
    from core.constants import C_MMNS
    from core.resolved_knot import (
        preserve_drift_remainders,
        preserve_drift_remainders_decimal,
    )

    rng = np.random.default_rng(39107)
    for _ in range(300):
        start = rng.normal(size=3) * 10.0 ** rng.uniform(-2, 11)
        end = start + rng.normal(size=3) * 1e-5
        gamma = np.hypot(1.0, np.linalg.norm(start))
        h = 10.0 ** rng.uniform(-10, -2)
        current = {"gamma": np.array([gamma]), "source_kinematics_ready": np.ones(1)}
        result = {"gamma": np.array([np.hypot(1.0, np.linalg.norm(end))])}
        for i, axis in enumerate(("t", "x", "y", "z")):
            current[axis] = np.array([rng.normal() * 100.0])
            if i == 0:
                increment = 0.5 * h * (gamma + result["gamma"][0])
                prefix, suffix = "source_time", "ns"
            else:
                increment = 0.5 * h * C_MMNS * (start[i - 1] + end[i - 1])
                prefix, suffix = "source_position", axis
                current[f"source_u_{axis}"] = np.array([start[i - 1]])
            result[axis] = current[axis] + increment
            current[f"{prefix}_low_{suffix}"] = np.array([rng.normal() * 1e-15])
            current[f"{prefix}_tail_{suffix}"] = np.array([rng.normal() * 1e-32])
        reference = copy.deepcopy(result)
        preserve_drift_remainders(
            result, current, 0, h, end, centered=centered, on_shell=on_shell
        )
        preserve_drift_remainders_decimal(
            reference, current, 0, h, end, centered=centered, on_shell=on_shell
        )
        same(result, reference)


def test_quad_operation_enclosures_against_90_digits():
    import core.certified_ballistic as quad

    rng = np.random.default_rng(6331)
    for _ in range(500):
        x, y = 10.0 ** rng.uniform(-80, 80, size=2)
        for result, operation in [
            (quad._add(quad._float(x), quad._float(y)), lambda a, b: a + b),
            (quad._mul(quad._float(x), quad._float(y)), lambda a, b: a * b),
            (quad._div(quad._float(x), quad._float(y)), lambda a, b: a / b),
            (quad._sqrt(quad._float(x)), lambda a, b: a.sqrt()),
        ]:
            with localcontext() as context:
                context.prec = 90
                reference = operation(Decimal.from_float(x), Decimal.from_float(y))
                context.prec = 600
                center = sum((Decimal.from_float(v) for v in result[:4]), Decimal(0))
                assert abs(reference - center) <= Decimal.from_float(result[4])


def test_cached_velocity_constants_keep_signed_zero_keys():
    positive = _compiled_velocity_constants((0.0, 0.0, 10.0))
    negative = _compiled_velocity_constants((-0.0, 0.0, 10.0))
    assert not np.signbit(positive[2, 0])
    assert np.signbit(negative[2, 0])
