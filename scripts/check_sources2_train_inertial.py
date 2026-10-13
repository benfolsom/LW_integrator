"""Independent inertial Lorentz-impulse control for the train probe.

This control integrates the closed uniform-motion field analytically. It does
not evolve coupled trajectories or apply radiation reaction, and is not a
replacement for the Medina/LAD timestep study.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.constants import C_MMNS
from core.exact_source_cloud import exact_cloud_history, transverse_offsets
from core.integration_runner import (
    _build_driver_train_initial_state,
    _build_inertial_coasting_history,
    _estimate_inertial_prehistory_duration_ns,
)
from core.retarded_fields import ObserverEvent, evaluate_retarded_charge_field_native
from core.types import (
    DriverTrainConfig,
    MacroparticleSmearingConfig,
    MagneticDipoleConfig,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    inputs = json.loads((args.directory / "resolved_inputs.json").read_text())
    states = {}
    for role in ("rider", "driver"):
        states[role] = {
            key: np.asarray(value) if isinstance(value, list) else value
            for key, value in inputs["init_" + role].items()
        }
    rider = states["rider"]
    driver = _build_driver_train_initial_state(
        states["driver"], DriverTrainConfig(**inputs["driver_train"])
    )
    assert len(rider["x"]) == 1
    assert not np.any(rider["bx"]) and not np.any(rider["by"])
    assert not np.any(driver["bx"]) and not np.any(driver["by"])
    assert np.all(driver["x"] == driver["x"][0])
    assert np.all(driver["y"] == driver["y"][0])
    duration = (inputs["steps"] - 1) * inputs["h_step"] * float(rider["gamma"][0])
    endpoint = copy.deepcopy(driver)
    endpoint["t"] += duration
    endpoint["z"] += C_MMNS * driver["bz"] * duration
    prehistory = duration + _estimate_inertial_prehistory_duration_ns(
        rider, driver, MagneticDipoleConfig(enabled=True)
    )
    history = _build_inertial_coasting_history(endpoint, prehistory, knot_count=4)
    config = MacroparticleSmearingConfig(
        enabled=True,
        subcharge_count=16,
        position_sigma_mm=0.1,
        longitudinal_sigma_mm=0,
        momentum_sigma_amu_mm_ns=0,
        use_momentum_errors=False,
    )
    offsets, fractions = transverse_offsets(np.array([0, 0, driver["bz"][0]]), config)
    impact = np.array([rider["x"][0] - driver["x"][0], rider["y"][0] - driver["y"][0]])
    radius2 = np.sum((offsets[:, :2] - impact) ** 2, axis=1)
    gamma = float(driver["gamma"][0])
    vr = float(rider["bz"][0]) * C_MMNS
    vd = float(driver["bz"][0]) * C_MMNS
    relative_speed = vr - vd
    crossing = (driver["z"][0] - rider["z"][0]) / relative_speed
    width_time = config.position_sigma_mm / (gamma * abs(relative_speed))
    times = [0.0, duration] + [
        crossing + k * width_time for k in (-10, -3, -1, -0.5, 0.5, 1, 3, 10)
    ]
    oracle = []
    wrapped = exact_cloud_history(history, config)
    for time in times:
        event = ObserverEvent(
            float(time),
            (
                float(rider["x"][0]),
                float(rider["y"][0]),
                float(rider["z"][0] + vr * time),
            ),
        )
        actual = evaluate_retarded_charge_field_native(
            wrapped, event, backend="numba_full_strict_serial"
        ).electric_field_native[2]
        dz = event.position_mm[2] - (driver["z"] + vd * time)
        terms = (
            driver["q_source"][:, None]
            * fractions[None, :]
            / gamma**2
            * dz[:, None]
            / (dz[:, None] ** 2 + radius2[None, :] / gamma**2) ** 1.5
        )
        expected = float(terms.sum())
        difference = abs(actual - expected)
        assert difference <= 2e-7 * abs(expected) + 1e-14, (time, actual, expected)
        oracle.append(
            dict(
                time_ns=float(time),
                exact_field_z=float(actual),
                uniform_motion_field_z=expected,
                absolute_difference=float(difference),
            )
        )
    impulse = []
    for width in (0.05, 0.1, 0.2):
        scaled_radius2 = np.sum((offsets[:, :2] * (width / 0.1) - impact) ** 2, axis=1)
        first = rider["z"][0] - driver["z"]
        last = first + relative_speed * duration
        primitive_difference = 1 / np.sqrt(
            last[:, None] ** 2 + scaled_radius2[None, :] / gamma**2
        ) - 1 / np.sqrt(first[:, None] ** 2 + scaled_radius2[None, :] / gamma**2)
        per_bunch = (
            -float(rider["q_observer"][0])
            * driver["q_source"]
            / gamma**2
            / relative_speed
            * np.sum(fractions[None, :] * primitive_difference, axis=1)
        )
        impulse.append(
            dict(
                width_mm=width,
                per_driver_bunch_impulse_z_native=per_bunch.tolist(),
                rider_impulse_z_native=float(per_bunch.sum()),
            )
        )
    # At the first centre crossing the longitudinal primitive contains
    # E[1/r]. For a continuous two-dimensional Gaussian this is
    # sqrt(pi/2)/sigma. Even tensor Hermite rules approach this slowly.
    peaks = []
    continuous_peak = None
    if not np.any(impact):
        first_separation = float(rider["z"][0] - driver["z"][0])
        coefficient = (
            -float(rider["q_observer"][0])
            * float(driver["q_source"][0])
            / gamma**2
            / relative_speed
        )
        width = 0.1
        initial_continuous = (
            1
            / abs(first_separation)
            * (1 - width**2 / (gamma**2 * first_separation**2))
        )
        continuous_peak = coefficient * (
            gamma * math.sqrt(math.pi / 2) / width - initial_continuous
        )
        peaks = []
        for count in (4, 16, 36, 64, 100):
            nodes, weights = transverse_offsets(
                np.array([0, 0, driver["bz"][0]]),
                MacroparticleSmearingConfig(
                    enabled=True,
                    subcharge_count=count,
                    position_sigma_mm=width,
                    longitudinal_sigma_mm=0,
                    momentum_sigma_amu_mm_ns=0,
                    use_momentum_errors=False,
                ),
            )
            radii = np.linalg.norm(nodes, axis=1)
            initial = np.sum(
                weights / np.sqrt(first_separation**2 + radii**2 / gamma**2)
            )
            value = coefficient * (np.sum(weights * gamma / radii) - initial)
            peaks.append(
                dict(
                    count=count,
                    first_bunch_inertial_peak_impulse_native=float(value),
                    relative_difference_from_continuous_gaussian=float(
                        abs(value - continuous_peak) / abs(continuous_peak)
                    ),
                )
            )
    result = dict(
        model="Prescribed inertial centres; analytic lab-time longitudinal Lorentz impulse. No coupled acceleration or radiation reaction.",
        duration_ns=duration,
        prehistory_ns=prehistory,
        first_encounter_ns=float(crossing),
        width_time_ns=float(width_time),
        initial_h_lab_ns=float(rider["gamma"][0] * inputs["h_step"]),
        samples_per_width=float(width_time / (rider["gamma"][0] * inputs["h_step"])),
        field_oracle=oracle,
        impulses=impulse,
        first_bunch_continuous_gaussian_peak_native=continuous_peak,
        peak_count_study=peaks,
    )
    (args.directory / "inertial_impulse_control.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(
        json.dumps(
            dict(
                duration_ns=duration,
                first_encounter_ns=float(crossing),
                impulses=impulse,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
