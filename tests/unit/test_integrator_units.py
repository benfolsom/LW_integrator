"""Unit tests for the :class:`core.trajectory_integrator` compatibility shim."""

from __future__ import annotations

import numpy as np
import pytest

from core.trajectory_integrator import LienardWiechertIntegrator
from tests.test_config import (
    BASIC_TWO_PARTICLE,
    PROTON,
    TestConfiguration,
    create_bunch_uniform_distribution,
)


def _basic_config() -> TestConfiguration:
    return TestConfiguration(
        particle_count=2,
        transverse_separation=5.0,
        starting_distance=100.0,
        step_size=1e-5,
        total_steps=4,
        sim_type=2,
        wall_z=1e5,
        aperture_r=1e5,
        z_cutoff=0.0,
    )


@pytest.mark.unit
def test_integrate_retarded_fields_respects_step_count() -> None:
    integrator = LienardWiechertIntegrator()
    config = _basic_config()
    bunch = create_bunch_uniform_distribution(config, PROTON, "line")

    trajectory, driver = integrator.integrate_retarded_fields(
        static_steps=2,
        ret_steps=2,
        h_step=config.step_size,
        wall_Z=config.wall_z,
        apt_R=config.aperture_r,
        sim_type=config.sim_type,
        init_rider=bunch,
        init_driver=bunch.copy(),
        bunch_dist=1e5,
        z_cutoff=config.z_cutoff,
    )

    assert len(trajectory) == 4
    assert len(driver) == 4
    assert np.all(np.isfinite(trajectory[0]["Pt"]))


@pytest.mark.unit
def test_integrate_retarded_fields_does_not_modify_inputs() -> None:
    integrator = LienardWiechertIntegrator()
    config = BASIC_TWO_PARTICLE
    bunch = create_bunch_uniform_distribution(config, PROTON, "line")
    original_pt = bunch["Pt"].copy()

    _ = integrator.integrate_retarded_fields(
        static_steps=1,
        ret_steps=2,
        h_step=config.step_size,
        wall_Z=config.wall_z,
        apt_R=config.aperture_r,
        sim_type=config.sim_type,
        init_rider=bunch,
        init_driver=bunch.copy(),
        bunch_dist=1e5,
        z_cutoff=config.z_cutoff,
    )

    assert np.array_equal(bunch["Pt"], original_pt)
