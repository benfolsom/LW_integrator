"""
Legacy-style unit tests for the original `LienardWiechertIntegrator` helpers.

These exercises cover behaviour that used to live in the monolithic integrator
class (charge conservation, static stepping, etc.).  They are preserved here to
validate the archival physics implementation when it is still available.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    from core.trajectory_integrator import LienardWiechertIntegrator
except ImportError:
    pytest.skip(
        "LienardWiechertIntegrator wrapper is not available in the modern core;"
        " skipping legacy integrator unit tests.",
        allow_module_level=True,
    )

try:
    from physics.constants import C_MMNS  # type: ignore[attr-defined]
    from physics.particle_initialization import ELEMENTARY_CHARGE  # type: ignore[attr-defined]
    from physics.simulation_types import SimulationConfig  # type: ignore[attr-defined]
except ImportError:
    from archive.physics.constants import C_MMNS  # type: ignore[attr-defined]
    from archive.physics.particle_initialization import (  # type: ignore[attr-defined]
        ELEMENTARY_CHARGE,
    )
    from archive.physics.simulation_types import SimulationConfig  # type: ignore[attr-defined]

from tests.test_config import PROTON, TestConfiguration, create_bunch_uniform_distribution


class TestTrajectoryIntegratorUnits:
    """Unit tests for individual integrator methods."""

    def setup_method(self) -> None:
        self.integrator = LienardWiechertIntegrator()
        self.test_particle = {
            "x": np.array([0.0]),
            "y": np.array([0.0]),
            "z": np.array([0.0]),
            "Px": np.array([0.0]),
            "Py": np.array([0.0]),
            "Pz": np.array([100.0]),
            "Pt": np.array([100.0]),
            "mass": np.array([938.3]),
            "q": np.array([ELEMENTARY_CHARGE]),
            "gamma": np.array([100.0 / 938.3]),
            "bx": np.array([0.0]),
            "by": np.array([0.0]),
            "bz": np.array([1.0]),
            "bdotx": np.array([0.0]),
            "bdoty": np.array([0.0]),
            "bdotz": np.array([0.0]),
            "t": np.array([0.0]),
        }

    @pytest.mark.physics
    def test_integrator_initialization(self) -> None:
        integrator = LienardWiechertIntegrator()
        assert hasattr(integrator, "c_mmns")
        assert integrator.c_mmns == C_MMNS

        config = SimulationConfig()
        integrator_with_config = LienardWiechertIntegrator(config)
        assert integrator_with_config.config is not None

    @pytest.mark.physics
    def test_static_step_calculation(self) -> None:
        initial_state = self.test_particle.copy()
        h_step = 1e-5

        final_state = self.integrator.equations_of_motion_static_internal(  # type: ignore[attr-defined]
            h_step, initial_state, 0
        )

        expected_z = initial_state["z"][0] + initial_state["bz"][0] * C_MMNS * h_step
        assert np.isclose(final_state["z"][0], expected_z, rtol=1e-10)
        assert np.isclose(final_state["Pt"][0], initial_state["Pt"][0], rtol=1e-10)
        assert np.isclose(
            final_state["t"][0], initial_state["t"][0] + h_step, rtol=1e-10
        )

    @pytest.mark.physics
    def test_charge_conservation(self) -> None:
        config = TestConfiguration(
            particle_count=5,
            transverse_separation=10.0,
            starting_distance=100.0,
            step_size=1e-5,
            total_steps=10,
            sim_type=2,
            wall_z=1e5,
            aperture_r=1e5,
            z_cutoff=0.0,
        )

        bunch = create_bunch_uniform_distribution(config, PROTON, "line")
        initial_charge = np.sum(bunch["q"])

        trajectory, _ = self.integrator.integrate_retarded_fields(  # type: ignore[attr-defined]
            static_steps=5,
            ret_steps=5,
            h_step=config.step_size,
            wall_Z=config.wall_z,
            apt_R=config.aperture_r,
            sim_type=config.sim_type,
            init_rider=bunch,
            init_driver=bunch.copy(),
            bunch_dist=1e5,
            z_cutoff=config.z_cutoff,
        )

        for state in trajectory:
            final_charge = np.sum(state["q"])
            assert np.isclose(final_charge, initial_charge, rtol=1e-12)

    @pytest.mark.physics
    def test_momentum_units_consistency(self) -> None:
        config = TestConfiguration(
            particle_count=3,
            transverse_separation=5.0,
            starting_distance=120.0,
            step_size=1e-5,
            total_steps=5,
            sim_type=2,
            wall_z=1e5,
            aperture_r=1e5,
            z_cutoff=0.0,
        )

        bunch = create_bunch_uniform_distribution(config, PROTON, "line")

        trajectory, _ = self.integrator.integrate_retarded_fields(  # type: ignore[attr-defined]
            static_steps=2,
            ret_steps=3,
            h_step=config.step_size,
            wall_Z=config.wall_z,
            apt_R=config.aperture_r,
            sim_type=config.sim_type,
            init_rider=bunch,
            init_driver=bunch.copy(),
            bunch_dist=1e5,
            z_cutoff=config.z_cutoff,
        )

        for state in trajectory:
            Pt = state["Pt"]
            Px = state["Px"]
            Py = state["Py"]
            Pz = state["Pz"]
            mass = state.get("m", state.get("mass"))
            gamma = state["gamma"]

            assert np.allclose(Pt**2, Px**2 + Py**2 + Pz**2, rtol=1e-6, atol=1e-6)
            assert np.all(gamma >= 1.0)
            assert np.all(mass > 0)

    @pytest.mark.physics
    def test_bunch_energy_progression(self) -> None:
        config = TestConfiguration(
            particle_count=2,
            transverse_separation=1.0,
            starting_distance=100.0,
            step_size=5e-6,
            total_steps=8,
            sim_type=2,
            wall_z=1e5,
            aperture_r=1e5,
            z_cutoff=0.0,
        )

        bunch = create_bunch_uniform_distribution(config, PROTON, "line")
        trajectory, _ = self.integrator.integrate_retarded_fields(  # type: ignore[attr-defined]
            static_steps=3,
            ret_steps=config.total_steps - 3,
            h_step=config.step_size,
            wall_Z=config.wall_z,
            apt_R=config.aperture_r,
            sim_type=config.sim_type,
            init_rider=bunch,
            init_driver=bunch.copy(),
            bunch_dist=1e5,
            z_cutoff=config.z_cutoff,
        )

        pt_series = [np.sum(state["Pt"]) for state in trajectory]
        assert all(np.isfinite(pt) for pt in pt_series)
        assert pt_series[-1] <= pt_series[0] * 1.1
