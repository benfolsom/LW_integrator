"""Independent charge, mass-shell, and finite-energy checks of the runner."""

from __future__ import annotations

import numpy as np
import pytest

from core.constants import C_MMNS
from core.self_consistency import SelfConsistencyConfig
from core.trajectory_integrator import retarded_integrator
from core.types import SimulationType
from tests.test_config import (
    PROTON,
    TestConfiguration,
    create_bunch_uniform_distribution,
)


class TestTrajectoryIntegratorUnits:
    """Physical invariants formerly checked through the mutable facade."""

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

        trajectory, _, *_ = retarded_integrator(
            steps=10,
            h_step=config.step_size,
            wall_z=config.wall_z,
            aperture_radius=config.aperture_r,
            sim_type=SimulationType(config.sim_type),
            init_rider=bunch,
            init_driver=bunch.copy(),
            mean=1e5,
            z_cutoff=config.z_cutoff,
            cav_spacing=0.0,
            self_consistency=SelfConsistencyConfig(),
            radiation_reaction_mode="medina_lad",
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

        trajectory, _, *_ = retarded_integrator(
            steps=5,
            h_step=config.step_size,
            wall_z=config.wall_z,
            aperture_radius=config.aperture_r,
            sim_type=SimulationType(config.sim_type),
            init_rider=bunch,
            init_driver=bunch.copy(),
            mean=1e5,
            z_cutoff=config.z_cutoff,
            cav_spacing=0.0,
            self_consistency=SelfConsistencyConfig(),
            radiation_reaction_mode="medina_lad",
        )

        for state in trajectory:
            Pt = state["Pt"]
            Px = state["Px"]
            Py = state["Py"]
            Pz = state["Pz"]
            mass = state.get("m", state.get("mass"))
            gamma = state["gamma"]

            # These particles have nonzero rest mass: E²/c² = p² + m²c².
            assert np.allclose(
                Pt**2,
                Px**2 + Py**2 + Pz**2 + (mass * C_MMNS) ** 2,
                rtol=1e-6,
                atol=1e-6,
            )
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
        trajectory, _, *_ = retarded_integrator(
            steps=config.total_steps,
            h_step=config.step_size,
            wall_z=config.wall_z,
            aperture_radius=config.aperture_r,
            sim_type=SimulationType(config.sim_type),
            init_rider=bunch,
            init_driver=bunch.copy(),
            mean=1e5,
            z_cutoff=config.z_cutoff,
            cav_spacing=0.0,
            self_consistency=SelfConsistencyConfig(),
            radiation_reaction_mode="medina_lad",
        )

        pt_series = [np.sum(state["Pt"]) for state in trajectory]
        assert all(np.isfinite(pt) for pt in pt_series)
        assert pt_series[-1] <= pt_series[0] * 1.1
