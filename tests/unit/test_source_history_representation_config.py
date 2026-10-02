"""Public configuration, rejection, and serialization of interpolation choices."""

import pytest

from core.integration_runner import retarded_integrator
from core.types import MagneticDipoleConfig, SimulationType, StartupMode
from lw_integrator.cli import (
    _build_integrator_config,
    _merge_simulation_payload,
    parse_args,
)
from lw_integrator.testbed_runner import SimulationOptions
from tests.unit.test_inertial_prehistory import _state

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("mode", ["light_cone_quintic", "proper_velocity"])
def test_cli_and_gui_options_share_representation(mode):
    options = SimulationOptions(source_history_representation=mode)
    serialized = options.to_dict()
    restored = SimulationOptions.from_dict(serialized)
    assert restored.source_history_representation == mode
    args = parse_args(["--source-history-representation", mode])
    merged = _merge_simulation_payload(serialized, args)
    config = _build_integrator_config(merged)
    assert (
        config.source_history_representation == restored.source_history_representation
    )


def test_unknown_representation_is_rejected():
    with pytest.raises(ValueError, match="source_history_representation"):
        SimulationOptions.from_dict({"source_history_representation": "typo"})
    args = parse_args([])
    payload = _merge_simulation_payload({"source_history_representation": "typo"}, args)
    with pytest.raises(ValueError, match="source_history_representation"):
        _build_integrator_config(payload)


@pytest.mark.parametrize(
    "backend,startup,error",
    [
        ("metal_certified_full_strict", StartupMode.INERTIAL_PREHISTORY, "Metal"),
        ("numba_full_strict_serial", StartupMode.COLD_START, "exact scalar"),
    ],
)
def test_unsupported_runner_combination_fails_before_integration(
    backend, startup, error
):
    with pytest.raises(ValueError, match=error):
        retarded_integrator(
            steps=2,
            h_step=1e-6,
            wall_z=0.0,
            aperture_radius=1e6,
            sim_type=SimulationType.BUNCH_TO_BUNCH,
            init_rider=_state(position_mm=(0.0, 0.0, 0.0)),
            init_driver=_state(position_mm=(1.0, 0.0, 0.0)),
            mean=1000.0,
            cav_spacing=0.0,
            z_cutoff=0.0,
            startup_mode=startup,
            source_history_representation="proper_velocity",
            magnetic_dipole=MagneticDipoleConfig(
                enabled=True, exact_retarded_backend=backend
            ),
        )


def test_checkpoint_rejects_representation_change(tmp_path):
    from core.integration_checkpoint import (
        IntegrationCheckpointStore,
        CheckpointCompatibilityError,
    )
    from tests.unit.test_high_gamma_stored_history import _coast

    history, _ = _coast(1e12, steps=12, representation="proper_velocity")
    arguments = dict(
        total_steps=12,
        requested_steps=12,
        active_start=0,
        interval_steps=1,
        interval_seconds=0.0,
    )
    directory = tmp_path / "checkpoint"
    store = IntegrationCheckpointStore(
        directory,
        compatibility_payload={"source_history_representation": "proper_velocity"},
        resume=False,
        **arguments,
    )
    store.write(step_index=11, rider=history, driver=history, loop_state={})
    with pytest.raises(CheckpointCompatibilityError, match="fingerprint"):
        IntegrationCheckpointStore(
            directory,
            compatibility_payload={
                "source_history_representation": "light_cone_quintic"
            },
            resume=True,
            **arguments,
        )
    from core.integration_runner import _slice_trajectory_arrays

    sliced = _slice_trajectory_arrays(history, 2, 8)
    assert sliced is not None
    assert sliced.source_history_mode.shape == sliced.t.shape
    assert (sliced.source_history_mode == 1).all()


@pytest.mark.parametrize("mode", ["light_cone_quintic", "proper_velocity"])
def test_sweep_and_gui_conversion_retain_interpolation_choice(tmp_path, mode):
    from optimization.config import OptimizationConfig
    from optimization.single_integration_helpers import build_single_integration_setup

    options = SimulationOptions(source_history_representation=mode)
    config = OptimizationConfig.from_simulation_options(options)
    assert config.source_history_representation == mode
    converted = build_single_integration_setup(
        config,
        aperture=1.0,
        energy_gev=1.0,
        start_z=0.0,
        transv_offset=0.0,
        timestep=1e-6,
        steps=12,
        run_output_dir=tmp_path,
        run_num=0,
        driver_params=None,
    )
    assert converted.options.source_history_representation == mode


def test_proper_velocity_cloud_rejected_before_checkpoint_creation(tmp_path):
    from core.types import CheckpointConfig, MacroparticleSmearingConfig

    checkpoint = tmp_path / "unused"
    with pytest.raises(ValueError, match="clouds do not support.*proper_velocity"):
        retarded_integrator(
            steps=2,
            h_step=1e-6,
            wall_z=0.0,
            aperture_radius=1e6,
            sim_type=SimulationType.BUNCH_TO_BUNCH,
            init_rider=_state(position_mm=(0.0, 0.0, 0.0)),
            init_driver=_state(position_mm=(1.0, 0.0, 0.0)),
            mean=1000.0,
            cav_spacing=0.0,
            z_cutoff=0.0,
            startup_mode=StartupMode.INERTIAL_PREHISTORY,
            source_history_representation="proper_velocity",
            magnetic_dipole=MagneticDipoleConfig(enabled=True),
            macroparticle_smearing=MacroparticleSmearingConfig(
                enabled=True, subcharge_count=4, position_sigma_mm=0.25
            ),
            checkpoint=CheckpointConfig(enabled=True, directory=str(checkpoint)),
        )
    assert not checkpoint.exists()
