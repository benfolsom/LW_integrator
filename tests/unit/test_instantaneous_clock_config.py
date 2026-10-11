"""The lab-time clock requires an explicit choice in every configuration path."""

from types import SimpleNamespace

import pytest

from core.types import SpaceChargeConfig
from lw_integrator.cli import _build_space_charge_config, parse_args
from lw_integrator.gui_config_mixins import IntegratorGUIConfigMixin
from lw_integrator.testbed_runner import (
    SimulationOptions,
    build_space_charge_config,
)
from optimization.config import OptimizationConfig
from optimization.plugin_persistence_helpers import apply_persisted_config_overrides
from optimization.single_integration_helpers import build_single_integration_setup


class _Var:
    def __init__(self):
        self.value = None

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class _SpaceChargeHarness(IntegratorGUIConfigMixin):
    def __init__(self):
        for name in (
            "enabled",
            "retarded",
            "softening_mm",
            "bunch_sigma_mm",
            "instantaneous_clock",
            "min_retarded_steps",
        ):
            setattr(self, f"space_charge_{name}_var", _Var())


@pytest.mark.parametrize("clock", [None, "legacy_same_index", "lab_time"])
def test_clock_roundtrip_has_cli_gui_and_sweep_parity(clock, tmp_path):
    payload = {"space_charge_enabled": True}
    if clock is not None:
        payload["space_charge_instantaneous_clock"] = clock
    expected = clock or "legacy_same_index"
    options = SimulationOptions.from_dict(payload)
    gui = _SpaceChargeHarness()
    gui._apply_space_charge_options_to_ui(options)
    gui_options = SimulationOptions(**gui._build_space_charge_options_from_ui())
    for key in gui._build_space_charge_options_from_ui():
        assert getattr(gui_options, key) == getattr(options, key)
    assert build_space_charge_config(gui_options).instantaneous_clock == expected
    assert _build_space_charge_config(payload).instantaneous_clock == expected
    assert (
        SimulationOptions.from_dict(
            gui_options.to_dict()
        ).space_charge_instantaneous_clock
        == expected
    )

    config = OptimizationConfig.from_simulation_options(options)
    assert config.space_charge_instantaneous_clock == expected
    persisted = apply_persisted_config_overrides(OptimizationConfig(), payload)
    assert persisted.space_charge_instantaneous_clock == expected
    setup = build_single_integration_setup(
        config,
        aperture=1e5,
        energy_gev=0.1,
        start_z=-1000,
        transv_offset=0,
        timestep=1e-7,
        steps=4,
        run_output_dir=tmp_path,
        run_num=0,
        driver_params=None,
    )
    assert setup.options.space_charge_instantaneous_clock == expected
    args = parse_args(
        [] if clock is None else ["--space-charge-instantaneous-clock", clock]
    )
    assert args.space_charge_instantaneous_clock == clock
