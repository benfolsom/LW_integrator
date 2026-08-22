"""Headless regressions for GUI macroparticle-smearing config mapping."""

from __future__ import annotations

from typing import Any

from lw_integrator.gui_config_mixins import IntegratorGUIConfigMixin
from lw_integrator.testbed_runner import SimulationOptions


class _Var:
    def __init__(self, value: Any = None) -> None:
        self.value: Any = value

    def get(self) -> Any:
        return self.value

    def set(self, value: Any) -> None:
        self.value = value


class _SmearingHarness(IntegratorGUIConfigMixin):
    macroparticle_use_momentum_errors_var: _Var
    macroparticle_smearing_enabled_var: _Var
    macroparticle_smearing_subcharge_count_var: _Var
    macroparticle_smearing_sigma_multiplier_var: _Var
    macroparticle_smearing_position_sigma_var: _Var
    macroparticle_smearing_longitudinal_sigma_var: _Var
    macroparticle_smearing_momentum_sigma_var: _Var
    macroparticle_smearing_use_position_errors_var: _Var
    macroparticle_smearing_use_momentum_errors_var: _Var
    macroparticle_smearing_use_centroid_errors_var: _Var
    macroparticle_smearing_use_internal_cloud_var: _Var
    macroparticle_smearing_apply_to_active_observers_var: _Var
    macroparticle_smearing_apply_to_active_sources_var: _Var
    macroparticle_smearing_apply_to_passive_sources_var: _Var
    macroparticle_smearing_apply_to_passive_updates_var: _Var
    macroparticle_smearing_seed_var: _Var
    macroparticle_smearing_refresh_policy_var: _Var

    def __init__(self) -> None:
        self.macroparticle_use_momentum_errors_var = _Var(True)
        self.macroparticle_smearing_enabled_var = _Var()
        self.macroparticle_smearing_subcharge_count_var = _Var()
        self.macroparticle_smearing_sigma_multiplier_var = _Var()
        self.macroparticle_smearing_position_sigma_var = _Var()
        self.macroparticle_smearing_longitudinal_sigma_var = _Var()
        self.macroparticle_smearing_momentum_sigma_var = _Var()
        self.macroparticle_smearing_use_position_errors_var = _Var()
        self.macroparticle_smearing_use_momentum_errors_var = _Var()
        self.macroparticle_smearing_use_centroid_errors_var = _Var()
        self.macroparticle_smearing_use_internal_cloud_var = _Var()
        self.macroparticle_smearing_apply_to_active_observers_var = _Var()
        self.macroparticle_smearing_apply_to_active_sources_var = _Var()
        self.macroparticle_smearing_apply_to_passive_sources_var = _Var()
        self.macroparticle_smearing_apply_to_passive_updates_var = _Var()
        self.macroparticle_smearing_seed_var = _Var()
        self.macroparticle_smearing_refresh_policy_var = _Var()

    def apply_smearing_options(self, options: SimulationOptions) -> None:
        self._apply_macroparticle_smearing_options_to_ui(options)

    def build_smearing_options(self) -> dict[str, Any]:
        return self._build_macroparticle_smearing_options_from_ui()


def _smearing_harness() -> _SmearingHarness:
    return _SmearingHarness()


def test_source_smearing_config_round_trips_without_touching_image_momentum_errors():
    smearing_config = {
        "enabled": True,
        "subcharge_count": 5,
        "sigma_multiplier": 1.25,
        "position_sigma_mm": 0.125,
        "longitudinal_sigma_mm": 0.25,
        "momentum_sigma_amu_mm_ns": 0.5,
        "use_position_errors": False,
        "use_momentum_errors": False,
        "use_centroid_errors": False,
        "use_internal_cloud": False,
        "apply_to_active_observers": False,
        "apply_to_active_sources": False,
        "apply_to_passive_sources": False,
        "apply_to_passive_updates": True,
        "seed": 9876,
        "refresh_policy": "per_step",
    }
    loaded = SimulationOptions.from_dict(
        {
            "macroparticle_use_momentum_errors": True,
            "macroparticle_smearing": smearing_config,
        }
    )
    harness = _smearing_harness()

    harness.apply_smearing_options(loaded)

    assert harness.macroparticle_smearing_use_momentum_errors_var.get() is False
    assert harness.macroparticle_use_momentum_errors_var.get() is True

    rebuilt = SimulationOptions(
        macroparticle_use_momentum_errors=bool(
            harness.macroparticle_use_momentum_errors_var.get()
        ),
        **harness.build_smearing_options(),
    )

    assert rebuilt.macroparticle_use_momentum_errors is True
    assert rebuilt.to_dict()["macroparticle_smearing"] == smearing_config


def test_source_smearing_optional_sigmas_round_trip_as_blank_gui_values():
    source = SimulationOptions(
        macroparticle_smearing_enabled=True,
        macroparticle_smearing_position_sigma_mm=None,
        macroparticle_smearing_longitudinal_sigma_mm=None,
        macroparticle_smearing_momentum_sigma_amu_mm_ns=None,
    )
    harness = _smearing_harness()

    harness.apply_smearing_options(source)

    assert harness.macroparticle_smearing_position_sigma_var.get() == ""
    assert harness.macroparticle_smearing_longitudinal_sigma_var.get() == ""
    assert harness.macroparticle_smearing_momentum_sigma_var.get() == ""

    rebuilt = SimulationOptions(**harness.build_smearing_options())

    assert rebuilt.macroparticle_smearing_position_sigma_mm is None
    assert rebuilt.macroparticle_smearing_longitudinal_sigma_mm is None
    assert rebuilt.macroparticle_smearing_momentum_sigma_amu_mm_ns is None
