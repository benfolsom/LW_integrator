"""Headless regressions for magnetic-dipole GUI configuration plumbing."""

from __future__ import annotations

import tkinter as tk
from typing import Any

import pytest

from core.external_fields import (
    magnetic_field_native_to_tesla,
    magnetic_field_tesla_to_native,
)
from core.particle_config import DEFAULT_DRIVER_PARAMS, DEFAULT_RIDER_PARAMS
from core.species import get_species, list_species
from lw_integrator import gui
from lw_integrator.gui_config_mixins import IntegratorGUIConfigMixin
from lw_integrator.gui_controller_mixins import IntegratorGUIControllerMixin
from lw_integrator.gui_state_mixins import IntegratorGUIStateMixin
from lw_integrator.testbed_runner import PARTICLE_PARAM_FIELDS, SimulationOptions


class _Var:
    def __init__(self, value: Any = None) -> None:
        self.value = value

    def get(self) -> Any:
        return self.value

    def set(self, value: Any) -> None:
        self.value = value


class _Widget:
    def __init__(self) -> None:
        self.config: dict[str, Any] = {}

    def configure(self, **kwargs: Any) -> None:
        self.config.update(kwargs)


class _MagneticHarness(IntegratorGUIConfigMixin):
    def __init__(self) -> None:
        species = list_species()
        self._magnetic_species_by_label = {
            item.display_name: item.name for item in species
        }
        self._magnetic_species_label_by_key = {
            item.name: item.display_name for item in species
        }
        self.magnetic_dipole_enabled_var = _Var()
        self.magnetic_dipole_spin_precession_enabled_var = _Var()
        self.magnetic_dipole_stern_gerlach_force_enabled_var = _Var()
        self.magnetic_dipole_source_model_var = _Var()
        self.magnetic_dipole_exact_retarded_backend_var = _Var()
        self.magnetic_dipole_source_minimum_separation_var = _Var()
        self.rider_magnetic_species_var = _Var()
        self.driver_magnetic_species_var = _Var()
        self.rider_rest_spin_vars = [_Var() for _axis in range(3)]
        self.driver_rest_spin_vars = [_Var() for _axis in range(3)]

    def apply(self, options: SimulationOptions) -> None:
        self._apply_magnetic_dipole_options_to_ui(options)

    def build(self) -> dict[str, Any]:
        return self._build_magnetic_dipole_options_from_ui()


class _ExternalMagneticHarness(IntegratorGUIConfigMixin):
    def __init__(self) -> None:
        self.external_field_input_mode_var = _Var("SI V/m")
        self.external_magnetic_native_vars = [_Var() for _axis in range(3)]
        self.external_magnetic_tesla_vars = [_Var() for _axis in range(3)]
        self.external_magnetic_gradient_vars = [
            [_Var() for _coordinate in range(3)] for _component in range(3)
        ]


def test_current_magnetic_dipole_config_round_trips_through_gui_fields() -> None:
    source = SimulationOptions.from_dict(
        {
            "magnetic_dipole": {
                "enabled": True,
                "spin_precession_enabled": False,
                "stern_gerlach_force_enabled": True,
                "exact_retarded_backend": "numba_roots_exact_serial",
                "source": {
                    "model": "covariant_retarded_point",
                    "minimum_separation_mm": 7.0e-9,
                    "relative_stencil_step": 2.0e-3,
                    "minimum_stencil_step_mm": 3.0e-15,
                    "root_tolerance_mm": 4.0e-21,
                    "max_root_iterations": 80,
                },
                "rider": {
                    "species": "neutron",
                    "rest_spin": [1.0, -2.0, 0.5],
                },
                "driver": {
                    "species": "antiproton",
                    "rest_spin": [-0.25, 0.75, 1.5],
                },
            }
        }
    )
    harness = _MagneticHarness()

    harness.apply(source)

    assert harness.rider_magnetic_species_var.get() == "Neutron"
    assert harness.driver_magnetic_species_var.get() == "Antiproton"
    rebuilt = SimulationOptions(**harness.build())

    assert rebuilt.magnetic_dipole_enabled is True
    assert rebuilt.magnetic_dipole_spin_precession_enabled is False
    assert rebuilt.magnetic_dipole_stern_gerlach_force_enabled is True
    assert rebuilt.magnetic_dipole_spin_model == "rfs_minimal_2021"
    assert rebuilt.magnetic_dipole_stern_gerlach_model == "rfs_full_g"
    assert harness.magnetic_dipole_source_model_var.get() == (
        "Full retarded point (experimental)"
    )
    assert harness.magnetic_dipole_exact_retarded_backend_var.get() == (
        "Numba roots-exact CPU"
    )
    assert float(harness.magnetic_dipole_source_minimum_separation_var.get()) == (
        pytest.approx(7.0e-9)
    )
    assert rebuilt.magnetic_dipole_source_model == "covariant_retarded_point"
    assert rebuilt.magnetic_dipole_exact_retarded_backend == (
        "numba_roots_exact_serial"
    )
    assert rebuilt.magnetic_dipole_source_minimum_separation_mm == pytest.approx(7.0e-9)
    assert rebuilt.magnetic_dipole_source_relative_stencil_step == pytest.approx(2.0e-3)
    assert rebuilt.magnetic_dipole_source_minimum_stencil_step_mm == pytest.approx(
        3.0e-15
    )
    assert rebuilt.magnetic_dipole_source_root_tolerance_mm == pytest.approx(4.0e-21)
    assert rebuilt.magnetic_dipole_source_max_root_iterations == 80
    assert rebuilt.rider_magnetic_species == "neutron"
    assert rebuilt.driver_magnetic_species == "antiproton"
    assert rebuilt.rider_rest_spin == pytest.approx((1.0, -2.0, 0.5))
    assert rebuilt.driver_rest_spin == pytest.approx((-0.25, 0.75, 1.5))
    assert rebuilt.to_dict()["magnetic_dipole"] == source.to_dict()["magnetic_dipole"]


def test_full_strict_backend_round_trips_through_gui_label() -> None:
    source = SimulationOptions(
        magnetic_dipole_exact_retarded_backend="numba_full_strict_serial"
    )
    harness = _MagneticHarness()

    harness.apply(source)
    rebuilt = SimulationOptions(**harness.build())

    assert harness.magnetic_dipole_exact_retarded_backend_var.get() == (
        "Numba full strict CPU"
    )
    assert rebuilt.magnetic_dipole_exact_retarded_backend == (
        "numba_full_strict_serial"
    )


def test_analytical_charge_backend_round_trips_through_gui_label() -> None:
    source = SimulationOptions(
        magnetic_dipole_exact_retarded_backend=("numba_analytic_charge_response_serial")
    )
    harness = _MagneticHarness()

    harness.apply(source)
    rebuilt = SimulationOptions(**harness.build())

    assert harness.magnetic_dipole_exact_retarded_backend_var.get() == (
        "Numba analytical charge response CPU"
    )
    assert rebuilt.magnetic_dipole_exact_retarded_backend == (
        "numba_analytic_charge_response_serial"
    )


def test_old_config_defaults_round_trip_with_magnetic_dipoles_off() -> None:
    old_config = SimulationOptions.from_dict({"steps": 12})
    harness = _MagneticHarness()

    harness.apply(old_config)
    rebuilt = SimulationOptions(**harness.build())

    assert rebuilt.magnetic_dipole_enabled is False
    assert rebuilt.magnetic_dipole_spin_precession_enabled is True
    assert rebuilt.magnetic_dipole_stern_gerlach_force_enabled is False
    assert rebuilt.magnetic_dipole_spin_model == "rfs_minimal_2021"
    assert rebuilt.magnetic_dipole_stern_gerlach_model == "rfs_full_g"
    assert rebuilt.magnetic_dipole_source_model == "off"
    assert rebuilt.magnetic_dipole_exact_retarded_backend == "python"
    assert rebuilt.magnetic_dipole_source_minimum_separation_mm == pytest.approx(2.0e-9)
    assert rebuilt.rider_magnetic_species == "electron"
    assert rebuilt.driver_magnetic_species == "proton"
    assert rebuilt.rider_rest_spin == (0.0, 0.0, 1.0)
    assert rebuilt.driver_rest_spin == (0.0, 0.0, 1.0)
    assert rebuilt.to_dict()["magnetic_dipole"]["enabled"] is False


def test_legacy_model_pair_round_trips_through_compact_gui() -> None:
    source = SimulationOptions.from_dict(
        {
            "magnetic_dipole": {
                "spin_model": "bmt_frenkel",
                "stern_gerlach_model": "static_rest_gradient",
            }
        }
    )
    harness = _MagneticHarness()

    harness.apply(source)
    rebuilt = SimulationOptions(**harness.build())

    assert rebuilt.magnetic_dipole_spin_model == "bmt_frenkel"
    assert rebuilt.magnetic_dipole_stern_gerlach_model == "static_rest_gradient"
    assert rebuilt.to_dict()["magnetic_dipole"] == source.to_dict()["magnetic_dipole"]


def test_magnetic_species_selector_is_backed_by_core_registry() -> None:
    harness = _MagneticHarness()

    assert set(harness._magnetic_species_label_by_key) == {
        item.name for item in list_species()
    }
    assert "neutron" in harness._magnetic_species_label_by_key
    assert "h_minus" in harness._magnetic_species_label_by_key


@pytest.mark.parametrize(
    ("role", "species_key"),
    (("rider", "electron"), ("driver", "neutron")),
)
def test_general_species_preset_synchronizes_magnetic_species(
    role: str, species_key: str
) -> None:
    species = get_species(species_key)
    species_by_label = {item.display_name: item.name for item in list_species()}
    magnetic_label_by_key = {item.name: item.display_name for item in list_species()}
    rider_params = {
        field: _Var(DEFAULT_RIDER_PARAMS[field]) for field in PARTICLE_PARAM_FIELDS
    }
    driver_params = {
        field: _Var(DEFAULT_DRIVER_PARAMS[field]) for field in PARTICLE_PARAM_FIELDS
    }
    harness = type("ControllerHarness", (IntegratorGUIControllerMixin,), {})()
    harness._species_by_label = species_by_label
    harness._magnetic_species_label_by_key = magnetic_label_by_key
    harness.rider_species_var = _Var(
        species.display_name if role == "rider" else "Electron"
    )
    harness.driver_species_var = _Var(
        species.display_name if role == "driver" else "Proton"
    )
    harness.rider_magnetic_species_var = _Var("Proton")
    harness.driver_magnetic_species_var = _Var("Electron")
    harness.rider_param_vars = rider_params
    harness.driver_param_vars = driver_params
    harness._refresh_initial_summary = lambda: None

    harness._apply_species(role)

    magnetic_var = getattr(harness, f"{role}_magnetic_species_var")
    param_vars = rider_params if role == "rider" else driver_params
    assert magnetic_var.get() == species.display_name
    assert float(param_vars["m_particle"].get()) == pytest.approx(species.mass_amu)
    assert float(param_vars["charge_sign"].get()) * float(
        param_vars["stripped_ions"].get()
    ) == pytest.approx(species.charge_e)


def test_named_magnetic_species_rejects_mismatched_general_particle_values() -> None:
    harness = _MagneticHarness()
    magnetic_options = {
        "magnetic_dipole_enabled": True,
        "rider_magnetic_species": "electron",
        "driver_magnetic_species": "proton",
    }

    with pytest.raises(ValueError, match="Magnetic dipole rider species mismatch"):
        harness._validate_magnetic_species_particle_matches(
            magnetic_options=magnetic_options,
            rider_params=dict(DEFAULT_RIDER_PARAMS),
            driver_params=None,
        )


def test_named_driver_magnetic_species_is_validated_in_bunch_to_bunch_mode() -> None:
    proton = get_species("proton")
    harness = _MagneticHarness()

    with pytest.raises(ValueError, match="Magnetic dipole driver species mismatch"):
        harness._validate_magnetic_species_particle_matches(
            magnetic_options={
                "magnetic_dipole_enabled": True,
                "rider_magnetic_species": "proton",
                "driver_magnetic_species": "proton",
            },
            rider_params={
                "mass_amu": proton.mass_amu,
                "charge_sign": 1.0,
                "stripped_ions": 1.0,
            },
            driver_params=dict(DEFAULT_DRIVER_PARAMS),
        )


@pytest.mark.parametrize("species_key", ("electron", "neutron", "deuteron"))
def test_custom_general_particle_values_may_numerically_match_named_magnetic_species(
    species_key: str,
) -> None:
    species = get_species(species_key)
    harness = _MagneticHarness()
    params = {
        "mass_amu": species.mass_amu,
        "charge_sign": -1.0 if species.charge_e < 0 else 1.0,
        "stripped_ions": abs(species.charge_e),
    }
    if species.charge_e == 0:
        params["charge_sign"] = 0.0
        params["stripped_ions"] = 1.0

    harness._validate_magnetic_species_particle_matches(
        magnetic_options={
            "magnetic_dipole_enabled": True,
            "rider_magnetic_species": species_key,
            "driver_magnetic_species": "proton",
        },
        rider_params=params,
        driver_params=None,
    )


def test_gui_build_rejects_mismatch_and_accepts_matching_custom_values() -> None:
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")

    root.withdraw()
    try:
        app = gui.IntegratorGUI(root)
        app.sim_type_var.set("CONDUCTING_WALL")
        app.magnetic_dipole_enabled_var.set(True)

        with pytest.raises(ValueError, match="Magnetic dipole rider species mismatch"):
            app._build_options_from_ui()

        electron = get_species("electron")
        app.rider_species_var.set(app._species_label_by_key["custom"])
        app.rider_param_vars["m_particle"].set(electron.mass_amu)
        app.rider_param_vars["charge_sign"].set(-1.0)
        app.rider_param_vars["stripped_ions"].set(1.0)
        rebuilt = app._build_options_from_ui()

        assert rebuilt.rider_magnetic_species == "electron"
        assert rebuilt.rider_params["m_particle"] == pytest.approx(electron.mass_amu)
    finally:
        root.destroy()


def test_gui_labels_present_compact_rfs_controls() -> None:
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")

    root.withdraw()
    try:
        app = gui.IntegratorGUI(root)

        assert "RFS" in app.magnetic_dipole_enable_check.master.cget("text")
        assert app.magnetic_dipole_precession_check.cget("text") == "Spin precession"
        assert app.magnetic_dipole_stern_gerlach_check.cget("text") == (
            "Fully coupled force (Stern–Gerlach)"
        )
        assert tuple(app.magnetic_dipole_source_model_combo.cget("values")) == (
            "Off",
            "Full retarded point (experimental)",
        )
        assert app.magnetic_dipole_exact_retarded_backend_label.cget("text") == (
            "Exact-retarded backend:"
        )
        assert tuple(
            app.magnetic_dipole_exact_retarded_backend_combo.cget("values")
        ) == (
            "Python reference",
            "Numba roots-exact CPU",
            "Numba full strict CPU",
            "Metal-certified roots + strict CPU",
        )
        assert app.magnetic_dipole_source_cutoff_label.cget("text") == (
            "Minimum separation abort (mm):"
        )
        assert not hasattr(app, "magnetic_dipole_spin_model_var")
        assert not hasattr(app, "magnetic_dipole_stern_gerlach_model_var")
    finally:
        root.destroy()


def test_magnetic_control_state_tracks_enable_and_bunch_to_bunch_mode() -> None:
    common = _Widget()
    rider_combo = _Widget()
    rider_spin = _Widget()
    driver_combo = _Widget()
    driver_spin = _Widget()
    driver_label = _Widget()
    exact_backend_combo = _Widget()
    source_cutoff = _Widget()
    source_label = _Widget()
    harness = type(
        "StateHarness",
        (IntegratorGUIStateMixin,),
        {},
    )()
    harness.magnetic_dipole_enabled_var = _Var(False)
    harness.magnetic_dipole_source_model_var = _Var("Off")
    harness.sim_type_var = _Var("BUNCH_TO_BUNCH")
    harness._magnetic_dipole_common_controls = [
        (common, "normal"),
        (exact_backend_combo, "readonly"),
    ]
    harness._magnetic_dipole_source_controls = [(source_cutoff, "normal")]
    harness._magnetic_dipole_source_labels = [source_label]
    harness._magnetic_dipole_rider_controls = [
        (rider_combo, "readonly"),
        (rider_spin, "normal"),
    ]
    harness._magnetic_dipole_driver_controls = [
        (driver_combo, "readonly"),
        (driver_spin, "normal"),
    ]
    harness._magnetic_dipole_driver_labels = [driver_label]

    harness._toggle_magnetic_dipole_controls()

    assert common.config["state"] == "disabled"
    assert rider_combo.config["state"] == "disabled"
    assert driver_combo.config["state"] == "disabled"
    assert exact_backend_combo.config["state"] == "disabled"
    assert source_cutoff.config["state"] == "disabled"

    harness.magnetic_dipole_enabled_var.set(True)
    harness._toggle_magnetic_dipole_controls()

    assert common.config["state"] == "normal"
    assert rider_combo.config["state"] == "readonly"
    assert rider_spin.config["state"] == "normal"
    assert driver_combo.config["state"] == "readonly"
    assert driver_spin.config["state"] == "normal"
    assert driver_label.config["foreground"] == "black"
    assert exact_backend_combo.config["state"] == "readonly"
    assert source_cutoff.config["state"] == "disabled"

    harness.magnetic_dipole_source_model_var.set("Full retarded point (experimental)")
    harness._toggle_magnetic_dipole_controls()

    assert source_cutoff.config["state"] == "normal"
    assert source_label.config["foreground"] == "black"

    harness.sim_type_var.set("CONDUCTING_WALL")
    harness._toggle_magnetic_dipole_controls()

    assert rider_combo.config["state"] == "readonly"
    assert exact_backend_combo.config["state"] == "readonly"
    assert driver_combo.config["state"] == "disabled"
    assert driver_spin.config["state"] == "disabled"
    assert driver_label.config["foreground"] == "gray"
    assert source_cutoff.config["state"] == "disabled"
    assert source_label.config["foreground"] == "gray"


def test_user_enabling_rfs_selects_rr_off() -> None:
    harness = type(
        "StateHarness",
        (IntegratorGUIStateMixin,),
        {},
    )()
    harness.magnetic_dipole_enabled_var = _Var(True)
    harness.radiation_reaction_mode_var = _Var("medina_lad")
    harness.adaptive_timestep_enabled_var = _Var(True)
    harness.sim_type_var = _Var("BUNCH_TO_BUNCH")
    harness._magnetic_dipole_spin_model = "rfs_minimal_2021"

    harness._on_magnetic_dipole_toggle()

    assert harness.radiation_reaction_mode_var.get() == "off"
    assert harness.adaptive_timestep_enabled_var.get() is False


def test_user_enabling_legacy_dipole_model_keeps_rr_choice() -> None:
    harness = type(
        "StateHarness",
        (IntegratorGUIStateMixin,),
        {},
    )()
    harness.magnetic_dipole_enabled_var = _Var(True)
    harness.radiation_reaction_mode_var = _Var("medina_lad")
    harness.sim_type_var = _Var("BUNCH_TO_BUNCH")
    harness._magnetic_dipole_spin_model = "bmt_frenkel"

    harness._on_magnetic_dipole_toggle()

    assert harness.radiation_reaction_mode_var.get() == "medina_lad"


def test_tesla_base_field_and_t_per_m_gradient_round_trip() -> None:
    harness = _ExternalMagneticHarness()
    source = SimulationOptions(
        external_magnetic_field_native=(1.0, -2.0, 3.0),
        external_magnetic_field_gradient_t_per_m=(
            (1.0, 2.0, 3.0),
            (4.0, 5.0, 6.0),
            (7.0, 8.0, -6.0),
        ),
    )

    harness._apply_external_magnetic_options_to_ui(source)
    assert float(harness.external_magnetic_tesla_vars[0].get()) == pytest.approx(
        magnetic_field_native_to_tesla(1.0)
    )
    harness.external_magnetic_tesla_vars[0].set("0.5")
    options = harness._build_external_magnetic_options_from_ui(enabled=True)

    assert options["external_magnetic_field_native"][0] == pytest.approx(
        magnetic_field_tesla_to_native(0.5)
    )
    assert options["external_magnetic_field_gradient_t_per_m"] == (
        (1.0, 2.0, 3.0),
        (4.0, 5.0, 6.0),
        (7.0, 8.0, -6.0),
    )
