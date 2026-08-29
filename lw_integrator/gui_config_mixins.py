"""Config load/save and UI<->options mapping helpers for the main GUI."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

from core.external_fields import (
    magnetic_field_native_to_tesla,
    magnetic_field_tesla_to_native,
)
from core.particle_config import DEFAULT_DRIVER_PARAMS, DEFAULT_RIDER_PARAMS
from core.species import get_species
from core.types import SimulationType
from optimization.mode_helpers import SWEEP_OR_OPTIMIZATION_MODES

from .testbed_runner import (
    CORE_PARAM_DEFAULTS,
    DIPOLE_SOURCE_MODEL_OPTIONS,
    EXACT_RETARDED_BACKEND_OPTIONS,
    EXACT_RETARDED_UPDATE_OPTIONS,
    PARTICLE_PARAM_FIELDS,
    SimulationOptions,
    load_config,
    save_config,
    supports_driver,
)

_SWEEP_OR_OPTIMIZATION_KEYS = {"sweep_parameters", "parameter_sweeps"}
_DIPOLE_SOURCE_MODEL_BY_LABEL = dict(DIPOLE_SOURCE_MODEL_OPTIONS)
_DIPOLE_SOURCE_LABEL_BY_MODEL = {
    model: label for label, model in DIPOLE_SOURCE_MODEL_OPTIONS
}
_DIPOLE_SOURCE_LABEL_BY_MODEL.update(
    {
        "retarded_point": _DIPOLE_SOURCE_LABEL_BY_MODEL["covariant_retarded_point"],
        "full_retarded_point": _DIPOLE_SOURCE_LABEL_BY_MODEL[
            "covariant_retarded_point"
        ],
    }
)
_EXACT_RETARDED_BACKEND_BY_LABEL = dict(EXACT_RETARDED_BACKEND_OPTIONS)
_EXACT_RETARDED_LABEL_BY_BACKEND = {
    backend: label for label, backend in EXACT_RETARDED_BACKEND_OPTIONS
}
_EXACT_RETARDED_UPDATE_BY_LABEL = dict(EXACT_RETARDED_UPDATE_OPTIONS)
_EXACT_RETARDED_LABEL_BY_UPDATE = {
    update: label for label, update in EXACT_RETARDED_UPDATE_OPTIONS
}


def _particle_params_require_manual_config(params: object) -> bool:
    if not isinstance(params, dict):
        return False
    return any(key not in PARTICLE_PARAM_FIELDS for key in params)


def _format_gui_float(value: object) -> str:
    return f"{float(value):.12g}"


def _format_gui_optional_float(value: object) -> str:
    return "" if value is None else _format_gui_float(value)


def _parse_gui_float(text: object, label: str) -> float:
    try:
        return float(str(text).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc


def _parse_gui_optional_float(text: object, label: str):
    cleaned = str(text).strip()
    if cleaned == "":
        return None
    return _parse_gui_float(cleaned, label)


def _parse_gui_float_lenient(text: object, default: float) -> float:
    try:
        return float(str(text).strip())
    except (TypeError, ValueError):
        return default


def _parse_gui_optional_float_lenient(text: object):
    cleaned = str(text).strip()
    if cleaned == "":
        return None
    try:
        return float(cleaned)
    except (TypeError, ValueError):
        return None


def _looks_like_sweep_or_optimization_config(path: Path) -> bool:
    """Return true for configs that must be handled by the sweep/optim tab."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    if not isinstance(data, dict):
        return False

    mode = data.get("mode")
    if isinstance(mode, str) and mode in SWEEP_OR_OPTIMIZATION_MODES:
        return True

    return any(key in data for key in _SWEEP_OR_OPTIMIZATION_KEYS)


class IntegratorGUIConfigMixin:
    """Translate between GUI state and ``SimulationOptions`` configs."""

    def _apply_magnetic_dipole_options_to_ui(
        self: Any, options: SimulationOptions
    ) -> None:
        """Populate the compact magnetic-dipole controls from run options."""
        if not hasattr(self, "magnetic_dipole_enabled_var"):
            return

        self.magnetic_dipole_enabled_var.set(
            getattr(options, "magnetic_dipole_enabled", False)
        )
        self.magnetic_dipole_spin_precession_enabled_var.set(
            getattr(options, "magnetic_dipole_spin_precession_enabled", True)
        )
        self.magnetic_dipole_stern_gerlach_force_enabled_var.set(
            getattr(
                options,
                "magnetic_dipole_stern_gerlach_force_enabled",
                False,
            )
        )
        # Model selection stays out of the compact GUI controls, but preserve
        # it when loading and saving diagnostic JSON configurations.
        self._magnetic_dipole_spin_model = str(
            getattr(options, "magnetic_dipole_spin_model", "rfs_minimal_2021")
        )
        self._magnetic_dipole_stern_gerlach_model = str(
            getattr(options, "magnetic_dipole_stern_gerlach_model", "rfs_full_g")
        )
        source_model = (
            str(getattr(options, "magnetic_dipole_source_model", "off"))
            .strip()
            .lower()
            .replace("-", "_")
        )
        self.magnetic_dipole_source_model_var.set(
            _DIPOLE_SOURCE_LABEL_BY_MODEL.get(source_model, source_model)
        )
        exact_retarded_backend = (
            str(
                getattr(
                    options,
                    "magnetic_dipole_exact_retarded_backend",
                    "python",
                )
            )
            .strip()
            .lower()
        )
        self.magnetic_dipole_exact_retarded_backend_var.set(
            _EXACT_RETARDED_LABEL_BY_BACKEND.get(
                exact_retarded_backend, exact_retarded_backend
            )
        )
        exact_retarded_update = (
            str(
                getattr(
                    options,
                    "magnetic_dipole_exact_retarded_update",
                    "first_order_endpoint",
                )
            )
            .strip()
            .lower()
            .replace("-", "_")
        )
        exact_retarded_update = {
            "second_order": "second_order_start_taylor_endpoint",
            "second_order_taylor": "second_order_start_taylor_endpoint",
            "second_order_taylor_endpoint": ("second_order_start_taylor_endpoint"),
        }.get(exact_retarded_update, exact_retarded_update)
        self.magnetic_dipole_exact_retarded_update_var.set(
            _EXACT_RETARDED_LABEL_BY_UPDATE.get(
                exact_retarded_update, exact_retarded_update
            )
        )
        self.magnetic_dipole_source_minimum_separation_var.set(
            _format_gui_float(
                getattr(
                    options,
                    "magnetic_dipole_source_minimum_separation_mm",
                    2.0e-9,
                )
            )
        )
        self._magnetic_dipole_source_relative_stencil_step = float(
            getattr(
                options,
                "magnetic_dipole_source_relative_stencil_step",
                1.0e-3,
            )
        )
        self._magnetic_dipole_source_minimum_stencil_step_mm = float(
            getattr(
                options,
                "magnetic_dipole_source_minimum_stencil_step_mm",
                1.0e-15,
            )
        )
        self._magnetic_dipole_source_root_tolerance_mm = float(
            getattr(
                options,
                "magnetic_dipole_source_root_tolerance_mm",
                1.0e-21,
            )
        )
        self._magnetic_dipole_source_max_root_iterations = int(
            getattr(
                options,
                "magnetic_dipole_source_max_root_iterations",
                96,
            )
        )

        rider_species = str(getattr(options, "rider_magnetic_species", "electron"))
        driver_species = str(getattr(options, "driver_magnetic_species", "proton"))
        self.rider_magnetic_species_var.set(
            self._magnetic_species_label_by_key.get(rider_species, rider_species)
        )
        self.driver_magnetic_species_var.set(
            self._magnetic_species_label_by_key.get(driver_species, driver_species)
        )

        rider_spin = getattr(options, "rider_rest_spin", (0.0, 0.0, 1.0))
        driver_spin = getattr(options, "driver_rest_spin", (0.0, 0.0, 1.0))
        for var, value in zip(self.rider_rest_spin_vars, rider_spin):
            var.set(_format_gui_float(value))
        for var, value in zip(self.driver_rest_spin_vars, driver_spin):
            var.set(_format_gui_float(value))

    def _build_magnetic_dipole_options_from_ui(self: Any) -> dict[str, Any]:
        """Return the magnetic-dipole fields represented by the GUI."""

        enabled = bool(self.magnetic_dipole_enabled_var.get())
        driver_enabled = enabled and (
            not hasattr(self, "sim_type_var")
            or self.sim_type_var.get() == "BUNCH_TO_BUNCH"
        )

        def selected_species(variable: Any, role: str) -> str:
            selection = str(variable.get()).strip()
            species = self._magnetic_species_by_label.get(selection, selection)
            if species not in self._magnetic_species_label_by_key:
                raise ValueError(f"Select a known magnetic species for the {role}.")
            return species

        def spin_vector(
            variables: Any, role: str, *, validate: bool
        ) -> tuple[float, float, float]:
            defaults = (0.0, 0.0, 1.0)
            values = []
            for var, axis, default in zip(variables, ("x", "y", "z"), defaults):
                if validate:
                    value = _parse_gui_float(var.get(), f"{role} rest spin {axis}")
                else:
                    value = _parse_gui_float_lenient(var.get(), default)
                values.append(value)
            return (values[0], values[1], values[2])

        source_selection = str(self.magnetic_dipole_source_model_var.get()).strip()
        source_model = _DIPOLE_SOURCE_MODEL_BY_LABEL.get(source_selection)
        if source_model is None:
            normalized_source_model = source_selection.lower().replace("-", "_")
            source_model = {
                "retarded_point": "covariant_retarded_point",
                "full_retarded_point": "covariant_retarded_point",
            }.get(normalized_source_model, normalized_source_model)
        if source_model not in _DIPOLE_SOURCE_LABEL_BY_MODEL:
            raise ValueError(
                "Select Off or Full retarded point (experimental) for the "
                "dipole source."
            )
        backend_selection = str(
            self.magnetic_dipole_exact_retarded_backend_var.get()
        ).strip()
        exact_retarded_backend = _EXACT_RETARDED_BACKEND_BY_LABEL.get(
            backend_selection, backend_selection.strip().lower()
        )
        if exact_retarded_backend not in _EXACT_RETARDED_LABEL_BY_BACKEND:
            raise ValueError(
                "Select Python reference, Numba roots-exact CPU, or Numba full "
                "strict CPU, or Metal-certified roots + strict CPU for the "
                "exact-retarded backend."
            )
        update_selection = str(
            self.magnetic_dipole_exact_retarded_update_var.get()
        ).strip()
        exact_retarded_update = _EXACT_RETARDED_UPDATE_BY_LABEL.get(
            update_selection,
            update_selection.strip().lower().replace("-", "_"),
        )
        if exact_retarded_update not in _EXACT_RETARDED_LABEL_BY_UPDATE:
            raise ValueError(
                "Select First-order endpoint or Second-order accepted-start "
                "Taylor for the exact-retarded update."
            )
        source_minimum_separation = _parse_gui_float(
            self.magnetic_dipole_source_minimum_separation_var.get(),
            "Dipole source minimum separation",
        )
        if (
            not math.isfinite(source_minimum_separation)
            or source_minimum_separation <= 0.0
        ):
            raise ValueError(
                "Dipole source minimum separation must be finite and positive."
            )

        return {
            "magnetic_dipole_enabled": enabled,
            "magnetic_dipole_spin_precession_enabled": bool(
                self.magnetic_dipole_spin_precession_enabled_var.get()
            ),
            "magnetic_dipole_stern_gerlach_force_enabled": bool(
                self.magnetic_dipole_stern_gerlach_force_enabled_var.get()
            ),
            "magnetic_dipole_spin_model": getattr(
                self, "_magnetic_dipole_spin_model", "rfs_minimal_2021"
            ),
            "magnetic_dipole_stern_gerlach_model": getattr(
                self, "_magnetic_dipole_stern_gerlach_model", "rfs_full_g"
            ),
            "magnetic_dipole_source_model": source_model,
            "magnetic_dipole_exact_retarded_backend": exact_retarded_backend,
            "magnetic_dipole_exact_retarded_update": exact_retarded_update,
            "magnetic_dipole_source_minimum_separation_mm": (source_minimum_separation),
            "magnetic_dipole_source_relative_stencil_step": getattr(
                self, "_magnetic_dipole_source_relative_stencil_step", 1.0e-3
            ),
            "magnetic_dipole_source_minimum_stencil_step_mm": getattr(
                self,
                "_magnetic_dipole_source_minimum_stencil_step_mm",
                1.0e-15,
            ),
            "magnetic_dipole_source_root_tolerance_mm": getattr(
                self, "_magnetic_dipole_source_root_tolerance_mm", 1.0e-21
            ),
            "magnetic_dipole_source_max_root_iterations": getattr(
                self, "_magnetic_dipole_source_max_root_iterations", 96
            ),
            "rider_magnetic_species": selected_species(
                self.rider_magnetic_species_var, "rider"
            ),
            "driver_magnetic_species": selected_species(
                self.driver_magnetic_species_var, "driver"
            ),
            "rider_rest_spin": spin_vector(
                self.rider_rest_spin_vars, "Rider", validate=enabled
            ),
            "driver_rest_spin": spin_vector(
                self.driver_rest_spin_vars, "Driver", validate=driver_enabled
            ),
        }

    def _validate_magnetic_species_particle_matches(
        self: Any,
        *,
        magnetic_options: dict[str, Any],
        rider_params: dict[str, Any],
        driver_params: dict[str, Any] | None,
    ) -> None:
        """Reject named magnetic presets that disagree with particle q and m."""
        if not magnetic_options["magnetic_dipole_enabled"]:
            return

        def validate_role(role: str, params: dict[str, Any], species_key: str) -> None:
            species = get_species(species_key)
            if not species.has_supported_magnetic_moment:
                raise ValueError(
                    f"The magnetic {role} species '{species.display_name}' has no "
                    "built-in magnetic-moment preset. The GUI does not expose "
                    "custom moments; choose a supported preset or use a documented "
                    "custom CLI/JSON configuration."
                )

            mass_value = params.get("mass_amu", params.get("m_particle"))
            charge_sign = params.get("charge_sign")
            stripped_ions = params.get("stripped_ions")
            if mass_value is None or charge_sign is None or stripped_ions is None:
                raise ValueError(
                    f"Magnetic dipole validation needs {role} particle mass, "
                    "charge_sign, and stripped_ions values."
                )
            try:
                actual_mass = float(mass_value)
                actual_charge_e = float(charge_sign) * float(stripped_ions)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"The {role} particle mass and charge values must be numeric "
                    "when magnetic dipoles are enabled."
                ) from exc

            mass_matches = math.isclose(
                actual_mass,
                species.mass_amu,
                rel_tol=1.0e-3,
                abs_tol=1.0e-12,
            )
            charge_matches = math.isclose(
                actual_charge_e,
                float(species.charge_e),
                rel_tol=0.0,
                abs_tol=1.0e-6,
            )
            if not mass_matches or not charge_matches:
                raise ValueError(
                    f"Magnetic dipole {role} species mismatch: "
                    f"'{species.display_name}' expects mass "
                    f"{species.mass_amu:.12g} amu and charge "
                    f"{species.charge_e:+d} e, but the current {role} particle "
                    f"values give {actual_mass:.12g} amu and "
                    f"{actual_charge_e:+.12g} e. Select the matching general "
                    f"{role} species preset, or make the Custom particle mass "
                    "and charge match the magnetic preset."
                )

        validate_role("rider", rider_params, magnetic_options["rider_magnetic_species"])
        if driver_params is not None:
            validate_role(
                "driver",
                driver_params,
                magnetic_options["driver_magnetic_species"],
            )

    def _apply_external_magnetic_options_to_ui(
        self: Any, options: SimulationOptions
    ) -> None:
        """Populate native, tesla, and T/m prescribed magnetic-field controls."""
        native = getattr(options, "external_magnetic_field_native", (0.0, 0.0, 0.0))
        for var, value in zip(self.external_magnetic_native_vars, native):
            var.set(_format_gui_float(value))
        for var, value in zip(self.external_magnetic_tesla_vars, native):
            var.set(_format_gui_float(magnetic_field_native_to_tesla(value)))

        gradient = getattr(
            options,
            "external_magnetic_field_gradient_t_per_m",
            ((0.0, 0.0, 0.0),) * 3,
        )
        for variable_row, value_row in zip(
            self.external_magnetic_gradient_vars, gradient
        ):
            for var, value in zip(variable_row, value_row):
                var.set(_format_gui_float(value))

    def _build_external_magnetic_options_from_ui(
        self: Any, *, enabled: bool
    ) -> dict[str, Any]:
        """Return prescribed B in native units and its static gradient in T/m."""
        strict_parser = _parse_gui_float if enabled else None

        def parse(variable: Any, label: str) -> float:
            if strict_parser is not None:
                return strict_parser(variable.get(), label)
            return _parse_gui_float_lenient(variable.get(), 0.0)

        if self.external_field_input_mode_var.get() == "SI V/m":
            magnetic_native = tuple(
                magnetic_field_tesla_to_native(parse(var, f"External B T {axis}"))
                for var, axis in zip(self.external_magnetic_tesla_vars, ("x", "y", "z"))
            )
        else:
            magnetic_native = tuple(
                parse(var, f"External B native {axis}")
                for var, axis in zip(
                    self.external_magnetic_native_vars, ("x", "y", "z")
                )
            )

        gradient = tuple(
            tuple(
                parse(var, f"External dB{component}/d{coordinate} T/m")
                for var, coordinate in zip(variable_row, ("x", "y", "z"))
            )
            for variable_row, component in zip(
                self.external_magnetic_gradient_vars, ("x", "y", "z")
            )
        )
        return {
            "external_magnetic_field_native": magnetic_native,
            "external_magnetic_field_gradient_t_per_m": gradient,
        }

    def _apply_macroparticle_smearing_options_to_ui(
        self: Any, options: SimulationOptions
    ) -> None:
        self.macroparticle_smearing_enabled_var.set(
            getattr(options, "macroparticle_smearing_enabled", False)
        )
        self.macroparticle_smearing_subcharge_count_var.set(
            getattr(options, "macroparticle_smearing_subcharge_count", 8)
        )
        self.macroparticle_smearing_sigma_multiplier_var.set(
            getattr(options, "macroparticle_smearing_sigma_multiplier", 1.0)
        )
        self.macroparticle_smearing_position_sigma_var.set(
            _format_gui_optional_float(
                getattr(options, "macroparticle_smearing_position_sigma_mm", None)
            )
        )
        self.macroparticle_smearing_longitudinal_sigma_var.set(
            _format_gui_optional_float(
                getattr(options, "macroparticle_smearing_longitudinal_sigma_mm", None)
            )
        )
        self.macroparticle_smearing_momentum_sigma_var.set(
            _format_gui_optional_float(
                getattr(
                    options,
                    "macroparticle_smearing_momentum_sigma_amu_mm_ns",
                    None,
                )
            )
        )
        self.macroparticle_smearing_use_position_errors_var.set(
            getattr(options, "macroparticle_smearing_use_position_errors", True)
        )
        self.macroparticle_smearing_use_momentum_errors_var.set(
            getattr(options, "macroparticle_smearing_use_momentum_errors", True)
        )
        self.macroparticle_smearing_use_centroid_errors_var.set(
            getattr(options, "macroparticle_smearing_use_centroid_errors", True)
        )
        self.macroparticle_smearing_use_internal_cloud_var.set(
            getattr(options, "macroparticle_smearing_use_internal_cloud", True)
        )
        self.macroparticle_smearing_apply_to_active_observers_var.set(
            getattr(options, "macroparticle_smearing_apply_to_active_observers", True)
        )
        self.macroparticle_smearing_apply_to_active_sources_var.set(
            getattr(options, "macroparticle_smearing_apply_to_active_sources", True)
        )
        self.macroparticle_smearing_apply_to_passive_sources_var.set(
            getattr(options, "macroparticle_smearing_apply_to_passive_sources", True)
        )
        self.macroparticle_smearing_apply_to_passive_updates_var.set(
            getattr(options, "macroparticle_smearing_apply_to_passive_updates", False)
        )
        self.macroparticle_smearing_seed_var.set(
            getattr(options, "macroparticle_smearing_seed", 12345)
        )
        self.macroparticle_smearing_refresh_policy_var.set(
            str(
                getattr(
                    options,
                    "macroparticle_smearing_refresh_policy",
                    "fixed_per_particle",
                )
            ).replace("-", "_")
        )

    def _build_macroparticle_smearing_options_from_ui(
        self: Any,
    ) -> dict[str, Any]:
        return {
            "macroparticle_smearing_enabled": bool(
                self.macroparticle_smearing_enabled_var.get()
            ),
            "macroparticle_smearing_subcharge_count": int(
                self.macroparticle_smearing_subcharge_count_var.get()
            ),
            "macroparticle_smearing_sigma_multiplier": float(
                self.macroparticle_smearing_sigma_multiplier_var.get()
            ),
            "macroparticle_smearing_position_sigma_mm": _parse_gui_optional_float(
                self.macroparticle_smearing_position_sigma_var.get(),
                "Macroparticle smearing position sigma",
            ),
            "macroparticle_smearing_longitudinal_sigma_mm": _parse_gui_optional_float(
                self.macroparticle_smearing_longitudinal_sigma_var.get(),
                "Macroparticle smearing longitudinal sigma",
            ),
            "macroparticle_smearing_momentum_sigma_amu_mm_ns": _parse_gui_optional_float(
                self.macroparticle_smearing_momentum_sigma_var.get(),
                "Macroparticle smearing momentum sigma",
            ),
            "macroparticle_smearing_use_position_errors": bool(
                self.macroparticle_smearing_use_position_errors_var.get()
            ),
            "macroparticle_smearing_use_momentum_errors": bool(
                self.macroparticle_smearing_use_momentum_errors_var.get()
            ),
            "macroparticle_smearing_use_centroid_errors": bool(
                self.macroparticle_smearing_use_centroid_errors_var.get()
            ),
            "macroparticle_smearing_use_internal_cloud": bool(
                self.macroparticle_smearing_use_internal_cloud_var.get()
            ),
            "macroparticle_smearing_apply_to_active_observers": bool(
                self.macroparticle_smearing_apply_to_active_observers_var.get()
            ),
            "macroparticle_smearing_apply_to_active_sources": bool(
                self.macroparticle_smearing_apply_to_active_sources_var.get()
            ),
            "macroparticle_smearing_apply_to_passive_sources": bool(
                self.macroparticle_smearing_apply_to_passive_sources_var.get()
            ),
            "macroparticle_smearing_apply_to_passive_updates": bool(
                self.macroparticle_smearing_apply_to_passive_updates_var.get()
            ),
            "macroparticle_smearing_seed": int(
                self.macroparticle_smearing_seed_var.get()
            ),
            "macroparticle_smearing_refresh_policy": str(
                self.macroparticle_smearing_refresh_policy_var.get()
            ).replace("-", "_"),
        }

    def _load_config(self) -> None:
        from tkinter import messagebox

        from .gui import _show_error_dialog

        filename = self._selected_config_filename()
        if not filename:
            messagebox.showinfo("Load config", "Select a configuration to load.")
            return

        path = Path(self.config_dir_var.get()) / filename
        if _looks_like_sweep_or_optimization_config(path):
            self._load_sweep_or_optimization_config(path)
            return

        try:
            options = load_config(path)
        except Exception as exc:
            _show_error_dialog(
                self.root, "Load config", f"Failed to load {filename}: {exc}"
            )
            return

        self._apply_options_to_ui(options, preserve_directories=True)
        self.config_name_var.set(filename)
        self.config_file_var.set(filename)

        self.run_mode_var.set("single")
        self._on_run_mode_changed()
        print("[INFO] Auto-switched to Single Run mode")

        self._refresh_config_list(selected=filename)
        self._refresh_initial_summary()
        self._update_driver_visibility()
        self._update_image_subcharge_state()
        self._update_cavity_spacing_state()
        self._toggle_z_cutoff_controls()
        self._toggle_macroparticle_controls()
        self._update_macroparticle_state()
        if hasattr(self, "_toggle_pseudo_grid_controls"):
            self._toggle_pseudo_grid_controls()
        if hasattr(self, "_update_pseudo_grid_state"):
            self._update_pseudo_grid_state()
        if hasattr(self, "_update_driver_train_state"):
            self._update_driver_train_state()

        current_value = self.sim_type_var.get()
        try:
            values_list = list(self.sim_type_combo["values"])
            if current_value in values_list:
                idx = values_list.index(current_value)
                self.sim_type_combo.current(idx)
                self.root.update_idletasks()
        except Exception:
            pass

        self._set_status(f"Loaded config: {filename}")
        self.current_config_label.config(text=filename, foreground="black")

    def _load_sweep_or_optimization_config(self, path: Path) -> None:
        from .gui import _show_error_dialog

        if not hasattr(self, "optimization_tab"):
            _show_error_dialog(
                self.root,
                "Load config",
                f"{path.name} is a sweep/optimization config, but the sweep tab is unavailable.",
            )
            return

        try:
            self.optimization_tab._load_config_from_path(str(path))
        except Exception as exc:
            _show_error_dialog(
                self.root, "Load config", f"Failed to load {path.name}: {exc}"
            )
            return

        if hasattr(self, "sweep_config_name_var"):
            self.sweep_config_name_var.set(path.name)
        if hasattr(self, "sweep_config_dir_var"):
            self.sweep_config_dir_var.set(str(path.parent))
        if hasattr(self.optimization_tab, "sweep_config_dir"):
            self.optimization_tab.sweep_config_dir = str(path.parent)
        if hasattr(self, "run_mode_var"):
            self.run_mode_var.set("sweep")
        if hasattr(self, "_on_run_mode_changed"):
            self._on_run_mode_changed()
        if hasattr(self, "_refresh_sweep_config_list"):
            self._refresh_sweep_config_list(selected=path.name)
        if hasattr(self, "current_sweep_config_label"):
            self.current_sweep_config_label.config(
                text=path.name, foreground="black", font=("TkDefaultFont", 9)
            )
        if hasattr(self, "_set_status"):
            self._set_status(f"Loaded sweep/optimization config: {path.name}")
        print("[INFO] Auto-switched to Sweep/Optim run mode")

    def _apply_options_to_ui(
        self, options: SimulationOptions, preserve_directories: bool = False
    ) -> None:
        self.options = options
        self.sim_type_var.set(options.simulation_type.name)
        self.steps_var.set(options.steps)
        self.seed_var.set(options.seed)
        self.energy_display_var.set(options.energy_display)
        self.energy_save_var.set(options.energy_save)
        self.energy_xaxis_var.set(getattr(options, "energy_xaxis", "z"))
        self.energy_yaxis_var.set(getattr(options, "energy_yaxis", "delta_total"))
        self.transverse_display_var.set(options.transverse_display)
        self.transverse_save_var.set(options.transverse_save)
        self.transverse_xaxis_var.set(getattr(options, "transverse_xaxis", "t"))
        self.beta_display_var.set(options.beta_display)
        self.beta_save_var.set(options.beta_save)
        self.beta_xaxis_var.set(getattr(options, "beta_xaxis", "t"))
        self.momentum_display_var.set(options.momentum_display)
        self.momentum_save_var.set(options.momentum_save)
        self.momentum_xaxis_var.set(getattr(options, "momentum_xaxis", "t"))
        self.gamma_display_var.set(getattr(options, "gamma_display", False))
        self.gamma_save_var.set(getattr(options, "gamma_save", False))
        self.gamma_xaxis_var.set(getattr(options, "gamma_xaxis", "t"))
        self.zposition_display_var.set(getattr(options, "zposition_display", False))
        self.zposition_save_var.set(getattr(options, "zposition_save", False))
        self.trajectory_save_var.set(options.trajectory_save)
        self.trajectory_interval_var.set(options.trajectory_interval)
        self.dpi_var.set(options.plot_dpi)
        self.image_subcharge_var.set(options.image_subcharge_count)
        self.image_weighting_var.set(options.use_image_weighting)
        self.use_numba_var.set(getattr(options, "use_numba", True))
        self.macroparticle_enabled_var.set(
            getattr(options, "macroparticle_enabled", False)
        )
        self.macroparticle_charge_multiplier_var.set(
            getattr(options, "macroparticle_charge_multiplier", 1.0)
        )
        self.macroparticle_sigma_multiplier_var.set(
            getattr(options, "macroparticle_sigma_multiplier", 1.0)
        )
        self.macroparticle_use_momentum_errors_var.set(
            getattr(options, "macroparticle_use_momentum_errors", True)
        )
        self._apply_macroparticle_smearing_options_to_ui(options)
        self._apply_magnetic_dipole_options_to_ui(options)
        if hasattr(self, "pseudo_grid_enabled_var"):
            self.pseudo_grid_enabled_var.set(
                getattr(options, "pseudo_grid_enabled", False)
            )
            self.pseudo_grid_active_rider_count_var.set(
                getattr(options, "pseudo_grid_active_rider_count", 4)
            )
            self.pseudo_grid_active_driver_count_var.set(
                getattr(options, "pseudo_grid_active_driver_count", 4)
            )
            self.pseudo_grid_field_rider_count_var.set(
                getattr(options, "pseudo_grid_field_rider_count", 0)
            )
            self.pseudo_grid_field_driver_count_var.set(
                getattr(options, "pseudo_grid_field_driver_count", 0)
            )
            self.pseudo_grid_field_deposition_neighbor_count_var.set(
                getattr(options, "pseudo_grid_field_deposition_neighbor_count", 4)
            )
            self.pseudo_grid_passive_neighbor_count_var.set(
                getattr(options, "pseudo_grid_passive_neighbor_count", 4)
            )
            self.pseudo_grid_coverage_strategy_var.set(
                getattr(
                    options,
                    "pseudo_grid_coverage_strategy",
                    "farthest_point_staleness",
                )
            )
            self.pseudo_grid_coverage_space_var.set(
                getattr(options, "pseudo_grid_coverage_space", "position")
            )
            self.pseudo_grid_pair_reuse_window_var.set(
                getattr(options, "pseudo_grid_pair_reuse_window", 16)
            )
            self.pseudo_grid_source_weighting_mode_var.set(
                getattr(
                    options,
                    "pseudo_grid_source_weighting_mode",
                    "inverse_distance",
                )
            )
            self.pseudo_grid_loss_tracking_enabled_var.set(
                getattr(options, "pseudo_grid_loss_tracking_enabled", True)
            )
            self.pseudo_grid_causal_history_pruning_enabled_var.set(
                getattr(
                    options,
                    "pseudo_grid_causal_history_pruning_enabled",
                    False,
                )
            )
            self.pseudo_grid_causal_history_safety_margin_steps_var.set(
                getattr(
                    options,
                    "pseudo_grid_causal_history_safety_margin_steps",
                    2,
                )
            )
        if hasattr(self, "driver_train_enabled_var"):
            self.driver_train_enabled_var.set(
                getattr(options, "driver_train_enabled", False)
            )
            self.driver_train_bunch_count_var.set(
                getattr(options, "driver_train_bunch_count", 1)
            )
            self.driver_train_z_spacing_mm_var.set(
                getattr(options, "driver_train_z_spacing_mm", 0.0)
            )
            self.driver_train_z_offsets_mm_var.set(
                " ".join(
                    _format_gui_float(value)
                    for value in getattr(options, "driver_train_z_offsets_mm", ())
                )
            )
            self.driver_train_prehistory_steps_var.set(
                getattr(options, "driver_train_prehistory_steps", 0)
            )
            self.driver_train_preserve_prehistory_var.set(
                getattr(
                    options,
                    "driver_train_preserve_prehistory_in_output",
                    False,
                )
            )
        self.self_consistency_enabled_var.set(options.self_consistency_enabled)
        self.self_consistency_convergence_mode_var.set(
            options.self_consistency_convergence_mode
        )
        self.self_consistency_mass_shell_relaxation_var.set(
            options.self_consistency_mass_shell_relaxation
        )
        self.self_consistency_target_ms_tolerance_var.set(
            options.self_consistency_target_ms_tolerance
        )
        self.self_consistency_max_iterations_var.set(
            options.self_consistency_max_iterations
        )
        self.self_consistency_mass_shell_tolerance_var.set(
            options.self_consistency_mass_shell_tolerance
        )
        self.self_consistency_verbosity_var.set(options.self_consistency_verbosity)
        self.chrono_interpolate_var.set(
            getattr(
                options,
                "chrono_interpolate",
                getattr(options, "self_consistency_chrono_interpolate", False),
            )
        )
        self.chrono_tolerance_var.set(
            getattr(
                options,
                "chrono_tolerance",
                getattr(options, "self_consistency_chrono_tolerance", 1e-3),
            )
        )
        self.chrono_high_precision_var.set(
            getattr(
                options,
                "chrono_high_precision",
                getattr(options, "self_consistency_chrono_high_precision", False),
            )
        )
        self.chrono_adaptive_tolerance_var.set(
            getattr(
                options,
                "chrono_adaptive_tolerance",
                getattr(options, "self_consistency_chrono_adaptive_tolerance", False),
            )
        )
        self.self_consistency_gamma_reconciliation_method_var.set(
            getattr(
                options,
                "self_consistency_gamma_reconciliation_method",
                "DISABLED",
            )
        )
        self.self_consistency_gamma_reconciliation_low_beta_threshold_var.set(
            getattr(
                options, "self_consistency_gamma_reconciliation_low_beta_threshold", 0.9
            )
        )
        self.self_consistency_gamma_reconciliation_high_beta_threshold_var.set(
            getattr(
                options,
                "self_consistency_gamma_reconciliation_high_beta_threshold",
                0.99,
            )
        )
        self.self_consistency_gamma_reconciliation_low_beta_weight_var.set(
            getattr(
                options, "self_consistency_gamma_reconciliation_low_beta_weight", 0.8
            )
        )
        self.self_consistency_gamma_reconciliation_high_beta_weight_var.set(
            getattr(
                options, "self_consistency_gamma_reconciliation_high_beta_weight", 0.2
            )
        )
        self.self_consistency_gamma_reconciliation_mid_beta_weight_var.set(
            getattr(
                options, "self_consistency_gamma_reconciliation_mid_beta_weight", 0.5
            )
        )
        self.self_consistency_gamma_reconciliation_fixed_weight_var.set(
            getattr(options, "self_consistency_gamma_reconciliation_fixed_weight", 0.5)
        )
        self.adaptive_timestep_enabled_var.set(options.adaptive_timestep_enabled)
        self.adaptive_timestep_halt_on_jump_var.set(options.energy_monitor_halt_on_jump)
        self.adaptive_timestep_threshold_var.set(options.adaptive_timestep_threshold)
        self.adaptive_timestep_reduction_factor_var.set(
            options.adaptive_timestep_reduction_factor
        )
        self._update_max_attempts_display()
        self.adaptive_timestep_min_factor_var.set(options.adaptive_timestep_min_factor)
        self.adaptive_timestep_cooldown_steps_var.set(
            options.adaptive_timestep_cooldown_steps
        )
        self.adaptive_timestep_probe_threshold_var.set(
            options.adaptive_timestep_probe_threshold
        )
        self.adaptive_timestep_max_probe_steps_var.set(
            options.adaptive_timestep_max_probe_steps
        )
        self.adaptive_timestep_debug_var.set(options.adaptive_timestep_debug)
        self.adaptive_timestep_bunch_proximity_enabled_var.set(
            getattr(options, "adaptive_timestep_bunch_proximity_enabled", False)
        )
        self.adaptive_timestep_bunch_proximity_sigma_mm_var.set(
            getattr(options, "adaptive_timestep_bunch_proximity_sigma_mm", 5.0)
        )
        self.adaptive_timestep_bunch_proximity_n_sigma_var.set(
            getattr(options, "adaptive_timestep_bunch_proximity_n_sigma", 5.0)
        )
        self.adaptive_timestep_bunch_proximity_reduction_factor_var.set(
            getattr(options, "adaptive_timestep_bunch_proximity_reduction_factor", 10.0)
        )
        self.adaptive_timestep_bunch_proximity_transition_n_sigma_var.set(
            getattr(
                options,
                "adaptive_timestep_bunch_proximity_transition_n_sigma",
                2.0,
            )
        )
        self._update_max_substeps_display()
        self.radiation_reaction_mode_var.set(
            getattr(options, "radiation_reaction_mode", "medina_lad")
        )
        self.space_charge_enabled_var.set(
            getattr(options, "space_charge_enabled", False)
        )
        self.space_charge_retarded_var.set(
            getattr(options, "space_charge_retarded", True)
        )
        self.space_charge_softening_mm_var.set(
            getattr(options, "space_charge_softening_mm", 0.0)
        )
        self.space_charge_bunch_sigma_mm_var.set(
            getattr(options, "space_charge_bunch_sigma_mm", 0.01)
        )
        min_ret_steps = getattr(options, "space_charge_min_retarded_steps", None)
        self.space_charge_min_retarded_steps_var.set(
            "" if min_ret_steps is None else str(min_ret_steps)
        )
        self.external_field_enabled_var.set(
            getattr(options, "external_field_enabled", False)
        )
        electric_si = getattr(options, "external_electric_field_v_per_m", None)
        electric_native = getattr(
            options, "external_electric_field_native", (0.0, 0.0, 0.0)
        )
        use_si_inputs = electric_si is not None or not any(electric_native)
        self.external_field_input_mode_var.set("SI V/m" if use_si_inputs else "Native")
        for var, value in zip(
            self.external_electric_native_vars,
            electric_native,
        ):
            var.set(_format_gui_float(value))
        for var, value in zip(
            self.external_electric_si_vars, electric_si or (0.0, 0.0, 0.0)
        ):
            var.set(_format_gui_float(value))
        self._apply_external_magnetic_options_to_ui(options)
        for axis in ("x", "y", "z", "t"):
            for bound in ("min", "max"):
                key = f"{axis}_{bound}"
                option_name = f"external_field_{key}"
                self.external_field_window_vars[key].set(
                    _format_gui_optional_float(getattr(options, option_name, None))
                )
        if hasattr(self, "cavity_exit_enabled_var"):
            self.cavity_exit_enabled_var.set(
                getattr(options, "cavity_exit_enabled", False)
            )
            self.cavity_exit_mode_var.set(
                getattr(options, "cavity_exit_mode", "first_exit")
            )
            cavity_length = getattr(options, "cavity_exit_length_mm", None)
            self.cavity_exit_length_mm_var.set(
                "" if cavity_length is None else _format_gui_float(cavity_length)
            )
            self._toggle_cavity_exit_controls()
        if hasattr(self, "beamline_geometry_enabled_var"):
            self.beamline_geometry_enabled_var.set(
                bool(getattr(options, "beamline_geometry_enabled", False))
            )
            if hasattr(self, "beamline_geometry_text"):
                geom_payload = {
                    "enabled": bool(
                        getattr(options, "beamline_geometry_enabled", False)
                    ),
                    "occluders": list(
                        getattr(options, "beamline_geometry_occluders", []) or []
                    ),
                }
                self._set_text_widget_content(
                    self.beamline_geometry_text,
                    json.dumps(geom_payload, indent=2),
                )
            self._toggle_beamline_geometry_controls()
        manual_particle_config_enabled = bool(
            getattr(options, "manual_particle_config_enabled", False)
        ) or _particle_params_require_manual_config(options.rider_params)
        if options.driver_params is not None:
            manual_particle_config_enabled = manual_particle_config_enabled or (
                _particle_params_require_manual_config(options.driver_params)
            )
        if hasattr(self, "manual_particle_config_enabled_var"):
            self.manual_particle_config_enabled_var.set(manual_particle_config_enabled)
            if hasattr(self, "manual_rider_config_text"):
                self._set_text_widget_content(
                    self.manual_rider_config_text,
                    json.dumps(dict(options.rider_params), indent=2),
                )
            if hasattr(self, "manual_driver_config_text"):
                driver_payload = (
                    dict(options.driver_params)
                    if options.driver_params is not None
                    else dict(DEFAULT_DRIVER_PARAMS)
                )
                self._set_text_widget_content(
                    self.manual_driver_config_text,
                    json.dumps(driver_payload, indent=2),
                )
            self._toggle_manual_particle_config_controls()
        self.auto_duration_enabled_var.set(
            getattr(options, "auto_duration_enabled", False)
        )
        self.auto_duration_crossing_steps_var.set(
            getattr(options, "auto_duration_crossing_steps", 200)
        )
        self.auto_duration_post_factor_var.set(
            getattr(options, "auto_duration_post_factor", 2.0)
        )
        if hasattr(self, "_toggle_macroparticle_smearing_controls"):
            self._toggle_macroparticle_smearing_controls()
        if hasattr(self, "_toggle_magnetic_dipole_controls"):
            self._toggle_magnetic_dipole_controls()
        self._toggle_space_charge_controls()
        self._toggle_external_field_controls()
        self._toggle_auto_duration_controls()
        if hasattr(self, "_toggle_pseudo_grid_controls"):
            self._toggle_pseudo_grid_controls()
        if hasattr(self, "_update_pseudo_grid_state"):
            self._update_pseudo_grid_state()
        if hasattr(self, "_update_driver_train_state"):
            self._update_driver_train_state()
        self.save_log_file_var.set(options.save_log_file)

        if not preserve_directories:
            self.output_dir_var.set(str(options.output_dir))
            self.config_dir_var.set(str(options.config_dir))

        self.config_name_var.set(options.config_name)
        self.checkpoint_enabled_var.set(options.checkpoint_enabled)
        self.checkpoint_directory_var.set(
            ""
            if options.checkpoint_directory is None
            else str(options.checkpoint_directory)
        )
        self.checkpoint_resume_from_var.set(
            ""
            if options.checkpoint_resume_from is None
            else str(options.checkpoint_resume_from)
        )
        self.checkpoint_interval_steps_var.set(options.checkpoint_interval_steps)
        self.checkpoint_interval_seconds_var.set(options.checkpoint_interval_seconds)

        default_species_label = self._species_label_by_key.get(
            "custom", next(iter(self._species_by_label))
        )
        self.rider_species_var.set(default_species_label)
        self.driver_species_var.set(default_species_label)

        for name in PARTICLE_PARAM_FIELDS:
            rider_value = options.rider_params.get(name, DEFAULT_RIDER_PARAMS[name])
            self.rider_param_vars[name].set(rider_value)
            driver_value = (
                options.driver_params[name]
                if options.driver_params is not None and name in options.driver_params
                else DEFAULT_DRIVER_PARAMS[name]
            )
            self.driver_param_vars[name].set(driver_value)
        for name in CORE_PARAM_DEFAULTS:
            self.core_param_vars[name].set(options.core_params[name])

        z_cutoff_val = options.core_params.get("z_cutoff", 0.0)
        self.z_cutoff_enabled_var.set(z_cutoff_val != 0.0)
        self._toggle_z_cutoff_controls()

    def _update_max_attempts_display(self):
        """Update derived adaptive-timestep attempt display."""
        try:
            reduction_factor = self.adaptive_timestep_reduction_factor_var.get()
            min_factor = self.adaptive_timestep_min_factor_var.get()

            if reduction_factor <= 1 or min_factor <= 0:
                self.adaptive_timestep_max_attempts_display_var.set("N/A")
                return

            attempts = math.ceil(
                math.log(1.0 / min_factor) / math.log(reduction_factor)
            )
            attempts = max(1, attempts)
            self.adaptive_timestep_max_attempts_display_var.set(
                f"{attempts} (from reduction & min factor)"
            )
        except (ValueError, ZeroDivisionError):
            self.adaptive_timestep_max_attempts_display_var.set("N/A")

    def _update_max_substeps_display(self):
        """Update derived adaptive-timestep substep display."""
        try:
            min_factor = self.adaptive_timestep_min_factor_var.get()
            theoretical_max = math.ceil(1.0 / min_factor)
            with_margin = int(theoretical_max * 1.1)
            self.adaptive_timestep_max_substeps_display_var.set(
                f"{with_margin} (from min factor)"
            )
        except (ValueError, ZeroDivisionError):
            self.adaptive_timestep_max_substeps_display_var.set("N/A")

    def _build_options_from_ui(self) -> SimulationOptions:
        sim_type = SimulationType[self.sim_type_var.get()]
        driver_supported = supports_driver(sim_type)
        manual_particle_config_enabled = bool(
            self.manual_particle_config_enabled_var.get()
        )
        if manual_particle_config_enabled:
            rider_params = self._collect_manual_particle_payload("rider", strict=True)
            driver_params = (
                self._collect_manual_particle_payload("driver", strict=True)
                if driver_supported
                else None
            )
        else:
            rider_params = {
                name: self.rider_param_vars[name].get()
                for name in PARTICLE_PARAM_FIELDS
            }
            driver_params = (
                {
                    name: self.driver_param_vars[name].get()
                    for name in PARTICLE_PARAM_FIELDS
                }
                if driver_supported
                else None
            )
        core_params = {}
        for name in CORE_PARAM_DEFAULTS:
            value = self.core_param_vars[name].get()
            if isinstance(CORE_PARAM_DEFAULTS[name], str):
                core_params[name] = value
            else:
                core_params[name] = float(value)

        if not self.z_cutoff_enabled_var.get():
            core_params["z_cutoff"] = 0.0

        config_name = self.config_name_var.get().strip() or "testbed_config"
        if not config_name.endswith(".json"):
            config_name += ".json"

        if self.random_seed_var.get():
            import random

            seed = random.randint(1, 2**31 - 1)
        else:
            seed = int(self.seed_var.get())

        external_field_enabled = bool(self.external_field_enabled_var.get())
        external_input_mode = self.external_field_input_mode_var.get()

        driver_train_offsets_text = self.driver_train_z_offsets_mm_var.get().strip()
        driver_train_offsets = tuple(
            _parse_gui_float(part, "Driver-train z offset")
            for part in driver_train_offsets_text.replace(",", " ").split()
        )
        driver_train_bunch_count = int(self.driver_train_bunch_count_var.get())
        driver_train_enabled = bool(self.driver_train_enabled_var.get())
        pseudo_grid_enabled = bool(self.pseudo_grid_enabled_var.get())
        if (
            driver_train_offsets
            and len(driver_train_offsets) != driver_train_bunch_count
        ):
            raise ValueError(
                "Driver-train explicit z offsets must match the driver bunch count."
            )

        if external_field_enabled:
            external_electric_native = tuple(
                _parse_gui_float(var.get(), f"External E native {axis}")
                for var, axis in zip(
                    self.external_electric_native_vars, ("x", "y", "z")
                )
            )
            external_electric_si = None
            if external_input_mode == "SI V/m":
                external_electric_si = tuple(
                    _parse_gui_float(var.get(), f"External E V/m {axis}")
                    for var, axis in zip(
                        self.external_electric_si_vars, ("x", "y", "z")
                    )
                )
            external_bounds = {
                f"{axis}_{bound}": _parse_gui_optional_float(
                    self.external_field_window_vars[f"{axis}_{bound}"].get(),
                    f"External field {axis}_{bound}",
                )
                for axis in ("x", "y", "z", "t")
                for bound in ("min", "max")
            }
        else:
            external_electric_native = tuple(
                _parse_gui_float_lenient(var.get(), 0.0)
                for var in self.external_electric_native_vars
            )
            external_electric_si = None
            if external_input_mode == "SI V/m":
                external_electric_si = tuple(
                    _parse_gui_float_lenient(var.get(), 0.0)
                    for var in self.external_electric_si_vars
                )
            external_bounds = {
                f"{axis}_{bound}": _parse_gui_optional_float_lenient(
                    self.external_field_window_vars[f"{axis}_{bound}"].get()
                )
                for axis in ("x", "y", "z", "t")
                for bound in ("min", "max")
            }

        external_magnetic_options = self._build_external_magnetic_options_from_ui(
            enabled=external_field_enabled
        )
        magnetic_dipole_options = self._build_magnetic_dipole_options_from_ui()
        self._validate_magnetic_species_particle_matches(
            magnetic_options=magnetic_dipole_options,
            rider_params=rider_params,
            driver_params=driver_params,
        )
        macroparticle_smearing_options = (
            self._build_macroparticle_smearing_options_from_ui()
        )

        checkpoint_resume_text = self.checkpoint_resume_from_var.get().strip()
        checkpoint_directory_text = self.checkpoint_directory_var.get().strip()

        return SimulationOptions(
            simulation_type=sim_type,
            steps=int(self.steps_var.get()),
            seed=seed,
            rider_params=rider_params,
            driver_params=driver_params,
            core_params=core_params,
            energy_display=bool(self.energy_display_var.get()),
            energy_save=bool(self.energy_save_var.get()),
            energy_xaxis=str(self.energy_xaxis_var.get()),
            energy_yaxis=str(self.energy_yaxis_var.get()),
            transverse_display=bool(self.transverse_display_var.get()),
            transverse_save=bool(self.transverse_save_var.get()),
            transverse_xaxis=str(self.transverse_xaxis_var.get()),
            beta_display=bool(self.beta_display_var.get()),
            beta_save=bool(self.beta_save_var.get()),
            beta_xaxis=str(self.beta_xaxis_var.get()),
            momentum_display=bool(self.momentum_display_var.get()),
            momentum_save=bool(self.momentum_save_var.get()),
            momentum_xaxis=str(self.momentum_xaxis_var.get()),
            gamma_display=bool(self.gamma_display_var.get()),
            gamma_save=bool(self.gamma_save_var.get()),
            gamma_xaxis=str(self.gamma_xaxis_var.get()),
            zposition_display=bool(self.zposition_display_var.get()),
            zposition_save=bool(self.zposition_save_var.get()),
            trajectory_save=bool(self.trajectory_save_var.get()),
            trajectory_interval=int(self.trajectory_interval_var.get()),
            plot_dpi=int(self.dpi_var.get()),
            output_dir=Path(self.output_dir_var.get()),
            config_dir=Path(self.config_dir_var.get()),
            config_name=config_name,
            checkpoint_enabled=bool(
                self.checkpoint_enabled_var.get() or checkpoint_resume_text
            ),
            checkpoint_directory=(
                None
                if checkpoint_resume_text or not checkpoint_directory_text
                else Path(checkpoint_directory_text)
            ),
            checkpoint_resume_from=(
                Path(checkpoint_resume_text) if checkpoint_resume_text else None
            ),
            checkpoint_interval_steps=int(self.checkpoint_interval_steps_var.get()),
            checkpoint_interval_seconds=float(
                self.checkpoint_interval_seconds_var.get()
            ),
            manual_particle_config_enabled=manual_particle_config_enabled,
            image_subcharge_count=int(self.image_subcharge_var.get()),
            use_image_weighting=bool(self.image_weighting_var.get()),
            macroparticle_enabled=bool(self.macroparticle_enabled_var.get()),
            macroparticle_charge_multiplier=float(
                self.macroparticle_charge_multiplier_var.get()
            ),
            macroparticle_sigma_multiplier=float(
                self.macroparticle_sigma_multiplier_var.get()
            ),
            macroparticle_use_momentum_errors=bool(
                self.macroparticle_use_momentum_errors_var.get()
            ),
            **macroparticle_smearing_options,
            **magnetic_dipole_options,
            self_consistency_enabled=bool(self.self_consistency_enabled_var.get()),
            self_consistency_convergence_mode=str(
                self.self_consistency_convergence_mode_var.get()
            ),
            self_consistency_mass_shell_relaxation=float(
                self.self_consistency_mass_shell_relaxation_var.get()
            ),
            self_consistency_target_ms_tolerance=float(
                self.self_consistency_target_ms_tolerance_var.get()
            ),
            self_consistency_max_iterations=int(
                self.self_consistency_max_iterations_var.get()
            ),
            self_consistency_mass_shell_tolerance=float(
                self.self_consistency_mass_shell_tolerance_var.get()
            ),
            self_consistency_verbosity=int(self.self_consistency_verbosity_var.get()),
            chrono_interpolate=bool(self.chrono_interpolate_var.get()),
            chrono_tolerance=float(self.chrono_tolerance_var.get()),
            chrono_high_precision=bool(self.chrono_high_precision_var.get()),
            chrono_adaptive_tolerance=bool(self.chrono_adaptive_tolerance_var.get()),
            self_consistency_chrono_interpolate=bool(self.chrono_interpolate_var.get()),
            self_consistency_chrono_tolerance=float(self.chrono_tolerance_var.get()),
            self_consistency_chrono_high_precision=bool(
                self.chrono_high_precision_var.get()
            ),
            self_consistency_chrono_adaptive_tolerance=bool(
                self.chrono_adaptive_tolerance_var.get()
            ),
            self_consistency_gamma_reconciliation_method=self.self_consistency_gamma_reconciliation_method_var.get(),
            self_consistency_gamma_reconciliation_low_beta_threshold=float(
                self.self_consistency_gamma_reconciliation_low_beta_threshold_var.get()
            ),
            self_consistency_gamma_reconciliation_high_beta_threshold=float(
                self.self_consistency_gamma_reconciliation_high_beta_threshold_var.get()
            ),
            self_consistency_gamma_reconciliation_low_beta_weight=float(
                self.self_consistency_gamma_reconciliation_low_beta_weight_var.get()
            ),
            self_consistency_gamma_reconciliation_high_beta_weight=float(
                self.self_consistency_gamma_reconciliation_high_beta_weight_var.get()
            ),
            self_consistency_gamma_reconciliation_mid_beta_weight=float(
                self.self_consistency_gamma_reconciliation_mid_beta_weight_var.get()
            ),
            self_consistency_gamma_reconciliation_fixed_weight=float(
                self.self_consistency_gamma_reconciliation_fixed_weight_var.get()
            ),
            chrono_matching_mode="FAST",
            self_consistency_chrono_matching_mode="FAST",
            energy_monitor_enabled=False,
            energy_monitor_threshold=2.0,
            energy_monitor_check_interval=10,
            energy_monitor_halt_on_jump=bool(
                self.adaptive_timestep_halt_on_jump_var.get()
            ),
            energy_monitor_debug=False,
            adaptive_timestep_enabled=bool(self.adaptive_timestep_enabled_var.get()),
            adaptive_timestep_threshold=float(
                self.adaptive_timestep_threshold_var.get()
            ),
            adaptive_timestep_reduction_factor=int(
                self.adaptive_timestep_reduction_factor_var.get()
            ),
            adaptive_timestep_min_factor=float(
                self.adaptive_timestep_min_factor_var.get()
            ),
            adaptive_timestep_cooldown_steps=int(
                self.adaptive_timestep_cooldown_steps_var.get()
            ),
            adaptive_timestep_probe_threshold=float(
                self.adaptive_timestep_probe_threshold_var.get()
            ),
            adaptive_timestep_max_probe_steps=int(
                self.adaptive_timestep_max_probe_steps_var.get()
            ),
            adaptive_timestep_debug=bool(self.adaptive_timestep_debug_var.get()),
            adaptive_timestep_bunch_proximity_enabled=bool(
                self.adaptive_timestep_bunch_proximity_enabled_var.get()
            ),
            adaptive_timestep_bunch_proximity_sigma_mm=float(
                self.adaptive_timestep_bunch_proximity_sigma_mm_var.get()
            ),
            adaptive_timestep_bunch_proximity_n_sigma=float(
                self.adaptive_timestep_bunch_proximity_n_sigma_var.get()
            ),
            adaptive_timestep_bunch_proximity_reduction_factor=float(
                self.adaptive_timestep_bunch_proximity_reduction_factor_var.get()
            ),
            adaptive_timestep_bunch_proximity_transition_n_sigma=float(
                self.adaptive_timestep_bunch_proximity_transition_n_sigma_var.get()
            ),
            space_charge_enabled=bool(self.space_charge_enabled_var.get()),
            space_charge_retarded=bool(self.space_charge_retarded_var.get()),
            space_charge_softening_mm=float(self.space_charge_softening_mm_var.get()),
            space_charge_bunch_sigma_mm=float(
                self.space_charge_bunch_sigma_mm_var.get()
            ),
            space_charge_min_retarded_steps=(
                int(self.space_charge_min_retarded_steps_var.get())
                if self.space_charge_min_retarded_steps_var.get().strip()
                else None
            ),
            external_field_enabled=external_field_enabled,
            external_electric_field_native=external_electric_native,
            external_electric_field_v_per_m=external_electric_si,
            **external_magnetic_options,
            external_field_x_min=external_bounds["x_min"],
            external_field_x_max=external_bounds["x_max"],
            external_field_y_min=external_bounds["y_min"],
            external_field_y_max=external_bounds["y_max"],
            external_field_z_min=external_bounds["z_min"],
            external_field_z_max=external_bounds["z_max"],
            external_field_t_min=external_bounds["t_min"],
            external_field_t_max=external_bounds["t_max"],
            radiation_reaction_mode=(
                str(self.radiation_reaction_mode_var.get())
                if hasattr(self, "radiation_reaction_mode_var")
                else getattr(
                    getattr(self, "options", None),
                    "radiation_reaction_mode",
                    "medina_lad",
                )
            ),
            cavity_exit_enabled=bool(self.cavity_exit_enabled_var.get()),
            cavity_exit_mode=str(self.cavity_exit_mode_var.get()),
            cavity_exit_length_mm=_parse_gui_optional_float_lenient(
                self.cavity_exit_length_mm_var.get()
            ),
            auto_duration_enabled=bool(self.auto_duration_enabled_var.get()),
            auto_duration_crossing_steps=int(
                self.auto_duration_crossing_steps_var.get()
            ),
            auto_duration_post_factor=float(self.auto_duration_post_factor_var.get()),
            pseudo_grid_enabled=pseudo_grid_enabled,
            pseudo_grid_active_rider_count=int(
                self.pseudo_grid_active_rider_count_var.get()
            ),
            pseudo_grid_active_driver_count=int(
                self.pseudo_grid_active_driver_count_var.get()
            ),
            pseudo_grid_field_rider_count=int(
                self.pseudo_grid_field_rider_count_var.get()
            ),
            pseudo_grid_field_driver_count=int(
                self.pseudo_grid_field_driver_count_var.get()
            ),
            pseudo_grid_field_deposition_neighbor_count=int(
                self.pseudo_grid_field_deposition_neighbor_count_var.get()
            ),
            pseudo_grid_passive_neighbor_count=int(
                self.pseudo_grid_passive_neighbor_count_var.get()
            ),
            pseudo_grid_coverage_strategy=str(
                self.pseudo_grid_coverage_strategy_var.get()
            ),
            pseudo_grid_coverage_space=str(self.pseudo_grid_coverage_space_var.get()),
            pseudo_grid_pair_reuse_window=int(
                self.pseudo_grid_pair_reuse_window_var.get()
            ),
            pseudo_grid_source_weighting_mode=str(
                self.pseudo_grid_source_weighting_mode_var.get()
            ),
            pseudo_grid_loss_tracking_enabled=bool(
                self.pseudo_grid_loss_tracking_enabled_var.get()
            ),
            pseudo_grid_causal_history_pruning_enabled=bool(
                self.pseudo_grid_causal_history_pruning_enabled_var.get()
            ),
            pseudo_grid_causal_history_safety_margin_steps=int(
                self.pseudo_grid_causal_history_safety_margin_steps_var.get()
            ),
            driver_train_enabled=driver_train_enabled,
            driver_train_bunch_count=driver_train_bunch_count,
            driver_train_z_spacing_mm=float(self.driver_train_z_spacing_mm_var.get()),
            driver_train_z_offsets_mm=driver_train_offsets,
            driver_train_prehistory_steps=int(
                self.driver_train_prehistory_steps_var.get()
            ),
            driver_train_preserve_prehistory_in_output=bool(
                self.driver_train_preserve_prehistory_var.get()
            ),
            save_log_file=bool(self.save_log_file_var.get()),
            beamline_geometry_enabled=bool(self.beamline_geometry_enabled_var.get()),
            beamline_geometry_occluders=self._collect_beamline_geometry_occluders(),
        )

    def _save_config(self) -> None:
        from tkinter import messagebox

        from .gui import _show_error_dialog

        try:
            options = self._build_options_from_ui()
        except ValueError as exc:
            _show_error_dialog(self.root, "Invalid configuration", str(exc))
            return

        filename = self.config_name_var.get().strip()
        if not filename:
            messagebox.showinfo("Save Run Config", "Enter a config name to save.")
            return

        if not filename.endswith(".json"):
            filename += ".json"

        config_dir = self.config_dir_var.get()
        os.makedirs(config_dir, exist_ok=True)

        filepath = os.path.join(config_dir, filename)
        if not self._check_override_warning(Path(filepath), "run"):
            return

        options.config_name = filename

        try:
            save_config(options, Path(filepath))
        except Exception as exc:
            _show_error_dialog(
                self.root, "Save config", f"Failed to save configuration: {exc}"
            )
            return

        self.config_name_var.set(filename)
        self.config_file_var.set(filename)
        self._refresh_config_list(selected=filename)
        self.current_config_label.config(text=filename, foreground="black")
        messagebox.showinfo("Save Run Config", f"Configuration saved as {filename}")
        self._set_status(f"Saved config: {filename}")
