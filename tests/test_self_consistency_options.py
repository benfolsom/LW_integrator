"""Self-consistency/chrono option parsing, defaults, validation and warnings.

See docs/self_consistency_option_audit.md (issues SC-1 to SC-17).
"""

from __future__ import annotations

import copy
import inspect
import json
import warnings
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from core.constants import C_MMNS, PROTON_MASS_AMU
from core.self_consistency import (
    ChronoHighPrecisionDeprecationWarning,
    SelfConsistencyConfig,
    SelfConsistencyOptionWarning,
    self_consistency_runtime_warnings,
)
from core.types import (
    ChronoMatchingMode,
    GammaReconciliationMethod,
    SimulationType,
    SpaceChargeConfig,
    StartupMode,
)
from lw_integrator import cli
from lw_integrator.sweep_runner import _convert_json_config_to_dataclass
from lw_integrator.testbed_runner import (
    SimulationOptions,
    build_self_consistency_config,
)
from optimization.config import OptimizationConfig
from optimization.plugin_persistence_helpers import apply_persisted_config_overrides

REPO_ROOT = Path(__file__).resolve().parents[1]

# A hand-edited saved config: the canonical keys were changed, the legacy
# aliases written alongside them were not.
CONFLICTING_CHRONO_PAYLOAD = {
    "self_consistency_enabled": True,
    "chrono_interpolate": False,
    "self_consistency_chrono_interpolate": True,
    "chrono_tolerance": 1e-3,
    "self_consistency_chrono_tolerance": 5e-3,
    "chrono_high_precision": False,
    "self_consistency_chrono_high_precision": True,
    "chrono_adaptive_tolerance": False,
    "self_consistency_chrono_adaptive_tolerance": True,
    "chrono_matching_mode": "FAST",
    "self_consistency_chrono_matching_mode": "AVERAGED",
}

EXPECTED_CANONICAL = {
    "chrono_interpolate": False,
    "chrono_tolerance": 1e-3,
    "chrono_high_precision": False,
    "chrono_adaptive_tolerance": False,
    "chrono_matching_mode": "FAST",
}


def _chrono_values(obj) -> dict:
    return {key: getattr(obj, key) for key in EXPECTED_CANONICAL}


def test_cli_and_testbed_prefer_canonical_chrono_keys():
    cli_config = cli._build_self_consistency_config(CONFLICTING_CHRONO_PAYLOAD)
    testbed_config = build_self_consistency_config(
        SimulationOptions.from_dict(dict(CONFLICTING_CHRONO_PAYLOAD))
    )

    assert _chrono_values(cli_config) == EXPECTED_CANONICAL
    assert _chrono_values(testbed_config) == EXPECTED_CANONICAL


def test_sweep_loader_prefers_canonical_chrono_keys_over_legacy_aliases():
    converted = _convert_json_config_to_dataclass(dict(CONFLICTING_CHRONO_PAYLOAD))
    valid = {field.name for field in fields(OptimizationConfig)}
    config = OptimizationConfig(**{k: v for k, v in converted.items() if k in valid})

    assert _chrono_values(config) == EXPECTED_CANONICAL
    for key, value in EXPECTED_CANONICAL.items():
        assert getattr(config, f"self_consistency_{key}") == value


def test_sweep_loader_still_accepts_legacy_only_chrono_keys():
    converted = _convert_json_config_to_dataclass(
        {
            "self_consistency_chrono_interpolate": True,
            "self_consistency_chrono_tolerance": 5e-3,
        }
    )

    assert converted["chrono_interpolate"] is True
    assert converted["chrono_tolerance"] == pytest.approx(5e-3)


def test_persisted_plugin_config_prefers_canonical_chrono_keys():
    config = apply_persisted_config_overrides(
        OptimizationConfig(), dict(CONFLICTING_CHRONO_PAYLOAD)
    )

    assert _chrono_values(config) == EXPECTED_CANONICAL
    for key, value in EXPECTED_CANONICAL.items():
        assert getattr(config, f"self_consistency_{key}") == value


def test_persisted_plugin_config_uses_legacy_alias_when_canonical_missing():
    config = apply_persisted_config_overrides(
        OptimizationConfig(),
        {
            "self_consistency_chrono_interpolate": True,
            "self_consistency_chrono_tolerance": 5e-3,
        },
    )

    assert config.chrono_interpolate is True
    assert config.self_consistency_chrono_interpolate is True
    assert config.chrono_tolerance == pytest.approx(5e-3)


def test_testbed_rejects_unknown_gamma_reconciliation_method():
    options = SimulationOptions.from_dict(
        {
            "self_consistency_enabled": True,
            "self_consistency_gamma_reconciliation_method": "NONE",
        }
    )

    with pytest.raises(ValueError, match="gamma_reconciliation_method"):
        build_self_consistency_config(options)


@pytest.mark.parametrize("name", [method.name for method in GammaReconciliationMethod])
def test_testbed_accepts_every_gamma_reconciliation_method(name: str):
    options = SimulationOptions.from_dict(
        {
            "self_consistency_enabled": True,
            "self_consistency_gamma_reconciliation_method": name.lower(),
        }
    )

    config = build_self_consistency_config(options)

    assert config is not None
    assert config.gamma_reconciliation_method is GammaReconciliationMethod[name]


# ---------------------------------------------------------------------------
# SC-1: one set of missing-key defaults everywhere
# ---------------------------------------------------------------------------

EXPECTED_SC_DEFAULTS = {
    "enabled": True,
    "convergence_mode": "fixed_geometry",
    "max_iterations": 2,
    "verbosity": 0,
    "gamma_reconciliation_method": GammaReconciliationMethod.DISABLED,
    "chrono_interpolate": False,
    "chrono_high_precision": False,
    "chrono_adaptive_tolerance": False,
    "chrono_matching_mode": "FAST",
}


def _sc_defaults(config) -> dict:
    return {key: getattr(config, key) for key in EXPECTED_SC_DEFAULTS}


def _option_defaults(options) -> dict:
    return {
        "enabled": options.self_consistency_enabled,
        "convergence_mode": options.self_consistency_convergence_mode,
        "max_iterations": options.self_consistency_max_iterations,
        "verbosity": options.self_consistency_verbosity,
        "gamma_reconciliation_method": GammaReconciliationMethod[
            options.self_consistency_gamma_reconciliation_method.upper()
        ],
        "chrono_interpolate": options.chrono_interpolate,
        "chrono_high_precision": options.chrono_high_precision,
        "chrono_adaptive_tolerance": options.chrono_adaptive_tolerance,
        "chrono_matching_mode": options.chrono_matching_mode,
    }


def test_core_and_standard_preset_use_unified_defaults():
    assert _sc_defaults(SelfConsistencyConfig()) == EXPECTED_SC_DEFAULTS
    assert _sc_defaults(SelfConsistencyConfig.standard()) == EXPECTED_SC_DEFAULTS
    # Named non-default presets are unchanged.
    assert SelfConsistencyConfig.aggressive().max_iterations == 20
    assert SelfConsistencyConfig.variable_geometry().max_iterations == 20


def test_keyless_cli_config_enables_self_consistency_with_unified_defaults():
    config = cli._build_self_consistency_config({})

    assert config is not None
    assert _sc_defaults(config) == EXPECTED_SC_DEFAULTS


def test_testbed_gui_and_sweep_entry_points_use_unified_defaults():
    assert _option_defaults(SimulationOptions()) == EXPECTED_SC_DEFAULTS
    assert _option_defaults(SimulationOptions.from_dict({})) == EXPECTED_SC_DEFAULTS
    assert (
        _sc_defaults(build_self_consistency_config(SimulationOptions.from_dict({})))
        == EXPECTED_SC_DEFAULTS
    )
    for config in (
        OptimizationConfig(),
        apply_persisted_config_overrides(OptimizationConfig(), {}),
    ):
        assert config.self_consistency_enabled is True
        assert config.self_consistency_convergence_mode == "fixed_geometry"
        assert config.self_consistency_max_iterations == 2
        assert config.self_consistency_verbosity == 0
        assert config.self_consistency_gamma_reconciliation_method == "DISABLED"
        assert config.chrono_matching_mode == "FAST"


# ---------------------------------------------------------------------------
# SC-2 / SC-14: validation when the config is built
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("iterations", [0, 1])
def test_enabled_self_consistency_rejects_fewer_than_two_iterations(iterations):
    with pytest.raises(ValueError, match="at least 2"):
        SelfConsistencyConfig(enabled=True, max_iterations=iterations)
    with pytest.raises(cli.SimulationConfigError, match="at least 2"):
        cli._build_self_consistency_config(
            {"self_consistency_max_iterations": iterations}
        )
    options = SimulationOptions.from_dict(
        {"self_consistency_max_iterations": iterations}
    )
    with pytest.raises(ValueError, match="at least 2"):
        build_self_consistency_config(options)


def test_disabled_self_consistency_does_not_check_iteration_count():
    config = SelfConsistencyConfig(enabled=False, max_iterations=1)

    assert config.enabled is False


def test_unknown_convergence_mode_is_rejected_after_alias_canonicalisation():
    assert (
        SelfConsistencyConfig(convergence_mode="full_iteration").convergence_mode
        == "variable_geometry"
    )
    with pytest.raises(ValueError, match="dual_independent"):
        SelfConsistencyConfig(convergence_mode="dual_independent")
    with pytest.raises(cli.SimulationConfigError, match="convergence_mode"):
        cli._build_self_consistency_config(
            {"self_consistency_convergence_mode": "dual_independent"}
        )


# ---------------------------------------------------------------------------
# SC-6 / SC-7 / SC-8 / SC-9: run-start warnings (no rejection)
# ---------------------------------------------------------------------------


def _warning_texts(config, **kwargs):
    return [
        (category, text)
        for category, text in self_consistency_runtime_warnings(config, **kwargs)
    ]


def test_default_config_produces_no_runtime_warnings():
    assert (
        _warning_texts(
            SelfConsistencyConfig(),
            radiation_reaction_mode="medina_lad",
            exact_path=True,
            chrono_mode=ChronoMatchingMode.FAST,
        )
        == []
    )


@pytest.mark.parametrize(
    ("config", "kwargs", "fragment"),
    [
        (
            SelfConsistencyConfig(
                gamma_reconciliation_method=GammaReconciliationMethod.FIXED_WEIGHTED
            ),
            {"radiation_reaction_mode": "medina_lad", "exact_path": False},
            "medina_lad",
        ),
        (
            SelfConsistencyConfig(
                gamma_reconciliation_method=GammaReconciliationMethod.ADAPTIVE_WEIGHTED
            ),
            {"radiation_reaction_mode": "off", "exact_path": True},
            "exact RFS/dipole",
        ),
        (
            SelfConsistencyConfig(
                chrono_interpolate=True, chrono_adaptive_tolerance=True
            ),
            {"radiation_reaction_mode": "off", "exact_path": True},
            "exact RFS/dipole",
        ),
        (
            SelfConsistencyConfig(chrono_adaptive_tolerance=True),
            {"radiation_reaction_mode": "off", "exact_path": False},
            "without chrono_interpolate",
        ),
    ],
)
def test_inapplicable_options_produce_one_option_warning(config, kwargs, fragment):
    messages = _warning_texts(config, **kwargs)

    assert len(messages) == 1
    category, text = messages[0]
    assert category is SelfConsistencyOptionWarning
    assert fragment in text


def test_gamma_reconciliation_is_not_flagged_where_it_can_act():
    config = SelfConsistencyConfig(
        gamma_reconciliation_method=GammaReconciliationMethod.FIXED_WEIGHTED
    )

    assert _warning_texts(config, radiation_reaction_mode="off", exact_path=False) == []


def test_high_precision_emits_deprecation_warning():
    messages = _warning_texts(
        SelfConsistencyConfig(chrono_interpolate=True, chrono_high_precision=True),
        radiation_reaction_mode="off",
        exact_path=False,
    )

    assert [category for category, _ in messages] == [
        ChronoHighPrecisionDeprecationWarning
    ]
    assert issubclass(ChronoHighPrecisionDeprecationWarning, FutureWarning)


def _small_b2b_run(self_consistency, *, radiation_reaction_mode="diagnostic_only"):
    from core.integration_runner import retarded_integrator
    from input_output.bunch_initialization import create_bunch_from_params
    from optimization.single_integration_helpers import calculate_rider_starting_pz

    pz = calculate_rider_starting_pz(
        0.0840444095085845, PROTON_MASS_AMU, SimulationType.BUNCH_TO_BUNCH
    )
    common = dict(
        transv_mom=0.0,
        stripped_ions=1.0,
        m_particle=PROTON_MASS_AMU,
        transv_dist=2.0,
        pcount=4,
        transverse_geometry="gaussian",
    )
    rider, _ = create_bunch_from_params(
        starting_distance=0.0,
        starting_Pz=pz,
        charge_sign=-1.0,
        seed=20260527,
        charge_multiplier=1.4e8,
        **common,
    )
    driver, _ = create_bunch_from_params(
        starting_distance=12.0,
        starting_Pz=-pz,
        charge_sign=1.0,
        seed=20260528,
        **common,
    )
    gamma = float(np.mean(rider["gamma"]))
    beta = abs(float(np.mean(rider["bz"])))
    steps = 20
    trajectory, *_ = retarded_integrator(
        steps=steps,
        h_step=10.0 / (gamma * beta * C_MMNS * (steps - 1)),
        wall_z=1e5,
        aperture_radius=5000.0,
        sim_type=SimulationType.BUNCH_TO_BUNCH,
        init_rider=copy.deepcopy(rider),
        init_driver=copy.deepcopy(driver),
        mean=1e5,
        cav_spacing=1e5,
        z_cutoff=1e5,
        self_consistency=self_consistency,
        chrono_mode=ChronoMatchingMode.FAST,
        startup_mode=StartupMode.COLD_START,
        space_charge=SpaceChargeConfig(
            enabled=True,
            retarded=True,
            softening_mm=0.1,
            bunch_sigma_mm=2.0,
            min_retarded_steps=0,
        ),
        radiation_reaction_mode=radiation_reaction_mode,
        use_numba=True,
    )
    return np.concatenate(
        [np.asarray(trajectory[-1][key], dtype=float) for key in ("x", "Pz", "Pt")]
    )


def test_run_start_warns_once_for_gamma_reconciliation_with_medina():
    config = SelfConsistencyConfig(
        gamma_reconciliation_method=GammaReconciliationMethod.FIXED_WEIGHTED
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _small_b2b_run(config, radiation_reaction_mode="medina_lad")

    option_warnings = [
        w for w in caught if issubclass(w.category, SelfConsistencyOptionWarning)
    ]
    assert len(option_warnings) == 1
    assert "medina_lad" in str(option_warnings[0].message)


def test_high_precision_has_no_effect_in_maintained_runner():
    base = SelfConsistencyConfig(
        chrono_interpolate=True, chrono_adaptive_tolerance=True
    )
    high = SelfConsistencyConfig(
        chrono_interpolate=True,
        chrono_adaptive_tolerance=True,
        chrono_high_precision=True,
    )
    reference = _small_b2b_run(base)
    with pytest.warns(ChronoHighPrecisionDeprecationWarning):
        deprecated = _small_b2b_run(high)

    np.testing.assert_array_equal(deprecated, reference)


# ---------------------------------------------------------------------------
# SC-10: FAST is the API default everywhere
# ---------------------------------------------------------------------------


def test_api_chrono_mode_defaults_are_fast():
    from core.distances import chrono_match_indices, chrono_match_indices_soa
    from core.equations import retarded_equations_of_motion
    from core.integration_runner import retarded_integrator

    assert (
        inspect.signature(retarded_integrator).parameters["chrono_mode"].default
        is ChronoMatchingMode.FAST
    )
    assert (
        inspect.signature(retarded_equations_of_motion)
        .parameters["chrono_mode"]
        .default
        is ChronoMatchingMode.FAST
    )
    for function in (chrono_match_indices, chrono_match_indices_soa):
        assert (
            inspect.signature(function).parameters["mode"].default
            is ChronoMatchingMode.FAST
        )


def test_trajectory_integrator_without_config_uses_fast(monkeypatch):
    import core.trajectory_integrator as trajectory_integrator
    from tests.unit.test_trajectory_integrator_helpers import (
        _make_single_particle_state,
    )

    captured = {}

    def fake_retarded_integrator(**kwargs):
        captured.update(kwargs)
        return [kwargs["init_rider"]], [kwargs["init_driver"]]

    monkeypatch.setattr(
        trajectory_integrator, "retarded_integrator", fake_retarded_integrator
    )
    state = _make_single_particle_state(t=0.0, z=0.0, charge=1.0)
    trajectory_integrator.LienardWiechertIntegrator(None).integrate_retarded_fields(
        0, 1, 1e-3, 0.0, 1.0, SimulationType.BUNCH_TO_BUNCH, state, state, 0.0, 0.0
    )

    assert captured["chrono_mode"] is ChronoMatchingMode.FAST


# ---------------------------------------------------------------------------
# SC-11 / SC-12: CLI passthrough and retardation-mode alias
# ---------------------------------------------------------------------------


def _cli_request(tmp_path, payload):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))
    return cli.build_request(cli.parse_args(["--config", str(path)]))


def test_cli_passes_through_adaptive_gamma_parameters(tmp_path):
    request = _cli_request(
        tmp_path,
        {
            "self_consistency_gamma_reconciliation_method": "ADAPTIVE_WEIGHTED",
            "self_consistency_gamma_reconciliation_low_beta_threshold": 0.5,
            "self_consistency_gamma_reconciliation_high_beta_threshold": 0.95,
            "self_consistency_gamma_reconciliation_low_beta_weight": 0.1,
            "self_consistency_gamma_reconciliation_high_beta_weight": 0.3,
            "self_consistency_gamma_reconciliation_mid_beta_weight": 0.4,
        },
    )
    config = request.self_consistency

    assert config.gamma_reconciliation_low_beta_threshold == pytest.approx(0.5)
    assert config.gamma_reconciliation_high_beta_threshold == pytest.approx(0.95)
    assert config.gamma_reconciliation_low_beta_weight == pytest.approx(0.1)
    assert config.gamma_reconciliation_high_beta_weight == pytest.approx(0.3)
    assert config.gamma_reconciliation_mid_beta_weight == pytest.approx(0.4)


@pytest.mark.parametrize(
    "payload",
    [
        {"chrono_matching_mode": "AVERAGED"},
        {"self_consistency_chrono_matching_mode": "AVERAGED"},
        {"chrono_mode": "averaged", "chrono_matching_mode": "AVERAGED"},
    ],
)
def test_cli_accepts_chrono_matching_mode_alias(tmp_path, payload):
    request = _cli_request(tmp_path, payload)

    assert request.config.chrono_mode is ChronoMatchingMode.AVERAGED
    assert request.self_consistency.chrono_matching_mode == "AVERAGED"


def test_cli_rejects_conflicting_retardation_modes(tmp_path):
    with pytest.raises(cli.SimulationConfigError, match="Conflicting"):
        _cli_request(
            tmp_path, {"chrono_mode": "fast", "chrono_matching_mode": "AVERAGED"}
        )


def test_gui_carries_loaded_retardation_mode_through():
    from lw_integrator.gui_config_mixins import IntegratorGUIConfigMixin

    loaded = SimpleNamespace(_loaded_chrono_matching_mode="AVERAGED")
    fresh = SimpleNamespace(options=SimulationOptions())

    assert IntegratorGUIConfigMixin._current_chrono_matching_mode(loaded) == "AVERAGED"
    assert IntegratorGUIConfigMixin._current_chrono_matching_mode(fresh) == "FAST"


# ---------------------------------------------------------------------------
# SC-13: strict booleans
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key",
    [
        "self_consistency_enabled",
        "chrono_interpolate",
        "self_consistency_chrono_interpolate",
        "chrono_high_precision",
        "chrono_adaptive_tolerance",
    ],
)
def test_string_booleans_are_rejected(key):
    with pytest.raises(cli.SimulationConfigError, match="JSON boolean"):
        cli._build_self_consistency_config({key: "false"})
    with pytest.raises(ValueError, match="JSON boolean"):
        SimulationOptions.from_dict({key: "false"})


def test_json_booleans_are_accepted():
    config = cli._build_self_consistency_config(
        {"self_consistency_enabled": False, "chrono_interpolate": True}
    )

    assert config.enabled is False
    assert config.chrono_interpolate is True


# ---------------------------------------------------------------------------
# SC-15: dead sweep stability dialog removed
# ---------------------------------------------------------------------------


def test_dead_stability_dialog_is_removed():
    pytest.importorskip("tkinter")
    import optimization.plugin_control_mixins as plugin_control_mixins

    assert not hasattr(
        plugin_control_mixins.OptimizationPluginControlMixin,
        "_confirm_stability_options",
    )
    assert not hasattr(plugin_control_mixins, "_stability_dialog_logging_defaults")


# ---------------------------------------------------------------------------
# SC-17: shipped example follows the AGENTS defaults
# ---------------------------------------------------------------------------


def test_shipped_b2b_example_uses_maintained_defaults():
    path = (
        REPO_ROOT
        / "configs"
        / "run_configs"
        / "example_b2b_counter_propagating_proton_bunches.json"
    )
    payload = json.loads(path.read_text())
    options = SimulationOptions.from_dict(payload)
    config = build_self_consistency_config(options)

    assert payload["radiation_reaction_mode"] == "medina_lad"
    assert options.radiation_reaction_mode == "medina_lad"
    assert _sc_defaults(config) == EXPECTED_SC_DEFAULTS
