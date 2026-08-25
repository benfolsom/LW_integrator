"""Configuration tests for selectable magnetic-dipole model pairs."""

from __future__ import annotations

import pytest

from core.types import DipoleSourceConfig, MagneticDipoleConfig


def test_magnetic_dipole_defaults_to_disabled_rfs_pair() -> None:
    config = MagneticDipoleConfig()

    assert config.enabled is False
    assert config.spin_model == "rfs_minimal_2021"
    assert config.stern_gerlach_model == "rfs_full_g"
    assert config.exact_retarded_backend == "python"
    assert config.source.model == "off"
    assert not hasattr(config.source, "backend")
    assert config.source.active is False


@pytest.mark.parametrize(
    ("spin_model", "stern_gerlach_model"),
    (
        ("rfs_minimal_2021", "rfs_full_g"),
        ("bmt_frenkel", "static_rest_gradient"),
    ),
)
def test_magnetic_dipole_accepts_matched_model_pairs(
    spin_model: str, stern_gerlach_model: str
) -> None:
    config = MagneticDipoleConfig(
        spin_model=spin_model,
        stern_gerlach_model=stern_gerlach_model,
    )

    assert config.spin_model == spin_model
    assert config.stern_gerlach_model == stern_gerlach_model


@pytest.mark.parametrize(
    ("spin_model", "stern_gerlach_model", "required_model"),
    (
        ("bmt_frenkel", "rfs_full_g", "rfs_minimal_2021"),
        ("rfs_minimal_2021", "static_rest_gradient", "bmt_frenkel"),
    ),
)
def test_magnetic_dipole_rejects_mismatched_model_pairs(
    spin_model: str, stern_gerlach_model: str, required_model: str
) -> None:
    with pytest.raises(ValueError, match=f"requires spin_model '{required_model}'"):
        MagneticDipoleConfig(
            spin_model=spin_model,
            stern_gerlach_model=stern_gerlach_model,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"spin_model": "unknown"}, "spin_model must be one of"),
        (
            {"stern_gerlach_model": "unknown"},
            "stern_gerlach_model must be one of",
        ),
    ),
)
def test_magnetic_dipole_rejects_unknown_models(
    overrides: dict[str, str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        MagneticDipoleConfig(**overrides)


def test_retarded_dipole_source_normalizes_alias_and_nested_mapping() -> None:
    config = MagneticDipoleConfig(
        enabled=True,
        exact_retarded_backend="numba_roots_exact_serial",
        source={
            "model": "full-retarded-point",
            "minimum_separation_mm": 3.0e-9,
            "relative_stencil_step": 5.0e-4,
        },
    )

    assert isinstance(config.source, DipoleSourceConfig)
    assert config.source.model == "covariant_retarded_point"
    assert config.exact_retarded_backend == "numba_roots_exact_serial"
    assert config.source.active is True
    assert config.source.minimum_separation_mm == 3.0e-9
    assert config.source.relative_stencil_step == 5.0e-4


def test_magnetic_dipole_accepts_full_strict_exact_retarded_backend() -> None:
    config = MagneticDipoleConfig(exact_retarded_backend="numba_full_strict_serial")

    assert config.exact_retarded_backend == "numba_full_strict_serial"


def test_magnetic_dipole_rejects_unknown_exact_retarded_backend() -> None:
    with pytest.raises(ValueError, match="exact_retarded_backend"):
        MagneticDipoleConfig(exact_retarded_backend="auto")


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"model": "quasistatic"}, "model"),
        ({"minimum_separation_mm": 0.0}, "minimum_separation"),
        ({"relative_stencil_step": 0.05}, "relative_stencil_step"),
        ({"minimum_stencil_step_mm": float("nan")}, "minimum_stencil"),
        ({"root_tolerance_mm": -1.0}, "root_tolerance"),
        ({"max_root_iterations": 0}, "max_root_iterations"),
    ),
)
def test_invalid_retarded_dipole_source_config_fails_explicitly(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        DipoleSourceConfig(**overrides)
