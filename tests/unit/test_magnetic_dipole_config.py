"""Configuration tests for selectable magnetic-dipole model pairs."""

from __future__ import annotations

import pytest

from core.types import MagneticDipoleConfig


def test_magnetic_dipole_defaults_to_disabled_rfs_pair() -> None:
    config = MagneticDipoleConfig()

    assert config.enabled is False
    assert config.spin_model == "rfs_minimal_2021"
    assert config.stern_gerlach_model == "rfs_full_g"


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
