from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from core.species import SPECIES, get_species, list_species, resolve_species


def test_common_species_registry_has_signed_free_particle_moments() -> None:
    expected_signs = {
        "electron": -1,
        "positron": 1,
        "proton": 1,
        "antiproton": -1,
        "neutron": -1,
        "deuteron": 1,
        "triton": 1,
        "helion": -1,
    }

    for name, expected_sign in expected_signs.items():
        moment = get_species(name).magnetic_moment_j_t
        assert moment is not None
        assert (moment > 0.0) - (moment < 0.0) == expected_sign


def test_conjugate_presets_reverse_charge_and_moment() -> None:
    for particle_name, antiparticle_name in (
        ("electron", "positron"),
        ("proton", "antiproton"),
    ):
        particle = get_species(particle_name)
        antiparticle = get_species(antiparticle_name)

        assert particle.conjugate_name == antiparticle.name
        assert antiparticle.conjugate_name == particle.name
        assert antiparticle.mass_amu == pytest.approx(particle.mass_amu)
        assert antiparticle.charge_e == -particle.charge_e
        assert antiparticle.magnetic_moment_j_t == pytest.approx(
            -particle.magnetic_moment_j_t, rel=5.0e-10
        )


def test_spin_quantum_number_is_not_inferred_from_moment() -> None:
    assert get_species("electron").spin_quantum_number == 0.5
    assert get_species("deuteron").spin_quantum_number == 1.0
    assert get_species("alpha").spin_quantum_number == 0.0
    assert get_species("alpha").magnetic_moment_j_t == 0.0


def test_neutral_neutron_keeps_nonzero_signed_moment() -> None:
    neutron = get_species("neutron")

    assert neutron.charge_e == 0
    assert neutron.charge_coulomb == 0.0
    assert neutron.magnetic_moment_j_t is not None
    assert neutron.magnetic_moment_j_t < 0.0


def test_h_minus_is_explicitly_unsupported_not_silently_zero() -> None:
    h_minus = get_species("H-")

    assert h_minus.name == "h_minus"
    assert h_minus.charge_e == -1
    assert h_minus.magnetic_moment_j_t is None
    assert not h_minus.has_supported_magnetic_moment
    assert "unsupported" in h_minus.moment_status
    assert "binding energy omitted" in h_minus.note


def test_registry_and_records_are_immutable() -> None:
    electron = get_species("e-")

    with pytest.raises(TypeError):
        SPECIES["test"] = electron
    with pytest.raises(FrozenInstanceError):
        electron.charge_e = 1


def test_aliases_and_supported_filter_are_deterministic() -> None:
    assert get_species("3He++") is get_species("helion")
    assert get_species("anti-proton") is get_species("antiproton")
    assert resolve_species("e-") is get_species("electron")
    assert all(
        item.has_supported_magnetic_moment
        for item in list_species(require_supported_moment=True)
    )
    assert get_species("h_minus") not in list_species(require_supported_moment=True)


def test_species_exposes_signed_gyromagnetic_ratio() -> None:
    assert get_species("electron").gyromagnetic_ratio_rad_s_t < 0.0
    assert get_species("proton").gyromagnetic_ratio_rad_s_t > 0.0
    assert get_species("alpha").gyromagnetic_ratio_rad_s_t == 0.0
    with pytest.raises(ValueError, match="no supported magnetic moment"):
        get_species("h_minus").gyromagnetic_ratio_rad_s_t


def test_unknown_species_reports_the_finite_choices() -> None:
    with pytest.raises(KeyError, match="available presets"):
        get_species("invented_particle")
