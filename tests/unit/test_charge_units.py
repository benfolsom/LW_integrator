from __future__ import annotations

import numpy as np
import pytest

from core.constants import (
    C_MMNS,
    ELECTRON_MASS_AMU,
    ELEMENTARY_CHARGE,
    ELEMENTARY_CHARGE_STATC,
    PROTON_MASS_AMU,
    STATCOULOMB_TO_NATIVE_CHARGE,
)
from core.particle_initialization import create_particle_state
from input_output.bunch_initialization import (
    ELEMENTARY_CHARGE_GU,
    create_bunch_from_energy,
    create_bunch_from_params,
)


def _char_time(charge: float, mass_amu: float) -> float:
    return (2.0 / 3.0) * charge**2 / (mass_amu * C_MMNS**3)


def test_cgs_charge_constant_is_conversion_only() -> None:
    converted = ELEMENTARY_CHARGE_STATC * STATCOULOMB_TO_NATIVE_CHARGE

    assert ELEMENTARY_CHARGE_GU == pytest.approx(ELEMENTARY_CHARGE_STATC)
    assert converted == pytest.approx(ELEMENTARY_CHARGE, rel=3e-5)
    assert ELEMENTARY_CHARGE_GU != pytest.approx(ELEMENTARY_CHARGE)


def test_energy_initializer_uses_native_charge_units() -> None:
    state, _ = create_bunch_from_energy(
        kinetic_energy_mev=0.0,
        mass_amu=ELECTRON_MASS_AMU,
        charge_sign=-1.0,
        particle_count=1,
    )

    expected_charge = -ELEMENTARY_CHARGE
    assert state["q"][0] == pytest.approx(expected_charge)
    assert state["char_time"][0] == pytest.approx(
        _char_time(expected_charge, ELECTRON_MASS_AMU)
    )


def test_parameter_initializers_agree_on_charge_units() -> None:
    stripped_ions = 2.0
    core_state, _ = create_particle_state(
        starting_distance=0.0,
        transv_momentum=0.0,
        starting_pz=0.0,
        stripped_ions=stripped_ions,
        particle_mass_amu=PROTON_MASS_AMU,
        transv_distance=0.0,
        particle_count=1,
        charge_sign=1.0,
    )
    io_state, _ = create_bunch_from_params(
        starting_distance=0.0,
        transv_mom=0.0,
        starting_Pz=0.0,
        stripped_ions=stripped_ions,
        m_particle=PROTON_MASS_AMU,
        pcount=1,
        charge_sign=1.0,
    )

    assert io_state["q"][0] == pytest.approx(core_state["q"][0])
    assert io_state["char_time"][0] == pytest.approx(core_state["char_time"][0])


def test_macroparticle_charge_split_preserves_multiply_charged_observer() -> None:
    stripped_ions = 54.0
    macro_population = 100.0

    state, _ = create_particle_state(
        starting_distance=0.0,
        transv_momentum=0.0,
        starting_pz=0.0,
        stripped_ions=stripped_ions,
        particle_mass_amu=197.0,
        transv_distance=0.0,
        particle_count=2,
        charge_sign=1.0,
        charge_multiplier=macro_population,
    )

    expected_q_observer = stripped_ions * ELEMENTARY_CHARGE
    expected_q_source = macro_population * expected_q_observer

    np.testing.assert_allclose(state["q_species"], expected_q_observer)
    np.testing.assert_allclose(state["q_observer"], expected_q_observer)
    np.testing.assert_allclose(state["q_source"], expected_q_source)
    np.testing.assert_allclose(state["q"], expected_q_source)
    np.testing.assert_allclose(state["macro_population"], macro_population)
    np.testing.assert_allclose(state["m_species"], 197.0)
    np.testing.assert_allclose(state["m"], 197.0)
    np.testing.assert_allclose(
        state["char_time"], _char_time(expected_q_observer, 197.0)
    )


def test_singly_charged_macroparticle_charge_split_is_backward_compatible() -> None:
    state, _ = create_particle_state(
        starting_distance=0.0,
        transv_momentum=0.0,
        starting_pz=0.0,
        stripped_ions=1.0,
        particle_mass_amu=PROTON_MASS_AMU,
        transv_distance=0.0,
        particle_count=1,
        charge_sign=-1.0,
        charge_multiplier=100.0,
    )

    expected_q_observer = -ELEMENTARY_CHARGE
    expected_q_source = 100.0 * expected_q_observer

    assert state["q_observer"][0] == pytest.approx(expected_q_observer)
    assert state["q_source"][0] == pytest.approx(expected_q_source)
    assert state["q"][0] == pytest.approx(expected_q_source)
    assert state["char_time"][0] == pytest.approx(
        _char_time(expected_q_observer, PROTON_MASS_AMU)
    )
