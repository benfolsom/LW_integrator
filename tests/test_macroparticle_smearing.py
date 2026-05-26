from __future__ import annotations

import numpy as np
import pytest

from core.constants import ELEMENTARY_CHARGE
from core.macroparticle_smearing import effective_observer_charge, smear_source_samples
from core.types import MacroparticleSmearingConfig
from core.vectorized_interactions import ExternalSampleBatch


def _samples(charge_multiplier: float = 100.0) -> ExternalSampleBatch:
    return ExternalSampleBatch(
        charge=np.array([ELEMENTARY_CHARGE * charge_multiplier, ELEMENTARY_CHARGE]),
        gamma=np.array([1.0, 1.0]),
        bx=np.array([0.0, 0.0]),
        by=np.array([0.0, 0.0]),
        bz=np.array([0.0, 0.0]),
        bdotx=np.array([0.0, 0.0]),
        bdoty=np.array([0.0, 0.0]),
        bdotz=np.array([0.0, 0.0]),
        valid_mask=np.array([True, True]),
        x=np.array([0.0, 10.0]),
        y=np.array([0.0, 0.0]),
        z=np.array([0.0, 0.0]),
        m=np.array([1.0, 1.0]),
    )


def test_smearing_disabled_is_noop() -> None:
    samples = _samples()
    result, nhat = smear_source_samples(
        samples=samples,
        observer_position=(0.0, 0.0, 10.0),
        config=MacroparticleSmearingConfig(enabled=False),
        step_index=1,
    )

    assert result is samples
    assert nhat == {}


def test_smearing_conserves_charge_and_is_deterministic() -> None:
    config = MacroparticleSmearingConfig(
        enabled=True,
        subcharge_count=4,
        seed=99,
        position_sigma_mm=1.0,
    )
    first, first_nhat = smear_source_samples(
        samples=_samples(),
        observer_position=(0.0, 0.0, 10.0),
        config=config,
        step_index=7,
    )
    second, second_nhat = smear_source_samples(
        samples=_samples(),
        observer_position=(0.0, 0.0, 10.0),
        config=config,
        step_index=7,
    )

    assert first.charge.size == 8
    assert np.sum(first.charge) == pytest.approx(np.sum(_samples().charge))
    np.testing.assert_allclose(first.x, second.x)
    np.testing.assert_allclose(first.y, second.y)
    np.testing.assert_allclose(first.z, second.z)
    np.testing.assert_allclose(first_nhat["R"], second_nhat["R"])


def test_smearing_stays_within_half_spacing_cap() -> None:
    config = MacroparticleSmearingConfig(
        enabled=True,
        subcharge_count=8,
        seed=123,
        position_sigma_mm=10.0,
        longitudinal_sigma_mm=10.0,
    )
    smeared, _ = smear_source_samples(
        samples=_samples(),
        observer_position=(0.0, 0.0, 10.0),
        config=config,
        step_index=0,
    )

    base_positions = np.repeat(np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]), 8, axis=0)
    displacements = np.column_stack((smeared.x, smeared.y, smeared.z)) - base_positions
    assert np.max(np.linalg.norm(displacements, axis=1)) <= 5.0 + 1e-12


def test_observer_charge_normalization_uses_unit_particle_equivalent_charge() -> None:
    macro_charge = 250.0 * ELEMENTARY_CHARGE

    assert effective_observer_charge(macro_charge) == pytest.approx(ELEMENTARY_CHARGE)
    assert effective_observer_charge(-macro_charge) == pytest.approx(-ELEMENTARY_CHARGE)
    assert effective_observer_charge(0.5 * ELEMENTARY_CHARGE) == pytest.approx(
        0.5 * ELEMENTARY_CHARGE
    )
