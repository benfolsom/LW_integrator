from __future__ import annotations

import numpy as np
import pytest

from core.constants import ELEMENTARY_CHARGE
from core.macroparticle_smearing import fixed_cloud_offsets, smear_source_samples
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
        macro_population=np.array([charge_multiplier, 1.0]),
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
    initial_samples = _samples()
    initial = {a: getattr(initial_samples, a).copy() for a in "xyz"}
    initial.update(
        q=initial_samples.charge, macro_population=initial_samples.macro_population
    )
    offsets = fixed_cloud_offsets([initial], config)
    first, first_nhat = smear_source_samples(
        samples=_samples(),
        observer_position=(0.0, 0.0, 10.0),
        config=config,
        step_index=7,
        fixed_offsets=offsets,
    )
    second, second_nhat = smear_source_samples(
        samples=_samples(),
        observer_position=(0.0, 0.0, 10.0),
        config=config,
        step_index=7,
        fixed_offsets=offsets,
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


def test_smearing_preserves_explicit_macro_population_metadata() -> None:
    config = MacroparticleSmearingConfig(
        enabled=True,
        subcharge_count=4,
        seed=11,
        position_sigma_mm=0.5,
    )

    smeared, _ = smear_source_samples(
        samples=_samples(charge_multiplier=100.0),
        observer_position=(0.0, 0.0, 10.0),
        config=config,
        step_index=0,
    )

    np.testing.assert_allclose(
        smeared.macro_population,
        np.repeat(np.array([100.0, 1.0]), 4),
    )


def test_fixed_cloud_ignores_event_spacing_masks_and_checkpoint_reconstruction():
    import copy
    from core.macroparticle_smearing import fixed_cloud_offsets, _initial_cloud

    samples = _samples()
    initial = {a: getattr(samples, a).copy() for a in "xyz"}
    initial.update(
        q=samples.charge.copy(), macro_population=samples.macro_population.copy()
    )
    config = MacroparticleSmearingConfig(
        enabled=True,
        subcharge_count=4,
        seed=42,
        position_sigma_mm=10.0,
        longitudinal_sigma_mm=10.0,
    )
    offsets = fixed_cloud_offsets([initial], config)
    moved = copy.deepcopy(samples)
    moved.x *= 100
    moved.valid_mask[1] = False
    smeared, _ = smear_source_samples(
        samples=moved,
        observer_position=(3.0, 4.0, 5.0),
        config=config,
        step_index=91,
        fixed_offsets=offsets,
    )
    actual = np.column_stack([getattr(smeared, a) for a in "xyz"]) - np.repeat(
        np.column_stack([getattr(moved, a) for a in "xyz"]), 4, axis=0
    )
    np.testing.assert_allclose(actual, offsets.reshape(-1, 3), rtol=0.0, atol=6e-14)
    # Evicting a cache or loading a detached initial checkpoint row changes
    # neither the cloud nor random draws at later accepted steps.
    _initial_cloud.cache_clear()
    reconstructed = fixed_cloud_offsets([copy.deepcopy(initial)], config)
    np.testing.assert_array_equal(reconstructed, offsets)
    assert not reconstructed.flags.writeable


def test_default_cloud_cannot_silently_resize_after_initialization():
    with pytest.raises(ValueError, match="retained initial cloud"):
        smear_source_samples(
            samples=_samples(),
            observer_position=(0.0, 0.0, 10.0),
            config=MacroparticleSmearingConfig(enabled=True),
            step_index=1,
        )
