"""A short curved step must not corrupt a long initial coasting history."""

import copy

import numpy as np
import pytest

from core.constants import C_MMNS as c
from core.integration_runner import (
    _build_inertial_coasting_history,
    _causal_c5_inertial_time_offsets_ns,
)
from core.retarded_fields import (
    _prepare_history_uncached,
    evaluate_retarded_charge_field_native,
    ObserverEvent,
)
from tests.unit.test_inertial_prehistory import (
    _state,
    _species_state,
    _run_simple_inertial,
)
from core.external_fields import magnetic_field_tesla_to_native
from core.types import (
    AdaptivePairReturnConfig,
    CheckpointConfig,
    ExternalFieldConfig,
    MagneticDipoleConfig,
    MagneticDipoleParticleConfig,
    ParticleLossConfig,
)


def test_tapered_startup_prevents_spurious_superluminal_charge_interpolation():
    beta, dt, omega = 0.99, 1e-8, 1e6
    initial = _state(position_mm=(0, 0, 0), beta=(beta, 0, 0), observer_charge=0.5)
    tail = copy.deepcopy(initial)
    phase = omega * dt
    tail["t"][:] = dt
    tail["x"][:] = beta * c * np.sin(phase) / omega
    tail["y"][:] = beta * c * 2 * np.sin(phase / 2) ** 2 / omega
    tail["bx"][:] = beta * np.cos(phase)
    tail["by"][:] = beta * np.sin(phase)

    def history(taper):
        offsets = _causal_c5_inertial_time_offsets_ns(1.0, 2 * dt) if taper else None
        return _build_inertial_coasting_history(initial, 1.0, time_offsets_ns=offsets)

    def maximum_sampled_speed(states):
        source = _prepare_history_uncached(states, ()).sources[0]
        fractions = np.linspace(0, 1, 21)
        derivative_coefficients = (
            source.position_coefficients_mm[:, 1:] * np.arange(1, 6)[None, :, None]
        )
        velocities = np.einsum(
            "ikc,ks->isc",
            derivative_coefficients,
            fractions[None, :] ** np.arange(5)[:, None],
        )
        velocities /= c * source.segment_duration_ns[:, None, None]
        return float(np.max(np.linalg.norm(velocities, axis=-1)))

    sparse, tapered = history(False), history(True)
    assert maximum_sampled_speed(sparse) < 1
    assert maximum_sampled_speed(sparse + [tail]) > 10
    assert maximum_sampled_speed(tapered + [tail]) < 1
    # The observer still sees the earlier coasting motion. Adding the short
    # bend must not change that already-covered retarded field materially.
    event = ObserverEvent(time_ns=dt, position_mm=(0, 10, 0))
    before, after = [
        evaluate_retarded_charge_field_native(states, event)
        for states in (tapered, tapered + [tail])
    ]
    np.testing.assert_allclose(
        after.four_potential, before.four_potential, rtol=1e-11, atol=1e-14
    )


@pytest.mark.parametrize("beta", [0.99, 0.9999])
def test_public_adaptive_start_completes_relativistic_bend(tmp_path, beta):
    """Exercise generated seed spacing through the normal public runner.

    Recoil is off to isolate the charge interpolation failure. The neutral
    partner still receives the electron's retarded charge field during endpoint
    evaluation, just as in the live experimental-recoil benchmark.
    """
    rider = _species_state("electron", position_mm=(0, 0, 0), beta=(beta, 0, 0))
    driver = _species_state("neutron", position_mm=(0, 10, 0))
    field = magnetic_field_tesla_to_native(1e7)
    omega = abs(rider["q"][0]) * field / (rider["m"][0] * c)
    angle = 0.02
    target = rider["gamma"][0] * angle / omega
    result = _run_simple_inertial(
        rider,
        driver,
        steps=9,
        h_step=target / 8,
        magnetic_dipole=MagneticDipoleConfig(
            enabled=True,
            exact_retarded_update="second_order_start_taylor_endpoint",
            rider=MagneticDipoleParticleConfig(species="electron"),
            driver=MagneticDipoleParticleConfig(species="neutron"),
        ),
        radiation_reaction_mode="off",
        external_field=ExternalFieldConfig(magnetic_field_native=(0, 0, field)),
        adaptive_pair_return=AdaptivePairReturnConfig(
            enabled=True,
            target_lab_time_ns=target,
            tolerance_scale=1e5,
            maximum_step_factor=1.0,
        ),
        checkpoint=CheckpointConfig(
            enabled=True, directory=str(tmp_path / "checkpoint")
        ),
        particle_loss=ParticleLossConfig(enabled=False),
    )
    trajectory = result[2]
    assert trajectory.t[-1, 0] == pytest.approx(target, rel=1e-12)
    speeds = np.sqrt(trajectory.bx**2 + trajectory.by**2 + trajectory.bz**2)
    assert np.all(np.isfinite(speeds)) and np.all(speeds < 1)
    assert np.arctan2(trajectory.by[-1, 0], trajectory.bx[-1, 0]) == pytest.approx(
        angle, rel=5e-5
    )
