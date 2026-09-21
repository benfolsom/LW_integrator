import numpy as np
import pytest

from core.constants import C_MMNS as c
from core.full_dipole_history import FullDipoleHistory
from core.full_dipole_radiation import FullDipoleRadiationSampler
from core.radiation_flux_oracle import gauss_legendre_sphere_quadrature


@pytest.mark.parametrize("indices", [(1, 2), (0, 1)])
def test_rest_electric_or_magnetic_dipole_has_expected_quadratic_radiation(indices):
    time = np.linspace(-1, 1, 41) / c
    moment = np.zeros((len(time), 4, 4))
    i, j = indices
    moment[:, i, j] = 0.1 * c * (c * time) ** 2
    moment[:, j, i] = -moment[:, i, j]
    h = FullDipoleHistory(
        time,
        np.zeros((len(time), 3)),
        np.zeros((len(time), 3)),
        moment,
        c,
        position_tolerance=1e-12,
    ).completed()
    sampler = FullDipoleRadiationSampler(h, 0.0)
    quad = gauss_legendre_sphere_quadrature(polar_order=4, azimuthal_order=8)
    flux = sampler.emission_flux(time_ns=0.275 / c, radius_mm=16, quadrature=quad)[2]
    np.testing.assert_allclose(
        flux.mu_squared.energy_rate_native / c**3, 2 / 3 * 0.2**2, rtol=2e-9
    )
    np.testing.assert_allclose(
        flux.mu_squared.momentum_rate_native / c**2, 0, atol=1e-11
    )
    assert flux.q_squared.energy_rate_native == 0
    assert flux.q_mu_interference.energy_rate_native == 0
