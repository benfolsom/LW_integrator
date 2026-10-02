"""First-order prescribed-orbit kick from an exact weighted electron source.

This checks field/accounting normalization, not coupled trajectory evolution or
radiation reaction. Source motion is exactly inertial for this analytic case.
"""

import numpy as np
import pytest
from scipy.integrate import quad

from core.constants import C_MMNS, ELEMENTARY_CHARGE
from core.integration_runner import _build_inertial_coasting_history
from core.retarded_fields import ObserverEvent, evaluate_retarded_charge_field_native
from tests.unit.test_inertial_prehistory import _species_state

pytestmark = pytest.mark.physics


@pytest.mark.parametrize("proton_energy_mev", [3.6, 20, 90, 216, 1000])
def test_weighted_electron_crossing_impulse_and_longitudinal_work(proton_energy_mev):
    gamma_e = 1 + 5 / 0.51099895
    beta_e = np.sqrt(1 - gamma_e**-2)
    gamma_p = 1 + proton_energy_mev / 938.27208816
    beta_p = np.sqrt(1 - gamma_p**-2)
    population, impact_mm = 1e8, 1.5
    end_ns = 10000.0
    electron = _species_state(
        "electron",
        position_mm=(0, 0, -beta_e * C_MMNS * end_ns),
        beta=(0, 0, -beta_e),
        source_charge=-population * ELEMENTARY_CHARGE,
    )
    electron["t"][:] = end_ns
    electron["macro_population"][:] = population
    history = _build_inertial_coasting_history(electron, 2 * end_ns, knot_count=4)
    time_scale = impact_mm / (gamma_e * (beta_e + beta_p) * C_MMNS)

    def force(s):
        lab_time = s * time_scale
        field = evaluate_retarded_charge_field_native(
            history,
            ObserverEvent(lab_time, (impact_mm, 0, beta_p * C_MMNS * lab_time)),
        )
        assert field.valid_sources.all()
        return ELEMENTARY_CHARGE * (
            field.electric_field_native
            + np.cross([0, 0, beta_p], field.magnetic_field_native)
        )

    # Compactify each half of the infinite encounter. The quadrature weights
    # include dt/ds and ds/du; no sampled field or shared retarded root is used.
    def impulse_integrand(u):
        s = u / (1 - u)
        return (force(s)[0] + force(-s)[0]) * time_scale / (1 - u) ** 2

    impulse, _ = quad(impulse_integrand, 0, 1, epsabs=1e-13, epsrel=1e-8)
    reference = (
        -2
        * population
        * ELEMENTARY_CHARGE**2
        * (1 + beta_e * beta_p)
        / ((beta_e + beta_p) * C_MMNS * impact_mm)
    )
    np.testing.assert_allclose(impulse, reference, rtol=1e-7, atol=0)

    def work_integrand(u):
        s = u / (1 - u)
        return (
            (force(s)[2] + force(-s)[2]) * beta_p * C_MMNS * time_scale / (1 - u) ** 2
        )

    work, _ = quad(work_integrand, 0, 1, epsabs=1e-12, epsrel=1e-8)
    assert abs(work) < 1e-8 * abs(reference * beta_p * C_MMNS)
