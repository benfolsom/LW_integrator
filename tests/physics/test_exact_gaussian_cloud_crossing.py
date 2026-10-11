"""Prescribed-orbit exact-field quadrature, independent of coupled stepping."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad

from core.constants import C_MMNS, ELEMENTARY_CHARGE
from core.exact_source_cloud import exact_cloud_history
from core.integration_runner import _build_inertial_coasting_history
from core.retarded_fields import ObserverEvent, evaluate_retarded_charge_field_native
from tests.unit.test_exact_source_cloud import cloud_config
from tests.unit.test_inertial_prehistory import _species_state

pytestmark = pytest.mark.physics


@pytest.mark.parametrize("width_mm", [0.1, 0.2, 0.25])
def test_gaussian_cloud_kick_converges_with_count_and_preserves_width(width_mm):
    gamma_e, gamma_p = 1 + 5 / 0.51099895, 1 + 20 / 938.27208816
    beta_e, beta_p = np.sqrt(1 - gamma_e**-2), np.sqrt(1 - gamma_p**-2)
    population, impact_mm, end_ns = 1e8, 1.5, 10000.0
    electron = _species_state(
        "electron",
        position_mm=(0, 0, -beta_e * C_MMNS * end_ns),
        beta=(0, 0, -beta_e),
        source_charge=-population * ELEMENTARY_CHARGE,
    )
    electron["t"][:] = end_ns
    history = _build_inertial_coasting_history(electron, 2 * end_ns, knot_count=4)
    time_scale = impact_mm / (gamma_e * (beta_e + beta_p) * C_MMNS)
    point_reference = (
        -2
        * population
        * ELEMENTARY_CHARGE**2
        * (1 + beta_e * beta_p)
        / ((beta_e + beta_p) * C_MMNS * impact_mm)
    )
    gaussian_reference = point_reference * (
        1 - np.exp(-(impact_mm**2) / (2 * width_mm**2))
    )
    rows = []
    # Refine the even-order tensor family: no central node lies on the
    # prescribed observer's axis. Odd-order rules have a different aliasing
    # pattern for the near-singular point kernels and are reported separately.
    for count in (4, 16, 64, 100):
        cloud = exact_cloud_history(history, cloud_config(count, width_mm))

        def force(s):
            lab_time = s * time_scale
            field = evaluate_retarded_charge_field_native(
                cloud,
                ObserverEvent(lab_time, (impact_mm, 0, beta_p * C_MMNS * lab_time)),
            )
            assert field.valid_sources.all()
            assert field.valid_sources.size == count
            return (
                ELEMENTARY_CHARGE
                * (
                    field.electric_field_native
                    + np.cross([0, 0, beta_p], field.magnetic_field_native)
                )[0]
            )

        def integrand(u):
            s = u / (1 - u)
            return (force(s) + force(-s)) * time_scale / (1 - u) ** 2

        kick, error = quad(integrand, 0, 1, epsabs=1e-13, epsrel=1e-8)
        relative = abs(kick / gaussian_reference - 1)
        rows.append(
            {
                "count": count,
                "width_mm": width_mm,
                "kick_amu_mm_ns": kick,
                "quadrature_error": error,
                "gaussian_reference_amu_mm_ns": gaussian_reference,
                "relative_error": relative,
            }
        )
    assert rows[1]["relative_error"] < rows[0]["relative_error"]
    assert rows[2]["relative_error"] < rows[1]["relative_error"]
    assert (
        rows[3]["relative_error"]
        <= rows[2]["relative_error"] + 64 * np.finfo(float).eps
    )
    np.testing.assert_allclose(
        rows[-1]["kick_amu_mm_ns"], gaussian_reference, rtol=1e-7, atol=0
    )
    # Optional evidence capture is explicit, rather than a normal test side effect.
    import os

    evidence_directory = os.environ.get("LW_EXACT_CLOUD_EVIDENCE")
    if evidence_directory:
        Path(evidence_directory, f"gaussian_width_{width_mm:g}.json").write_text(
            json.dumps(rows, indent=2) + "\n"
        )
