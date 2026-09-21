"""Independent outward self-radiation from accepted full dipole histories.

Uses the full retarded potential response, not a local reaction formula.
Emission-time sampling is per source; coherent inter-particle radiation
requires common observation cuts and must not be formed by adding these fluxes.
"""

from dataclasses import replace
from typing import Any

import numpy as np

from .constants import C_MMNS as c
from .full_dipole_history import FullDipoleHistory
from .momentum_center_pair import FullDipoleProvider
from .radiation_flux_oracle import integrate_radiation_sphere_flux_native


def _fields(tensor: np.ndarray) -> np.ndarray:
    return np.array([tensor[1:, 0], [-tensor[2, 3], tensor[1, 3], -tensor[1, 2]]])


class FullDipoleRadiationSampler:
    """Prepared q and D sources sharing exactly the same accepted worldline."""

    def __init__(self, history: FullDipoleHistory, charge_native: float):
        self.history = history
        charge_history = replace(
            history,
            dipole=np.zeros_like(history.dipole),
            segments=tuple(
                replace(s, dipole=np.zeros_like(s.dipole)) for s in history.segments
            ),
        )
        # No externally applied startup multiplier belongs on the emitted field.
        self.total = FullDipoleProvider(history, charge_native)
        self.charge = FullDipoleProvider(charge_history, charge_native)

    def emission_flux(
        self, *, time_ns: float, radius_mm: float, quadrature: Any
    ) -> dict[int, Any]:
        """Three-radius extraction at one emission time; two extrapolation orders.

        Sphere centers coincide with this source's position at the chosen
        emission. The factor 1-n.beta converts observation to emission-time
        flux in the far limit. Angular momentum is NOT determined by the
        leading 1/r fields: ignore the generic flux object's angular member.
        """
        if not np.isfinite(radius_mm) or radius_mm <= 0 or not np.isfinite(time_ns):
            raise ValueError("Finite emission time and positive radius required")
        segment = self.history.segment_at(time_ns)
        point, _ = segment.sample(time_ns)
        velocity, _ = segment.sample(time_ns, 1)
        beta = velocity / c
        jacobian = 1 - quadrature.directions @ beta
        samples = []
        for radius in radius_mm * np.array([1.0, 2.0, 4.0]):
            rays = []
            for direction in quadrature.directions:
                position = point + radius * direction
                time = time_ns + radius / c
                charge = _fields(self.charge(time, position)[2])
                dipole = _fields(self.total(time, position)[2]) - charge
                rays.append(radius * np.array([charge, dipole]))
            samples.append(rays)
        result = {}
        for degree, weights in ((1, [0.0, -1.0, 2.0]), (2, [1 / 3, -2.0, 8 / 3])):
            amplitudes = np.einsum("r,rnkij->nkij", weights, samples)
            amplitudes *= np.sqrt(jacobian)[:, None, None, None]
            result[degree] = integrate_radiation_sphere_flux_native(
                quadrature=quadrature,
                radius_mm=1.0,
                charge_electric_field_native=amplitudes[:, 0, 0],
                charge_magnetic_field_native=amplitudes[:, 0, 1],
                dipole_electric_field_native=amplitudes[:, 1, 0],
                dipole_magnetic_field_native=amplitudes[:, 1, 1],
            )
        return result
