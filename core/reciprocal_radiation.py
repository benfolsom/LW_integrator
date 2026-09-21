"""Diagnostic far-radiation extraction from frozen accepted source histories.

This does not feed fields back into the equations of motion. Observation
times obey t = cut_time + radius/c; a cut is one common outgoing light-time
coordinate, not simultaneous emission times at the particles.
"""

import numpy as np
from scipy.optimize import brentq
from .constants import C_MMNS as c
from .jakobsen_pair import FrozenSource, RetardedSourceProvider
from .radiation_flux_oracle import integrate_radiation_sphere_flux_native


def _fields(tensor):
    return np.array([tensor[1:, 0], [-tensor[2, 3], tensor[1, 3], -tensor[1, 2]]])


class ReciprocalRadiationSampler:
    """Keep source fields separate until interference can be accounted for."""

    def __init__(self, sources):
        if not sources:
            raise ValueError("At least one frozen source required")
        self.providers = [RetardedSourceProvider(source) for source in sources]
        self.charges = [
            RetardedSourceProvider(
                FrozenSource(
                    source.particle,
                    source.rows,
                    source.segments,
                    [np.zeros_like(s) for s in source.spin_segments],
                )
            )
            for source in sources
        ]

    def amplitudes(self, *, cut_time_ns, radius_mm, quadrature):
        """Return linear and quadratic estimates, indexed source, ray, q/mu, E/B, xyz.

        Fit r times each field in 1/r at r, 2r and 4r. Its intercept is the
        coefficient of the radiation field. Both estimates remain available
        so callers can report extrapolation sensitivity.
        """
        if not np.isfinite(radius_mm) or radius_mm <= 0 or not np.isfinite(cut_time_ns):
            raise ValueError("Finite cut time and positive radius required")
        values = []
        for radius in radius_mm * np.array([1.0, 2.0, 4.0]):
            each_source = []
            for total, charge in zip(self.providers, self.charges):
                rays = []
                for n in quadrature.directions:
                    t, x = cut_time_ns + radius / c, radius * n
                    q = _fields(charge(t, x)[2])
                    magnetic = _fields(total(t, x)[2]) - q
                    rays.append(np.array([q, magnetic]) * radius)
                each_source.append(rays)
            values.append(each_source)
        values = np.asarray(values)
        return {
            1: np.einsum("r,rsnkij->snkij", [0.0, -1.0, 2.0], values),
            2: np.einsum("r,rsnkij->snkij", [1 / 3, -2.0, 8 / 3], values),
        }

    def emission_self_flux(self, *, emission_time_ns, radius_mm, quadrature):
        """Match each particle's own emission time and include dt_observation/dt_source.

        Return per-source fluxes, never a coherent pair sum: different sources
        here are observed at different times. This is the appropriate self-
        radiation comparison for simultaneous particle endpoint accounting.
        """
        if (
            not np.isfinite(radius_mm)
            or radius_mm <= 0
            or not np.isfinite(emission_time_ns)
        ):
            raise ValueError("Finite emission time and positive radius required")
        results = {1: [], 2: []}
        for provider, charge in zip(self.providers, self.charges):
            source = provider.source
            k = int(np.searchsorted(source.times, emission_time_ns, side="right") - 1)
            if not 0 <= k < len(source.durations):
                raise ValueError("Emission time outside accepted history")
            width = source.durations[k]
            fraction = (emission_time_ns - source.times[k]) / width
            point = np.polynomial.polynomial.polyval(fraction, source.coefficients[k])
            beta = np.polynomial.polynomial.polyval(
                fraction, np.polynomial.polynomial.polyder(source.coefficients[k])
            ) / (width * c)
            jacobian = 1 - quadrature.directions @ beta
            samples = []
            for radius in radius_mm * np.array([1.0, 2.0, 4.0]):
                rays = []
                for n in quadrature.directions:
                    x = radius * n
                    time = emission_time_ns + np.linalg.norm(x - point) / c
                    q = _fields(charge(time, x)[2])
                    magnetic = _fields(provider(time, x)[2]) - q
                    rays.append(np.array([q, magnetic]) * radius)
                samples.append(rays)
            for degree, weights in ((1, [0.0, -1.0, 2.0]), (2, [1 / 3, -2.0, 8 / 3])):
                amplitude = np.einsum("r,rnkij->nkij", weights, samples)
                # The square root weights all quadratic sectors, including q*mu.
                amplitude *= np.sqrt(jacobian)[:, None, None, None]
                results[degree].append(amplitude_flux(amplitude[None, :], quadrature))
        return results


def amplitude_flux(amplitudes, quadrature):
    """Sum fields BEFORE forming Maxwell flux; radius one removes factored r.

    Amplitudes are coefficients of 1/r fields, not actual fields at 1 mm.
    Angular momentum cannot be inferred from this leading coefficient alone;
    callers must not use the generic flux result's angular momentum member.
    """
    summed = np.sum(amplitudes, axis=0)
    return integrate_radiation_sphere_flux_native(
        quadrature=quadrature,
        radius_mm=1.0,
        charge_electric_field_native=summed[:, 0, 0],
        charge_magnetic_field_native=summed[:, 0, 1],
        dipole_electric_field_native=summed[:, 1, 0],
        dipole_magnetic_field_native=summed[:, 1, 1],
    )


def charge_radiation_amplitudes(sources, *, cut_time_ns, quadrature):
    """Independent charge reference using the far-zone Lienard-Wiechert formula.

    Solve t_source - n.x_source/c = cut_time per ray. This is the limiting
    light cone, not a sphere of finite radius. No potential derivative or
    radius extrapolation is used. Magnetic-source amplitudes are zero here.
    """
    amplitudes = np.zeros((len(sources), quadrature.sample_count, 2, 2, 3))
    for i, source in enumerate(sources):

        def state(t):
            k = int(
                np.clip(
                    np.searchsorted(source.times, t, side="right") - 1,
                    0,
                    len(source.durations) - 1,
                )
            )
            width = source.durations[k]
            fraction = (t - source.times[k]) / width
            coeff = source.coefficients[k]
            x = np.polynomial.polynomial.polyval(fraction, coeff)
            beta = np.polynomial.polynomial.polyval(
                fraction, np.polynomial.polynomial.polyder(coeff)
            ) / (width * c)
            beta_rate = np.polynomial.polynomial.polyval(
                fraction, np.polynomial.polynomial.polyder(coeff, 2)
            ) / (width**2 * c)
            return x, beta, beta_rate

        for j, n in enumerate(quadrature.directions):
            root = brentq(
                lambda t: t - n @ state(t)[0] / c - cut_time_ns,
                source.times[0],
                source.times[-1],
                xtol=1e-15,
                rtol=1e-14,
            )
            _, beta, beta_rate = state(root)
            electric = (
                source.particle.charge_native
                / c
                * np.cross(n, np.cross(n - beta, beta_rate))
                / (1 - n @ beta) ** 3
            )
            amplitudes[i, j, 0] = [electric, np.cross(n, electric)]
    return amplitudes
