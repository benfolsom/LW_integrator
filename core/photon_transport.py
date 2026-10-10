"""Opt-in photon particles, unpolarised Compton scattering, and LCFA conversion.

Native units are mm, ns, amu, and scaled Gaussian charge. Photons never enter
LW source arrays. Pair momenta are on shell; unresolved field exchange is logged.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any, Callable

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.special import kv

from .constants import C_MMNS as C, ELECTRON_MASS_AMU as M, ELEMENTARY_CHARGE as Q
from .magnetic_dipole import HBAR_NATIVE
from .stochastic_emission import bessel_integral

ALPHA = Q**2 / (HBAR_NATIVE * C)


@dataclass(frozen=True)
class PhotonTransportConfig:
    enabled: bool = False
    seed: int = 0
    scattering: bool = True
    pair_conversion: bool = True
    interaction_radius_mm: float = 1.0
    max_probability: float = 0.01
    max_step_mm: float = 0.1
    domain_min_mm: tuple[float, float, float] = (-1000.0, -1000.0, -1000.0)
    domain_max_mm: tuple[float, float, float] = (1000.0, 1000.0, 1000.0)
    # Explicit additional uniform background; these fields do not drive charges.
    electric_native: tuple[float, float, float] = (0.0, 0.0, 0.0)
    magnetic_native: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        for name in ("enabled", "scattering", "pair_conversion"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"photon transport {name} must be boolean")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("photon transport seed must be a nonnegative integer")
        for name in ("interaction_radius_mm", "max_probability", "max_step_mm"):
            value = getattr(self, name)
            if isinstance(value, (str, bool)) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"photon transport {name} must be positive and finite")
        if self.max_probability > 0.1:
            raise ValueError("photon transport max_probability must be <= 0.1")
        for name in (
            "domain_min_mm",
            "domain_max_mm",
            "electric_native",
            "magnetic_native",
        ):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (3,) or not np.isfinite(value).all():
                raise ValueError(
                    f"photon transport {name} must be a finite three-vector"
                )
            object.__setattr__(self, name, tuple(float(x) for x in value))
        if np.any(np.asarray(self.domain_min_mm) >= self.domain_max_mm):
            raise ValueError("photon domain must have positive extent on all axes")

    @classmethod
    def from_dict(cls, value: Any) -> PhotonTransportConfig:
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            raise ValueError("photon_transport must be an object")
        return cls(**value)


def boost(four: np.ndarray, beta: np.ndarray, gamma: float | None = None) -> np.ndarray:
    """Lorentz boost to a frame moving at beta; components are (E/c, p)."""
    b2 = float(beta @ beta)
    if b2 == 0:
        return four.copy()
    if not 0 < b2 < 1:
        raise ValueError("Finite subluminal boost required")
    gamma = 1 / np.sqrt(1 - b2) if gamma is None else gamma
    dot = float(beta @ four[1:])
    return np.r_[
        gamma * (four[0] - dot),
        four[1:] + (gamma**2 / (gamma + 1) * dot - gamma * four[0]) * beta,
    ]


def klein_nishina_total(energy: float, mass: float = M, charge: float = Q) -> float:
    """Total free-target cross-section in mm², at rest-frame photon energy."""
    x = energy / (mass * C**2)
    if not np.isfinite(x) or x <= 0 or mass <= 0:
        raise ValueError("Positive finite photon energy and target mass required")
    radius = charge**2 / (mass * C**2)
    if x < 1e-3:
        return float(
            8 * np.pi / 3 * radius**2 * (1 - 2 * x + 26 * x * x / 5 - 133 * x**3 / 10)
        )
    log = np.log1p(2 * x)
    return float(
        2
        * np.pi
        * radius**2
        * (
            (1 + x) / x**3 * (2 * x * (1 + x) / (1 + 2 * x) - log)
            + log / (2 * x)
            - (1 + 3 * x) / (1 + 2 * x) ** 2
        )
    )


def compton_scatter(
    energy: float,
    direction: np.ndarray,
    momentum: np.ndarray,
    mass: float,
    rng: np.random.Generator,
) -> tuple[float, np.ndarray, np.ndarray, dict[str, Any]]:
    """Sample unpolarised KN in the target rest frame; apply exact recoil."""
    before = np.r_[np.hypot(mass * C, np.linalg.norm(momentum)), momentum]
    photon = energy / C * np.r_[1.0, direction]
    beta = momentum / before[0]
    gamma = before[0] / (mass * C)
    rest = boost(photon, beta, gamma)
    axis = rest[1:] / np.linalg.norm(rest[1:])
    # Evaluate p0-p.n without cancellation for nearly co-propagating beams.
    norm = np.linalg.norm(momentum)
    p_axis = momentum / norm if norm else np.zeros(3)
    rest_gap = (mass * C) ** 2 / (before[0] + norm)
    incident_gap = rest_gap + (norm / 2) * np.sum((p_axis - direction) ** 2)
    rest_energy_c = photon[0] * incident_gap / (mass * C)
    x = rest_energy_c / (mass * C)
    # At high energy use the KN energy-ratio mixture (1/r + r), as in
    # Geant4, so narrow forward peaks do not require O(x) uniform proposals.
    while True:
        if x < 1:
            mu = rng.uniform(-1.0, 1.0)
            ratio = 1 / (1 + x * (1 - mu))
            acceptance = ratio**2 * (ratio + 1 / ratio - (1 - mu**2)) / 2
        else:
            minimum = 1 / (1 + 2 * x)
            logarithmic = np.log1p(2 * x)
            linear = (1 - minimum**2) / 2
            if rng.random() * (logarithmic + linear) < logarithmic:
                ratio = np.exp(-rng.random() * logarithmic)
            else:
                ratio = np.sqrt(minimum**2 + rng.random() * (1 - minimum**2))
            mu = float(np.clip(1 - (1 / ratio - 1) / x, -1, 1))
            acceptance = 1 - (1 - mu**2) / (ratio + 1 / ratio)
        if rng.random() <= acceptance:
            break
    helper = np.eye(3)[int(np.argmin(np.abs(axis)))]
    e1 = np.cross(axis, helper)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    phi = rng.uniform(0, 2 * np.pi)
    out_direction = mu * axis + np.sqrt(max(0.0, 1 - mu**2)) * (
        np.cos(phi) * e1 + np.sin(phi) * e2
    )
    rest_out = rest_energy_c * ratio * np.r_[1.0, out_direction]
    boosted = boost(rest_out, -beta, gamma)
    lab_direction = boosted[1:] / np.linalg.norm(boosted[1:])
    outgoing_gap = rest_gap + (norm / 2) * np.sum((p_axis - lab_direction) ** 2)
    angular_gap = np.sum((direction - lab_direction) ** 2) / 2
    # Solve the invariant Compton energy relation in the lab, then put the
    # target exactly on shell. This avoids two cancelling inverse boosts.
    out_norm = photon[0] * incident_gap / (outgoing_gap + photon[0] * angular_gap)
    out = out_norm * np.r_[1.0, lab_direction]
    target_p = momentum + photon[1:] - out[1:]
    target = np.r_[np.hypot(mass * C, np.linalg.norm(target_p)), target_p]
    return (
        float(C * out[0]),
        out[1:] / np.linalg.norm(out[1:]),
        target[1:],
        {
            "target_before_four_native": before.tolist(),
            "target_after_four_native": target.tolist(),
            "photon_before_four_native": photon.tolist(),
            "photon_after_four_native": out.tolist(),
            "rest_frame_cos_theta": float(mu),
        },
    )


def photon_chi(
    energy: float, direction: np.ndarray, electric: np.ndarray, magnetic: np.ndarray
) -> float:
    """chi_gamma from local Gaussian fields (B has the same units as E)."""
    effective = electric + np.cross(direction, magnetic)
    transverse = effective - (direction @ electric) * direction
    return float(
        energy
        / (M * C**2)
        * HBAR_NATIVE
        * abs(Q)
        * np.linalg.norm(transverse)
        / (M**2 * C**3)
    )


def breit_wheeler_density(
    fraction: np.ndarray, chi: float, energy: float
) -> np.ndarray:
    """Spin/polarisation averaged CCF dW/df, per ns; high-energy energy fraction.

    Ritus kernel: [(f/(1-f)+(1-f)/f) K_2/3(z) + integral_z^inf K_1/3],
    z=2/[3 chi f(1-f)]. Finite-energy on-shell support is applied by spectrum.
    """
    f = np.asarray(fraction)
    result = np.zeros_like(f, dtype=float)
    if chi <= 0:
        return result
    valid = (f > 0) & (f < 1)
    fv = f[valid]
    z = 2 / (3 * chi * fv * (1 - fv))
    prefactor = ALPHA * (M * C**2) ** 2 / (np.sqrt(3) * np.pi * HBAR_NATIVE * energy)
    result[valid] = prefactor * (
        (fv / (1 - fv) + (1 - fv) / fv) * kv(2 / 3, z) + bessel_integral(z)
    )
    return result


@lru_cache(maxsize=128)
def breit_wheeler_spectrum(
    chi: float, energy: float
) -> tuple[np.ndarray, np.ndarray, float]:
    """Numerical CDF on finite energy support; no rounded-chi cache key."""
    lower = M * C**2 / energy
    if lower >= 0.5 or chi <= 0:
        return np.array([0.5, 0.5]), np.zeros(2), 0.0
    # Cosine grid resolves both endpoint tails and symmetric energy sharing.
    t = np.linspace(0, 1, 4097)
    f = lower + (1 - 2 * lower) * (1 - np.cos(np.pi * t)) / 2
    density = breit_wheeler_density(f, chi, energy)
    cdf = cumulative_trapezoid(density, f, initial=0)
    rate = float(cdf[-1])
    return f, cdf / rate if rate else cdf, rate


def pair_kinematics(
    energy: float, direction: np.ndarray, fraction: float
) -> dict[str, Any]:
    """Collinear LCFA prescription with exact mass shells and field ledger.

    A photon alone cannot make two massive particles. Q=p_minus+p_plus-k is
    the unresolved background impulse; no finite-source recoil is invented.
    """
    energies = np.array([fraction, 1 - fraction]) * energy
    if np.any(energies < M * C**2):
        raise ValueError("Pair energy split must retain both rest energies")
    magnitudes = np.sqrt(energies - M * C**2) * np.sqrt(energies + M * C**2) / C
    pair = np.column_stack([energies / C, magnitudes[:, None] * direction])
    photon = energy / C * np.r_[1.0, direction]
    return {
        "electron_four_native": pair[0].tolist(),
        "positron_four_native": pair[1].tolist(),
        "photon_before_four_native": photon.tolist(),
        "background_impulse_four_native": (pair.sum(axis=0) - photon).tolist(),
        "energy_fraction": float(fraction),
        "particles_inserted": False,
        "remainder_receiver": "unresolved_field_source_system",
    }


@dataclass
class PhotonTarget:
    key: str
    position: np.ndarray
    momentum: np.ndarray
    mass: float
    charge: float


class PhotonTransportRuntime:
    """Accepted-barrier state, independent RNG, and JSON checkpoint payload."""

    def __init__(self, config: PhotonTransportConfig, payload: dict | None = None):
        self.config = config
        self.rng = np.random.Generator(np.random.PCG64(config.seed))
        self.photons: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.emission_cursor = 0
        if payload is not None:
            if (
                payload.get("schema_version") != 1
                or PhotonTransportConfig.from_dict(payload["config"]) != config
            ):
                raise ValueError("Incompatible photon transport checkpoint")
            self.rng.bit_generator.state = copy.deepcopy(payload["rng_state"])
            self.photons = copy.deepcopy(payload["photons"])
            self.events = copy.deepcopy(payload["events"])
            self.emission_cursor = int(payload["emission_cursor"])

    def ingest(self, emission_events: list[dict[str, Any]]) -> None:
        if len(emission_events) < self.emission_cursor:
            raise ValueError("Emission event log shrank below transport cursor")
        for index in range(self.emission_cursor, len(emission_events)):
            event = emission_events[index]
            if (
                event.get("event_type", "photon") != "photon"
                or "energy_native" not in event
            ):
                continue
            direction = np.asarray(event["direction"], dtype=float)
            position = np.asarray(event["position_mm"], dtype=float)
            energy = float(event["energy_native"])
            time = float(event["time_ns"])
            if (
                direction.shape != (3,)
                or position.shape != (3,)
                or not np.isfinite(np.r_[direction, position, energy, time]).all()
                or energy <= 0
                or not np.isclose(np.linalg.norm(direction), 1, atol=1e-12, rtol=0)
            ):
                raise ValueError("Invalid photon emission event")
            self.photons.append(
                dict(
                    creation_event_id=f"emission:{index}",
                    time_ns=time,
                    creation_time_ns=time,
                    position_mm=position.tolist(),
                    direction=direction.tolist(),
                    energy_native=energy,
                    polarisation=copy.deepcopy(event.get("polarisation")),
                    status="alive",
                )
            )
            if self._escape_distance(position, direction) <= 0:
                self.photons[-1]["status"] = "escaped"
                self.events.append(
                    dict(
                        event_type="escape",
                        creation_event_id=f"emission:{index}",
                        time_ns=time,
                        position_mm=position.tolist(),
                    )
                )
        self.emission_cursor = len(emission_events)

    def _escape_distance(self, position: np.ndarray, direction: np.ndarray) -> float:
        low, high = np.asarray(self.config.domain_min_mm), np.asarray(
            self.config.domain_max_mm
        )
        if np.any(position < low) or np.any(position > high):
            return 0.0
        distances = [
            (
                (high[i] - position[i]) / direction[i]
                if direction[i] > 0
                else (low[i] - position[i]) / direction[i]
            )
            for i in range(3)
            if direction[i] != 0
        ]
        return float(min(distances))

    def advance(
        self,
        endpoint_ns: float,
        targets: list[PhotonTarget],
        fields: (
            Callable[[float, np.ndarray], tuple[np.ndarray, np.ndarray]] | None
        ) = None,
    ) -> None:
        """Bounded Bernoulli W dt, competing channels, and straight flight at c.

        Local density is one physical target per top-hat sphere of radius R;
        rate n sigma_rest c (1-beta.n) uses lab density and invariant flux.
        Positions/fields are sampled at substep starts; target endpoints are
        frozen by the adapter. max_step_mm also bounds geometry sampling.
        """
        if not np.isfinite(endpoint_ns):
            raise ValueError("Finite photon endpoint required")
        radius = self.config.interaction_radius_mm
        volume = 4 * np.pi * radius**3 / 3
        for photon in self.photons:
            if photon["status"] not in ("alive", "scattered"):
                continue
            while photon["time_ns"] < endpoint_ns:
                position = np.asarray(photon["position_mm"], dtype=float)
                direction = np.asarray(photon["direction"], dtype=float)
                energy = photon["energy_native"]
                escape = self._escape_distance(position, direction) / C
                if escape <= 0:
                    photon["status"] = "escaped"
                    self.events.append(
                        dict(
                            event_type="escape",
                            creation_event_id=photon["creation_event_id"],
                            time_ns=photon["time_ns"],
                            position_mm=position.tolist(),
                        )
                    )
                    break
                rates, eligible = [], []
                if self.config.scattering:
                    for target in targets:
                        if (
                            target.charge == 0
                            or np.linalg.norm(position - target.position) > radius
                        ):
                            continue
                        norm = np.linalg.norm(target.momentum)
                        p0 = np.hypot(target.mass * C, norm)
                        axis = target.momentum / norm if norm else np.zeros(3)
                        gap = (target.mass * C) ** 2 / (p0 + norm) + norm / 2 * np.sum(
                            (axis - direction) ** 2
                        )
                        flux = gap / p0
                        rest_energy = energy * gap / (target.mass * C)
                        rates.append(
                            klein_nishina_total(rest_energy, target.mass, target.charge)
                            * C
                            * flux
                            / volume
                        )
                        eligible.append(target)
                chi, pair_rate = 0.0, 0.0
                if self.config.pair_conversion and energy > 2 * M * C**2:
                    electric = np.asarray(self.config.electric_native)
                    magnetic = np.asarray(self.config.magnetic_native)
                    if fields is not None:
                        # A birth position can coincide with its charged emitter.
                        # Query strictly along the outgoing flight, not at that pole.
                        field_dt = (
                            min(
                                endpoint_ns - photon["time_ns"],
                                escape,
                                self.config.max_step_mm / C,
                                radius / (4 * C),
                            )
                            / 2
                        )
                        e, b = fields(
                            photon["time_ns"] + field_dt,
                            position + C * field_dt * direction,
                        )
                        electric, magnetic = electric + e, magnetic + b
                    if not np.isfinite(np.r_[electric, magnetic]).all():
                        raise ValueError("Nonfinite photon field sample")
                    chi = photon_chi(energy, direction, electric, magnetic)
                    _, _, pair_rate = breit_wheeler_spectrum(chi, energy)
                total = float(sum(rates) + pair_rate)
                duration = min(
                    endpoint_ns - photon["time_ns"],
                    escape,
                    self.config.max_step_mm / C,
                    radius / (4 * C),
                )
                if total > 0:
                    duration = min(duration, self.config.max_probability / total)
                new_time = photon["time_ns"] + duration
                if new_time == photon["time_ns"]:
                    raise RuntimeError(
                        "Photon substep is below floating-point time resolution"
                    )
                # Endpoint events: escape wins ties at the domain surface.
                position = position + C * duration * direction
                photon.update(time_ns=float(new_time), position_mm=position.tolist())
                if duration == escape:
                    photon["status"] = "escaped"
                    self.events.append(
                        dict(
                            event_type="escape",
                            creation_event_id=photon["creation_event_id"],
                            time_ns=float(new_time),
                            position_mm=position.tolist(),
                        )
                    )
                    break
                if total <= 0 or self.rng.random() >= total * duration:
                    continue
                draw = self.rng.random() * total
                event = dict(
                    creation_event_id=photon["creation_event_id"],
                    time_ns=float(new_time),
                    position_mm=position.tolist(),
                    probability=float(total * duration),
                )
                if draw < sum(rates):
                    selected = int(
                        np.searchsorted(np.cumsum(rates), draw, side="right")
                    )
                    target = eligible[selected]
                    out_energy, out_direction, out_momentum, data = compton_scatter(
                        energy, direction, target.momentum, target.mass, self.rng
                    )
                    target.momentum[:] = out_momentum
                    photon.update(
                        energy_native=out_energy,
                        direction=out_direction.tolist(),
                        polarisation=None,
                        status="scattered",
                    )
                    event.update(event_type="scattering", target=target.key, **data)
                else:
                    f, cdf, _ = breit_wheeler_spectrum(chi, energy)
                    fraction = float(np.interp(self.rng.random(), cdf, f))
                    photon["status"] = "converted"
                    event.update(
                        event_type="pair_conversion",
                        chi_gamma=chi,
                        **pair_kinematics(energy, direction, fraction),
                    )
                self.events.append(event)
                if photon["status"] == "converted":
                    break

    def to_payload(self) -> dict[str, Any]:
        return copy.deepcopy(
            dict(
                schema_version=1,
                config=asdict(self.config),
                rng_state=self.rng.bit_generator.state,
                photons=self.photons,
                events=self.events,
                emission_cursor=self.emission_cursor,
            )
        )
