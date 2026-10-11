"""Opt-in, spin-averaged local quantum synchrotron emission.

Native units: mm, ns, amu, and Gaussian charge. See the documentation page
for the formation-length, beaming, and unresolved-background limitations.
No generator or spectrum is constructed on the disabled deterministic path.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any, cast

import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.special import gamma as gamma_function
from scipy.special import kv

from .constants import C_MMNS as C
from .constants import ELECTRON_MASS_AMU, ELEMENTARY_CHARGE
from .magnetic_dipole import HBAR_NATIVE


@dataclass(frozen=True)
class StochasticEmissionConfig:
    """Single-particle emission controls; defaults never alter deterministic runs."""

    enabled: bool = False
    chi_threshold: float = 0.1
    recoil_threshold: float = 0.1
    deactivation_fraction: float = 0.8
    rate_model: str = "quantum_lcfa"
    seed: int = 0
    max_probability: float = 0.05
    test_force_next_emission_fraction: float | None = None

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise TypeError("stochastic emission enabled must be boolean")
        for name in ("chi_threshold", "recoil_threshold", "max_probability"):
            value = getattr(self, name)
            if isinstance(value, (str, bool)) or not np.isfinite(value) or value <= 0:
                raise ValueError(
                    f"stochastic emission {name} must be positive and finite"
                )
        if self.max_probability > 0.1:
            raise ValueError("stochastic emission max_probability must be <= 0.1")
        if (
            isinstance(self.deactivation_fraction, (str, bool))
            or not np.isfinite(self.deactivation_fraction)
            or not 0 < self.deactivation_fraction <= 1
        ):
            raise ValueError(
                "stochastic emission deactivation_fraction must be in (0, 1]"
            )
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("stochastic emission seed must be a nonnegative integer")
        if self.rate_model not in ("quantum_lcfa", "classical_capped"):
            raise ValueError(
                "emission rate_model must be quantum_lcfa or classical_capped"
            )
        if self.test_force_next_emission_fraction is not None and (
            not np.isfinite(self.test_force_next_emission_fraction)
            or not 0 < self.test_force_next_emission_fraction <= 1
        ):
            raise ValueError("test forced emission fraction must be in (0, 1]")

    @classmethod
    def from_dict(cls, value: Any) -> StochasticEmissionConfig:
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            raise ValueError("stochastic_emission must be an object")
        return cls(**value)


def is_electron(mass: float, charge: float) -> bool:
    """Electron/positron rates do not silently apply to ions or macro charges."""
    return bool(
        np.isclose(mass, ELECTRON_MASS_AMU, rtol=1e-5, atol=0)
        and np.isclose(abs(charge), ELEMENTARY_CHARGE, rtol=1e-5, atol=0)
    )


def kinetic_energy(momentum: np.ndarray, mass: float) -> float:
    """Stable on-shell kinetic energy, including near-rest particles."""
    pnorm = float(np.linalg.norm(momentum))
    mc = mass * C
    return float(C * pnorm * (pnorm / (np.hypot(mc, pnorm) + mc)))


def local_indicators(
    momentum: np.ndarray, mass: float, rest_force: float
) -> dict[str, Any]:
    """Rest Lorentz-force invariant and the local formation-time proxy."""
    if (
        np.shape(momentum) != (3,)
        or not np.isfinite(momentum).all()
        or not np.isfinite([mass, rest_force]).all()
        or mass <= 0
        or rest_force < 0
    ):
        raise ValueError(
            "Finite momentum, positive mass, and nonnegative force required"
        )
    gamma = float(np.hypot(1.0, np.linalg.norm(momentum) / (mass * C)))
    chi = float(HBAR_NATIVE * rest_force / (mass**2 * C**3))
    characteristic = 1.5 * chi * gamma * mass * C**2
    kinetic = kinetic_energy(momentum, mass)
    # A resting particle has no available kinetic energy to radiate.
    recoil = characteristic / kinetic if kinetic > 0 else 0.0
    return {"chi": chi, "recoil_parameter": recoil, "gamma": gamma}


def indicators_from_force(
    momentum: np.ndarray, mass: float, force: np.ndarray
) -> dict[str, Any]:
    """Transform an ordinary lab three-force to the instantaneous rest frame."""
    force = np.asarray(force, dtype=float)
    if force.shape != (3,) or not np.isfinite(force).all():
        raise ValueError("Finite three-force required")
    norm = float(np.linalg.norm(momentum))
    direction = momentum / norm if norm else np.zeros(3)
    parallel = np.dot(force, direction) * direction
    gamma = np.hypot(1.0, norm / (mass * C))
    rest = parallel + gamma * (force - parallel)
    result = local_indicators(momentum, mass, float(np.linalg.norm(rest)))
    result["rest_force_direction"] = (
        (rest / np.linalg.norm(rest)).tolist()
        if np.linalg.norm(rest)
        else [0.0, 0.0, 1.0]
    )
    return result


def crosses(config: StochasticEmissionConfig, indicators: dict[str, Any]) -> bool:
    return bool(
        indicators["chi"] > config.chi_threshold
        or indicators["recoil_parameter"] > config.recoil_threshold
    )


def activation_state(
    config: StochasticEmissionConfig,
    indicators: dict[str, Any],
    active: bool,
    persistent_legacy: bool = False,
) -> bool:
    """Preview an accepted transition without changing state or consuming RNG."""
    if persistent_legacy:
        return bool(
            active
            or indicators["chi"] >= config.chi_threshold
            or indicators["recoil_parameter"] >= config.recoil_threshold
        )
    if not active:
        return crosses(config, indicators)
    h = config.deactivation_fraction
    return not (
        indicators["chi"] < h * config.chi_threshold
        and indicators["recoil_parameter"] < h * config.recoil_threshold
    )


def indicators_from_fields(
    momentum: np.ndarray,
    mass: float,
    charge: float,
    electric: np.ndarray,
    magnetic: np.ndarray,
) -> dict[str, Any]:
    gamma = np.hypot(1.0, np.linalg.norm(momentum) / (mass * C))
    beta = momentum / (gamma * mass * C)
    force = charge * (np.asarray(electric) + np.cross(beta, magnetic))
    return indicators_from_force(momentum, mass, force)


def sampled_lw_fields(
    nhat: dict[str, np.ndarray], samples: Any
) -> tuple[np.ndarray, np.ndarray]:
    """Standard LW E and n cross E, using the solver's retarded samples.

    Source bdot is d beta / d(ct), so the radiation term has no extra c.
    This retains the sampled-path limitations; it is not an exact root query.
    """
    n = np.stack([nhat[f"n{axis}"] for axis in "xyz"], axis=-1)
    beta = np.stack([getattr(samples, f"b{axis}") for axis in "xyz"], axis=-1)
    bdot = np.stack([getattr(samples, f"bdot{axis}") for axis in "xyz"], axis=-1)
    r = np.asarray(nhat["R"])
    kappa = 1 - np.einsum("ij,ij->i", n, beta)
    valid = samples.valid_mask & (r > 0) & (kappa > 0)
    n, beta, bdot, r, kappa = (v[valid] for v in (n, beta, bdot, r, kappa))
    electric = (
        samples.charge[valid, None]
        / kappa[:, None] ** 3
        * (
            (n - beta) / (samples.gamma[valid, None] ** 2 * r[:, None] ** 2)
            + np.cross(n, np.cross(n - beta, bdot)) / r[:, None]
        )
    )
    return np.sum(electric, axis=0), np.sum(np.cross(n, electric), axis=0)


def photon_direction(
    momentum: np.ndarray,
    mass: float,
    rest_force_direction: Any,
    rng: np.random.Generator,
) -> np.ndarray:
    """Collinear for gamma >= 10; otherwise aberrate a rest dipole pattern."""
    norm = np.linalg.norm(momentum)
    axis = momentum / norm
    gamma = np.hypot(1.0, norm / (mass * C))
    if gamma >= 10:
        return cast(np.ndarray, axis)
    acceleration = np.asarray(rest_force_direction, dtype=float)
    acceleration /= np.linalg.norm(acceleration)
    while True:
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        if rng.random() < 1 - np.dot(direction, acceleration) ** 2:
            break
    cosine = np.dot(direction, axis)
    beta = norm / (gamma * mass * C)
    lab = ((direction - cosine * axis) / gamma + (cosine + beta) * axis) / (
        1 + beta * cosine
    )
    return cast(np.ndarray, np.asarray(lab / np.linalg.norm(lab)))


@lru_cache(maxsize=1)
def _bessel_integral_table() -> tuple[np.ndarray, np.ndarray]:
    # Integral K_1/3 from z to infinity. A log grid resolves soft photons;
    # above 700 the contribution is numerically negligible.
    z = np.geomspace(1e-14, 700.0, 32769)
    tail = -cumulative_trapezoid(kv(1 / 3, z)[::-1], z[::-1], initial=0)[::-1]
    return np.log(z), tail


def bessel_integral(z: np.ndarray) -> np.ndarray:
    logs, values = _bessel_integral_table()
    return cast(
        np.ndarray,
        np.asarray(
            np.interp(np.log(z), logs, values, left=np.pi / np.sqrt(3), right=0.0)
        ),
    )


def lcfa_spectral_brackets(
    fraction: np.ndarray, chi: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the three dimensionless LCFA number-spectrum brackets.

    ``fraction`` is the photon fraction of total electron energy, ``s = k/E``.
    The returned arrays are, in order: spin-averaged quantum LCFA, the
    recoil-only scalar diagnostic, and the classical spectrum.  The first is
    the bracket sampled by ``quantum_lcfa``.  The latter two are exposed for
    cross-code diagnostics; neither changes the runner's model choices.

    This provides the direct mapping used by thread C's ``spectral_brackets``:
    its ``quantum``, ``scalar``, and ``classical`` outputs respectively map to
    these outputs.  All brackets are zero outside ``0 < s < 1``.
    """
    s = np.asarray(fraction, dtype=float)
    quantum = np.zeros_like(s)
    scalar = np.zeros_like(s)
    classical = np.zeros_like(s)
    if chi <= 0:
        return quantum, scalar, classical
    valid = (s > 0) & (s < 1)
    sv = s[valid]
    quantum_z = 2 * sv / (3 * chi * (1 - sv))
    quantum_k = kv(2 / 3, quantum_z)
    quantum_integral = bessel_integral(quantum_z)
    scalar[valid] = np.maximum(2 * quantum_k - quantum_integral, 0)
    quantum[valid] = np.maximum(
        (1 - sv + 1 / (1 - sv)) * quantum_k - quantum_integral, 0
    )
    classical_z = 2 * sv / (3 * chi)
    classical[valid] = np.maximum(
        2 * kv(2 / 3, classical_z) - bessel_integral(classical_z), 0
    )
    return quantum, scalar, classical


def differential_rate(
    fraction: np.ndarray,
    chi: float,
    gamma: float,
    mass: float,
    charge: float,
    model: str = "quantum_lcfa",
) -> np.ndarray:
    """dW/ds per lab ns, s = photon energy / total electron energy.

    Ritus/Baier–Katkov spin-averaged LCFA; classical_capped removes recoil
    from the Bessel argument and spin prefactor. The caller enforces k < T.
    """
    s = np.asarray(fraction, dtype=float)
    if model not in ("quantum_lcfa", "classical_capped"):
        raise ValueError("Unknown photon spectrum")
    out: np.ndarray = np.zeros_like(s)
    if chi <= 0:
        return out
    quantum, _, classical = lcfa_spectral_brackets(s, chi)
    alpha = charge**2 / (HBAR_NATIVE * C)
    prefactor = alpha * mass * C**2 / (np.sqrt(3) * np.pi * HBAR_NATIVE * gamma)
    out = prefactor * (quantum if model == "quantum_lcfa" else classical)
    return out


@dataclass(frozen=True)
class PhotonSpectrum:
    """Quadrature in t = z**(1/3), removing the soft-photon singularity."""

    t: np.ndarray
    fraction: np.ndarray
    cdf: np.ndarray
    rate_per_ns: float
    mean_fraction: float

    def sample_fraction(self, uniform: float) -> float:
        if self.rate_per_ns <= 0 or not 0 <= uniform < 1:
            raise ValueError("Nonzero spectrum and uniform in [0, 1) required")
        coordinate = float(np.interp(uniform, self.cdf, self.t))
        return float(np.interp(coordinate, self.t, self.fraction))


def photon_spectrum(
    chi: float,
    gamma: float,
    mass: float,
    charge: float,
    model: str = "quantum_lcfa",
    points: int = 2049,
) -> PhotonSpectrum:
    """Integrate the capped local number rate; no rounded-chi cache is used."""
    if (
        not np.isfinite([chi, gamma, mass, charge]).all()
        or chi < 0
        or gamma < 1
        or mass <= 0
        or points < 65
        or model not in ("quantum_lcfa", "classical_capped")
    ):
        raise ValueError("Invalid photon spectrum inputs")
    cap = (gamma - 1) / gamma
    if chi == 0 or cap <= 0:
        zero = np.zeros(2)
        return PhotonSpectrum(zero, zero.copy(), zero.copy(), 0.0, 0.0)
    quantum = model == "quantum_lcfa"
    zmax = min(600.0, 2 * cap / (3 * chi * (1 - cap if quantum else 1)))
    t = np.linspace(0, np.cbrt(zmax), points)
    z = t**3
    s = 1.5 * chi * z / (1 + 1.5 * chi * z if quantum else 1)
    jacobian = 4.5 * chi * t**2 / ((1 + 1.5 * chi * z) ** 2 if quantum else 1)
    density = np.empty_like(t)
    density[1:] = (
        differential_rate(s[1:], chi, gamma, mass, charge, model) * jacobian[1:]
    )
    alpha = charge**2 / (HBAR_NATIVE * C)
    prefactor = alpha * mass * C**2 / (np.sqrt(3) * np.pi * HBAR_NATIVE * gamma)
    density[0] = prefactor * 9 * chi * 2 ** (-1 / 3) * gamma_function(2 / 3)
    integrated = cumulative_trapezoid(density, t, initial=0)
    rate = float(integrated[-1])
    mean = float(trapezoid(density * s, t) / rate) if rate > 0 else 0.0
    return PhotonSpectrum(t, s, integrated / rate if rate else integrated, rate, mean)


def quantum_power_suppression(chi: float) -> float:
    """LCFA Gaunt factor g(chi) = quantum power / classical Larmor power.

    Integrate the same Ritus/Baier–Katkov bracket as ``photon_spectrum`` over
    its full quantum support. No fitted or rounded-chi approximation is used.
    This is the usual quantum correction to continuous LL reaction, applied
    here to the selected classical reaction below stochastic activation.
    The finite kinetic-energy cap remains a separate emission limitation.
    """
    if not np.isfinite(chi) or chi < 0:
        raise ValueError("Finite nonnegative chi required")
    if chi == 0:
        return 1.0
    t = np.linspace(0, np.cbrt(600.0), 2049)
    z = t**3
    fraction = 1.5 * chi * z / (1 + 1.5 * chi * z)
    jacobian = 4.5 * chi * t**2 / (1 + 1.5 * chi * z) ** 2
    power_density = np.zeros_like(t)
    power_density[1:] = (
        lcfa_spectral_brackets(fraction[1:], chi)[0] * jacobian[1:] * fraction[1:]
    )
    return float(trapezoid(power_density, t) / (2 * np.pi * chi**2 / np.sqrt(3)))


def recoil(
    momentum: np.ndarray,
    mass: float,
    energy: float,
    direction: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Exact on-shell energy loss, with unresolved background momentum recorded."""
    p = np.asarray(momentum, dtype=float)
    kinetic = kinetic_energy(p, mass)
    if not np.isfinite(energy) or not 0 < energy <= kinetic:
        raise ValueError(
            "Photon energy must be positive and <= available kinetic energy"
        )
    direction = (
        p / np.linalg.norm(p)
        if direction is None
        else np.asarray(direction, dtype=float)
    )
    if (
        direction.shape != (3,)
        or not np.isfinite(direction).all()
        or not np.isclose(np.linalg.norm(direction), 1.0, rtol=0, atol=1e-12)
    ):
        raise ValueError("Photon direction must be a finite unit vector")
    remaining = max(0.0, kinetic - energy)
    norm = np.sqrt(remaining) * np.sqrt(remaining + 2 * mass * C**2) / C
    residual = p - energy / C * direction
    after = norm * residual / np.linalg.norm(residual)
    before_four = np.r_[np.hypot(mass * C, np.linalg.norm(p)), p]
    after_four = np.r_[np.hypot(mass * C, norm), after]
    photon = np.r_[energy / C, energy / C * direction]
    background = after_four + photon - before_four
    return after, {
        "energy_native": float(energy),
        "kinetic_before_native": float(kinetic),
        "direction": direction.tolist(),
        "polarisation": None,
        "particle_before_four_momentum_native": before_four.tolist(),
        "particle_after_four_momentum_native": after_four.tolist(),
        "photon_four_momentum_native": photon.tolist(),
        "background_impulse_four_native": background.tolist(),
        "remainder_receiver": "unresolved_field_source_system",
    }


class EmissionRuntime:
    """Accepted-step state only; payloads are detached and JSON serialisable."""

    def __init__(self, config: StochasticEmissionConfig, payload: dict | None = None):
        self.config = config
        self.rng = np.random.Generator(np.random.PCG64(config.seed))
        self.particles: dict[str, dict[str, Any]] = {}
        self.events: list[dict[str, Any]] = []
        self.persistent_legacy = False
        if payload is not None:
            saved_config = dict(payload.get("config", {}))
            self.persistent_legacy = (
                "deactivation_fraction" not in saved_config
                or payload.get("activation_policy") == "persistent_legacy"
            )
            saved_config.setdefault("deactivation_fraction", 0.8)
            saved_config.setdefault("test_force_next_emission_fraction", None)
            if payload.get("schema_version") != 1 or saved_config != asdict(config):
                raise ValueError("Incompatible stochastic emission checkpoint")
            self.rng.bit_generator.state = copy.deepcopy(payload["rng_state"])
            self.particles = copy.deepcopy(payload["particles"])
            self.events = copy.deepcopy(payload["events"])

    def active(self, key: str) -> bool:
        return bool(self.particles.get(key, {}).get("active", False))

    def observe(self, key: str, indicators: dict[str, Any], time_ns: float) -> bool:
        row = self.particles.setdefault(key, {"active": False, "first_crossing": None})
        row["last"] = dict(indicators)
        row["max_chi"] = max(row.get("max_chi", 0.0), indicators["chi"])
        row["max_recoil_parameter"] = max(
            row.get("max_recoil_parameter", 0.0), indicators["recoil_parameter"]
        )
        active = activation_state(
            self.config, indicators, row["active"], self.persistent_legacy
        )
        if active != row["active"]:
            row["active"] = active
            crossing = {"time_ns": float(time_ns), **indicators}
            if active and row["first_crossing"] is None:
                row["first_crossing"] = crossing
            self.events.append(
                {
                    "event_type": "activation" if active else "deactivation",
                    "particle": key,
                    **crossing,
                }
            )
        return bool(row["active"])

    def emit(
        self,
        key: str,
        momentum: np.ndarray,
        mass: float,
        charge: float,
        indicators: dict[str, Any],
        dt_ns: float,
        time_ns: float,
        position: np.ndarray,
    ) -> np.ndarray:
        """Subdivide an accepted interval into bounded Bernoulli emission trials.

        The runner's field indicators and endpoint position are held fixed.
        After each recoil, rebuild the spectrum with the updated momentum;
        repartition the remaining lab time so every W dt stays below the cap.
        All events are published at the accepted orbit endpoint, as before.
        """
        if not np.isfinite(dt_ns) or dt_ns <= 0:
            raise ValueError("Positive finite lab emission interval required")
        if not self.config.enabled or not is_electron(mass, charge):
            return momentum
        if not self.observe(key, indicators, time_ns):
            return momentum
        remaining = float(dt_ns)
        after = momentum
        spectrum = None
        subintervals = 0
        while remaining > 0:
            if spectrum is None:
                gamma = float(np.hypot(1.0, np.linalg.norm(after) / (mass * C)))
                spectrum = photon_spectrum(
                    indicators["chi"], gamma, mass, charge, self.config.rate_model
                )
            interval_probability = spectrum.rate_per_ns * remaining
            if subintervals == 0:
                self.particles[key]["last_interval_probability"] = float(
                    interval_probability
                )
            count = max(
                1, int(np.ceil(interval_probability / self.config.max_probability))
            )
            duration = remaining / count
            probability = spectrum.rate_per_ns * duration
            # Roundoff must not place a sub-interval above the configured cap.
            if probability > self.config.max_probability:
                count += 1
                duration = remaining / count
                probability = spectrum.rate_per_ns * duration
            self.particles[key]["last_probability"] = float(probability)
            subintervals += 1
            remaining = 0.0 if count == 1 else remaining - duration
            force_fraction = self.config.test_force_next_emission_fraction
            forced = force_fraction is not None and not self.particles[key].get(
                "test_forced_emission_used", False
            )
            if not forced and (probability <= 0 or self.rng.random() >= probability):
                continue
            fraction = (
                float(force_fraction)
                if forced
                else spectrum.sample_fraction(float(self.rng.random()))
            )
            if forced:
                self.particles[key]["test_forced_emission_used"] = True
            energy = min(
                fraction * gamma * mass * C**2,
                np.nextafter(kinetic_energy(after, mass), 0),
            )
            if energy <= 0:
                continue
            direction = photon_direction(
                after, mass, indicators.get("rest_force_direction", [0, 0, 1]), self.rng
            )
            after, event = recoil(after, mass, energy, direction)
            event.update(
                event_type="photon",
                particle=key,
                time_ns=float(time_ns),
                position_mm=np.asarray(position).tolist(),
                chi=float(indicators["chi"]),
                rate_model=self.config.rate_model,
            )
            self.events.append(event)
            spectrum = None
        self.particles[key]["last_subintervals"] = subintervals
        return after

    def to_payload(self) -> dict[str, Any]:
        return copy.deepcopy(
            {
                "schema_version": 1,
                "activation_policy": (
                    "persistent_legacy" if self.persistent_legacy else "hysteresis"
                ),
                "config": asdict(self.config),
                "rng_state": self.rng.bit_generator.state,
                "particles": self.particles,
                "events": self.events,
            }
        )


def prepare_general_state(
    state: dict[str, Any], role: str, runtime: EmissionRuntime
) -> None:
    """Attach immutable trial settings; stochastic state stays at the barrier."""
    state["_stochastic_emission_config"] = asdict(runtime.config)
    state["_stochastic_emission_persistent_legacy"] = runtime.persistent_legacy
    state["_stochastic_emission_active"] = np.array(
        [runtime.active(f"{role}:{i}") for i in range(len(state["x"]))], dtype=bool
    )


def commit_general_state(
    previous: dict[str, Any], state: dict[str, Any], role: str, runtime: EmissionRuntime
) -> None:
    """Recoil at a fixed-step accepted endpoint, preserving canonical offsets."""
    samples = state.get("_stochastic_emission_samples")
    if samples is None:
        raise RuntimeError("Accepted step did not provide stochastic emission samples")
    for i, sample in enumerate(samples):
        if sample is None:
            continue
        if bool(state.get("_dead_particles", np.zeros(len(samples), dtype=bool))[i]):
            continue
        momentum = np.asarray(sample["mechanical_momentum"], dtype=float)
        mass = float(state["m"][i])
        charge = float(state.get("q_observer", state["q"])[i])
        dt = float(state["t"][i] - previous["t"][i])
        after = runtime.emit(
            f"{role}:{i}",
            momentum,
            mass,
            charge,
            sample["indicators"],
            dt,
            float(state["t"][i]),
            np.array([state[axis][i] for axis in "xyz"]),
        )
        if np.array_equal(momentum, after):
            continue
        # Reuse the accepted-force restart used for synthetic/gated branches.
        # The next continuous interval must prime a post-kick sample.
        from .integration_runner import _clear_medina_force_history

        mask = np.arange(len(samples)) == i
        _clear_medina_force_history(state, particle_mask=mask)
        before_pt = np.hypot(mass * C, np.linalg.norm(momentum))
        after_pt = np.hypot(mass * C, np.linalg.norm(after))
        state["Pt"][i] += after_pt - before_pt
        state["gamma"][i] = after_pt / (mass * C)
        proper = after / (mass * C)
        magnitude = np.linalg.norm(proper)
        for j, axis in enumerate("xyz"):
            state[f"P{axis}"][i] += after[j] - momentum[j]
            state[f"b{axis}"][i] = after[j] / after_pt
            if f"source_u_{axis}" in state:
                state[f"source_u_{axis}"][i] = proper[j]
        if "source_speed_deficit" in state:
            state["source_speed_deficit"][i] = 1 / (
                state["gamma"][i] * (state["gamma"][i] + magnitude)
            )
        if "source_kinematics_ready" in state:
            state["source_kinematics_ready"][i] = 1.0
        state["radiation_energy_applied"][i] = C * (before_pt - after_pt)
    state.pop("_stochastic_emission_samples", None)
