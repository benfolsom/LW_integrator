"""Immutable particle-species data used by magnetic-moment models.

The free-particle masses and magnetic moments below use the 2022 CODATA
adjustment published as Rev. Mod. Phys. 97, 025002 (2025):
https://doi.org/10.1103/RevModPhys.97.025002.  The directly measured
antiproton moment is from the BASE collaboration:
https://doi.org/10.1038/nature24048.

Magnetic moments are signed projections for the maximally polarized state,
in joules per tesla.  The sign is physically important: it says whether the
moment is parallel or antiparallel to the spin.  ``spin_quantum_number`` is
stored separately so callers do not silently treat every species as spin-1/2.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Dict, Mapping, Optional, Tuple

from .external_fields import AMU_KG, ELEMENTARY_CHARGE_COULOMB

CODATA_2022_URL = "https://doi.org/10.1103/RevModPhys.97.025002"
ANTIPROTON_MOMENT_URL = "https://doi.org/10.1038/nature24048"
HBAR_J_S = 1.054571817e-34


@dataclass(frozen=True)
class ParticleSpecies:
    """A read-only free-particle species preset.

    ``magnetic_moment_j_t=None`` means that this registry deliberately does
    not provide a usable moment.  It must not be interpreted as zero.
    ``magnetic_moment_j_t=0.0`` instead denotes a supported zero moment, as
    for the spin-zero alpha-particle ground state.
    """

    name: str
    display_name: str
    aliases: Tuple[str, ...]
    mass_amu: float
    charge_e: int
    spin_quantum_number: float
    magnetic_moment_j_t: Optional[float]
    moment_status: str
    reference_url: str
    note: str = ""
    conjugate_name: Optional[str] = None

    @property
    def charge_coulomb(self) -> float:
        """Return the single-particle charge in coulombs."""

        return self.charge_e * ELEMENTARY_CHARGE_COULOMB

    @property
    def mass_kg(self) -> float:
        """Return the single-particle mass in kilograms."""

        return self.mass_amu * AMU_KG

    @property
    def has_supported_magnetic_moment(self) -> bool:
        """Whether this preset supplies a moment for dynamics."""

        return self.magnetic_moment_j_t is not None

    @property
    def gyromagnetic_ratio_rad_s_t(self) -> float:
        """Return signed ``mu/(I hbar)`` in rad/s/T.

        Raises:
            ValueError: if the preset deliberately has no supported moment.
        """

        if self.magnetic_moment_j_t is None:
            raise ValueError(
                "Species {!r} has no supported magnetic moment preset".format(self.name)
            )
        if self.spin_quantum_number == 0.0:
            if self.magnetic_moment_j_t == 0.0:
                return 0.0
            raise ValueError("A nonzero moment cannot be assigned to zero spin")
        return self.magnetic_moment_j_t / (self.spin_quantum_number * HBAR_J_S)


_SPECIES_DATA = (
    ParticleSpecies(
        name="electron",
        display_name="Electron",
        aliases=("e-", "electron-"),
        mass_amu=5.485799090441e-4,
        charge_e=-1,
        spin_quantum_number=0.5,
        magnetic_moment_j_t=-9.2847646917e-24,
        moment_status="codata_2022_free_particle",
        reference_url=CODATA_2022_URL,
        conjugate_name="positron",
    ),
    ParticleSpecies(
        name="positron",
        display_name="Positron",
        aliases=("e+", "antielectron"),
        mass_amu=5.485799090441e-4,
        charge_e=1,
        spin_quantum_number=0.5,
        magnetic_moment_j_t=9.2847646917e-24,
        moment_status="cpt_conjugate_of_codata_2022_electron",
        reference_url=CODATA_2022_URL,
        note="Mass and moment magnitude use the electron values with CPT-conjugate signs.",
        conjugate_name="electron",
    ),
    ParticleSpecies(
        name="proton",
        display_name="Proton",
        aliases=("p", "p+"),
        mass_amu=1.0072764665789,
        charge_e=1,
        spin_quantum_number=0.5,
        magnetic_moment_j_t=1.41060679545e-26,
        moment_status="codata_2022_free_particle",
        reference_url=CODATA_2022_URL,
        conjugate_name="antiproton",
    ),
    ParticleSpecies(
        name="antiproton",
        display_name="Antiproton",
        aliases=("pbar", "anti-proton", "p-"),
        mass_amu=1.0072764665789,
        charge_e=-1,
        spin_quantum_number=0.5,
        magnetic_moment_j_t=-1.41060679519e-26,
        moment_status="base_2017_measured_free_particle",
        reference_url=ANTIPROTON_MOMENT_URL,
        note=(
            "Converted from the measured signed ratio "
            "mu_antiproton/mu_N=-2.7928473441."
        ),
        conjugate_name="proton",
    ),
    ParticleSpecies(
        name="neutron",
        display_name="Neutron",
        aliases=("n",),
        mass_amu=1.00866491606,
        charge_e=0,
        spin_quantum_number=0.5,
        magnetic_moment_j_t=-9.6623653e-27,
        moment_status="codata_2022_free_particle",
        reference_url=CODATA_2022_URL,
    ),
    ParticleSpecies(
        name="deuteron",
        display_name="Deuteron",
        aliases=("d", "deuterium_nucleus", "h-2_nucleus"),
        mass_amu=2.013553212544,
        charge_e=1,
        spin_quantum_number=1.0,
        magnetic_moment_j_t=4.330735087e-27,
        moment_status="codata_2022_free_particle",
        reference_url=CODATA_2022_URL,
    ),
    ParticleSpecies(
        name="triton",
        display_name="Triton",
        aliases=("t", "tritium_nucleus", "h-3_nucleus"),
        mass_amu=3.01550071597,
        charge_e=1,
        spin_quantum_number=0.5,
        magnetic_moment_j_t=1.5046095178e-26,
        moment_status="codata_2022_free_particle",
        reference_url=CODATA_2022_URL,
    ),
    ParticleSpecies(
        name="helion",
        display_name="Helium-3 nucleus (helion)",
        aliases=("helium-3_nucleus", "he3_nucleus", "3he++"),
        mass_amu=3.014932246932,
        charge_e=2,
        spin_quantum_number=0.5,
        magnetic_moment_j_t=-1.07461755198e-26,
        moment_status="codata_2022_free_particle",
        reference_url=CODATA_2022_URL,
        note="This is the free helion moment, not the bound or shielded value.",
    ),
    ParticleSpecies(
        name="alpha",
        display_name="Alpha particle (helium-4 nucleus)",
        aliases=("alpha_particle", "helium-4_nucleus", "he4_nucleus", "4he++"),
        mass_amu=4.001506179129,
        charge_e=2,
        spin_quantum_number=0.0,
        magnetic_moment_j_t=0.0,
        moment_status="spin_zero_ground_state",
        reference_url=CODATA_2022_URL,
    ),
    ParticleSpecies(
        name="h_minus",
        display_name="Negative hydrogen ion (H-)",
        aliases=("h-", "hydrogen_negative_ion", "hydride"),
        mass_amu=1.0072764665789 + 2.0 * 5.485799090441e-4,
        charge_e=-1,
        spin_quantum_number=0.5,
        magnetic_moment_j_t=None,
        moment_status="unsupported_composite_bound_state",
        reference_url=CODATA_2022_URL,
        note=(
            "Approximate constituent-sum mass (binding energy omitted). No effective "
            "H- magnetic moment is assumed; supply a documented bound-state model."
        ),
    ),
)


def _normalize_species_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def _build_registries() -> Tuple[Mapping[str, ParticleSpecies], Mapping[str, str]]:
    species: Dict[str, ParticleSpecies] = {}
    aliases: Dict[str, str] = {}
    for item in _SPECIES_DATA:
        canonical = _normalize_species_name(item.name)
        if canonical in species:
            raise RuntimeError("Duplicate canonical species name: {}".format(canonical))
        species[canonical] = item
        for alias in (item.name,) + item.aliases:
            normalized = _normalize_species_name(alias)
            previous = aliases.get(normalized)
            if previous is not None and previous != canonical:
                raise RuntimeError(
                    "Species alias {!r} maps to both {!r} and {!r}".format(
                        alias, previous, canonical
                    )
                )
            aliases[normalized] = canonical
    return MappingProxyType(species), MappingProxyType(aliases)


SPECIES, _SPECIES_ALIASES = _build_registries()
"""Read-only mapping from canonical names to :class:`ParticleSpecies`."""


def get_species(name: str) -> ParticleSpecies:
    """Look up a species by canonical name or documented alias.

    Raises:
        KeyError: if the name is not present in the finite preset registry.
    """

    normalized = _normalize_species_name(name)
    try:
        canonical = _SPECIES_ALIASES[normalized]
    except KeyError as exc:
        choices = ", ".join(SPECIES)
        raise KeyError(
            "Unknown particle species {!r}; available presets: {}".format(name, choices)
        ) from exc
    return SPECIES[canonical]


def resolve_species(name: str) -> ParticleSpecies:
    """Integration-facing alias for :func:`get_species`."""

    return get_species(name)


def list_species(
    *, require_supported_moment: bool = False
) -> Tuple[ParticleSpecies, ...]:
    """Return the immutable presets, optionally excluding unsupported moments."""

    if not require_supported_moment:
        return tuple(SPECIES.values())
    return tuple(
        item for item in SPECIES.values() if item.has_supported_magnetic_moment
    )


__all__ = [
    "ANTIPROTON_MOMENT_URL",
    "CODATA_2022_URL",
    "HBAR_J_S",
    "ParticleSpecies",
    "SPECIES",
    "get_species",
    "list_species",
    "resolve_species",
]
