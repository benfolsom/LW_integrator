"""Beam-current / macroparticle-weight conversions for bunched beams.

These helpers translate between an average beam current and the per-bunch
physical particle population, and from there to the per-macroparticle charge
weight (``stripped_ions``) used when initializing a simulated bunch.

The relation for a bunched beam at RF frequency ``f_RF`` (one bunch per RF
bucket) is::

    N_phys_per_bunch = I / (e * f_RF)
    stripped_ions    = N_phys_per_bunch / pcount

where ``e`` is the elementary charge in coulombs and ``I`` the average current
in amperes. ``stripped_ions`` is therefore the number of real charges each
simulated macroparticle represents.

Note: ``e`` here is the SI elementary charge in coulombs, used purely for the
current-to-count relation. It is distinct from
``core.constants.ELEMENTARY_CHARGE`` (the solver's native charge unit), which
is what individual particle charges are expressed in inside the integrator.
"""

from __future__ import annotations

ELEMENTARY_CHARGE_COULOMB: float = 1.602176634e-19
"""Elementary charge in coulombs (SI), for current/population conversions."""


def physical_population_per_bunch(current_a: float, rf_hz: float) -> float:
    """Physical particles per bunch for an average current at an RF frequency.

    Parameters
    ----------
    current_a:
        Average beam current in amperes.
    rf_hz:
        Bunch (RF) repetition frequency in hertz, one bunch per RF bucket.

    Returns
    -------
    float
        Number of real charges contained in one bunch.
    """
    if rf_hz <= 0.0:
        raise ValueError("rf_hz must be positive")
    return current_a / (ELEMENTARY_CHARGE_COULOMB * rf_hz)


def macro_weight_per_particle(
    current_a: float, rf_hz: float, pcount: int
) -> float:
    """Per-macroparticle charge weight (``stripped_ions``) for a bunched beam.

    Parameters
    ----------
    current_a:
        Average beam current in amperes.
    rf_hz:
        Bunch (RF) repetition frequency in hertz.
    pcount:
        Number of simulated macroparticles representing the bunch.

    Returns
    -------
    float
        Real charges represented by each macroparticle, suitable for the
        ``stripped_ions`` particle parameter.
    """
    if pcount <= 0:
        raise ValueError("pcount must be positive")
    return physical_population_per_bunch(current_a, rf_hz) / pcount


def current_from_macro_weight(
    stripped_ions: float, rf_hz: float, pcount: int
) -> float:
    """Inverse of :func:`macro_weight_per_particle`: recover average current.

    Parameters
    ----------
    stripped_ions:
        Real charges represented by each macroparticle.
    rf_hz:
        Bunch (RF) repetition frequency in hertz.
    pcount:
        Number of simulated macroparticles representing the bunch.

    Returns
    -------
    float
        Average beam current in amperes.
    """
    return stripped_ions * pcount * ELEMENTARY_CHARGE_COULOMB * rf_hz


__all__ = [
    "ELEMENTARY_CHARGE_COULOMB",
    "physical_population_per_bunch",
    "macro_weight_per_particle",
    "current_from_macro_weight",
]
