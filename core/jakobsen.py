"""Experimental ordinary linear-spin response, Jakobsen (2024), Eqs. 6-8.

Native amu/mm/ns scaled-Gaussian units. Inputs are physical spin angular
momentum, NOT unit polarization. No self-force, susceptibility or S^2 terms.
This local kernel does not select a production integration mode.
"""

from dataclasses import dataclass
from itertools import permutations

import numpy as np

from .constants import C_MMNS

_SIGNS = np.array([1.0, -1.0, -1.0, -1.0])
_EPS = np.zeros((4, 4, 4, 4))
for _indices in permutations(range(4)):
    _parity = sum(_indices[i] > _indices[j] for i in range(4) for j in range(i + 1, 4))
    _EPS[_indices] = -((-1.0) ** _parity)


def _dot(a, b):
    return (a * _SIGNS) @ b


def _dual(field):
    return 0.5 * np.einsum("abcd,c,d,cd->ab", _EPS, _SIGNS, _SIGNS, field)


def _cross(a, b, u):
    return np.einsum("abcd,b,c,d->a", _EPS, _SIGNS * a, _SIGNS * b, _SIGNS * u)


@dataclass(frozen=True)
class JakobsenResponse:
    """Force dp/dtau, spin dS/dtau, and canonical spin offset in native units."""

    four_force: np.ndarray
    spin_four_rate: np.ndarray
    spin_momentum_offset: np.ndarray
    interaction_energy: float
    spin_momentum_offset_rate: np.ndarray
    charge_radiative_balance_correction: np.ndarray
    spin_four_force: np.ndarray


def ordinary_response_native(
    *,
    four_velocity_mm_ns,
    spin_angular_momentum,
    field_tensor,
    partial_f,
    charge_native,
    mass_amu,
    g,
    leading_charge_reaction_four_force_native=None,
):
    """Evaluate the order-reduced ordinary equation and matching momentum.

    Inside explicitly spin-linear terms use leading Lorentz acceleration and
    BMT precession. Replacing them with spin-corrected values adds S^2 terms.
    The spin rate here is strictly linear: higher-order constraint transport
    must not be mistaken for an independently derived S^2 torque.
    """
    u, spin, field, gradient = [
        np.asarray(x, dtype=float)
        for x in (four_velocity_mm_ns, spin_angular_momentum, field_tensor, partial_f)
    ]
    if any(
        x.shape != shape
        for x, shape in zip((u, spin, field, gradient), ((4,), (4,), (4, 4), (4, 4, 4)))
    ):
        raise ValueError("Invalid velocity, spin or derivative shapes")
    if not all(np.isfinite(x).all() for x in (u, spin, field, gradient)):
        raise ValueError("Finite response inputs required")
    if not np.isfinite([charge_native, mass_amu, g]).all() or mass_amu <= 0:
        raise ValueError("Finite coefficients and positive mass required")
    u = u / C_MMNS
    if u[0] <= 0 or abs(_dot(u, u) - 1) > 1e-11 * (u @ u):
        raise ValueError("Future mass-shell velocity required")
    if abs(_dot(spin, u)) > 1e-11 * max(
        np.linalg.norm(spin) * np.linalg.norm(u), 1e-300
    ):
        raise ValueError("Spin must be transverse to velocity")
    for tensor in (field, gradient):
        if not np.allclose(tensor, -tensor.swapaxes(-1, -2), rtol=0, atol=1e-15):
            raise ValueError("Antisymmetric field and gradient required")
    # c=1 length-time variables: tau_length=c*tau_ns; retain mm and amu.
    spin = spin / C_MMNS
    field = field / C_MMNS**2
    gradient = gradient / C_MMNS**2
    extra = (
        np.zeros(4)
        if leading_charge_reaction_four_force_native is None
        else np.asarray(leading_charge_reaction_four_force_native, dtype=float)
        / C_MMNS**2
    )
    if extra.shape != (4,) or not np.isfinite(extra).all():
        raise ValueError("Finite leading charge reaction four-force required")
    if abs(_dot(u, extra)) > 1e-11 * max(
        np.linalg.norm(u) * np.linalg.norm(extra), 1e-300
    ):
        raise ValueError("Leading charge reaction must be transverse")
    return _response_algebra(
        u, spin, field, gradient, charge_native, mass_amu, g, extra
    )


def _response_algebra(u, spin, field, gradient, charge_native, mass_amu, g, extra):
    """Unchecked c=1 algebra; complex inputs support directional differentiation."""
    kappa = charge_native / mass_amu
    alpha = g * kappa / 2
    electric = field @ (_SIGNS * u)
    magnetic = -_dual(field) @ (_SIGNS * u)
    a0 = kappa * electric + extra / mass_amu
    spin_rate = alpha * field @ (_SIGNS * spin) + (alpha - kappa) * u * _dot(
        spin, electric
    )
    spin_rate -= u * _dot(spin, extra / mass_amu)
    energy = alpha * _dot(spin, magnetic)
    energy_gradient = np.array(
        [alpha * _dot(spin, -_dual(entry) @ (_SIGNS * u)) for entry in gradient]
    )
    field_rate = np.einsum("a,aij->ij", u, gradient)
    electric_rate = field_rate @ (_SIGNS * u) + field @ (_SIGNS * a0)
    cross_rate = (
        _cross(spin_rate, electric, u)
        + _cross(spin, electric_rate, u)
        + _cross(spin, electric, a0)
    )

    def projected(v):
        return v - u * _dot(u, v)

    spin_force = (
        projected(_SIGNS * energy_gradient)
        - energy * a0
        - (alpha - kappa) * projected(cross_rate)
    )
    force = charge_native * electric + extra + spin_force
    offset = energy * u + (alpha - kappa) * _cross(spin, electric, u)
    magnetic_rate = -_dual(field_rate) @ (_SIGNS * u) - _dual(field) @ (_SIGNS * a0)
    energy_rate = alpha * (_dot(spin_rate, magnetic) + _dot(spin, magnetic_rate))
    offset_rate = energy_rate * u + energy * a0 + (alpha - kappa) * cross_rate
    # This is an energy/momentum balance contribution, NOT an extra force.
    balance = u / mass_amu * _dot(spin, _cross(kappa * electric, extra, u))
    return JakobsenResponse(
        force * C_MMNS**2,
        spin_rate * C_MMNS**2,
        offset * C_MMNS,
        energy * C_MMNS**2,
        offset_rate * C_MMNS**2,
        balance * C_MMNS**2,
        spin_force * C_MMNS**2,
    )


def ordinary_force_rate_native(
    *,
    four_velocity_mm_ns,
    spin_angular_momentum,
    field_tensor,
    partial_f,
    partial_f_proper_rate,
    charge_native,
    mass_amu,
    g,
):
    """Differentiate the ordinary force through first order in spin.

    The caller supplies d(partial_F)/d(tau) along the leading worldline.
    Complex-step differentiation evaluates local polynomial algebra only;
    it never samples a displaced observer or a future source history.
    """
    leading, spin = _ordinary_force_rate_parts_native(
        four_velocity_mm_ns=four_velocity_mm_ns,
        spin_angular_momentum=spin_angular_momentum,
        field_tensor=field_tensor,
        partial_f=partial_f,
        partial_f_proper_rate=partial_f_proper_rate,
        charge_native=charge_native,
        mass_amu=mass_amu,
        g=g,
    )
    return leading + spin


def _ordinary_force_rate_parts_native(
    *,
    four_velocity_mm_ns,
    spin_angular_momentum,
    field_tensor,
    partial_f,
    partial_f_proper_rate,
    charge_native,
    mass_amu,
    g,
):
    """Separate charge/spin derivatives without subtracting large charge terms."""
    response = ordinary_response_native(
        four_velocity_mm_ns=four_velocity_mm_ns,
        spin_angular_momentum=spin_angular_momentum,
        field_tensor=field_tensor,
        partial_f=partial_f,
        charge_native=charge_native,
        mass_amu=mass_amu,
        g=g,
    )
    u = np.asarray(four_velocity_mm_ns, dtype=float)
    spin = np.asarray(spin_angular_momentum, dtype=float)
    field = np.asarray(field_tensor, dtype=float)
    gradient = np.asarray(partial_f, dtype=float)
    gradient_rate = np.asarray(partial_f_proper_rate, dtype=float)
    if gradient_rate.shape != (4, 4, 4) or not np.isfinite(gradient_rate).all():
        raise ValueError("Finite proper-time field-gradient derivative required")
    if not np.allclose(
        gradient_rate, -gradient_rate.swapaxes(-1, -2), rtol=0, atol=1e-15
    ):
        raise ValueError("Antisymmetric field-gradient derivative required")
    leading_force = charge_native / C_MMNS * field @ (_SIGNS * u)
    leading_acceleration = leading_force / mass_amu
    field_rate = np.einsum("a,aij->ij", u, gradient)
    leading_rate = (
        charge_native
        / C_MMNS
        * (field_rate @ (_SIGNS * u) + field @ (_SIGNS * leading_acceleration))
    )
    spin_force = response.spin_four_force
    # Include spin acceleration in the charge force derivative, but never
    # inside a term already proportional to spin.
    charge_spin_rate = (
        charge_native / (C_MMNS * mass_amu) * field @ (_SIGNS * spin_force)
    )
    eps = 1e-24
    uc = (u + 1j * eps * leading_acceleration) / C_MMNS
    fc = (field + 1j * eps * field_rate) / C_MMNS**2
    differentiated = _response_algebra(
        uc,
        (spin + 1j * eps * response.spin_four_rate) / C_MMNS,
        fc,
        (gradient + 1j * eps * gradient_rate) / C_MMNS**2,
        charge_native,
        mass_amu,
        g,
        np.zeros(4),
    )
    spin_rate = differentiated.spin_four_force.imag / eps
    return leading_rate, charge_spin_rate + spin_rate


def canonical_momentum_native(
    *,
    four_velocity_mm_ns,
    rest_spin_angular_momentum,
    four_potential,
    field_tensor,
    charge_native,
    mass_amu,
    g,
):
    """P = m u + q A/c + Q for this model only; never apply Q to RFS states."""
    u = np.asarray(four_velocity_mm_ns, dtype=float)
    rest = np.asarray(rest_spin_angular_momentum, dtype=float)
    potential = np.asarray(four_potential, dtype=float)
    if rest.shape != (3,) or potential.shape != (4,) or u.shape != (4,):
        raise ValueError("Invalid canonical input shape")
    if not np.isfinite(rest).all() or not np.isfinite(potential).all():
        raise ValueError("Finite canonical inputs required")
    w = u[1:] / C_MMNS
    s0 = w @ rest
    spin = np.r_[s0, rest + w * s0 / (1 + u[0] / C_MMNS)]
    response = ordinary_response_native(
        four_velocity_mm_ns=u,
        spin_angular_momentum=spin,
        field_tensor=field_tensor,
        partial_f=np.zeros((4, 4, 4)),
        charge_native=charge_native,
        mass_amu=mass_amu,
        g=g,
    )
    return (
        mass_amu * u
        + charge_native * potential / C_MMNS
        + response.spin_momentum_offset
    )


def velocity_from_canonical_spatial_native(
    *,
    canonical_spatial_momentum,
    rest_spin_angular_momentum,
    four_potential,
    field_tensor,
    charge_native,
    mass_amu,
    g,
):
    """Invert the spatial momentum through first order in spin, not exactly.

    Q is evaluated at the zero-spin inverse velocity; the omitted inversion
    correction is S^2. Temporal canonical momentum is dependent and must be
    rebuilt, not interpreted as m*gamma*c or enforced as a second velocity.
    """
    p = np.asarray(canonical_spatial_momentum, dtype=float)
    a = np.asarray(four_potential, dtype=float)
    if (
        p.shape != (3,)
        or a.shape != (4,)
        or not np.isfinite(p).all()
        or not np.isfinite(a).all()
    ):
        raise ValueError("Finite spatial momentum and four-potential required")
    if not np.isfinite(mass_amu) or mass_amu <= 0:
        raise ValueError("Positive mass required")
    mechanical = p - charge_native * a[1:] / C_MMNS
    spatial = mechanical / mass_amu
    u0 = np.r_[np.sqrt(C_MMNS**2 + spatial @ spatial), spatial]
    canonical = canonical_momentum_native(
        four_velocity_mm_ns=u0,
        rest_spin_angular_momentum=rest_spin_angular_momentum,
        four_potential=a,
        field_tensor=field_tensor,
        charge_native=charge_native,
        mass_amu=mass_amu,
        g=g,
    )
    spatial -= (canonical[1:] - p) / mass_amu
    return np.r_[np.sqrt(C_MMNS**2 + spatial @ spatial), spatial]
