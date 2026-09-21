import numpy as np
import pytest

from core import momentum_center as m
from core.full_dipole_self_field import regular_self_field, reaction_sectors


def rest():
    u = np.zeros((5, 4))
    u[0, 0] = 1
    return u, np.zeros((5, 4, 4))


def test_coasting_constant_full_dipole_has_no_self_field():
    u, d = rest()
    u[0] = [1.25, 0.75, 0, 0]
    d[0] = m.unpack(np.arange(6) * 0.1)
    result = regular_self_field(u, d, charge=0.4)
    np.testing.assert_array_equal(result["field"], 0)
    np.testing.assert_array_equal(result["gradient"], 0)


def test_charge_recovers_lad_instantaneous_rest_frame():
    u, d = rest()
    a = np.array([0.0, 0.2, -0.3, 0.1])
    jerk = np.array([np.dot(a[1:], a[1:]), 0.4, 0.1, -0.2])
    u[1], u[2] = a, jerk / 2
    q = 0.7
    result = regular_self_field(u, d, charge=q)
    force = q * result["field"] @ (m.METRIC * u[0])
    expected = 2 * q * q / 3 * (jerk + m.dot(a, a) * u[0])
    np.testing.assert_allclose(force, expected, rtol=3e-14, atol=1e-16)


@pytest.mark.parametrize("electric", [True, False])
def test_rest_dipole_cubic_regular_field_and_torque(electric):
    u, d = rest()
    if electric:
        d[0, 0, 1], d[0, 1, 0] = -1, 1
        d[3, 0, 2], d[3, 2, 0] = -1 / 6, 1 / 6
    else:
        d[0, 2, 3], d[0, 3, 2] = -1, 1
        d[3, 3, 1], d[3, 1, 3] = -1 / 6, 1 / 6
    result = regular_self_field(u, d, charge=0)
    # E_self=2/3 p''' and B_self=2/3 m''' in Gaussian c=1 units.
    # D^{ij}=epsilon^{ijk} m_k whereas F^{ij}=-epsilon^{ijk} B_k.
    expected = (2 / 3) * 6 * d[3] * (1 if electric else -1)
    np.testing.assert_allclose(result["field"], expected, rtol=2e-14, atol=1e-16)
    sectors = reaction_sectors(result, u[0], d[0], charge=0)
    assert np.linalg.norm(sectors["D_squared"]["torque"]) > 0.5
    np.testing.assert_allclose(sectors["D_squared"]["force"], 0, atol=1e-16)


@pytest.mark.parametrize("beta", [0.6, 0.99, 0.999])
def test_covariant_full_tensor_and_gradient(beta):
    u, d = rest()
    u[1] = [0, 0.1, 0.2, -0.1]
    u[2] = [0.03, 0.2, -0.1, 0.3]
    d[:] = np.array([m.unpack(np.arange(6) * (k + 1) / 20) for k in range(5)])
    b = np.eye(4)
    gamma = 1 / np.sqrt(1 - beta * beta)
    b[0, 0] = b[1, 1] = gamma
    b[0, 1] = b[1, 0] = gamma * beta
    inv = m.METRIC[:, None] * b.T * m.METRIC[None, :]
    original = regular_self_field(u, d, charge=0.2)
    changed = regular_self_field(
        u @ b.T, np.einsum("ai,kij,bj->kab", b, d, b), charge=0.2
    )
    for key in ("charge_field", "dipole_field"):
        expected = b @ original[key] @ b.T
        assert (
            np.linalg.norm(changed[key] - expected) / np.linalg.norm(expected) < 1e-12
        )
        np.testing.assert_array_equal(changed[key], -changed[key].T)
    for key in ("charge_gradient", "dipole_gradient"):
        expected = np.einsum("ak,ib,jc,abc->kij", inv, b, b, original[key])
        assert (
            np.linalg.norm(changed[key] - expected) / np.linalg.norm(expected) < 1e-12
        )


def test_source_balance_is_covariant():
    u, d = rest()
    u[1] = [0, 0.1, 0.2, -0.1]
    u[2] = [0.03, 0.2, -0.1, 0.3]
    d[:] = np.array([m.unpack(np.arange(6) * (n + 1) * 0.01) for n in range(5)])
    transform = np.eye(4)
    transform[0, 0] = transform[1, 1] = 1.25
    transform[0, 1] = transform[1, 0] = 0.75
    original = regular_self_field(u, d, charge=0.2)["source_balance"]
    changed = regular_self_field(
        u @ transform.T,
        np.einsum("ai,kij,bj->kab", transform, d, transform),
        charge=0.2,
    )["source_balance"]
    for sector in original:
        for name in ("bound_momentum", "bound_momentum_rate", "outward_radiation_rate"):
            expected = transform @ original[sector][name]
            assert (
                np.linalg.norm(changed[sector][name] - expected)
                / np.linalg.norm(expected)
                < 1e-12
            )


def test_regular_field_obeys_homogeneous_maxwell_and_sector_scaling():
    u, d = rest()
    u[1, 1], u[2, 0], u[2, 2] = 0.1, 0.005, 0.1
    d[:] = np.array([m.unpack(np.arange(6) * (k + 1) / 100) for k in range(5)])
    r = regular_self_field(u, d, charge=0.2)
    lower = r["gradient"] * m.METRIC[None, :, None] * m.METRIC[None, None, :]
    np.testing.assert_allclose(
        lower + lower.transpose(1, 2, 0) + lower.transpose(2, 0, 1), 0, atol=1e-14
    )
    s = reaction_sectors(r, u[0], d[0], charge=0.2)
    doubled = regular_self_field(u, 2 * d, charge=0.2)
    t = reaction_sectors(doubled, u[0], 2 * d[0], charge=0.2)
    for key, factor in [("q_squared", 1), ("qD", 2), ("D_squared", 4)]:
        for component in ("force", "torque"):
            np.testing.assert_allclose(
                t[key][component], factor * s[key][component], atol=1e-14
            )


def test_independent_retarded_minus_advanced_full_dipole():
    from math import comb
    from core.full_dipole_response import response

    u, d = rest()
    d[0] = m.unpack(np.arange(6) * 0.03)
    d[3] = m.unpack(np.arange(6)[::-1] * 0.02)
    reversal = np.diag([-1.0, 1, 1, 1])

    def sampled(reverse):
        # Polynomial source over [-1,1]; observer at t=0, r=0.3.
        coefficients = np.zeros((5, 4, 4))
        for n in range(5):
            tensor = (-1) ** n * (-reversal @ d[n] @ reversal) if reverse else d[n]
            for k in range(n + 1):
                coefficients[k] += tensor * comb(n, k) * 2**k * (-1) ** (n - k)
        return response([0, 0.3, 0.1, -0.2], -1, 2, np.zeros((2, 3)), coefficients)

    ret, rev = sampled(False), sampled(True)
    advanced = -reversal @ rev["field_tensor"] @ reversal
    actual = regular_self_field(u, d, charge=0)["field"]
    np.testing.assert_allclose(
        actual, 0.5 * (ret["field_tensor"] - advanced), atol=1e-13, rtol=1e-12
    )


def test_rotating_magnetic_dipole_energy_and_angular_momentum_loss():
    from math import factorial

    u, d = rest()
    # Prescribed m=(cos(t),sin(t),0), t=0: m'' squared=1 and
    # m' cross m''=(0,0,1). Standard outward power and angular momentum
    # are both 2/3. Reversible endpoint derivatives vanish for this motion.
    for n in range(5):
        moment = np.array(
            [np.cos(n * np.pi / 2), np.sin(n * np.pi / 2), 0]
        ) / factorial(n)
        d[n, 2, 3], d[n, 3, 1], d[n, 1, 2] = moment
        d[n] -= d[n].T
    result = regular_self_field(u, d, charge=0)
    sector = reaction_sectors(result, u[0], d[0], charge=0)["D_squared"]
    assert sector["force"][0] == pytest.approx(-2 / 3, abs=1e-14)
    assert sector["torque"][1, 2] == pytest.approx(-2 / 3, abs=1e-14)
    np.testing.assert_allclose(sector["force"][1:], 0, atol=1e-14)
    assert result["source_balance"]["D_squared"]["outward_radiation_rate"][
        0
    ] == pytest.approx(2 / 3, abs=1e-14)


def test_full_current_bound_radiation_identity():
    u, d = rest()
    u[1] = [0, 0.2, -0.1, 0.3]
    u[2] = [0.07, 0.1, 0.2, -0.2]
    d[:] = np.array([m.unpack(np.arange(6) * (n + 1) * 0.01) for n in range(5)])
    result = regular_self_field(u, d, charge=0.3)
    forces = reaction_sectors(result, u[0], d[0], charge=0.3)
    for key, row in result["source_balance"].items():
        force = forces[key]["force"]
        residue = force + row["bound_momentum_rate"] + row["outward_radiation_rate"]
        denominator = sum(
            np.linalg.norm(x)
            for x in (force, row["bound_momentum_rate"], row["outward_radiation_rate"])
        )
        assert np.linalg.norm(residue) / denominator < 1e-13


def test_charge_bound_momentum_matches_schott_term():
    u, d = rest()
    u[1] = [0, 0.2, -0.1, 0.3]
    u[2] = [0.07, 0.1, 0.2, -0.2]
    result = regular_self_field(u, d, charge=0.3)
    np.testing.assert_allclose(
        result["source_balance"]["q_squared"]["bound_momentum"],
        -2 * 0.3**2 / 3 * u[1],
        atol=1e-15,
    )
