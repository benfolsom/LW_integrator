"""Compiled building blocks preserve the reference arithmetic, including tangents."""

import numpy as np
import pytest

from core import dipole_hertz_jet as jet


@pytest.mark.parametrize("seed", range(5))
def test_compiled_coefficients_are_bitwise_equal(seed):
    rng = np.random.default_rng(seed)
    for scale in (1e-10, 1.0, 1e10):
        left, right = rng.normal(size=(2, len(jet._MULTIINDICES))) * scale
        left[0] = scale
        np.testing.assert_array_equal(
            jet._multiply_coefficients(left, right), jet._python_multiply(left, right)
        )
        np.testing.assert_array_equal(
            jet._reciprocal_coefficients(left), jet._python_reciprocal(left)
        )


def test_complete_expression_and_tangent_match_python(monkeypatch):
    rng = np.random.default_rng(71)
    coefficients, tangent = rng.normal(size=(2, len(jet._MULTIINDICES)))
    coefficients[0] = 2
    x = jet._CompiledJet3(coefficients, tangent)
    actual = ((x * x + 3) / x).sqrt()
    monkeypatch.setattr(jet, "_multiply_coefficients", jet._python_multiply)
    monkeypatch.setattr(jet, "_reciprocal_coefficients", jet._python_reciprocal)
    expected = ((x * x + 3) / x).sqrt()
    np.testing.assert_array_equal(actual.coefficients, expected.coefficients)
    np.testing.assert_array_equal(actual.tangent, expected.tangent)


def test_reference_path_does_not_call_compiled_arithmetic(monkeypatch):
    def unexpected(*args):
        raise AssertionError("Reference arithmetic must remain Python")

    monkeypatch.setattr(jet, "_multiply_coefficients", unexpected)
    monkeypatch.setattr(jet, "_reciprocal_coefficients", unexpected)
    x = jet._Jet3.variable(2.0, 0)
    assert ((x * x + 3) / x).sqrt().value > 0


def test_compiled_type_survives_reverse_operations_and_polynomials():
    x = jet._CompiledJet3.variable(2.0, 0)
    for value in (1 - x, 1 / x, x / 2, x.sqrt(), jet._polynomial([1, 2, 3], x)):
        assert isinstance(value, jet._CompiledJet3)
    assert isinstance(jet._dot([x], [x]), jet._CompiledJet3)
    assert isinstance(jet._norm([x]), jet._CompiledJet3)
