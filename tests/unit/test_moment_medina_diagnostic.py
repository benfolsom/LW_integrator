import numpy as np
import pytest

from core.moment_medina_diagnostic import match_medina_force


def state(force, *, capped=False, ready=True):
    return {
        "_moment_applied_medina_force_native": np.asarray(force),
        "medina_impulse_capped": np.array([capped]),
        "medina_force_derivative_ready": np.array([ready]),
    }


def test_matches_same_force_without_accumulating_trial_kicks():
    calls = []

    def step(force):
        calls.append(force.copy())
        return state([[1.0, 2.0, 3.0]])

    result = match_medina_force(step, particle_count=1)
    assert result["_moment_medina_iterations"] == 2
    np.testing.assert_array_equal(calls, [[[0, 0, 0]], [[1, 2, 3]]])
    np.testing.assert_array_equal(result["_moment_medina_force_relative_mismatch"], 0)


def test_zero_force_converges_without_divide_by_zero():
    assert (
        match_medina_force(lambda f: state(f), particle_count=1)[
            "_moment_medina_iterations"
        ]
        == 1
    )


@pytest.mark.parametrize("kwargs", [{"capped": True}, {"ready": False}])
def test_rejects_unusable_medina_step(kwargs):
    with pytest.raises(ValueError, match="primed, uncapped"):
        match_medina_force(lambda f: state(f, **kwargs), particle_count=1)


@pytest.mark.parametrize("force", [[[np.nan, 0, 0]], [1, 2, 3]])
def test_rejects_bad_force(force):
    with pytest.raises(ValueError, match="invalid applied"):
        match_medina_force(lambda f: state(force), particle_count=1)


def test_iteration_limit_fails_instead_of_accepting_mismatch():
    with pytest.raises(ValueError, match="did not converge"):
        match_medina_force(lambda f: state(f + 1), particle_count=1)
