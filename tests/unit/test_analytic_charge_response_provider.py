"""Production contracts for the one-root analytical charge response."""

from __future__ import annotations

import numpy as np
import pytest

from core.analytic_charge_response_diagnostics import (
    analytic_charge_response_diagnostics,
    reset_analytic_charge_response_diagnostics,
)
from core.antisymmetric_response_rfs import (
    materialize_antisymmetric_response_native,
    materialize_partial_antisymmetric_response_native,
)
from core.charge_source_interactions import (
    evaluate_retarded_charge_source_interaction_native,
)
from core.constants import C_MMNS
from core.exact_retarded_backend import ExactRetardedBackendUnavailableError
from core.retarded_fields import (
    ObserverEvent,
    evaluate_retarded_charge_field_gradient_native,
    evaluate_retarded_charge_response_gradient_native,
)


def _uniform_history(
    *,
    times_ns: np.ndarray,
    beta: np.ndarray,
    charge_native: float = -1.3,
) -> list[dict[str, np.ndarray]]:
    positions = C_MMNS * times_ns[:, np.newaxis] * beta[np.newaxis, :]
    return [
        {
            "t": np.asarray((time_ns,)),
            "x": np.asarray((position[0],)),
            "y": np.asarray((position[1],)),
            "z": np.asarray((position[2],)),
            "bx": np.asarray((beta[0],)),
            "by": np.asarray((beta[1],)),
            "bz": np.asarray((beta[2],)),
            "bdotx": np.zeros(1),
            "bdoty": np.zeros(1),
            "bdotz": np.zeros(1),
            "q": np.asarray((charge_native,)),
            "q_source": np.asarray((charge_native,)),
            "_dead_particles": np.zeros(1, dtype=bool),
        }
        for time_ns, position in zip(times_ns, positions)
    ]


def test_analytical_provider_matches_fine_stencil_and_avoids_fallback() -> None:
    pytest.importorskip("numba")
    history = _uniform_history(
        times_ns=np.linspace(-0.04, 0.004, 121),
        beta=np.asarray((0.31, -0.12, 0.07)),
    )
    event = ObserverEvent(-1.0e-5, (0.9, 1.1, -0.5))
    analytical = evaluate_retarded_charge_response_gradient_native(
        history,
        event,
        relative_step=2.5e-6,
    )
    stencil = evaluate_retarded_charge_field_gradient_native(
        history,
        event,
        relative_step=2.5e-6,
        backend="numba_full_strict_serial",
    )

    assert analytical.fallback_used is False
    assert analytical.minimum_segment_margin_ratio > 1.0
    np.testing.assert_allclose(
        materialize_antisymmetric_response_native(analytical.antisymmetric_response),
        stencil.field.field_tensor,
        rtol=4.0e-15,
        atol=1.0e-16,
    )
    np.testing.assert_allclose(
        materialize_partial_antisymmetric_response_native(
            analytical.partial_antisymmetric_response
        ),
        stencil.partial_f,
        rtol=2.0e-8,
        atol=2.0e-8,
    )


def test_root_on_interpolation_knot_uses_maintained_fallback() -> None:
    pytest.importorskip("numba")
    radius_mm = 1.0
    history = _uniform_history(
        times_ns=np.asarray((-0.01, 0.0, 0.01)),
        beta=np.zeros(3),
    )
    event = ObserverEvent(radius_mm / C_MMNS, (radius_mm, 0.0, 0.0))

    reset_analytic_charge_response_diagnostics()
    analytical = evaluate_retarded_charge_response_gradient_native(history, event)

    assert analytical.fallback_used is True
    assert analytical.fallback_reason == "source_0:retarded_root_near_segment_boundary"
    assert analytical.fallback_stencil_step_mm is not None
    diagnostics = analytic_charge_response_diagnostics()
    assert diagnostics.calls == 1
    assert diagnostics.fallback_calls == 1
    assert diagnostics.fallback_segment_boundary == 1


def test_analytical_interaction_contracts_without_field_tensor() -> None:
    pytest.importorskip("numba")
    history = _uniform_history(
        times_ns=np.linspace(-0.04, 0.004, 121),
        beta=np.asarray((0.17, -0.08, 0.03)),
    )
    observer_beta = np.asarray((-0.08, 0.13, 0.03))
    gamma = 1.0 / np.sqrt(1.0 - float(observer_beta @ observer_beta))
    velocity = gamma * C_MMNS * np.concatenate(((1.0,), observer_beta))
    interaction = evaluate_retarded_charge_source_interaction_native(
        history,
        ObserverEvent(-1.0e-5, (1.2, 0.5, -0.4)),
        four_velocity_mm_ns=velocity,
        observer_charge_native=-0.7,
        proper_time_step_ns=0.025,
        backend="numba_analytic_charge_response_serial",
    )

    assert interaction.field is None
    assert interaction.response is not None
    assert interaction.analytical_fallback_used is False
    np.testing.assert_allclose(
        interaction.mechanical_four_impulse,
        0.025 * interaction.mechanical_four_force,
        rtol=2.0e-15,
        atol=1.0e-18,
    )
    np.testing.assert_array_equal(
        interaction.four_potential,
        interaction.response.four_potential,
    )


def test_compatibility_gradient_materializes_only_at_api_boundary() -> None:
    pytest.importorskip("numba")
    history = _uniform_history(
        times_ns=np.linspace(-0.04, 0.004, 121),
        beta=np.asarray((0.11, 0.04, -0.02)),
    )
    event = ObserverEvent(-1.0e-5, (0.8, -0.6, 0.3))
    direct = evaluate_retarded_charge_response_gradient_native(history, event)
    compatibility = evaluate_retarded_charge_field_gradient_native(
        history,
        event,
        backend="numba_analytic_charge_response_serial",
    )

    np.testing.assert_array_equal(
        compatibility.field.field_tensor,
        materialize_antisymmetric_response_native(direct.antisymmetric_response),
    )
    np.testing.assert_array_equal(
        compatibility.partial_f,
        materialize_partial_antisymmetric_response_native(
            direct.partial_antisymmetric_response
        ),
    )
    assert compatibility.stencil_step_mm == 0.0


def test_initial_analytical_jit_failure_has_named_capability_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    numba = pytest.importorskip("numba")
    import core.charge_response_jet_numba as compiled

    def failed_compilation(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise numba.core.errors.TypingError("synthetic compilation failure")

    failed_compilation.signatures = ()
    monkeypatch.setattr(
        compiled,
        "evaluate_charge_response_coefficients_one_event_strict_serial",
        failed_compilation,
    )
    with pytest.raises(
        ExactRetardedBackendUnavailableError,
        match="failed during initial JIT compilation",
    ):
        evaluate_retarded_charge_response_gradient_native(
            _uniform_history(
                times_ns=np.linspace(-0.04, 0.004, 121),
                beta=np.asarray((0.13, -0.06, 0.04)),
            ),
            ObserverEvent(-1.0e-5, (0.9, 1.1, -0.5)),
        )
