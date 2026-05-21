from __future__ import annotations

import numpy as np
import pytest

import core.vectorized_interactions as vectorized_interactions
from core.constants import C_MMNS
from core.types import TrajectoryBuilder


def _make_external_state(
    value: float,
    *,
    charge: float = 1.0,
    gamma: float = 1.0,
    particle_count: int = 1,
) -> dict[str, np.ndarray | float]:
    values = np.full(particle_count, value, dtype=float)
    return {
        "x": values.copy(),
        "y": values.copy(),
        "z": values.copy(),
        "bx": values.copy(),
        "by": np.zeros(particle_count, dtype=float),
        "bz": np.zeros(particle_count, dtype=float),
        "bdotx": (values * 10.0).copy(),
        "bdoty": np.zeros(particle_count, dtype=float),
        "bdotz": np.zeros(particle_count, dtype=float),
        "q": charge,
        "gamma": gamma,
    }


def _make_samples(
    *,
    charge: list[float] | None = None,
    gamma: list[float] | None = None,
    bx: list[float] | None = None,
    by: list[float] | None = None,
    bz: list[float] | None = None,
    bdotx: list[float] | None = None,
    bdoty: list[float] | None = None,
    bdotz: list[float] | None = None,
    valid_mask: list[bool] | None = None,
) -> vectorized_interactions.ExternalSampleBatch:
    charge_vals = [1.0] if charge is None else charge
    size = len(charge_vals)

    def _arr(values: list[float] | None, default: float = 0.0) -> np.ndarray:
        if values is None:
            values = [default] * size
        return np.array(values, dtype=float)

    return vectorized_interactions.ExternalSampleBatch(
        charge=_arr(charge),
        gamma=_arr(gamma, 1.0),
        bx=_arr(bx),
        by=_arr(by),
        bz=_arr(bz),
        bdotx=_arr(bdotx),
        bdoty=_arr(bdoty),
        bdotz=_arr(bdotz),
        valid_mask=np.array(
            [True] * size if valid_mask is None else valid_mask, dtype=bool
        ),
    )


def test_external_sample_batch_any_valid_reflects_mask() -> None:
    assert _make_samples(valid_mask=[False, True]).any_valid is True
    assert _make_samples(valid_mask=[False, False]).any_valid is False


def test_compute_small_k_forces_series_matches_direct_inputs() -> None:
    result = vectorized_interactions._compute_small_k_forces_series(
        k_factor=np.array([0.5]),
        charge_factor_base=np.array([2.0]),
        v_betas_scalar=np.array([3.0]),
        v_beta_dot_mixed_scalar=np.array([5.0]),
        bx_ext=np.array([0.0]),
        by_ext=np.array([0.0]),
        bz_ext=np.array([0.0]),
        bdotx_ext=np.array([0.0]),
        bdoty_ext=np.array([0.0]),
        bdotz_ext=np.array([0.0]),
        nx=np.array([1.0]),
        ny=np.array([0.0]),
        nz=np.array([0.0]),
        R_sep=np.array([7.0]),
        gamma_ext=np.array([1.0]),
        c=10.0,
    )

    assert result[0].tolist() == pytest.approx([760.0])
    assert result[1].tolist() == pytest.approx([0.0])
    assert result[2].tolist() == pytest.approx([0.0])
    assert result[3].tolist() == pytest.approx([520.0])


def test_gather_external_samples_linearly_interpolates_and_skips_invalid_slots() -> (
    None
):
    trajectory_ext = [
        _make_external_state(0.0, charge=2.0, gamma=1.0),
        _make_external_state(2.0, charge=2.0, gamma=3.0),
    ]

    samples = vectorized_interactions.gather_external_samples(
        trajectory_ext,
        indices=np.array([1, 1]),
        indices_next=np.array([0, 0]),
        weights=np.array([0.25, 0.25]),
    )

    assert samples.valid_mask.tolist() == [True, False]
    assert samples.charge.tolist() == pytest.approx([2.0, 0.0])
    assert samples.gamma.tolist() == pytest.approx([1.5, 0.0])
    assert samples.bx.tolist() == pytest.approx([0.5, 0.0])
    assert samples.bdotx.tolist() == pytest.approx([5.0, 0.0])


def test_gather_external_samples_supports_cubic_interpolation() -> None:
    trajectory_ext = [
        _make_external_state(0.0, gamma=1.0),
        _make_external_state(1.0, gamma=2.0),
        _make_external_state(2.0, gamma=3.0),
        _make_external_state(3.0, gamma=4.0),
    ]

    samples = vectorized_interactions.gather_external_samples(
        trajectory_ext,
        indices=np.array([2]),
        indices_next=np.array([1]),
        weights=np.array([0.5]),
        indices_prev=np.array([0]),
        indices_next2=np.array([3]),
        use_cubic=True,
    )

    assert samples.valid_mask.tolist() == [True]
    assert samples.gamma.tolist() == pytest.approx([2.5])
    assert samples.bx.tolist() == pytest.approx([1.5])
    assert samples.bdotx.tolist() == pytest.approx([15.0])


def test_gather_external_samples_soa_skips_negative_and_out_of_range_indices() -> None:
    builder = TrajectoryBuilder(2, 2)
    state0 = {
        "x": np.array([0.0, 1.0], dtype=float),
        "y": np.zeros(2, dtype=float),
        "z": np.zeros(2, dtype=float),
        "t": np.zeros(2, dtype=float),
        "Px": np.zeros(2, dtype=float),
        "Py": np.zeros(2, dtype=float),
        "Pz": np.zeros(2, dtype=float),
        "Pt": np.ones(2, dtype=float),
        "gamma": np.array([2.0, 3.0], dtype=float),
        "bx": np.array([0.1, 0.2], dtype=float),
        "by": np.zeros(2, dtype=float),
        "bz": np.zeros(2, dtype=float),
        "bdotx": np.array([1.0, 2.0], dtype=float),
        "bdoty": np.zeros(2, dtype=float),
        "bdotz": np.zeros(2, dtype=float),
        "q": np.array([1.0, 2.0], dtype=float),
        "m": np.array([1.0, 1.0], dtype=float),
        "char_time": np.array([1e-3, 1e-3], dtype=float),
    }
    state1 = {
        k: (
            v + 1.0
            if isinstance(v, np.ndarray) and k in {"x", "gamma", "bx", "bdotx"}
            else v.copy() if isinstance(v, np.ndarray) else v
        )
        for k, v in state0.items()
    }
    builder.set_step(0, state0)
    builder.set_step(1, state1)
    traj_ext = builder.build()

    samples = vectorized_interactions.gather_external_samples_soa(
        traj_ext,
        indices=np.array([1, 5]),
        indices_next=np.array([0, 10]),
        weights=np.array([0.25, 0.25]),
        needs_interpolation=np.array([True, True]),
    )

    assert samples.valid_mask.tolist() == [True, False]
    assert samples.charge.tolist() == pytest.approx([1.0, 0.0])
    assert samples.gamma.tolist()[1] == pytest.approx(0.0)
    assert samples.bx.tolist()[1] == pytest.approx(0.0)


def test_gather_external_samples_skips_negative_and_out_of_range_indices() -> None:
    trajectory_ext = [
        _make_external_state(
            1.0,
            charge=np.array([1.0, 2.0, 3.0], dtype=float),
            gamma=np.array([1.0, 1.5, 2.0], dtype=float),
            particle_count=3,
        )
    ]

    samples = vectorized_interactions.gather_external_samples(
        trajectory_ext,
        indices=np.array([-1, 1, 0]),
    )

    assert samples.valid_mask.tolist() == [False, False, True]
    assert samples.charge.tolist() == pytest.approx([0.0, 0.0, 3.0])
    assert samples.gamma.tolist() == pytest.approx([0.0, 0.0, 2.0])
    assert samples.bx.tolist() == pytest.approx([0.0, 0.0, 1.0])


def test_gather_external_samples_linearly_interpolates_array_gamma_with_positions() -> (
    None
):
    trajectory_ext = [
        _make_external_state(
            0.0,
            charge=np.array([2.0], dtype=float),
            gamma=np.array([1.0], dtype=float),
        ),
        _make_external_state(
            2.0,
            charge=np.array([2.0], dtype=float),
            gamma=np.array([3.0], dtype=float),
        ),
    ]

    samples = vectorized_interactions.gather_external_samples(
        trajectory_ext,
        indices=np.array([1]),
        indices_next=np.array([0]),
        weights=np.array([0.25]),
        interpolate_positions=True,
    )

    assert samples.valid_mask.tolist() == [True]
    assert samples.charge.tolist() == pytest.approx([2.0])
    assert samples.gamma.tolist() == pytest.approx([1.5])
    assert samples.bx.tolist() == pytest.approx([0.5])
    assert samples.bdotx.tolist() == pytest.approx([5.0])


def test_gather_external_samples_supports_cubic_position_interpolation() -> None:
    trajectory_ext = [
        _make_external_state(0.0, gamma=np.array([1.0], dtype=float)),
        _make_external_state(1.0, gamma=np.array([2.0], dtype=float)),
        _make_external_state(2.0, gamma=np.array([3.0], dtype=float)),
        _make_external_state(3.0, gamma=np.array([4.0], dtype=float)),
    ]

    samples = vectorized_interactions.gather_external_samples(
        trajectory_ext,
        indices=np.array([2]),
        indices_next=np.array([1]),
        weights=np.array([0.5]),
        indices_prev=np.array([0]),
        indices_next2=np.array([3]),
        use_cubic=True,
        interpolate_positions=True,
    )

    assert samples.valid_mask.tolist() == [True]
    assert samples.gamma.tolist() == pytest.approx([2.5])
    assert samples.bx.tolist() == pytest.approx([1.5])
    assert samples.bdotx.tolist() == pytest.approx([15.0])


def test_compute_vectorized_contributions_returns_zero_for_guard_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(vectorized_interactions, "NUMBA_AVAILABLE", False)
    samples = _make_samples(valid_mask=[True])

    assert vectorized_interactions.compute_vectorized_contributions(
        h=1.0,
        charge_i=1.0,
        mass_i=1.0,
        gamma_i=1.0,
        beta_vec=(0.0, 0.0, 0.0),
        nhat_nx=np.array([1.0]),
        nhat_ny=np.array([0.0]),
        nhat_nz=np.array([0.0]),
        R_separation=np.array([1.0]),
        samples=samples,
        apply_external=False,
    ) == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    assert vectorized_interactions.compute_vectorized_contributions(
        h=1.0,
        charge_i=1.0,
        mass_i=1.0,
        gamma_i=1e7,
        beta_vec=(0.0, 0.0, 0.0),
        nhat_nx=np.array([1.0]),
        nhat_ny=np.array([0.0]),
        nhat_nz=np.array([0.0]),
        R_separation=np.array([1.0]),
        samples=samples,
        apply_external=True,
    ) == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def test_compute_vectorized_contributions_returns_zero_for_empty_and_filtered_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(vectorized_interactions, "NUMBA_AVAILABLE", False)

    empty_samples = _make_samples(charge=[], valid_mask=[])
    assert vectorized_interactions.compute_vectorized_contributions(
        h=1.0,
        charge_i=1.0,
        mass_i=1.0,
        gamma_i=1.0,
        beta_vec=(0.0, 0.0, 0.0),
        nhat_nx=np.array([], dtype=float),
        nhat_ny=np.array([], dtype=float),
        nhat_nz=np.array([], dtype=float),
        R_separation=np.array([], dtype=float),
        samples=empty_samples,
        apply_external=True,
    ) == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    filtered_samples = _make_samples(
        charge=[1.0], gamma=[1.0], bx=[0.0], valid_mask=[True]
    )
    assert vectorized_interactions.compute_vectorized_contributions(
        h=1.0,
        charge_i=1.0,
        mass_i=1.0,
        gamma_i=1.0,
        beta_vec=(0.0, 0.0, 0.0),
        nhat_nx=np.array([1.0]),
        nhat_ny=np.array([0.0]),
        nhat_nz=np.array([0.0]),
        R_separation=np.array([0.0]),
        samples=filtered_samples,
        apply_external=True,
    ) == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def test_compute_vectorized_contributions_avoids_parallel_kernel_for_small_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(vectorized_interactions, "NUMBA_AVAILABLE", True)

    def _unexpected_parallel_call(*args: object) -> tuple[float, ...]:
        raise AssertionError("small batches should not use the parallel numba kernel")

    monkeypatch.setattr(
        vectorized_interactions,
        "_compute_forces_numba_kernel",
        _unexpected_parallel_call,
    )
    samples = _make_samples(
        charge=[2.0],
        gamma=[1.0],
        bx=[0.0],
        valid_mask=[True],
    )

    result = vectorized_interactions.compute_vectorized_contributions(
        h=0.5,
        charge_i=1.0,
        mass_i=2.0,
        gamma_i=1.0,
        beta_vec=(0.0, 0.0, 0.0),
        nhat_nx=np.array([1.0]),
        nhat_ny=np.array([0.0]),
        nhat_nz=np.array([0.0]),
        R_separation=np.array([4.0]),
        samples=samples,
        apply_external=True,
    )

    assert result == pytest.approx((0.0625, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5))


def test_compute_vectorized_contributions_matches_simple_normal_regime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(vectorized_interactions, "NUMBA_AVAILABLE", False)
    samples = _make_samples(
        charge=[2.0],
        gamma=[1.0],
        bx=[0.0],
        valid_mask=[True],
    )

    result = vectorized_interactions.compute_vectorized_contributions(
        h=0.5,
        charge_i=1.0,
        mass_i=2.0,
        gamma_i=1.0,
        beta_vec=(0.0, 0.0, 0.0),
        nhat_nx=np.array([1.0]),
        nhat_ny=np.array([0.0]),
        nhat_nz=np.array([0.0]),
        R_separation=np.array([4.0]),
        samples=samples,
        apply_external=True,
    )

    assert result == pytest.approx((0.0625, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5))


def test_compute_vectorized_contributions_uses_series_helper_for_small_k(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(vectorized_interactions, "NUMBA_AVAILABLE", False)

    calls: list[np.ndarray] = []

    def _fake_series(k_factor: np.ndarray, *args: object) -> tuple[np.ndarray, ...]:
        calls.append(k_factor.copy())
        return (
            np.array([10.0]),
            np.array([20.0]),
            np.array([30.0]),
            np.array([40.0]),
        )

    monkeypatch.setattr(
        vectorized_interactions,
        "_compute_small_k_forces_series",
        _fake_series,
    )

    samples = _make_samples(
        charge=[2.0],
        gamma=[1.0],
        bx=[1.0 - 5e-4],
        valid_mask=[True],
    )

    result = vectorized_interactions.compute_vectorized_contributions(
        h=1.0,
        charge_i=1.0,
        mass_i=2.0,
        gamma_i=1.0,
        beta_vec=(0.0, 0.0, 0.0),
        nhat_nx=np.array([1.0]),
        nhat_ny=np.array([0.0]),
        nhat_nz=np.array([0.0]),
        R_separation=np.array([4.0]),
        samples=samples,
        apply_external=True,
    )

    assert calls[0].tolist() == pytest.approx([5e-4])
    assert result[:4] == pytest.approx((10.0, 20.0, 30.0, 40.0))
    assert result[4] == pytest.approx((499.75 / C_MMNS))
    assert result[5:7] == pytest.approx((0.0, 0.0))
    assert result[7] == pytest.approx(1000.0)


def test_compute_vectorized_contributions_filters_hard_k_cutoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(vectorized_interactions, "NUMBA_AVAILABLE", False)
    samples = _make_samples(
        charge=[2.0],
        gamma=[1.0],
        bx=[1.0],
        valid_mask=[True],
    )

    result = vectorized_interactions.compute_vectorized_contributions(
        h=1.0,
        charge_i=1.0,
        mass_i=1.0,
        gamma_i=1.0,
        beta_vec=(0.0, 0.0, 0.0),
        nhat_nx=np.array([1.0]),
        nhat_ny=np.array([0.0]),
        nhat_nz=np.array([0.0]),
        R_separation=np.array([1.0]),
        samples=samples,
        apply_external=True,
    )

    assert result == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    python_kernel = getattr(
        vectorized_interactions._compute_forces_numba_kernel,
        "py_func",
        vectorized_interactions._compute_forces_numba_kernel,
    )
    kernel_result = python_kernel(
        1.0,
        1.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        np.array([1.0]),
        np.array([0.0]),
        np.array([0.0]),
        np.array([1.0]),
        np.array([1.0]),
        np.array([0.0]),
        np.array([0.0]),
        np.array([0.0]),
        np.array([0.0]),
        np.array([0.0]),
        np.array([2.0]),
        np.array([1.0]),
        C_MMNS,
    )

    assert kernel_result == pytest.approx((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))


def test_compute_vectorized_contributions_reports_verbose_k_regimes(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(vectorized_interactions, "NUMBA_AVAILABLE", False)
    samples = _make_samples(
        charge=[1.0, 1.0, 1.0],
        gamma=[1.0, 1.0, 1.0],
        bx=[0.0, 1.0 - 5e-4, 1.0],
        valid_mask=[True, True, True],
    )

    result = vectorized_interactions.compute_vectorized_contributions(
        h=1.0,
        charge_i=1.0,
        mass_i=1.0,
        gamma_i=1.0,
        beta_vec=(0.0, 0.0, 0.0),
        nhat_nx=np.array([1.0, 1.0, 1.0]),
        nhat_ny=np.array([0.0, 0.0, 0.0]),
        nhat_nz=np.array([0.0, 0.0, 0.0]),
        R_separation=np.array([1.0, 1.0, 1.0]),
        samples=samples,
        apply_external=True,
        verbosity=4,
    )

    output = capsys.readouterr().out
    assert "compute_vectorized_contributions called" in output
    assert "k-factor hard cutoff triggered" in output
    assert "series approximation" in output
    assert "Force calculation (normal k regime)" in output
    assert "Force components (normal k regime)" in output
    assert np.all(np.isfinite(result))


def test_numba_force_kernel_matches_numpy_path_for_nonzero_acceleration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(vectorized_interactions, "NUMBA_AVAILABLE", False)
    samples = _make_samples(
        charge=[2.0],
        gamma=[2.5],
        bx=[0.6],
        by=[0.3],
        bz=[0.2],
        bdotx=[0.5],
        bdoty=[0.4],
        bdotz=[0.3],
        valid_mask=[True],
    )

    expected = vectorized_interactions.compute_vectorized_contributions(
        h=0.5,
        charge_i=1.0,
        mass_i=2.0,
        gamma_i=1.7,
        beta_vec=(0.5, 0.2, 0.1),
        nhat_nx=np.array([0.6]),
        nhat_ny=np.array([0.3]),
        nhat_nz=np.array([0.2]),
        R_separation=np.array([4.0]),
        samples=samples,
        apply_external=True,
    )

    actual = vectorized_interactions._compute_forces_numba_kernel(
        0.5,
        1.0,
        2.0,
        1.7,
        0.5,
        0.2,
        0.1,
        np.array([0.6]),
        np.array([0.3]),
        np.array([0.2]),
        np.array([4.0]),
        np.array([0.6]),
        np.array([0.3]),
        np.array([0.2]),
        np.array([0.5]),
        np.array([0.4]),
        np.array([0.3]),
        np.array([2.0]),
        np.array([2.5]),
        C_MMNS,
    )

    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)

    python_kernel = getattr(
        vectorized_interactions._compute_forces_numba_kernel,
        "py_func",
        vectorized_interactions._compute_forces_numba_kernel,
    )
    py_func_result = python_kernel(
        0.5,
        1.0,
        2.0,
        1.7,
        0.5,
        0.2,
        0.1,
        np.array([0.6]),
        np.array([0.3]),
        np.array([0.2]),
        np.array([4.0]),
        np.array([0.6]),
        np.array([0.3]),
        np.array([0.2]),
        np.array([0.5]),
        np.array([0.4]),
        np.array([0.3]),
        np.array([2.0]),
        np.array([2.5]),
        C_MMNS,
    )

    assert py_func_result == pytest.approx(expected, rel=1e-12, abs=1e-12)
