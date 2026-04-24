from __future__ import annotations

import numpy as np
import pytest

import core.vectorized_interactions as vectorized_interactions
from core.constants import C_MMNS


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


def test_gather_external_samples_linearly_interpolates_and_skips_invalid_slots() -> None:
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
