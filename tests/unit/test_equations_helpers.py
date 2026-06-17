from __future__ import annotations

import io
from contextlib import redirect_stdout
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

import core.equations as equations
from core.constants import C_MMNS
from core.integration_runner import IntegrationCancelled
from core.particle_status import mark_particle_dead
from core.self_consistency import SelfConsistencyConfig
from core.types import (
    ChronoMatchingMode,
    GammaReconciliationMethod,
    SimulationType,
    SpaceChargeConfig,
    StartupMode,
    TrajectoryBuilder,
)


def _make_state(
    *,
    x: list[float] | None = None,
    y: list[float] | None = None,
    z: list[float] | None = None,
    t: list[float] | None = None,
    bx: list[float] | None = None,
    by: list[float] | None = None,
    bz: list[float] | None = None,
    gamma: list[float] | None = None,
    charge: list[float] | None = None,
    mass: list[float] | None = None,
    char_time: list[float] | None = None,
    pt: list[float] | None = None,
) -> dict[str, np.ndarray]:
    x_vals = [0.0] if x is None else x
    size = len(x_vals)

    def _arr(values: list[float] | None, default: float = 0.0) -> np.ndarray:
        if values is None:
            values = [default] * size
        return np.array(values, dtype=float)

    gamma_vals = _arr(gamma, 1.0)
    mass_vals = _arr(mass, 1.0)

    pt_vals = gamma_vals * mass_vals * C_MMNS if pt is None else _arr(pt)

    return {
        "x": _arr(x),
        "y": _arr(y),
        "z": _arr(z),
        "t": _arr(t),
        "Px": np.zeros(size, dtype=float),
        "Py": np.zeros(size, dtype=float),
        "Pz": np.zeros(size, dtype=float),
        "Pt": pt_vals,
        "gamma": gamma_vals,
        "bx": _arr(bx),
        "by": _arr(by),
        "bz": _arr(bz),
        "bdotx": np.zeros(size, dtype=float),
        "bdoty": np.zeros(size, dtype=float),
        "bdotz": np.zeros(size, dtype=float),
        "q": _arr(charge, 1.0),
        "m": mass_vals,
        "char_time": _arr(char_time, 1e-3),
        "origin_x": _arr(x),
        "origin_y": _arr(y),
        "origin_z": _arr(z),
        "beta_avg_x": _arr(bx),
        "beta_avg_y": _arr(by),
        "beta_avg_z": _arr(bz),
        "beta_samples": np.ones(size, dtype=float),
    }


def _patch_mass_shell_convergence_sequence(
    monkeypatch: pytest.MonkeyPatch,
    sequence: list[tuple[bool, float]],
) -> None:
    calls = {"count": 0}

    def fake(*args: object, **kwargs: object) -> tuple[bool, float]:
        del args, kwargs
        idx = min(calls["count"], len(sequence) - 1)
        calls["count"] += 1
        return sequence[idx]

    monkeypatch.setattr(equations, "_check_mass_shell_convergence", fake)


def test_initialize_result_state_copies_charge_arrays_before_death_marking() -> None:
    state = _make_state(charge=[1.0, 2.0])

    result = equations._initialize_result_state(state)
    mark_particle_dead(result, 0, step=1, reason="synthetic")

    assert result["q"][0] == pytest.approx(0.0)
    assert state["q"][0] == pytest.approx(1.0)
    assert result["q"] is not state["q"]
    assert result["m"] is not state["m"]
    assert result["char_time"] is not state["char_time"]


def test_initialize_result_state_casts_continuous_fields_to_float() -> None:
    state = _make_state()
    for key in ("gamma", "Pt", "Pz", "x", "radiation_power"):
        if key in state:
            state[key] = state[key].astype(int)

    result = equations._initialize_result_state(state)

    for key in ("gamma", "Pt", "Pz", "x", "radiation_power"):
        assert np.issubdtype(result[key].dtype, np.floating), key


def test_particle_scalar_extractors_handle_arrays_and_scalars() -> None:
    array_state = _make_state(charge=[1.0, 2.0], mass=[3.0, 4.0], char_time=[5.0, 6.0])
    scalar_state = {
        "q": 7.0,
        "m": 8.0,
        "char_time": 9.0,
    }

    assert equations._get_particle_charge(array_state, 1) == pytest.approx(2.0)
    assert equations._get_particle_mass(array_state, 1) == pytest.approx(4.0)
    assert equations._get_particle_char_time(array_state, 1) == pytest.approx(6.0)
    assert equations._get_particle_charge(scalar_state, 0) == pytest.approx(7.0)
    assert equations._get_particle_mass(scalar_state, 0) == pytest.approx(8.0)
    assert equations._get_particle_char_time(scalar_state, 0) == pytest.approx(9.0)


def test_scalar_potential_contribution_uses_momentum_units() -> None:
    charge = 3.0
    scalar_potential = 5.0

    assert equations._scalar_potential_momentum_contribution(
        charge, scalar_potential
    ) == pytest.approx(charge * scalar_potential / C_MMNS)


def test_compute_approximate_retarded_distance_uses_factored_correction() -> None:
    current_state = _make_state(x=[1.0], y=[0.0], z=[0.0])
    external_state = _make_state(x=[0.0], y=[0.0], z=[0.0], bx=[0.5])

    nhat, indices = equations._compute_approximate_retarded_distance(
        current_state,
        external_state,
        particle_idx=0,
        time_step_idx=4,
    )

    assert indices.tolist() == [4]
    assert nhat["nx"][0] == pytest.approx(1.0)
    assert nhat["R"][0] == pytest.approx(2.0)


def test_compute_full_retarded_distance_handles_plain_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trajectory = [_make_state(x=[0.0]), _make_state(x=[1.0])]
    trajectory_ext = [_make_state(x=[10.0]), _make_state(x=[11.0])]

    captured: dict[str, object] = {}

    def fake_chrono(*args: object, **kwargs: object) -> np.ndarray:
        captured["chrono_kwargs"] = kwargs
        return np.array([5])

    def fake_distance(*args: object) -> dict[str, np.ndarray]:
        captured["distance_indices"] = args[-1]
        return {
            "R": np.array([42.0]),
            "nx": np.array([0.1]),
            "ny": np.array([0.2]),
            "nz": np.array([0.3]),
        }

    monkeypatch.setattr(equations, "chrono_match_indices", fake_chrono)
    monkeypatch.setattr(equations, "compute_retarded_distance", fake_distance)

    nhat, bounded, chrono_result = equations._compute_full_retarded_distance(
        trajectory,
        trajectory_ext,
        time_step_idx=1,
        particle_idx=0,
        chrono_mode=ChronoMatchingMode.FAST,
    )

    assert chrono_result is None
    assert bounded.tolist() == [1]
    assert captured["distance_indices"].tolist() == [1]
    assert captured["chrono_kwargs"]["interpolate"] is False
    assert nhat["R"].tolist() == pytest.approx([42.0])


def test_compute_full_retarded_distance_handles_interpolation_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trajectory = [_make_state(x=[0.0]), _make_state(x=[1.0])]
    trajectory_ext = [_make_state(x=[10.0]), _make_state(x=[11.0])]
    config = SelfConsistencyConfig(
        chrono_interpolate=True,
        chrono_tolerance=5e-4,
        chrono_high_precision=True,
        chrono_adaptive_tolerance=True,
        verbosity=2,
    )
    chrono_payload = equations.ChronoMatchResult(
        indices=np.array([-1]),
        indices_next=np.array([0]),
        weights=np.array([0.5]),
        residuals=np.array([0.25]),
        max_residual=0.25,
        needs_interpolation=np.array([True]),
        use_cubic=True,
        indices_prev=np.array([0]),
        indices_next2=np.array([1]),
    )
    captured: dict[str, object] = {}

    def fake_chrono(*args: object, **kwargs: object) -> equations.ChronoMatchResult:
        captured["chrono_kwargs"] = kwargs
        return chrono_payload

    monkeypatch.setattr(equations, "chrono_match_indices", fake_chrono)
    monkeypatch.setattr(
        equations,
        "compute_retarded_distance",
        lambda *args: {
            "R": np.array([2.0]),
            "nx": np.array([1.0]),
            "ny": np.array([0.0]),
            "nz": np.array([0.0]),
        },
    )

    nhat, bounded, returned_payload = equations._compute_full_retarded_distance(
        trajectory,
        trajectory_ext,
        time_step_idx=1,
        particle_idx=0,
        chrono_mode=ChronoMatchingMode.AVERAGED,
        self_consistency=config,
        timestep_h=0.2,
    )

    assert returned_payload is chrono_payload
    assert bounded.tolist() == [0]
    assert captured["chrono_kwargs"]["interpolate"] is True
    assert captured["chrono_kwargs"]["tolerance"] == pytest.approx(5e-4)
    assert captured["chrono_kwargs"]["high_precision"] is True
    assert captured["chrono_kwargs"]["adaptive_tolerance"] is True
    assert captured["chrono_kwargs"]["verbosity"] == 2
    assert captured["chrono_kwargs"]["timestep_h"] == pytest.approx(0.2)
    assert nhat["R"].tolist() == pytest.approx([2.0])


def test_b2b_cold_start_applies_external_force_without_observer_travel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trajectory = [
        _make_state(
            x=[0.0],
            z=[0.0],
            t=[0.0],
            gamma=[2.0],
            charge=[1.0],
            mass=[1.0],
            pt=[2.0 * C_MMNS],
        ),
        _make_state(
            x=[0.0],
            z=[0.0],
            t=[0.0],
            gamma=[2.0],
            charge=[1.0],
            mass=[1.0],
            pt=[2.0 * C_MMNS],
        ),
    ]
    driver = [
        _make_state(
            x=[0.0],
            z=[10.0],
            t=[0.0],
            gamma=[2.0],
            charge=[-1.0],
            mass=[1.0],
            pt=[2.0 * C_MMNS],
        ),
        _make_state(
            x=[0.0],
            z=[9.5],
            t=[0.1],
            gamma=[2.0],
            charge=[-1.0],
            mass=[1.0],
            pt=[2.0 * C_MMNS],
        ),
    ]

    monkeypatch.setattr(
        equations,
        "_compute_full_retarded_distance",
        lambda *args, **kwargs: (
            {
                "R": np.array([10.0]),
                "nx": np.array([0.0]),
                "ny": np.array([0.0]),
                "nz": np.array([-1.0]),
            },
            np.array([0], dtype=int),
            None,
        ),
    )

    monkeypatch.setattr(
        equations,
        "gather_external_samples",
        lambda *args, **kwargs: SimpleNamespace(
            x=np.array([0.0]),
            y=np.array([0.0]),
            z=np.array([10.0]),
            bx=np.array([0.0]),
            by=np.array([0.0]),
            bz=np.array([-0.5]),
            bdotx=np.array([0.0]),
            bdoty=np.array([0.0]),
            bdotz=np.array([0.0]),
            q=np.array([-1.0]),
            m=np.array([1.0]),
            char_time=np.array([1.0e-3]),
        ),
    )

    monkeypatch.setattr(
        equations,
        "compute_vectorized_contributions",
        lambda **kwargs: (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )

    updated = equations.retarded_equations_of_motion(
        0.1,
        trajectory,
        driver,
        1,
        aperture_radius=1.0,
        sim_type=SimulationType.BUNCH_TO_BUNCH,
        chrono_mode=ChronoMatchingMode.FAST,
        startup_mode=StartupMode.COLD_START,
        self_consistency=SelfConsistencyConfig(enabled=False),
    )

    assert updated["Pz"][0] > trajectory[1]["Pz"][0]


def test_self_consistency_nonconvergence_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    trajectory = [
        _make_state(x=[0.0], z=[0.0], t=[0.0], gamma=[2.0], charge=[1.0], mass=[1.0]),
        _make_state(x=[0.0], z=[0.5], t=[0.1], gamma=[2.0], charge=[1.0], mass=[1.0]),
    ]
    driver = [
        _make_state(x=[0.0], z=[1.0], t=[0.0], gamma=[2.0], charge=[-1.0], mass=[1.0]),
        _make_state(x=[0.0], z=[1.1], t=[0.1], gamma=[2.0], charge=[-1.0], mass=[1.0]),
    ]

    with pytest.raises(equations.SelfConsistencyNonConvergenceError):
        equations.retarded_equations_of_motion(
            0.1,
            trajectory,
            driver,
            1,
            aperture_radius=1.0,
            sim_type=SimulationType.BUNCH_TO_BUNCH,
            chrono_mode=ChronoMatchingMode.FAST,
            startup_mode=StartupMode.COLD_START,
            self_consistency=SelfConsistencyConfig(
                enabled=True,
                max_iterations=1,
                verbosity=0,
            ),
        )


def test_retarded_space_charge_uses_pseudo_grid_source_charge_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trajectory = [
        _make_state(x=[0.0, 10.0], charge=[1.0, 1.0]),
        _make_state(x=[1.0, 11.0], t=[1.0, 1.0], charge=[1.0, 1.0]),
    ]
    driver = [
        _make_state(x=[100.0], charge=[0.0]),
        _make_state(x=[101.0], t=[1.0], charge=[0.0]),
    ]
    seen_nonzero_charges: list[float] = []

    def fake_contrib(**kwargs: object) -> tuple[float, ...]:
        charges = np.asarray(kwargs["samples"].charge, dtype=float)
        nonzero = charges[np.abs(charges) > 1.0e-30]
        seen_nonzero_charges.extend(nonzero.tolist())
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    monkeypatch.setattr(equations, "compute_vectorized_contributions", fake_contrib)

    equations.retarded_equations_of_motion(
        0.1,
        trajectory,
        driver,
        1,
        aperture_radius=1.0,
        sim_type=SimulationType.BUNCH_TO_BUNCH,
        chrono_mode=ChronoMatchingMode.FAST,
        startup_mode=StartupMode.COLD_START,
        self_consistency=SelfConsistencyConfig(enabled=False),
        space_charge=SpaceChargeConfig(
            enabled=True,
            retarded=False,
        ),
        pseudo_grid_space_charge_source_charges=np.array(
            [[0.0, 2.0], [2.0, 0.0]],
            dtype=float,
        ),
    )

    assert seen_nonzero_charges == pytest.approx([2.0, 2.0])


def test_retarded_space_charge_batches_same_bunch_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trajectory = [
        _make_state(x=[0.0, 10.0, 20.0], charge=[1.0, 1.0, 1.0]),
        _make_state(
            x=[1.0, 11.0, 21.0],
            t=[1.0, 1.0, 1.0],
            charge=[1.0, 1.0, 1.0],
        ),
    ]
    driver = [
        _make_state(x=[100.0, 110.0, 120.0], charge=[0.0, 0.0, 0.0]),
        _make_state(
            x=[101.0, 111.0, 121.0],
            t=[1.0, 1.0, 1.0],
            charge=[0.0, 0.0, 0.0],
        ),
    ]
    same_bunch_source_counts: list[int] = []

    def fake_chrono(*args: object, **kwargs: object) -> np.ndarray:
        observer_history = args[0]
        source_history = cast(list[dict[str, np.ndarray]], args[1])
        source_count = len(source_history[-1]["x"])
        if source_history is observer_history:
            same_bunch_source_counts.append(source_count)
        return np.zeros(source_count, dtype=int)

    def fake_contrib(**kwargs: object) -> tuple[float, ...]:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    monkeypatch.setattr(equations, "chrono_match_indices", fake_chrono)
    monkeypatch.setattr(equations, "compute_vectorized_contributions", fake_contrib)

    equations.retarded_equations_of_motion(
        0.1,
        trajectory,
        driver,
        1,
        aperture_radius=1.0,
        sim_type=SimulationType.BUNCH_TO_BUNCH,
        chrono_mode=ChronoMatchingMode.FAST,
        startup_mode=StartupMode.COLD_START,
        self_consistency=SelfConsistencyConfig(enabled=False),
        space_charge=SpaceChargeConfig(
            enabled=True,
            retarded=True,
            min_retarded_steps=0,
        ),
    )

    assert same_bunch_source_counts == [3, 3, 3]


def _build_soa(trajectory: list[dict[str, np.ndarray]]):
    builder = TrajectoryBuilder(len(trajectory), len(trajectory[0]["x"]))
    for step_idx, state in enumerate(trajectory):
        builder.set_step(step_idx, state)
    return builder.build()


def test_retarded_space_charge_uses_chrono_matching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trajectory = [
        _make_state(x=[0.0, 10.0], charge=[1.0, 1.0]),
        _make_state(x=[1.0, 11.0], t=[1.0, 1.0], charge=[1.0, 1.0]),
    ]
    driver = [
        _make_state(x=[100.0], charge=[0.0]),
        _make_state(x=[101.0], t=[1.0], charge=[0.0]),
    ]
    sc_calls: list[dict[str, object]] = []

    def fake_chrono(*args: object, **kwargs: object) -> np.ndarray:
        observer_history = args[0]
        source_history = cast(list[dict[str, np.ndarray]], args[1])
        source_count = len(source_history[-1]["x"])
        if source_history is observer_history:
            sc_calls.append(kwargs)
        return np.zeros(source_count, dtype=int)

    def fake_contrib(**kwargs: object) -> tuple[float, ...]:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    monkeypatch.setattr(equations, "chrono_match_indices", fake_chrono)
    monkeypatch.setattr(equations, "compute_vectorized_contributions", fake_contrib)

    equations.retarded_equations_of_motion(
        0.1,
        trajectory,
        driver,
        1,
        aperture_radius=1.0,
        sim_type=SimulationType.BUNCH_TO_BUNCH,
        chrono_mode=ChronoMatchingMode.FAST,
        startup_mode=StartupMode.COLD_START,
        self_consistency=SelfConsistencyConfig(
            enabled=False,
            chrono_interpolate=True,
            chrono_high_precision=True,
            chrono_adaptive_tolerance=True,
        ),
        space_charge=SpaceChargeConfig(
            enabled=True,
            retarded=True,
            min_retarded_steps=0,
        ),
    )

    assert sc_calls
    assert all(call["mode"] is ChronoMatchingMode.FAST for call in sc_calls)
    assert all(call["interpolate"] is True for call in sc_calls)
    assert all(call["high_precision"] is True for call in sc_calls)


def test_retarded_space_charge_uses_soa_helpers_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trajectory = [
        _make_state(x=[0.0, 10.0], charge=[1.0, 1.0]),
        _make_state(x=[1.0, 11.0], t=[1.0, 1.0], charge=[1.0, 1.0]),
    ]
    driver = [
        _make_state(x=[100.0], charge=[0.0]),
        _make_state(x=[101.0], t=[1.0], charge=[0.0]),
    ]
    traj_soa = _build_soa(trajectory)
    soa_calls: dict[str, int] = {
        "chrono": 0,
        "distance": 0,
        "gather": 0,
    }

    def fake_chrono_soa(*args: object, **kwargs: object) -> np.ndarray:
        soa_calls["chrono"] += 1
        assert args[0] is traj_soa
        assert args[1] is traj_soa
        return np.zeros(traj_soa.n_particles, dtype=int)

    def fake_distance_soa(*args: object, **kwargs: object) -> dict[str, np.ndarray]:
        soa_calls["distance"] += 1
        n = traj_soa.n_particles
        return {
            "R": np.ones(n, dtype=float),
            "nx": np.ones(n, dtype=float),
            "ny": np.zeros(n, dtype=float),
            "nz": np.zeros(n, dtype=float),
        }

    def fake_gather_soa(*args: object, **kwargs: object):
        soa_calls["gather"] += 1
        n = traj_soa.n_particles
        return SimpleNamespace(
            charge=np.ones(n, dtype=float),
            gamma=np.ones(n, dtype=float),
            bx=np.zeros(n, dtype=float),
            by=np.zeros(n, dtype=float),
            bz=np.zeros(n, dtype=float),
            bdotx=np.zeros(n, dtype=float),
            bdoty=np.zeros(n, dtype=float),
            bdotz=np.zeros(n, dtype=float),
            valid_mask=np.ones(n, dtype=bool),
        )

    def fake_contrib(**kwargs: object) -> tuple[float, ...]:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    monkeypatch.setattr(equations, "chrono_match_indices_soa", fake_chrono_soa)
    monkeypatch.setattr(equations, "compute_retarded_distance_soa", fake_distance_soa)
    monkeypatch.setattr(equations, "gather_external_samples_soa", fake_gather_soa)
    monkeypatch.setattr(equations, "compute_vectorized_contributions", fake_contrib)

    equations.retarded_equations_of_motion(
        0.1,
        trajectory,
        driver,
        1,
        aperture_radius=1.0,
        sim_type=SimulationType.BUNCH_TO_BUNCH,
        chrono_mode=ChronoMatchingMode.FAST,
        startup_mode=StartupMode.COLD_START,
        self_consistency=SelfConsistencyConfig(enabled=False),
        space_charge=SpaceChargeConfig(
            enabled=True,
            retarded=True,
            min_retarded_steps=0,
        ),
        traj_soa=traj_soa,
    )

    assert soa_calls == {"chrono": 2, "distance": 2, "gather": 2}


def test_gating_threshold_and_force_application_follow_travel_distance() -> None:
    nhat = {
        "R": np.array([10.0]),
        "nx": np.array([-1.0]),
        "ny": np.array([0.0]),
        "nz": np.array([0.0]),
    }
    current_state = _make_state(
        x=[3.0],
        y=[0.0],
        z=[0.0],
        bx=[0.5],
        by=[0.0],
        bz=[0.0],
    )
    current_state["origin_x"] = np.array([0.0], dtype=float)
    current_state["origin_y"] = np.array([0.0], dtype=float)
    current_state["origin_z"] = np.array([0.0], dtype=float)
    current_state["beta_avg_x"] = np.array([0.5], dtype=float)
    current_state["beta_avg_y"] = np.array([0.0], dtype=float)
    current_state["beta_avg_z"] = np.array([0.0], dtype=float)

    threshold = equations._compute_gating_threshold(nhat, 0.5, 0.0, 0.0)

    assert threshold == pytest.approx(10.0 / 3.0)
    assert (
        equations._should_apply_external_forces(
            StartupMode.COLD_START,
            SimulationType.CONDUCTING_WALL,
            nhat,
            current_state,
            0,
        )
        is False
    )

    current_state["x"][0] = 4.0
    assert (
        equations._should_apply_external_forces(
            StartupMode.COLD_START,
            SimulationType.CONDUCTING_WALL,
            nhat,
            current_state,
            0,
        )
        is True
    )


def test_b2b_cold_start_force_gate_waits_for_observer_travel() -> None:
    nhat = {
        "R": np.array([10.0]),
        "nx": np.array([1.0]),
        "ny": np.array([0.0]),
        "nz": np.array([0.0]),
    }
    current_state = _make_state(x=[0.0], bx=[0.5], by=[0.0], bz=[0.0])

    assert (
        equations._should_apply_external_forces(
            StartupMode.COLD_START,
            SimulationType.BUNCH_TO_BUNCH,
            nhat,
            current_state,
            0,
        )
        is False
    )

    current_state["x"][0] = 10.0
    assert (
        equations._should_apply_external_forces(
            StartupMode.COLD_START,
            SimulationType.BUNCH_TO_BUNCH,
            nhat,
            current_state,
            0,
        )
        is True
    )


def test_b2b_cold_start_early_skip_avoids_retarded_work_before_travel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trajectory = [
        _make_state(x=[0.0], y=[0.0], z=[0.0], bx=[0.5], by=[0.0], bz=[0.0], t=[0.0]),
        _make_state(x=[0.0], y=[0.0], z=[0.0], bx=[0.5], by=[0.0], bz=[0.0], t=[1.0]),
    ]
    driver = [
        _make_state(
            x=[10.0],
            y=[0.0],
            z=[0.0],
            bx=[-0.5],
            by=[0.0],
            bz=[0.0],
            t=[0.0],
        ),
        _make_state(
            x=[10.0],
            y=[0.0],
            z=[0.0],
            bx=[-0.5],
            by=[0.0],
            bz=[0.0],
            t=[1.0],
        ),
    ]

    chrono_calls = {"count": 0}

    def fail_if_called(*args: object, **kwargs: object) -> object:
        chrono_calls["count"] += 1
        raise AssertionError("retarded distance machinery should be skipped")

    monkeypatch.setattr(equations, "_compute_full_retarded_distance", fail_if_called)
    result = equations.retarded_equations_of_motion(
        0.1,
        trajectory,
        driver,
        1,
        aperture_radius=1.0,
        sim_type=SimulationType.BUNCH_TO_BUNCH,
        chrono_mode=ChronoMatchingMode.FAST,
        startup_mode=StartupMode.COLD_START,
        self_consistency=SelfConsistencyConfig(enabled=False),
    )

    assert chrono_calls["count"] == 0
    assert result["Px"][0] == pytest.approx(trajectory[1]["Px"][0])


def test_gating_threshold_handles_receding_case_with_large_threshold() -> None:
    nhat = {
        "R": np.array([5.0]),
        "nx": np.array([1.0]),
        "ny": np.array([0.0]),
        "nz": np.array([0.0]),
    }

    threshold = equations._compute_gating_threshold(nhat, 1.0, 0.0, 0.0)

    assert threshold == pytest.approx(1e12)


def test_gating_threshold_small_array_matches_vector_path() -> None:
    nhat_small = {
        "R": np.array([2.0, 4.0, 8.0, 16.0]),
        "nx": np.array([0.2, -0.3, 0.4, -0.5]),
        "ny": np.array([0.1, 0.2, -0.1, -0.2]),
        "nz": np.array([0.9, 0.8, -0.7, -0.6]),
    }
    nhat_large = {
        key: np.concatenate([value, value, value]) for key, value in nhat_small.items()
    }

    small_threshold = equations._compute_gating_threshold(nhat_small, 0.25, -0.05, 0.35)
    large_threshold = equations._compute_gating_threshold(nhat_large, 0.25, -0.05, 0.35)

    assert small_threshold == pytest.approx(large_threshold)


def test_get_current_particle_gamma_and_beta_switches_to_iterated_state() -> None:
    current_state = _make_state(bx=[0.1], by=[0.2], bz=[0.3], gamma=[4.0])
    result_state = _make_state(bx=[0.4], by=[0.5], bz=[0.6], gamma=[7.0])

    gamma0, beta0 = equations._get_current_particle_gamma_and_beta(
        current_state,
        result_state,
        particle_idx=0,
        sc_iteration=0,
        sc_enabled=True,
    )
    gamma1, beta1 = equations._get_current_particle_gamma_and_beta(
        current_state,
        result_state,
        particle_idx=0,
        sc_iteration=1,
        sc_enabled=True,
    )

    assert gamma0 == pytest.approx(4.0)
    assert beta0 == pytest.approx((0.1, 0.2, 0.3))
    assert gamma1 == pytest.approx(7.0)
    assert beta1 == pytest.approx((0.4, 0.5, 0.6))


def test_beta_helpers_limit_and_recover_gamma() -> None:
    limited = equations._limit_beta_magnitude(1.0, 1.0, 1.0)
    assert max(abs(component) for component in limited) < 1.0
    assert equations._calculate_one_minus_beta_squared(*limited) > 0.0
    assert equations._calculate_one_minus_beta_squared(0.0, 0.0, 0.0) == pytest.approx(
        1.0
    )
    assert equations._calculate_gamma_from_beta(0.0, 0.0, 0.0) == pytest.approx(1.0)
    assert equations._calculate_one_minus_beta_squared(1.0, 0.0, 0.0) > 0.0


def test_medina_kinematics_preserve_longitudinal_cancellation() -> None:
    gamma = 10.0
    mass = 1.0
    charge = 1.0
    beta_z = np.sqrt(1.0 - 1.0 / gamma**2)
    force_z = 5.0

    beta_dot_t, dgamma_dt = equations._derive_relativistic_kinematics_from_force(
        (0.0, 0.0, force_z),
        (0.0, 0.0, beta_z),
        gamma,
        mass,
    )

    assert beta_dot_t[2] == pytest.approx(force_z / (gamma**3 * mass * C_MMNS))
    assert dgamma_dt == pytest.approx(beta_z * force_z / (mass * C_MMNS))

    impulse, capped = equations._compute_medina_radiation_reaction_impulse(
        external_force=(0.0, 0.0, force_z),
        beta=(0.0, 0.0, beta_z),
        beta_dot_t=beta_dot_t,
        gamma=gamma,
        dgamma_dt=dgamma_dt,
        mass=mass,
        charge=charge,
        coordinate_dt=1.0,
        max_impulse_fraction=0.0,
    )

    assert capped is False
    assert np.linalg.norm(impulse) == pytest.approx(0.0, abs=1.0e-20)


def test_medina_kinematics_give_transverse_synchrotron_damping() -> None:
    gamma = 10.0
    mass = 1.0
    charge = 1.0
    beta_z = np.sqrt(1.0 - 1.0 / gamma**2)
    force_x = 5.0

    beta_dot_t, dgamma_dt = equations._derive_relativistic_kinematics_from_force(
        (force_x, 0.0, 0.0),
        (0.0, 0.0, beta_z),
        gamma,
        mass,
    )
    impulse, capped = equations._compute_medina_radiation_reaction_impulse(
        external_force=(force_x, 0.0, 0.0),
        beta=(0.0, 0.0, beta_z),
        beta_dot_t=beta_dot_t,
        gamma=gamma,
        dgamma_dt=dgamma_dt,
        mass=mass,
        charge=charge,
        coordinate_dt=1.0,
        max_impulse_fraction=0.0,
    )

    assert capped is False
    assert dgamma_dt == pytest.approx(0.0)
    assert beta_dot_t[0] == pytest.approx(force_x / (gamma * mass * C_MMNS))
    assert impulse[0] == pytest.approx(0.0)
    assert impulse[1] == pytest.approx(0.0)
    assert impulse[2] < 0.0


def test_running_average_helper_matches_closed_form() -> None:
    average, sample_count = equations._update_beta_running_average(
        previous_avg=(0.2, 0.4, 0.6),
        previous_sample_count=2.0,
        new_beta=(0.8, 1.0, 1.2),
    )

    assert average == pytest.approx((0.4, 0.6, 0.8))
    assert sample_count == pytest.approx(3.0)


def test_convergence_helpers_and_logging_report_expected_values() -> None:
    converged, mass_shell_error = equations._check_mass_shell_convergence(
        Pt=5.0,
        Px=3.0,
        Py=0.0,
        Pz=0.0,
        particle_mass=4.0 / C_MMNS,
        C_MMNS=C_MMNS,
        tolerance=1e-9,
    )
    gamma_ok, gamma_error = equations._check_gamma_consistency(
        gamma_velocity=10.0,
        gamma_energy=10.1,
        tolerance=0.02,
    )

    assert converged is True
    assert mass_shell_error == pytest.approx(0.0)
    assert gamma_ok is True
    assert gamma_error == pytest.approx(0.01)

    stream = io.StringIO()
    with redirect_stdout(stream):
        equations._print_convergence_info(
            particle_idx=2,
            iteration=1,
            gamma_from_velocity=10.0,
            gamma_from_energy=9.9,
            gamma_mass_shell=10.1,
            mass_shell_error=1e-4,
            gamma_consistency_error=2e-4,
            converged=False,
            max_iterations=5,
            verbosity=2,
            step_idx=7,
            particle_position=(1.0, 2.0, 3.0),
            particle_time=4.0,
        )

    output = stream.getvalue()
    assert "Step 7, Particle 2" in output
    assert "Mass-shell error" in output
    assert "Position: x=1.000000e+00 mm" in output


def test_mass_shell_convergence_uses_kinetic_and_mechanical_momentum() -> None:
    scalar_potential_momentum = 0.25 * C_MMNS
    vector_potential_x = 0.5 * C_MMNS

    converged, mass_shell_error = equations._check_mass_shell_convergence(
        Pt=C_MMNS + scalar_potential_momentum,
        Px=vector_potential_x,
        Py=0.0,
        Pz=0.0,
        particle_mass=1.0,
        C_MMNS=C_MMNS,
        tolerance=1e-12,
        scalar_potential_contribution=scalar_potential_momentum,
        field_x=vector_potential_x,
    )

    assert converged is True
    assert mass_shell_error == pytest.approx(0.0)


def test_convergence_logging_summary_and_full_detail_modes() -> None:
    summary_stream = io.StringIO()
    with redirect_stdout(summary_stream):
        equations._print_convergence_info(
            particle_idx=1,
            iteration=0,
            gamma_from_velocity=2.0,
            gamma_from_energy=2.0,
            gamma_mass_shell=2.0,
            mass_shell_error=1e-5,
            gamma_consistency_error=2e-5,
            converged=True,
            max_iterations=4,
            verbosity=1,
            step_idx=2,
            convergence_mode="variable_geometry",
        )

    full_stream = io.StringIO()
    with redirect_stdout(full_stream):
        equations._print_convergence_info(
            particle_idx=3,
            iteration=2,
            gamma_from_velocity=4.0,
            gamma_from_energy=3.5,
            gamma_mass_shell=3.8,
            mass_shell_error=1e-3,
            gamma_consistency_error=2e-3,
            converged=False,
            max_iterations=4,
            verbosity=3,
            particle_position=(1.0, -2.0, 3.0),
            particle_time=4.5,
        )

    assert "Step 2, P1: converged in 1 iter" in summary_stream.getvalue()
    full_output = full_stream.getvalue()
    assert "Particle 3: max iter (4) reached" in full_output
    assert "Time: t=4.500000e+00 ns" in full_output


def test_retarded_equations_of_motion_respects_cancellation_callback() -> None:
    state = _make_state()

    with pytest.raises(IntegrationCancelled):
        equations.retarded_equations_of_motion(
            h=1e-3,
            trajectory=[state],
            trajectory_ext=[state],
            index_traj=0,
            aperture_radius=1.0,
            sim_type=SimulationType.CONDUCTING_WALL,
            cancel_callback=lambda: True,
        )


def test_retarded_equations_of_motion_skips_already_dead_particles() -> None:
    state = _make_state(x=[1.0], y=[2.0], z=[3.0], gamma=[4.0], charge=[5.0])
    mark_particle_dead(state, 0, step=5, reason="synthetic")

    result = equations.retarded_equations_of_motion(
        h=1e-3,
        trajectory=[state],
        trajectory_ext=[state],
        index_traj=0,
        aperture_radius=1.0,
        sim_type=SimulationType.CONDUCTING_WALL,
    )

    for key in [
        "x",
        "y",
        "z",
        "t",
        "bx",
        "by",
        "bz",
        "gamma",
        "Px",
        "Py",
        "Pz",
        "Pt",
        "bdotx",
        "bdoty",
        "bdotz",
    ]:
        assert result[key][0] == pytest.approx(state[key][0])
    assert result["_dead_particles"][0]


def test_retarded_equations_of_motion_raises_gamma_blowup_for_sc_runs() -> None:
    state = _make_state(gamma=[1e9])

    with pytest.raises(equations.GammaBlowupError):
        equations.retarded_equations_of_motion(
            h=1e-3,
            trajectory=[state],
            trajectory_ext=[state],
            index_traj=0,
            aperture_radius=1.0,
            sim_type=SimulationType.CONDUCTING_WALL,
            self_consistency=SelfConsistencyConfig(enabled=True, max_iterations=1),
            step_idx=3,
        )


def test_retarded_equations_of_motion_applies_final_mass_shell_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _make_state(gamma=[1.0], pt=[10.0], charge=[0.0])
    _patch_mass_shell_convergence_sequence(
        monkeypatch,
        [(True, 0.0), (False, 0.69)],
    )

    result = equations.retarded_equations_of_motion(
        h=1e-3,
        trajectory=[state],
        trajectory_ext=[state],
        index_traj=0,
        aperture_radius=1.0,
        sim_type=SimulationType.CONDUCTING_WALL,
        self_consistency=SelfConsistencyConfig(
            enabled=True,
            max_iterations=2,
            mass_shell_tolerance=1e-9,
        ),
    )

    expected_pt = pytest.approx(C_MMNS)
    assert result["Pt"][0] == expected_pt
    assert result["gamma"][0] == pytest.approx(1.0)


def test_final_mass_shell_projection_uses_mechanical_momentum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _make_state(gamma=[1.0], charge=[1.0], mass=[1.0])
    source = _make_state(x=[1.0], charge=[1.0], mass=[1.0])
    vector_potential_x = 0.5 * C_MMNS
    scalar_potential = 0.25 * C_MMNS
    scalar_potential_momentum = equations._scalar_potential_momentum_contribution(
        equations.effective_observer_charge(float(state["q"][0])), scalar_potential
    )

    monkeypatch.setattr(
        equations,
        "_compute_approximate_retarded_distance",
        lambda *args, **kwargs: (
            {
                "R": np.array([1.0]),
                "nx": np.array([1.0]),
                "ny": np.array([0.0]),
                "nz": np.array([0.0]),
            },
            np.array([0]),
        ),
    )
    monkeypatch.setattr(equations, "_should_apply_external_forces", lambda *args: True)

    def fake_contributions(**kwargs: object) -> tuple[float, ...]:
        del kwargs
        return (
            vector_potential_x,
            0.0,
            0.0,
            scalar_potential_momentum,
            vector_potential_x,
            0.0,
            0.0,
            scalar_potential,
        )

    monkeypatch.setattr(
        equations,
        "compute_vectorized_contributions",
        fake_contributions,
    )
    _patch_mass_shell_convergence_sequence(
        monkeypatch,
        [(True, 0.0), (False, 0.69)],
    )

    result = equations.retarded_equations_of_motion(
        h=1e-3,
        trajectory=[state],
        trajectory_ext=[source],
        index_traj=0,
        aperture_radius=1.0,
        sim_type=SimulationType.BUNCH_TO_BUNCH,
        startup_mode=StartupMode.APPROXIMATE_BACK_HISTORY,
        self_consistency=SelfConsistencyConfig(
            enabled=True,
            max_iterations=2,
            mass_shell_tolerance=1e-12,
        ),
    )

    assert result["Px"][0] == pytest.approx(vector_potential_x)
    assert result["Pt"][0] == pytest.approx(C_MMNS + scalar_potential_momentum)
    assert result["gamma"][0] == pytest.approx(1.0)


def test_sc_iteration_projection_uses_mechanical_momentum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _make_state(gamma=[1.0], charge=[1.0], mass=[1.0])
    source = _make_state(x=[1.0], charge=[1.0], mass=[1.0])
    vector_potential_x = 0.5 * C_MMNS
    scalar_potential = 0.25 * C_MMNS
    scalar_potential_momentum = equations._scalar_potential_momentum_contribution(
        equations.effective_observer_charge(float(state["q"][0])), scalar_potential
    )

    monkeypatch.setattr(
        equations,
        "_compute_approximate_retarded_distance",
        lambda *args, **kwargs: (
            {
                "R": np.array([1.0]),
                "nx": np.array([1.0]),
                "ny": np.array([0.0]),
                "nz": np.array([0.0]),
            },
            np.array([0]),
        ),
    )
    monkeypatch.setattr(equations, "_should_apply_external_forces", lambda *args: True)

    def fake_contributions(**kwargs: object) -> tuple[float, ...]:
        del kwargs
        return (
            vector_potential_x,
            0.0,
            0.0,
            scalar_potential_momentum,
            vector_potential_x,
            0.0,
            0.0,
            scalar_potential,
        )

    monkeypatch.setattr(
        equations, "compute_vectorized_contributions", fake_contributions
    )

    result = equations.retarded_equations_of_motion(
        h=1e-3,
        trajectory=[state],
        trajectory_ext=[source],
        index_traj=0,
        aperture_radius=1.0,
        sim_type=SimulationType.BUNCH_TO_BUNCH,
        startup_mode=StartupMode.APPROXIMATE_BACK_HISTORY,
        self_consistency=SelfConsistencyConfig(
            enabled=True,
            max_iterations=2,
            mass_shell_relaxation=1.0,
            mass_shell_tolerance=1e-12,
        ),
    )

    assert result["Px"][0] == pytest.approx(vector_potential_x)
    assert result["Pt"][0] == pytest.approx(C_MMNS + scalar_potential_momentum)
    assert result["gamma"][0] == pytest.approx(1.0)


def test_retarded_equations_of_motion_can_cancel_during_sc_iteration() -> None:
    state = _make_state()
    calls = {"count": 0}

    def _cancel() -> bool:
        calls["count"] += 1
        return calls["count"] >= 2

    with pytest.raises(IntegrationCancelled):
        equations.retarded_equations_of_motion(
            h=1e-3,
            trajectory=[state],
            trajectory_ext=[state],
            index_traj=0,
            aperture_radius=1.0,
            sim_type=SimulationType.CONDUCTING_WALL,
            self_consistency=SelfConsistencyConfig(enabled=True, max_iterations=2),
            cancel_callback=_cancel,
        )


@pytest.mark.parametrize(
    ("method", "extra_kwargs", "expected_gamma"),
    [
        # USE_VELOCITY: Px=0 so mass-shell is violated (Pt=2mc but P=0 implies
        # gamma=1). Post-loop projection resets gamma to 1.0.
        (GammaReconciliationMethod.USE_VELOCITY, {}, 1.0),
        # USE_ENERGY: Px=sqrt(3)*mc, gamma=2 satisfies mass shell. gamma=2.0.
        (GammaReconciliationMethod.USE_ENERGY, {}, 2.0),
        # FIXED_WEIGHTED / ADAPTIVE_WEIGHTED: reconciliation now only seeds the
        # next SC iteration, not the stored result.  Mass shell is satisfied
        # (Px=sqrt(3)*mc, gamma=2), so gamma stays at 2.0.
        (
            GammaReconciliationMethod.FIXED_WEIGHTED,
            {"gamma_reconciliation_fixed_weight": 0.25},
            2.0,
        ),
        (GammaReconciliationMethod.ADAPTIVE_WEIGHTED, {}, 2.0),
    ],
)
def test_retarded_equations_of_motion_reconciles_gamma_modes(
    monkeypatch: pytest.MonkeyPatch,
    method: GammaReconciliationMethod,
    extra_kwargs: dict[str, float],
    expected_gamma: float,
) -> None:
    state = _make_state(gamma=[2.0], charge=[0.0])
    if method is not GammaReconciliationMethod.USE_VELOCITY:
        state["Px"][0] = np.sqrt(3.0) * C_MMNS
    monkeypatch.setattr(equations, "_calculate_gamma_from_beta", lambda *args: 1.0)
    _patch_mass_shell_convergence_sequence(
        monkeypatch,
        [(True, 0.0), (False, 0.69)],
    )
    config = SelfConsistencyConfig(
        enabled=True,
        max_iterations=2,
        gamma_reconciliation_method=method,
        **extra_kwargs,
    )

    result = equations.retarded_equations_of_motion(
        h=1e-3,
        trajectory=[state],
        trajectory_ext=[state],
        index_traj=0,
        aperture_radius=1.0,
        sim_type=SimulationType.CONDUCTING_WALL,
        self_consistency=config,
    )

    assert result["gamma"][0] == pytest.approx(expected_gamma)
    assert result["Pt"][0] == pytest.approx(expected_gamma * C_MMNS)


def test_gamma_reconciliation_refreshes_kinematics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _make_state(gamma=[2.0], charge=[0.0])
    state["Px"][0] = np.sqrt(3.0) * C_MMNS
    monkeypatch.setattr(equations, "_calculate_gamma_from_beta", lambda *args: 1.0)
    _patch_mass_shell_convergence_sequence(
        monkeypatch,
        [(True, 0.0), (True, 0.0)],
    )
    config = SelfConsistencyConfig(
        enabled=True,
        max_iterations=2,
        gamma_reconciliation_method=GammaReconciliationMethod.FIXED_WEIGHTED,
        gamma_reconciliation_fixed_weight=0.25,
    )

    result = equations.retarded_equations_of_motion(
        h=1e-3,
        trajectory=[state],
        trajectory_ext=[state],
        index_traj=0,
        aperture_radius=1.0,
        sim_type=SimulationType.CONDUCTING_WALL,
        self_consistency=config,
    )

    # Reconciliation no longer modifies the stored result — it only seeds the
    # working state for subsequent SC iterations.  With Px=sqrt(3)*mc and
    # gamma=2 the mass shell is satisfied (no post-loop projection), so the
    # final kinematics are derived from the energy-based gamma (2.0) and the
    # spatial momentum, not from the blended reconciliation value.
    expected_gamma = 2.0
    expected_beta_x = np.sqrt(expected_gamma**2 - 1.0) / expected_gamma
    assert result["gamma"][0] == pytest.approx(expected_gamma)
    assert result["t"][0] == pytest.approx(1e-3 * expected_gamma)
    assert result["bx"][0] == pytest.approx(expected_beta_x)
    assert result["x"][0] == pytest.approx(
        1e-3 * C_MMNS * np.sqrt(expected_gamma**2 - 1.0)
    )


def test_gamma_reconciliation_preserves_potential_bookkeeping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _make_state(gamma=[2.0], charge=[1.0], mass=[1.0])
    source = _make_state(x=[1.0], charge=[1.0], mass=[1.0])

    monkeypatch.setattr(
        equations,
        "_compute_approximate_retarded_distance",
        lambda *args, **kwargs: (
            {
                "R": np.array([1.0]),
                "nx": np.array([1.0]),
                "ny": np.array([0.0]),
                "nz": np.array([0.0]),
            },
            np.array([0]),
        ),
    )
    monkeypatch.setattr(equations, "_should_apply_external_forces", lambda *args: True)

    scalar_potential = 5.0
    scalar_potential_momentum = equations._scalar_potential_momentum_contribution(
        equations.effective_observer_charge(float(state["q"][0])), scalar_potential
    )

    def fake_contributions(**kwargs: object) -> tuple[float, ...]:
        del kwargs
        # Canonical Px=17 with vector-potential field Ax=7 means mechanical Px=10.
        # Delta Pt equals qPhi/c, so gamma_energy remains the state's initial gamma=2.
        return (
            17.0,
            0.0,
            0.0,
            scalar_potential_momentum,
            7.0,
            0.0,
            0.0,
            scalar_potential,
        )

    monkeypatch.setattr(
        equations,
        "compute_vectorized_contributions",
        fake_contributions,
    )
    monkeypatch.setattr(equations, "_calculate_gamma_from_beta", lambda *args: 1.0)
    _patch_mass_shell_convergence_sequence(
        monkeypatch,
        [(True, 0.0), (False, 0.69)],
    )
    config = SelfConsistencyConfig(
        enabled=True,
        max_iterations=2,
        gamma_reconciliation_method=GammaReconciliationMethod.FIXED_WEIGHTED,
        gamma_reconciliation_fixed_weight=0.25,
    )

    result = equations.retarded_equations_of_motion(
        h=1e-3,
        trajectory=[state],
        trajectory_ext=[source],
        index_traj=0,
        aperture_radius=1.0,
        sim_type=SimulationType.BUNCH_TO_BUNCH,
        startup_mode=StartupMode.APPROXIMATE_BACK_HISTORY,
        self_consistency=config,
    )

    # Reconciliation now only seeds the working state; it does not write back
    # to result["gamma"]/result["Pt"]/result["Px"].
    #
    # After the force step: canonical Px=17, accumulated_field_x=7, so
    # mechanical Px=10.  gamma_from_energy=2 (since dPt = scalar_potential_momentum
    # exactly cancels phi contribution).  The mass-shell check sees
    # Pt_kinetic=2mc vs P_mech=10 → large error, so the post-loop projection
    # fires and resets gamma to sqrt(1+(10/mc)^2) and Pt accordingly.
    expected_mechanical_px = 10.0
    expected_gamma = float(
        np.sqrt(1.0 + (expected_mechanical_px / C_MMNS) ** 2)
    )
    expected_pt = float(
        np.sqrt(expected_mechanical_px**2 + C_MMNS**2) + scalar_potential_momentum
    )
    assert result["gamma"][0] == pytest.approx(expected_gamma, rel=1e-9)
    assert result["Pt"][0] == pytest.approx(expected_pt, rel=1e-9)
    # Canonical Px is unchanged by the post-loop projection
    assert result["Px"][0] == pytest.approx(17.0)


def test_retarded_equations_of_motion_variable_geometry_uses_updated_observer_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _make_state(x=[1.0], gamma=[2.0], charge=[0.0])
    state["Px"][0] = 2.0
    recorded_x: list[float] = []
    original_compute = equations._compute_full_retarded_distance

    def fake_compute(
        trajectory: list[dict[str, np.ndarray]],
        trajectory_ext: list[dict[str, np.ndarray]],
        time_step_idx: int,
        particle_idx: int,
        chrono_mode: ChronoMatchingMode,
        self_consistency: SelfConsistencyConfig | None = None,
        timestep_h: float = 1e-3,
    ) -> tuple[dict[str, np.ndarray], np.ndarray, None]:
        recorded_x.append(float(trajectory[time_step_idx]["x"][particle_idx]))
        return original_compute(
            trajectory,
            trajectory_ext,
            time_step_idx,
            particle_idx,
            chrono_mode,
            self_consistency,
            timestep_h,
        )

    monkeypatch.setattr(equations, "_compute_full_retarded_distance", fake_compute)
    _patch_mass_shell_convergence_sequence(
        monkeypatch,
        [(True, 0.0), (True, 0.0)],
    )

    equations.retarded_equations_of_motion(
        h=1e-3,
        trajectory=[state],
        trajectory_ext=[state],
        index_traj=0,
        aperture_radius=1.0,
        sim_type=SimulationType.CONDUCTING_WALL,
        self_consistency=SelfConsistencyConfig(
            enabled=True,
            convergence_mode="variable_geometry",
            max_iterations=2,
        ),
    )

    assert recorded_x[0] == pytest.approx(1.0)
    assert recorded_x[1] > recorded_x[0]
