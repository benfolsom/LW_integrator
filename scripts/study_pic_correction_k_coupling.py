"""Serial coupled PIC/Medina evidence; all artifacts stay outside the checkout."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.pic import simulation
from core.pic.coupling import coupled_kick_ledger
from core.pic.grid import ElectrostaticPIC, Species
from core.pic.kernels import C, COULOMB
from core.pic.medina import MedinaPIC

OUT = Path.home() / "compute-data/pic_correction_k/8_coupling"


def config(steps: int, rr: str = "off", midpoint: bool = True) -> dict:
    rng = np.random.default_rng(804)
    species = []
    for name, charge, mass, center, beta in (
        ("electron", -1.602176634e-19, 9.1093837015e-31, [-1, 0, 2], -0.35),
        ("proton", 1.602176634e-19, 1.67262192369e-27, [1, 0, -2], 0.30),
    ):
        local = rng.normal(size=(4, 3)) * 0.04
        position = np.concatenate((local, -local)) + center
        u = np.tile([0, 0, beta / np.sqrt(1 - beta**2)], (8, 1))
        u[:, 0] = np.linspace(-0.001, 0.001, 8)
        species.append(
            dict(
                name=name,
                charge_c=charge,
                mass_kg=mass,
                positions_mm=position.tolist(),
                momentum_mc=u.tolist(),
                population=np.linspace(0.8e7, 1.2e7, 8).tolist(),
            )
        )
    return dict(
        mode="pic",
        backend="cpu",
        radiation_reaction_mode=rr,
        grid=dict(shape=[12] * 3, half_extent_mm=[4] * 3),
        species=species,
        steps=steps,
        propagation_distance_mm=5,
        sample_every=max(1, (steps - 1) // 12),
        correction=dict(
            clouds_per_species=2,
            subcharge_count=4,
            lattice_shape=[4] * 3,
            half_extent_mm=[0.4] * 3,
            midpoint_predictor=midpoint,
        ),
    )


def coupled_run(steps: int, rr: str, midpoint: bool = True) -> dict:
    non_rr_forces, rr_traces = {}, []
    call_counts = {}
    previous_baseline = {}
    previous_full = {}
    kick_totals = []
    original_solver, original_medina = simulation.ElectrostaticPIC, simulation.MedinaPIC
    original_ledger = simulation.coupled_kick_ledger

    class RecordingSolver(ElectrostaticPIC):
        def push(self, species, electric, magnetic, dt):
            pushed = super().push(species, electric, magnetic, dt)
            count = call_counts.get(species.name, 0)
            if count % 2 == 0:
                non_rr_forces[species.name] = (
                    (pushed - species.momentum_mc) * species.mass_kg * C / dt
                )
            call_counts[species.name] = count + 1
            return pushed

    class RecordingMedina(MedinaPIC):
        def apply(self, species, non_rr_u, sample_time, dt):
            full = (non_rr_u - species.momentum_mc) * species.mass_kg * C / dt
            baseline = non_rr_forces[species.name]
            previous = previous_baseline.get(species.name)
            after, ledger = super().apply(species, non_rr_u, sample_time, dt)
            if previous is not None:
                dbase = (baseline - previous) / dt
                dfull = (full - previous_full[species.name]) / dt
                rr_traces.append(
                    dict(
                        time_s=sample_time,
                        species=species.name,
                        full_derivative_norm=float(np.linalg.norm(dfull)),
                        correction_derivative_norm=float(np.linalg.norm(dfull - dbase)),
                        rr_work_residual_j=ledger["work_residual_j"],
                        rr_kick_energy_j=ledger["kick_energy_j"],
                    )
                )
            previous_baseline[species.name] = baseline.copy()
            previous_full[species.name] = full.copy()
            return after, ledger

    def record_kick(*args):
        ledger = coupled_kick_ledger(*args)
        kick_totals.append(ledger)
        return ledger

    simulation.ElectrostaticPIC, simulation.MedinaPIC = RecordingSolver, RecordingMedina
    simulation.coupled_kick_ledger = record_kick
    start = time.perf_counter()
    try:
        result = simulation.run_pic(config(steps, rr, midpoint))
    finally:
        simulation.ElectrostaticPIC, simulation.MedinaPIC = (
            original_solver,
            original_medina,
        )
        simulation.coupled_kick_ledger = original_ledger
    result["elapsed_s"] = time.perf_counter() - start
    assert all(
        np.all(np.isfinite(s["momentum_mc"])) and np.all(np.isfinite(s["positions_mm"]))
        for s in result["final_species"]
    )
    projections = [row["cloud_projection"] for row in result["diagnostics"]]
    assert all(
        row["max_momentum_fit_error_mc"] == 0
        and row["max_velocity_fit_error_m_s"] == 0
        and row["max_position_fit_error_m"] == 0
        for row in projections
    )
    final = result["diagnostics"][-1]
    scale_energy = sum(abs(k["kick_energy_j"]) for k in kick_totals)
    scale_impulse = sum(np.linalg.norm(k["kick_momentum_kg_m_s"]) for k in kick_totals)
    coupled = final["coupled_kick"]
    summary = dict(
        steps=steps,
        rr=rr,
        midpoint=midpoint,
        elapsed_s=result["elapsed_s"],
        dt_s=result["lab_timestep_s"],
        correction_energy_scale_j=scale_energy,
        correction_impulse_scale_kg_m_s=scale_impulse,
        correction_work_relative_residual=abs(coupled["work_residual_j"])
        / scale_energy,
        correction_impulse_relative_residual=np.linalg.norm(
            coupled["impulse_residual_kg_m_s"]
        )
        / scale_impulse,
        old_work_relative_residual=abs(final["correction"]["work_minus_kick_energy_j"])
        / scale_energy,
        total_work_relative_residual=abs(
            coupled["total_work_j"] - coupled["total_kick_energy_j"]
        )
        / sum(abs(k["total_kick_energy_j"]) for k in kick_totals),
        total_impulse_relative_residual=np.linalg.norm(
            np.array(coupled["total_impulse_kg_m_s"])
            - coupled["total_kick_momentum_kg_m_s"]
        )
        / sum(np.linalg.norm(k["total_kick_momentum_kg_m_s"]) for k in kick_totals),
        max_projection_error_mc=max(
            p["max_momentum_fit_error_mc"] for p in projections
        ),
        max_cloud_momentum_residual_kg_m_s=max(
            np.linalg.norm(p["cloud_minus_particle_momentum_kg_m_s"])
            for p in projections
        ),
        coupled_kick=coupled,
    )
    if rr_traces:
        summary["correction_derivative_fraction"] = float(
            np.linalg.norm([r["correction_derivative_norm"] for r in rr_traces])
            / np.linalg.norm([r["full_derivative_norm"] for r in rr_traces])
        )
        summary["medina"] = final["radiation_reaction"]
    stem = f"coupled_{steps}_{rr}_{'midpoint' if midpoint else 'accepted'}"
    (OUT / f"{stem}.json").write_text(json.dumps(result, indent=2) + "\n")
    if rr_traces:
        (OUT / f"{stem}_force_trace.json").write_text(
            json.dumps(rr_traces, indent=2) + "\n"
        )
    return summary


def derivative_control() -> list[dict]:
    """Smooth full-force secants against an independent analytic derivative.

    Force contains a constant component and a rotating correction. The
    physical electron response time is retained; this prescribed strong
    force control makes the RR kick numerically resolvable.
    """
    rows = []
    duration, amplitude = 2e-13, 1e-9
    for intervals in (32, 64, 128, 256):
        dt, rr = duration / intervals, MedinaPIC()
        s = Species(
            "electron",
            [[0, 0, 0]],
            [[0.2, 0.1, 0.4]],
            -1.602176634e-19,
            9.1093837015e-31,
            [7e5],
        )
        errors, exacts, force_errors = [], [], []
        totals = {}
        for j in range(intervals):
            t = (j + 0.5) * dt
            phase = 2 * np.pi * t / duration
            f = amplitude * np.array([[1, 0.3 * np.sin(phase), 0.2 * np.cos(phase)]])
            df = (
                amplitude
                * 2
                * np.pi
                / duration
                * np.array([[0, 0.3 * np.cos(phase), -0.2 * np.sin(phase)]])
            )
            predicted = s.momentum_mc + dt * f / (s.mass_kg * C)
            before = rr.previous.get(s.name)
            after, ledger = rr.apply(s, predicted, t, dt)
            if before is not None:
                actual_force = rr.previous[s.name][1]
                secant = (actual_force - before[1]) / dt
                errors.append(secant - df)
                exacts.append(df)
                gamma = np.sqrt(1 + np.sum(predicted**2, axis=1))[:, None]
                beta = predicted / gamma
                tau = 2 * COULOMB * s.charge_c**2 / (3 * s.mass_kg * C**3)
                # Only dF/dt changes between analytic and secant RR inputs.
                force_errors.append(tau * gamma * (secant - df))
            for key, value in ledger.items():
                totals[key] = totals.get(key, 0) + np.asarray(value)
            s.momentum_mc = after
        residual = abs(float(totals["model_energy_residual_j"]))
        rows.append(
            dict(
                intervals=intervals,
                dt_s=dt,
                derivative_relative_error=float(
                    np.linalg.norm(errors) / np.linalg.norm(exacts)
                ),
                rr_force_error_n=float(np.linalg.norm(force_errors)),
                model_energy_relative_residual=residual
                / max(
                    abs(float(totals["model_work_j"])),
                    float(totals["far_radiated_energy_j"]),
                ),
                model_momentum_residual_kg_m_s=np.asarray(
                    totals["model_momentum_residual_kg_m_s"]
                ).tolist(),
                applied_work_j=float(totals["applied_work_j"]),
                kick_energy_j=float(totals["kick_energy_j"]),
                far_radiated_energy_j=float(totals["far_radiated_energy_j"]),
                cross_field_energy_change_j=float(
                    totals["cross_field_energy_change_j"]
                ),
                work_quadrature_difference_j=float(
                    totals["work_quadrature_difference_j"]
                ),
                rr_work_residual_j=float(totals["work_residual_j"]),
            )
        )
    for coarse, fine in zip(rows, rows[1:]):
        fine["derivative_order"] = float(
            np.log2(
                coarse["derivative_relative_error"] / fine["derivative_relative_error"]
            )
        )
    assert rows[-1]["derivative_order"] > 0.95
    return rows


def default_replay() -> dict:
    """Read the committed runner, then compare unchanged RR-off trajectories."""
    source = subprocess.check_output(["git", "show", "HEAD:core/pic/simulation.py"])
    path = OUT / "checkpoint_simulation.py"
    path.write_bytes(source)
    spec = importlib.util.spec_from_file_location(
        "core.pic.checkpoint_simulation", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checks = []
    for mode in ("disabled", "default", "midpoint"):
        cfg = config(9, midpoint=mode == "midpoint")
        cfg["propagation_distance_mm"] = 0.01
        if mode == "disabled":
            cfg.pop("correction")
        old, new = module.run_pic(cfg), simulation.run_pic(cfg)
        assert old["final_species"] == new["final_species"]
        for a, b in zip(old["diagnostics"], new["diagnostics"]):
            for key in (
                "kinetic_energy_j",
                "momentum_kg_m_s",
                "grid_field_energy_j",
                "max_relative_continuity_residual",
            ):
                if key in a:
                    assert a[key] == b[key]
            if "correction" in a:
                assert a["correction"] == b["correction"]
        checks.append(mode)
    return dict(
        modes=checks,
        final_states_bitwise_equal=True,
        checkpoint_sha256=hashlib.sha256(source).hexdigest(),
    )


def main() -> None:
    assert int(Path("/private/tmp/compute_cap").read_text()) >= 2
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(OUT / "mpl_cache"))
    result = dict(
        default_replay=default_replay(),
        derivative_control=derivative_control(),
        coupled=[],
    )
    print("Replay and analytic derivative controls passed", flush=True)
    for steps, rr, midpoint in (
        (1201, "off", False),
        (1201, "off", True),
        (301, "medina_lad", True),
        (601, "medina_lad", True),
        (1201, "medina_lad", True),
    ):
        row = coupled_run(steps, rr, midpoint)
        result["coupled"].append(row)
        (OUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        print(
            {
                key: value
                for key, value in row.items()
                if key not in ("medina", "coupled_kick")
            },
            flush=True,
        )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    rows = result["derivative_control"]
    axes[0].loglog(
        [r["dt_s"] * 1e15 for r in rows],
        [r["derivative_relative_error"] for r in rows],
        "o-",
    )
    axes[0].set(
        xlabel="Lab timestep (fs)",
        ylabel="Full-force derivative relative error",
        title="Causal Medina force secant",
    )
    rows = result["coupled"]
    for key, label in (
        ("correction_work_relative_residual", "Discrete work"),
        ("correction_impulse_relative_residual", "Discrete impulse"),
        ("old_work_relative_residual", "Earlier sampled-work estimate"),
    ):
        axes[1].semilogy(
            range(len(rows)), [max(r[key], 1e-17) for r in rows], "o-", label=label
        )
    axes[1].set_xticks(
        range(len(rows)),
        ["Off\naccepted", "Off\nmidpoint", "RR\n301", "RR\n601", "RR\n1201"],
    )
    axes[1].set(
        ylabel="Residual / sum of kick magnitudes",
        title="Both bunches, population weighted",
    )
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"coupling_ledgers.{suffix}", dpi=180)
    manifest = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (
            Path("core/pic/correction.py"),
            Path("core/pic/coupling.py"),
            Path("core/pic/medina.py"),
            Path("core/pic/simulation.py"),
            Path(__file__),
        )
    }
    (OUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
