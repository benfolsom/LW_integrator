"""Independent endpoint, rounding, and reservoir checks of task 8 artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.pic.coupling import kinetic_change
from core.pic.grid import Species
from core.pic.kernels import C
from core.pic.medina import MedinaPIC
from core.pic.simulation import species_from_config
from core.pic import simulation
from scripts.study_pic_correction_k_coupling import config

OUT = Path.home() / "compute-data/pic_correction_k/8_coupling"


def reservoir_control() -> list[dict]:
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
        energy_rate_integral, momentum_rate_integral = 0.0, np.zeros(3)
        initial = None
        for j in range(intervals):
            t = (j + 0.5) * dt
            phase = 2 * np.pi * t / duration
            force = amplitude * np.array(
                [[1, 0.3 * np.sin(phase), 0.2 * np.cos(phase)]]
            )
            predicted = s.momentum_mc + dt * force / (s.mass_kg * C)
            after, ledger = rr.apply(s, predicted, t, dt)
            sample = rr.last_samples[s.name]
            if initial is None:
                initial = sample.copy()
            energy_rate_integral += ledger["cross_field_energy_change_j"]
            momentum_rate_integral += ledger["cross_field_momentum_change_kg_m_s"]
            s.momentum_mc = after
        energy_change = sample["cross_field_energy_j"] - initial["cross_field_energy_j"]
        momentum_change = (
            np.array(sample["cross_field_momentum_kg_m_s"])
            - initial["cross_field_momentum_kg_m_s"]
        )
        rows.append(
            dict(
                intervals=intervals,
                dt_s=dt,
                energy_rate_integral_j=energy_rate_integral,
                energy_sample_change_j=energy_change,
                energy_relative_difference=abs(energy_rate_integral - energy_change)
                / abs(energy_change),
                momentum_rate_integral_kg_m_s=momentum_rate_integral.tolist(),
                momentum_sample_change_kg_m_s=momentum_change.tolist(),
                momentum_relative_difference=float(
                    np.linalg.norm(momentum_rate_integral - momentum_change)
                    / np.linalg.norm(momentum_change)
                ),
            )
        )
    for coarse, fine in zip(rows, rows[1:]):
        fine["energy_order"] = float(
            np.log2(
                coarse["energy_relative_difference"]
                / fine["energy_relative_difference"]
            )
        )
        fine["momentum_order"] = float(
            np.log2(
                coarse["momentum_relative_difference"]
                / fine["momentum_relative_difference"]
            )
        )
    assert rows[-1]["energy_order"] > 0.9
    assert rows[-1]["momentum_order"] > 0.9
    return rows


def trajectory_balance(result: dict) -> dict:
    before = [species_from_config(s) for s in result["config"]["species"]]
    final = result["diagnostics"][-1]
    measured_energy, measured_momentum = 0.0, np.zeros(3)
    rest_energy = 0.0
    for initial, terminal in zip(before, result["final_species"]):
        u = np.array(terminal["momentum_mc"])
        measured_energy += kinetic_change(initial, initial.momentum_mc, u)
        measured_momentum += np.sum(
            initial.population[:, None]
            * initial.mass_kg
            * C
            * (u - initial.momentum_mc),
            axis=0,
        )
        rest_energy += float(initial.population.sum() * initial.mass_kg * C**2)
    kick, rr = final["coupled_kick"], final.get("radiation_reaction", {})
    field_work = kick["total_work_j"] + rr.get("applied_work_j", 0.0)
    field_impulse = np.array(kick["total_impulse_kg_m_s"]) + rr.get(
        "applied_impulse_kg_m_s", np.zeros(3)
    )
    step_energy_scale = (
        max(row["kinetic_energy_j"] for row in result["diagnostics"]) + rest_energy
    )
    # Conservative storage/accumulation bound, not a fitted balancing term.
    # Gamma bounds each component of u and its energy sensitivity.
    energy_rounding_scale = (
        64 * np.finfo(float).eps * (result["config"]["steps"] - 1) * step_energy_scale
    )
    impulse_rounding_scale = energy_rounding_scale / C
    energy_residual = field_work - measured_energy
    momentum_residual = field_impulse - measured_momentum
    assert abs(energy_residual) < energy_rounding_scale
    assert np.linalg.norm(momentum_residual) < impulse_rounding_scale
    assert abs(kick["work_residual_j"]) < energy_rounding_scale
    assert np.linalg.norm(kick["impulse_residual_kg_m_s"]) < impulse_rounding_scale
    return dict(
        measured_energy_change_j=measured_energy,
        measured_momentum_change_kg_m_s=measured_momentum.tolist(),
        non_rr_plus_rr_work_j=field_work,
        non_rr_plus_rr_impulse_kg_m_s=field_impulse.tolist(),
        work_residual_j=energy_residual,
        impulse_residual_kg_m_s=momentum_residual.tolist(),
        conservative_energy_rounding_scale_j=energy_rounding_scale,
        conservative_impulse_rounding_scale_kg_m_s=impulse_rounding_scale,
        energy_residual_over_rounding_scale=abs(energy_residual)
        / energy_rounding_scale,
        impulse_residual_over_rounding_scale=float(
            np.linalg.norm(momentum_residual) / impulse_rounding_scale
        ),
        model_rr_energy_residual_j=rr.get("model_energy_residual_j", 0.0),
        rr_work_quadrature_difference_j=rr.get("work_quadrature_difference_j", 0.0),
    )


def precision_control() -> dict:
    """Short, strong electron/positron control with resolved correction kicks.

    This isolates discrete ledger precision, not trajectory convergence or
    a physical close-encounter model. The standard study remains 1,201 steps.
    """
    cfg = config(41, rr="medina_lad", midpoint=True)
    cfg["propagation_distance_mm"] = 0.02
    for s in cfg["species"]:
        s["population"] = (np.array(s["population"]) * 1e4).tolist()
    cfg["species"][1]["name"] = "positron"
    cfg["species"][1]["mass_kg"] = cfg["species"][0]["mass_kg"]
    kicks = []
    original = simulation.coupled_kick_ledger

    def record(*args):
        ledger = original(*args)
        kicks.append(ledger)
        return ledger

    simulation.coupled_kick_ledger = record
    try:
        result = simulation.run_pic(cfg)
    finally:
        simulation.coupled_kick_ledger = original
    (OUT / "coupled_precision.json").write_text(json.dumps(result, indent=2) + "\n")
    final = result["diagnostics"][-1]
    ledger = final["coupled_kick"]
    energy_scale = sum(abs(k["kick_energy_j"]) for k in kicks)
    impulse_scale = sum(np.linalg.norm(k["kick_momentum_kg_m_s"]) for k in kicks)
    work_residual = abs(ledger["work_residual_j"]) / energy_scale
    impulse_residual = np.linalg.norm(ledger["impulse_residual_kg_m_s"]) / impulse_scale
    assert work_residual < 1e-7
    assert impulse_residual < 1e-7
    return dict(
        steps=41,
        source_population_scale=1e4,
        propagation_distance_mm=0.02,
        work_relative_residual=work_residual,
        impulse_relative_residual=float(impulse_residual),
        old_work_relative_residual=abs(final["correction"]["work_minus_kick_energy_j"])
        / energy_scale,
        cloud_projection=final["cloud_projection"],
        coupled_kick=ledger,
        medina=final["radiation_reaction"],
        trajectory_balance=trajectory_balance(result),
    )


def main() -> None:
    assert int(Path("/private/tmp/compute_cap").read_text()) >= 2
    result = json.loads((OUT / "summary.json").read_text())
    result["reservoir_control"] = reservoir_control()
    result["precision_control"] = precision_control()
    for row in result["coupled"]:
        mode = "midpoint" if row["midpoint"] else "accepted"
        path = OUT / f"coupled_{row['steps']}_{row['rr']}_{mode}.json"
        row["trajectory_balance"] = trajectory_balance(json.loads(path.read_text()))
    medina = [row for row in result["coupled"] if row["rr"] == "medina_lad"]
    refinement = []
    for coarse, fine in zip(medina, medina[1:]):
        a = json.loads(
            (OUT / f"coupled_{coarse['steps']}_medina_lad_midpoint.json").read_text()
        )
        b = json.loads(
            (OUT / f"coupled_{fine['steps']}_medina_lad_midpoint.json").read_text()
        )
        ua = np.concatenate([np.array(s["momentum_mc"]) for s in a["final_species"]])
        ub = np.concatenate([np.array(s["momentum_mc"]) for s in b["final_species"]])
        refinement.append(
            dict(
                coarse_steps=coarse["steps"],
                fine_steps=fine["steps"],
                final_u_relative_change=float(
                    np.linalg.norm(ub - ua) / np.linalg.norm(ub)
                ),
                far_radiation_relative_change=abs(
                    fine["medina"]["far_radiated_energy_j"]
                    - coarse["medina"]["far_radiated_energy_j"]
                )
                / fine["medina"]["far_radiated_energy_j"],
            )
        )
    result["coupled_refinement"] = refinement
    (OUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    os.environ.setdefault("MPLCONFIGDIR", str(OUT / "mpl_cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for key, label in (
        ("energy_relative_difference", "Cross-field energy"),
        ("momentum_relative_difference", "Cross-field momentum"),
    ):
        ax.loglog(
            [r["dt_s"] * 1e15 for r in result["reservoir_control"]],
            [r[key] for r in result["reservoir_control"]],
            "o-",
            label=label,
        )
    ax.set(
        xlabel="Lab timestep (fs)",
        ylabel="Integrated rate versus sample change",
        title="Medina reservoir quadrature convergence",
    )
    ax.legend()
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"reservoir_convergence.{suffix}", dpi=180)
    print(
        json.dumps(
            dict(
                reservoir_control=result["reservoir_control"],
                coupled_refinement=refinement,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
