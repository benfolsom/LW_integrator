"""Sequential causal midpoint timestep control on the prescribed acceleration case."""

from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from core.pic import CloudCorrection, CorrectionConfig, ElectrostaticPIC, Species
from core.pic.correction import correction_ledger
from core.pic.kernels import C
from scripts.study_pic_correction_k import CASES, quiet, species, trajectory

OUT = Path.home() / "compute-data/pic_correction_k/6_predictor"


def set_source(src, offsets, case, t):
    _, displacement, beta = trajectory(case, t / case["duration"])
    src.position_m[:] = offsets + displacement
    src.momentum_mc[:] = beta / np.sqrt(1 - beta @ beta)


def convergence(divisions=(16, 32, 64, 128), reference_steps=512):
    case = CASES["acceleration"]
    start, stop = 24e-12, 36e-12
    offsets = quiet(256)
    src = species(offsets.copy(), case)
    seed = CloudCorrection(
        [src],
        CorrectionConfig(
            clouds_per_species=4,
            subcharge_count=4,
            lattice_shape=(4,) * 3,
            half_extent_m=(0.15e-3,) * 3,
            midpoint_predictor=True,
        ),
    )
    for t in np.linspace(0, start, 2401)[1:]:
        set_source(src, offsets, case, t)
        seed.accept([src], t)
    _, displacement, _ = trajectory(case, 1)
    center = displacement + case["displacement"]
    obs = Species(
        "observer",
        center + quiet(64, sigma=0.025e-3),
        np.tile(
            [0, 0, case["observer_beta"] / np.sqrt(1 - case["observer_beta"] ** 2)],
            (64, 1),
        ),
        src.charge_c,
        src.mass_kg,
        np.linspace(1e3, 2e3, 64),
    )
    local = obs.position_m - center
    solver = ElectrostaticPIC()

    def run(n, mode):
        corr = deepcopy(seed)
        dt = (stop - start) / n
        total = np.zeros((2, len(local), 3))
        impulse = np.zeros(3)
        work, energy = 0.0, 0.0
        for j in range(n):
            t = start + j * dt
            if j:
                set_source(src, offsets, case, t)
                corr.accept([src], t)
            mid = t + dt / 2
            obs.position_m[:] = (
                center
                + local
                + np.array([0, 0, case["observer_beta"] * C * (mid - case["duration"])])
            )
            if mode == "reference":
                # This offline reference alone may use a future prescribed state.
                set_source(src, offsets, case, mid)
                corr.accept([src], mid)
                evaluation = corr
            else:
                evaluation = corr.predict_midpoint(dt) if mode == "midpoint" else corr
            e, b = evaluation.sample(obs)
            total += dt * np.stack((e, C * b))
            pushed = solver.push(obs, e, b, dt)
            ledger = correction_ledger(obs, obs.momentum_mc, pushed, e, b, dt)
            impulse += ledger["kick_momentum_kg_m_s"]
            work += ledger["electric_work_j"]
            energy += ledger["kick_energy_j"]
        return total, impulse, abs(work - energy) / max(abs(energy), 1e-300)

    reference, ref_impulse, _ = run(reference_steps, "reference")
    ref_half, ref_half_impulse, _ = run(reference_steps // 2, "reference")
    rows = []
    for mode in ("accepted", "midpoint"):
        previous = None
        for n in divisions:
            field, impulse, closure = run(n, mode)
            error = float(np.linalg.norm(field - reference) / np.linalg.norm(reference))
            row = dict(
                mode=mode,
                divisions=n,
                dt_s=(stop - start) / n,
                integrated_field_error=error,
                impulse_error=float(
                    np.linalg.norm(impulse - ref_impulse) / np.linalg.norm(ref_impulse)
                ),
                work_relative_residual=closure,
                order=None if previous is None else float(np.log2(previous / error)),
            )
            rows.append(row)
            previous = error
            print(row, flush=True)
    return dict(
        rows=rows,
        reference_steps=reference_steps,
        reference_field_change=float(
            np.linalg.norm(reference - ref_half) / np.linalg.norm(reference)
        ),
        reference_impulse_change=float(
            np.linalg.norm(ref_impulse - ref_half_impulse) / np.linalg.norm(ref_impulse)
        ),
    )


if __name__ == "__main__":
    assert int(Path("/private/tmp/compute_cap").read_text()) >= 2
    OUT.mkdir(parents=True, exist_ok=True)
    result = convergence()
    (OUT / "convergence.json").write_text(json.dumps(result, indent=2) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    for mode in ("accepted", "midpoint"):
        rows = [r for r in result["rows"] if r["mode"] == mode]
        ax.loglog(
            [r["dt_s"] * 1e12 for r in rows],
            [r["integrated_field_error"] for r in rows],
            "o-",
            label=mode,
        )
    ax.set(xlabel="Lab timestep (ps)", ylabel="Integrated correction relative error")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "timestep_order.png", dpi=180)
