"""Integrator check: COLD_START gate vs guard window for a long source bunch.

Rider: one proton at z=0, x=1 mm, beta=+0.866. Driver: 41 protons on axis from
z=50 to 2050 mm, beta=-0.866. The rider passes the first drivers within
~0.1-1 ns; the guard's nearest-pair missing window is 0.09 ns.
"""

import sys
import warnings

import numpy as np

sys.path.insert(0, sys.argv[1])
from core.constants import C_MMNS  # noqa: E402
from core.integration_runner import retarded_integrator  # noqa: E402
from core.self_consistency import SelfConsistencyConfig  # noqa: E402
from core.startup_history_guard import check_startup_history  # noqa: E402
from core.types import SimulationType, StartupMode  # noqa: E402
from tests.unit.test_potential_inclusion import state  # noqa: E402

beta = np.sqrt(0.75)
rider = state(x=1.0, z=0.0, beta=beta)
parts = [state(x=0.0, z=z, beta=-beta) for z in np.linspace(50, 2050, 41)]
driver = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
gamma = float(rider["gamma"][0])
dt_lab = 0.01  # ns
h = dt_lab / gamma
steps = int(sys.argv[2])

rep = check_startup_history(
    rider, driver, sim_type=SimulationType.BUNCH_TO_BUNCH,
    startup_mode=StartupMode.COLD_START, steps=steps, h_step=h,
    allow_incomplete=True,
)
print(rep.failures)
print(f"guard: passed={rep.passed} worst_ratio={rep.worst_ratio:.3g}")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    traj, drv, *_ = retarded_integrator(
        steps=steps, h_step=h, init_rider=rider, init_driver=driver,
        wall_z=1e6, aperture_radius=1e5, mean=1000, cav_spacing=1e6, z_cutoff=0,
        sim_type=SimulationType.BUNCH_TO_BUNCH,
        startup_mode=StartupMode.COLD_START,
        self_consistency=SelfConsistencyConfig(
            enabled=True, convergence_mode="fixed_geometry", max_iterations=2,
            target_ms_tolerance=1e-6, mass_shell_tolerance=0.01,
            chrono_interpolate=False,
        ),
        radiation_reaction_mode="medina_lad",
        allow_incomplete_startup_history=True,
    )
t = np.array([float(s["t"][0]) for s in traj])
z = np.array([float(s["z"][0]) for s in traj])
px = np.array([float(s["Px"][0]) for s in traj])
first = int(np.argmax(np.abs(px - px[0]) > 0))
passed = int(np.sum(np.linspace(50, 2050, 41) < (z[first] + beta * C_MMNS * t[first])))
print(f"rider Px first changes at t={t[first]:.3f} ns (step {first}), z={z[first]:.1f}"
      f" mm; drivers already crossed by then: {passed} of 41")
crossed = np.linspace(50, 2050, 41) <= 2 * z[first]
print(f"drivers whose crossing (z_cross = z0/2) precedes gate: {int(crossed.sum())}")
print(f"final rider Px = {px[-1]:.4g}")

for i in (0, 1, 5, 10, 20, 50, 100, 150, 200, 250, 300, len(traj) - 1):
    s = traj[i]
    print(i, f"t={float(s['t'][0]):.3f} z={float(s['z'][0]):.2f} x={float(s['x'][0]):.6g} "
          f"bx={float(s['bx'][0]):.4g} Pz={float(s['Pz'][0]):.8g} g={float(s['gamma'][0]):.10g}")
