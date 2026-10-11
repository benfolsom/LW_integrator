"""Dump trajectories for parity cases. Run with cwd = repo tree; arg = out .npz."""

import sys

import numpy as np

sys.path.insert(0, ".")
import tests.unit.test_exact_visibility_gates as g  # noqa: E402
from core.constants import ELECTRON_MASS_AMU as ELECTRON_MASS  # noqa: E402
from core.constants import ELEMENTARY_CHARGE  # noqa: E402
from core.stochastic_emission import StochasticEmissionConfig  # noqa: E402
from core.types import ExternalFieldConfig, MagneticDipoleConfig, StartupMode  # noqa: E402
from tests.unit.test_inertial_prehistory import _state  # noqa: E402

out = {}


def dump(name, result):
    for role, traj in (("r", result[0]), ("d", result[1])):
        for i, row in enumerate(traj):
            for key, value in row.items():
                if isinstance(value, np.ndarray) and value.dtype.kind in "fiub":
                    out[f"{name}/{role}/{i}/{key}"] = value
                elif key == "potential_inclusion_state":
                    out[f"{name}/{role}/{i}/meta"] = np.asarray(value, dtype=str)


dump("gate_quintic", g.crossing_run(8, g.gate(radius=1.0)))
# Sampled path with a forced photon.
orig = g.retarded_integrator


def cold(**kw):
    kw["startup_mode"] = StartupMode.COLD_START
    return orig(**kw)


g.retarded_integrator = cold
rider = _state(
    position_mm=(0.1, 0, -0.5),
    beta=(0, 0, np.sqrt(0.75)),
    mass_amu=ELECTRON_MASS,
    observer_charge=-ELEMENTARY_CHARGE,
)
driver = _state(
    position_mm=(0, 0, 0.5),
    beta=(0, 0, -0.2),
    mass_amu=12,
    observer_charge=ELEMENTARY_CHARGE,
)
dump(
    "sampled_emission",
    g.crossing_run(
        6,
        None,
        rider_state=rider,
        driver_state=driver,
        duration=1e-6,
        stochastic_emission=StochasticEmissionConfig(
            enabled=True,
            seed=73,
            chi_threshold=1e-30,
            recoil_threshold=1e-30,
            test_force_next_emission_fraction=0.01,
        ),
        external_field=ExternalFieldConfig(magnetic_field_native=(0, 100, 0)),
        magnetic_dipole=MagneticDipoleConfig(),
    ),
)
np.savez(sys.argv[1], **out)
print("saved", len(out))
