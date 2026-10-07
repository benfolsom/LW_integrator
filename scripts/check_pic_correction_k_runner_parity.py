"""Replay default run_pic correction against the complete 9e898dc core.

Each replay runs serially in a fresh interpreter, capturing particle events,
PIC field arrays, and sampled additive correction fields without changing them.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def worker(source_root, config_path, output):
    sys.path.insert(0, str(source_root))
    import numpy as np
    from core.pic import CloudCorrection, ElectrostaticPIC, run_pic

    captured = {}

    def save(label, values):
        for i, value in enumerate(values):
            if value is not None:
                captured[f"{label}_{len(captured)}_{i}"] = np.asarray(value).copy()

    original_fields = ElectrostaticPIC.fields
    original_sample = ElectrostaticPIC.sample
    original_correction = CloudCorrection.sample

    def fields(self, species, *args, **kwargs):
        for s in species:
            save("particles", (s.position_m, s.momentum_mc, s.population))
        result = original_fields(self, species, *args, **kwargs)
        for field in result:
            save(
                "pic",
                (
                    field.charge,
                    field.electric_rest,
                    field.potential_rest,
                    field.center_m,
                    field.basis,
                    field.beta_vector,
                ),
            )
        return result

    def sample(self, *args, **kwargs):
        result = original_sample(self, *args, **kwargs)
        save("pic_sample", result)
        return result

    def correction(self, *args, **kwargs):
        result = original_correction(self, *args, **kwargs)
        save("correction_sample", result)
        return result

    ElectrostaticPIC.fields = fields
    ElectrostaticPIC.sample = sample
    CloudCorrection.sample = correction
    result = run_pic(json.loads(config_path.read_text()))
    for s in result["final_species"]:
        save("final", (s["positions_mm"], s["momentum_mc"], s["population"]))
    np.savez(output, **captured)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker-root", type=Path)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    if args.worker_root:
        worker(args.worker_root, args.config, args.output)
        return
    import numpy as np

    args.output.mkdir(parents=True, exist_ok=True)
    baseline = args.output / "baseline_9e898dc"
    baseline.mkdir(exist_ok=True)
    archive = subprocess.check_output(["git", "archive", "9e898dc", "core"], cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
        bundle.extractall(baseline, filter="data")
    comparisons = []
    for label, chirp in (("uniform", 0), ("chirped", 0.15)):
        config = dict(
            mode="pic",
            backend="cpu",
            radiation_reaction_mode="off",
            steps=21,
            timestep_s=1e-13,
            sample_every=1,
            grid=dict(shape=[8] * 3, half_extent_mm=[4] * 3),
            correction={},
            species=[
                dict(
                    name="electrons",
                    charge_c=-1.602176634e-19,
                    mass_kg=9.1093837015e-31,
                    population=1e5,
                    gaussian=dict(
                        count=8,
                        seed=42,
                        rms_size_mm=[0.1] * 3,
                        kinetic_energy_mev=0.1,
                        fractional_momentum_chirp=chirp,
                    ),
                )
            ],
        )
        config_path = args.output / f"{label}_config.json"
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        paths = []
        for name, source_root in (("baseline", baseline), ("current", ROOT)):
            cap = int(Path("/private/tmp/compute_cap").read_text().strip())
            if cap < 2:
                raise RuntimeError(
                    "compute cap below 2; rerun when a slot is available"
                )
            path = args.output / f"{label}_{name}.npz"
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker-root",
                    str(source_root),
                    "--config",
                    str(config_path),
                    "--output",
                    str(path),
                ],
                check=True,
                cwd=source_root,
            )
            paths.append(path)
        with np.load(paths[0]) as expected, np.load(paths[1]) as actual:
            assert expected.files == actual.files
            for key in expected.files:
                assert expected[key].dtype == actual[key].dtype, key
                assert expected[key].shape == actual[key].shape, key
                assert expected[key].tobytes() == actual[key].tobytes(), key
            comparisons.append(
                dict(
                    case=label,
                    arrays=len(expected.files),
                    particle_and_field_byte_parity=True,
                )
            )
    (args.output / "runner_parity.json").write_text(
        json.dumps(comparisons, indent=2) + "\n"
    )
    print(comparisons, flush=True)


if __name__ == "__main__":
    main()
