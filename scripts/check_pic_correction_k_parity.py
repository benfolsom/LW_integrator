"""Replay the stage 2 controls against the frozen development node evaluator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import core.pic.correction as correction_module
from scripts import validate_pic_correction as stage2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    baseline = types.ModuleType("core.pic._baseline_correction_fields")
    baseline.__package__ = "core.pic"
    code = subprocess.check_output(
        ["git", "show", "9e898dc:core/pic/correction_fields.py"], cwd=ROOT
    )
    exec(compile(code, "correction_fields_9e898dc.py", "exec"), baseline.__dict__)
    production = correction_module.correction_node_fields
    results = []
    for label in ("cold", "warm"):
        stage2.OUT = args.output / label
        stage2.OUT.mkdir(parents=True, exist_ok=True)
        if label == "cold":

            def cold(history, t, positions, **kwargs):
                return baseline.correction_node_fields(history, t, positions)

            correction_module.correction_node_fields = cold
        else:
            correction_module.correction_node_fields = production
        stage2.uniform()
        stage2.acceleration()
        stage2.crossing()
        for name in ("uniform", "acceleration", "crossing_m24_s1201_n6"):
            path = stage2.OUT / f"{name}.json"
            results.append((label, path.name, json.loads(path.read_text())))
    comparisons = []
    for (_, name, expected), (_, warm_name, actual) in zip(results[:3], results[3:]):
        assert name == warm_name
        assert expected == actual, name
        comparisons.append(dict(case=name, json_bit_parity=True))
    cold_npz = args.output / "cold/crossing_m24_s1201_histories.npz"
    warm_npz = args.output / "warm/crossing_m24_s1201_histories.npz"
    with np.load(cold_npz) as a, np.load(warm_npz) as b:
        assert a.files == b.files
        for key in a.files:
            assert a[key].dtype == b[key].dtype
            assert a[key].tobytes() == b[key].tobytes(), key
        comparisons.append(
            dict(case="crossing histories", arrays=len(a.files), byte_parity=True)
        )
    (args.output / "parity.json").write_text(json.dumps(comparisons, indent=2) + "\n")
    print(comparisons, flush=True)


if __name__ == "__main__":
    main()
