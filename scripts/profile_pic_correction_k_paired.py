"""Interleave cold and warm queries on identical accepted histories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import retarded_fields as rf
import core.pic.correction as cm
from core.exact_source_cloud import exact_cloud_history
from core.pic import ElectrostaticPIC
from scripts.profile_pic_correction_k import StageTimers, cases, pic_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    timers = StageTimers()
    baseline = types.ModuleType("core.pic._paired_baseline")
    baseline.__package__ = "core.pic"
    code = subprocess.check_output(
        ["git", "show", "9e898dc:core/pic/correction_fields.py"], cwd=ROOT
    )
    exec(compile(code, "correction_fields_9e898dc.py", "exec"), baseline.__dict__)
    timers.patch(baseline, "node_segment_hints", "batch_bracket_proposals")
    production = cm.correction_node_fields

    def cold(history, t, positions, **kwargs):
        return baseline.correction_node_fields(history, t, positions)

    rows = []
    for name, sources, corr, observers, dt in cases(8, 5):
        cm.correction_node_fields = production
        for obs in observers:
            corr.sample(obs)
        for step in range(5):
            # Both variants receive the same already prepared accepted history.
            rf._prepare_history(
                exact_cloud_history(corr.builder.build_current(), corr.smearing), ()
            )
            values, measurements = {}, {}
            order = ("cold", "warm") if step % 2 == 0 else ("warm", "cold")
            for label in order:
                cm.correction_node_fields = cold if label == "cold" else production
                timers.reset()
                start = time.perf_counter()
                values[label] = [f for obs in observers for f in corr.sample(obs)]
                measurements[label] = dict(
                    seconds=time.perf_counter() - start, stages_s=dict(timers.seconds)
                )
            assert all(
                a.dtype == b.dtype and a.tobytes() == b.tobytes()
                for a, b in zip(values["cold"], values["warm"])
            )
            rows.append(
                dict(
                    case=name, step=step, order=order, byte_parity=True, **measurements
                )
            )
            (args.output / "paired.json").write_text(json.dumps(rows, indent=2) + "\n")
            cm.correction_node_fields = production
            pic_step(ElectrostaticPIC((24,) * 3, (0.004,) * 3, 2), sources, dt)
            corr.accept(sources, corr.time_s + dt)
            print(name, step, measurements, flush=True)


if __name__ == "__main__":
    main()
