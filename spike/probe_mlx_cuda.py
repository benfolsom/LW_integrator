"""Separate Colab capability probe; keep float64 failures visible."""

import argparse
import json
import traceback
from pathlib import Path

from .provenance import metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"metadata": metadata(), "checks": []}
    try:
        import mlx.core as mx

        mx.set_default_device(mx.gpu)
        report["device"] = str(mx.device_info())
        for name, operation in (
            ("float32_array", lambda: mx.ones((32,), dtype=mx.float32) + 1),
            ("fft_float32", lambda: mx.fft.fftn(mx.ones((8, 8, 8), dtype=mx.float32))),
            ("float64_array", lambda: mx.ones((32,), dtype=mx.float64) + 1),
            ("fft_float64", lambda: mx.fft.fftn(mx.ones((8, 8, 8), dtype=mx.float64))),
        ):
            try:
                out = operation()
                mx.eval(out)
                mx.synchronize()
                report["checks"].append(
                    {"test": name, "passed": True, "dtype": str(out.dtype)}
                )
            except Exception as exc:  # noqa: BLE001
                report["checks"].append(
                    {"test": name, "passed": False, "error": str(exc)}
                )
        report["custom_cuda_api"] = hasattr(mx.fast, "cuda_kernel")
    except Exception as exc:  # noqa: BLE001 - preserve import/device diagnostics
        report["error"], report["traceback"] = str(exc), traceback.format_exc()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return int("error" in report)


if __name__ == "__main__":
    raise SystemExit(main())
