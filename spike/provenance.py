"""Version, hardware, and uncommitted source identity for every artifact."""

import hashlib
import importlib.metadata
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def command(args):
    try:
        result = subprocess.run(
            args, capture_output=True, text=True, timeout=15, check=False
        )
        return (
            result.stdout.strip() if result.returncode == 0 else result.stderr.strip()
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return str(exc)


def metadata():
    versions = {}
    for name in (
        "numpy",
        "numba",
        "llvmlite",
        "mlx",
        "mlx-metal",
        "mlx-cuda-12",
        "mlx-cuda-13",
        "taichi",
        "cupy-cuda12x",
        "pytest",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    root = Path(__file__).resolve().parent
    hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.glob("*.py"))
    }
    identity = hashlib.sha256(str(sorted(hashes.items())).encode()).hexdigest()
    info = {
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": versions,
        "base_commit": command(["git", "rev-parse", "HEAD"]),
        "git_status": command(["git", "status", "--short"]),
        "source_sha256": identity,
        "source_files_sha256": hashes,
    }
    if platform.system() == "Darwin":
        info["hardware"] = command(["sysctl", "-n", "machdep.cpu.brand_string"])
        info["memory_bytes"] = command(["sysctl", "-n", "hw.memsize"])
        info["macos"] = command(["sw_vers", "-productVersion"])
    else:
        info["nvidia_smi"] = command(["nvidia-smi"])
        info["cuda_toolkit"] = command(["nvcc", "--version"])
    return info
