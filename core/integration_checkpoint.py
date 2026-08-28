"""Atomic append-only checkpoints for accepted integration steps.

The checkpoint directory contains immutable NumPy chunks and one atomically
replaced JSON manifest.  A crash can leave an unreferenced chunk, but it cannot
advance the committed step until every array in that chunk has been flushed.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np

from .types import TrajectoryArrays, TrajectoryBuilder


SCHEMA_VERSION = 1


class CheckpointError(RuntimeError):
    """Base class for checkpoint creation and restart errors."""


class CheckpointCompatibilityError(CheckpointError):
    """Raised when a checkpoint belongs to different integration inputs."""


_PARTICLE_CONSTANT_FIELDS = (
    "q",
    "q_species",
    "q_observer",
    "q_source",
    "macro_population",
    "m",
    "m_species",
    "char_time",
    "magnetic_moment_j_per_t",
    "magnetic_moment_native",
    "spin_quantum_number",
    "gyromagnetic_ratio_rad_s_t",
    "magnetic_dipole_active",
    "spin_precession_active",
    "stern_gerlach_active",
)
_NON_ARRAY_FIELDS = {
    "halt_reason",
    "particle_failure_info",
    "pseudo_grid_schedule",
    "_storage_state",
    "_storage_array_revision",
}
_ROW_ARRAY_FIELDS = tuple(
    descriptor.name
    for descriptor in fields(TrajectoryArrays)
    if descriptor.name not in _PARTICLE_CONSTANT_FIELDS
    and descriptor.name not in _NON_ARRAY_FIELDS
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_hash(payload: dict[str, Any]) -> str:
    """Return the stable SHA-256 of a JSON-compatible compatibility payload."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _fsync_directory(directory: Path) -> None:
    """Flush a directory entry where the platform permits it."""

    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        # Windows and some network filesystems do not allow opening a
        # directory as a file descriptor. The atomic replacement still gives
        # process-crash safety there, even when power-loss durability cannot be
        # strengthened with a directory fsync.
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    data = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode(
        "utf-8"
    )
    with temporary.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> str:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)
    return _sha256(path)


class IntegrationCheckpointStore:
    """Write and restore one append-only integration checkpoint directory."""

    def __init__(
        self,
        directory: str | Path,
        *,
        compatibility_payload: dict[str, Any],
        total_steps: int,
        requested_steps: int,
        active_start: int,
        interval_steps: int,
        interval_seconds: float,
        resume: bool,
    ) -> None:
        self.directory = Path(directory).expanduser().resolve()
        self.chunks_directory = self.directory / "chunks"
        self.manifest_path = self.directory / "manifest.json"
        self.constants_path = self.directory / "constants.npz"
        self.compatibility_payload = compatibility_payload
        self.compatibility_hash = canonical_json_hash(compatibility_payload)
        self.total_steps = int(total_steps)
        self.requested_steps = int(requested_steps)
        self.active_start = int(active_start)
        self.interval_steps = int(interval_steps)
        self.interval_seconds = float(interval_seconds)
        self._last_write_monotonic = time.monotonic()

        if resume:
            self.manifest = self._load_and_validate_manifest()
        else:
            if self.directory.exists() and any(self.directory.iterdir()):
                raise CheckpointError(
                    f"checkpoint directory is not empty: {self.directory}; "
                    "use resume_from or choose a new directory"
                )
            self.chunks_directory.mkdir(parents=True, exist_ok=True)
            self.manifest = {
                "schema_version": SCHEMA_VERSION,
                "status": "running",
                "created_utc": _utc_now(),
                "updated_utc": _utc_now(),
                "compatibility_hash": self.compatibility_hash,
                "compatibility": compatibility_payload,
                "total_steps": self.total_steps,
                "requested_steps": self.requested_steps,
                "active_start": self.active_start,
                "committed_internal_step": -1,
                "completed_public_steps": 0,
                "constants": None,
                "chunks": [],
                "loop_state": {},
            }
            _atomic_json(self.manifest_path, self.manifest)

    @property
    def committed_internal_step(self) -> int:
        return int(self.manifest["committed_internal_step"])

    @property
    def next_internal_step(self) -> int:
        return self.committed_internal_step + 1

    @property
    def loop_state(self) -> dict[str, Any]:
        value = self.manifest.get("loop_state", {})
        if not isinstance(value, dict):
            raise CheckpointError("checkpoint loop_state must be a JSON object")
        return cast(dict[str, Any], dict(value))

    def _load_and_validate_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.is_file():
            raise CheckpointError(
                f"checkpoint manifest not found: {self.manifest_path}"
            )
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CheckpointError(f"cannot read checkpoint manifest: {exc}") from exc
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise CheckpointCompatibilityError(
                "unsupported checkpoint schema "
                f"{manifest.get('schema_version')!r}; expected {SCHEMA_VERSION}"
            )
        if manifest.get("compatibility_hash") != self.compatibility_hash:
            raise CheckpointCompatibilityError(
                "checkpoint physics/configuration fingerprint does not match this run"
            )
        expected = {
            "total_steps": self.total_steps,
            "requested_steps": self.requested_steps,
            "active_start": self.active_start,
        }
        for name, value in expected.items():
            if int(manifest.get(name, -1)) != value:
                raise CheckpointCompatibilityError(
                    f"checkpoint {name}={manifest.get(name)!r}, expected {value}"
                )
        committed = int(manifest.get("committed_internal_step", -1))
        if committed < self.active_start or committed >= self.total_steps:
            raise CheckpointError(
                f"checkpoint committed step {committed} is outside the active run"
            )
        return cast(dict[str, Any], manifest)

    def due(self, completed_public_steps: int, *, force: bool = False) -> bool:
        if force:
            return True
        step_due = bool(
            self.interval_steps > 0
            and completed_public_steps
            >= int(self.manifest["completed_public_steps"]) + self.interval_steps
        )
        time_due = bool(
            self.interval_seconds > 0.0
            and time.monotonic() - self._last_write_monotonic >= self.interval_seconds
        )
        return step_due or time_due

    def _validate_side_channels(self, trajectory: TrajectoryArrays, role: str) -> None:
        if trajectory.particle_failure_info:
            raise CheckpointError(
                f"cannot checkpoint {role} particle failure side-channel state"
            )
        if any(value is not None for value in trajectory.pseudo_grid_schedule):
            raise CheckpointError(
                f"cannot checkpoint {role} pseudo-grid schedule side-channel state"
            )
        if any(value is not None for value in trajectory.halt_reason):
            raise CheckpointError(f"cannot checkpoint halted {role} trajectory state")

    def _write_constants(
        self, rider: TrajectoryArrays, driver: TrajectoryArrays
    ) -> None:
        if self.manifest.get("constants") is not None:
            return
        arrays: dict[str, np.ndarray] = {}
        for role, trajectory in (("rider", rider), ("driver", driver)):
            for name in _PARTICLE_CONSTANT_FIELDS:
                arrays[f"{role}__{name}"] = np.array(
                    getattr(trajectory, name), copy=True
                )
        digest = _atomic_npz(self.constants_path, arrays)
        self.manifest["constants"] = {
            "file": self.constants_path.name,
            "sha256": digest,
        }

    def write(
        self,
        *,
        step_index: int,
        rider: TrajectoryArrays,
        driver: TrajectoryArrays,
        loop_state: dict[str, Any],
        complete: bool = False,
    ) -> None:
        """Commit every accepted row after the previous manifest boundary."""

        step_index = int(step_index)
        start = self.committed_internal_step + 1
        stop = step_index + 1
        if stop <= start:
            if complete and self.manifest.get("status") != "complete":
                self.manifest["status"] = "complete"
                self.manifest["updated_utc"] = _utc_now()
                _atomic_json(self.manifest_path, self.manifest)
            return
        if rider.n_steps < stop or driver.n_steps < stop:
            raise CheckpointError(
                "checkpoint trajectory does not contain the accepted row"
            )
        self._validate_side_channels(rider, "rider")
        self._validate_side_channels(driver, "driver")
        self._write_constants(rider, driver)

        arrays: dict[str, np.ndarray] = {}
        for role, trajectory in (("rider", rider), ("driver", driver)):
            for name in _ROW_ARRAY_FIELDS:
                values = np.asarray(getattr(trajectory, name))
                arrays[f"{role}__{name}"] = np.array(values[start:stop], copy=True)
        filename = f"rows_{start:09d}_{stop:09d}.npz"
        chunk_path = self.chunks_directory / filename
        digest = _atomic_npz(chunk_path, arrays)
        self.manifest["chunks"].append(
            {
                "file": str(Path("chunks") / filename),
                "start": start,
                "stop": stop,
                "sha256": digest,
            }
        )
        self.manifest["committed_internal_step"] = step_index
        self.manifest["completed_public_steps"] = max(
            0, step_index - self.active_start + 1
        )
        self.manifest["loop_state"] = loop_state
        self.manifest["status"] = "complete" if complete else "running"
        self.manifest["updated_utc"] = _utc_now()
        _atomic_json(self.manifest_path, self.manifest)
        self._last_write_monotonic = time.monotonic()

    def _verified_npz(self, relative_path: str, expected_hash: str) -> Any:
        path = self.directory / relative_path
        if not path.is_file():
            raise CheckpointError(f"checkpoint data file is missing: {path}")
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            raise CheckpointError(
                f"checkpoint data hash mismatch for {path.name}: "
                f"{actual_hash} != {expected_hash}"
            )
        return np.load(path, allow_pickle=False)

    def restore_builder(self, builder: TrajectoryBuilder, role: str) -> None:
        """Restore one rider/driver builder through the committed row."""

        if role not in {"rider", "driver"}:
            raise ValueError("checkpoint role must be rider or driver")
        constants_meta = self.manifest.get("constants")
        if not isinstance(constants_meta, dict):
            raise CheckpointError("checkpoint constants metadata is missing")
        with self._verified_npz(
            str(constants_meta["file"]), str(constants_meta["sha256"])
        ) as archive:
            constants = {
                name: np.array(archive[f"{role}__{name}"], copy=True)
                for name in _PARTICLE_CONSTANT_FIELDS
            }

        expected_start = 0
        for chunk_index, chunk in enumerate(self.manifest.get("chunks", [])):
            start = int(chunk["start"])
            stop = int(chunk["stop"])
            if start != expected_start or stop <= start:
                raise CheckpointError("checkpoint chunks are not contiguous")
            with self._verified_npz(
                str(chunk["file"]), str(chunk["sha256"])
            ) as archive:
                row_arrays = {
                    name: np.array(archive[f"{role}__{name}"], copy=True)
                    for name in _ROW_ARRAY_FIELDS
                }
            builder.restore_checkpoint_rows(
                start,
                row_arrays,
                particle_constants=constants if chunk_index == 0 else None,
            )
            expected_start = stop
        if expected_start != self.next_internal_step:
            raise CheckpointError(
                "checkpoint chunk boundary does not match committed step"
            )


__all__ = [
    "CheckpointCompatibilityError",
    "CheckpointError",
    "IntegrationCheckpointStore",
    "canonical_json_hash",
]
