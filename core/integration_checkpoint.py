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
ACCEPTED_PAIR_SCHEMA_VERSION = 1


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


class AcceptedPairCheckpointStore:
    """Append-only checkpoint store for a variable-length accepted pair history.

    Unlike :class:`IntegrationCheckpointStore`, this format does not require a
    final step count.  Each immutable chunk contains an equal number of rider
    and driver source-history knots.  The atomically replaced manifest also
    records the adaptive controller and public-output cursor needed to resume
    at the next shared lab-time barrier.

    The class is a storage substrate only.  No current CLI/GUI option or
    integrator mode selects it yet.
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        compatibility_payload: dict[str, Any],
        interval_knots: int,
        interval_seconds: float,
        resume: bool,
    ) -> None:
        self.directory = Path(directory).expanduser().resolve()
        self.chunks_directory = self.directory / "chunks"
        self.manifest_path = self.directory / "manifest.json"
        self.constants_path = self.directory / "constants.npz"
        self.compatibility_payload = compatibility_payload
        self.compatibility_hash = canonical_json_hash(compatibility_payload)
        self.interval_knots = int(interval_knots)
        self.interval_seconds = float(interval_seconds)
        if self.interval_knots < 0:
            raise ValueError("interval_knots must be non-negative")
        if not np.isfinite(self.interval_seconds) or self.interval_seconds < 0.0:
            raise ValueError("interval_seconds must be finite and non-negative")
        if self.interval_knots == 0 and self.interval_seconds == 0.0:
            raise ValueError("accepted-pair checkpoint needs a positive interval")
        self._last_write_monotonic = time.monotonic()

        if resume:
            self.manifest = self._load_and_validate_manifest()
        else:
            if self.directory.exists() and any(self.directory.iterdir()):
                raise CheckpointError(
                    f"checkpoint directory is not empty: {self.directory}; "
                    "resume it or choose a new directory"
                )
            self.chunks_directory.mkdir(parents=True, exist_ok=True)
            self.manifest = {
                "schema_version": ACCEPTED_PAIR_SCHEMA_VERSION,
                "checkpoint_kind": "accepted_pair_history",
                "status": "running",
                "created_utc": _utc_now(),
                "updated_utc": _utc_now(),
                "compatibility_hash": self.compatibility_hash,
                "compatibility": compatibility_payload,
                "committed_knots": 0,
                "constants": None,
                "chunks": [],
                "controller_state": {},
                "public_output_state": {},
            }
            _atomic_json(self.manifest_path, self.manifest)

    @property
    def committed_knots(self) -> int:
        return int(self.manifest["committed_knots"])

    @property
    def controller_state(self) -> dict[str, Any]:
        value = self.manifest.get("controller_state", {})
        if not isinstance(value, dict):
            raise CheckpointError("controller_state must be a JSON object")
        return cast(dict[str, Any], dict(value))

    @property
    def public_output_state(self) -> dict[str, Any]:
        value = self.manifest.get("public_output_state", {})
        if not isinstance(value, dict):
            raise CheckpointError("public_output_state must be a JSON object")
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
        if (
            manifest.get("schema_version") != ACCEPTED_PAIR_SCHEMA_VERSION
            or manifest.get("checkpoint_kind") != "accepted_pair_history"
        ):
            raise CheckpointCompatibilityError(
                "unsupported accepted-pair checkpoint schema or kind"
            )
        if manifest.get("compatibility_hash") != self.compatibility_hash:
            raise CheckpointCompatibilityError(
                "checkpoint physics/configuration fingerprint does not match this run"
            )
        committed = int(manifest.get("committed_knots", -1))
        if committed < 1:
            raise CheckpointError(
                "resumable accepted-pair history has no committed knots"
            )
        return cast(dict[str, Any], manifest)

    def due(self, accepted_knots: int, *, force: bool = False) -> bool:
        if force:
            return True
        knot_due = bool(
            self.interval_knots > 0
            and int(accepted_knots) >= self.committed_knots + self.interval_knots
        )
        time_due = bool(
            self.interval_seconds > 0.0
            and time.monotonic() - self._last_write_monotonic >= self.interval_seconds
        )
        return knot_due or time_due

    @staticmethod
    def _validate_side_channels(
        trajectory: TrajectoryArrays,
        role: str,
    ) -> None:
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

    @staticmethod
    def _json_state(value: dict[str, Any], label: str) -> dict[str, Any]:
        """Validate and detach one JSON checkpoint state object."""

        try:
            encoded = json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            decoded = json.loads(encoded)
        except (TypeError, ValueError) as exc:
            raise CheckpointError(
                f"{label} must contain only finite JSON-compatible values"
            ) from exc
        if not isinstance(decoded, dict):
            raise CheckpointError(f"{label} must be a JSON object")
        return cast(dict[str, Any], decoded)

    def _constants_metadata(
        self,
        rider: TrajectoryArrays,
        driver: TrajectoryArrays,
    ) -> dict[str, str]:
        existing = self.manifest.get("constants")
        if existing is not None:
            if not isinstance(existing, dict):
                raise CheckpointError("checkpoint constants metadata is invalid")
            with self._verified_npz(
                str(existing["file"]), str(existing["sha256"])
            ) as archive:
                for role, trajectory in (("rider", rider), ("driver", driver)):
                    for name in _PARTICLE_CONSTANT_FIELDS:
                        stored = np.asarray(archive[f"{role}__{name}"])
                        current = np.asarray(getattr(trajectory, name))
                        if not np.array_equal(stored, current, equal_nan=True):
                            raise CheckpointCompatibilityError(
                                f"{role} particle constant {name} changed after "
                                "the accepted-pair checkpoint was created"
                            )
            return {
                "file": str(existing["file"]),
                "sha256": str(existing["sha256"]),
            }

        arrays: dict[str, np.ndarray] = {}
        for role, trajectory in (("rider", rider), ("driver", driver)):
            for name in _PARTICLE_CONSTANT_FIELDS:
                arrays[f"{role}__{name}"] = np.array(
                    getattr(trajectory, name), copy=True
                )
        digest = _atomic_npz(self.constants_path, arrays)
        return {
            "file": self.constants_path.name,
            "sha256": digest,
        }

    def write(
        self,
        *,
        rider: TrajectoryArrays,
        driver: TrajectoryArrays,
        controller_state: dict[str, Any],
        public_output_state: dict[str, Any],
        complete: bool = False,
    ) -> None:
        """Commit all jointly accepted knots after the manifest boundary."""

        if self.manifest.get("status") == "complete":
            raise CheckpointError("cannot append to a completed checkpoint")
        if rider.n_steps != driver.n_steps:
            raise CheckpointError(
                "rider and driver accepted histories must have equal knot counts"
            )
        normalized_controller = self._json_state(controller_state, "controller_state")
        normalized_output = self._json_state(public_output_state, "public_output_state")
        start = self.committed_knots
        stop = int(rider.n_steps)
        if stop < start:
            raise CheckpointError("accepted history is shorter than the checkpoint")
        if stop == start:
            if complete:
                next_manifest = dict(self.manifest)
                next_manifest["status"] = "complete"
                next_manifest["controller_state"] = normalized_controller
                next_manifest["public_output_state"] = normalized_output
                next_manifest["updated_utc"] = _utc_now()
                _atomic_json(self.manifest_path, next_manifest)
                self.manifest = next_manifest
            return

        self._validate_side_channels(rider, "rider")
        self._validate_side_channels(driver, "driver")
        constants = self._constants_metadata(rider, driver)
        arrays: dict[str, np.ndarray] = {}
        for role, trajectory in (("rider", rider), ("driver", driver)):
            for name in _ROW_ARRAY_FIELDS:
                values = np.asarray(getattr(trajectory, name))
                arrays[f"{role}__{name}"] = np.array(values[start:stop], copy=True)
        filename = f"knots_{start:09d}_{stop:09d}.npz"
        chunk_path = self.chunks_directory / filename
        digest = _atomic_npz(chunk_path, arrays)
        chunks = list(self.manifest.get("chunks", []))
        chunks.append(
            {
                "file": str(Path("chunks") / filename),
                "start": start,
                "stop": stop,
                "sha256": digest,
            }
        )
        next_manifest = dict(self.manifest)
        next_manifest["constants"] = constants
        next_manifest["chunks"] = chunks
        next_manifest["committed_knots"] = stop
        next_manifest["controller_state"] = normalized_controller
        next_manifest["public_output_state"] = normalized_output
        next_manifest["status"] = "complete" if complete else "running"
        next_manifest["updated_utc"] = _utc_now()
        _atomic_json(self.manifest_path, next_manifest)
        self.manifest = next_manifest
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

    def _restore_builder(self, builder: TrajectoryBuilder, role: str) -> None:
        constants_meta = self.manifest.get("constants")
        if not isinstance(constants_meta, dict):
            raise CheckpointError("checkpoint constants metadata is missing")
        with self._verified_npz(
            str(constants_meta["file"]),
            str(constants_meta["sha256"]),
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
                raise CheckpointError("accepted-pair chunks are not contiguous")
            with self._verified_npz(
                str(chunk["file"]),
                str(chunk["sha256"]),
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
        if expected_start != self.committed_knots:
            raise CheckpointError(
                "accepted-pair chunk boundary does not match committed knot count"
            )

    def restore_pair(
        self,
        rider_builder: TrajectoryBuilder,
        driver_builder: TrajectoryBuilder,
    ) -> None:
        """Restore both histories through the same committed knot boundary."""

        self._restore_builder(rider_builder, "rider")
        self._restore_builder(driver_builder, "driver")


__all__ = [
    "AcceptedPairCheckpointStore",
    "CheckpointCompatibilityError",
    "CheckpointError",
    "IntegrationCheckpointStore",
    "canonical_json_hash",
]
