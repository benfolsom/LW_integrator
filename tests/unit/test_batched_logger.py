"""Unit tests for batched GUI logging helpers."""

from __future__ import annotations

import pytest

from core import batched_logger as batched_logger_module
from core.batched_logger import BatchedLogger, ThrottledProgressCallback


def test_batched_logger_batches_messages_and_tracks_stats():
    received_batches: list[str] = []
    logger = BatchedLogger(
        gui_callback=received_batches.append,
        batch_size=3,
        flush_interval_ms=60000,
        enable_batching=True,
    )

    try:
        for index in range(7):
            logger.log(f"message {index}")

        logger.flush()

        assert received_batches == [
            "message 0\nmessage 1\nmessage 2",
            "message 3\nmessage 4\nmessage 5",
            "message 6",
        ]

        stats = logger.get_stats()
        assert stats["total_messages"] == 7
        assert stats["total_batches"] == 3
        assert stats["dropped_messages"] == 0
        assert stats["batching_enabled"] is True
    finally:
        logger.shutdown()


def test_batched_logger_can_bypass_batching():
    received_messages: list[str] = []
    logger = BatchedLogger(
        gui_callback=received_messages.append,
        batch_size=100,
        flush_interval_ms=60000,
        enable_batching=False,
    )

    try:
        for index in range(3):
            logger.log(f"message {index}")

        assert received_messages == ["message 0", "message 1", "message 2"]
    finally:
        logger.shutdown()


def test_throttled_progress_callback_throttles_but_forces_final(monkeypatch):
    fake_time = {"now": 0.0}
    received_updates: list[float] = []

    monkeypatch.setattr(
        batched_logger_module.time, "time", lambda: fake_time["now"]
    )

    callback = ThrottledProgressCallback(
        gui_callback=received_updates.append,
        min_interval_ms=100,
        force_final=True,
    )

    fake_time["now"] = 0.11
    callback(0, 1000)

    fake_time["now"] = 0.12
    callback(100, 1000)

    fake_time["now"] = 0.25
    callback(200, 1000)

    fake_time["now"] = 0.26
    callback(999, 1000)

    assert received_updates == pytest.approx([0.0, 20.0, 99.9])
