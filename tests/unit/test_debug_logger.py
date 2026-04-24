"""Unit tests for debug log file creation, rotation, and purging."""

from __future__ import annotations

import logging
import os
import sys
from io import StringIO
from pathlib import Path

from core import debug_logger as debug_logger_module
from core.debug_logger import DebugLogger, TeeStream


def _reset_logging_state() -> None:
    try:
        debug_logger_module.close_debug_logging()
    except Exception:
        pass

    instance = DebugLogger._instance
    if instance is not None:
        try:
            instance.close()
        except Exception:
            pass

    DebugLogger._instance = None
    debug_logger_module._global_logger = None

    handler = debug_logger_module._logging_handler
    if handler is not None and handler in logging.root.handlers:
        logging.root.removeHandler(handler)
    debug_logger_module._logging_handler = None


def test_debug_logger_creates_log_and_rotates_on_context_change(tmp_path: Path):
    _reset_logging_state()
    logger = DebugLogger()

    try:
        logger.initialize(working_dir=str(tmp_path), context="unit")
        first_path = logger._log_path

        assert first_path is not None
        assert first_path.parent == tmp_path / "logcache"
        assert "unit" in first_path.name

        logger.write("hello world\n")
        logger.set_context("second")
        second_path = logger._log_path

        assert second_path is not None
        assert second_path != first_path
        assert "second" in second_path.name
        assert "hello world" in first_path.read_text(encoding="utf-8")
    finally:
        logger.close()
        _reset_logging_state()


def test_debug_logger_purges_old_logs_but_keeps_current_log(tmp_path: Path):
    _reset_logging_state()
    logger = DebugLogger()
    logcache = tmp_path / "logcache"
    logcache.mkdir()

    current = logcache / "20260103_current.log"
    current.write_text("current" * 100, encoding="utf-8")

    old_one = logcache / "20260101_old.log"
    old_one.write_text("old-one" * 100, encoding="utf-8")

    old_two = logcache / "20260102_old.log"
    old_two.write_text("old-two" * 100, encoding="utf-8")

    now = 1_700_000_000
    os.utime(old_one, (now - 30, now - 30))
    os.utime(old_two, (now - 20, now - 20))
    os.utime(current, (now - 10, now - 10))

    logger._log_path = current
    logger.MAX_CACHE_SIZE_MB = 0.0001

    try:
        logger._purge_old_logs(logcache)

        assert current.exists()
        assert not old_one.exists()
        assert not old_two.exists()
    finally:
        logger.close()
        _reset_logging_state()


def test_initialize_debug_logging_force_new_log_reuses_single_tee_wrapper(
    tmp_path: Path,
):
    _reset_logging_state()
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    try:
        debug_logger_module.initialize_debug_logging(
            working_dir=str(tmp_path),
            context="first",
        )
        first_stdout = sys.stdout
        first_stderr = sys.stderr
        first_log = debug_logger_module.get_current_log_path()

        debug_logger_module.initialize_debug_logging(
            working_dir=str(tmp_path),
            context="first",
            force_new_log=True,
        )
        second_log = debug_logger_module.get_current_log_path()

        assert isinstance(first_stdout, TeeStream)
        assert isinstance(first_stderr, TeeStream)
        assert sys.stdout is first_stdout
        assert sys.stderr is first_stderr
        assert first_log is not None
        assert second_log is not None
        assert second_log != first_log
    finally:
        debug_logger_module.close_debug_logging()
        assert sys.stdout is original_stdout
        assert sys.stderr is original_stderr
        _reset_logging_state()


def test_initialize_debug_logging_preserves_stdout_and_stderr_streams(
    tmp_path: Path, monkeypatch
):
    _reset_logging_state()
    stdout_buffer = StringIO()
    stderr_buffer = StringIO()
    monkeypatch.setattr(sys, "stdout", stdout_buffer)
    monkeypatch.setattr(sys, "stderr", stderr_buffer)

    try:
        debug_logger_module.initialize_debug_logging(
            working_dir=str(tmp_path),
            context="streams",
        )
        sys.stdout.write("stdout line\n")
        sys.stderr.write("stderr line\n")
        log_path = debug_logger_module.get_current_log_path()
    finally:
        debug_logger_module.close_debug_logging()
        _reset_logging_state()

    assert stdout_buffer.getvalue() == "stdout line\n"
    assert stderr_buffer.getvalue() == "stderr line\n"
    assert log_path is not None
    log_text = log_path.read_text(encoding="utf-8")
    assert "stdout line" in log_text
    assert "stderr line" in log_text
