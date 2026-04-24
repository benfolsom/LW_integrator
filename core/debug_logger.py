"""Centralized debug logging system with automatic rotation and size management.

This module provides a singleton logger that:
- Captures all debug print statements to rotating log files
- Stores logs in /logcache/ directory with timestamped filenames
- Automatically rotates to a new log file when size limit is exceeded
- Purges oldest logs when total cache size exceeds limit
- Thread-safe for concurrent access during optimization/sweeps
- Integrates with Python's logging module to capture logger.info() calls
"""

import atexit
import logging
import os
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, TextIO


class DebugLogger:
    """Singleton debug logger with automatic rotation and cache management.

    Features:
    - Writes to both stdout and rotating log files
    - Thread-safe write operations
    - Automatic log rotation when file size exceeds limit
    - Automatic purge of oldest logs when cache size exceeds limit
    - Contextual log naming (GUI, testbed, optimization, etc.)

    Attributes
    ----------
    MAX_LOG_SIZE_MB : int
        Maximum size of a single log file before rotation (default: 10 MB)
    MAX_CACHE_SIZE_MB : int
        Maximum total size of logcache directory before purging (default: 100 MB)
    LOGCACHE_DIR : str
        Directory name for log storage (default: "logcache")
    """

    _instance = None
    _lock = threading.Lock()

    # Configuration
    MAX_LOG_SIZE_MB = 50  # Max size per log file (10-20 min at full verbosity)
    MAX_CACHE_SIZE_MB = 500  # Max total cache size
    LOGCACHE_DIR = "logcache"

    def __new__(cls):
        """Ensure only one instance exists (singleton pattern)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(DebugLogger, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the debug logger (only once)."""
        if self._initialized:
            return

        self._initialized = True
        self._write_lock = threading.Lock()
        self._log_file: Optional[TextIO] = None
        self._log_path: Optional[Path] = None
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        self._is_active = False
        self._context_name = "default"

        # Register cleanup on exit
        atexit.register(self.close)

    def initialize(
        self,
        working_dir: Optional[str] = None,
        context: str = "default",
        force_new_log: bool = False,
    ):
        """Initialize logging to the logcache directory.

        Parameters
        ----------
        working_dir : str, optional
            Working directory where logcache/ will be created.
            If None, uses current working directory.
        context : str, optional
            Context name for log file (e.g., "gui", "testbed", "optimization")
        force_new_log : bool, optional
            Rotate to a fresh log file even if the context is unchanged.
        """
        with self._lock:
            if self._is_active:
                # Already initialized, rotate when the caller requests a fresh run log
                if force_new_log or context != self._context_name:
                    self._context_name = context
                    self._rotate_log(force=True)
                return

            self._context_name = context

            # Set up logcache directory
            if working_dir is None:
                working_dir = os.getcwd()

            logcache_path = Path(working_dir) / self.LOGCACHE_DIR
            logcache_path.mkdir(parents=True, exist_ok=True)

            # Purge old logs if cache is too large
            self._purge_old_logs(logcache_path)

            # Create initial log file
            self._create_new_log(logcache_path)

            self._is_active = True

    def _create_new_log(self, logcache_path: Path):
        """Create a new log file with timestamp first for chronological sorting."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_filename = f"{timestamp}_{self._context_name}.log"
        self._log_path = logcache_path / log_filename

        try:
            self._log_file = open(self._log_path, "w", encoding="utf-8", buffering=1)
            self._write_to_file(
                f"=== Debug Log Started: {datetime.now().isoformat()} ===\n"
            )
            self._write_to_file(f"=== Context: {self._context_name} ===\n")
            self._write_to_file(f"=== Max log size: {self.MAX_LOG_SIZE_MB} MB ===\n")
            self._write_to_file(
                f"=== Max cache size: {self.MAX_CACHE_SIZE_MB} MB ===\n\n"
            )
        except Exception as e:
            print(
                f"[WARNING] Failed to create log file {self._log_path}: {e}",
                file=self._original_stdout,
            )
            self._log_file = None
            self._log_path = None

    def _rotate_log(self, force: bool = False):
        """Rotate to a new log file if size limit exceeded or forced.

        Parameters
        ----------
        force : bool
            Force rotation even if size limit not exceeded
        """
        if self._log_file is None or self._log_path is None:
            return

        # Check if rotation needed
        try:
            current_size = self._log_path.stat().st_size
            size_mb = current_size / (1024 * 1024)

            if not force and size_mb < self.MAX_LOG_SIZE_MB:
                return  # No rotation needed

            # Close current log
            self._write_to_file(
                f"\n=== Log Rotated: {datetime.now().isoformat()} ===\n"
            )
            self._write_to_file(f"=== Final size: {size_mb:.2f} MB ===\n")
            self._log_file.close()

            # Create new log
            logcache_path = self._log_path.parent
            self._create_new_log(logcache_path)

            # Purge old logs if needed
            self._purge_old_logs(logcache_path)

        except Exception as e:
            print(f"[WARNING] Failed to rotate log: {e}", file=self._original_stdout)

    def _purge_old_logs(self, logcache_path: Path):
        """Remove oldest logs if total cache size exceeds limit.

        Parameters
        ----------
        logcache_path : Path
            Path to logcache directory
        """
        try:
            # Get all log files sorted by modification time (oldest first)
            # Match both old format (debug_*) and new format (timestamp_*)
            log_files = sorted(
                list(logcache_path.glob("debug_*.log"))
                + list(logcache_path.glob("2*.log")),
                key=lambda p: p.stat().st_mtime,
            )

            if not log_files:
                return

            # Calculate total size
            total_size = sum(f.stat().st_size for f in log_files)
            max_size_bytes = self.MAX_CACHE_SIZE_MB * 1024 * 1024

            # Remove oldest files until under limit
            while total_size > max_size_bytes and len(log_files) > 1:
                # Keep at least the current log file
                oldest = log_files.pop(0)

                # Don't delete the current log
                if self._log_path and oldest == self._log_path:
                    continue

                try:
                    file_size = oldest.stat().st_size
                    oldest.unlink()
                    total_size -= file_size
                    print(
                        f"[LOGCACHE] Purged old log: {oldest.name} "
                        f"({file_size / (1024 * 1024):.2f} MB)",
                        file=self._original_stdout,
                    )
                except Exception as e:
                    print(
                        f"[WARNING] Failed to delete {oldest}: {e}",
                        file=self._original_stdout,
                    )

        except Exception as e:
            print(
                f"[WARNING] Failed to purge old logs: {e}", file=self._original_stdout
            )

    def _write_to_file(self, text: str):
        """Write text to log file (internal, no locking).

        Parameters
        ----------
        text : str
            Text to write
        """
        if self._log_file is not None:
            try:
                self._log_file.write(text)
                self._log_file.flush()
            except Exception as e:
                print(
                    f"[WARNING] Failed to write to log: {e}", file=self._original_stdout
                )

    def write(self, text: str, original_stream: Optional[TextIO] = None):
        """Write text to both stdout and log file (thread-safe).

        Parameters
        ----------
        text : str
            Text to write
        original_stream : TextIO, optional
            Stream that should still receive the console copy. Defaults to the
            original stdout captured when the logger was created.
        """
        with self._write_lock:
            stream = original_stream or self._original_stdout

            # Always write to the original console stream
            stream.write(text)
            stream.flush()

            # Write to log file if active
            if self._is_active and self._log_file is not None:
                self._write_to_file(text)

                # Check if rotation needed (only on newlines to avoid checking too often)
                if "\n" in text:
                    self._rotate_log()

    def flush(self):
        """Flush output streams."""
        with self._write_lock:
            self._original_stdout.flush()
            if self._log_file is not None:
                self._log_file.flush()

    def close(self):
        """Close the log file and cleanup."""
        with self._lock:
            if self._log_file is not None:
                try:
                    self._write_to_file(
                        f"\n=== Log Closed: {datetime.now().isoformat()} ===\n"
                    )
                    self._log_file.close()
                except Exception:
                    pass
                self._log_file = None
                self._log_path = None

            self._is_active = False

    def set_context(self, context: str):
        """Update the logging context and rotate to a new log file.

        Parameters
        ----------
        context : str
            New context name (e.g., "optimization_run_5")
        """
        if context != self._context_name:
            self._context_name = context
            if self._is_active:
                self._rotate_log(force=True)


class TeeStream:
    """Stream wrapper that writes to both original stream and debug logger.

    This class allows print() statements to be captured by the debug logger
    while still appearing in the console.
    """

    def __init__(self, original_stream: TextIO, logger: DebugLogger):
        """Initialize the tee stream.

        Parameters
        ----------
        original_stream : TextIO
            Original stdout/stderr stream
        logger : DebugLogger
            Debug logger instance
        """
        self.original_stream = original_stream
        self.logger = logger

    def write(self, text: str):
        """Write to both original stream and logger."""
        self.logger.write(text, original_stream=self.original_stream)

    def flush(self):
        """Flush both streams."""
        self.logger.flush()

    def __getattr__(self, name):
        """Delegate unknown attributes to original stream."""
        return getattr(self.original_stream, name)


class DebugLoggerHandler(logging.Handler):
    """Python logging handler that redirects to DebugLogger.

    This allows logger.info(), logger.warning(), etc. calls to be captured
    by the debug logging system alongside print() statements.
    """

    def __init__(self, debug_logger: DebugLogger):
        """Initialize the handler.

        Parameters
        ----------
        debug_logger : DebugLogger
            DebugLogger instance to write to
        """
        super().__init__()
        self.debug_logger = debug_logger
        # Set a simple formatter
        formatter = logging.Formatter("%(levelname)s: %(message)s")
        self.setFormatter(formatter)

    def emit(self, record):
        """Emit a log record to the debug logger.

        Parameters
        ----------
        record : logging.LogRecord
            Log record to emit
        """
        try:
            msg = self.format(record)
            # Write to debug logger (which writes to both stdout and file)
            self.debug_logger.write(msg + "\n")
        except Exception:
            self.handleError(record)


# Global logger instance
_global_logger: Optional[DebugLogger] = None
_logging_handler: Optional[DebugLoggerHandler] = None


def initialize_debug_logging(
    working_dir: Optional[str] = None,
    context: str = "default",
    force_new_log: bool = False,
):
    """Initialize global debug logging system.

    This function should be called once at application startup (GUI or testbed).
    It redirects stdout/stderr to capture all print statements to rotating log files.

    Parameters
    ----------
    working_dir : str, optional
        Working directory where logcache/ will be created.
        If None, uses current working directory.
    context : str, optional
        Context name for log file (e.g., "gui", "testbed", "optimization")

    Examples
    --------
    In GUI startup:
    >>> initialize_debug_logging(context="gui")

    In testbed:
    >>> initialize_debug_logging(working_dir="/path/to/results", context="testbed")

    In optimization run:
    >>> initialize_debug_logging(context="optimization_ga_run1")
    """
    global _global_logger

    if _global_logger is None:
        _global_logger = DebugLogger()

    _global_logger.initialize(
        working_dir=working_dir,
        context=context,
        force_new_log=force_new_log,
    )

    # Redirect stdout and stderr to tee streams once. Re-initialization should
    # reuse the same wrapper instead of nesting more TeeStreams.
    base_stdout = _unwrap_tee_stream(sys.stdout)
    base_stderr = _unwrap_tee_stream(sys.stderr)
    if not (
        isinstance(sys.stdout, TeeStream)
        and sys.stdout.logger is _global_logger
        and sys.stdout.original_stream is base_stdout
    ):
        sys.stdout = TeeStream(base_stdout, _global_logger)
    if not (
        isinstance(sys.stderr, TeeStream)
        and sys.stderr.logger is _global_logger
        and sys.stderr.original_stream is base_stderr
    ):
        sys.stderr = TeeStream(base_stderr, _global_logger)

    # Configure Python's logging module to also use our debug logger
    _configure_python_logging(_global_logger)


def set_logging_context(context: str, force_new_log: bool = False):
    """Update the logging context (creates new log file).

    Parameters
    ----------
    context : str
        New context name

    Examples
    --------
    >>> set_logging_context("optimization_nelder_mead_run5")
    """
    global _global_logger

    if _global_logger is not None:
        if force_new_log:
            _global_logger.initialize(context=context, force_new_log=True)
        else:
            _global_logger.set_context(context)


def close_debug_logging():
    """Close the debug logging system and restore original stdout/stderr.

    This is called automatically on program exit via atexit, but can also
    be called manually if needed.
    """
    global _global_logger, _logging_handler

    if _global_logger is not None:
        _global_logger.close()

    if _logging_handler is not None:
        logging.root.removeHandler(_logging_handler)
        _logging_handler = None

    # Restore original streams even if older nested wrappers are present.
    sys.stdout = _unwrap_tee_stream(sys.stdout)
    sys.stderr = _unwrap_tee_stream(sys.stderr)


def get_current_log_path() -> Optional[Path]:
    """Get the path to the current log file.

    Returns
    -------
    Path or None
        Path to current log file, or None if logging not initialized
    """
    global _global_logger

    if _global_logger is not None:
        return _global_logger._log_path
    return None


def _configure_python_logging(debug_logger: DebugLogger):
    """Configure Python's logging module to use DebugLogger.

    This adds a handler to the root logger so that logger.info(), logger.warning(),
    etc. calls from any module (like optimization/optimizer.py) are captured.

    Parameters
    ----------
    debug_logger : DebugLogger
        DebugLogger instance to send log messages to
    """
    global _logging_handler

    # Remove existing handler if present
    if _logging_handler is not None:
        logging.root.removeHandler(_logging_handler)

    # Create and add new handler
    _logging_handler = DebugLoggerHandler(debug_logger)
    _logging_handler.setLevel(logging.INFO)  # Capture INFO and above

    # Add to root logger so all loggers inherit it
    logging.root.addHandler(_logging_handler)
    logging.root.setLevel(logging.INFO)  # Set root level to INFO


def _unwrap_tee_stream(stream: TextIO) -> TextIO:
    """Collapse nested TeeStreams back to their base stream."""
    while isinstance(stream, TeeStream):
        stream = stream.original_stream
    return stream


__all__ = [
    "DebugLogger",
    "TeeStream",
    "DebugLoggerHandler",
    "initialize_debug_logging",
    "set_logging_context",
    "close_debug_logging",
    "get_current_log_path",
]
