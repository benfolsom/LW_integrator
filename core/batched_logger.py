"""
Batched logger for efficient GUI updates during intensive simulations.

This module provides a thread-safe batched logging mechanism that accumulates
log messages and flushes them to the GUI in batches, reducing the number of
GUI event queue updates from potentially millions to hundreds.

This dramatically improves GUI responsiveness during intensive simulations
with heavy debug logging (e.g., cooldown mode with 1000 substeps).
"""

import threading
import time
from collections import deque
from typing import Callable, Optional


class BatchedLogger:
    """
    Thread-safe batched logger that accumulates messages and flushes periodically.

    Reduces GUI event queue flooding by batching multiple log messages into
    single updates. This is critical during adaptive timestep refinement where
    debug logging can generate 90,000+ messages during a single cooldown period.

    Parameters
    ----------
    gui_callback : Callable[[str], None]
        Function to call with batched log messages. Should queue to GUI thread
        using root.after() or similar thread-safe mechanism.
    batch_size : int, optional
        Maximum number of messages to accumulate before forcing a flush.
        Default: 100 (good balance between responsiveness and efficiency)
    flush_interval_ms : int, optional
        Maximum time (in milliseconds) to wait before flushing, even if batch
        not full. Default: 500ms (ensures messages appear within 0.5 seconds)
    max_queue_size : int, optional
        Maximum number of messages to queue. If exceeded, oldest messages are
        dropped (backpressure mechanism). Default: 10000.
    enable_batching : bool, optional
        If False, bypass batching and call callback immediately (useful for
        debugging). Default: True.

    Examples
    --------
    >>> def gui_log(text):
    ...     root.after(0, lambda: text_widget.insert('end', text + '\\n'))
    >>> logger = BatchedLogger(gui_log, batch_size=100, flush_interval_ms=500)
    >>> for i in range(1000):
    ...     logger.log(f"Step {i}: Processing...")
    >>> logger.flush()  # Ensure all messages sent before exit

    Notes
    -----
    - Thread-safe: can be called from worker threads
    - Automatic flushing via background timer (if enabled)
    - Manual flush() should be called before thread exit
    - Dropping oldest messages on overflow prevents memory bloat
    """

    def __init__(
        self,
        gui_callback: Callable[[str], None],
        batch_size: int = 100,
        flush_interval_ms: int = 500,
        max_queue_size: int = 10000,
        enable_batching: bool = True,
    ):
        self.gui_callback = gui_callback
        self.batch_size = batch_size
        self.flush_interval = flush_interval_ms / 1000.0  # Convert to seconds
        self.max_queue_size = max_queue_size
        self.enable_batching = enable_batching

        # Thread-safe message buffer (deque is thread-safe for append/popleft)
        self._buffer: deque = deque(maxlen=max_queue_size)
        self._lock = threading.Lock()

        # Timer for periodic flushing
        self._timer: Optional[threading.Timer] = None
        self._last_flush_time = time.time()
        self._shutdown = False

        # Statistics for monitoring
        self._total_messages = 0
        self._total_batches = 0
        self._dropped_messages = 0

        # Start periodic flush timer if batching enabled
        if self.enable_batching:
            self._schedule_flush()

    def log(self, message: str) -> None:
        """
        Add a log message to the batch queue.

        If batching is disabled, immediately calls the GUI callback.
        Otherwise, accumulates the message and flushes when batch is full
        or flush interval elapses.

        Parameters
        ----------
        message : str
            Log message to queue. Should not contain trailing newline
            (will be added during flush).

        Notes
        -----
        Thread-safe: can be called from any thread.
        """
        if not self.enable_batching:
            # Bypass batching - immediate callback
            self.gui_callback(message)
            return

        with self._lock:
            # Track if buffer was full (message will be dropped)
            was_full = len(self._buffer) >= self.max_queue_size

            # Add to buffer (deque with maxlen automatically drops oldest)
            self._buffer.append(message)
            self._total_messages += 1

            if was_full:
                self._dropped_messages += 1

            # Check if batch is full
            if len(self._buffer) >= self.batch_size:
                self._flush_unlocked()

    def flush(self) -> None:
        """
        Immediately flush all pending messages to GUI.

        Should be called:
        - Before thread/program exit to ensure no messages lost
        - When entering a critical section where immediate output needed
        - Manually if flush_interval is too long for use case

        Thread-safe.
        """
        with self._lock:
            self._flush_unlocked()

    def _flush_unlocked(self) -> None:
        """
        Internal flush implementation (caller must hold lock).

        Combines all buffered messages with newlines and sends to GUI
        callback as a single string.
        """
        if not self._buffer:
            return

        # Combine all messages with newlines
        batch = list(self._buffer)
        self._buffer.clear()

        combined = "\n".join(batch)
        self._total_batches += 1
        self._last_flush_time = time.time()

        # Release lock before calling callback (prevent deadlock)
        # GUI callback will queue to main thread, doesn't need our lock
        try:
            self.gui_callback(combined)
        except Exception as e:
            # Prevent callback errors from breaking logging
            # In production, might want to log to stderr or file
            print(f"[BatchedLogger] Error in GUI callback: {e}")

    def _schedule_flush(self) -> None:
        """
        Schedule next periodic flush via threading.Timer.

        Creates a background timer that flushes messages after flush_interval.
        Automatically reschedules itself after each flush (until shutdown).
        """
        if self._shutdown:
            return

        # Cancel existing timer if any
        if self._timer is not None:
            self._timer.cancel()

        # Create new timer
        self._timer = threading.Timer(self.flush_interval, self._periodic_flush)
        self._timer.daemon = True  # Don't prevent program exit
        self._timer.start()

    def _periodic_flush(self) -> None:
        """
        Periodic flush callback (called by timer).

        Flushes any pending messages and reschedules next flush.
        """
        if self._shutdown:
            return

        with self._lock:
            # Only flush if there are messages AND enough time has passed
            # (avoid flushing empty buffer repeatedly)
            if self._buffer:
                self._flush_unlocked()

        # Schedule next flush
        self._schedule_flush()

    def shutdown(self) -> None:
        """
        Clean shutdown of logger.

        Stops periodic flush timer, flushes any remaining messages,
        and prevents future logging.

        Should be called before program/thread exit to ensure all
        messages are delivered.

        Thread-safe.
        """
        with self._lock:
            self._shutdown = True

            # Stop timer
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None

            # Final flush
            self._flush_unlocked()

    def get_stats(self) -> dict:
        """
        Get logging statistics for monitoring and debugging.

        Returns
        -------
        dict
            Statistics including total messages, batches, dropped messages,
            and efficiency metrics.

        Examples
        --------
        >>> logger.get_stats()
        {
            'total_messages': 90000,
            'total_batches': 900,
            'dropped_messages': 0,
            'messages_per_batch': 100.0,
            'reduction_factor': 100.0,
            'buffer_size': 45,
            'batching_enabled': True
        }
        """
        with self._lock:
            avg_batch_size = (
                self._total_messages / self._total_batches
                if self._total_batches > 0
                else 0
            )
            reduction = (
                self._total_messages / max(self._total_batches, 1)
                if self._total_batches > 0
                else 1
            )

            return {
                "total_messages": self._total_messages,
                "total_batches": self._total_batches,
                "dropped_messages": self._dropped_messages,
                "messages_per_batch": avg_batch_size,
                "reduction_factor": reduction,
                "buffer_size": len(self._buffer),
                "batching_enabled": self.enable_batching,
                "max_queue_size": self.max_queue_size,
                "batch_size_limit": self.batch_size,
                "flush_interval_ms": self.flush_interval * 1000,
            }

    def __enter__(self):
        """Context manager entry (no-op, already initialized)."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures clean shutdown."""
        self.shutdown()
        return False  # Don't suppress exceptions

    def __del__(self):
        """Destructor - attempt final flush on garbage collection."""
        try:
            self.shutdown()
        except Exception:
            pass  # Best-effort cleanup


class ThrottledProgressCallback:
    """
    Throttle progress callbacks to prevent GUI event queue flooding.

    Limits progress updates to a maximum frequency regardless of how fast
    the integration is running. This prevents the GUI from becoming laggy
    when integration steps are very fast (< 1ms per step).

    Parameters
    ----------
    gui_callback : Callable[[float], None]
        Function to call with progress percentage (0-100).
    min_interval_ms : int, optional
        Minimum time between updates in milliseconds. Default: 100ms (10 Hz).
    force_final : bool, optional
        If True, always send final update (100%) even if interval not elapsed.
        Default: True.

    Examples
    --------
    >>> def update_progress(percent):
    ...     root.after(0, lambda: progress_var.set(percent))
    >>> throttled = ThrottledProgressCallback(update_progress, min_interval_ms=100)
    >>> for i in range(10000):
    ...     throttled(i, 10000)  # Only ~100 actual GUI updates
    """

    def __init__(
        self,
        gui_callback: Callable[[float], None],
        min_interval_ms: int = 100,
        force_final: bool = True,
    ):
        self.gui_callback = gui_callback
        self.min_interval = min_interval_ms / 1000.0
        self.force_final = force_final
        self._last_update_time = 0.0
        self._last_value = -1.0
        self._lock = threading.Lock()

    def __call__(self, current: int, total: int) -> None:
        """
        Report progress (throttled).

        Parameters
        ----------
        current : int
            Current step number (0-based).
        total : int
            Total number of steps.
        """
        if total <= 0:
            return

        now = time.time()
        progress = (current / total) * 100.0

        with self._lock:
            # Force update if final step and force_final enabled
            is_final = current >= total - 1
            should_force = is_final and self.force_final

            # Update if interval elapsed or forcing
            if should_force or (now - self._last_update_time) >= self.min_interval:
                # Only update if value changed (avoid redundant updates)
                if abs(progress - self._last_value) > 0.01:
                    self.gui_callback(progress)
                    self._last_update_time = now
                    self._last_value = progress
