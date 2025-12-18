"""Camera statistics collection and reporting.

Provides metrics collection for camera operations including:
- Capture success/failure rates
- Timing statistics (min, max, avg, p95)
- Error categorization
- Rolling windows for recent performance

Thread-safe for concurrent camera access.

Example:
    stats = CameraStats()

    # Record captures
    stats.record_capture(camera_id=0, duration_ms=150, success=True)
    stats.record_capture(camera_id=0, duration_ms=200, success=True)
    stats.record_capture(camera_id=0, duration_ms=0, success=False,
                         error_type="timeout")

    # Get summary
    summary = stats.get_summary(camera_id=0)
    print(f"Success rate: {summary.success_rate:.1%}")
    print(f"Avg duration: {summary.avg_duration_ms:.1f}ms")

    # Export for session storage
    data = stats.to_dict()
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

# =============================================================================
# Constants
# =============================================================================

#: Default number of capture records to retain in rolling window.
#: Provides ~10min history at typical capture rates. Increase for
#: longer history or more accurate percentile calculations.
DEFAULT_STATS_WINDOW_SIZE: int = 1000


# =============================================================================
# Helpers
# =============================================================================


def _utc_now() -> datetime:
    """Return current UTC datetime.

    Centralized timestamp creation for consistency across the module.
    All timestamps in telescope-mcp statistics use UTC to avoid
    timezone confusion in observation logs.

    Args:
        None.

    Returns:
        Current datetime with UTC timezone attached.

    Example:
        >>> ts = _utc_now()
        >>> ts.tzinfo
        datetime.timezone.utc
    """
    return datetime.now(UTC)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class StatsSummary:
    """Summary statistics for a camera.

    Attributes:
        camera_id: Camera identifier
        total_captures: Total capture attempts
        successful_captures: Successful captures
        failed_captures: Failed captures
        success_rate: Success rate (0.0 to 1.0)
        min_duration_ms: Minimum capture duration
        max_duration_ms: Maximum capture duration
        avg_duration_ms: Average capture duration
        p95_duration_ms: 95th percentile duration
        error_counts: Count by error type
        last_capture_time: Time of last capture
        uptime_seconds: Time since stats reset
    """

    camera_id: int
    total_captures: int = 0
    successful_captures: int = 0
    failed_captures: int = 0
    success_rate: float = 0.0
    min_duration_ms: float = 0.0
    max_duration_ms: float = 0.0
    avg_duration_ms: float = 0.0
    p95_duration_ms: float = 0.0
    error_counts: dict[str, int] = field(default_factory=dict)
    last_capture_time: datetime | None = None
    uptime_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert statistics summary to a serializable dictionary.

        Creates a plain dictionary representation suitable for JSON
        serialization, session storage, or API responses. All values
        are converted to JSON-compatible types.

        This is the primary export format for camera performance metrics,
        enabling persistence to ASDF session files and real-time
        dashboard updates.

        Args:
            None (uses self attributes).

        Returns:
            Dictionary containing all statistics fields:
            - camera_id (int): Camera identifier
            - total_captures (int): Total attempts
            - successful_captures (int): Success count
            - failed_captures (int): Failure count
            - success_rate (float): 0.0-1.0 ratio
            - min_duration_ms (float): Fastest capture
            - max_duration_ms (float): Slowest capture
            - avg_duration_ms (float): Mean duration
            - p95_duration_ms (float): 95th percentile
            - error_counts (dict): {error_type: count}
            - last_capture_time (str|None): ISO timestamp
            - uptime_seconds (float): Collector uptime

        Raises:
            None.

        Example:
            >>> summary = collector.get_summary()
            >>> data = summary.to_dict()
            >>> json.dumps(data)  # Safe to serialize
            >>> data["success_rate"]
            0.95
        """
        return {
            "camera_id": self.camera_id,
            "total_captures": self.total_captures,
            "successful_captures": self.successful_captures,
            "failed_captures": self.failed_captures,
            "success_rate": self.success_rate,
            "min_duration_ms": self.min_duration_ms,
            "max_duration_ms": self.max_duration_ms,
            "avg_duration_ms": self.avg_duration_ms,
            "p95_duration_ms": self.p95_duration_ms,
            "error_counts": self.error_counts.copy(),
            "last_capture_time": (
                self.last_capture_time.isoformat() if self.last_capture_time else None
            ),
            "uptime_seconds": self.uptime_seconds,
        }


@dataclass
class CaptureRecord:
    """Single capture record for statistics."""

    timestamp: float  # monotonic time
    duration_ms: float
    success: bool
    error_type: str | None = None


class CameraStatsCollector:
    """Statistics collector for a single camera.

    Maintains rolling window of recent captures and computes
    summary statistics on demand.
    """

    def __init__(
        self,
        camera_id: int,
        window_size: int = DEFAULT_STATS_WINDOW_SIZE,
    ) -> None:
        """Initialize a statistics collector for a single camera.

        Creates a thread-safe collector that maintains a rolling window
        of capture records. Older records are automatically discarded
        when the window fills, ensuring bounded memory usage while
        preserving recent performance data for analysis.

        The collector tracks both cumulative totals (for overall success
        rate) and windowed records (for duration percentiles), balancing
        historical accuracy with recent performance visibility.

        Args:
            camera_id: Unique identifier for the camera (0=finder, 1=main).
                Used to label statistics in multi-camera setups.
            window_size: Maximum capture records to retain for rolling
                statistics. Default 1000 provides ~10min of history at
                typical capture rates. Larger windows increase memory
                but improve percentile accuracy.

        Returns:
            None (constructor).

        Raises:
            None.

        Example:
            >>> collector = CameraStatsCollector(camera_id=0, window_size=500)
            >>> collector.record(duration_ms=150, success=True)
            >>> summary = collector.get_summary()
        """
        self.camera_id = camera_id
        self._window_size = window_size
        self._records: deque[CaptureRecord] = deque(maxlen=window_size)
        self._error_counts: dict[str, int] = {}
        self._total_captures = 0
        self._successful_captures = 0
        self._start_time = time.monotonic()
        self._last_capture_time: datetime | None = None
        self._lock = threading.Lock()

    def record(
        self,
        duration_ms: float,
        success: bool,
        error_type: str | None = None,
    ) -> None:
        """Record a camera capture attempt for statistics tracking.

        Thread-safe method to log capture outcomes. Each call updates
        both cumulative counters (total, successful, failed) and the
        rolling window of detailed records. Error types are tracked
        separately for failure analysis.

        This is the primary data ingestion point for the statistics
        system, typically called by camera capture code after each
        frame acquisition attempt.

        Args:
            duration_ms: Capture duration in milliseconds. For successful
                captures, this should be the actual time. For failures,
                use 0 or the time until failure was detected.
            success: True if capture completed successfully and produced
                a valid frame. False for any failure condition.
            error_type: Categorized error string for failures. Common
                values: 'timeout', 'disconnected', 'buffer_overflow',
                'exposure_failed'. None for successful captures or
                uncategorized failures.

        Returns:
            None.

        Raises:
            None. Thread-safe via internal lock.

        Example:
            >>> collector = CameraStatsCollector(camera_id=0)
            >>> collector.record(duration_ms=150.5, success=True)
            >>> collector.record(duration_ms=0, success=False, error_type="timeout")
        """
        record = CaptureRecord(
            timestamp=time.monotonic(),
            duration_ms=duration_ms,
            success=success,
            error_type=error_type,
        )

        with self._lock:
            self._records.append(record)
            self._total_captures += 1
            if success:
                self._successful_captures += 1
            elif error_type:
                self._error_counts[error_type] = (
                    self._error_counts.get(error_type, 0) + 1
                )
            self._last_capture_time = _utc_now()

    def get_summary(self) -> StatsSummary:
        """Compute and return current statistics summary.

        Calculates comprehensive statistics from the cumulative counters
        and rolling window of capture records. Duration statistics (min,
        max, avg, p95) are computed from successful captures only.

        This is the primary statistics query method, used by dashboards,
        session logging, and diagnostic tools to assess camera health
        and performance.

        The computation is thread-safe and returns a snapshot of the
        current state; subsequent captures won't affect the returned
        summary.

        Args:
            None (uses internal state).

        Returns:
            StatsSummary dataclass containing:
            - camera_id: This collector's camera ID
            - total_captures: All-time capture attempts
            - successful_captures: All-time successes
            - failed_captures: All-time failures
            - success_rate: successful/total (0.0 if no captures)
            - min/max/avg/p95_duration_ms: From windowed successes
            - error_counts: {error_type: count} copy
            - last_capture_time: UTC datetime of most recent capture
            - uptime_seconds: Time since collector creation/reset

        Raises:
            None.

        Example:
            >>> collector = CameraStatsCollector(camera_id=0)
            >>> collector.record(duration_ms=100, success=True)
            >>> collector.record(duration_ms=200, success=True)
            >>> summary = collector.get_summary()
            >>> summary.avg_duration_ms
            150.0
            >>> summary.success_rate
            1.0
        """
        # Copy data under lock, compute statistics outside lock
        # This minimizes lock hold time (O(n log n) sort happens unlocked)
        with self._lock:
            total = self._total_captures
            successful = self._successful_captures
            error_counts = self._error_counts.copy()
            last_capture_time = self._last_capture_time
            start_time = self._start_time
            camera_id = self.camera_id
            # Copy durations from successful records
            durations = [
                r.duration_ms for r in self._records if r.success and r.duration_ms > 0
            ]

        # Expensive computations outside the lock
        failed = total - successful

        if durations:
            min_dur = min(durations)
            max_dur = max(durations)
            avg_dur = sum(durations) / len(durations)
            p95_dur = _percentile(sorted(durations), 95)  # O(n log n)
        else:
            min_dur = max_dur = avg_dur = p95_dur = 0.0

        return StatsSummary(
            camera_id=camera_id,
            total_captures=total,
            successful_captures=successful,
            failed_captures=failed,
            success_rate=successful / total if total > 0 else 0.0,
            min_duration_ms=min_dur,
            max_duration_ms=max_dur,
            avg_duration_ms=avg_dur,
            p95_duration_ms=p95_dur,
            error_counts=error_counts,
            last_capture_time=last_capture_time,
            uptime_seconds=time.monotonic() - start_time,
        )

    def reset(self) -> None:
        """Reset all statistics to initial state.

        Clears all capture records, counters, and error tracking. Resets
        the uptime timer to zero. Use this when starting a new observation
        session or when previous statistics are no longer relevant.

        Thread-safe; any concurrent record() calls will see the reset
        state. The camera_id and window_size configuration are preserved.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.

        Example:
            >>> collector = CameraStatsCollector(camera_id=0)
            >>> collector.record(duration_ms=100, success=True)
            >>> collector.get_summary().total_captures
            1
            >>> collector.reset()
            >>> collector.get_summary().total_captures
            0
        """
        with self._lock:
            self._records.clear()
            self._error_counts.clear()
            self._total_captures = 0
            self._successful_captures = 0
            self._start_time = time.monotonic()
            self._last_capture_time = None


class CameraStats:
    """Statistics manager for multiple cameras.

    Thread-safe container for per-camera statistics collectors.

    Usage:
        stats = CameraStats()

        # Record captures
        stats.record_capture(camera_id=0, duration_ms=150, success=True)

        # Get per-camera summary
        summary = stats.get_summary(camera_id=0)

        # Get all summaries
        all_stats = stats.get_all_summaries()

        # Export for session
        data = stats.to_dict()
    """

    def __init__(self, window_size: int = DEFAULT_STATS_WINDOW_SIZE) -> None:
        """Initialize the multi-camera statistics manager.

        Creates a thread-safe container that lazily creates per-camera
        collectors on first access. All cameras share the same window
        size configuration.

        This is the main entry point for statistics in telescope-mcp.
        Inject via dependency injection to camera classes.

        Args:
            window_size: Maximum capture records per camera for rolling
                statistics. Passed to each CameraStatsCollector created.
                Default 1000 balances memory with statistical accuracy.

        Returns:
            None (constructor).

        Raises:
            None.

        Example:
            >>> stats = CameraStats(window_size=500)
            >>> stats.record_capture(camera_id=0, duration_ms=100, success=True)
        """
        self._window_size = window_size
        self._collectors: dict[int, CameraStatsCollector] = {}
        self._lock = threading.Lock()

    def _get_collector(self, camera_id: int) -> CameraStatsCollector:
        """Get or create the statistics collector for a camera.

        Thread-safe lazy initialization of per-camera collectors. If a
        collector doesn't exist for the given camera_id, one is created
        with the manager's configured window_size.

        This internal method is used by all public methods that need
        camera-specific statistics access.

        Args:
            camera_id: Camera identifier (0=finder, 1=main, etc.).

        Returns:
            CameraStatsCollector instance for the specified camera.
            Always returns a valid collector (creates if needed).

        Raises:
            None.

        Example:
            >>> stats = CameraStats()
            >>> collector = stats._get_collector(0)
            >>> collector.camera_id
            0
        """
        with self._lock:
            if camera_id not in self._collectors:
                self._collectors[camera_id] = CameraStatsCollector(
                    camera_id, self._window_size
                )
            return self._collectors[camera_id]

    def record_capture(
        self,
        camera_id: int,
        duration_ms: float,
        success: bool,
        error_type: str | None = None,
    ) -> None:
        """Record a capture attempt for statistics tracking.

        Primary method for logging camera captures. Routes the record
        to the appropriate per-camera collector, creating it if needed.
        Thread-safe for concurrent multi-camera operations.

        Call this after every capture attempt (successful or not) to
        maintain accurate performance metrics. The statistics enable
        session analysis, alerting on degradation, and capacity planning.

        Args:
            camera_id: Camera identifier (0=finder, 1=main). Determines
                which collector receives the record.
            duration_ms: Capture duration in milliseconds. Use actual
                time for successes, 0 or time-to-failure for errors.
            success: True if a valid frame was captured, False otherwise.
            error_type: Error category string for failures. Use consistent
                values like 'timeout', 'disconnected', 'hardware_error'
                to enable error analysis. None for successes.

        Returns:
            None.

        Raises:
            None. Thread-safe; creates collector if needed.

        Example:
            >>> stats = CameraStats()
            >>> stats.record_capture(camera_id=0, duration_ms=150, success=True)
            >>> stats.record_capture(
            ...     camera_id=0, duration_ms=5000,
            ...     success=False, error_type="timeout"
            ... )
        """
        collector = self._get_collector(camera_id)
        collector.record(duration_ms, success, error_type)

    def get_summary(self, camera_id: int) -> StatsSummary:
        """Get statistics summary for a specific camera.

        Retrieves computed statistics for the specified camera. If no
        captures have been recorded for this camera, returns a summary
        with zero values (a new collector is created).

        Use this for single-camera queries. For all cameras, use
        get_all_summaries() instead.

        Args:
            camera_id: Camera identifier to query (0=finder, 1=main).

        Returns:
            StatsSummary dataclass with computed statistics including
            success rate, duration percentiles, error counts, etc.
            See StatsSummary.to_dict() for full field list.

        Raises:
            None. Creates empty collector if camera not seen before.

        Example:
            >>> stats = CameraStats()
            >>> stats.record_capture(0, 100, True)
            >>> summary = stats.get_summary(camera_id=0)
            >>> print(f"Success rate: {summary.success_rate:.0%}")
            Success rate: 100%
        """
        collector = self._get_collector(camera_id)
        return collector.get_summary()

    def get_all_summaries(self) -> dict[int, StatsSummary]:
        """Get statistics summaries for all tracked cameras.

        Returns summaries only for cameras that have recorded at least
        one capture. Thread-safe snapshot of current state.

        Use this for dashboard displays, session exports, or any
        operation needing a complete view of system performance.

        Args:
            None.

        Returns:
            Dictionary mapping camera_id (int) to StatsSummary.
            Empty dict if no captures recorded for any camera.
            Example: {0: StatsSummary(...), 1: StatsSummary(...)}

        Raises:
            None.

        Example:
            >>> stats = CameraStats()
            >>> stats.record_capture(0, 100, True)
            >>> stats.record_capture(1, 200, True)
            >>> summaries = stats.get_all_summaries()
            >>> for cam_id, summary in summaries.items():
            ...     print(f"Camera {cam_id}: {summary.total_captures} captures")
            Camera 0: 1 captures
            Camera 1: 1 captures
        """
        with self._lock:
            return {
                camera_id: collector.get_summary()
                for camera_id, collector in self._collectors.items()
            }

    def reset(self, camera_id: int | None = None) -> None:
        """Reset statistics for one or all cameras.

        Clears capture records and counters. Use when starting a new
        observation session or to clear stale data after configuration
        changes. Collector objects are preserved (not deleted).

        Thread-safe; concurrent record_capture() calls may interleave
        with reset but won't cause errors.

        Args:
            camera_id: Specific camera to reset (0, 1, etc.), or None
                to reset all tracked cameras. If the camera_id hasn't
                been seen, this is a no-op (no error).

        Returns:
            None.

        Raises:
            None.

        Example:
            >>> stats = CameraStats()
            >>> stats.record_capture(0, 100, True)
            >>> stats.record_capture(1, 100, True)
            >>> stats.reset(camera_id=0)  # Reset only camera 0
            >>> stats.get_summary(0).total_captures
            0
            >>> stats.get_summary(1).total_captures
            1
            >>> stats.reset()  # Reset all
        """
        with self._lock:
            if camera_id is not None:
                if camera_id in self._collectors:
                    self._collectors[camera_id].reset()
            else:
                for collector in self._collectors.values():
                    collector.reset()

    def to_dict(self) -> dict[str, Any]:
        """Export all camera statistics for serialization.

        Creates a complete snapshot of statistics for all tracked
        cameras in a JSON-serializable format. Used for session
        persistence, API responses, and debugging.

        The output format is designed for ASDF session files and
        integrates with telescope-mcp's data management system.

        Args:
            None.

        Returns:
            Dictionary with structure:
            {
                "cameras": {
                    "0": {StatsSummary fields...},
                    "1": {StatsSummary fields...}
                },
                "timestamp": "2025-01-15T10:30:00+00:00"
            }
            Camera IDs are string keys for JSON compatibility.

        Raises:
            None.

        Example:
            >>> stats = CameraStats()
            >>> stats.record_capture(0, 150, True)
            >>> data = stats.to_dict()
            >>> json.dumps(data)  # Safe to serialize
            >>> data["cameras"]["0"]["success_rate"]
            1.0
        """
        summaries = self.get_all_summaries()
        return {
            "cameras": {
                str(camera_id): summary.to_dict()
                for camera_id, summary in summaries.items()
            },
            "timestamp": _utc_now().isoformat(),
        }


def _percentile(sorted_data: list[float], p: float) -> float:
    """Calculate a percentile value from pre-sorted data.

    Uses linear interpolation between data points for percentiles
    that fall between values. This matches numpy's 'linear'
    interpolation method.

    Common use in telescope-mcp is p95 capture duration to identify
    outlier captures without being skewed by maximum values.

    Args:
        sorted_data: List of float values, must be sorted ascending.
            Empty list returns 0.0. Caller must ensure sorted order.
        p: Percentile to calculate, 0-100 inclusive. Common values:
            50 (median), 90, 95, 99.

    Returns:
        The percentile value. Returns 0.0 for empty input.
        For single-element input, returns that element.
        For p=0 returns minimum, p=100 returns maximum.

    Raises:
        ValueError: If p is outside the range [0, 100].

    Example:
        >>> _percentile([1.0, 2.0, 3.0, 4.0, 5.0], 50)
        3.0
        >>> _percentile([100.0, 150.0, 200.0], 95)
        195.0
        >>> _percentile([], 50)
        0.0
    """
    if not 0 <= p <= 100:
        raise ValueError(f"Percentile must be between 0 and 100, got {p}")

    if not sorted_data:
        return 0.0

    # Calculate position in data: k ∈ [0, len-1]
    k = (len(sorted_data) - 1) * (p / 100)
    floor_idx = int(k)
    ceil_idx = min(floor_idx + 1, len(sorted_data) - 1)

    # No interpolation needed if k is exact integer or at boundary
    if floor_idx == ceil_idx:
        return sorted_data[floor_idx]

    # Linear interpolation: weight by fractional distance from floor
    fraction = k - floor_idx  # Distance from floor to k (0.0 to 1.0)
    return sorted_data[floor_idx] * (1 - fraction) + sorted_data[ceil_idx] * fraction
