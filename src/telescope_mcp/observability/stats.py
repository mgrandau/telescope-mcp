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
from datetime import datetime, timezone
from typing import Any


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
        """Convert to dictionary for serialization."""
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
                self.last_capture_time.isoformat() 
                if self.last_capture_time else None
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
        window_size: int = 1000,
    ) -> None:
        """Initialize collector.
        
        Args:
            camera_id: Camera identifier
            window_size: Max records to keep for rolling stats
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
        """Record a capture attempt.
        
        Args:
            duration_ms: Capture duration in milliseconds
            success: Whether capture succeeded
            error_type: Error type if failed (e.g., "timeout", "disconnected")
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
            self._last_capture_time = datetime.now(timezone.utc)
            
            if success:
                self._successful_captures += 1
            elif error_type:
                self._error_counts[error_type] = (
                    self._error_counts.get(error_type, 0) + 1
                )
    
    def get_summary(self) -> StatsSummary:
        """Get current statistics summary.
        
        Returns:
            StatsSummary with computed statistics
        """
        with self._lock:
            total = self._total_captures
            successful = self._successful_captures
            failed = total - successful
            
            # Compute duration stats from recent records
            durations = [
                r.duration_ms for r in self._records 
                if r.success and r.duration_ms > 0
            ]
            
            if durations:
                min_dur = min(durations)
                max_dur = max(durations)
                avg_dur = sum(durations) / len(durations)
                p95_dur = _percentile(sorted(durations), 95)
            else:
                min_dur = max_dur = avg_dur = p95_dur = 0.0
            
            return StatsSummary(
                camera_id=self.camera_id,
                total_captures=total,
                successful_captures=successful,
                failed_captures=failed,
                success_rate=successful / total if total > 0 else 0.0,
                min_duration_ms=min_dur,
                max_duration_ms=max_dur,
                avg_duration_ms=avg_dur,
                p95_duration_ms=p95_dur,
                error_counts=self._error_counts.copy(),
                last_capture_time=self._last_capture_time,
                uptime_seconds=time.monotonic() - self._start_time,
            )
    
    def reset(self) -> None:
        """Reset all statistics."""
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
    
    def __init__(self, window_size: int = 1000) -> None:
        """Initialize stats manager.
        
        Args:
            window_size: Max records per camera for rolling stats
        """
        self._window_size = window_size
        self._collectors: dict[int, CameraStatsCollector] = {}
        self._lock = threading.Lock()
    
    def _get_collector(self, camera_id: int) -> CameraStatsCollector:
        """Get or create collector for camera."""
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
        """Record a capture attempt for a camera.
        
        Args:
            camera_id: Camera identifier
            duration_ms: Capture duration in milliseconds
            success: Whether capture succeeded
            error_type: Error type if failed
        """
        collector = self._get_collector(camera_id)
        collector.record(duration_ms, success, error_type)
    
    def get_summary(self, camera_id: int) -> StatsSummary:
        """Get statistics summary for a camera.
        
        Args:
            camera_id: Camera identifier
            
        Returns:
            StatsSummary for the camera
        """
        collector = self._get_collector(camera_id)
        return collector.get_summary()
    
    def get_all_summaries(self) -> dict[int, StatsSummary]:
        """Get summaries for all cameras.
        
        Returns:
            Dict mapping camera_id to StatsSummary
        """
        with self._lock:
            return {
                camera_id: collector.get_summary()
                for camera_id, collector in self._collectors.items()
            }
    
    def reset(self, camera_id: int | None = None) -> None:
        """Reset statistics.
        
        Args:
            camera_id: Camera to reset, or None for all cameras
        """
        with self._lock:
            if camera_id is not None:
                if camera_id in self._collectors:
                    self._collectors[camera_id].reset()
            else:
                for collector in self._collectors.values():
                    collector.reset()
    
    def to_dict(self) -> dict[str, Any]:
        """Export all statistics for serialization.
        
        Returns:
            Dict with all camera statistics
        """
        summaries = self.get_all_summaries()
        return {
            "cameras": {
                str(camera_id): summary.to_dict()
                for camera_id, summary in summaries.items()
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


def _percentile(sorted_data: list[float], p: float) -> float:
    """Calculate percentile from sorted data.
    
    Args:
        sorted_data: Sorted list of values
        p: Percentile (0-100)
        
    Returns:
        Percentile value
    """
    if not sorted_data:
        return 0.0
    
    k = (len(sorted_data) - 1) * (p / 100)
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_data) else f
    
    if f == c:
        return sorted_data[f]
    
    return sorted_data[f] * (c - k) + sorted_data[c] * (k - f)


# =============================================================================
# Global Stats Instance
# =============================================================================

_global_stats: CameraStats | None = None
_stats_lock = threading.Lock()


def get_camera_stats() -> CameraStats:
    """Get the global camera statistics instance.
    
    Returns:
        Global CameraStats instance (created on first call)
    """
    global _global_stats
    
    with _stats_lock:
        if _global_stats is None:
            _global_stats = CameraStats()
        return _global_stats
