"""Tests for the observability module (logging and statistics)."""

import io
import json
import logging
import threading
import time
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from telescope_mcp.observability.logging import (
    JSONFormatter,
    LogContext,
    SessionLogHandler,
    StructuredFormatter,
    StructuredLogger,
    _format_value,
    _log_context,
    configure_logging,
    get_logger,
)
from telescope_mcp.observability.stats import (
    CameraStats,
    CameraStatsCollector,
    CaptureRecord,
    StatsSummary,
    _percentile,
    get_camera_stats,
)


# =============================================================================
# Structured Logging Tests
# =============================================================================


class TestStructuredFormatter:
    """Tests for StructuredFormatter."""

    def test_basic_format(self):
        """Basic log message formats correctly."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        result = formatter.format(record)
        assert "Test message" in result
        assert "INFO" in result

    def test_format_with_structured_data(self):
        """Structured data appended to message."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Camera connected",
            args=(),
            exc_info=None,
        )
        record.structured_data = {"camera_id": 0, "name": "ASI120"}
        result = formatter.format(record)
        assert "Camera connected" in result
        assert "camera_id=0" in result
        assert "name=ASI120" in result
        assert "|" in result  # Separator

    def test_format_without_structured_data(self):
        """No structured data means no separator."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Simple message",
            args=(),
            exc_info=None,
        )
        result = formatter.format(record)
        assert "Simple message" in result
        assert "|" not in result

    def test_include_structured_false(self):
        """Can disable structured data output."""
        formatter = StructuredFormatter(include_structured=False)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Message",
            args=(),
            exc_info=None,
        )
        record.structured_data = {"key": "value"}
        result = formatter.format(record)
        assert "key=value" not in result


class TestJSONFormatter:
    """Tests for JSONFormatter."""

    def test_basic_json_format(self):
        """Log formats as valid JSON."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test.module",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        result = formatter.format(record)
        parsed = json.loads(result)
        assert parsed["message"] == "Test message"
        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "test.module"
        assert "timestamp" in parsed

    def test_json_with_structured_data(self):
        """Structured data included in JSON."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.WARNING,
            pathname="test.py",
            lineno=1,
            msg="Warning",
            args=(),
            exc_info=None,
        )
        record.structured_data = {"camera_id": 1, "error": "timeout"}
        result = formatter.format(record)
        parsed = json.loads(result)
        assert parsed["camera_id"] == 1
        assert parsed["error"] == "timeout"
        assert parsed["level"] == "WARNING"

    def test_json_timestamp_is_iso(self):
        """Timestamp is ISO format."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )
        result = formatter.format(record)
        parsed = json.loads(result)
        # Should parse as ISO datetime
        datetime.fromisoformat(parsed["timestamp"])


class TestFormatValue:
    """Tests for _format_value helper."""

    def test_simple_string(self):
        """Simple strings pass through."""
        assert _format_value("hello") == "hello"

    def test_string_with_spaces(self):
        """Strings with spaces get quoted."""
        assert _format_value("hello world") == '"hello world"'

    def test_integer(self):
        """Integers convert to string."""
        assert _format_value(42) == "42"

    def test_float(self):
        """Floats convert to string."""
        assert _format_value(3.14) == "3.14"

    def test_dict(self):
        """Dicts serialize as JSON."""
        result = _format_value({"key": "value"})
        assert json.loads(result) == {"key": "value"}

    def test_list(self):
        """Lists serialize as JSON."""
        result = _format_value([1, 2, 3])
        assert json.loads(result) == [1, 2, 3]


class TestLogContext:
    """Tests for LogContext context manager."""

    def test_context_sets_values(self):
        """Context sets values in context var."""
        with LogContext(camera_id=0, operation="capture"):
            ctx = _log_context.get()
            assert ctx["camera_id"] == 0
            assert ctx["operation"] == "capture"

    def test_context_restores_on_exit(self):
        """Context restores previous values on exit."""
        _log_context.set({})
        with LogContext(camera_id=0):
            pass
        assert _log_context.get() == {}

    def test_nested_contexts(self):
        """Nested contexts merge correctly."""
        with LogContext(camera_id=0):
            with LogContext(operation="capture"):
                ctx = _log_context.get()
                assert ctx["camera_id"] == 0
                assert ctx["operation"] == "capture"
            # Inner context removed
            ctx = _log_context.get()
            assert ctx["camera_id"] == 0
            assert "operation" not in ctx

    def test_context_override(self):
        """Inner context can override outer values."""
        with LogContext(camera_id=0):
            with LogContext(camera_id=1):
                ctx = _log_context.get()
                assert ctx["camera_id"] == 1
            # Restored
            ctx = _log_context.get()
            assert ctx["camera_id"] == 0


class TestStructuredLogger:
    """Tests for StructuredLogger."""

    @pytest.fixture
    def logger_and_stream(self):
        """Create a logger with captured output."""
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(StructuredFormatter())
        
        logger = StructuredLogger("test_logger")
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)
        logger.propagate = False
        
        yield logger, stream
        
        # Cleanup
        logger.handlers.clear()

    def test_basic_logging(self, logger_and_stream):
        """Basic logging works."""
        logger, stream = logger_and_stream
        logger.info("Test message")
        output = stream.getvalue()
        assert "Test message" in output

    def test_structured_kwargs(self, logger_and_stream):
        """Keyword arguments become structured data."""
        logger, stream = logger_and_stream
        logger.info("Frame captured", camera_id=0, exposure_us=100000)
        output = stream.getvalue()
        assert "camera_id=0" in output
        assert "exposure_us=100000" in output

    def test_context_included(self, logger_and_stream):
        """LogContext values included in output."""
        logger, stream = logger_and_stream
        with LogContext(request_id="abc123"):
            logger.info("Processing")
        output = stream.getvalue()
        assert "request_id=abc123" in output

    def test_kwargs_override_context(self, logger_and_stream):
        """Kwargs override context values."""
        logger, stream = logger_and_stream
        with LogContext(camera_id=0):
            logger.info("Changed", camera_id=1)
        output = stream.getvalue()
        assert "camera_id=1" in output


class TestSessionLogHandler:
    """Tests for SessionLogHandler."""

    def test_forwards_to_session_manager(self):
        """Handler forwards logs to session manager."""
        mock_manager = MagicMock()
        handler = SessionLogHandler(lambda: mock_manager)
        
        record = logging.LogRecord(
            name="test.module",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.structured_data = {"key": "value"}
        
        handler.emit(record)
        
        mock_manager.log.assert_called_once()
        call_kwargs = mock_manager.log.call_args[1]
        assert call_kwargs["level"] == "INFO"
        assert call_kwargs["message"] == "Test message"
        assert call_kwargs["source"] == "test.module"
        assert call_kwargs["key"] == "value"

    def test_handles_none_manager(self):
        """Handler handles None manager gracefully."""
        handler = SessionLogHandler(lambda: None)
        
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )
        
        # Should not raise
        handler.emit(record)


# =============================================================================
# Camera Statistics Tests
# =============================================================================


class TestStatsSummary:
    """Tests for StatsSummary dataclass."""

    def test_default_values(self):
        """Default values are sensible."""
        summary = StatsSummary(camera_id=0)
        assert summary.camera_id == 0
        assert summary.total_captures == 0
        assert summary.success_rate == 0.0

    def test_to_dict(self):
        """to_dict produces serializable output."""
        summary = StatsSummary(
            camera_id=0,
            total_captures=10,
            successful_captures=9,
            failed_captures=1,
            success_rate=0.9,
            avg_duration_ms=150.0,
            error_counts={"timeout": 1},
            last_capture_time=datetime(2025, 12, 17, 20, 0, 0, tzinfo=timezone.utc),
        )
        result = summary.to_dict()
        assert result["camera_id"] == 0
        assert result["success_rate"] == 0.9
        assert result["error_counts"] == {"timeout": 1}
        assert "2025-12-17" in result["last_capture_time"]

    def test_to_dict_none_timestamp(self):
        """to_dict handles None timestamp."""
        summary = StatsSummary(camera_id=0, last_capture_time=None)
        result = summary.to_dict()
        assert result["last_capture_time"] is None


class TestCameraStatsCollector:
    """Tests for CameraStatsCollector."""

    def test_initial_state(self):
        """New collector has zero stats."""
        collector = CameraStatsCollector(camera_id=0)
        summary = collector.get_summary()
        assert summary.total_captures == 0
        assert summary.success_rate == 0.0

    def test_record_success(self):
        """Successful capture recorded correctly."""
        collector = CameraStatsCollector(camera_id=0)
        collector.record(duration_ms=150.0, success=True)
        
        summary = collector.get_summary()
        assert summary.total_captures == 1
        assert summary.successful_captures == 1
        assert summary.success_rate == 1.0

    def test_record_failure(self):
        """Failed capture recorded with error type."""
        collector = CameraStatsCollector(camera_id=0)
        collector.record(duration_ms=0, success=False, error_type="timeout")
        
        summary = collector.get_summary()
        assert summary.total_captures == 1
        assert summary.failed_captures == 1
        assert summary.success_rate == 0.0
        assert summary.error_counts["timeout"] == 1

    def test_duration_statistics(self):
        """Duration stats calculated correctly."""
        collector = CameraStatsCollector(camera_id=0)
        collector.record(duration_ms=100.0, success=True)
        collector.record(duration_ms=200.0, success=True)
        collector.record(duration_ms=150.0, success=True)
        
        summary = collector.get_summary()
        assert summary.min_duration_ms == 100.0
        assert summary.max_duration_ms == 200.0
        assert summary.avg_duration_ms == 150.0

    def test_p95_duration(self):
        """P95 duration calculated correctly."""
        collector = CameraStatsCollector(camera_id=0)
        # Record 100 captures with durations 1-100
        for i in range(1, 101):
            collector.record(duration_ms=float(i), success=True)
        
        summary = collector.get_summary()
        # P95 should be around 95
        assert 94 <= summary.p95_duration_ms <= 96

    def test_reset(self):
        """Reset clears all statistics."""
        collector = CameraStatsCollector(camera_id=0)
        collector.record(duration_ms=150.0, success=True)
        collector.reset()
        
        summary = collector.get_summary()
        assert summary.total_captures == 0

    def test_window_size(self):
        """Window size limits stored records."""
        collector = CameraStatsCollector(camera_id=0, window_size=10)
        for i in range(20):
            collector.record(duration_ms=float(i), success=True)
        
        # Should have recorded all 20
        summary = collector.get_summary()
        assert summary.total_captures == 20
        
        # But duration stats from last 10 only (10-19)
        assert summary.min_duration_ms == 10.0
        assert summary.max_duration_ms == 19.0

    def test_thread_safety(self):
        """Collector is thread-safe."""
        collector = CameraStatsCollector(camera_id=0)
        
        def record_captures():
            for _ in range(100):
                collector.record(duration_ms=100.0, success=True)
        
        threads = [threading.Thread(target=record_captures) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        summary = collector.get_summary()
        assert summary.total_captures == 1000


class TestCameraStats:
    """Tests for CameraStats manager."""

    def test_record_creates_collector(self):
        """Recording creates collector for camera."""
        stats = CameraStats()
        stats.record_capture(camera_id=0, duration_ms=150.0, success=True)
        
        summary = stats.get_summary(camera_id=0)
        assert summary.total_captures == 1

    def test_multiple_cameras(self):
        """Stats tracked per camera."""
        stats = CameraStats()
        stats.record_capture(camera_id=0, duration_ms=100.0, success=True)
        stats.record_capture(camera_id=1, duration_ms=200.0, success=True)
        
        summary0 = stats.get_summary(camera_id=0)
        summary1 = stats.get_summary(camera_id=1)
        
        assert summary0.avg_duration_ms == 100.0
        assert summary1.avg_duration_ms == 200.0

    def test_get_all_summaries(self):
        """Can get all camera summaries."""
        stats = CameraStats()
        stats.record_capture(camera_id=0, duration_ms=100.0, success=True)
        stats.record_capture(camera_id=1, duration_ms=200.0, success=True)
        
        all_summaries = stats.get_all_summaries()
        assert 0 in all_summaries
        assert 1 in all_summaries

    def test_reset_single_camera(self):
        """Can reset single camera."""
        stats = CameraStats()
        stats.record_capture(camera_id=0, duration_ms=100.0, success=True)
        stats.record_capture(camera_id=1, duration_ms=200.0, success=True)
        
        stats.reset(camera_id=0)
        
        assert stats.get_summary(camera_id=0).total_captures == 0
        assert stats.get_summary(camera_id=1).total_captures == 1

    def test_reset_all_cameras(self):
        """Can reset all cameras."""
        stats = CameraStats()
        stats.record_capture(camera_id=0, duration_ms=100.0, success=True)
        stats.record_capture(camera_id=1, duration_ms=200.0, success=True)
        
        stats.reset()
        
        assert stats.get_summary(camera_id=0).total_captures == 0
        assert stats.get_summary(camera_id=1).total_captures == 0

    def test_to_dict(self):
        """to_dict exports all stats."""
        stats = CameraStats()
        stats.record_capture(camera_id=0, duration_ms=100.0, success=True)
        
        result = stats.to_dict()
        assert "cameras" in result
        assert "0" in result["cameras"]
        assert "timestamp" in result


class TestPercentile:
    """Tests for _percentile helper."""

    def test_empty_list(self):
        """Empty list returns 0."""
        assert _percentile([], 50) == 0.0

    def test_single_value(self):
        """Single value returns that value."""
        assert _percentile([100.0], 50) == 100.0

    def test_median(self):
        """P50 is median."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert _percentile(data, 50) == 3.0

    def test_p0(self):
        """P0 is minimum."""
        data = [1.0, 2.0, 3.0]
        assert _percentile(data, 0) == 1.0

    def test_p100(self):
        """P100 is maximum."""
        data = [1.0, 2.0, 3.0]
        assert _percentile(data, 100) == 3.0

    def test_p95(self):
        """P95 interpolates correctly."""
        # 100 values from 1 to 100
        data = list(range(1, 101))
        data = [float(x) for x in data]
        result = _percentile(data, 95)
        assert 94 <= result <= 96


class TestGetCameraStats:
    """Tests for get_camera_stats singleton."""

    def test_returns_same_instance(self):
        """Returns same instance on multiple calls."""
        stats1 = get_camera_stats()
        stats2 = get_camera_stats()
        assert stats1 is stats2

    def test_is_camera_stats(self):
        """Returns CameraStats instance."""
        stats = get_camera_stats()
        assert isinstance(stats, CameraStats)


# =============================================================================
# Integration Tests
# =============================================================================


class TestLoggingIntegration:
    """Integration tests for logging system."""

    def test_json_logging_end_to_end(self):
        """JSON logging produces parseable output."""
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(JSONFormatter())
        
        logger = StructuredLogger("integration_test")
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        logger.propagate = False
        
        try:
            with LogContext(session_id="test123"):
                logger.info("Frame captured", camera_id=0, duration_ms=150.5)
            
            output = stream.getvalue()
            parsed = json.loads(output)
            
            assert parsed["message"] == "Frame captured"
            assert parsed["session_id"] == "test123"
            assert parsed["camera_id"] == 0
            assert parsed["duration_ms"] == 150.5
        finally:
            logger.handlers.clear()


class TestStatsIntegration:
    """Integration tests for statistics system."""

    def test_realistic_capture_session(self):
        """Simulate realistic capture session."""
        stats = CameraStats()
        
        # Simulate 100 captures with realistic timings
        import random
        random.seed(42)  # Reproducible
        
        for _ in range(100):
            duration = random.gauss(150, 20)  # Mean 150ms, stddev 20ms
            success = random.random() > 0.05  # 95% success rate
            error = None if success else "random_error"
            stats.record_capture(
                camera_id=0,
                duration_ms=duration if success else 0,
                success=success,
                error_type=error,
            )
        
        summary = stats.get_summary(camera_id=0)
        
        # Should have reasonable stats
        assert 90 <= summary.total_captures <= 100
        assert 0.90 <= summary.success_rate <= 1.0
        assert 100 <= summary.avg_duration_ms <= 200
        assert summary.p95_duration_ms > summary.avg_duration_ms
