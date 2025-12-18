"""Tests for the observability module (logging and statistics)."""

import io
import json
import logging
import threading
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from telescope_mcp.observability.logging import (
    JSONFormatter,
    LogContext,
    SessionLogHandler,
    StructuredFormatter,
    StructuredLogger,
    StructuredLogRecord,
    _format_value,
    _log_context,
    configure_logging,
    get_logger,
)
from telescope_mcp.observability.stats import (
    CameraStats,
    CameraStatsCollector,
    StatsSummary,
    _percentile,
)

# =============================================================================
# Structured Logging Tests
# =============================================================================


class TestStructuredLogRecord:
    """Tests for StructuredLogRecord."""

    def test_basic_creation(self):
        """Can create record with structured data."""
        record = StructuredLogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
            structured_data={"key": "value"},
        )
        assert record.structured_data == {"key": "value"}
        assert record.name == "test"
        assert record.levelno == logging.INFO

    def test_without_structured_data(self):
        """Record without structured_data has empty dict."""
        record = StructuredLogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )
        assert record.structured_data == {}

    def test_with_func_and_sinfo(self):
        """Can create record with optional func and sinfo."""
        record = StructuredLogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
            func="my_function",
            sinfo="stack info",
            structured_data={"camera_id": 0},
        )
        assert record.funcName == "my_function"
        assert record.stack_info == "stack info"
        assert record.structured_data == {"camera_id": 0}


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

    def test_custom_format_string(self):
        """Can use custom format string."""
        formatter = StructuredFormatter(fmt="%(levelname)s: %(message)s")
        record = logging.LogRecord(
            name="test",
            level=logging.WARNING,
            pathname="test.py",
            lineno=1,
            msg="Custom",
            args=(),
            exc_info=None,
        )
        record.structured_data = {}
        result = formatter.format(record)
        assert result.startswith("WARNING: Custom")

    def test_custom_date_format(self):
        """Can use custom date format."""
        formatter = StructuredFormatter(datefmt="%Y-%m-%d")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )
        record.structured_data = {}
        result = formatter.format(record)
        # Should contain date in YYYY-MM-DD format
        import re

        assert re.search(r"\d{4}-\d{2}-\d{2}", result)


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

    def test_json_with_exception(self):
        """JSON formatter includes exception info."""
        formatter = JSONFormatter()
        try:
            raise ValueError("Test error")
        except ValueError:
            import sys

            exc_info = sys.exc_info()
            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname="test.py",
                lineno=1,
                msg="Error occurred",
                args=(),
                exc_info=exc_info,
            )
            record.structured_data = {}

            result = formatter.format(record)
            parsed = json.loads(result)

            assert "exception" in parsed
            assert "ValueError" in parsed["exception"]
            assert "Test error" in parsed["exception"]

    def test_json_without_structured_data(self):
        """JSON formatter works without structured data attribute."""
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
        # No structured_data attribute
        result = formatter.format(record)
        parsed = json.loads(result)
        assert parsed["message"] == "Test"


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

    def test_context_exit_without_token(self):
        """Context handles exit when token is None."""
        ctx = LogContext(key="value")
        # Exit without entering (token is None)
        ctx.__exit__(None, None, None)
        # Should not raise


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

    def test_log_with_args_formatting(self, logger_and_stream):
        """Logger supports % formatting."""
        logger, stream = logger_and_stream
        logger.info("Captured %d frames", 5)
        output = stream.getvalue()
        assert "Captured 5 frames" in output

    def test_log_with_exc_info(self, logger_and_stream):
        """Logger captures exception info."""
        logger, stream = logger_and_stream
        try:
            raise RuntimeError("Test error")
        except RuntimeError:
            logger.error("Error occurred", exc_info=True)
        output = stream.getvalue()
        assert "Error occurred" in output
        assert "RuntimeError" in output

    def test_log_with_stack_info(self, logger_and_stream):
        """Logger captures stack info."""
        logger, stream = logger_and_stream
        logger.info("With stack", stack_info=True)
        output = stream.getvalue()
        assert "With stack" in output

    def test_log_with_extra_dict(self, logger_and_stream):
        """Logger handles extra dict."""
        logger, stream = logger_and_stream
        logger.info("Test", extra={"custom": "value"}, camera_id=0)
        output = stream.getvalue()
        assert "camera_id=0" in output

    def test_log_without_context(self, logger_and_stream):
        """Logger works without LogContext."""
        logger, stream = logger_and_stream
        # Clear any context
        _log_context.set({})
        logger.info("No context", key="value")
        output = stream.getvalue()
        assert "key=value" in output


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

    def test_handles_exception_in_emit(self):
        """Handler catches exceptions during emit."""
        mock_manager = MagicMock()
        mock_manager.log.side_effect = RuntimeError("Session error")
        handler = SessionLogHandler(lambda: mock_manager)

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )
        record.structured_data = {}

        # Should not raise (calls handleError internally)
        handler.emit(record)

    def test_handler_with_custom_level(self):
        """Handler respects custom level setting."""
        mock_manager = MagicMock()
        handler = SessionLogHandler(lambda: mock_manager, level=logging.WARNING)

        # Level should be set
        assert handler.level == logging.WARNING


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
            last_capture_time=datetime(2025, 12, 17, 20, 0, 0, tzinfo=UTC),
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

    def test_record_failure_without_error_type(self):
        """Failed capture without error type."""
        collector = CameraStatsCollector(camera_id=0)
        collector.record(duration_ms=0, success=False, error_type=None)

        summary = collector.get_summary()
        assert summary.total_captures == 1
        assert summary.failed_captures == 1
        assert summary.success_rate == 0.0
        assert summary.error_counts == {}  # No error type tracked

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

    def test_reset_nonexistent_camera(self):
        """Reset on non-existent camera is a no-op."""
        stats = CameraStats()
        stats.record_capture(camera_id=0, duration_ms=100.0, success=True)

        # Reset camera that doesn't exist - should not raise
        stats.reset(camera_id=99)

        # Camera 0 should be unaffected
        assert stats.get_summary(camera_id=0).total_captures == 1

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


# =============================================================================
# Configuration Tests
# =============================================================================


class TestConfigureLogging:
    """Tests for configure_logging function."""

    def test_configure_with_defaults(self):
        """Configure logging with default settings."""
        from telescope_mcp.observability import logging as log_module

        # Reset configured flag for testing
        log_module._configured = False

        stream = io.StringIO()
        configure_logging(stream=stream)

        # Should be marked as configured
        assert log_module._configured

        # Get a logger and test it works
        logger = logging.getLogger("telescope_mcp.test")
        logger.info("Test message")

        output = stream.getvalue()
        assert "Test message" in output

    def test_configure_with_json_format(self):
        """Configure with JSON formatter."""
        from telescope_mcp.observability import logging as log_module

        log_module._configured = False

        stream = io.StringIO()
        configure_logging(json_format=True, stream=stream)

        logger = logging.getLogger("telescope_mcp.test_json")
        logger.info("JSON test")

        output = stream.getvalue()
        parsed = json.loads(output)
        assert parsed["message"] == "JSON test"

    def test_configure_with_debug_level(self):
        """Configure with DEBUG level."""
        from telescope_mcp.observability import logging as log_module

        log_module._configured = False

        stream = io.StringIO()
        configure_logging(level=logging.DEBUG, stream=stream)

        logger = logging.getLogger("telescope_mcp.test_debug")
        logger.debug("Debug message")

        output = stream.getvalue()
        assert "Debug message" in output

    def test_configure_with_string_level(self):
        """Configure with string level name."""
        from telescope_mcp.observability import logging as log_module

        log_module._configured = False

        stream = io.StringIO()
        configure_logging(level="WARNING", stream=stream)

        logger = logging.getLogger("telescope_mcp.test_warning")
        logger.info("Info message")
        logger.warning("Warning message")

        output = stream.getvalue()
        assert "Info message" not in output
        assert "Warning message" in output

    def test_configure_without_structured(self):
        """Configure without structured data in output."""
        from telescope_mcp.observability import logging as log_module

        log_module._configured = False

        stream = io.StringIO()
        configure_logging(include_structured=False, stream=stream)

        logger = get_logger("telescope_mcp.test_nostruct")
        logger.info("Message", key="value")

        output = stream.getvalue()
        assert "Message" in output
        assert "key=value" not in output

    def test_configure_idempotent(self):
        """Calling configure_logging multiple times is safe."""
        from telescope_mcp.observability import logging as log_module

        log_module._configured = False

        stream1 = io.StringIO()
        configure_logging(stream=stream1)

        stream2 = io.StringIO()
        configure_logging(stream=stream2)  # Should have no effect

        logger = logging.getLogger("telescope_mcp.test_idempotent")
        logger.info("Test")

        # Only stream1 should have output (first config wins)
        assert "Test" in stream1.getvalue()
        assert stream2.getvalue() == ""


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_logger_returns_structured_logger(self):
        """get_logger returns StructuredLogger instance."""
        from telescope_mcp.observability import logging as log_module

        log_module._configured = False

        logger = get_logger("telescope_mcp.test_get")
        assert isinstance(logger, StructuredLogger)

    def test_get_logger_auto_configures(self):
        """get_logger auto-configures if not configured."""
        from telescope_mcp.observability import logging as log_module

        log_module._configured = False

        _logger = get_logger("telescope_mcp.test_auto")  # noqa: F841

        # Should be configured now
        assert log_module._configured

    def test_get_logger_with_name(self):
        """get_logger uses provided name."""
        logger = get_logger("telescope_mcp.custom.module")
        assert logger.name == "telescope_mcp.custom.module"

    def test_get_logger_when_already_configured(self):
        """get_logger works when already configured."""
        from telescope_mcp.observability import logging as log_module

        # Ensure configured
        if not log_module._configured:
            configure_logging()

        logger = get_logger("telescope_mcp.test_already")
        assert isinstance(logger, StructuredLogger)

    def test_get_logger_double_check_locking(self):
        """Test that double-checked locking works correctly.

        The inner check (line 676) is defensive for race conditions.
        We verify the overall behavior is correct.
        """
        import threading

        from telescope_mcp.observability import logging as log_module

        # Ensure we start configured
        if not log_module._configured:
            configure_logging()

        # Multiple threads calling get_logger should all succeed
        results = []

        def get_logger_thread(n):
            logger = get_logger(f"telescope_mcp.concurrent.{n}")
            results.append(logger.name)

        threads = [
            threading.Thread(target=get_logger_thread, args=(i,)) for i in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 10
        assert log_module._configured


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
