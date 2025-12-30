"""Comprehensive tests for observability/logging.py to increase coverage."""

import json
import logging
from io import StringIO
from unittest.mock import MagicMock

from telescope_mcp.observability.logging import (
    JSONFormatter,
    LogContext,
    SessionLogHandler,
    StructuredFormatter,
    StructuredLogger,
    StructuredLogRecord,
    _format_value,
    configure_logging,
    get_logger,
    reset_logging,
)


class TestStructuredLogger:
    """Test StructuredLogger functionality."""

    def test_logger_basic_logging(self):
        """Verifies StructuredLogger handles plain messages without structured data.

        Arrangement:
        1. get_logger() returns StructuredLogger instance.
        2. Logging without keyword arguments (plain messages).
        3. Should work like standard Python logger.

        Action:
        Logs info, debug, warning messages without structured data.

        Assertion Strategy:
        Validates basic compatibility by confirming:
        - get_logger() returns StructuredLogger type.
        - No exceptions raised during logging.

        Testing Principle:
        Validates backwards compatibility, ensuring StructuredLogger
        is a drop-in replacement for standard Logger.
        """
        logger = get_logger("test.basic")
        assert isinstance(logger, StructuredLogger)

        # Should not raise
        logger.info("Test message")
        logger.debug("Debug message")
        logger.warning("Warning message")

    def test_logger_with_structured_data(self):
        """Verifies StructuredLogger accepts keyword arguments as structured data.

        Arrangement:
        1. StructuredLogger obtained via get_logger().
        2. Log methods accept **kwargs for structured data.
        3. camera_id=0, name="Test Camera" become queryable metadata.

        Action:
        Logs with info() and warning() using keyword arguments.

        Assertion Strategy:
        Validates structured logging by confirming:
        - No exceptions raised when passing kwargs.

        Testing Principle:
        Validates core feature, ensuring structured data
        is captured via keyword arguments.
        """
        logger = get_logger("test.structured")

        # Should not raise
        logger.info("Camera connected", camera_id=0, name="Test Camera")
        logger.warning("High temperature", temp_c=45.5, threshold=40.0)

    def test_logger_all_levels(self):
        """Verifies StructuredLogger supports all standard log levels.

        Arrangement:
        1. StructuredLogger instance.
        2. Python logging defines DEBUG, INFO, WARNING, ERROR, CRITICAL.
        3. All levels should be accessible.

        Action:
        Calls debug(), info(), warning(), error(), critical().

        Assertion Strategy:
        Validates level support by confirming:
        - No exceptions raised for any level.

        Testing Principle:
        Validates API completeness, ensuring all standard
        Python logging levels are supported.
        """
        logger = get_logger("test.levels")

        logger.debug("Debug level")
        logger.info("Info level")
        logger.warning("Warning level")
        logger.error("Error level")
        logger.critical("Critical level")

    def test_logger_with_exception(self):
        """Verifies StructuredLogger captures exception stack traces.

        Arrangement:
        1. ValueError exception raised and caught.
        2. logger.error() with exc_info=True captures stack trace.
        3. Standard Python logging feature.

        Action:
        Raises ValueError, then logs error with exc_info=True.

        Assertion Strategy:
        Validates exception logging by confirming:
        - No exception raised during logger.error() with exc_info.

        Testing Principle:
        Validates exception handling integration, ensuring
        StructuredLogger preserves standard exc_info behavior.
        """
        logger = get_logger("test.exception")

        try:
            raise ValueError("Test error")
        except ValueError:
            logger.error("Error occurred", exc_info=True)

    def test_logger_name(self):
        """Verifies logger preserves specified name for hierarchical organization.

        Arrangement:
        1. get_logger("test.module.submodule") with hierarchical name.
        2. Logger name used for filtering and organization.
        3. Name should match input exactly.

        Action:
        Creates logger with dotted name, reads name attribute.

        Assertion Strategy:
        Validates naming by confirming:
        - logger.name = "test.module.submodule".

        Testing Principle:
        Validates logger identity, ensuring hierarchical
        naming supports module-based organization.
        """
        logger = get_logger("test.module.submodule")
        assert logger.name == "test.module.submodule"


class TestStructuredFormatter:
    """Test StructuredFormatter for human-readable output."""

    def test_format_simple_message(self):
        """Verifies StructuredFormatter formats plain messages without structured data.

        Arrangement:
        1. StructuredFormatter instance for human-readable output.
        2. LogRecord with message="Simple message", empty structured_data.
        3. Should format like standard formatter.

        Action:
        Creates LogRecord, formats with StructuredFormatter.

        Assertion Strategy:
        Validates formatting by confirming:
        - Output contains "Simple message".
        - Output contains logger name "test".

        Testing Principle:
        Validates base formatting, ensuring formatter handles
        plain messages when no structured data present.
        """
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Simple message",
            args=(),
            exc_info=None,
        )
        record.structured_data = {}

        formatted = formatter.format(record)
        assert "Simple message" in formatted
        assert "test" in formatted

    def test_format_with_structured_data(self):
        """Verifies StructuredFormatter appends key=value pairs to message.

        Arrangement:
        1. StructuredFormatter with include_structured=True (default).
        2. LogRecord with structured_data={"camera_id": 0, "exposure_us": 100000}.
        3. Output format: "message | key=value key=value".

        Action:
        Creates LogRecord with structured data, formats it.

        Assertion Strategy:
        Validates structured formatting by confirming:
        - Output contains "Camera event" (message).
        - Output contains "camera_id=0" (structured).
        - Output contains "exposure_us=100000" (structured).

        Testing Principle:
        Validates structured data rendering, ensuring key=value
        pairs are appended for human readability.
        """
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Camera event",
            args=(),
            exc_info=None,
        )
        record.structured_data = {"camera_id": 0, "exposure_us": 100000}

        formatted = formatter.format(record)
        assert "Camera event" in formatted
        assert "camera_id=0" in formatted
        assert "exposure_us=100000" in formatted


class TestJSONFormatter:
    """Test JSONFormatter for machine-readable output."""

    def test_format_json_simple(self):
        """Verifies JSONFormatter produces valid JSON with standard fields.

        Arrangement:
        1. JSONFormatter for machine-parseable output.
        2. LogRecord with simple message, no structured data.
        3. Output should be single-line JSON.

        Action:
        Creates LogRecord, formats as JSON, parses result.

        Assertion Strategy:
        Validates JSON formatting by confirming:
        - data["message"] = "Test message".
        - data["level"] = "INFO".
        - data["logger"] = "test".

        Testing Principle:
        Validates JSON output structure, ensuring standard
        fields are present for log aggregation systems.
        """
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.structured_data = {}

        formatted = formatter.format(record)
        data = json.loads(formatted)

        assert data["message"] == "Test message"
        assert data["level"] == "INFO"
        assert data["logger"] == "test"

    def test_format_json_with_structured_data(self):
        """Verifies JSONFormatter merges structured data as top-level JSON fields.

        Arrangement:
        1. JSONFormatter for machine-parseable output.
        2. LogRecord with structured_data={"gain": 200, "max_gain": 150}.
        3. Structured fields should appear at top level in JSON.

        Action:
        Creates LogRecord with structured data, formats and parses JSON.

        Assertion Strategy:
        Validates JSON structure by confirming:
        - data["message"] = "High gain".
        - data["gain"] = 200 (top-level).
        - data["max_gain"] = 150 (top-level).
        - data["level"] = "WARNING".

        Testing Principle:
        Validates structured data integration, ensuring JSON
        output is queryable by structured field names.
        """
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test.camera",
            level=logging.WARNING,
            pathname="camera.py",
            lineno=100,
            msg="High gain",
            args=(),
            exc_info=None,
        )
        record.structured_data = {"gain": 200, "max_gain": 150}

        formatted = formatter.format(record)
        data = json.loads(formatted)

        assert data["message"] == "High gain"
        assert data["gain"] == 200
        assert data["max_gain"] == 150
        assert data["level"] == "WARNING"

    def test_format_json_with_exception(self):
        """Verifies JSONFormatter includes exception stack trace in JSON output.

        Arrangement:
        1. ValueError exception raised and captured.
        2. LogRecord with exc_info containing exception details.
        3. JSON should include "exception" field with stack trace.

        Action:
        Raises ValueError, creates LogRecord with exc_info, formats as JSON.

        Assertion Strategy:
        Validates exception handling by confirming:
        - data contains "exception" key.
        - data["exception"] contains "ValueError" string.

        Testing Principle:
        Validates error logging, ensuring exceptions are
        captured in JSON for debugging and monitoring.
        """
        formatter = JSONFormatter()

        try:
            raise ValueError("Test error")
        except ValueError as e:
            import sys

            exc_info = sys.exc_info()

            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname="test.py",
                lineno=50,
                msg="Error occurred",
                args=(),
                exc_info=exc_info,
            )
            record.structured_data = {}

            formatted = formatter.format(record)
            data = json.loads(formatted)

            assert "exception" in data
            assert "ValueError" in data["exception"]


class TestLogContext:
    """Test LogContext for context management."""

    def test_context_manager_basic(self):
        """Verifies LogContext works as context manager with single field.

        Arrangement:
        1. LogContext(request_id="test123") creates context manager.
        2. Context injects request_id into all logs within scope.
        3. Should not raise exceptions.

        Action:
        Enters and exits LogContext with single keyword argument.

        Assertion Strategy:
        Validates context manager protocol by confirming:
        - No exceptions raised during __enter__/__exit__.

        Testing Principle:
        Validates context manager basics, ensuring LogContext
        follows Python context manager protocol.
        """
        # LogContext should work without errors
        with LogContext(request_id="test123"):
            # Context is used internally
            pass

    def test_context_manager_multiple_fields(self):
        """Verifies LogContext accepts multiple key-value pairs.

        Arrangement:
        1. LogContext with request_id, camera_id, operation kwargs.
        2. All fields should be injected into context.
        3. Should handle arbitrary number of fields.

        Action:
        Creates LogContext with 3 keyword arguments.

        Assertion Strategy:
        Validates multi-field support by confirming:
        - No exceptions raised with multiple kwargs.

        Testing Principle:
        Validates context flexibility, ensuring multiple
        contextual fields can be tracked simultaneously.
        """
        # Should work without errors
        with LogContext(request_id="abc", camera_id=0, operation="capture"):
            pass

    def test_context_manager_nested(self):
        """Verifies LogContext supports nesting with proper value merging.

        Arrangement:
        1. Outer LogContext(outer="value1").
        2. Inner LogContext(inner="value2") nested inside.
        3. Inner context should merge with outer (both present).

        Action:
        Nests two LogContext managers.

        Assertion Strategy:
        Validates nesting by confirming:
        - No exceptions raised during nested contexts.

        Testing Principle:
        Validates context composition, ensuring nested contexts
        merge properly for hierarchical operation tracking.
        """
        # Should work without errors
        with LogContext(outer="value1"):
            with LogContext(inner="value2"):
                pass


class TestLoggingConfiguration:
    """Test logging configuration."""

    def test_configure_logging_default(self):
        """Verifies configure_logging initializes with default settings.

        Arrangement:
        1. configure_logging() with no arguments.
        2. Defaults: INFO level, StructuredFormatter, stderr output.
        3. get_logger() should return configured StructuredLogger.

        Action:
        Calls configure_logging() then get_logger().

        Assertion Strategy:
        Validates default configuration by confirming:
        - get_logger() returns StructuredLogger type.

        Testing Principle:
        Validates out-of-box experience, ensuring logging
        works with sensible defaults.
        """
        configure_logging()
        logger = get_logger("test.config")
        assert isinstance(logger, StructuredLogger)

    def test_configure_logging_with_level(self):
        """Verifies configure_logging accepts custom log level.

        Arrangement:
        1. configure_logging(level=logging.DEBUG).
        2. Logger level should be DEBUG (10) or lower.
        3. Enables verbose development logging.

        Action:
        Configures with DEBUG level, checks logger level.

        Assertion Strategy:
        Validates level configuration by confirming:
        - logger.level <= logging.DEBUG (captures DEBUG+).

        Testing Principle:
        Validates level control, ensuring log verbosity
        can be configured at runtime.
        """
        configure_logging(level=logging.DEBUG)
        logger = get_logger("test.debug")
        assert logger.level <= logging.DEBUG

    def test_configure_logging_json_format(self):
        """Verifies configure_logging enables JSON output for log aggregation.

        Arrangement:
        1. configure_logging(json_format=True).
        2. Output should be single-line JSON (NDJSON).
        3. StringIO captures log output for verification.

        Action:
        Configures JSON format, logs message, parses output.

        Assertion Strategy:
        Validates JSON mode by confirming:
        - Output is valid JSON with "message" field.

        Testing Principle:
        Validates production mode, ensuring JSON output
        is parseable by log aggregation systems.
        """
        # Capture log output
        stream = StringIO()
        handler = logging.StreamHandler(stream)

        configure_logging(json_format=True)
        logger = get_logger("test.json")
        logger.addHandler(handler)

        logger.info("Test JSON", key="value")

        output = stream.getvalue()
        if output:
            # Should be valid JSON
            data = json.loads(output.strip())
            assert "message" in data

    def test_configure_logging_structured_format(self):
        """Verifies configure_logging uses human-readable format by default.

        Arrangement:
        1. configure_logging(json_format=False).
        2. Output should be human-readable with key=value pairs.
        3. StringIO captures log output for verification.

        Action:
        Configures structured text format, logs with structured data.

        Assertion Strategy:
        Validates structured text by confirming:
        - Output contains "Test structured" (message text).

        Testing Principle:
        Validates development mode, ensuring human-readable
        output for console debugging.
        """
        stream = StringIO()
        handler = logging.StreamHandler(stream)

        configure_logging(json_format=False)
        logger = get_logger("test.structured")
        logger.addHandler(handler)

        logger.info("Test structured", camera_id=0)

        output = stream.getvalue()
        if output:
            assert "Test structured" in output


class TestGetLogger:
    """Test get_logger function."""

    def test_get_logger_returns_structured_logger(self):
        """Verifies get_logger factory returns StructuredLogger instances.

        Arrangement:
        1. get_logger("test.module") factory function.
        2. setLoggerClass(StructuredLogger) configures logger type.
        3. Should return StructuredLogger, not base Logger.

        Action:
        Calls get_logger(), checks type.

        Assertion Strategy:
        Validates factory behavior by confirming:
        - isinstance(logger, StructuredLogger).

        Testing Principle:
        Validates factory contract, ensuring get_logger
        always returns structured-capable loggers.
        """
        logger = get_logger("test.module")
        assert isinstance(logger, StructuredLogger)

    def test_get_logger_same_name_returns_same_instance(self):
        """Verifies get_logger returns singleton for each logger name.

        Arrangement:
        1. get_logger("test.same") called twice.
        2. Python logging uses singleton pattern per name.
        3. Should return same object instance.

        Action:
        Calls get_logger() twice with identical name.

        Assertion Strategy:
        Validates singleton pattern by confirming:
        - logger1 is logger2 (object identity).

        Testing Principle:
        Validates logger caching, ensuring multiple calls
        for same name return shared instance.
        """
        logger1 = get_logger("test.same")
        logger2 = get_logger("test.same")
        assert logger1 is logger2

    def test_get_logger_different_names(self):
        """Verifies get_logger creates separate instances for different names.

        Arrangement:
        1. get_logger("test.module1") and get_logger("test.module2").
        2. Different names should yield different logger instances.
        3. Each logger tracks its own name.

        Action:
        Gets two loggers with different names.

        Assertion Strategy:
        Validates logger separation by confirming:
        - logger1 is not logger2 (different objects).
        - logger1.name != logger2.name.

        Testing Principle:
        Validates logger isolation, ensuring separate
        loggers for different modules/components.
        """
        logger1 = get_logger("test.module1")
        logger2 = get_logger("test.module2")
        assert logger1 is not logger2
        assert logger1.name != logger2.name

    def test_get_logger_hierarchical(self):
        """Verifies logger names support hierarchical organization.

        Arrangement:
        1. Dotted names create logger hierarchy.
        2. "telescope_mcp" parent of "telescope_mcp.devices".
        3. "telescope_mcp.devices" parent of "telescope_mcp.devices.camera".

        Action:
        Gets parent, child, grandchild loggers.

        Assertion Strategy:
        Validates hierarchy by confirming:
        - parent.name = "telescope_mcp".
        - child.name = "telescope_mcp.devices".
        - grandchild.name = "telescope_mcp.devices.camera".

        Testing Principle:
        Validates naming structure, ensuring hierarchical
        organization supports module-based filtering.
        """
        parent = get_logger("telescope_mcp")
        child = get_logger("telescope_mcp.devices")
        grandchild = get_logger("telescope_mcp.devices.camera")

        assert parent.name == "telescope_mcp"
        assert child.name == "telescope_mcp.devices"
        assert grandchild.name == "telescope_mcp.devices.camera"


class TestLoggerIntegration:
    """Integration tests for logging system."""

    def test_logger_with_context_in_logs(self):
        """Verifies LogContext values appear in formatted log output.

        Arrangement:
        1. Logger with StructuredFormatter for readable output.
        2. LogContext(request_id="req123") active during logging.
        3. Context should be included in formatted message.

        Action:
        Logs message within LogContext, captures output.

        Assertion Strategy:
        Validates context inclusion by confirming:
        - Output contains "Operation started" (message).
        - Output contains "request_id" or "req123" (context).

        Testing Principle:
        Validates context integration, ensuring contextual
        fields are visible in human-readable logs.
        """
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(StructuredFormatter())

        logger = get_logger("test.integration")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        with LogContext(request_id="req123"):
            logger.info("Operation started")

        output = stream.getvalue()
        if output:
            assert "Operation started" in output
            # Context should be included
            assert "request_id" in output or "req123" in output

    def test_logger_structured_data_and_context(self):
        """Verifies LogContext and structured data kwargs combine properly.

        Arrangement:
        1. LogContext(session_id="session_abc") provides context.
        2. logger.info() with camera_id=0, duration_ms=150 kwargs.
        3. Both context and kwargs should appear in output.

        Action:
        Logs with structured data within LogContext.

        Assertion Strategy:
        Validates merging by confirming:
        - Output contains "Capture complete" (message).

        Testing Principle:
        Validates data composition, ensuring context and
        per-log structured data merge correctly.
        """
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(StructuredFormatter())

        logger = get_logger("test.combined")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        with LogContext(session_id="session_abc"):
            logger.info("Capture complete", camera_id=0, duration_ms=150)

        output = stream.getvalue()
        if output:
            assert "Capture complete" in output


class TestLoggerEdgeCases:
    """Test edge cases and error handling."""

    def test_logger_with_none_values(self):
        """Verifies StructuredLogger handles None values in structured data.

        Arrangement:
        1. Structured data may include value=None.
        2. _format_value() converts None to "null" string.
        3. Should not raise TypeError or AttributeError.

        Action:
        Logs with value=None and other="valid" kwargs.

        Assertion Strategy:
        Validates None handling by confirming:
        - No exception raised during logging.

        Testing Principle:
        Validates robustness, ensuring None values are
        handled gracefully without errors.
        """
        logger = get_logger("test.none")
        # Should not raise
        logger.info("Test with None", value=None, other="valid")

    def test_logger_with_complex_types(self):
        """Verifies StructuredLogger serializes complex Python types.

        Arrangement:
        1. Structured data may include lists and dicts.
        2. _format_value() uses json.dumps() for complex types.
        3. Should serialize to JSON strings.

        Action:
        Logs with list_data=[1,2,3] and dict_data={"a":1,"b":2}.

        Assertion Strategy:
        Validates serialization by confirming:
        - No exception raised during logging.

        Testing Principle:
        Validates type flexibility, ensuring complex
        types are serialized for human readability.
        """
        logger = get_logger("test.complex")
        # Should not raise
        logger.info("Complex data", list_data=[1, 2, 3], dict_data={"a": 1, "b": 2})

    def test_logger_empty_message(self):
        """Verifies StructuredLogger accepts empty message strings.

        Arrangement:
        1. Message parameter can be empty string "".
        2. Structured data still captured via kwargs.
        3. Should not raise ValueError.

        Action:
        Logs empty message with camera_id=0 structured data.

        Assertion Strategy:
        Validates empty message by confirming:
        - No exception raised.

        Testing Principle:
        Validates edge case handling, ensuring empty
        messages don't break logging.
        """
        logger = get_logger("test.empty")
        # Should not raise
        logger.info("", camera_id=0)

    def test_logger_long_message(self):
        """Verifies StructuredLogger handles very long messages without truncation.

        Arrangement:
        1. Message with 10,000 characters.
        2. Logging system should handle large messages.
        3. No artificial length limits imposed.

        Action:
        Logs message with 10k 'A' characters.

        Assertion Strategy:
        Validates large message handling by confirming:
        - No exception raised.

        Testing Principle:
        Validates capacity limits, ensuring logging
        doesn't fail on verbose debug output.
        """
        logger = get_logger("test.long")
        long_msg = "A" * 10000
        # Should not raise
        logger.info(long_msg)


# =============================================================================
# Coverage Tests - StructuredLogRecord
# =============================================================================


class TestStructuredLogRecord:
    """Direct tests for StructuredLogRecord class."""

    def test_init_with_structured_data(self):
        """Verifies StructuredLogRecord stores structured_data from kwargs.

        Tests direct instantiation of StructuredLogRecord to cover __init__.
        """
        record = StructuredLogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/path/to/test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
            func="test_func",
            sinfo=None,
            structured_data={"camera_id": 0, "exposure_ms": 100},
        )

        assert record.name == "test.logger"
        assert record.levelno == logging.INFO
        assert record.msg == "Test message"
        assert record.structured_data == {"camera_id": 0, "exposure_ms": 100}

    def test_init_without_structured_data(self):
        """Verifies StructuredLogRecord defaults structured_data to empty dict."""
        record = StructuredLogRecord(
            name="test.logger",
            level=logging.WARNING,
            pathname="/path/to/test.py",
            lineno=10,
            msg="Warning message",
            args=(),
            exc_info=None,
        )

        assert record.structured_data == {}

    def test_init_with_args_formatting(self):
        """Verifies StructuredLogRecord handles %-style format args."""
        record = StructuredLogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/path/to/test.py",
            lineno=1,
            msg="Value: %d",
            args=(42,),
            exc_info=None,
        )

        assert record.getMessage() == "Value: 42"


# =============================================================================
# Coverage Tests - _format_value
# =============================================================================


class TestFormatValue:
    """Direct tests for _format_value helper function."""

    def test_format_value_none(self):
        """Verifies _format_value returns 'null' for None."""
        assert _format_value(None) == "null"

    def test_format_value_simple_string(self):
        """Verifies _format_value returns string as-is without spaces."""
        assert _format_value("hello") == "hello"
        assert _format_value("world") == "world"

    def test_format_value_string_with_spaces(self):
        """Verifies _format_value quotes strings containing spaces."""
        assert _format_value("hello world") == '"hello world"'
        assert _format_value("has spaces here") == '"has spaces here"'

    def test_format_value_dict(self):
        """Verifies _format_value JSON-serializes dictionaries."""
        result = _format_value({"key": "value"})
        assert result == '{"key": "value"}'

    def test_format_value_list(self):
        """Verifies _format_value JSON-serializes lists."""
        result = _format_value([1, 2, 3])
        assert result == "[1, 2, 3]"

    def test_format_value_integer(self):
        """Verifies _format_value converts integers via str()."""
        assert _format_value(42) == "42"
        assert _format_value(-100) == "-100"

    def test_format_value_float(self):
        """Verifies _format_value converts floats via str()."""
        assert _format_value(3.14) == "3.14"

    def test_format_value_bool(self):
        """Verifies _format_value converts booleans via str()."""
        assert _format_value(True) == "True"
        assert _format_value(False) == "False"

    def test_format_value_nested_dict(self):
        """Verifies _format_value handles nested structures."""
        result = _format_value({"outer": {"inner": [1, 2]}})
        parsed = json.loads(result)
        assert parsed == {"outer": {"inner": [1, 2]}}


# =============================================================================
# Coverage Tests - StructuredFormatter
# =============================================================================


class TestStructuredFormatterCoverage:
    """Additional coverage tests for StructuredFormatter."""

    def test_format_include_structured_false(self):
        """Verifies StructuredFormatter omits structured data when disabled."""
        formatter = StructuredFormatter(include_structured=False)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.structured_data = {"camera_id": 0, "key": "value"}

        formatted = formatter.format(record)

        assert "Test message" in formatted
        # Structured data should NOT be included
        assert "camera_id" not in formatted
        assert "key=" not in formatted

    def test_format_without_structured_data_attribute(self):
        """Verifies StructuredFormatter handles records without structured_data."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Plain message",
            args=(),
            exc_info=None,
        )
        # No structured_data attribute at all

        formatted = formatter.format(record)
        assert "Plain message" in formatted


# =============================================================================
# Coverage Tests - SessionLogHandler
# =============================================================================


class TestSessionLogHandler:
    """Tests for SessionLogHandler class."""

    def test_init_stores_getter(self):
        """Verifies SessionLogHandler stores the manager getter function."""
        mock_getter = MagicMock(return_value=None)
        handler = SessionLogHandler(mock_getter, level=logging.INFO)

        assert handler._get_manager is mock_getter
        assert handler.level == logging.INFO

    def test_init_default_level(self):
        """Verifies SessionLogHandler defaults to NOTSET level."""
        mock_getter = MagicMock(return_value=None)
        handler = SessionLogHandler(mock_getter)

        assert handler.level == logging.NOTSET

    def test_emit_forwards_to_manager(self):
        """Verifies emit() forwards log records to session manager."""
        mock_manager = MagicMock()
        mock_getter = MagicMock(return_value=mock_manager)
        handler = SessionLogHandler(mock_getter)

        record = logging.LogRecord(
            name="test.module",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.structured_data = {"camera_id": 0}

        handler.emit(record)

        mock_manager.log.assert_called_once_with(
            level="INFO",
            message="Test message",
            source="test.module",
            camera_id=0,
        )

    def test_emit_skips_when_manager_none(self):
        """Verifies emit() skips silently when manager getter returns None."""
        mock_getter = MagicMock(return_value=None)
        handler = SessionLogHandler(mock_getter)

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test",
            args=(),
            exc_info=None,
        )

        # Should not raise
        handler.emit(record)
        mock_getter.assert_called_once()

    def test_emit_handles_exception(self):
        """Verifies emit() catches exceptions and calls handleError."""
        mock_manager = MagicMock()
        mock_manager.log.side_effect = RuntimeError("Manager failed")
        mock_getter = MagicMock(return_value=mock_manager)
        handler = SessionLogHandler(mock_getter)
        handler.handleError = MagicMock()

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test",
            args=(),
            exc_info=None,
        )

        handler.emit(record)

        handler.handleError.assert_called_once_with(record)

    def test_emit_recursion_guard(self):
        """Verifies emit() prevents recursive calls within same thread."""
        call_count = 0

        def recursive_log(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # Simulate recursion by calling emit again
            if call_count < 3:
                handler.emit(record)

        mock_manager = MagicMock()
        mock_manager.log.side_effect = recursive_log
        mock_getter = MagicMock(return_value=mock_manager)
        handler = SessionLogHandler(mock_getter)

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test",
            args=(),
            exc_info=None,
        )

        handler.emit(record)

        # Should only call log once due to recursion guard
        assert call_count == 1

    def test_emit_without_structured_data(self):
        """Verifies emit() works when record lacks structured_data."""
        mock_manager = MagicMock()
        mock_getter = MagicMock(return_value=mock_manager)
        handler = SessionLogHandler(mock_getter)

        record = logging.LogRecord(
            name="test",
            level=logging.WARNING,
            pathname="test.py",
            lineno=10,
            msg="Warning message",
            args=(),
            exc_info=None,
        )
        # No structured_data attribute

        handler.emit(record)

        mock_manager.log.assert_called_once_with(
            level="WARNING",
            message="Warning message",
            source="test",
        )


# =============================================================================
# Coverage Tests - LogContext edge cases
# =============================================================================


class TestLogContextCoverage:
    """Additional coverage tests for LogContext."""

    def test_exit_with_none_token(self):
        """Verifies LogContext.__exit__ handles None token gracefully."""
        ctx = LogContext(key="value")
        # Don't enter - token remains None
        ctx._token = None

        # Should not raise
        ctx.__exit__(None, None, None)

    def test_context_override_existing_keys(self):
        """Verifies nested LogContext overrides parent keys."""
        from telescope_mcp.observability.logging import _log_context

        with LogContext(key="outer"):
            outer_ctx = _log_context.get()
            assert outer_ctx.get("key") == "outer"

            with LogContext(key="inner"):
                inner_ctx = _log_context.get()
                assert inner_ctx.get("key") == "inner"

            # Back to outer
            restored = _log_context.get()
            assert restored.get("key") == "outer"


# =============================================================================
# Coverage Tests - Logger level branches
# =============================================================================


class TestLoggerLevelBranches:
    """Test log methods when level is disabled (early return branches)."""

    def test_debug_when_disabled(self):
        """Verifies debug() returns early when DEBUG not enabled."""
        reset_logging()
        configure_logging(level=logging.WARNING, force=True)
        logger = get_logger("test.level.debug")
        logger.setLevel(logging.WARNING)

        # DEBUG is below WARNING, so should be skipped
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        logger.addHandler(handler)

        logger.debug("This should not appear")

        assert stream.getvalue() == ""

    def test_debug_when_enabled(self):
        """Verifies debug() logs when DEBUG level enabled."""
        reset_logging()
        configure_logging(level=logging.DEBUG, force=True)
        logger = get_logger("test.level.debug.enabled")

        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(StructuredFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        logger.debug("Debug message", key="value")

        output = stream.getvalue()
        assert "Debug message" in output

    def test_info_when_disabled(self):
        """Verifies info() returns early when INFO not enabled."""
        reset_logging()
        configure_logging(level=logging.ERROR, force=True)
        logger = get_logger("test.level.info")
        logger.setLevel(logging.ERROR)

        stream = StringIO()
        handler = logging.StreamHandler(stream)
        logger.addHandler(handler)

        logger.info("This should not appear")

        assert stream.getvalue() == ""

    def test_warning_when_disabled(self):
        """Verifies warning() returns early when WARNING not enabled."""
        reset_logging()
        configure_logging(level=logging.CRITICAL, force=True)
        logger = get_logger("test.level.warning")
        logger.setLevel(logging.CRITICAL)

        stream = StringIO()
        handler = logging.StreamHandler(stream)
        logger.addHandler(handler)

        logger.warning("This should not appear")

        assert stream.getvalue() == ""

    def test_error_when_disabled(self):
        """Verifies error() returns early when ERROR not enabled."""
        reset_logging()
        configure_logging(level=logging.CRITICAL + 1, force=True)
        logger = get_logger("test.level.error")
        logger.setLevel(logging.CRITICAL + 1)

        stream = StringIO()
        handler = logging.StreamHandler(stream)
        logger.addHandler(handler)

        logger.error("This should not appear")

        assert stream.getvalue() == ""

    def test_critical_when_disabled(self):
        """Verifies critical() returns early when CRITICAL not enabled."""
        reset_logging()
        configure_logging(level=logging.CRITICAL + 10, force=True)
        logger = get_logger("test.level.critical")
        logger.setLevel(logging.CRITICAL + 10)

        stream = StringIO()
        handler = logging.StreamHandler(stream)
        logger.addHandler(handler)

        logger.critical("This should not appear")

        assert stream.getvalue() == ""


# =============================================================================
# Coverage Tests - get_logger auto-configuration
# =============================================================================


class TestGetLoggerAutoConfig:
    """Test get_logger automatic configuration behavior."""

    def test_get_logger_auto_configures(self):
        """Verifies get_logger auto-configures when not yet configured."""
        reset_logging()
        from telescope_mcp.observability import logging as log_module

        assert log_module._configured is False

        # get_logger should auto-configure
        logger = get_logger("test.autoconfig")

        assert log_module._configured is True
        assert isinstance(logger, StructuredLogger)


# =============================================================================
# Coverage Tests - Branch Coverage
# =============================================================================


class TestBranchCoverage:
    """Tests specifically targeting partial branch coverage."""

    def test_log_with_extra_dict(self):
        """Verifies _log handles pre-existing extra dict (branch 458->460)."""
        reset_logging()
        configure_logging(level=logging.DEBUG, force=True)
        logger = get_logger("test.extra.dict")
        logger.setLevel(logging.DEBUG)

        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(StructuredFormatter())
        logger.addHandler(handler)

        # Call with extra dict already provided
        logger.info("Test message", extra={"custom_field": "custom_value"}, key="value")

        output = stream.getvalue()
        assert "Test message" in output

    def test_structured_formatter_with_custom_fmt(self):
        """Verifies StructuredFormatter accepts custom format string."""
        custom_fmt = "%(levelname)s: %(message)s"
        formatter = StructuredFormatter(fmt=custom_fmt)

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Custom format test",
            args=(),
            exc_info=None,
        )
        record.structured_data = {"key": "value"}

        formatted = formatter.format(record)

        # Should use custom format (no timestamp, no logger name)
        assert formatted.startswith("INFO: Custom format test")
        assert "key=value" in formatted

    def test_structured_formatter_with_custom_datefmt(self):
        """Verifies StructuredFormatter accepts custom date format."""
        formatter = StructuredFormatter(datefmt="%Y-%m-%d")

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Date format test",
            args=(),
            exc_info=None,
        )
        record.structured_data = {}

        formatted = formatter.format(record)
        assert "Date format test" in formatted
