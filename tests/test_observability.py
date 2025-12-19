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
        """Verifies StructuredLogRecord stores structured data dict.

        Arrangement:
        1. StructuredLogRecord with structured_data kwarg.
        2. Record inherits from logging.LogRecord.
        3. structured_data stored as instance attribute.

        Action:
        Creates log record with structured_data parameter.

        Assertion Strategy:
        Validates record creation by confirming:
        - structured_data equals {"key": "value"}.
        - Standard LogRecord fields preserved (name, level).

        Testing Principle:
        Validates extension pattern, ensuring StructuredLogRecord
        adds metadata without breaking base LogRecord functionality.
        """
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
        """Verifies record defaults to empty dict when no structured data.

        Arrangement:
        1. StructuredLogRecord created without structured_data kwarg.
        2. __init__ should handle missing parameter gracefully.
        3. Default value: empty dict {}.

        Action:
        Creates record without structured_data parameter.

        Assertion Strategy:
        Validates default handling by confirming:
        - structured_data equals {} (empty dict).

        Testing Principle:
        Validates interface flexibility, ensuring structured_data
        optional for backward compatibility with standard logging.
        """
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
        """Verifies record accepts optional func and sinfo parameters.

        Arrangement:
        1. StructuredLogRecord supports func (function name).
        2. Supports sinfo (stack info string).
        3. All parameters coexist with structured_data.

        Action:
        Creates record with func, sinfo, and structured_data.

        Assertion Strategy:
        Validates full feature support by confirming:
        - funcName equals "my_function".
        - stack_info equals "stack info".
        - structured_data equals {"camera_id": 0}.

        Testing Principle:
        Validates complete interface, ensuring StructuredLogRecord
        supports all LogRecord features plus structured data.
        """
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
        """Verifies formatter produces readable output for simple message.

        Arrangement:
        1. StructuredFormatter with default settings.
        2. Basic LogRecord without structured_data attribute.
        3. Output format: timestamp - name - level - message.

        Action:
        Formats basic LogRecord without structured data.

        Assertion Strategy:
        Validates base formatting by confirming:
        - "Test message" appears in output.
        - "INFO" level name appears.

        Testing Principle:
        Validates formatter baseline, ensuring standard
        logging functionality preserved.
        """
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
        """Verifies structured data appended with pipe separator.

        Arrangement:
        1. StructuredFormatter with default settings.
        2. Record with structured_data = {"camera_id": 0, "name": "ASI120"}.
        3. Format: message | key1=value1 key2=value2.

        Action:
        Formats record with structured_data dict containing key=value pairs.

        Assertion Strategy:
        Validates structured output by confirming:
        - "Camera connected" message appears.
        - "camera_id=0" appears after pipe.
        - "name=ASI120" appears after pipe.
        - "|" separator present.

        Testing Principle:
        Validates structured data formatting, ensuring
        key=value pairs appended for human readability.
        """
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
        """Verifies no separator when structured_data is empty.

        Arrangement:
        1. StructuredFormatter with default settings.
        2. Record without structured_data attribute.
        3. Should format base message only.

        Action:
        Formats record without structured_data attribute.

        Assertion Strategy:
        Validates clean output by confirming:
        - "Simple message" appears.
        - "|" separator NOT present.

        Testing Principle:
        Validates conditional formatting, ensuring pipe
        separator only appears when structured data exists.
        """
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
        """Verifies include_structured=False suppresses key=value output.

        Arrangement:
        1. StructuredFormatter with include_structured=False.
        2. Record has structured_data = {"key": "value"}.
        3. Structured data should NOT appear in output.

        Action:
        Creates formatter with include_structured=False parameter.

        Assertion Strategy:
        Validates suppression by confirming:
        - "key=value" NOT in output.

        Testing Principle:
        Validates configuration control, ensuring structured
        data can be disabled for simpler output.
        """
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
        """Verifies custom format string is applied to base message.

        Arrangement:
        1. StructuredFormatter with fmt="%(levelname)s: %(message)s".
        2. Simpler format than default (no timestamp/name).
        3. Record at WARNING level.

        Action:
        Creates formatter with custom fmt parameter.

        Assertion Strategy:
        Validates format customization by confirming:
        - Output starts with "WARNING: Custom".

        Testing Principle:
        Validates format flexibility, ensuring StructuredFormatter
        respects standard logging formatter configuration.
        """
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
        """Verifies custom date format is applied to timestamps.

        Arrangement:
        1. StructuredFormatter with datefmt="%Y-%m-%d".
        2. Date-only format (no time component).
        3. Record created with timestamp.

        Action:
        Creates formatter with datefmt=%Y-%m-%d parameter.

        Assertion Strategy:
        Validates date formatting by confirming:
        - Output contains YYYY-MM-DD pattern.

        Testing Principle:
        Validates timestamp customization, ensuring date
        format configurable via standard logging interface.
        """
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
        """Verifies log produces valid parseable JSON object.

        Arrangement:
        1. JSONFormatter configured for machine-readable output.
        2. Basic LogRecord with INFO level.
        3. Expected fields: message, level, logger, timestamp.

        Action:
        Formats basic record as JSON, parses to verify structure.

        Assertion Strategy:
        Validates JSON structure by confirming:
        - message equals "Test message".
        - level equals "INFO".
        - logger equals "test.module".
        - timestamp field present.

        Testing Principle:
        Validates JSON formatter baseline, ensuring NDJSON
        output suitable for log aggregation systems.
        """
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
        """Verifies structured data merged as top-level JSON fields.

        Arrangement:
        1. JSONFormatter merges structured_data into top level.
        2. Record with camera_id=1, error="timeout" metadata.
        3. Flat JSON structure for easy querying.

        Action:
        Formats record with structured_data as JSON.

        Assertion Strategy:
        Validates data merging by confirming:
        - camera_id=1 at top level.
        - error="timeout" at top level.
        - level="WARNING" preserved.

        Testing Principle:
        Validates structured data integration, ensuring
        queryable fields for log aggregation filtering.
        """
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
        """Verifies timestamp uses ISO 8601 format.

        Arrangement:
        1. JSONFormatter uses UTC ISO 8601 for timestamps.
        2. Format: YYYY-MM-DDTHH:MM:SS+00:00.
        3. Parseable by datetime.fromisoformat().

        Action:
        Formats record and validates timestamp format.

        Assertion Strategy:
        Validates timestamp by confirming:
        - fromisoformat() parses without error.

        Testing Principle:
        Validates timestamp standardization for
        cross-system log correlation.
        """
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
        """Verifies JSON formatter includes exception tracebacks.

        Arrangement:
        1. ValueError raised and caught.
        2. Record created with exc_info from sys.exc_info().
        3. Exception formatted as string in JSON.

        Action:
        Formats error record with exception info.

        Assertion Strategy:
        Validates exception capture by confirming:
        - "exception" field present.
        - "ValueError" in exception string.
        - "Test error" message in exception.

        Testing Principle:
        Validates error tracking, ensuring full stack traces
        captured for debugging production issues.
        """
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
        """Verifies JSON formatter handles missing structured_data.

        Arrangement:
        1. Record without structured_data attribute.
        2. Formatter should handle gracefully.
        3. Base fields still present.

        Action:
        Formats record without structured_data attribute.

        Assertion Strategy:
        Validates robustness by confirming:
        - message="Test" in output.
        - No error on missing attribute.

        Testing Principle:
        Validates backward compatibility with standard
        LogRecords lacking structured_data.
        """
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

    def test_none_value(self):
        """Verifies None formats as 'null' string.

        Arrangement:
        Helper function _format_value() with None input.

        Action:
        Calls _format_value(None) and checks output.

        Assertion Strategy:
        Validates null formatting by confirming output equals 'null'.

        Testing Principle:
        Validates JSON-compatible null representation for log output.
        """
        assert _format_value(None) == "null"

    def test_simple_string(self):
        """Verifies simple strings pass through unchanged.

        Arrangement:
        Helper function _format_value() with simple string.

        Action:
        Calls _format_value("hello") and checks output.

        Assertion Strategy:
        Validates passthrough by confirming output equals input string.

        Testing Principle:
        Validates string preservation without quotes for readability.
        """
        assert _format_value("hello") == "hello"

    def test_string_with_spaces(self):
        """Verifies strings with spaces get quoted.

        Arrangement:
        Helper function _format_value() with string containing spaces.

        Action:
        Calls _format_value("hello world") and checks output.

        Assertion Strategy:
        Validates quoting by confirming output has surrounding quotes.

        Testing Principle:
        Validates quoting for unambiguous parsing in log output.
        """
        assert _format_value("hello world") == '"hello world"'

    def test_integer(self):
        """Verifies integers convert to string representation.

        Arrangement:
        Helper function _format_value() with integer input.

        Action:
        Calls _format_value(42) and checks string conversion.

        Assertion Strategy:
        Validates conversion by confirming output equals "42".

        Testing Principle:
        Validates numeric formatting for log output.
        """
        assert _format_value(42) == "42"

    def test_float(self):
        """Verifies floats convert to string representation.

        Arrangement:
        Helper function _format_value() with float input.

        Action:
        Calls _format_value(3.14) and checks string conversion.

        Assertion Strategy:
        Validates conversion by confirming output equals "3.14".

        Testing Principle:
        Validates decimal formatting for log output.
        """
        assert _format_value(3.14) == "3.14"

    def test_dict(self):
        """Verifies dicts serialize as JSON strings.

        Arrangement:
        Helper function _format_value() with dict input.

        Action:
        Calls _format_value({"key": "value"}) and parses result.

        Assertion Strategy:
        Validates JSON serialization by parsing output back to dict.

        Testing Principle:
        Validates structured data preservation through JSON encoding.
        """
        result = _format_value({"key": "value"})
        assert json.loads(result) == {"key": "value"}

    def test_list(self):
        """Verifies lists serialize as JSON arrays.

        Arrangement:
        Helper function _format_value() with list input.

        Action:
        Calls _format_value([1, 2, 3]) and parses result.

        Assertion Strategy:
        Validates JSON array serialization by parsing output back to list.

        Testing Principle:
        Validates array formatting through JSON encoding.
        """
        result = _format_value([1, 2, 3])
        assert json.loads(result) == [1, 2, 3]


class TestLogContext:
    """Tests for LogContext context manager."""

    def test_context_sets_values(self):
        """Verifies LogContext activates key-value pairs.

        Arrangement:
        1. LogContext initialized with camera_id=0, operation="capture".
        2. Context variable holds active context.
        3. Expected: both keys injected into all logs.

        Action:
        Enters context, reads active context value.

        Assertion Strategy:
        Validates context activation by confirming:
        - camera_id=0 in context.
        - operation="capture" in context.

        Testing Principle:
        Validates context injection, ensuring metadata
        propagates to all nested log calls.
        """
        with LogContext(camera_id=0, operation="capture"):
            ctx = _log_context.get()
            assert ctx["camera_id"] == 0
            assert ctx["operation"] == "capture"

    def test_context_restores_on_exit(self):
        """Verifies LogContext restores previous state on exit.

        Arrangement:
        1. Empty context set initially.
        2. LogContext entered with camera_id=0.
        3. Expected: empty context restored after exit.

        Action:
        Enters and exits context, checks restoration.

        Assertion Strategy:
        Validates cleanup by confirming:
        - Context empty {} after exit.

        Testing Principle:
        Validates isolation, ensuring contexts don't
        leak beyond their scope.
        """
        _log_context.set({})
        with LogContext(camera_id=0):
            pass
        assert _log_context.get() == {}

    def test_nested_contexts(self):
        """Verifies nested LogContexts merge properly.

        Arrangement:
        1. Outer context with camera_id=0.
        2. Inner context with operation="capture".
        3. Expected: both values active in inner scope.

        Action:
        Nests contexts, checks merged values.

        Assertion Strategy:
        Validates nesting by confirming:
        - Inner scope has both camera_id and operation.
        - Outer scope restored after inner exit.

        Testing Principle:
        Validates composition, ensuring contexts
        stack naturally for complex operations.
        """
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
        """Verifies inner context overrides outer values.

        Arrangement:
        1. Outer context with camera_id=0.
        2. Inner context with camera_id=1.
        3. Expected: inner value wins.

        Action:
        Nests contexts with conflicting keys.

        Assertion Strategy:
        Validates precedence by confirming:
        - Inner context shows camera_id=1.
        - Outer context restored to camera_id=0 after exit.

        Testing Principle:
        Validates override semantics, ensuring specific
        operations can specialize ambient context.
        """
        with LogContext(camera_id=0):
            with LogContext(camera_id=1):
                ctx = _log_context.get()
                assert ctx["camera_id"] == 1
            # Restored
            ctx = _log_context.get()
            assert ctx["camera_id"] == 0

    def test_context_exit_without_token(self):
        """Verifies LogContext handles exit when token is None.

        Arrangement:
        1. LogContext created but __enter__ not called.
        2. Simulates edge case where token unset.
        3. Expected: __exit__ handles gracefully.

        Action:
        Calls __exit__ directly without prior __enter__.

        Assertion Strategy:
        Validates robustness by confirming:
        - No exception raised when token is None.

        Testing Principle:
        Validates defensive programming, ensuring
        cleanup never crashes.
        """
        ctx = LogContext(key="value")
        # Exit without entering (token is None)
        ctx.__exit__(None, None, None)
        # Should not raise


class TestStructuredLogger:
    """Tests for StructuredLogger."""

    @pytest.fixture
    def logger_and_stream(self):
        """Create a logger with captured output for testing structured logging.

        Business Context:
            Structured logging is critical for telescope operations debugging,
            providing machine-parseable logs for automated analysis. Testing
            requires capturing output for assertion without polluting console.

        Creates a StructuredLogger configured with StringIO stream capture
        and StructuredFormatter for human-readable output testing. Manages
        full lifecycle with automatic handler cleanup.

        Arrangement:
        1. Create StringIO stream for output capture.
        2. Attach StreamHandler with StructuredFormatter.
        3. Create StructuredLogger with DEBUG level.
        4. Disable propagation to isolate test output.

        Args:
            None (pytest fixture with implicit request parameter).

        Returns:
            Tuple[StructuredLogger, StringIO]: Logger configured for testing
                and output stream for assertion on logged content.

        Raises:
            None. Stream and logger creation don't raise exceptions.

        Example:
            >>> def test_example(logger_and_stream):
            ...     logger, stream = logger_and_stream
            ...     logger.info("Test", camera_id=0)
            ...     assert "camera_id=0" in stream.getvalue()

        Implementation Details:
            - propagate=False prevents output leaking to root logger
            - Cleanup ensures no handlers accumulate across tests
            - StringIO provides in-memory capture without file I/O

        Testing Principle:
            Validates fixture reusability, ensuring each test receives
            clean logger/stream pair without cross-test contamination.
        """ ""
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
        """Verifies StructuredLogger produces standard log output.

        Arrangement:
        1. StructuredLogger configured with StringIO stream.
        2. StructuredFormatter attached for human-readable output.
        3. Logger ready for basic message logging.

        Action:
        Logs simple INFO message without structured data.

        Assertion Strategy:
        Validates basic functionality by confirming:
        - Message appears in output stream.
        """
        logger, stream = logger_and_stream
        logger.info("Test message")
        output = stream.getvalue()
        assert "Test message" in output

    def test_structured_kwargs(self, logger_and_stream):
        """Verifies keyword arguments are appended as structured data.

        Arrangement:
        1. Logger configured with StructuredFormatter.
        2. Formatter appends key=value pairs after message.
        3. Test kwargs: camera_id=0, exposure_us=100000.

        Action:
        Logs message with multiple keyword arguments.

        Assertion Strategy:
        Validates structured logging by confirming:
        - camera_id=0 appears in output.
        - exposure_us=100000 appears in output.
        """
        logger, stream = logger_and_stream
        logger.info("Frame captured", camera_id=0, exposure_us=100000)
        output = stream.getvalue()
        assert "camera_id=0" in output
        assert "exposure_us=100000" in output

    def test_context_included(self, logger_and_stream):
        """Verifies LogContext values propagate to log output.

        Arrangement:
        1. Logger configured with structured formatter.
        2. LogContext with request_id="abc123" active.
        3. Context values merged with log record.

        Action:
        Logs message within LogContext scope.

        Assertion Strategy:
        Validates context propagation by confirming:
        - request_id=abc123 appears in output.
        """
        logger, stream = logger_and_stream
        with LogContext(request_id="abc123"):
            logger.info("Processing")
        output = stream.getvalue()
        assert "request_id=abc123" in output

    def test_kwargs_override_context(self, logger_and_stream):
        """Verifies explicit kwargs take precedence over context.

        Arrangement:
        1. LogContext sets camera_id=0.
        2. Log call provides camera_id=1 as kwarg.
        3. Merge order: context < kwargs (kwargs win).

        Action:
        Logs with kwarg that conflicts with context value.

        Assertion Strategy:
        Validates precedence by confirming:
        - camera_id=1 appears (kwarg value, not context).
        """
        logger, stream = logger_and_stream
        with LogContext(camera_id=0):
            logger.info("Changed", camera_id=1)
        output = stream.getvalue()
        assert "camera_id=1" in output

    def test_log_with_args_formatting(self, logger_and_stream):
        """Verifies logger supports standard % formatting.

        Arrangement:
        1. Logger inherits Python logging % formatting.
        2. Message with %d placeholder for integer.
        3. Argument 5 provided for substitution.

        Action:
        Logs message with % formatting and argument.

        Assertion Strategy:
        Validates formatting by confirming:
        - "Captured 5 frames" in output (formatted).
        """
        logger, stream = logger_and_stream
        logger.info("Captured %d frames", 5)
        output = stream.getvalue()
        assert "Captured 5 frames" in output

    def test_log_with_exc_info(self, logger_and_stream):
        """Verifies logger captures exception tracebacks.

        Arrangement:
        1. RuntimeError raised and caught.
        2. Logger called with exc_info=True in except block.
        3. Formatter includes exception traceback.

        Action:
        Logs error message with exception info.

        Assertion Strategy:
        Validates exception logging by confirming:
        - "Error occurred" message in output.
        - "RuntimeError" type appears in traceback.
        """
        logger, stream = logger_and_stream
        try:
            raise RuntimeError("Test error")
        except RuntimeError:
            logger.error("Error occurred", exc_info=True)
        output = stream.getvalue()
        assert "Error occurred" in output
        assert "RuntimeError" in output

    def test_log_with_stack_info(self, logger_and_stream):
        """Verifies logger captures call stack traces.

        Arrangement:
        1. Logger configured to capture stack info.
        2. stack_info=True parameter provided.
        3. Formatter includes stack trace in output.

        Action:
        Logs message with stack_info flag.

        Assertion Strategy:
        Validates stack capture by confirming:
        - "With stack" message appears.
        """
        logger, stream = logger_and_stream
        logger.info("With stack", stack_info=True)
        output = stream.getvalue()
        assert "With stack" in output

    def test_log_with_extra_dict(self, logger_and_stream):
        """Verifies logger handles both extra dict and kwargs.

        Arrangement:
        1. Logger accepts standard extra dict parameter.
        2. Also accepts structured kwargs (camera_id=0).
        3. Both mechanisms coexist without conflict.

        Action:
        Logs with both extra dict and structured kwargs.

        Assertion Strategy:
        Validates compatibility by confirming:
        - camera_id=0 appears in structured output.
        """
        logger, stream = logger_and_stream
        logger.info("Test", extra={"custom": "value"}, camera_id=0)
        output = stream.getvalue()
        assert "camera_id=0" in output

    def test_log_without_context(self, logger_and_stream):
        """Verifies logger functions without active LogContext.

        Arrangement:
        1. Log context explicitly cleared (empty dict).
        2. Logger called with only explicit kwargs.
        3. No ambient context to merge.

        Action:
        Logs message with kwargs but no context.

        Assertion Strategy:
        Validates standalone operation by confirming:
        - key=value appears in output.
        """
        logger, stream = logger_and_stream
        # Clear any context
        _log_context.set({})
        logger.info("No context", key="value")
        output = stream.getvalue()
        assert "key=value" in output


class TestSessionLogHandler:
    """Tests for SessionLogHandler."""

    def test_forwards_to_session_manager(self):
        """Verifies handler forwards log records to session manager.

        Arrangement:
        1. Mock session manager with log() method.
        2. SessionLogHandler configured with manager getter.
        3. LogRecord with structured_data created.

        Action:
        Emits record through handler.

        Assertion Strategy:
        Validates forwarding by confirming:
        - manager.log called once.
        - level="INFO" passed.
        - message="Test message" passed.
        - source="test.module" passed.
        - structured key="value" passed.

        Testing Principle:
        Validates dual-write pattern, ensuring logs
        persisted to both console and session storage.
        """
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
        """Verifies handler handles missing session manager gracefully.

        Arrangement:
        1. Handler getter returns None (no active session).
        2. LogRecord ready to emit.
        3. Expected: silent no-op.

        Action:
        Emits record with None manager.

        Assertion Strategy:
        Validates robustness by confirming:
        - No exception raised.

        Testing Principle:
        Validates graceful degradation when logging
        occurs outside session scope.
        """
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
        """Verifies handler catches exceptions during emit.

        Arrangement:
        1. Mock manager raises RuntimeError on log().
        2. Handler should catch via handleError().
        3. Exception should not propagate.

        Action:
        Emits record that triggers exception.

        Assertion Strategy:
        Validates error handling by confirming:
        - No exception propagates (handler.emit returns).

        Testing Principle:
        Validates defensive logging, ensuring session
        errors never crash application.
        """
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
        """Verifies handler respects custom level setting.

        Arrangement:
        1. Handler created with level=logging.WARNING.
        2. Level stored in handler.level attribute.
        3. Only WARNING+ logs should process.

        Action:
        Creates handler and checks level attribute.

        Assertion Strategy:
        Validates configuration by confirming:
        - handler.level == logging.WARNING.

        Testing Principle:
        Validates level configuration, enabling selective
        session logging by severity.
        """
        mock_manager = MagicMock()
        handler = SessionLogHandler(lambda: mock_manager, level=logging.WARNING)

        # Level should be set
        assert handler.level == logging.WARNING

    def test_recursion_guard_prevents_reentry(self):
        """Verifies recursion guard prevents handler re-entry.

        Arrangement:
        1. Handler with emitting flag manually set True.
        2. Simulates in-progress emit() call.
        3. Second emit() should return early.

        Action:
        Calls emit() with emitting flag active.

        Assertion Strategy:
        Validates guard by confirming:
        - manager.log not called.

        Testing Principle:
        Validates recursion protection via thread-local
        flag, preventing infinite loops.
        """
        mock_manager = MagicMock()
        handler = SessionLogHandler(lambda: mock_manager)

        # Simulate recursion by manually setting the emitting flag
        handler._local.emitting = True

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

        # Should return early without calling manager.log
        handler.emit(record)

        # Manager should NOT have been called due to recursion guard
        mock_manager.log.assert_not_called()


# =============================================================================
# Camera Statistics Tests
# =============================================================================


class TestStatsSummary:
    """Tests for StatsSummary dataclass."""

    def test_default_values(self):
        """Verifies StatsSummary has sensible defaults.

        Arrangement:
        1. StatsSummary dataclass with camera_id only.
        2. Other fields should initialize to zero/empty.
        3. Safe to construct incrementally.

        Action:
        Creates StatsSummary with minimal args.

        Assertion Strategy:
        Validates defaults by confirming:
        - camera_id=0 preserved.
        - total_captures=0 (default).
        - success_rate=0.0 (default).

        Testing Principle:
        Validates dataclass initialization, ensuring
        safe construction without all fields.
        """
        summary = StatsSummary(camera_id=0)
        assert summary.camera_id == 0
        assert summary.total_captures == 0
        assert summary.success_rate == 0.0

    def test_to_dict(self):
        """Verifies to_dict produces serializable output.

        Arrangement:
        1. StatsSummary with all fields populated.
        2. to_dict() converts to plain dict.
        3. Datetime converted to ISO string.

        Action:
        Calls to_dict() on fully populated summary.

        Assertion Strategy:
        Validates serialization by confirming:
        - camera_id=0 in dict.
        - All numeric fields present.
        - Datetime as ISO string.

        Testing Principle:
        Validates JSON-safe serialization for API
        responses and logging.
        """
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
        """Verifies to_dict handles None timestamp gracefully.

        Arrangement:
        1. StatsSummary with last_capture_time=None.
        2. to_dict() should preserve None value.
        3. No error on missing timestamp.

        Action:
        Calls to_dict() with None timestamp.

        Assertion Strategy:
        Validates null handling by confirming:
        - last_capture_time=None in output dict.

        Testing Principle:
        Validates defensive serialization, handling
        uninitialized/optional fields.
        """
        summary = StatsSummary(camera_id=0, last_capture_time=None)
        result = summary.to_dict()
        assert result["last_capture_time"] is None


class TestCameraStatsCollector:
    """Tests for CameraStatsCollector."""

    def test_initial_state(self):
        """Verifies new collector starts with zero stats.

        Arrangement:
        1. CameraStatsCollector constructed for camera_id=0.
        2. No captures recorded yet.
        3. Expected: all counters at zero.

        Action:
        Creates collector and gets summary.

        Assertion Strategy:
        Validates initialization by confirming:
        - total_captures=0.
        - success_rate=0.0.

        Testing Principle:
        Validates clean slate initialization for new
        camera tracking.
        """
        collector = CameraStatsCollector(camera_id=0)
        summary = collector.get_summary()
        assert summary.total_captures == 0
        assert summary.success_rate == 0.0

    def test_record_success(self):
        """Verifies successful capture updates all counters.

        Arrangement:
        1. Fresh collector with zero captures.
        2. record() with success=True, duration_ms=150.
        3. Success counter should increment.

        Action:
        Records one successful capture.

        Assertion Strategy:
        Validates success tracking by confirming:
        - total_captures=1.
        - successful_captures=1.
        - success_rate=1.0 (100%).

        Testing Principle:
        Validates baseline success tracking for
        reliability monitoring.
        """
        collector = CameraStatsCollector(camera_id=0)
        collector.record(duration_ms=150.0, success=True)

        summary = collector.get_summary()
        assert summary.total_captures == 1
        assert summary.successful_captures == 1
        assert summary.success_rate == 1.0

    def test_record_failure(self):
        """Verifies failed capture tracks error type.

        Arrangement:
        1. Collector with zero failures.
        2. record() with success=False, error_type="timeout".
        3. Error counts dict should track category.

        Action:
        Records failed capture with error type.

        Assertion Strategy:
        Validates failure tracking by confirming:
        - total_captures=1.
        - failed_captures=1.
        - success_rate=0.0.
        - error_counts["timeout"]=1.

        Testing Principle:
        Validates error categorization for root
        cause analysis.
        """
        collector = CameraStatsCollector(camera_id=0)
        collector.record(duration_ms=0, success=False, error_type="timeout")

        summary = collector.get_summary()
        assert summary.total_captures == 1
        assert summary.failed_captures == 1
        assert summary.success_rate == 0.0
        assert summary.error_counts["timeout"] == 1

    def test_record_failure_without_error_type(self):
        """Verifies failed capture handles None error_type.

        Arrangement:
        1. Collector ready to track failures.
        2. record() with success=False, error_type=None.
        3. Should not crash on None error.

        Action:
        Records failure without error type.

        Assertion Strategy:
        Validates graceful handling by confirming:
        - failed_captures=1 (counted).
        - error_counts={} (empty, no type tracked).

        Testing Principle:
        Validates defensive programming, handling
        unspecified error categories.
        """
        collector = CameraStatsCollector(camera_id=0)
        collector.record(duration_ms=0, success=False, error_type=None)

        summary = collector.get_summary()
        assert summary.total_captures == 1
        assert summary.failed_captures == 1
        assert summary.success_rate == 0.0
        assert summary.error_counts == {}  # No error type tracked

    def test_duration_statistics(self):
        """Verifies min/max/avg duration calculations.

        Arrangement:
        1. Collector records 3 captures.
        2. Durations: 100ms, 200ms, 150ms.
        3. Stats: min=100, max=200, avg=150.

        Action:
        Records captures with varying durations.

        Assertion Strategy:
        Validates stats by confirming:
        - min_duration_ms=100.0.
        - max_duration_ms=200.0.
        - avg_duration_ms=150.0.

        Testing Principle:
        Validates timing analytics for performance
        optimization insights.
        """
        collector = CameraStatsCollector(camera_id=0)
        collector.record(duration_ms=100.0, success=True)
        collector.record(duration_ms=200.0, success=True)
        collector.record(duration_ms=150.0, success=True)

        summary = collector.get_summary()
        assert summary.min_duration_ms == 100.0
        assert summary.max_duration_ms == 200.0
        assert summary.avg_duration_ms == 150.0

    def test_p95_duration(self):
        """Verifies P95 percentile calculation.

        Arrangement:
        1. Collector records 100 captures.
        2. Durations: 1ms through 100ms.
        3. P95 should be ~95ms (95th percentile).

        Action:
        Records 100 samples with linear distribution.

        Assertion Strategy:
        Validates percentile by confirming:
        - 94 <= p95_duration_ms <= 96.

        Testing Principle:
        Validates tail latency tracking for SLA
        monitoring and performance analysis.
        """
        collector = CameraStatsCollector(camera_id=0)
        # Record 100 captures with durations 1-100
        for i in range(1, 101):
            collector.record(duration_ms=float(i), success=True)

        summary = collector.get_summary()
        # P95 should be around 95
        assert 94 <= summary.p95_duration_ms <= 96

    def test_reset(self):
        """Verifies reset clears all statistics.

        Arrangement:
        1. Collector with recorded captures.
        2. reset() called to clear state.
        3. Should return to initial zero state.

        Action:
        Records capture then calls reset().

        Assertion Strategy:
        Validates reset by confirming:
        - total_captures=0 after reset.

        Testing Principle:
        Validates state reset for session boundaries
        or testing scenarios.
        """
        collector = CameraStatsCollector(camera_id=0)
        collector.record(duration_ms=150.0, success=True)
        collector.reset()

        summary = collector.get_summary()
        assert summary.total_captures == 0

    def test_window_size(self):
        """Verifies window size limits duration storage.

        Arrangement:
        1. Collector with window_size=10.
        2. Records 20 captures (exceeds window).
        3. Total count includes all, but duration stats
           calculated from last 10 only.

        Action:
        Records 20 captures with window_size=10.

        Assertion Strategy:
        Validates windowing by confirming:
        - total_captures=20 (all counted).
        - Duration stats from last 10 samples only.

        Testing Principle:
        Validates sliding window for memory-bounded
        statistics in long-running sessions.
        """
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
        """Verifies collector is thread-safe for concurrent recording.

        Arrangement:
        1. Single CameraStatsCollector instance shared across threads.
        2. 10 threads each recording 100 captures concurrently.
        3. Expected total: 1000 captures without data corruption.

        Action:
        Spawns 10 concurrent threads recording to same collector.

        Assertion Strategy:
        Validates thread-safety by confirming total_captures equals 1000.

        Testing Principle:
        Validates concurrent access safety for multi-threaded telescope operations.
        """
        collector = CameraStatsCollector(camera_id=0)

        def record_captures():
            """Thread worker that records 100 captures to shared collector.

            Business context:
            Validates telescope camera stats collection is thread-safe
            for concurrent capture operations during observations.

            Args:
                None (closure captures collector instance).

            Returns:
                None. Modifies collector state via record() calls.

            Raises:
                None.
            """
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
        """Verifies recording auto-creates collector for new camera.

        Arrangement:
        1. CameraStats manager with no collectors.
        2. record_capture() called for camera_id=0.
        3. Should create collector on first use.

        Action:
        Records capture for previously unseen camera.

        Assertion Strategy:
        Validates auto-initialization by confirming:
        - get_summary() succeeds for camera_id=0.
        - total_captures=1 recorded.

        Testing Principle:
        Validates lazy initialization pattern for
        dynamic camera discovery.
        """
        stats = CameraStats()
        stats.record_capture(camera_id=0, duration_ms=150.0, success=True)

        summary = stats.get_summary(camera_id=0)
        assert summary.total_captures == 1

    def test_multiple_cameras(self):
        """Verifies stats tracked independently per camera.

        Arrangement:
        1. Two cameras with different performance.
        2. Camera 0: 100ms, Camera 1: 200ms.
        3. Stats should not cross-contaminate.

        Action:
        Records captures for two cameras.

        Assertion Strategy:
        Validates isolation by confirming:
        - Camera 0 avg_duration=100ms.
        - Camera 1 avg_duration=200ms.

        Testing Principle:
        Validates multi-camera tracking for systems
        with finder + main cameras.
        """
        stats = CameraStats()
        stats.record_capture(camera_id=0, duration_ms=100.0, success=True)
        stats.record_capture(camera_id=1, duration_ms=200.0, success=True)

        summary0 = stats.get_summary(camera_id=0)
        summary1 = stats.get_summary(camera_id=1)

        assert summary0.avg_duration_ms == 100.0
        assert summary1.avg_duration_ms == 200.0

    def test_get_all_summaries(self):
        """Verifies get_all_summaries returns all cameras.

        Arrangement:
        1. Stats recorded for cameras 0 and 1.
        2. get_all_summaries() retrieves both.
        3. Returns dict keyed by camera_id.

        Action:
        Records for multiple cameras, gets all summaries.

        Assertion Strategy:
        Validates aggregation by confirming:
        - Key 0 present in returned dict.
        - Key 1 present in returned dict.

        Testing Principle:
        Validates bulk export for monitoring dashboards
        and API responses.
        """
        stats = CameraStats()
        stats.record_capture(camera_id=0, duration_ms=100.0, success=True)
        stats.record_capture(camera_id=1, duration_ms=200.0, success=True)

        all_summaries = stats.get_all_summaries()
        assert 0 in all_summaries
        assert 1 in all_summaries

    def test_reset_single_camera(self):
        """Verifies reset clears single camera without affecting others.

        Arrangement:
        1. Two cameras with recorded stats.
        2. reset(camera_id=0) called.
        3. Camera 1 should remain unaffected.

        Action:
        Resets one camera's stats.

        Assertion Strategy:
        Validates selective reset by confirming:
        - Camera 0 total_captures=0 (reset).
        - Camera 1 total_captures=1 (preserved).

        Testing Principle:
        Validates targeted reset for camera
        reconnection/restart scenarios.
        """
        stats = CameraStats()
        stats.record_capture(camera_id=0, duration_ms=100.0, success=True)
        stats.record_capture(camera_id=1, duration_ms=200.0, success=True)

        stats.reset(camera_id=0)

        assert stats.get_summary(camera_id=0).total_captures == 0
        assert stats.get_summary(camera_id=1).total_captures == 1

    def test_reset_nonexistent_camera(self):
        """Verifies reset handles nonexistent camera gracefully.

        Arrangement:
        1. Camera 0 has stats.
        2. reset() called for camera 99 (doesn't exist).
        3. Should not crash or affect camera 0.

        Action:
        Resets nonexistent camera.

        Assertion Strategy:
        Validates robustness by confirming:
        - No exception raised.
        - Camera 0 unaffected (total_captures=1).

        Testing Principle:
        Validates defensive programming for invalid
        camera IDs from user input.
        """
        stats = CameraStats()
        stats.record_capture(camera_id=0, duration_ms=100.0, success=True)

        # Reset camera that doesn't exist - should not raise
        stats.reset(camera_id=99)

        # Camera 0 should be unaffected
        assert stats.get_summary(camera_id=0).total_captures == 1

    def test_reset_all_cameras(self):
        """Verifies reset without camera_id clears all cameras.

        Arrangement:
        1. Multiple cameras with recorded stats.
        2. reset() called without arguments.
        3. All collectors should clear.

        Action:
        Resets all cameras via parameterless reset().

        Assertion Strategy:
        Validates global reset by confirming:
        - Camera 0 total_captures=0.
        - Camera 1 total_captures=0.

        Testing Principle:
        Validates system-wide reset for session
        boundaries or testing.
        """
        stats = CameraStats()
        stats.record_capture(camera_id=0, duration_ms=100.0, success=True)
        stats.record_capture(camera_id=1, duration_ms=200.0, success=True)

        stats.reset()

        assert stats.get_summary(camera_id=0).total_captures == 0
        assert stats.get_summary(camera_id=1).total_captures == 0

    def test_to_dict(self):
        """Verifies to_dict exports all camera stats.

        Arrangement:
        1. Stats recorded for camera 0.
        2. to_dict() should serialize for JSON export.
        3. Output includes cameras dict and timestamp.

        Action:
        Records stats and exports to dict.

        Assertion Strategy:
        Validates serialization by confirming:
        - "cameras" key present.
        - "0" (str) key in cameras dict.
        - "timestamp" key present.

        Testing Principle:
        Validates JSON-safe export for API responses
        and persistent storage.
        """
        stats = CameraStats()
        stats.record_capture(camera_id=0, duration_ms=100.0, success=True)

        result = stats.to_dict()
        assert "cameras" in result
        assert "0" in result["cameras"]
        assert "timestamp" in result


class TestPercentile:
    """Tests for _percentile helper."""

    def test_empty_list(self):
        """Verifies empty list returns 0.

        Arrangement:
        Helper function _percentile() with empty list input.

        Action:
        Calls _percentile([], 50) for median of empty dataset.

        Assertion Strategy:
        Validates edge case by confirming return value equals 0.0.

        Testing Principle:
        Validates edge case handling for uninitialized/empty datasets.
        """
        assert _percentile([], 50) == 0.0

    def test_single_value(self):
        """Verifies single value returns itself for any percentile.

        Arrangement:
        Helper function _percentile() with single-element list.

        Action:
        Calls _percentile([100.0], 50) for median.

        Assertion Strategy:
        Validates degenerate case by confirming output equals 100.0.

        Testing Principle:
        Validates degenerate case where all percentiles collapse to single value.
        """
        assert _percentile([100.0], 50) == 100.0

    def test_median(self):
        """Verifies P50 (median) calculation.

        Arrangement:
        Helper function _percentile() with 5-element sorted list.

        Action:
        Calls _percentile([1.0, 2.0, 3.0, 4.0, 5.0], 50).

        Assertion Strategy:
        Validates median by confirming middle value equals 3.0.

        Testing Principle:
        Validates median as middle value in sorted odd-length list.
        """
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert _percentile(data, 50) == 3.0

    def test_p0(self):
        """Verifies P0 returns minimum value.

        Arrangement:
        Helper function _percentile() with 3-element list.

        Action:
        Calls _percentile([1.0, 2.0, 3.0], 0) for 0th percentile.

        Assertion Strategy:
        Validates boundary by confirming output equals minimum 1.0.

        Testing Principle:
        Validates boundary condition where 0th percentile equals minimum.
        """
        data = [1.0, 2.0, 3.0]
        assert _percentile(data, 0) == 1.0

    def test_p100(self):
        """Verifies P100 returns maximum value.

        Arrangement:
        Helper function _percentile() with 3-element list.

        Action:
        Calls _percentile([1.0, 2.0, 3.0], 100) for 100th percentile.

        Assertion Strategy:
        Validates boundary by confirming output equals maximum 3.0.

        Testing Principle:
        Validates boundary condition where 100th percentile equals maximum.
        """
        data = [1.0, 2.0, 3.0]
        assert _percentile(data, 100) == 3.0

    def test_p95(self):
        """Verifies P95 interpolation for large datasets.

        Arrangement:
        Helper function _percentile() with 100 linearly-distributed values.

        Action:
        Calls _percentile(1.0..100.0, 95) for 95th percentile.

        Assertion Strategy:
        Validates interpolation by confirming result in range [94, 96].

        Testing Principle:
        Validates linear interpolation for tail latency analysis (SLA monitoring).
        """
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
        """Verifies default configuration creates working logger.

        Arrangement:
        1. Module _configured flag reset to False.
        2. configure_logging() called with defaults.
        3. Expected: INFO level, structured format, stderr.

        Action:
        Configures logging and tests output.

        Assertion Strategy:
        Validates setup by confirming:
        - _configured flag set True.
        - Test message appears in output stream.

        Testing Principle:
        Validates default initialization for typical
        development usage.
        """
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
        """Verifies JSON formatter configuration.

        Arrangement:
        1. configure_logging(json_format=True).
        2. Output should be parseable NDJSON.
        3. For production log aggregation.

        Action:
        Configures with JSON, logs message, parses output.

        Assertion Strategy:
        Validates JSON output by confirming:
        - Output parses as valid JSON.
        - message field equals "JSON test".

        Testing Principle:
        Validates production logging configuration
        for log aggregation systems.
        """
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
        """Verifies DEBUG level configuration.

        Arrangement:
        1. configure_logging(level=logging.DEBUG).
        2. DEBUG messages should appear in output.
        3. For development/troubleshooting.

        Action:
        Configures DEBUG level, logs debug message.

        Assertion Strategy:
        Validates level by confirming:
        - Debug message appears in output.

        Testing Principle:
        Validates verbose logging for development
        and debugging scenarios.
        """
        from telescope_mcp.observability import logging as log_module

        log_module._configured = False

        stream = io.StringIO()
        configure_logging(level=logging.DEBUG, stream=stream)

        logger = logging.getLogger("telescope_mcp.test_debug")
        logger.debug("Debug message")

        output = stream.getvalue()
        assert "Debug message" in output

    def test_configure_with_string_level(self):
        """Verifies string level name configuration.

        Arrangement:
        1. configure_logging(level="WARNING").
        2. INFO messages filtered, WARNING+ pass.
        3. Convenience for config files.

        Action:
        Configures WARNING level, logs INFO and WARNING.

        Assertion Strategy:
        Validates filtering by confirming:
        - Info message not in output (filtered).
        - Warning message in output (passed).

        Testing Principle:
        Validates string-based level configuration
        for user-friendly config files.
        """
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
        """Verifies include_structured=False suppresses key=value.

        Arrangement:
        1. configure_logging(include_structured=False).
        2. Structured kwargs should not appear in output.
        3. For minimal/legacy logging.

        Action:
        Logs with kwargs, checks output.

        Assertion Strategy:
        Validates suppression by confirming:
        - Base message appears.
        - key=value does not appear.

        Testing Principle:
        Validates backward compatibility mode
        for traditional logging output.
        """
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
        """Verifies repeated configure_logging calls are safe.

        Arrangement:
        1. First configure_logging(stream=stream1).
        2. Second configure_logging(stream=stream2).
        3. Second call should be no-op (first config wins).

        Action:
        Configures twice, logs message.

        Assertion Strategy:
        Validates idempotence by confirming:
        - stream1 has output (first config active).
        - stream2 empty (second config ignored).

        Testing Principle:
        Validates safe repeated initialization
        in complex applications.
        """
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
        """Verifies get_logger returns StructuredLogger instance.

        Arrangement:
        1. Reset _configured flag to unconfigured state.
        2. Request logger with unique name.
        3. Expected: StructuredLogger instance, not base Logger.

        Action:
        Calls get_logger("telescope_mcp.test_get").

        Assertion Strategy:
        Validates return type by confirming:
        - isinstance(logger, StructuredLogger) is True.

        Testing Principle:
        Validates type contract, ensuring get_logger always
        returns enhanced StructuredLogger for structured data.
        """ ""
        from telescope_mcp.observability import logging as log_module

        log_module._configured = False

        logger = get_logger("telescope_mcp.test_get")
        assert isinstance(logger, StructuredLogger)

    def test_get_logger_auto_configures(self):
        """Verifies get_logger auto-configures logging if unconfigured.

        Arrangement:
        1. Reset _configured flag to False.
        2. Logging module in unconfigured state.
        3. Expected: get_logger triggers configure_logging().

        Action:
        Calls get_logger("telescope_mcp.test_auto").

        Assertion Strategy:
        Validates auto-configuration by confirming:
        - log_module._configured is True after call.

        Testing Principle:
        Validates lazy initialization, ensuring first get_logger
        call configures logging without explicit setup.
        """ ""
        from telescope_mcp.observability import logging as log_module

        log_module._configured = False

        _logger = get_logger("telescope_mcp.test_auto")  # noqa: F841

        # Should be configured now
        assert log_module._configured

    def test_get_logger_with_name(self):
        """Verifies get_logger uses provided name for logger identity.

        Arrangement:
        1. Request logger with custom module-style name.
        2. Expected: logger.name matches provided string exactly.
        3. Name used for logger hierarchy and filtering.

        Action:
        Calls get_logger("telescope_mcp.custom.module").

        Assertion Strategy:
        Validates name assignment by confirming:
        - logger.name == "telescope_mcp.custom.module".

        Testing Principle:
        Validates naming contract, ensuring logger names match
        module paths for hierarchical log filtering.
        """ ""
        logger = get_logger("telescope_mcp.custom.module")
        assert logger.name == "telescope_mcp.custom.module"

    def test_get_logger_when_already_configured(self):
        """Verifies get_logger works when logging already configured.

        Arrangement:
        1. Ensure _configured is True via configure_logging().
        2. Logging system in configured state.
        3. Expected: get_logger skips configuration.

        Action:
        Calls get_logger("telescope_mcp.test_already").

        Assertion Strategy:
        Validates idempotence by confirming:
        - isinstance(logger, StructuredLogger) is True.
        - No re-configuration side effects.

        Testing Principle:
        Validates idempotence, ensuring multiple get_logger
        calls do not reconfigure or corrupt logging state.
        """
        from telescope_mcp.observability import logging as log_module

        # Ensure configured
        if not log_module._configured:
            configure_logging()

        logger = get_logger("telescope_mcp.test_already")
        assert isinstance(logger, StructuredLogger)

    def test_get_logger_double_check_locking(self):
        """Verifies double-checked locking pattern works correctly.

        The inner check (line 676) is defensive for race conditions.
        We verify the overall behavior is correct.

        Arrangement:
        1. Ensure logging configured at start.
        2. Spawn 10 concurrent threads.
        3. Each thread requests unique logger.
        4. Expected: all succeed, no race conditions.

        Action:
        Multiple threads call get_logger() concurrently.

        Assertion Strategy:
        Validates thread-safety by confirming:
        - All 10 threads complete successfully.
        - results list has 10 entries.
        - _configured remains True.

        Testing Principle:
        Validates thread-safety, ensuring concurrent get_logger
        calls don't corrupt configuration or logger creation.
        """
        import threading

        from telescope_mcp.observability import logging as log_module

        # Ensure we start configured
        if not log_module._configured:
            configure_logging()

        # Multiple threads calling get_logger should all succeed
        results = []

        def get_logger_thread(n):
            """Thread worker that gets logger and records name.

            Helper function for concurrent logger creation testing.
            Simulates multi-threaded telescope operations where
            different subsystems request loggers simultaneously.

            Arrangement:
            1. Thread receives unique identifier n.
            2. Requests logger with thread-specific name.
            3. Appends logger name to shared results list.

            Action:
            Gets logger from concurrent thread context.

            Assertion Strategy:
            Validates thread-safety by appending logger name to results.

            Args:
                n (int): Thread identifier for unique logger name.

            Returns:
                None. Appends logger.name to shared results list.

            Raises:
                None. Thread-safe get_logger shouldn't raise.

            Example:
                >>> results = []
                >>> t = threading.Thread(target=get_logger_thread, args=(1,))
                >>> t.start()
                >>> t.join()
                >>> assert "telescope_mcp.concurrent.1" in results

            Testing Principle:
            Validates concurrent initialization, ensuring
            get_logger() is thread-safe under contention.
            """
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
        """Verifies JSON logging produces parseable output end-to-end.

        Arrangement:
        1. StructuredLogger with JSONFormatter.
        2. LogContext active with session_id.
        3. Log with structured kwargs (camera_id, duration_ms).

        Action:
        Logs message within context, parses JSON output.

        Assertion Strategy:
        Validates integration by confirming:
        - message="Frame captured".
        - session_id="test123" (from context).
        - camera_id=0 (from kwargs).
        - duration_ms=150.5 (from kwargs).

        Testing Principle:
        Validates full stack integration: context +
        structured logger + JSON formatter.
        """
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
        """Verifies stats system handles realistic capture session.

        Arrangement:
        1. CameraStats manager for camera_id=0.
        2. Simulate 100 captures with realistic timings.
        3. Mean duration 150ms (stddev 20ms), 95% success rate.

        Action:
        Records 100 captures with Gaussian distribution.

        Assertion Strategy:
        Validates realism by confirming:
        - total_captures ~100 (all recorded).
        - success_rate ~0.90-1.0 (95% target).
        - avg_duration ~100-200ms (150ms target).

        Testing Principle:
        Validates full system integration under
        realistic telescope operation load.
        """
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
