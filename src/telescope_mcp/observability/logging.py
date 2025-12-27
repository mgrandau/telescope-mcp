"""Structured logging for telescope-mcp.

Builds on Python's standard logging module with:
- Structured data support (key-value pairs in logs)
- JSON formatting option for log aggregation
- Context management for request/operation tracking
- Integration with session logging

Design Principles:
- Compatible with standard logging (drop-in replacement)
- Structured data via extra dict (Python logging standard)
- Thread-safe context management
- Optional JSON output for production

Security Note:
    When logging user-provided or untrusted data, always use structured
    keyword arguments rather than including them in the message string:

    # SAFE - structured data is properly escaped
    logger.info("User action", user_id=untrusted_id, action=untrusted_action)

    # UNSAFE - could inject fake log entries with CRLF
    logger.info(f"User {untrusted_id} performed {untrusted_action}")

Example:
    # Get a logger (like standard logging)
    logger = get_logger(__name__)

    # Simple logging works as expected
    logger.info("Server started")

    # Structured logging with keyword arguments
    logger.info("Camera connected", camera_id=0, name="ZWO ASI120")

    # Context manager for operation tracking
    with LogContext(request_id="abc123", camera_id=0):
        logger.info("Starting capture")  # Automatically includes context
        do_capture()
        logger.info("Capture complete", duration_ms=150)

    # JSON output for production
    configure_logging(json_format=True)
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
import threading
from collections.abc import MutableMapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, cast

# Context variable for structured logging context
_log_context: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "log_context", default={}
)


# =============================================================================
# Structured Log Record
# =============================================================================


class StructuredLogRecord(logging.LogRecord):
    """LogRecord with structured data support.

    Extends standard LogRecord to include structured key-value data
    that can be formatted as JSON or human-readable text.
    """

    structured_data: dict[str, Any]

    def __init__(
        self,
        name: str,
        level: int,
        pathname: str,
        lineno: int,
        msg: object,
        args: tuple[Any, ...] | MutableMapping[str, Any] | None,
        exc_info: Any,
        func: str | None = None,
        sinfo: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a structured log record with optional structured data.

        Creates a standard LogRecord and attaches structured key-value data
        for downstream formatting. This enables logs to carry contextual
        metadata (camera IDs, durations, etc.) that can be formatted as
        JSON or human-readable key=value pairs.

        Args:
            name: Logger name (e.g., 'telescope_mcp.devices.camera').
            level: Numeric log level (e.g., logging.INFO = 20).
            pathname: Full path to source file where log was created.
            lineno: Line number in source file.
            msg: Log message (may contain % formatting placeholders).
            args: Arguments for % formatting of msg, or None.
            exc_info: Exception info tuple from sys.exc_info(), or None.
            func: Function name where log was created, or None.
            sinfo: Stack info string, or None.
            **kwargs: Additional keyword arguments. If 'structured_data' key
                is present, its value (dict) is stored; otherwise empty dict.

        Returns:
            None (constructor).

        Example:
            >>> record = StructuredLogRecord(
            ...     name="test", level=20, pathname="test.py", lineno=1,
            ...     msg="Captured", args=(), exc_info=None,
            ...     structured_data={"camera_id": 0, "exposure_ms": 100}
            ... )
            >>> record.structured_data
            {'camera_id': 0, 'exposure_ms': 100}
        """
        super().__init__(
            name, level, pathname, lineno, msg, args, exc_info, func, sinfo
        )
        # Store structured data from kwargs
        self.structured_data = kwargs.get("structured_data", {})


# =============================================================================
# Structured Logger
# =============================================================================


class StructuredLogger(logging.Logger):
    """Logger with structured data support.

    Extends standard Logger to accept keyword arguments that become
    structured data in the log record.

    Usage:
        logger = StructuredLogger("my.module")
        logger.info("User logged in", user_id=123, ip="192.168.1.1")
    """

    def debug(
        self,
        msg: object,
        *args: Any,
        exc_info: Any = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Log debug message with optional structured data kwargs."""
        if self.isEnabledFor(logging.DEBUG):
            self._log(
                logging.DEBUG,
                msg,
                args,
                exc_info=exc_info,
                extra=extra,
                stack_info=stack_info,
                stacklevel=stacklevel + 1,
                **kwargs,
            )

    def info(
        self,
        msg: object,
        *args: Any,
        exc_info: Any = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Log info message with optional structured data kwargs."""
        if self.isEnabledFor(logging.INFO):
            self._log(
                logging.INFO,
                msg,
                args,
                exc_info=exc_info,
                extra=extra,
                stack_info=stack_info,
                stacklevel=stacklevel + 1,
                **kwargs,
            )

    def warning(
        self,
        msg: object,
        *args: Any,
        exc_info: Any = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Log warning message with optional structured data kwargs."""
        if self.isEnabledFor(logging.WARNING):
            self._log(
                logging.WARNING,
                msg,
                args,
                exc_info=exc_info,
                extra=extra,
                stack_info=stack_info,
                stacklevel=stacklevel + 1,
                **kwargs,
            )

    def error(
        self,
        msg: object,
        *args: Any,
        exc_info: Any = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Log error message with optional structured data kwargs."""
        if self.isEnabledFor(logging.ERROR):
            self._log(
                logging.ERROR,
                msg,
                args,
                exc_info=exc_info,
                extra=extra,
                stack_info=stack_info,
                stacklevel=stacklevel + 1,
                **kwargs,
            )

    def critical(
        self,
        msg: object,
        *args: Any,
        exc_info: Any = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Log critical message with optional structured data kwargs."""
        if self.isEnabledFor(logging.CRITICAL):
            self._log(
                logging.CRITICAL,
                msg,
                args,
                exc_info=exc_info,
                extra=extra,
                stack_info=stack_info,
                stacklevel=stacklevel + 1,
                **kwargs,
            )

    def _log(
        self,
        level: int,
        msg: object,
        args: tuple[Any, ...] | MutableMapping[str, Any] | None = None,
        exc_info: Any = None,
        extra: dict[str, Any] | None = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        **kwargs: Any,
    ) -> None:
        """Log a message with structured data support.

        Overrides the standard Logger._log to capture keyword arguments as
        structured data. The structured data is merged with any active
        LogContext values, with explicit kwargs taking precedence. This
        enables rich, queryable logs for telescope operations.

        The merge order is: LogContext values < explicit kwargs, allowing
        operation-specific data to override ambient context.

        Args:
            level: Numeric log level (e.g., logging.DEBUG=10, INFO=20).
            msg: Log message, may contain % formatting placeholders.
            args: Arguments for % formatting, or None for no formatting.
            exc_info: Exception tuple from sys.exc_info(), True to capture
                current exception, or None for no exception info.
            extra: Additional context dict passed to LogRecord. The
                'structured_data' key will be added/overwritten.
            stack_info: If True, include stack trace in log.
            stacklevel: Stack frames to skip for determining caller info.
                Incremented by 1 internally for accurate source location.
            **kwargs: Arbitrary key-value pairs to include as structured
                data. Common keys: camera_id, exposure_us, duration_ms.

        Returns:
            None.

        Example:
            >>> logger = StructuredLogger("telescope_mcp.camera")
            >>> with LogContext(session_id="abc123"):
            ...     logger.info("Frame captured", camera_id=0, duration_ms=150)
            # Output includes: session_id=abc123 camera_id=0 duration_ms=150
        """
        # Merge context + kwargs into structured_data
        context = _log_context.get()
        structured_data = {**context, **kwargs}

        # Put structured_data in extra for the formatter
        if extra is None:
            extra = {}
        extra["structured_data"] = structured_data

        super()._log(
            level,
            msg,
            args,
            exc_info=exc_info,
            extra=extra,
            stack_info=stack_info,
            stacklevel=stacklevel + 1,
        )


# =============================================================================
# Formatters
# =============================================================================


class StructuredFormatter(logging.Formatter):
    """Human-readable formatter with structured data.

    Format: timestamp - name - level - message | key=value key=value
    """

    def __init__(
        self,
        fmt: str | None = None,
        datefmt: str | None = None,
        include_structured: bool = True,
    ) -> None:
        """Initialize the structured formatter.

        Creates a human-readable log formatter that appends structured
        key-value data after the main message. Designed for development
        and debugging where readability is prioritized over parseability.

        This formatter is the default for telescope-mcp development
        environments, providing readable console output while preserving
        structured metadata for debugging camera operations, timing
        analysis, and session tracking.

        Args:
            fmt: Format string using LogRecord attributes. If None, uses:
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'.
                See logging.Formatter for available attributes.
            datefmt: Date/time format string for %(asctime)s. If None,
                uses ISO 8601 format. Example: '%Y-%m-%d %H:%M:%S'.
            include_structured: If True (default), appends structured data
                as ' | key=value key=value' after the message. Set False
                to output only the base message format.

        Returns:
            None (constructor).

        Raises:
            None.

        Example:
            >>> formatter = StructuredFormatter(
            ...     fmt="%(levelname)s: %(message)s",
            ...     include_structured=True
            ... )
            >>> handler.setFormatter(formatter)
            # Output: "INFO: Camera connected | camera_id=0 name=ASI120"
        """
        if fmt is None:
            fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        super().__init__(fmt, datefmt)
        self.include_structured = include_structured

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as human-readable text with structured data.

        Produces output suitable for console viewing during development.
        The base message is formatted according to the configured format
        string, then structured data (if any) is appended as key=value
        pairs separated by ' | '.

        Structured data values are formatted via _format_value():
        - Strings with spaces are quoted
        - Dicts/lists become JSON
        - Other types use str()

        Args:
            record: The LogRecord to format. May have a 'structured_data'
                attribute (dict) containing key-value pairs to append.
                If missing or empty, only the base format is returned.

        Returns:
            Formatted log string. Examples:
            - Without structured: '2025-01-15 10:30:00 - app - INFO - Started'
            - With structured: '... - INFO - Captured | camera_id=0 ms=150'

        Raises:
            None. Gracefully handles missing structured_data attribute.

        Example:
            >>> formatter = StructuredFormatter()
            >>> record = logging.LogRecord(...)
            >>> record.structured_data = {"camera_id": 0}
            >>> formatter.format(record)
            '2025-01-15 10:30:00 - test - INFO - Message | camera_id=0'
        """
        base = super().format(record)

        if not self.include_structured:
            return base

        # Get structured data from record
        structured = getattr(record, "structured_data", {})
        if not structured:
            return base

        # Format as key=value pairs
        pairs = " ".join(f"{k}={_format_value(v)}" for k, v in structured.items())
        return f"{base} | {pairs}"


class JSONFormatter(logging.Formatter):
    """JSON formatter for log aggregation systems.

    Outputs each log record as a single JSON line with:
    - timestamp (ISO format)
    - level
    - logger name
    - message
    - All structured data as top-level keys
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as a single-line JSON object.

        Produces machine-parseable JSON output suitable for log aggregation
        systems (Elasticsearch, CloudWatch, etc.). Each log entry becomes
        one JSON line (NDJSON format) for easy ingestion and querying.

        The output JSON contains:
        - timestamp: ISO 8601 format in UTC (e.g., '2025-01-15T10:30:00+00:00')
        - level: Log level name (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        - logger: Logger name (e.g., 'telescope_mcp.devices.camera')
        - message: The formatted log message
        - exception: Stack trace string (only if exc_info is set)
        - ...plus all structured_data keys merged at top level

        Non-serializable values are converted via str() as fallback.

        Args:
            record: The LogRecord to format. The 'structured_data' attribute
                (if present) is merged into the output dict. Exception info
                from record.exc_info is formatted and included if present.

        Returns:
            Single-line JSON string with no trailing newline. Example:
            '{"timestamp":"2025-01-15T10:30:00+00:00","level":"INFO",...}'

        Raises:
            None. Uses json.dumps(default=str) for safe serialization.

        Example:
            >>> formatter = JSONFormatter()
            >>> record = logging.LogRecord(...)
            >>> record.structured_data = {"camera_id": 0, "exposure_us": 100000}
            >>> output = formatter.format(record)
            >>> json.loads(output)["camera_id"]
            0
        """
        # Base fields
        log_dict: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add structured data
        structured = getattr(record, "structured_data", {})
        log_dict.update(structured)

        # Add exception info if present
        if record.exc_info:
            log_dict["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_dict, default=str)


def _format_value(value: Any) -> str:
    """Format a value for human-readable structured log output.

    Converts arbitrary Python values to strings suitable for the
    key=value format used by StructuredFormatter. Ensures output
    is unambiguous and parseable while remaining human-readable.

    Formatting rules:
    - None: returns 'null' (JSON-compatible)
    - Strings: returned as-is, unless containing spaces (then quoted)
    - Dicts/lists: JSON-serialized for structure preservation
    - Other types: converted via str()

    Args:
        value: Any Python value to format. Common types include:
            str, int, float, bool, dict, list, None.

    Returns:
        String representation suitable for log output. Examples:
        - None → 'null'
        - 'hello' → 'hello'
        - 'hello world' → '"hello world"'
        - {'key': 1} → '{"key": 1}'
        - [1, 2, 3] → '[1, 2, 3]'
        - 42 → '42'

    Raises:
        None. Non-serializable objects fall back to str() representation.

    Example:
        >>> _format_value(None)
        'null'
        >>> _format_value("simple")
        'simple'
        >>> _format_value("has spaces")
        '"has spaces"'
        >>> _format_value({"camera": 0})
        '{"camera": 0}'
    """
    if value is None:
        return "null"
    if isinstance(value, str):
        # Quote strings with spaces
        if " " in value:
            return f'"{value}"'
        return value
    if isinstance(value, dict | list):
        return json.dumps(value, default=str)
    return str(value)


# =============================================================================
# Context Management
# =============================================================================


@dataclass
class LogContext:
    """Context manager for structured logging context.

    Adds key-value pairs to all log messages within the context.
    Thread-safe and supports nesting.

    Usage:
        with LogContext(request_id="abc", camera_id=0):
            logger.info("Processing")  # Includes request_id and camera_id

            with LogContext(step="capture"):
                logger.info("Capturing")  # Includes all three
    """

    _kwargs: dict[str, Any] = field(default_factory=dict, init=False, repr=True)
    _token: contextvars.Token[dict[str, Any]] | None = field(
        default=None, init=False, repr=False
    )

    def __init__(self, **kwargs: Any) -> None:
        """Initialize a logging context with key-value pairs.

        Creates a context manager that will inject the provided key-value
        pairs into all log messages emitted within its scope. Supports
        nesting, where inner contexts merge with (and can override) outer
        context values.

        Common use cases:
        - Request tracking: LogContext(request_id="abc123")
        - Camera operations: LogContext(camera_id=0, operation="capture")
        - Session scoping: LogContext(session_id=session.id)

        Args:
            **kwargs: Arbitrary key-value pairs to include in log context.
                Keys should be valid Python identifiers. Values can be any
                JSON-serializable type (str, int, float, bool, dict, list).

        Returns:
            None (constructor).

        Example:
            >>> ctx = LogContext(camera_id=0, operation="capture")
            >>> with ctx:
            ...     logger.info("Starting")  # includes camera_id, operation
        """
        self._kwargs = kwargs

    def __enter__(self) -> LogContext:
        """Enter the context and activate structured logging values.

        Merges this context's key-value pairs into the current logging
        context. Uses Python's contextvars for proper async/thread
        isolation. The previous context is preserved via token for
        restoration on exit.

        The merge uses {**current, **self._kwargs}, so values from this
        LogContext override any existing values with the same keys.

        Args:
            None (uses self._kwargs set during __init__).

        Returns:
            Self, allowing use as 'with LogContext(...) as ctx:'.

        Raises:
            None.

        Example:
            >>> with LogContext(camera_id=0) as ctx:
            ...     # All logs now include camera_id=0
            ...     logger.info("Processing")
        """
        current = _log_context.get()
        new_context = {**current, **self._kwargs}
        self._token = _log_context.set(new_context)
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit the context and restore the previous logging context.

        Uses the saved contextvars token to reset the log context to its
        state before this LogContext was entered. This ensures proper
        nesting behavior and cleanup even if exceptions occur.

        Args:
            *args: Exception info (exc_type, exc_val, exc_tb) from context
                manager protocol. Ignored; exceptions are not suppressed.

        Returns:
            None. Does not suppress exceptions (no return True).

        Raises:
            None.

        Example:
            >>> with LogContext(session_id="abc"):
            ...     pass  # session_id active here
            ... # session_id no longer in context
        """
        if self._token is not None:
            _log_context.reset(self._token)


# =============================================================================
# Configuration
# =============================================================================

# Track if logging has been configured
_configured = False
_config_lock = threading.Lock()


def configure_logging(
    level: int | str = logging.INFO,
    json_format: bool = False,
    stream: Any = None,
    include_structured: bool = True,
    force: bool = False,
) -> None:
    """Configure the telescope-mcp structured logging system.

    Initializes the logging infrastructure for the entire telescope-mcp
    package. Sets up handlers, formatters, and configures the root
    'telescope_mcp' logger. Should be called once at application startup.

    This function is idempotent: subsequent calls after the first have
    no effect unless force=True. The configuration is protected by a
    threading lock for safe concurrent initialization.

    The logging system provides structured data support via keyword
    arguments to log methods, enabling rich queryable logs for
    telescope operations monitoring.

    Args:
        level: Minimum log level to capture. Can be int (e.g., logging.DEBUG=10,
            logging.INFO=20) or string ('DEBUG', 'INFO', etc.). Default: INFO.
        json_format: If True, use JSONFormatter for machine-parseable output
            (NDJSON). If False (default), use StructuredFormatter for
            human-readable output with key=value pairs.
        stream: Output stream for logs. Default: sys.stderr. Can be any
            file-like object with write() method (e.g., sys.stdout, StringIO).
        include_structured: If True (default) and json_format=False, append
            structured data as ' | key=value' to log messages. Ignored when
            json_format=True (structured data always included in JSON).
        force: If True, reconfigure logging even if already configured.
            Use for testing or runtime reconfiguration. Default: False.

    Returns:
        None.

    Raises:
        None. Thread-safe; handles concurrent calls gracefully.

    Example:
        >>> # Development setup (human-readable)
        >>> configure_logging(level=logging.DEBUG)
        >>>
        >>> # Production setup (JSON for log aggregation)
        >>> configure_logging(level=logging.INFO, json_format=True)
        >>>
        >>> # Testing setup (capture to StringIO)
        >>> import io
        >>> buffer = io.StringIO()
        >>> configure_logging(stream=buffer, level=logging.DEBUG, force=True)
    """
    global _configured

    with _config_lock:
        if force:
            _reset_logging_impl()
        _configure_logging_impl(level, json_format, stream, include_structured)


def _configure_logging_impl(
    level: int | str = logging.INFO,
    json_format: bool = False,
    stream: Any = None,
    include_structured: bool = True,
) -> None:
    """Internal implementation of configure_logging (assumes lock is held)."""
    global _configured

    if _configured:
        return

    # Set our custom logger class as default
    logging.setLoggerClass(StructuredLogger)

    # Create handler
    if stream is None:
        stream = sys.stderr
    handler = logging.StreamHandler(stream)

    # Select formatter
    formatter: logging.Formatter
    if json_format:
        formatter = JSONFormatter()
    else:
        formatter = StructuredFormatter(include_structured=include_structured)
    handler.setFormatter(formatter)

    # Configure root logger for telescope_mcp
    root = logging.getLogger("telescope_mcp")
    root.setLevel(level)
    root.addHandler(handler)

    # Prevent propagation to root logger (avoid duplicate logs)
    root.propagate = False

    _configured = True


def _reset_logging_impl() -> None:
    """Internal implementation of reset_logging (assumes lock is held)."""
    global _configured

    # Remove all handlers from telescope_mcp logger
    root = logging.getLogger("telescope_mcp")
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()

    _configured = False


def reset_logging() -> None:
    """Reset the logging system to unconfigured state (for testing).

    Removes all handlers from the telescope_mcp logger and marks the
    system as unconfigured. The next call to configure_logging() or
    get_logger() will reinitialize the logging system.

    Warning:
        This is primarily for testing. Production code should not need
        to reset logging configuration.

    Args:
        None.

    Returns:
        None.

    Raises:
        None. Thread-safe.

    Example:
        >>> # In test teardown
        >>> reset_logging()
        >>> # Now can reconfigure with different settings
        >>> configure_logging(level=logging.DEBUG, json_format=True)
    """
    with _config_lock:
        _reset_logging_impl()


def get_logger(name: str) -> StructuredLogger:
    """Get a structured logger for the given module or component.

    Factory function to obtain a StructuredLogger instance configured
    for the telescope-mcp logging system. The returned logger supports
    keyword arguments that become structured data in log output.

    If configure_logging() hasn't been called yet, this function
    automatically configures logging with default settings (INFO level,
    human-readable format, stderr output).

    Logger names form a hierarchy (dot-separated), inheriting settings
    from parent loggers. Using __name__ ensures logs are properly
    attributed to their source module.

    Args:
        name: Logger name, typically __name__ for automatic module
            attribution. Examples: 'telescope_mcp.devices.camera',
            'telescope_mcp.tools.sessions'. The name appears in log
            output for traceability.

    Returns:
        StructuredLogger instance supporting keyword arguments:
            logger.info("message", key=value, ...)
        The logger inherits from the 'telescope_mcp' root logger.

    Raises:
        None. Automatically initializes logging if needed.

    Example:
        >>> # In telescope_mcp/devices/camera.py
        >>> logger = get_logger(__name__)
        >>> logger.info("Camera initialized", camera_id=0, model="ASI120")
        # Output: 2025-01-15 10:30:00 - telescope_mcp.devices.camera - INFO
        #         - Camera initialized | camera_id=0 model=ASI120
    """
    # Double-checked locking pattern for thread-safe lazy initialization
    if not _configured:
        with _config_lock:
            if not _configured:  # pragma: no branch
                _configure_logging_impl()

    # Get logger (will be StructuredLogger due to setLoggerClass in configure)
    logger = logging.getLogger(name)

    # Explicit cast: logging.getLogger returns Logger, but setLoggerClass()
    # ensures it's actually a StructuredLogger. Cast documents this intent.
    return cast(StructuredLogger, logger)


# =============================================================================
# Session Integration
# =============================================================================

# Protocol for session manager (avoids circular import)
from collections.abc import Callable  # noqa: E402
from typing import Protocol, runtime_checkable  # noqa: E402


@runtime_checkable
class SessionManagerProtocol(Protocol):  # pragma: no cover
    """Protocol for session manager log method."""

    def log(
        self,
        level: str,
        message: str,
        source: str,
        **kwargs: Any,
    ) -> None:
        """Log a message to the session."""
        ...  # pragma: no cover


class SessionLogHandler(logging.Handler):
    """Handler that forwards logs to the active session.

    Enables dual-write: logs go to both console and session storage.
    Includes recursion guard to prevent infinite loops if session
    manager logging triggers additional log events.

    Usage:
        from telescope_mcp.drivers.config import get_session_manager

        handler = SessionLogHandler(get_session_manager)
        logging.getLogger("telescope_mcp").addHandler(handler)
    """

    # Thread-local recursion guard to prevent infinite loops
    _local = threading.local()

    def __init__(
        self,
        session_manager_getter: Callable[[], SessionManagerProtocol | None],
        level: int = logging.NOTSET,
    ) -> None:
        """Initialize the session log handler with a manager accessor.

        Uses a getter function (rather than direct reference) to support
        lazy initialization and avoid circular imports. The getter is
        called on each emit() to get the current session manager.

        This design enables:
        - Handler creation before session manager exists
        - Dynamic session switching during runtime
        - Graceful handling when no session is active (getter returns None)

        Args:
            session_manager_getter: Callable that returns the current
                SessionManager instance, or None if no session is active.
                Typically: lambda: get_session_manager() or similar.
            level: Minimum log level to forward to sessions. Default:
                logging.NOTSET (0) forwards all levels. Use logging.INFO
                to skip DEBUG messages in session storage.

        Returns:
            None (constructor).

        Raises:
            None.

        Example:
            >>> from telescope_mcp.drivers.config import get_session_manager
            >>> handler = SessionLogHandler(
            ...     session_manager_getter=get_session_manager,
            ...     level=logging.INFO
            ... )
            >>> logging.getLogger("telescope_mcp").addHandler(handler)
        """
        super().__init__(level)
        self._get_manager = session_manager_getter

    def emit(self, record: logging.LogRecord) -> None:
        """Forward a log record to the active observation session.

        Implements the logging.Handler protocol to capture log entries
        and persist them in the current telescope observation session.
        This enables session replay, debugging, and audit trails.

        The handler extracts structured data from the record and passes
        it to the session manager's log() method. If no session is active
        (manager returns None), the record is silently skipped.

        Includes a recursion guard to prevent infinite loops when the
        session manager's log() method triggers additional log events.

        Following Python logging best practices, exceptions during emit
        are caught and passed to handleError() rather than propagated,
        ensuring logging failures never crash the application.

        Args:
            record: LogRecord to forward. Expected to have:
                - levelname: Log level string (INFO, ERROR, etc.)
                - name: Logger name (source module)
                - getMessage(): Formatted message
                - structured_data (optional): Dict of key-value metadata

        Returns:
            None.

        Raises:
            None. Exceptions are caught and passed to self.handleError().

        Example:
            >>> handler = SessionLogHandler(get_session_manager)
            >>> logging.getLogger("telescope_mcp").addHandler(handler)
            >>> # Now all telescope_mcp logs are also saved to sessions
        """
        # Recursion guard: skip if we're already emitting in this thread
        if getattr(self._local, "emitting", False):
            return

        try:
            self._local.emitting = True

            manager = self._get_manager()
            if manager is None:
                return

            # Get structured data
            structured = getattr(record, "structured_data", {})

            # Forward to session
            manager.log(
                level=record.levelname,
                message=record.getMessage(),
                source=record.name,
                **structured,
            )
        except Exception:
            # Don't raise exceptions in logging
            self.handleError(record)
        finally:
            self._local.emitting = False
