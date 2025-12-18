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
from datetime import datetime, timezone
from typing import Any, ClassVar

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
        """Override _log to handle structured data.
        
        Accepts keyword arguments as structured data and merges with
        context from LogContext.
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
        if fmt is None:
            fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        super().__init__(fmt, datefmt)
        self.include_structured = include_structured
    
    def format(self, record: logging.LogRecord) -> str:
        """Format record with structured data appended."""
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
        """Format record as JSON."""
        # Base fields
        log_dict: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
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
    """Format a value for human-readable output."""
    if isinstance(value, str):
        # Quote strings with spaces
        if " " in value:
            return f'"{value}"'
        return value
    if isinstance(value, (dict, list)):
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
    
    _token: contextvars.Token[dict[str, Any]] | None = field(
        default=None, init=False, repr=False
    )
    
    def __init__(self, **kwargs: Any) -> None:
        """Initialize with context key-value pairs."""
        self._kwargs = kwargs
        self._token = None
    
    def __enter__(self) -> LogContext:
        """Enter context and add to log context."""
        current = _log_context.get()
        new_context = {**current, **self._kwargs}
        self._token = _log_context.set(new_context)
        return self
    
    def __exit__(self, *args: Any) -> None:
        """Exit context and restore previous context."""
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
) -> None:
    """Configure the telescope-mcp logging system.
    
    Call once at application startup. Safe to call multiple times
    (subsequent calls are no-ops unless force=True).
    
    Args:
        level: Log level (default: INFO)
        json_format: Use JSON formatter for production (default: False)
        stream: Output stream (default: sys.stderr)
        include_structured: Include structured data in output (default: True)
    """
    global _configured
    
    with _config_lock:
        if _configured:
            return
        
        # Set our custom logger class as default
        logging.setLoggerClass(StructuredLogger)
        
        # Create handler
        if stream is None:
            stream = sys.stderr
        handler = logging.StreamHandler(stream)
        
        # Select formatter
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


def get_logger(name: str) -> StructuredLogger:
    """Get a structured logger for the given name.
    
    Returns a StructuredLogger that supports keyword arguments
    for structured data.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        StructuredLogger instance
    
    Example:
        logger = get_logger(__name__)
        logger.info("Connected", camera_id=0)
    """
    # Ensure logging is configured with defaults if not already
    if not _configured:
        configure_logging()
    
    # Get logger (will be StructuredLogger due to setLoggerClass)
    logger = logging.getLogger(name)
    
    # Cast for type checker (it's actually a StructuredLogger)
    return logger  # type: ignore[return-value]


# =============================================================================
# Session Integration
# =============================================================================

class SessionLogHandler(logging.Handler):
    """Handler that forwards logs to the active session.
    
    Enables dual-write: logs go to both console and session storage.
    
    Usage:
        from telescope_mcp.drivers.config import get_session_manager
        
        handler = SessionLogHandler(get_session_manager)
        logging.getLogger("telescope_mcp").addHandler(handler)
    """
    
    def __init__(
        self,
        session_manager_getter: Any,  # Callable[[], SessionManager]
        level: int = logging.NOTSET,
    ) -> None:
        """Initialize with session manager getter.
        
        Args:
            session_manager_getter: Callable that returns SessionManager
            level: Minimum level to forward to session
        """
        super().__init__(level)
        self._get_manager = session_manager_getter
    
    def emit(self, record: logging.LogRecord) -> None:
        """Forward log record to session."""
        try:
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
