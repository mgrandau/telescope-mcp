"""MCP Tools for session management.

Provides tools to start, end, and query telescope sessions.
Sessions are the core abstraction for telescope data storage.
"""

import json
from typing import Any

from mcp.server import Server
from mcp.types import TextContent, Tool

from telescope_mcp.data import LogLevel, SessionType
from telescope_mcp.drivers.config import get_factory, get_session_manager
from telescope_mcp.observability import get_logger

logger = get_logger(__name__)


# Tool definitions
TOOLS = [
    Tool(
        name="start_session",
        description="Start a new telescope session. Closes any existing session and creates a new one of the specified type.",
        inputSchema={
            "type": "object",
            "properties": {
                "session_type": {
                    "type": "string",
                    "enum": ["observation", "alignment", "experiment", "maintenance"],
                    "description": "Type of session to start",
                },
                "target": {
                    "type": "string",
                    "description": "Target object name (e.g., 'M31') - used for observation sessions",
                },
                "purpose": {
                    "type": "string",
                    "description": "Purpose description - used for alignment/experiment sessions",
                },
            },
            "required": ["session_type"],
        },
    ),
    Tool(
        name="end_session",
        description="End the current session and write the ASDF file. Returns to idle session automatically.",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
    Tool(
        name="get_session_info",
        description="Get information about the currently active session",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
    Tool(
        name="session_log",
        description="Log a message to the current session",
        inputSchema={
            "type": "object",
            "properties": {
                "level": {
                    "type": "string",
                    "enum": ["DEBUG", "INFO", "WARNING", "ERROR"],
                    "description": "Log level",
                    "default": "INFO",
                },
                "message": {
                    "type": "string",
                    "description": "Log message",
                },
                "source": {
                    "type": "string",
                    "description": "Source component name",
                    "default": "user",
                },
            },
            "required": ["message"],
        },
    ),
    Tool(
        name="session_event",
        description="Record a significant event in the current session (e.g., 'tracking_lost', 'cloud_detected')",
        inputSchema={
            "type": "object",
            "properties": {
                "event": {
                    "type": "string",
                    "description": "Event name",
                },
                "details": {
                    "type": "object",
                    "description": "Additional event details",
                    "additionalProperties": True,
                },
            },
            "required": ["event"],
        },
    ),
    Tool(
        name="get_data_dir",
        description="Get the current data directory path where session files are stored",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
    Tool(
        name="set_data_dir",
        description="Set the data directory path for session storage",
        inputSchema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the data directory",
                },
            },
            "required": ["path"],
        },
    ),
]


def register(server: Server) -> None:
    """Register session management tools with the MCP server.

    Attaches session-related tools to the MCP server instance, enabling
    AI assistants to manage telescope observation sessions. Sessions are
    the core data abstraction for organizing telescope data.

    Tools registered:
    - start_session: Begin new observation/alignment/experiment session
    - end_session: Close session and write ASDF file
    - get_session_info: Query active session status
    - session_log: Add log entries to session
    - session_event: Record significant events
    - get_data_dir / set_data_dir: Manage storage location

    Args:
        server: MCP Server instance to register tools with. Must be
            initialized but not yet running.

    Returns:
        None. Modifies server in-place by adding handlers.

    Raises:
        None. Registration itself doesn't access storage.

    Example:
        >>> server = Server("telescope-mcp")
        >>> register(server)
    """

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """Return the list of available session tools.

        MCP handler that provides tool definitions to clients.

        Returns:
            List of Tool objects defining session capabilities.
        """
        return TOOLS

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        """Route tool calls to appropriate session implementations.

        MCP handler that dispatches incoming tool calls based on name.
        Primary entry point for all session management operations from
        AI agents through the Model Context Protocol.
        
        Business context: Enables AI agents to manage observation sessions,
        tracking all data capture, events, and metadata in structured ASDF
        files. Sessions provide context for multi-frame captures, enabling
        correlation of frames with environmental conditions, equipment settings,
        and observation goals. Critical for scientific reproducibility and
        automated data organization in telescope operations.

        Args:
            name: Tool name from TOOLS definitions (start_session, end_session,
                get_session_info, session_log, session_event, get/set_data_dir).
            arguments: Dict of arguments matching tool's inputSchema. Validated
                by MCP framework before dispatch.

        Returns:
            List containing single TextContent with JSON result string or error
            message. Success responses contain structured JSON with session data,
            errors contain descriptive text.
        
        Raises:
            None. All errors are caught and returned as TextContent with details.
        
        Example:
            # AI agent starts an imaging session
            result = await call_tool(
                "start_session",
                {"session_type": "imaging", "target": "M31"}
            )
            # Returns: [TextContent(text='{"session_id": "...", ...}')]
        """
        if name == "start_session":
            return await _start_session(
                arguments["session_type"],
                arguments.get("target"),
                arguments.get("purpose"),
            )
        elif name == "end_session":
            return await _end_session()
        elif name == "get_session_info":
            return await _get_session_info()
        elif name == "session_log":
            return await _session_log(
                arguments.get("level", "INFO"),
                arguments["message"],
                arguments.get("source", "user"),
            )
        elif name == "session_event":
            return await _session_event(
                arguments["event"],
                arguments.get("details", {}),
            )
        elif name == "get_data_dir":
            return await _get_data_dir()
        elif name == "set_data_dir":
            return await _set_data_dir(arguments["path"])
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def _start_session(
    session_type: str,
    target: str | None,
    purpose: str | None,
) -> list[TextContent]:
    """Start a new telescope session of the specified type.

    Creates a new session, automatically closing any existing active
    session first. Sessions organize telescope data collection and
    provide context for captured frames, events, and telemetry.

    Session types determine data organization and metadata:
    - observation: Target-focused imaging (requires target name)
    - alignment: Telescope/mount calibration (uses purpose)
    - experiment: Testing and development (uses purpose)
    - maintenance: System maintenance (uses purpose)

    Args:
        session_type: One of 'observation', 'alignment', 'experiment',
            'maintenance'. Case-insensitive. Cannot be 'idle'.
        target: Target object name for observation sessions (e.g., 'M31',
            'Jupiter'). Optional for other session types.
        purpose: Description of session purpose for non-observation
            sessions. Optional but recommended.

    Returns:
        List with TextContent containing JSON:
        {"status": "started", "session_id": str, "session_type": str,
         "target": str|null, "purpose": str|null, "start_time": str}
        Returns error message for invalid session_type or exceptions.

    Raises:
        None. Exceptions caught and returned as error text.

    Example:
        >>> result = await _start_session("observation", "M31", None)
        >>> data = json.loads(result[0].text)
        >>> data["session_id"]
        "obs_20250115_103000"
    """
    try:
        manager = get_session_manager()
        
        # Validate session type
        try:
            st = SessionType(session_type.lower())
        except ValueError:
            valid = [t.value for t in SessionType if t != SessionType.IDLE]
            return [TextContent(
                type="text",
                text=f"Invalid session type: {session_type}. Valid types: {valid}"
            )]
        
        # Don't allow starting idle sessions manually
        if st == SessionType.IDLE:
            return [TextContent(
                type="text",
                text="Cannot manually start an idle session. Use end_session() to return to idle."
            )]
        
        session = manager.start_session(st, target=target, purpose=purpose)
        
        result = {
            "status": "started",
            "session_id": session.session_id,
            "session_type": session.session_type.value,
            "target": session.target,
            "purpose": session.purpose,
            "start_time": session.start_time.isoformat(),
        }
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    except Exception as e:
        logger.error(f"Error starting session: {e}")
        return [TextContent(type="text", text=f"Error starting session: {e}")]


async def _end_session() -> list[TextContent]:
    """End the current active session and persist data.

    Closes the active session, writes all collected data to an ASDF
    file, and automatically starts an idle session. The ASDF file
    contains frames, logs, events, telemetry, and metadata.

    Cannot end an idle session (no-op with informative message).
    Session files are named with session ID and stored in the
    configured data directory.

    Args:
        None.

    Returns:
        List with TextContent containing JSON:
        {"status": "ended", "session_id": str, "session_type": str,
         "file_path": str}
        Returns message if already in idle state.
        Returns error message on file write failures.

    Raises:
        None. Exceptions caught and returned as error text.

    Example:
        >>> result = await _end_session()
        >>> data = json.loads(result[0].text)
        >>> data["file_path"]
        "/data/telescope/obs_20250115_103000.asdf"
    """
    try:
        manager = get_session_manager()
        
        session_id = manager.active_session_id
        session_type = manager.active_session_type
        
        if session_type == SessionType.IDLE:
            return [TextContent(
                type="text",
                text="No active session to end (currently in idle)"
            )]
        
        path = manager.end_session()
        
        result = {
            "status": "ended",
            "session_id": session_id,
            "session_type": session_type.value if session_type else None,
            "file_path": str(path),
        }
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    except Exception as e:
        logger.error(f"Error ending session: {e}")
        return [TextContent(type="text", text=f"Error ending session: {e}")]


async def _get_session_info() -> list[TextContent]:
    """Get information about the currently active session.

    Returns detailed status of the active session including type,
    target, duration, and accumulated metrics. Works for both
    active sessions and idle state.

    Metrics include frame count, log entries, events, and error/
    warning counts - useful for monitoring session health.

    Args:
        None.

    Returns:
        List with TextContent containing JSON:
        {"session_id": str, "session_type": str, "target": str|null,
         "purpose": str|null, "start_time": str, "duration_seconds": float,
         "is_idle": bool, "metrics": {"frames_captured": int,
         "log_entries": int, "events": int, "errors": int, "warnings": int}}
        Returns "No active session" if session manager unavailable.

    Raises:
        None. Exceptions caught and returned as error text.

    Example:
        >>> result = await _get_session_info()
        >>> data = json.loads(result[0].text)
        >>> print(f"Duration: {data['duration_seconds']:.1f}s")
    """
    try:
        manager = get_session_manager()
        session = manager.active_session
        
        if session is None:
            return [TextContent(type="text", text="No active session")]
        
        result = {
            "session_id": session.session_id,
            "session_type": session.session_type.value,
            "target": session.target,
            "purpose": session.purpose,
            "start_time": session.start_time.isoformat(),
            "duration_seconds": session.duration_seconds,
            "is_idle": session.session_type == SessionType.IDLE,
            "metrics": {
                "frames_captured": session._frames_captured,
                "log_entries": len(session._logs),
                "events": len(session._events),
                "errors": session._error_count,
                "warnings": session._warning_count,
            },
        }
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    except Exception as e:
        logger.error(f"Error getting session info: {e}")
        return [TextContent(type="text", text=f"Error getting session info: {e}")]


async def _session_log(level: str, message: str, source: str) -> list[TextContent]:
    """Log a message to the current telescope session.

    Adds a timestamped log entry to the active session's log collection.
    Logs are persisted in the ASDF file when the session ends, providing
    an audit trail of operations and observations.

    Use for recording observations, decisions, anomalies, or any
    information relevant to the session that should be preserved.

    Args:
        level: Log severity, one of 'DEBUG', 'INFO', 'WARNING', 'ERROR'.
            Case-insensitive. INFO is typical for user messages.
        message: Log message text. Can include any relevant details.
            No length limit but keep reasonable for readability.
        source: Identifier for the log source component. Use 'user'
            for manual entries, or component name for automated logs.

    Returns:
        List with TextContent containing JSON:
        {"status": "logged", "level": str, "message": str,
         "source": str, "session_id": str}
        Returns error message for invalid level or exceptions.

    Raises:
        None. Exceptions caught and returned as error text.

    Example:
        >>> result = await _session_log("INFO", "Started M31 imaging", "user")
    """
    try:
        manager = get_session_manager()
        
        try:
            log_level = LogLevel(level.upper())
        except ValueError:
            valid = [l.value for l in LogLevel]
            return [TextContent(
                type="text",
                text=f"Invalid log level: {level}. Valid levels: {valid}"
            )]
        
        manager.log(log_level, message, source=source)
        
        result = {
            "status": "logged",
            "level": log_level.value,
            "message": message,
            "source": source,
            "session_id": manager.active_session_id,
        }
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    except Exception as e:
        logger.error(f"Error logging: {e}")
        return [TextContent(type="text", text=f"Error logging: {e}")]


async def _session_event(event: str, details: dict[str, Any]) -> list[TextContent]:
    """Record a significant event in the current session.

    Events are discrete occurrences during a session that warrant
    special attention in analysis. Unlike logs (continuous stream),
    events mark specific moments with structured metadata.

    Common events: 'tracking_lost', 'cloud_detected', 'focus_changed',
    'meridian_flip', 'filter_changed', 'guide_star_acquired'.

    Args:
        event: Event type/name string. Use consistent naming for
            analysis (e.g., snake_case category_action format).
        details: Additional event metadata as key-value pairs.
            Contents are event-specific. Example for tracking_lost:
            {"duration_seconds": 30, "recovery_action": "manual"}.

    Returns:
        List with TextContent containing JSON:
        {"status": "recorded", "event": str, "details": dict,
         "session_id": str}
        Returns error message on exceptions.

    Raises:
        None. Exceptions caught and returned as error text.

    Example:
        >>> result = await _session_event(
        ...     "cloud_detected",
        ...     {"coverage_percent": 40, "action": "paused_capture"}
        ... )
    """
    try:
        manager = get_session_manager()
        manager.add_event(event, **details)
        
        result = {
            "status": "recorded",
            "event": event,
            "details": details,
            "session_id": manager.active_session_id,
        }
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    except Exception as e:
        logger.error(f"Error recording event: {e}")
        return [TextContent(type="text", text=f"Error recording event: {e}")]


async def _get_data_dir() -> list[TextContent]:
    """Get the current data directory path for session storage.

    Returns the configured path where session ASDF files are stored.
    This directory contains all telescope data organized by session.

    The path can be changed with set_data_dir for different storage
    locations (e.g., different drives for different observation runs).

    Args:
        None.

    Returns:
        List with TextContent containing JSON:
        {"data_dir": str, "exists": bool}
        Returns error message on configuration access failure.

    Raises:
        None. Exceptions caught and returned as error text.

    Example:
        >>> result = await _get_data_dir()
        >>> data = json.loads(result[0].text)
        >>> data["data_dir"]
        "/home/user/telescope-data"
    """
    try:
        factory = get_factory()
        data_dir = factory.config.data_dir
        
        result = {
            "data_dir": str(data_dir),
            "exists": data_dir.exists(),
        }
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    except Exception as e:
        logger.error(f"Error getting data dir: {e}")
        return [TextContent(type="text", text=f"Error getting data dir: {e}")]


async def _set_data_dir(path: str) -> list[TextContent]:
    """Set the data directory path for session storage.

    Changes where session ASDF files are saved. The directory is
    created if it doesn't exist. Changing the directory resets
    the session manager, starting a fresh idle session.

    Use this to direct data to different drives or organize data
    by observation campaign. Previous sessions are not moved.

    Args:
        path: Absolute filesystem path for data storage. The path
            will be created if it doesn't exist. Must be writable.

    Returns:
        List with TextContent containing JSON:
        {"status": "updated", "data_dir": str, "exists": bool,
         "note": str (about session manager reset)}
        Returns error message on invalid path or permission errors.

    Raises:
        None. Exceptions caught and returned as error text.

    Example:
        >>> result = await _set_data_dir("/mnt/astro/tonight")
        >>> data = json.loads(result[0].text)
        >>> data["status"]
        "updated"
    """
    try:
        from pathlib import Path
        from telescope_mcp.drivers.config import set_data_dir
        
        new_path = Path(path)
        set_data_dir(new_path)
        
        result = {
            "status": "updated",
            "data_dir": str(new_path),
            "exists": new_path.exists(),
            "note": "Session manager has been reset. A new idle session will start on next operation.",
        }
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    except Exception as e:
        logger.error(f"Error setting data dir: {e}")
        return [TextContent(type="text", text=f"Error setting data dir: {e}")]
