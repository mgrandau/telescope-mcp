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
    """Register session tools with the MCP server."""

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return TOOLS

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
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
    """Start a new session."""
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
    """End the current session."""
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
    """Get current session info."""
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
    """Log a message to the current session."""
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
    """Record an event in the current session."""
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
    """Get the current data directory."""
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
    """Set the data directory."""
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
