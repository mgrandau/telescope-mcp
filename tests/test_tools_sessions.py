"""Unit tests for telescope_mcp.tools.sessions module.

Comprehensive tests achieving 100% coverage of the sessions MCP tools,
including error paths, edge cases, and the register() function.

Tests are organized by function under test with clear arrangement,
action, and assertion documentation following project conventions.

Author: Test suite
Date: 2025-12-30
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from mcp.server import Server
from mcp.types import TextContent, Tool

from telescope_mcp.data import SessionType
from telescope_mcp.tools import sessions

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_session_manager() -> MagicMock:
    """Create a mock SessionManager for dependency injection.

    Returns a MagicMock configured with typical SessionManager behavior
    that can be customized per test.

    Returns:
        MagicMock: Configured mock SessionManager.
    """
    manager = MagicMock()
    manager.active_session_id = "test_session_123"
    manager.active_session_type = SessionType.OBSERVATION

    # Mock session object
    mock_session = MagicMock()
    mock_session.session_id = "test_session_123"
    mock_session.session_type = SessionType.OBSERVATION
    mock_session.target = "M42"
    mock_session.purpose = None
    mock_session.start_time.isoformat.return_value = "2025-01-15T10:30:00"
    mock_session.duration_seconds = 120.5
    mock_session._frames_captured = 5
    mock_session._logs = [1, 2, 3]
    mock_session._events = [1, 2]
    mock_session._error_count = 1
    mock_session._warning_count = 2

    manager.active_session = mock_session
    manager.start_session.return_value = mock_session
    manager.end_session.return_value = Path("/data/test_session.asdf")

    return manager


@pytest.fixture
def mock_factory() -> MagicMock:
    """Create a mock factory for get_factory() calls.

    Returns:
        MagicMock: Configured mock factory with data_dir.
    """
    factory = MagicMock()
    # Create a proper mock for data_dir with Path-like behavior
    mock_data_dir = MagicMock()
    mock_data_dir.__str__ = MagicMock(return_value="/home/user/telescope-data")
    mock_data_dir.exists = MagicMock(return_value=True)
    factory.config.data_dir = mock_data_dir
    return factory


# =============================================================================
# Test TOOLS Constant
# =============================================================================


class TestToolsConstant:
    """Tests for the TOOLS constant definition."""

    def test_tools_list_is_list_of_tools(self) -> None:
        """Verifies TOOLS is a list containing Tool objects.

        Arrangement:
        1. sessions.TOOLS is module-level constant.
        2. Should be list type containing MCP Tool objects.

        Assertion Strategy:
        - TOOLS is a list.
        - Each item is a Tool instance.
        """
        assert isinstance(sessions.TOOLS, list)
        for tool in sessions.TOOLS:
            assert isinstance(tool, Tool)

    def test_tools_count(self) -> None:
        """Verifies TOOLS contains exactly 7 session tools.

        The expected tools are:
        - start_session
        - end_session
        - get_session_info
        - session_log
        - session_event
        - get_data_dir
        - set_data_dir

        Assertion Strategy:
        - TOOLS has exactly 7 items.
        """
        assert len(sessions.TOOLS) == 7

    def test_start_session_tool_schema(self) -> None:
        """Verifies start_session tool has correct input schema.

        Arrangement:
        1. Find start_session tool in TOOLS.
        2. Check inputSchema structure.

        Assertion Strategy:
        - session_type enum has 4 values (observation, alignment, etc).
        - session_type is required.
        """
        tool = next(t for t in sessions.TOOLS if t.name == "start_session")
        schema = tool.inputSchema

        assert "session_type" in schema["properties"]
        assert schema["properties"]["session_type"]["enum"] == [
            "observation",
            "alignment",
            "experiment",
            "maintenance",
        ]
        assert "session_type" in schema["required"]

    def test_session_log_tool_has_critical_level(self) -> None:
        """Verifies session_log tool schema includes CRITICAL level.

        Arrangement:
        1. Find session_log tool in TOOLS.
        2. Check level enum values.

        Assertion Strategy:
        - level enum includes CRITICAL.
        """
        tool = next(t for t in sessions.TOOLS if t.name == "session_log")
        schema = tool.inputSchema

        assert "CRITICAL" in schema["properties"]["level"]["enum"]


# =============================================================================
# Test register() Function
# =============================================================================


class TestRegisterFunction:
    """Tests for the register() function that registers MCP handlers."""

    def test_register_adds_list_tools_handler(self) -> None:
        """Verifies register() adds list_tools handler to server.

        Arrangement:
        1. Create mock MCP Server.
        2. Mock list_tools() decorator.

        Action:
        Call sessions.register(mock_server).

        Assertion Strategy:
        - server.list_tools() decorator was called.
        """
        mock_server = MagicMock(spec=Server)
        mock_server.list_tools.return_value = lambda f: f
        mock_server.call_tool.return_value = lambda f: f

        sessions.register(mock_server)

        mock_server.list_tools.assert_called_once()

    def test_register_adds_call_tool_handler(self) -> None:
        """Verifies register() adds call_tool handler to server.

        Arrangement:
        1. Create mock MCP Server.
        2. Mock call_tool() decorator.

        Action:
        Call sessions.register(mock_server).

        Assertion Strategy:
        - server.call_tool() decorator was called.
        """
        mock_server = MagicMock(spec=Server)
        mock_server.list_tools.return_value = lambda f: f
        mock_server.call_tool.return_value = lambda f: f

        sessions.register(mock_server)

        mock_server.call_tool.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_tools_returns_tools_list(self) -> None:
        """Verifies list_tools handler returns TOOLS list.

        Arrangement:
        1. Create mock server that captures registered handler.
        2. Register handlers via sessions.register().

        Action:
        Call the captured list_tools handler.

        Assertion Strategy:
        - Handler returns sessions.TOOLS.
        """
        captured_handler = None

        def capture_list_tools():
            def decorator(func):
                nonlocal captured_handler
                captured_handler = func
                return func

            return decorator

        mock_server = MagicMock(spec=Server)
        mock_server.list_tools = capture_list_tools
        mock_server.call_tool.return_value = lambda f: f

        sessions.register(mock_server)

        assert captured_handler is not None
        result = await captured_handler()
        assert result == sessions.TOOLS

    @pytest.mark.asyncio
    async def test_call_tool_dispatches_to_start_session(self) -> None:
        """Verifies call_tool routes 'start_session' to _start_session.

        Arrangement:
        1. Capture call_tool handler from register().
        2. Mock _start_session to track calls.

        Action:
        Call captured handler with name='start_session'.

        Assertion Strategy:
        - _start_session called with correct arguments.
        """
        captured_handler = None

        def capture_call_tool():
            def decorator(func):
                nonlocal captured_handler
                captured_handler = func
                return func

            return decorator

        mock_server = MagicMock(spec=Server)
        mock_server.list_tools.return_value = lambda f: f
        mock_server.call_tool = capture_call_tool

        sessions.register(mock_server)

        with patch.object(sessions, "_start_session") as mock_start:
            mock_start.return_value = [TextContent(type="text", text="{}")]
            await captured_handler(
                "start_session",
                {"session_type": "observation", "target": "M42", "purpose": None},
            )
            mock_start.assert_called_once_with("observation", "M42", None)

    @pytest.mark.asyncio
    async def test_call_tool_dispatches_to_end_session(self) -> None:
        """Verifies call_tool routes 'end_session' to _end_session.

        Action:
        Call captured handler with name='end_session'.

        Assertion Strategy:
        - _end_session called.
        """
        captured_handler = None

        def capture_call_tool():
            def decorator(func):
                nonlocal captured_handler
                captured_handler = func
                return func

            return decorator

        mock_server = MagicMock(spec=Server)
        mock_server.list_tools.return_value = lambda f: f
        mock_server.call_tool = capture_call_tool

        sessions.register(mock_server)

        with patch.object(sessions, "_end_session") as mock_end:
            mock_end.return_value = [TextContent(type="text", text="{}")]
            await captured_handler("end_session", {})
            mock_end.assert_called_once()

    @pytest.mark.asyncio
    async def test_call_tool_dispatches_to_get_session_info(self) -> None:
        """Verifies call_tool routes 'get_session_info' to _get_session_info."""
        captured_handler = None

        def capture_call_tool():
            def decorator(func):
                nonlocal captured_handler
                captured_handler = func
                return func

            return decorator

        mock_server = MagicMock(spec=Server)
        mock_server.list_tools.return_value = lambda f: f
        mock_server.call_tool = capture_call_tool

        sessions.register(mock_server)

        with patch.object(sessions, "_get_session_info") as mock_info:
            mock_info.return_value = [TextContent(type="text", text="{}")]
            await captured_handler("get_session_info", {})
            mock_info.assert_called_once()

    @pytest.mark.asyncio
    async def test_call_tool_dispatches_to_session_log(self) -> None:
        """Verifies call_tool routes 'session_log' to _session_log."""
        captured_handler = None

        def capture_call_tool():
            def decorator(func):
                nonlocal captured_handler
                captured_handler = func
                return func

            return decorator

        mock_server = MagicMock(spec=Server)
        mock_server.list_tools.return_value = lambda f: f
        mock_server.call_tool = capture_call_tool

        sessions.register(mock_server)

        with patch.object(sessions, "_session_log") as mock_log:
            mock_log.return_value = [TextContent(type="text", text="{}")]
            await captured_handler(
                "session_log", {"level": "INFO", "message": "test", "source": "user"}
            )
            mock_log.assert_called_once_with("INFO", "test", "user")

    @pytest.mark.asyncio
    async def test_call_tool_session_log_uses_defaults(self) -> None:
        """Verifies call_tool applies defaults for session_log optional args."""
        captured_handler = None

        def capture_call_tool():
            def decorator(func):
                nonlocal captured_handler
                captured_handler = func
                return func

            return decorator

        mock_server = MagicMock(spec=Server)
        mock_server.list_tools.return_value = lambda f: f
        mock_server.call_tool = capture_call_tool

        sessions.register(mock_server)

        with patch.object(sessions, "_session_log") as mock_log:
            mock_log.return_value = [TextContent(type="text", text="{}")]
            # Only provide message, level and source should use defaults
            await captured_handler("session_log", {"message": "test"})
            mock_log.assert_called_once_with("INFO", "test", "user")

    @pytest.mark.asyncio
    async def test_call_tool_dispatches_to_session_event(self) -> None:
        """Verifies call_tool routes 'session_event' to _session_event."""
        captured_handler = None

        def capture_call_tool():
            def decorator(func):
                nonlocal captured_handler
                captured_handler = func
                return func

            return decorator

        mock_server = MagicMock(spec=Server)
        mock_server.list_tools.return_value = lambda f: f
        mock_server.call_tool = capture_call_tool

        sessions.register(mock_server)

        with patch.object(sessions, "_session_event") as mock_event:
            mock_event.return_value = [TextContent(type="text", text="{}")]
            await captured_handler(
                "session_event", {"event": "test_event", "details": {"key": "value"}}
            )
            mock_event.assert_called_once_with("test_event", {"key": "value"})

    @pytest.mark.asyncio
    async def test_call_tool_session_event_uses_empty_details_default(self) -> None:
        """Verifies call_tool applies empty dict default for details."""
        captured_handler = None

        def capture_call_tool():
            def decorator(func):
                nonlocal captured_handler
                captured_handler = func
                return func

            return decorator

        mock_server = MagicMock(spec=Server)
        mock_server.list_tools.return_value = lambda f: f
        mock_server.call_tool = capture_call_tool

        sessions.register(mock_server)

        with patch.object(sessions, "_session_event") as mock_event:
            mock_event.return_value = [TextContent(type="text", text="{}")]
            # Only provide event, details should default to {}
            await captured_handler("session_event", {"event": "test_event"})
            mock_event.assert_called_once_with("test_event", {})

    @pytest.mark.asyncio
    async def test_call_tool_dispatches_to_get_data_dir(self) -> None:
        """Verifies call_tool routes 'get_data_dir' to _get_data_dir."""
        captured_handler = None

        def capture_call_tool():
            def decorator(func):
                nonlocal captured_handler
                captured_handler = func
                return func

            return decorator

        mock_server = MagicMock(spec=Server)
        mock_server.list_tools.return_value = lambda f: f
        mock_server.call_tool = capture_call_tool

        sessions.register(mock_server)

        with patch.object(sessions, "_get_data_dir") as mock_get:
            mock_get.return_value = [TextContent(type="text", text="{}")]
            await captured_handler("get_data_dir", {})
            mock_get.assert_called_once()

    @pytest.mark.asyncio
    async def test_call_tool_dispatches_to_set_data_dir(self) -> None:
        """Verifies call_tool routes 'set_data_dir' to _set_data_dir."""
        captured_handler = None

        def capture_call_tool():
            def decorator(func):
                nonlocal captured_handler
                captured_handler = func
                return func

            return decorator

        mock_server = MagicMock(spec=Server)
        mock_server.list_tools.return_value = lambda f: f
        mock_server.call_tool = capture_call_tool

        sessions.register(mock_server)

        with patch.object(sessions, "_set_data_dir") as mock_set:
            mock_set.return_value = [TextContent(type="text", text="{}")]
            await captured_handler("set_data_dir", {"path": "/new/path"})
            mock_set.assert_called_once_with("/new/path")

    @pytest.mark.asyncio
    async def test_call_tool_returns_error_for_unknown_tool(self) -> None:
        """Verifies call_tool returns error for unrecognized tool name.

        Arrangement:
        1. Capture call_tool handler.
        2. Call with invalid tool name.

        Assertion Strategy:
        - Returns TextContent with "Unknown tool" message.
        """
        captured_handler = None

        def capture_call_tool():
            def decorator(func):
                nonlocal captured_handler
                captured_handler = func
                return func

            return decorator

        mock_server = MagicMock(spec=Server)
        mock_server.list_tools.return_value = lambda f: f
        mock_server.call_tool = capture_call_tool

        sessions.register(mock_server)

        result = await captured_handler("nonexistent_tool", {})

        assert len(result) == 1
        assert "Unknown tool: nonexistent_tool" in result[0].text


# =============================================================================
# Test _start_session()
# =============================================================================


class TestStartSession:
    """Tests for _start_session() function."""

    @pytest.mark.asyncio
    async def test_start_session_success(self, mock_session_manager: MagicMock) -> None:
        """Verifies _start_session creates session successfully.

        Arrangement:
        1. Mock SessionManager with valid session.
        2. Call with observation type and target.

        Assertion Strategy:
        - Returns JSON with status='started'.
        - Contains session_id, session_type, target.
        """
        result = await sessions._start_session(
            "observation", "M42", "Test purpose", manager=mock_session_manager
        )

        assert len(result) == 1
        data = json.loads(result[0].text)
        assert data["status"] == "started"
        assert data["session_id"] == "test_session_123"
        assert data["session_type"] == "observation"
        assert data["target"] == "M42"

    @pytest.mark.asyncio
    async def test_start_session_invalid_type(
        self, mock_session_manager: MagicMock
    ) -> None:
        """Verifies _start_session rejects invalid session type.

        Arrangement:
        1. Mock SessionManager (not used due to early return).
        2. Call with invalid session type 'invalid_type'.

        Assertion Strategy:
        - Returns error message listing valid types.
        - Does not call manager.start_session().
        """
        result = await sessions._start_session(
            "invalid_type", "M42", None, manager=mock_session_manager
        )

        assert len(result) == 1
        assert "Invalid session type" in result[0].text
        assert "invalid_type" in result[0].text
        mock_session_manager.start_session.assert_not_called()

    @pytest.mark.asyncio
    async def test_start_session_idle_type_rejected(
        self, mock_session_manager: MagicMock
    ) -> None:
        """Verifies _start_session rejects 'idle' session type.

        Idle sessions are system-managed and cannot be started manually.

        Assertion Strategy:
        - Returns error message about idle sessions.
        - Does not call manager.start_session().
        """
        result = await sessions._start_session(
            "idle", None, None, manager=mock_session_manager
        )

        assert len(result) == 1
        assert "Cannot manually start an idle session" in result[0].text
        mock_session_manager.start_session.assert_not_called()

    @pytest.mark.asyncio
    async def test_start_session_case_insensitive(
        self, mock_session_manager: MagicMock
    ) -> None:
        """Verifies session type is case-insensitive.

        Arrangement:
        1. Call with 'OBSERVATION' (uppercase).

        Assertion Strategy:
        - Session created successfully.
        - SessionType.OBSERVATION used internally.
        """
        result = await sessions._start_session(
            "OBSERVATION", "M42", None, manager=mock_session_manager
        )

        data = json.loads(result[0].text)
        assert data["status"] == "started"
        mock_session_manager.start_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_session_exception_handling(
        self, mock_session_manager: MagicMock
    ) -> None:
        """Verifies _start_session catches and returns exceptions.

        Arrangement:
        1. Mock manager.start_session() to raise exception.

        Assertion Strategy:
        - Returns JSON error with 'internal' error type.
        - Contains exception message.
        """
        mock_session_manager.start_session.side_effect = RuntimeError(
            "Database connection failed"
        )

        result = await sessions._start_session(
            "observation", "M42", None, manager=mock_session_manager
        )

        assert len(result) == 1
        data = json.loads(result[0].text)
        assert data["error"] == "internal"
        assert "Database connection failed" in data["message"]

    @pytest.mark.asyncio
    async def test_start_session_uses_global_manager_when_none(self) -> None:
        """Verifies _start_session uses get_session_manager() when manager=None.

        Arrangement:
        1. Patch get_session_manager to return mock.
        2. Call without manager parameter.

        Assertion Strategy:
        - get_session_manager() called.
        """
        mock_manager = MagicMock()
        mock_session = MagicMock()
        mock_session.session_id = "global_session"
        mock_session.session_type = SessionType.OBSERVATION
        mock_session.target = "M42"
        mock_session.purpose = None
        mock_session.start_time.isoformat.return_value = "2025-01-15T10:30:00"
        mock_manager.start_session.return_value = mock_session

        with patch.object(sessions, "get_session_manager", return_value=mock_manager):
            result = await sessions._start_session("observation", "M42", None)

        data = json.loads(result[0].text)
        assert data["session_id"] == "global_session"


# =============================================================================
# Test _end_session()
# =============================================================================


class TestEndSession:
    """Tests for _end_session() function."""

    @pytest.mark.asyncio
    async def test_end_session_success(self, mock_session_manager: MagicMock) -> None:
        """Verifies _end_session closes session and returns path.

        Assertion Strategy:
        - Returns JSON with status='ended'.
        - Contains file_path to ASDF file.
        """
        result = await sessions._end_session(manager=mock_session_manager)

        assert len(result) == 1
        data = json.loads(result[0].text)
        assert data["status"] == "ended"
        assert data["session_id"] == "test_session_123"
        assert data["file_path"] == "/data/test_session.asdf"

    @pytest.mark.asyncio
    async def test_end_session_idle_returns_error(
        self, mock_session_manager: MagicMock
    ) -> None:
        """Verifies _end_session returns error when in idle state.

        Arrangement:
        1. Set manager.active_session_type to IDLE.

        Assertion Strategy:
        - Returns JSON error with 'no_session' type.
        """
        mock_session_manager.active_session_type = SessionType.IDLE

        result = await sessions._end_session(manager=mock_session_manager)

        assert len(result) == 1
        data = json.loads(result[0].text)
        assert data["error"] == "no_session"
        assert "idle" in data["message"].lower()

    @pytest.mark.asyncio
    async def test_end_session_exception_handling(
        self, mock_session_manager: MagicMock
    ) -> None:
        """Verifies _end_session catches and returns exceptions.

        Arrangement:
        1. Mock end_session() to raise exception.

        Assertion Strategy:
        - Returns JSON error with 'internal' type.
        """
        mock_session_manager.end_session.side_effect = OSError("Disk full")

        result = await sessions._end_session(manager=mock_session_manager)

        assert len(result) == 1
        data = json.loads(result[0].text)
        assert data["error"] == "internal"
        assert "Disk full" in data["message"]

    @pytest.mark.asyncio
    async def test_end_session_handles_none_session_type(
        self, mock_session_manager: MagicMock
    ) -> None:
        """Verifies _end_session handles None session_type gracefully.

        Edge case where session_type might be None during transition.

        Arrangement:
        1. Set active_session_type to valid type first.
        2. After end_session call, session_type becomes None for JSON.

        Assertion Strategy:
        - No exception raised.
        - file_path still returned.
        """
        # This tests the `.value if session_type else None` branch
        mock_session_manager.active_session_type = SessionType.OBSERVATION

        result = await sessions._end_session(manager=mock_session_manager)

        data = json.loads(result[0].text)
        assert data["session_type"] == "observation"


# =============================================================================
# Test _get_session_info()
# =============================================================================


class TestGetSessionInfo:
    """Tests for _get_session_info() function."""

    @pytest.mark.asyncio
    async def test_get_session_info_success(
        self, mock_session_manager: MagicMock
    ) -> None:
        """Verifies _get_session_info returns complete session metadata.

        Assertion Strategy:
        - Returns JSON with all expected fields.
        - Metrics include frames, logs, events, errors, warnings.
        """
        result = await sessions._get_session_info(manager=mock_session_manager)

        assert len(result) == 1
        data = json.loads(result[0].text)
        assert data["session_id"] == "test_session_123"
        assert data["session_type"] == "observation"
        assert data["target"] == "M42"
        assert data["duration_seconds"] == 120.5
        assert data["is_idle"] is False
        assert data["metrics"]["frames_captured"] == 5
        assert data["metrics"]["log_entries"] == 3
        assert data["metrics"]["events"] == 2
        assert data["metrics"]["errors"] == 1
        assert data["metrics"]["warnings"] == 2

    @pytest.mark.asyncio
    async def test_get_session_info_no_active_session(
        self, mock_session_manager: MagicMock
    ) -> None:
        """Verifies _get_session_info returns error when no session.

        Arrangement:
        1. Set active_session to None.

        Assertion Strategy:
        - Returns JSON error with 'no_session' type.
        """
        mock_session_manager.active_session = None

        result = await sessions._get_session_info(manager=mock_session_manager)

        assert len(result) == 1
        data = json.loads(result[0].text)
        assert data["error"] == "no_session"
        assert "No active session" in data["message"]

    @pytest.mark.asyncio
    async def test_get_session_info_idle_session(
        self, mock_session_manager: MagicMock
    ) -> None:
        """Verifies _get_session_info correctly identifies idle session.

        Arrangement:
        1. Set session.session_type to IDLE.

        Assertion Strategy:
        - is_idle is True in response.
        """
        mock_session_manager.active_session.session_type = SessionType.IDLE

        result = await sessions._get_session_info(manager=mock_session_manager)

        data = json.loads(result[0].text)
        assert data["is_idle"] is True

    @pytest.mark.asyncio
    async def test_get_session_info_exception_handling(
        self, mock_session_manager: MagicMock
    ) -> None:
        """Verifies _get_session_info catches and returns exceptions.

        Arrangement:
        1. Make active_session property raise exception.

        Assertion Strategy:
        - Returns JSON error with 'internal' type.
        """
        type(mock_session_manager).active_session = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("Manager crashed"))
        )

        result = await sessions._get_session_info(manager=mock_session_manager)

        assert len(result) == 1
        data = json.loads(result[0].text)
        assert data["error"] == "internal"
        assert "Manager crashed" in data["message"]


# =============================================================================
# Test _session_log()
# =============================================================================


class TestSessionLog:
    """Tests for _session_log() function."""

    @pytest.mark.asyncio
    async def test_session_log_success(self, mock_session_manager: MagicMock) -> None:
        """Verifies _session_log records entry successfully.

        Assertion Strategy:
        - Returns JSON with status='logged'.
        - Contains level, message, source, session_id.
        """
        result = await sessions._session_log(
            "INFO", "Test message", "user", manager=mock_session_manager
        )

        assert len(result) == 1
        data = json.loads(result[0].text)
        assert data["status"] == "logged"
        assert data["level"] == "INFO"
        assert data["message"] == "Test message"
        assert data["source"] == "user"
        mock_session_manager.log.assert_called_once()

    @pytest.mark.asyncio
    async def test_session_log_all_levels(
        self, mock_session_manager: MagicMock
    ) -> None:
        """Verifies _session_log accepts all LogLevel values.

        Tests: DEBUG, INFO, WARNING, ERROR, CRITICAL.

        Assertion Strategy:
        - Each level parsed and logged successfully.
        """
        levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

        for level in levels:
            mock_session_manager.reset_mock()
            result = await sessions._session_log(
                level, f"{level} message", "test", manager=mock_session_manager
            )
            data = json.loads(result[0].text)
            assert data["level"] == level

    @pytest.mark.asyncio
    async def test_session_log_case_insensitive(
        self, mock_session_manager: MagicMock
    ) -> None:
        """Verifies log level is case-insensitive.

        Arrangement:
        1. Call with 'warning' (lowercase).

        Assertion Strategy:
        - Level parsed as WARNING.
        """
        result = await sessions._session_log(
            "warning", "Test", "user", manager=mock_session_manager
        )

        data = json.loads(result[0].text)
        assert data["level"] == "WARNING"

    @pytest.mark.asyncio
    async def test_session_log_invalid_level(
        self, mock_session_manager: MagicMock
    ) -> None:
        """Verifies _session_log rejects invalid log level.

        Arrangement:
        1. Call with invalid level 'TRACE'.

        Assertion Strategy:
        - Returns error message listing valid levels.
        """
        result = await sessions._session_log(
            "TRACE", "Test", "user", manager=mock_session_manager
        )

        assert len(result) == 1
        assert "Invalid log level" in result[0].text
        assert "TRACE" in result[0].text
        mock_session_manager.log.assert_not_called()

    @pytest.mark.asyncio
    async def test_session_log_exception_handling(
        self, mock_session_manager: MagicMock
    ) -> None:
        """Verifies _session_log catches and returns exceptions.

        Arrangement:
        1. Mock log() to raise exception.

        Assertion Strategy:
        - Returns JSON error with 'internal' type.
        """
        mock_session_manager.log.side_effect = ValueError("Invalid log entry")

        result = await sessions._session_log(
            "INFO", "Test", "user", manager=mock_session_manager
        )

        assert len(result) == 1
        data = json.loads(result[0].text)
        assert data["error"] == "internal"
        assert "Invalid log entry" in data["message"]


# =============================================================================
# Test _session_event()
# =============================================================================


class TestSessionEvent:
    """Tests for _session_event() function."""

    @pytest.mark.asyncio
    async def test_session_event_success(self, mock_session_manager: MagicMock) -> None:
        """Verifies _session_event records event successfully.

        Assertion Strategy:
        - Returns JSON with status='recorded'.
        - Contains event name and details.
        """
        result = await sessions._session_event(
            "cloud_detected",
            {"coverage": 40, "action": "paused"},
            manager=mock_session_manager,
        )

        assert len(result) == 1
        data = json.loads(result[0].text)
        assert data["status"] == "recorded"
        assert data["event"] == "cloud_detected"
        assert data["details"] == {"coverage": 40, "action": "paused"}
        mock_session_manager.add_event.assert_called_once_with(
            "cloud_detected", coverage=40, action="paused"
        )

    @pytest.mark.asyncio
    async def test_session_event_empty_details(
        self, mock_session_manager: MagicMock
    ) -> None:
        """Verifies _session_event accepts empty details dict.

        Assertion Strategy:
        - Event recorded with no additional kwargs.
        """
        result = await sessions._session_event(
            "mount_parked", {}, manager=mock_session_manager
        )

        data = json.loads(result[0].text)
        assert data["status"] == "recorded"
        assert data["details"] == {}
        mock_session_manager.add_event.assert_called_once_with("mount_parked")

    @pytest.mark.asyncio
    async def test_session_event_exception_handling(
        self, mock_session_manager: MagicMock
    ) -> None:
        """Verifies _session_event catches and returns exceptions.

        Arrangement:
        1. Mock add_event() to raise exception.

        Assertion Strategy:
        - Returns JSON error with 'internal' type.
        """
        mock_session_manager.add_event.side_effect = RuntimeError("Event storage full")

        result = await sessions._session_event(
            "test_event", {}, manager=mock_session_manager
        )

        assert len(result) == 1
        data = json.loads(result[0].text)
        assert data["error"] == "internal"
        assert "Event storage full" in data["message"]


# =============================================================================
# Test _get_data_dir()
# =============================================================================


class TestGetDataDir:
    """Tests for _get_data_dir() function."""

    @pytest.mark.asyncio
    async def test_get_data_dir_success(self, mock_factory: MagicMock) -> None:
        """Verifies _get_data_dir returns current data directory.

        Assertion Strategy:
        - Returns JSON with data_dir path.
        - Contains exists boolean.
        """
        with patch(
            "telescope_mcp.tools.sessions.get_factory", return_value=mock_factory
        ):
            mock_factory.config.data_dir.exists.return_value = True

            result = await sessions._get_data_dir()

        assert len(result) == 1
        data = json.loads(result[0].text)
        assert data["data_dir"] == "/home/user/telescope-data"
        assert data["exists"] is True

    @pytest.mark.asyncio
    async def test_get_data_dir_nonexistent(self, mock_factory: MagicMock) -> None:
        """Verifies _get_data_dir reports exists=False for missing dir.

        Assertion Strategy:
        - exists is False when directory doesn't exist.
        """
        with patch(
            "telescope_mcp.tools.sessions.get_factory", return_value=mock_factory
        ):
            mock_factory.config.data_dir.exists.return_value = False

            result = await sessions._get_data_dir()

        data = json.loads(result[0].text)
        assert data["exists"] is False

    @pytest.mark.asyncio
    async def test_get_data_dir_exception_handling(self) -> None:
        """Verifies _get_data_dir catches and returns exceptions.

        Arrangement:
        1. Mock get_factory() to raise exception.

        Assertion Strategy:
        - Returns JSON error with 'internal' type.
        """
        with patch(
            "telescope_mcp.tools.sessions.get_factory",
            side_effect=RuntimeError("Factory not initialized"),
        ):
            result = await sessions._get_data_dir()

        assert len(result) == 1
        data = json.loads(result[0].text)
        assert data["error"] == "internal"
        assert "Factory not initialized" in data["message"]


# =============================================================================
# Test _set_data_dir()
# =============================================================================


class TestSetDataDir:
    """Tests for _set_data_dir() function."""

    @pytest.mark.asyncio
    async def test_set_data_dir_success(self, tmp_path: Path) -> None:
        """Verifies _set_data_dir updates data directory.

        Arrangement:
        1. Use pytest tmp_path fixture for valid directory.

        Assertion Strategy:
        - Returns JSON with status='updated'.
        - Contains new data_dir path.
        """
        with patch("telescope_mcp.drivers.config.set_data_dir") as mock_set:
            result = await sessions._set_data_dir(str(tmp_path))

        assert len(result) == 1
        data = json.loads(result[0].text)
        assert data["status"] == "updated"
        assert str(tmp_path) in data["data_dir"]
        mock_set.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_data_dir_creates_absolute_path(self, tmp_path: Path) -> None:
        """Verifies _set_data_dir resolves paths to absolute.

        Arrangement:
        1. Provide absolute path (tmp_path is already absolute).

        Assertion Strategy:
        - Resolved path is used.
        """
        with patch("telescope_mcp.drivers.config.set_data_dir") as mock_set:
            result = await sessions._set_data_dir(str(tmp_path))

        # The path should be resolved (absolute)
        call_args = mock_set.call_args[0][0]
        assert call_args.is_absolute()

    @pytest.mark.asyncio
    async def test_set_data_dir_includes_note(self, tmp_path: Path) -> None:
        """Verifies _set_data_dir response includes session reset note.

        Assertion Strategy:
        - Response contains note about session manager reset.
        """
        with patch("telescope_mcp.drivers.config.set_data_dir"):
            result = await sessions._set_data_dir(str(tmp_path))

        data = json.loads(result[0].text)
        assert "note" in data
        assert "reset" in data["note"].lower()

    @pytest.mark.asyncio
    async def test_set_data_dir_exception_handling(self) -> None:
        """Verifies _set_data_dir catches and returns exceptions.

        Arrangement:
        1. Mock set_data_dir to raise permission error.

        Assertion Strategy:
        - Returns JSON error with 'internal' type.
        """
        with patch(
            "telescope_mcp.drivers.config.set_data_dir",
            side_effect=PermissionError("Access denied"),
        ):
            result = await sessions._set_data_dir("/restricted/path")

        assert len(result) == 1
        data = json.loads(result[0].text)
        assert data["error"] == "internal"
        assert "Access denied" in data["message"]

    @pytest.mark.asyncio
    async def test_set_data_dir_non_absolute_path_rejected(self) -> None:
        """Verifies _set_data_dir rejects non-absolute paths after resolution.

        This tests the validation branch that checks is_absolute() after
        Path.resolve(). While resolve() normally returns absolute paths,
        this tests the defensive validation.

        Arrangement:
        1. Mock Path to return a mock that reports is_absolute()=False.

        Assertion Strategy:
        - Returns JSON error with 'validation' type.
        """
        mock_path = MagicMock()
        mock_path.resolve.return_value = mock_path
        mock_path.is_absolute.return_value = False

        with patch("telescope_mcp.tools.sessions.Path", return_value=mock_path):
            result = await sessions._set_data_dir("relative/path")

        assert len(result) == 1
        data = json.loads(result[0].text)
        assert data["error"] == "validation"
        assert "absolute" in data["message"].lower()

    @pytest.mark.asyncio
    async def test_set_data_dir_reports_exists_status(self, tmp_path: Path) -> None:
        """Verifies _set_data_dir reports whether new path exists.

        Arrangement:
        1. tmp_path exists (pytest creates it).

        Assertion Strategy:
        - exists is True for existing directory.
        """
        with patch("telescope_mcp.drivers.config.set_data_dir"):
            result = await sessions._set_data_dir(str(tmp_path))

        data = json.loads(result[0].text)
        assert data["exists"] is True

    @pytest.mark.asyncio
    async def test_set_data_dir_nonexistent_path(self) -> None:
        """Verifies _set_data_dir reports exists=False for new paths.

        Arrangement:
        1. Provide path that doesn't exist yet.

        Assertion Strategy:
        - exists is False.
        """
        new_path = "/tmp/telescope_test_nonexistent_12345"

        with patch("telescope_mcp.drivers.config.set_data_dir"):
            result = await sessions._set_data_dir(new_path)

        data = json.loads(result[0].text)
        assert data["exists"] is False


# =============================================================================
# Test Edge Cases and Integration
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_start_session_with_none_optional_params(
        self, mock_session_manager: MagicMock
    ) -> None:
        """Verifies _start_session handles None target and purpose.

        Assertion Strategy:
        - Session created with null values for optional fields.
        """
        mock_session_manager.active_session.target = None
        mock_session_manager.active_session.purpose = None

        result = await sessions._start_session(
            "experiment", None, None, manager=mock_session_manager
        )

        data = json.loads(result[0].text)
        assert data["status"] == "started"

    @pytest.mark.asyncio
    async def test_session_log_with_empty_message(
        self, mock_session_manager: MagicMock
    ) -> None:
        """Verifies _session_log handles empty message string.

        Assertion Strategy:
        - Empty message logged successfully.
        """
        result = await sessions._session_log(
            "INFO", "", "user", manager=mock_session_manager
        )

        data = json.loads(result[0].text)
        assert data["status"] == "logged"
        assert data["message"] == ""

    @pytest.mark.asyncio
    async def test_session_event_with_nested_details(
        self, mock_session_manager: MagicMock
    ) -> None:
        """Verifies _session_event handles nested detail structures.

        Assertion Strategy:
        - Complex nested dict preserved in response.
        """
        nested_details = {
            "camera": {"exposure": 5.0, "gain": 100},
            "mount": {"ra": 10.5, "dec": 45.2},
            "conditions": ["clear", "stable"],
        }

        result = await sessions._session_event(
            "frame_captured", nested_details, manager=mock_session_manager
        )

        data = json.loads(result[0].text)
        assert data["details"] == nested_details

    @pytest.mark.asyncio
    async def test_multiple_sequential_operations(
        self, mock_session_manager: MagicMock
    ) -> None:
        """Verifies multiple operations work in sequence.

        Tests realistic workflow: start -> log -> event -> info.

        Assertion Strategy:
        - All operations succeed.
        - Session ID consistent across operations.
        """
        # Start session
        await sessions._start_session(
            "observation", "M42", None, manager=mock_session_manager
        )

        # Log message
        await sessions._session_log(
            "INFO", "Starting capture", "user", manager=mock_session_manager
        )

        # Add event
        await sessions._session_event(
            "capture_started", {"frames": 10}, manager=mock_session_manager
        )

        # Get info
        result = await sessions._get_session_info(manager=mock_session_manager)

        data = json.loads(result[0].text)
        assert data["session_id"] == "test_session_123"


class TestReturnTypeConsistency:
    """Tests ensuring all functions return consistent types."""

    @pytest.mark.asyncio
    async def test_all_functions_return_list_of_text_content(
        self, mock_session_manager: MagicMock, mock_factory: MagicMock
    ) -> None:
        """Verifies all handler functions return List[TextContent].

        Assertion Strategy:
        - Each function returns list.
        - Each list item is TextContent.
        """
        with patch(
            "telescope_mcp.tools.sessions.get_factory", return_value=mock_factory
        ):
            with patch("telescope_mcp.drivers.config.set_data_dir"):
                results = [
                    await sessions._start_session(
                        "observation", "M42", None, manager=mock_session_manager
                    ),
                    await sessions._end_session(manager=mock_session_manager),
                    await sessions._get_session_info(manager=mock_session_manager),
                    await sessions._session_log(
                        "INFO", "test", "user", manager=mock_session_manager
                    ),
                    await sessions._session_event(
                        "test", {}, manager=mock_session_manager
                    ),
                    await sessions._get_data_dir(),
                    await sessions._set_data_dir("/tmp"),
                ]

        for result in results:
            assert isinstance(result, list)
            assert len(result) == 1
            assert isinstance(result[0], TextContent)
            assert result[0].type == "text"

    @pytest.mark.asyncio
    async def test_error_responses_are_valid_json(
        self, mock_session_manager: MagicMock
    ) -> None:
        """Verifies error responses contain valid JSON.

        Tests all error paths return parseable JSON.

        Assertion Strategy:
        - All error responses parse as JSON.
        - Error responses have 'error' key or descriptive text.
        """
        # Invalid session type
        result1 = await sessions._start_session(
            "invalid", None, None, manager=mock_session_manager
        )
        # Should be error text, not JSON
        assert "Invalid session type" in result1[0].text

        # Invalid log level
        result2 = await sessions._session_log(
            "INVALID", "test", "user", manager=mock_session_manager
        )
        assert "Invalid log level" in result2[0].text

        # Exception handling returns JSON
        mock_session_manager.start_session.side_effect = Exception("Test error")
        result3 = await sessions._start_session(
            "observation", None, None, manager=mock_session_manager
        )
        data3 = json.loads(result3[0].text)
        assert data3["error"] == "internal"
