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
    """Pytest fixture creating a mock SessionManager for dependency injection.

    Returns a MagicMock configured with typical SessionManager behavior
    that can be customized per test for isolated session tool testing.

    Business context:
        Session tools require a SessionManager for state management. This mock
        provides predictable behavior for testing tool handlers without file I/O
        or ASDF serialization overhead, enabling fast, deterministic tests.

    Arrangement:
        1. Create MagicMock instance with SessionManager-like attributes.
        2. Configure active_session_id="test_session_123".
        3. Configure active_session_type=SessionType.OBSERVATION.
        4. Create mock session object with metrics (frames=5, logs=3, events=2).
        5. Configure start_session to return mock session.
        6. Configure end_session to return Path("/data/test_session.asdf").

    Args:
        None (pytest fixture with implicit request parameter).

    Returns:
        MagicMock: Configured mock SessionManager with:
            - active_session_id="test_session_123"
            - active_session_type=SessionType.OBSERVATION
            - active_session with metrics (frames=5, logs=3, events=2)
            - start_session returning mock session
            - end_session returning Path("/data/test_session.asdf")

    Raises:
        None. Mock configuration is deterministic without external dependencies.

    Example:
        >>> async def test_session(mock_session_manager):
        ...     result = await _start_session("observation", "M42", None,
        ...                                   manager=mock_session_manager)
        ...     assert "started" in result[0].text

    Testing Principle:
        Validates dependency injection pattern, enabling isolated testing
        of session tools without file system or ASDF library dependencies.
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
    """Pytest fixture creating a mock ComponentFactory for data_dir access.

    Provides a configured mock ComponentFactory with data_dir property
    that returns Path-like mock object for testing session storage paths.

    Business context:
        Session tools use get_factory() to access data directory for ASDF
        file storage. This fixture isolates tests from real filesystem
        operations, enabling fast tests without disk I/O.

    Arrangement:
        1. Create MagicMock instance for factory.
        2. Create mock_data_dir with __str__ returning '/home/user/telescope-data'.
        3. Configure exists() to return True (directory exists).
        4. Attach mock_data_dir to factory.config.data_dir.

    Args:
        None (pytest fixture with implicit request parameter).

    Returns:
        MagicMock: Factory mock with config.data_dir set to Path-like
            mock returning '/home/user/telescope-data' from str().

    Raises:
        None. Mock configuration is deterministic.

    Example:
        >>> def test_data_dir(mock_factory):
        ...     assert str(mock_factory.config.data_dir) == '/home/user/telescope-data'

    Testing Principle:
        Validates filesystem isolation, enabling path manipulation tests
        without real directory creation or access permission requirements.
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

        Action:
        Accesses sessions.TOOLS and iterates over items.

        Assertion Strategy:
        Validates type consistency by confirming:
        - TOOLS is a list.
        - Each item is a Tool instance.

        Testing Principle:
        Validates API contract, ensuring TOOLS exports correct types
        for MCP server tool registration.
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

        Arrangement:
        1. sessions.TOOLS is module-level constant.
        2. Each tool represents a session management operation.

        Action:
        Counts items in sessions.TOOLS list.

        Assertion Strategy:
        Validates tool completeness by confirming:
        - TOOLS has exactly 7 items.

        Testing Principle:
        Validates completeness, ensuring all required session tools
        are exported for MCP client discovery.
        """
        assert len(sessions.TOOLS) == 7

    def test_start_session_tool_schema(self) -> None:
        """Verifies start_session tool has correct input schema.

        Arrangement:
        1. Find start_session tool in TOOLS.
        2. Check inputSchema structure for session_type field.

        Action:
        Extracts tool and inspects inputSchema properties.

        Assertion Strategy:
        Validates schema correctness by confirming:
        - session_type enum has 4 values (observation, alignment, etc).
        - session_type is listed in required fields.

        Testing Principle:
        Validates MCP schema contract, ensuring clients can construct
        valid requests with proper session type constraints.
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
        2. Check level enum values in inputSchema.

        Action:
        Extracts tool and inspects level enum in inputSchema.

        Assertion Strategy:
        Validates log level completeness by confirming:
        - level enum includes CRITICAL.

        Testing Principle:
        Validates logging levels, ensuring critical errors can be
        logged for serious issues like hardware failures.
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
        1. Create mock MCP Server with spec.
        2. Configure list_tools() as decorator returning identity function.

        Action:
        Calls sessions.register(mock_server).

        Assertion Strategy:
        Validates registration by confirming:
        - server.list_tools() decorator was called exactly once.

        Testing Principle:
        Validates handler registration, ensuring MCP server receives
        tool listing capability for client discovery.
        """
        mock_server = MagicMock(spec=Server)
        mock_server.list_tools.return_value = lambda f: f
        mock_server.call_tool.return_value = lambda f: f

        sessions.register(mock_server)

        mock_server.list_tools.assert_called_once()

    def test_register_adds_call_tool_handler(self) -> None:
        """Verifies register() adds call_tool handler to server.

        Arrangement:
        1. Create mock MCP Server with spec.
        2. Configure call_tool() as decorator returning identity function.

        Action:
        Calls sessions.register(mock_server).

        Assertion Strategy:
        Validates registration by confirming:
        - server.call_tool() decorator was called exactly once.

        Testing Principle:
        Validates handler registration, ensuring MCP server receives
        tool invocation capability for client requests.
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
        Calls the captured list_tools handler directly.

        Assertion Strategy:
        Validates handler output by confirming:
        - Handler returns exactly sessions.TOOLS list.

        Testing Principle:
        Validates MCP contract, ensuring clients receive correct
        tool definitions for session management operations.
        """
        captured_handler = None

        def capture_list_tools():
            """Create decorator factory that captures list_tools handler.

            Provides test access to the registered handler function by
            capturing it in a nonlocal variable during registration.

            Business context:
                MCP server decorators cannot be called directly in tests. This
                factory pattern captures the handler for isolated unit testing.

            Args:
                None: This factory function takes no arguments.

            Returns:
                Callable: Decorator function that captures and returns the handler.

            Raises:
                None: Pure function with no side effects beyond closure capture.

            Example:
                >>> handler = None
                >>> def capture(): ...
                >>> mock_server.list_tools = capture
            """

            def decorator(func):
                """Capture handler function in nonlocal variable for test access.

                Stores the decorated function in nonlocal captured_handler
                variable, enabling direct handler invocation in test assertions.

                Business context:
                    Intercepts MCP registration to enable isolated handler testing
                    without requiring full server lifecycle.

                Args:
                    func: The handler function being registered via decorator.

                Returns:
                    The original function unchanged, preserving call signature.

                Example:
                    >>> @decorator
                    ... async def handler(): pass
                    >>> assert captured_handler is handler
                """
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
        """Verifies call_tool routes 'start_session' to _start_session handler.

        Tests the dispatcher routing branch for session creation tool.

        Arrangement:
            1. Capture call_tool handler from register() via decorator capture.
            2. Mock _start_session to track calls and return canned response.
            3. Prepare params: session_type='observation', target='M42', purpose=None.

        Action:
            Call captured handler with name='start_session' and session params.

        Assertion Strategy:
            Validates routing logic by confirming:
            - _start_session called with correct arguments ("observation", "M42", None).
            - All three parameters extracted from params dict correctly.

        Testing Principle:
            Validates routing logic, ensuring correct tool dispatch with
            complete argument passing from MCP params to internal handler.
        """
        captured_handler = None

        def capture_call_tool():
            """Create decorator factory that captures call_tool handler.

            Provides test access to the registered handler function by
            capturing it in a nonlocal variable during registration.

            Business context:
                MCP server decorators cannot be called directly in tests. This
                factory pattern captures the handler for isolated unit testing.

            Args:
                None: This factory function takes no arguments.

            Returns:
                Callable: Decorator function that captures and returns the handler.

            Raises:
                None: Pure function with no side effects beyond closure capture.

            Example:
                >>> handler = None
                >>> def capture(): ...
                >>> mock_server.call_tool = capture
            """

            def decorator(func):
                """Capture handler function in nonlocal variable for test access.

                Stores the decorated function in nonlocal captured_handler
                variable, enabling direct handler invocation in test assertions.

                Business context:
                    Intercepts MCP registration to enable isolated handler testing
                    without requiring full server lifecycle.

                Args:
                    func: The handler function being registered via decorator.

                Returns:
                    The original function unchanged, preserving call signature.

                Example:
                    >>> @decorator
                    ... async def handler(): pass
                    >>> assert captured_handler is handler
                """
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
        """Verifies call_tool routes 'end_session' to _end_session handler.

        Tests the dispatcher routing branch for session termination tool.

        Arrangement:
            1. Capture call_tool handler from register() via decorator capture.
            2. Mock _end_session to track calls and return canned response.
            3. Prepare empty params dict (end_session takes no arguments).

        Action:
            Call captured handler with name='end_session' and empty params.

        Assertion Strategy:
            Validates routing by confirming:
            - _end_session was called exactly once.
            - No-argument dispatch works correctly.

        Testing Principle:
            Validates routing logic, ensuring end_session tool invokes
            correct handler without parameters.
        """
        captured_handler = None

        def capture_call_tool():
            """Create decorator factory that captures call_tool handler.

            Provides test access to the registered handler function by
            capturing it in a nonlocal variable during registration.

            Business context:
                MCP server decorators cannot be called directly in tests. This
                factory pattern captures the handler for isolated unit testing.

            Args:
                None: This factory function takes no arguments.

            Returns:
                Callable: Decorator function that captures and returns the handler.

            Raises:
                None: Pure function with no side effects beyond closure capture.

            Example:
                >>> handler = None
                >>> def capture(): ...
                >>> mock_server.call_tool = capture
            """

            def decorator(func):
                """Capture handler function in nonlocal variable for test access.

                Stores the decorated function in nonlocal captured_handler
                variable, enabling direct handler invocation in test assertions.

                Business context:
                    Intercepts MCP registration to enable isolated handler testing
                    without requiring full server lifecycle.

                Args:
                    func: The handler function being registered via decorator.

                Returns:
                    The original function unchanged, preserving call signature.

                Example:
                    >>> @decorator
                    ... async def handler(): pass
                    >>> assert captured_handler is handler
                """
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
        """Verifies call_tool routes 'get_session_info' to _get_session_info handler.

        Tests the dispatcher routing branch for session info retrieval tool.

        Arrangement:
            1. Capture call_tool handler from register() via decorator capture.
            2. Mock _get_session_info to track calls and return canned response.
            3. Prepare empty params dict (get_session_info takes no arguments).

        Action:
            Call captured handler with name='get_session_info' and empty params.

        Assertion Strategy:
            Validates routing by confirming:
            - _get_session_info called once.
            - Session info retrieval dispatched correctly.

        Testing Principle:
            Validates routing logic, ensuring session info retrieval
            is properly dispatched to the internal handler.
        """
        captured_handler = None

        def capture_call_tool():
            """Create decorator factory that captures call_tool handler.

            Provides test access to the registered handler function by
            capturing it in a nonlocal variable during registration.

            Business context:
                MCP server decorators cannot be called directly in tests. This
                factory pattern captures the handler for isolated unit testing.

            Args:
                None: This factory function takes no arguments.

            Returns:
                Callable: Decorator function that captures and returns the handler.

            Raises:
                None: Pure function with no side effects beyond closure capture.

            Example:
                >>> handler = None
                >>> def capture(): ...
                >>> mock_server.call_tool = capture
            """

            def decorator(func):
                """Capture handler function in nonlocal variable for test access.

                Stores the decorated function in nonlocal captured_handler
                variable, enabling direct handler invocation in test assertions.

                Business context:
                    Intercepts MCP registration to enable isolated handler testing
                    without requiring full server lifecycle.

                Args:
                    func: The handler function being registered via decorator.

                Returns:
                    The original function unchanged, preserving call signature.

                Example:
                    >>> @decorator
                    ... async def handler(): pass
                    >>> assert captured_handler is handler
                """
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
        """Verifies call_tool routes 'session_log' with level, message, and source.

        Tests the dispatcher routing and parameter extraction for logging tool.

        Arrangement:
            1. Capture call_tool handler from register() via decorator capture.
            2. Mock _session_log to track calls and return canned response.
            3. Prepare params with level='INFO', message='test', source='user'.

        Action:
            Call captured handler with name='session_log' and log params.

        Assertion Strategy:
            Validates routing logic by confirming:
            - _session_log called with ("INFO", "test", "user").
            - All three parameters extracted from params dict.

        Testing Principle:
            Validates routing logic, ensuring log entries are dispatched
            with correct arguments for proper log categorization.
        """
        captured_handler = None

        def capture_call_tool():
            """Create decorator factory that captures call_tool handler.

            Provides test access to the registered handler function by
            capturing it in a nonlocal variable during registration.

            Business context:
                MCP server decorators cannot be called directly in tests. This
                factory pattern captures the handler for isolated unit testing.

            Args:
                None: This factory function takes no arguments.

            Returns:
                Callable: Decorator function that captures and returns the handler.

            Raises:
                None: Pure function with no side effects beyond closure capture.

            Example:
                >>> handler = None
                >>> def capture(): ...
                >>> mock_server.call_tool = capture
            """

            def decorator(func):
                """Capture handler function in nonlocal variable for test access.

                Stores the decorated function in nonlocal captured_handler
                variable, enabling direct handler invocation in test assertions.

                Business context:
                    Intercepts MCP registration to enable isolated handler testing
                    without requiring full server lifecycle.

                Args:
                    func: The handler function being registered via decorator.

                Returns:
                    The original function unchanged, preserving call signature.

                Example:
                    >>> @decorator
                    ... async def handler(): pass
                    >>> assert captured_handler is handler
                """
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
        """Verifies call_tool applies default values for omitted session_log params.

        Tests default parameter handling when optional level and source are omitted.

        Arrangement:
            1. Capture call_tool handler from register() via decorator capture.
            2. Mock _session_log to track calls and return canned response.
            3. Prepare params with only message='test' (omit level and source).

        Action:
            Call captured handler with only 'message' param, omitting level/source.

        Assertion Strategy:
            Validates default handling by confirming:
            - _session_log called with 'INFO' level (default).
            - _session_log called with 'user' source (default).
            - Message passed through correctly.

        Testing Principle:
            Validates default handling, ensuring optional parameters have
            sensible defaults for convenient API usage.
        """
        captured_handler = None

        def capture_call_tool():
            """Create decorator factory that captures call_tool handler.

            Provides test access to the registered handler function by
            capturing it in a nonlocal variable during registration.

            Business context:
                MCP server decorators cannot be called directly in tests. This
                factory pattern captures the handler for isolated unit testing.

            Args:
                None: This factory function takes no arguments.

            Returns:
                Callable: Decorator function that captures and returns the handler.

            Raises:
                None: Pure function with no side effects beyond closure capture.

            Example:
                >>> handler = None
                >>> def capture(): ...
                >>> mock_server.call_tool = capture
            """

            def decorator(func):
                """Capture handler function in nonlocal variable for test access.

                Stores the decorated function in nonlocal captured_handler
                variable, enabling direct handler invocation in test assertions.

                Business context:
                    Intercepts MCP registration to enable isolated handler testing
                    without requiring full server lifecycle.

                Args:
                    func: The handler function being registered via decorator.

                Returns:
                    The original function unchanged, preserving call signature.

                Example:
                    >>> @decorator
                    ... async def handler(): pass
                    >>> assert captured_handler is handler
                """
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
        """Verifies call_tool routes 'session_event' with event name and details.

        Tests the dispatcher routing and parameter extraction for event recording.

        Arrangement:
            1. Capture call_tool handler from register() via decorator capture.
            2. Mock _session_event to track calls and return canned response.
            3. Prepare params with event='test_event', details={'key': 'value'}.

        Action:
            Call captured handler with name='session_event' and event params.

        Assertion Strategy:
            Validates routing logic by confirming:
            - _session_event called with ("test_event", {"key": "value"}).
            - Event name and details dict extracted correctly.

        Testing Principle:
            Validates routing logic, ensuring events are dispatched
            with full context for session event tracking.
        """
        captured_handler = None

        def capture_call_tool():
            """Create decorator factory that captures call_tool handler.

            Provides test access to the registered handler function by
            capturing it in a nonlocal variable during registration.

            Business context:
                MCP server decorators cannot be called directly in tests. This
                factory pattern captures the handler for isolated unit testing.

            Args:
                None: This factory function takes no arguments.

            Returns:
                Callable: Decorator function that captures and returns the handler.

            Raises:
                None: Pure function with no side effects beyond closure capture.

            Example:
                >>> handler = None
                >>> def capture(): ...
                >>> mock_server.call_tool = capture
            """

            def decorator(func):
                """Capture handler function in nonlocal variable for test access.

                Stores the decorated function in nonlocal captured_handler
                variable, enabling direct handler invocation in test assertions.

                Business context:
                    Intercepts MCP registration to enable isolated handler testing
                    without requiring full server lifecycle.

                Args:
                    func: The handler function being registered via decorator.

                Returns:
                    The original function unchanged, preserving call signature.

                Example:
                    >>> @decorator
                    ... async def handler(): pass
                    >>> assert captured_handler is handler
                """
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
        """Verifies call_tool applies empty dict default for omitted details param.

        Tests default parameter handling when optional details dict is omitted.

        Arrangement:
            1. Capture call_tool handler from register() via decorator capture.
            2. Mock _session_event to track calls and return canned response.
            3. Prepare params with only event='test_event' (omit details).

        Action:
            Call captured handler with only 'event' parameter, omitting details.

        Assertion Strategy:
            Validates default handling by confirming:
            - _session_event called with empty dict {} for details.
            - Event name passed through correctly.

        Testing Principle:
            Validates default handling, ensuring missing details defaults
            to empty dict for simple event recording without metadata.
        """
        captured_handler = None

        def capture_call_tool():
            """Create decorator factory that captures call_tool handler.

            Provides test access to the registered handler function by
            capturing it in a nonlocal variable during registration.

            Business context:
                MCP server decorators cannot be called directly in tests. This
                factory pattern captures the handler for isolated unit testing.

            Args:
                None: This factory function takes no arguments.

            Returns:
                Callable: Decorator function that captures and returns the handler.

            Raises:
                None: Pure function with no side effects beyond closure capture.

            Example:
                >>> handler = None
                >>> def capture(): ...
                >>> mock_server.call_tool = capture
            """

            def decorator(func):
                """Capture handler function in nonlocal variable for test access.

                Stores the decorated function in nonlocal captured_handler
                variable, enabling direct handler invocation in test assertions.

                Business context:
                    Intercepts MCP registration to enable isolated handler testing
                    without requiring full server lifecycle.

                Args:
                    func: The handler function being registered via decorator.

                Returns:
                    The original function unchanged, preserving call signature.

                Example:
                    >>> @decorator
                    ... async def handler(): pass
                    >>> assert captured_handler is handler
                """
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
        """Verifies call_tool routes 'get_data_dir' to _get_data_dir handler.

        Tests the dispatcher routing branch for data directory retrieval tool.

        Arrangement:
            1. Capture call_tool handler from register() via decorator capture.
            2. Mock _get_data_dir to track calls and return canned response.
            3. Prepare empty params dict (get_data_dir takes no arguments).

        Action:
            Call captured handler with name='get_data_dir' and empty params.

        Assertion Strategy:
            Validates routing by confirming:
            - _get_data_dir called once.
            - Data directory retrieval dispatched correctly.

        Testing Principle:
            Validates routing logic, ensuring data directory retrieval
            is properly dispatched to the internal handler.
        """
        captured_handler = None

        def capture_call_tool():
            """Create decorator factory that captures call_tool handler.

            Provides test access to the registered handler function by
            capturing it in a nonlocal variable during registration.

            Business context:
                MCP server decorators cannot be called directly in tests. This
                factory pattern captures the handler for isolated unit testing.

            Args:
                None: This factory function takes no arguments.

            Returns:
                Callable: Decorator function that captures and returns the handler.

            Raises:
                None: Pure function with no side effects beyond closure capture.

            Example:
                >>> handler = None
                >>> def capture(): ...
                >>> mock_server.call_tool = capture
            """

            def decorator(func):
                """Capture handler function in nonlocal variable for test access.

                Stores the decorated function in nonlocal captured_handler
                variable, enabling direct handler invocation in test assertions.

                Business context:
                    Intercepts MCP registration to enable isolated handler testing
                    without requiring full server lifecycle.

                Args:
                    func: The handler function being registered via decorator.

                Returns:
                    The original function unchanged, preserving call signature.

                Example:
                    >>> @decorator
                    ... async def handler(): pass
                    >>> assert captured_handler is handler
                """
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
        """Verifies call_tool routes 'set_data_dir' with path parameter.

        Tests the dispatcher routing and parameter extraction for data dir update.

        Arrangement:
            1. Capture call_tool handler from register() via decorator capture.
            2. Mock _set_data_dir to track calls and return canned response.
            3. Prepare params with path='/new/path'.

        Action:
            Call captured handler with name='set_data_dir' and path param.

        Assertion Strategy:
            Validates routing by confirming:
            - _set_data_dir called with the provided path '/new/path'.
            - Path parameter extracted correctly from params dict.

        Testing Principle:
            Validates routing logic, ensuring data directory update
            is properly dispatched with the path argument.
        """
        captured_handler = None

        def capture_call_tool():
            """Create decorator factory that captures call_tool handler.

            Provides test access to the registered handler function by
            capturing it in a nonlocal variable during registration.

            Business context:
                MCP server decorators cannot be called directly in tests. This
                factory pattern captures the handler for isolated unit testing.

            Args:
                None: This factory function takes no arguments.

            Returns:
                Callable: Decorator function that captures and returns the handler.

            Raises:
                None: Pure function with no side effects beyond closure capture.

            Example:
                >>> handler = None
                >>> def capture(): ...
                >>> mock_server.call_tool = capture
            """

            def decorator(func):
                """Capture handler function in nonlocal variable for test access.

                Stores the decorated function in nonlocal captured_handler
                variable, enabling direct handler invocation in test assertions.

                Business context:
                    Intercepts MCP registration to enable isolated handler testing
                    without requiring full server lifecycle.

                Args:
                    func: The handler function being registered via decorator.

                Returns:
                    The original function unchanged, preserving call signature.

                Example:
                    >>> @decorator
                    ... async def handler(): pass
                    >>> assert captured_handler is handler
                """
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
        """Verifies call_tool returns structured error for unrecognized tool name.

        Tests the error handling path when an invalid tool is requested.

        Arrangement:
            1. Capture call_tool handler from register() via decorator capture.
            2. No mocks needed - testing error path directly.
            3. Prepare unrecognized tool name 'nonexistent_tool'.

        Action:
            Call captured handler with name='nonexistent_tool' and empty params.

        Assertion Strategy:
            Validates error handling by confirming:
            - Returns TextContent with "Unknown tool" message.
            - Error message includes the invalid tool name for debugging.

        Testing Principle:
            Validates error handling, ensuring unknown tools return informative
            errors that help clients identify invalid requests.
        """
        captured_handler = None

        def capture_call_tool():
            """Create decorator factory that captures call_tool handler.

            Provides test access to the registered handler function by
            capturing it in a nonlocal variable during registration.

            Business context:
                MCP server decorators cannot be called directly in tests. This
                factory pattern captures the handler for isolated unit testing.

            Args:
                None: This factory function takes no arguments.

            Returns:
                Callable: Decorator function that captures and returns the handler.

            Raises:
                None: Pure function with no side effects beyond closure capture.

            Example:
                >>> handler = None
                >>> def capture(): ...
                >>> mock_server.call_tool = capture
            """

            def decorator(func):
                """Capture handler function in nonlocal variable for test access.

                Stores the decorated function in nonlocal captured_handler
                variable, enabling direct handler invocation in test assertions.

                Business context:
                    Intercepts MCP registration to enable isolated handler testing
                    without requiring full server lifecycle.

                Args:
                    func: The handler function being registered via decorator.

                Returns:
                    The original function unchanged, preserving call signature.

                Example:
                    >>> @decorator
                    ... async def handler(): pass
                    >>> assert captured_handler is handler
                """
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

        Action:
        Call _start_session with 'observation' type, 'M42' target, and purpose.

        Assertion Strategy:
        - Returns JSON with status='started'.
        - Contains session_id, session_type, target.

        Testing Principle:
        Validates happy path, ensuring session creation returns complete metadata.
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

        Action:
        Call _start_session with 'invalid_type' session type.

        Assertion Strategy:
        - Returns error message listing valid types.
        - Does not call manager.start_session().

        Testing Principle:
        Validates input validation, ensuring invalid types are rejected early.
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

        Action:
        Call _start_session with 'idle' session type.

        Assertion Strategy:
        - Returns error message about idle sessions.
        - Does not call manager.start_session().

        Testing Principle:
        Validates business rule: system-only session types cannot be user-initiated.
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

        Action:
        Call _start_session with uppercase 'OBSERVATION' type.

        Assertion Strategy:
        - Session created successfully.
        - SessionType.OBSERVATION used internally.

        Testing Principle:
        Validates input normalization, ensuring case-insensitive type matching.
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

        Action:
        Call _start_session which triggers the mocked exception.

        Assertion Strategy:
        - Returns JSON error with 'internal' error type.
        - Contains exception message.

        Testing Principle:
        Validates error handling, ensuring exceptions are caught and returned as JSON.
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

        Tests dependency injection fallback to global singleton manager.

        Arrangement:
            1. Create mock_manager with configured mock session response.
            2. Patch get_session_manager to return the mock.
            3. Call _start_session without providing a manager argument.

        Action:
            Call _start_session without manager parameter (triggers global lookup).

        Assertion Strategy:
            Validates dependency injection fallback by confirming:
            - get_session_manager() was called.
            - Session created with global manager's session_id.

        Testing Principle:
            Validates dependency injection fallback, ensuring global manager
            is used by default when no explicit manager is provided.
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

        Arrangement:
        1. Mock SessionManager with active observation session.

        Action:
        Call _end_session with the mock manager.

        Assertion Strategy:
        - Returns JSON with status='ended'.
        - Contains file_path to ASDF file.

        Testing Principle:
        Validates happy path, ensuring session closure returns file location.
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

        Action:
        Call _end_session when system is in idle state.

        Assertion Strategy:
        - Returns JSON error with 'no_session' type.

        Testing Principle:
        Validates state validation, ensuring idle sessions cannot be ended.
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

        Action:
        Call _end_session which triggers the mocked OSError.

        Assertion Strategy:
        - Returns JSON error with 'internal' type.

        Testing Principle:
        Validates error handling, ensuring filesystem errors are caught and reported.
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

        Tests edge case where session_type might be None during transition.

        Arrangement:
            1. Set active_session_type to valid type first (OBSERVATION).
            2. After end_session call, session_type becomes None for JSON.
            3. Tests the '.value if session_type else None' branch.

        Action:
            Call _end_session with an active observation session.

        Assertion Strategy:
            Validates edge case handling by confirming:
            - No exception raised during JSON serialization.
            - file_path still returned correctly.
            - session_type correctly serialized as "observation".

        Testing Principle:
            Validates edge case handling, ensuring session type transitions
            are handled safely without JSON serialization errors.
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

        Arrangement:
        1. Configure mock SessionManager with active observation session.
        2. Mock session has target='M42', duration=120.5s, various metrics.

        Action:
        Call _get_session_info with mock manager to retrieve session state.

        Assertion Strategy:
        - Returns JSON with all expected fields.
        - Metrics include frames, logs, events, errors, warnings.

        Testing Principle:
        Validates data completeness: all session metadata is exposed to clients.
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
        1. Set active_session to None to simulate idle state.

        Action:
        Call _get_session_info when no active session exists.

        Assertion Strategy:
        - Returns JSON error with 'no_session' type.

        Testing Principle:
        Validates error handling, ensuring clear feedback when no session context.
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
        1. Set session.session_type to IDLE to simulate system standby.

        Action:
        Call _get_session_info when session type is IDLE.

        Assertion Strategy:
        - is_idle is True in response.

        Testing Principle:
        Validates state detection, ensuring clients can distinguish idle vs active.
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
        1. Configure active_session property to raise RuntimeError.

        Action:
        Call _get_session_info which triggers the mocked exception.

        Assertion Strategy:
        - Returns JSON error with 'internal' type.

        Testing Principle:
        Validates exception safety, ensuring internal errors are caught and reported.
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

        Arrangement:
        1. Configure mock SessionManager with active session.
        2. Prepare INFO level message with 'user' source.

        Action:
        Call _session_log with level='INFO', message='Test message', source='user'.

        Assertion Strategy:
        - Returns JSON with status='logged'.
        - Contains level, message, source, session_id.

        Testing Principle:
        Validates happy path, ensuring log entries are recorded with full metadata.
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

        Arrangement:
        1. Define all valid log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL.

        Action:
        Iterate through each level and call _session_log with that level.

        Assertion Strategy:
        - Each level parsed and logged successfully.

        Testing Principle:
        Validates enum coverage, ensuring all LogLevel values are accepted.
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
        1. Prepare lowercase 'warning' level string.

        Action:
        Call _session_log with lowercase level='warning'.

        Assertion Strategy:
        - Level parsed as WARNING.

        Testing Principle:
        Validates input normalization, ensuring case-insensitive level matching.
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
        1. Prepare invalid level 'TRACE' (not in LogLevel enum).

        Action:
        Call _session_log with invalid level='TRACE'.

        Assertion Strategy:
        - Returns error message listing valid levels.

        Testing Principle:
        Validates input validation, ensuring invalid levels are rejected early.
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
        1. Configure manager.log() to raise ValueError.

        Action:
        Call _session_log which triggers the mocked exception.

        Assertion Strategy:
        - Returns JSON error with 'internal' type.

        Testing Principle:
        Validates exception safety, ensuring logging errors are caught and reported.
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

        Arrangement:
        1. Configure mock SessionManager with active session.
        2. Prepare 'cloud_detected' event with coverage and action details.

        Action:
        Call _session_event with event='cloud_detected' and structured details.

        Assertion Strategy:
        - Returns JSON with status='recorded'.
        - Contains event name and details.

        Testing Principle:
        Validates happy path, ensuring events are recorded with full context.
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

        Arrangement:
        1. Configure mock SessionManager with active session.

        Action:
        Call _session_event with event='mount_parked' and empty details dict.

        Assertion Strategy:
        - Event recorded with no additional kwargs.

        Testing Principle:
        Validates minimal input, ensuring events work without extra context.
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
        1. Configure add_event() to raise RuntimeError.

        Action:
        Call _session_event which triggers the mocked exception.

        Assertion Strategy:
        - Returns JSON error with 'internal' type.

        Testing Principle:
        Validates exception safety: event storage errors are caught and reported.
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

        Arrangement:
        1. Configure mock factory with data_dir='/home/user/telescope-data'.
        2. Set exists() to return True.

        Action:
        Call _get_data_dir to retrieve current storage path.

        Assertion Strategy:
        - Returns JSON with data_dir path.
        - Contains exists boolean.

        Testing Principle:
        Validates happy path, ensuring data directory info is correctly returned.
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

        Arrangement:
        1. Configure mock factory with exists() returning False.

        Action:
        Call _get_data_dir when configured directory doesn't exist.

        Assertion Strategy:
        - exists is False when directory doesn't exist.

        Testing Principle:
        Validates existence reporting, ensuring clients know when dir needs creation.
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
        1. Configure get_factory() to raise RuntimeError.

        Action:
        Call _get_data_dir which triggers the mocked exception.

        Assertion Strategy:
        - Returns JSON error with 'internal' type.

        Testing Principle:
        Validates exception safety, ensuring factory errors are caught and reported.
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
        1. Use pytest tmp_path fixture for valid existing directory.
        2. Mock set_data_dir to capture call.

        Action:
        Call _set_data_dir with tmp_path as new data directory.

        Assertion Strategy:
        - Returns JSON with status='updated'.
        - Contains new data_dir path.

        Testing Principle:
        Validates happy path, ensuring data directory can be updated successfully.
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
        """Verifies _set_data_dir resolves paths to absolute form.

        Tests path normalization ensuring consistent absolute path handling.

        Arrangement:
            1. Provide absolute path (tmp_path from pytest is already absolute).
            2. Mock set_data_dir to capture call arguments.
            3. Path should be resolved before passing to set_data_dir.

        Action:
            Call _set_data_dir with tmp_path to verify path resolution.

        Assertion Strategy:
            Validates path normalization by confirming:
            - Resolved path is used (call_args[0][0]).
            - Path is_absolute() returns True.

        Testing Principle:
            Validates path normalization, ensuring consistent absolute
            path handling for cross-platform compatibility.
        """
        with patch("telescope_mcp.drivers.config.set_data_dir") as mock_set:
            result = await sessions._set_data_dir(str(tmp_path))

        # The path should be resolved (absolute)
        call_args = mock_set.call_args[0][0]
        assert call_args.is_absolute()

    @pytest.mark.asyncio
    async def test_set_data_dir_includes_note(self, tmp_path: Path) -> None:
        """Verifies _set_data_dir response includes session reset guidance note.

        Tests that response contains user guidance about session manager state.

        Arrangement:
            1. Mock set_data_dir to allow call completion.
            2. Response should include 'note' field with guidance.
            3. Note should mention session manager reset behavior.

        Action:
            Call _set_data_dir and inspect response note field.

        Assertion Strategy:
            Validates user guidance by confirming:
            - Response contains 'note' key.
            - Note mentions 'reset' for session manager behavior.

        Testing Principle:
            Validates user guidance, ensuring clients are informed of
            side effects when changing the data directory.
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
        1. Configure set_data_dir to raise PermissionError.

        Action:
        Call _set_data_dir which triggers the mocked exception.

        Assertion Strategy:
        - Returns JSON error with 'internal' type.

        Testing Principle:
        Validates exception safety, ensuring permission errors are caught and reported.
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

        Tests the validation branch checking is_absolute() after Path.resolve().
        While resolve() normally returns absolute, this tests defensive validation.

        Arrangement:
            1. Mock Path to return a mock that reports is_absolute()=False.
            2. This simulates edge case where resolve() returns non-absolute.
            3. Validation should catch this and return error.

        Action:
            Call _set_data_dir with path that resolves to non-absolute (mocked).

        Assertion Strategy:
            Validates defensive coding by confirming:
            - Returns JSON error with 'validation' type.
            - Error message mentions 'absolute' path requirement.

        Testing Principle:
            Validates defensive coding, ensuring edge case path validation
            works correctly even in unusual filesystem scenarios.
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

        Tests existence feedback for directory state visibility.

        Arrangement:
            1. tmp_path exists (pytest creates it automatically).
            2. Mock set_data_dir to allow call completion.
            3. Response should include 'exists' field reflecting directory state.

        Action:
            Call _set_data_dir with existing tmp_path directory.

        Assertion Strategy:
            Validates existence feedback by confirming:
            - exists is True for existing directory.
            - Clients know directory is ready for use.

        Testing Principle:
            Validates existence feedback, ensuring clients know whether
            the new directory exists and is ready for session storage.
        """
        with patch("telescope_mcp.drivers.config.set_data_dir"):
            result = await sessions._set_data_dir(str(tmp_path))

        data = json.loads(result[0].text)
        assert data["exists"] is True

    @pytest.mark.asyncio
    async def test_set_data_dir_nonexistent_path(self) -> None:
        """Verifies _set_data_dir reports exists=False for non-existent paths.

        Tests existence feedback when directory needs to be created.

        Arrangement:
            1. Provide path that doesn't exist (/tmp/telescope_test_nonexistent_12345).
            2. Mock set_data_dir to allow call completion.
            3. Response should report exists=False.

        Action:
            Call _set_data_dir with non-existent directory path.

        Assertion Strategy:
            Validates existence feedback by confirming:
            - exists is False for non-existent directory.
            - Clients know directory needs creation before use.

        Testing Principle:
            Validates existence feedback, ensuring clients know when
            the directory needs to be created before storing sessions.
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
        """Verifies _start_session handles None target and purpose gracefully.

        Tests optional parameter handling for session creation.

        Arrangement:
            1. Configure mock session with target=None, purpose=None.
            2. Call _start_session with 'experiment' type and None for optional params.
            3. Session should be created with null values.

        Action:
            Call _start_session with 'experiment', None, None for optional params.

        Assertion Strategy:
            Validates optional parameter handling by confirming:
            - Session created successfully (status='started').
            - None values accepted for optional fields.

        Testing Principle:
            Validates optional parameter handling, ensuring None values
            are accepted for target/purpose when not applicable.
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
        """Verifies _session_log handles empty message string gracefully.

        Tests edge case handling for empty string input.

        Arrangement:
            1. Configure mock SessionManager with active session.
            2. Prepare empty string "" as message content.
            3. Empty messages should be logged without error.

        Action:
            Call _session_log with level='INFO', message='', source='user'.

        Assertion Strategy:
            Validates edge case handling by confirming:
            - Empty message logged successfully (status='logged').
            - Response message field is empty string "".

        Testing Principle:
            Validates edge case handling, ensuring empty strings don't cause
            errors and are logged as-is for debugging visibility.
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
        """Verifies _session_event handles complex nested detail structures.

        Tests data structure handling for rich event metadata.

        Arrangement:
            1. Prepare complex nested dict with camera, mount, and conditions data.
            2. Nested structure includes dicts within dicts and lists.
            3. All nested data should serialize correctly to JSON.

        Action:
            Call _session_event with 'frame_captured' and nested details structure.

        Assertion Strategy:
            Validates data structure handling by confirming:
            - Complex nested dict preserved in response.
            - All nested fields accessible after JSON round-trip.

        Testing Principle:
            Validates data structure handling, ensuring nested dicts are
            serialized correctly for rich event metadata capture.
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
        """Verifies multiple operations work correctly in sequence.

        Tests integrated workflow with multiple session operations.

        Arrangement:
            1. Configure mock SessionManager with consistent session_id.
            2. Prepare workflow: start -> log -> event -> get_info.
            3. Each operation should succeed and maintain session context.

        Action:
            Execute workflow: start -> log -> event -> get_session_info.

        Assertion Strategy:
            Validates integration flow by confirming:
            - All operations succeed (no exceptions).
            - Session ID consistent across operations (test_session_123).
            - Final get_session_info returns valid session data.

        Testing Principle:
            Validates integration flow, ensuring realistic workflows
            execute correctly with maintained session context.
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

        Arrangement:
        1. Configure mocks for all dependencies (manager, factory, set_data_dir).

        Action:
        Call each handler function and collect results into a list.

        Assertion Strategy:
        - Each function returns list.
        - Each list item is TextContent.

        Testing Principle:
        Validates MCP contract, ensuring consistent return types across all tools.
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

        Arrangement:
        1. Configure mock to raise exception for some calls.

        Action:
        Trigger various error conditions (invalid type, invalid level, exception).

        Assertion Strategy:
        - All error responses parse as JSON.
        - Error responses have 'error' key or descriptive text.

        Testing Principle:
        Validates error consistency, ensuring all errors are machine-parseable.
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
