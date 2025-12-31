"""Tests for tools/cameras.py targeting 100% coverage.

Covers edge cases and error paths not tested in test_tools.py:
- Empty camera registry
- Exception handling in each handler
- camera_info None path
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from telescope_mcp.tools import cameras


class TestCameraToolsCoverage:
    """Tests targeting uncovered lines in tools/cameras.py."""

    # -------------------------------------------------------------------------
    # _list_cameras coverage
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_list_cameras_empty_registry(self):
        """Verifies _list_cameras returns empty list when no cameras found.

        Arrangement:
        1. Mock registry that returns empty dict from discover().
        2. No cameras in system.

        Action:
        Calls _list_cameras with mock empty registry.

        Assertion Strategy:
        Validates empty response:
        - count is 0
        - cameras list is empty

        Testing Principle:
        Covers line 284-287 (empty camera branch).
        """
        mock_registry = MagicMock()
        mock_registry.discover.return_value = {}

        result = await cameras._list_cameras(registry=mock_registry)

        assert len(result) == 1
        data = json.loads(result[0].text)
        assert data["count"] == 0
        assert data["cameras"] == []

    @pytest.mark.asyncio
    async def test_list_cameras_exception(self):
        """Verifies _list_cameras handles exceptions gracefully.

        Arrangement:
        1. Mock registry that raises RuntimeError.
        2. Simulates hardware failure or driver error.

        Action:
        Calls _list_cameras with failing registry.

        Assertion Strategy:
        Validates error response:
        - Contains "error" key
        - Contains error message

        Testing Principle:
        Covers exception handler in _list_cameras.
        """
        mock_registry = MagicMock()
        mock_registry.discover.side_effect = RuntimeError("Hardware failure")

        result = await cameras._list_cameras(registry=mock_registry)

        assert len(result) == 1
        data = json.loads(result[0].text)
        assert data["error"] == "internal"
        assert "Hardware failure" in data["message"]

    # -------------------------------------------------------------------------
    # _get_camera_info coverage
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_get_camera_info_none_info(self):
        """Verifies _get_camera_info handles None camera.info.

        Arrangement:
        1. Mock camera with info property returning None.
        2. Camera connected but info unavailable.

        Action:
        Calls _get_camera_info with camera that has no info.

        Assertion Strategy:
        Validates error response:
        - error is "internal"
        - message indicates info unavailable

        Testing Principle:
        Covers lines 350-358 (camera_info is None branch).
        """
        mock_camera = MagicMock()
        mock_camera.info = None

        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_camera

        result = await cameras._get_camera_info(0, registry=mock_registry)

        assert len(result) == 1
        data = json.loads(result[0].text)
        assert data["error"] == "internal"
        assert "unavailable" in data["message"].lower()

    @pytest.mark.asyncio
    async def test_get_camera_info_exception(self):
        """Verifies _get_camera_info handles exceptions gracefully.

        Arrangement:
        1. Mock registry that raises KeyError.
        2. Simulates invalid camera_id.

        Action:
        Calls _get_camera_info with failing registry.

        Assertion Strategy:
        Validates error response:
        - Contains "error" key
        - Contains error message

        Testing Principle:
        Covers exception handler in _get_camera_info.
        """
        mock_registry = MagicMock()
        mock_registry.get.side_effect = KeyError("Camera 99 not found")

        result = await cameras._get_camera_info(99, registry=mock_registry)

        assert len(result) == 1
        data = json.loads(result[0].text)
        assert data["error"] == "internal"
        assert "99" in data["message"] or "not found" in data["message"].lower()

    # -------------------------------------------------------------------------
    # _capture_frame coverage
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_capture_frame_exception(self):
        """Verifies _capture_frame handles exceptions gracefully.

        Arrangement:
        1. Mock camera that raises CameraError on capture.
        2. Simulates capture failure (timeout, hardware error).

        Action:
        Calls _capture_frame with failing camera.

        Assertion Strategy:
        Validates error response:
        - Contains "error" key
        - Contains error message

        Testing Principle:
        Covers exception handler in _capture_frame (lines 441-448).
        """
        mock_camera = MagicMock()
        mock_camera.capture.side_effect = RuntimeError("Capture timeout")

        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_camera

        result = await cameras._capture_frame(0, 100000, 50, registry=mock_registry)

        assert len(result) == 1
        data = json.loads(result[0].text)
        assert data["error"] == "internal"
        assert "timeout" in data["message"].lower()

    # -------------------------------------------------------------------------
    # _set_camera_control coverage
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_set_camera_control_exception(self):
        """Verifies _set_camera_control handles exceptions gracefully.

        Arrangement:
        1. Mock camera that raises ValueError on set_control_info.
        2. Simulates invalid control name.

        Action:
        Calls _set_camera_control with failing camera.

        Assertion Strategy:
        Validates error response:
        - Contains "error" key
        - Contains error message

        Testing Principle:
        Covers exception handler in _set_camera_control (lines 499-506).
        """
        mock_camera = MagicMock()
        mock_camera.set_control_info.side_effect = ValueError(
            "Unknown control: InvalidControl"
        )

        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_camera

        result = await cameras._set_camera_control(
            0, "InvalidControl", 100, registry=mock_registry
        )

        assert len(result) == 1
        data = json.loads(result[0].text)
        assert data["error"] == "internal"
        assert "InvalidControl" in data["message"]

    # -------------------------------------------------------------------------
    # _get_camera_control coverage
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_get_camera_control_exception(self):
        """Verifies _get_camera_control handles exceptions gracefully.

        Arrangement:
        1. Mock camera that raises ValueError on get_control_info.
        2. Simulates invalid control name.

        Action:
        Calls _get_camera_control with failing camera.

        Assertion Strategy:
        Validates error response:
        - Contains "error" key
        - Contains error message

        Testing Principle:
        Covers exception handler in _get_camera_control.
        """
        mock_camera = MagicMock()
        mock_camera.get_control_info.side_effect = ValueError(
            "Unknown control: BadControl"
        )

        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_camera

        result = await cameras._get_camera_control(
            0, "BadControl", registry=mock_registry
        )

        assert len(result) == 1
        data = json.loads(result[0].text)
        assert data["error"] == "internal"
        assert "BadControl" in data["message"]

    # -------------------------------------------------------------------------
    # call_tool unknown tool coverage
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # call_tool dispatcher coverage (all branches)
    # -------------------------------------------------------------------------

    @pytest.fixture
    def call_tool_handler(self):
        """Pytest fixture capturing the registered call_tool handler function.

        Creates a mock MCP server to intercept handler registration,
        allowing direct invocation of the call_tool dispatcher for testing
        tool routing without full MCP server setup.

        Business context:
            Tool dispatch testing requires access to the registered handler
            function. This fixture captures it during registration so tests
            can invoke specific tool branches directly, validating routing
            logic and parameter extraction independently of MCP transport.

        Arrangement:
            1. Create MockServer class mimicking MCP Server decorator API.
            2. list_tools() returns identity decorator (captures but passes through).
            3. call_tool() decorator captures handler function in nonlocal variable.
            4. Call cameras.register(mock_server) to trigger registration.

        Args:
            self: Test class instance (implicit for pytest method fixtures).

        Returns:
            Callable: The async function registered with server.call_tool(),
                which dispatches tool invocations to handler functions.
                Signature: async def handler(name: str, params: dict)
                -> list[TextContent]

        Raises:
            None. Fixture setup is deterministic with no external dependencies.

        Example:
            >>> async def test_dispatch(self, call_tool_handler):
            ...     result = await call_tool_handler("list_cameras", {})
            ...     assert len(result) == 1

        Testing Principle:
            Validates registration lifecycle, capturing handlers for isolated
            testing of dispatch logic without full MCP server infrastructure.
        """
        handler = None

        class MockServer:
            def list_tools(self):
                """Return decorator for list_tools registration.

                Business context:
                Mock MCP server list_tools() for handler capture.

                Args:
                    None: This method takes no arguments.

                Returns:
                    Callable: Identity decorator that passes through function.

                Raises:
                    None: This method does not raise any exceptions.

                Example:
                    >>> mock = MockServer()
                    >>> @mock.list_tools()
                    ... async def handler(): pass
                """

                def decorator(fn):
                    """Identity decorator returning function unchanged.

                    Stores the decorated function unchanged, enabling
                    registration without modification.

                    Business context:
                    Intercepts MCP registration to enable isolated handler testing
                    without requiring full server lifecycle.

                    Args:
                        fn: The handler function being registered via decorator.

                    Returns:
                        The original function unchanged, preserving call signature.

                    Raises:
                        None: This function does not raise any exceptions.

                    Example:
                        >>> @decorator
                        ... async def handler(): pass
                        >>> assert handler is not None
                    """
                    return fn

                return decorator

            def call_tool(self):
                """Return decorator that captures call_tool handler.

                Business context:
                Captures registered handler for isolated testing.

                Args:
                    None: This method takes no arguments.

                Returns:
                    Callable: Decorator that stores handler in nonlocal variable.

                Raises:
                    None: This method does not raise any exceptions.

                Example:
                    >>> mock = MockServer()
                    >>> @mock.call_tool()
                    ... async def handler(name, args): pass
                """

                def decorator(fn):
                    """Capture handler function in nonlocal variable for test access.

                    Stores the decorated function in nonlocal variable,
                    enabling direct handler invocation in test assertions.

                    Business context:
                    Intercepts MCP registration to enable isolated handler testing
                    without requiring full server lifecycle.

                    Args:
                        fn: The handler function being registered via decorator.

                    Returns:
                        The original function unchanged, preserving call signature.

                    Raises:
                        None: This function does not raise any exceptions.

                    Example:
                        >>> @decorator
                        ... async def handler(): pass
                        >>> assert captured_handler is handler
                    """
                    nonlocal handler
                    handler = fn
                    return fn

                return decorator

        mock_server = MockServer()
        cameras.register(mock_server)
        return handler

    @pytest.mark.asyncio
    async def test_call_tool_list_cameras(self, call_tool_handler):
        """Verifies call_tool dispatches list_cameras correctly to _list_cameras.

        Tests the dispatcher routing branch for camera enumeration tool.

        Arrangement:
            1. Capture call_tool handler via fixture.
            2. Mock _list_cameras to return canned response.
            3. Prepare empty params dict (list_cameras takes no arguments).

        Action:
            Invokes call_tool handler with name="list_cameras" and empty params.

        Assertion Strategy:
            Validates dispatcher routing by confirming:
            - _list_cameras was called exactly once.
            - Correct handler invoked for tool name.

        Testing Principle:
            Validates dispatcher routing, ensuring list_cameras branch executes
            and routes to the correct internal handler function.
        """
        with patch.object(cameras, "_list_cameras") as mock_fn:
            mock_fn.return_value = [MagicMock(text='{"count": 0}')]
            result = await call_tool_handler("list_cameras", {})
            mock_fn.assert_called_once()

    @pytest.mark.asyncio
    async def test_call_tool_get_camera_info(self, call_tool_handler):
        """Verifies call_tool dispatches get_camera_info with camera_id extraction.

        Tests the dispatcher routing and parameter extraction for camera info tool.

        Arrangement:
            1. Capture call_tool handler via fixture.
            2. Mock _get_camera_info to return empty dict response.
            3. Prepare params dict with camera_id=0.

        Action:
            Invokes call_tool handler with name="get_camera_info" and params.

        Assertion Strategy:
            Validates parameter passing by confirming:
            - _get_camera_info called with correct camera_id (0).
            - Argument extracted correctly from params dict.

        Testing Principle:
            Validates parameter passing, ensuring camera_id flows through
            dispatcher to the internal handler function correctly.
        """
        with patch.object(cameras, "_get_camera_info") as mock_fn:
            mock_fn.return_value = [MagicMock(text="{}")]
            result = await call_tool_handler("get_camera_info", {"camera_id": 0})
            mock_fn.assert_called_once_with(0)

    @pytest.mark.asyncio
    async def test_call_tool_capture_frame(self, call_tool_handler):
        """Verifies call_tool dispatches capture_frame with all parameters.

        Tests the dispatcher routing and multi-parameter extraction for capture tool.

        Arrangement:
            1. Capture call_tool handler via fixture.
            2. Mock _capture_frame to return empty response.
            3. Prepare params dict with camera_id=0, exposure_us=50000, gain=25.

        Action:
            Invokes call_tool handler with name="capture_frame" and full params.

        Assertion Strategy:
            Validates multi-parameter extraction by confirming:
            - _capture_frame called with (0, 50000, 25).
            - All three parameters (camera_id, exposure_us, gain) extracted.

        Testing Principle:
            Validates multi-parameter extraction, ensuring all capture arguments
            flow through dispatcher correctly to the internal handler.
        """
        with patch.object(cameras, "_capture_frame") as mock_fn:
            mock_fn.return_value = [MagicMock(text="{}")]
            result = await call_tool_handler(
                "capture_frame", {"camera_id": 0, "exposure_us": 50000, "gain": 25}
            )
            mock_fn.assert_called_once_with(0, 50000, 25)

    @pytest.mark.asyncio
    async def test_call_tool_capture_frame_defaults(self, call_tool_handler):
        """Verifies call_tool applies default values for omitted capture_frame params.

        Tests default parameter handling when optional arguments are not provided.

        Arrangement:
            1. Capture call_tool handler via fixture.
            2. Mock _capture_frame to return empty response.
            3. Prepare params with only camera_id=1 (omit exposure_us and gain).

        Action:
            Invokes call_tool handler with name="capture_frame" missing optional params.

        Assertion Strategy:
            Validates default parameter handling by confirming:
            - _capture_frame called with camera_id=1.
            - Default exposure_us=100000 applied.
            - Default gain=50 applied.

        Testing Principle:
            Validates default parameter handling, ensuring omitted optional arguments
            receive sensible defaults for convenient API usage.
        """
        with patch.object(cameras, "_capture_frame") as mock_fn:
            mock_fn.return_value = [MagicMock(text="{}")]
            result = await call_tool_handler("capture_frame", {"camera_id": 1})
            mock_fn.assert_called_once_with(1, 100000, 50)  # defaults

    @pytest.mark.asyncio
    async def test_call_tool_set_camera_control(self, call_tool_handler):
        """Verifies call_tool dispatches set_camera_control with all arguments.

        Tests the dispatcher routing for camera control write operations.

        Arrangement:
            1. Capture call_tool handler via fixture.
            2. Mock _set_camera_control to return empty response.
            3. Prepare params with camera_id=0, control="Gain", value=100.

        Action:
            Invokes call_tool handler with name="set_camera_control" and params.

        Assertion Strategy:
            Validates control mutation routing by confirming:
            - _set_camera_control called with (0, "Gain", 100).
            - Control name and value extracted correctly from params.

        Testing Principle:
            Validates control mutation routing, ensuring write operations
            dispatch correctly to the internal handler with all arguments.
        """
        with patch.object(cameras, "_set_camera_control") as mock_fn:
            mock_fn.return_value = [MagicMock(text="{}")]
            result = await call_tool_handler(
                "set_camera_control",
                {"camera_id": 0, "control": "Gain", "value": 100},
            )
            mock_fn.assert_called_once_with(0, "Gain", 100)

    @pytest.mark.asyncio
    async def test_call_tool_get_camera_control(self, call_tool_handler):
        """Verifies call_tool dispatches get_camera_control with correct arguments.

        Tests the dispatcher routing for camera control read operations.

        Arrangement:
            1. Capture call_tool handler via fixture.
            2. Mock _get_camera_control to return empty response.
            3. Prepare params with camera_id=0, control="Temperature".

        Action:
            Invokes call_tool handler with name="get_camera_control" and params.

        Assertion Strategy:
            Validates control query routing by confirming:
            - _get_camera_control called with (0, "Temperature").
            - Control name extracted correctly for read operation.

        Testing Principle:
            Validates control query routing, ensuring read operations
            dispatch correctly to the internal handler function.
        """
        with patch.object(cameras, "_get_camera_control") as mock_fn:
            mock_fn.return_value = [MagicMock(text="{}")]
            result = await call_tool_handler(
                "get_camera_control", {"camera_id": 0, "control": "Temperature"}
            )
            mock_fn.assert_called_once_with(0, "Temperature")

    @pytest.mark.asyncio
    async def test_call_tool_unknown_tool(self, call_tool_handler):
        """Verifies call_tool returns structured error for unknown tool names.

        Tests the error handling path when an unrecognized tool is requested.

        Arrangement:
            1. Capture call_tool handler via fixture.
            2. No mocks needed - testing error path directly.
            3. Prepare unrecognized tool name "nonexistent_tool".

        Action:
            Invokes call_tool handler with name="nonexistent_tool" and empty params.

        Assertion Strategy:
            Validates error handling by confirming:
            - Response contains error="unknown_tool".
            - Message includes the unknown tool name for debugging.

        Testing Principle:
            Validates error handling, ensuring unknown tools return clear,
            structured errors that help clients identify invalid requests.
        """
        result = await call_tool_handler("nonexistent_tool", {})

        assert len(result) == 1
        data = json.loads(result[0].text)
        assert data["error"] == "unknown_tool"
        assert "nonexistent_tool" in data["message"]

    # -------------------------------------------------------------------------
    # TOOLS constant coverage
    # -------------------------------------------------------------------------

    def test_tools_constant_exists(self):
        """Verifies TOOLS constant is properly defined with all camera tools.

        Tests the module-level constant that defines available camera MCP tools.

        Arrangement:
            1. No setup needed - testing module-level constant.
            2. cameras.TOOLS should be populated at module load time.

        Action:
            Accesses cameras.TOOLS and extracts tool names via list comprehension.

        Assertion Strategy:
            Validates module API surface by confirming:
            - TOOLS attribute exists on module.
            - Exactly 5 tools defined (list_cameras, get_camera_info, capture_frame,
              set_camera_control, get_camera_control).
            - All expected tool names present in the list.

        Testing Principle:
            Validates module API surface, ensuring TOOLS constant is complete
            for MCP server registration and client discovery protocols.
        """
        assert hasattr(cameras, "TOOLS")
        assert len(cameras.TOOLS) == 5
        tool_names = [t.name for t in cameras.TOOLS]
        assert "list_cameras" in tool_names
        assert "get_camera_info" in tool_names
        assert "capture_frame" in tool_names
        assert "set_camera_control" in tool_names
        assert "get_camera_control" in tool_names

    # -------------------------------------------------------------------------
    # register function coverage
    # -------------------------------------------------------------------------

    def test_register_attaches_handlers(self):
        """Verifies register() attaches both list_tools and call_tool handlers.

        Tests the registration function that connects handlers to MCP server.

        Arrangement:
            1. Create MockServer with list_tools() and call_tool() decorator methods.
            2. Track decorator invocations via nonlocal boolean flags.
            3. Decorators return identity function to allow registration.

        Action:
            Calls cameras.register(mock_server) to attach handlers to server.

        Assertion Strategy:
            Validates registration lifecycle by confirming:
            - list_tools() decorator was invoked (tool listing capability).
            - call_tool() decorator was invoked (tool invocation capability).

        Testing Principle:
            Validates registration lifecycle, ensuring both handlers attach
            correctly for MCP server tool discovery and invocation protocols.
        """
        list_tools_called = False
        call_tool_called = False

        class MockServer:
            def list_tools(self):
                """Track list_tools decorator invocation.

                Business context:
                Verifies register() calls list_tools() on server.

                Args:
                    None: This method takes no arguments.

                Returns:
                    Callable: Identity decorator for registration.

                Raises:
                    None: This method does not raise any exceptions.

                Example:
                    >>> mock = MockServer()
                    >>> @mock.list_tools()
                    ... async def handler(): pass
                """
                nonlocal list_tools_called
                list_tools_called = True

                def decorator(fn):
                    """Identity decorator returning function unchanged.

                    Stores the decorated function unchanged, enabling
                    registration without modification.

                    Business context:
                    Intercepts MCP registration to enable isolated handler testing
                    without requiring full server lifecycle.

                    Args:
                        fn: The handler function being registered via decorator.

                    Returns:
                        The original function unchanged, preserving call signature.

                    Raises:
                        None: This function does not raise any exceptions.

                    Example:
                        >>> @decorator
                        ... async def handler(): pass
                        >>> assert handler is not None
                    """
                    return fn

                return decorator

            def call_tool(self):
                """Track call_tool decorator invocation.

                Business context:
                Verifies register() calls call_tool() on server.

                Args:
                    None: This method takes no arguments.

                Returns:
                    Callable: Identity decorator for registration.

                Raises:
                    None: This method does not raise any exceptions.

                Example:
                    >>> mock = MockServer()
                    >>> @mock.call_tool()
                    ... async def handler(name, args): pass
                """
                nonlocal call_tool_called
                call_tool_called = True

                def decorator(fn):
                    """Identity decorator returning function unchanged.

                    Stores the decorated function unchanged, enabling
                    registration without modification.

                    Business context:
                    Intercepts MCP registration to enable isolated handler testing
                    without requiring full server lifecycle.

                    Args:
                        fn: The handler function being registered via decorator.

                    Returns:
                        The original function unchanged, preserving call signature.

                    Raises:
                        None: This function does not raise any exceptions.

                    Example:
                        >>> @decorator
                        ... async def handler(): pass
                        >>> assert handler is not None
                    """
                    return fn

                return decorator

        mock_server = MockServer()
        cameras.register(mock_server)

        assert list_tools_called
        assert call_tool_called

    @pytest.mark.asyncio
    async def test_list_tools_returns_tools(self):
        """Verifies list_tools handler returns TOOLS constant for discovery.

        Tests that the registered list_tools handler provides complete tool list.

        Arrangement:
            1. Create MockServer that captures list_tools handler via decorator.
            2. Register cameras module to trigger handler registration.
            3. Captured handler stored in nonlocal variable.

        Action:
            Invokes the captured list_tools handler directly via await.

        Assertion Strategy:
            Validates tool discovery by confirming:
            - Returned list equals cameras.TOOLS exactly.
            - All defined tools exposed for client enumeration.

        Testing Principle:
            Validates tool discovery, ensuring clients receive complete
            tool list for camera operations via MCP protocol.
        """
        list_tools_handler = None

        class MockServer:
            def list_tools(self):
                """Return decorator that captures list_tools handler.

                Business context:
                Captures handler for direct invocation testing.

                Args:
                    None: This method takes no arguments.

                Returns:
                    Callable: Decorator that stores handler in nonlocal variable.

                Raises:
                    None: This method does not raise any exceptions.

                Example:
                    >>> mock = MockServer()
                    >>> @mock.list_tools()
                    ... async def handler(): pass
                """

                def decorator(fn):
                    """Capture handler function in nonlocal variable for test access.

                    Stores the decorated function in nonlocal variable,
                    enabling direct handler invocation in test assertions.

                    Business context:
                    Intercepts MCP registration to enable isolated handler testing
                    without requiring full server lifecycle.

                    Args:
                        fn: The handler function being registered via decorator.

                    Returns:
                        The original function unchanged, preserving call signature.

                    Raises:
                        None: This function does not raise any exceptions.

                    Example:
                        >>> @decorator
                        ... async def handler(): pass
                        >>> assert captured_handler is handler
                    """
                    nonlocal list_tools_handler
                    list_tools_handler = fn
                    return fn

                return decorator

            def call_tool(self):
                """Return decorator for call_tool registration.

                Business context:
                Mock call_tool() - not needed for this test.

                Args:
                    None: This method takes no arguments.

                Returns:
                    Callable: Identity decorator for registration.

                Raises:
                    None: This method does not raise any exceptions.

                Example:
                    >>> mock = MockServer()
                    >>> @mock.call_tool()
                    ... async def handler(name, args): pass
                """

                def decorator(fn):
                    """Identity decorator returning function unchanged.

                    Stores the decorated function unchanged, enabling
                    registration without modification.

                    Business context:
                    Intercepts MCP registration to enable isolated handler testing
                    without requiring full server lifecycle.

                    Args:
                        fn: The handler function being registered via decorator.

                    Returns:
                        The original function unchanged, preserving call signature.

                    Raises:
                        None: This function does not raise any exceptions.

                    Example:
                        >>> @decorator
                        ... async def handler(): pass
                        >>> assert handler is not None
                    """
                    return fn

                return decorator

        mock_server = MockServer()
        cameras.register(mock_server)

        result = await list_tools_handler()
        assert result == cameras.TOOLS
