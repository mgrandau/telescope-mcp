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
        """Fixture that captures the registered call_tool handler.

        Returns the async function registered with server.call_tool().
        """
        handler = None

        class MockServer:
            def list_tools(self):
                def decorator(fn):
                    return fn

                return decorator

            def call_tool(self):
                def decorator(fn):
                    nonlocal handler
                    handler = fn
                    return fn

                return decorator

        mock_server = MockServer()
        cameras.register(mock_server)
        return handler

    @pytest.mark.asyncio
    async def test_call_tool_list_cameras(self, call_tool_handler):
        """Verifies call_tool dispatches list_cameras correctly.

        Testing Principle:
        Covers line 215 (list_cameras branch).
        """
        with patch.object(cameras, "_list_cameras") as mock_fn:
            mock_fn.return_value = [MagicMock(text='{"count": 0}')]
            result = await call_tool_handler("list_cameras", {})
            mock_fn.assert_called_once()

    @pytest.mark.asyncio
    async def test_call_tool_get_camera_info(self, call_tool_handler):
        """Verifies call_tool dispatches get_camera_info correctly.

        Testing Principle:
        Covers line 217 (get_camera_info branch).
        """
        with patch.object(cameras, "_get_camera_info") as mock_fn:
            mock_fn.return_value = [MagicMock(text="{}")]
            result = await call_tool_handler("get_camera_info", {"camera_id": 0})
            mock_fn.assert_called_once_with(0)

    @pytest.mark.asyncio
    async def test_call_tool_capture_frame(self, call_tool_handler):
        """Verifies call_tool dispatches capture_frame correctly.

        Testing Principle:
        Covers lines 219-223 (capture_frame branch).
        """
        with patch.object(cameras, "_capture_frame") as mock_fn:
            mock_fn.return_value = [MagicMock(text="{}")]
            result = await call_tool_handler(
                "capture_frame", {"camera_id": 0, "exposure_us": 50000, "gain": 25}
            )
            mock_fn.assert_called_once_with(0, 50000, 25)

    @pytest.mark.asyncio
    async def test_call_tool_capture_frame_defaults(self, call_tool_handler):
        """Verifies call_tool uses defaults for capture_frame.

        Testing Principle:
        Covers default values in capture_frame dispatch.
        """
        with patch.object(cameras, "_capture_frame") as mock_fn:
            mock_fn.return_value = [MagicMock(text="{}")]
            result = await call_tool_handler("capture_frame", {"camera_id": 1})
            mock_fn.assert_called_once_with(1, 100000, 50)  # defaults

    @pytest.mark.asyncio
    async def test_call_tool_set_camera_control(self, call_tool_handler):
        """Verifies call_tool dispatches set_camera_control correctly.

        Testing Principle:
        Covers lines 225-229 (set_camera_control branch).
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
        """Verifies call_tool dispatches get_camera_control correctly.

        Testing Principle:
        Covers lines 231-234 (get_camera_control branch).
        """
        with patch.object(cameras, "_get_camera_control") as mock_fn:
            mock_fn.return_value = [MagicMock(text="{}")]
            result = await call_tool_handler(
                "get_camera_control", {"camera_id": 0, "control": "Temperature"}
            )
            mock_fn.assert_called_once_with(0, "Temperature")

    @pytest.mark.asyncio
    async def test_call_tool_unknown_tool(self, call_tool_handler):
        """Verifies call_tool returns error for unknown tool names.

        Testing Principle:
        Covers else branch in call_tool dispatcher (lines 235-241).
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
        """Verifies TOOLS constant is properly defined.

        Testing Principle:
        Ensures module-level TOOLS list is accessible and non-empty.
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
        """Verifies register() attaches handlers to server.

        Testing Principle:
        Ensures register() calls server.list_tools() and server.call_tool().
        """
        list_tools_called = False
        call_tool_called = False

        class MockServer:
            def list_tools(self):
                nonlocal list_tools_called
                list_tools_called = True

                def decorator(fn):
                    return fn

                return decorator

            def call_tool(self):
                nonlocal call_tool_called
                call_tool_called = True

                def decorator(fn):
                    return fn

                return decorator

        mock_server = MockServer()
        cameras.register(mock_server)

        assert list_tools_called
        assert call_tool_called

    @pytest.mark.asyncio
    async def test_list_tools_returns_tools(self):
        """Verifies list_tools handler returns TOOLS constant.

        Testing Principle:
        Ensures the list_tools handler returns the module TOOLS.
        """
        list_tools_handler = None

        class MockServer:
            def list_tools(self):
                def decorator(fn):
                    nonlocal list_tools_handler
                    list_tools_handler = fn
                    return fn

                return decorator

            def call_tool(self):
                def decorator(fn):
                    return fn

                return decorator

        mock_server = MockServer()
        cameras.register(mock_server)

        result = await list_tools_handler()
        assert result == cameras.TOOLS
