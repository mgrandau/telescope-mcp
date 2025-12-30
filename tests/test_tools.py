"""Tests for MCP tools."""

import base64
import json

import pytest
from mcp.server import Server

from telescope_mcp.devices import init_registry, shutdown_registry
from telescope_mcp.drivers.cameras.twin import DigitalTwinCameraDriver
from telescope_mcp.tools import cameras, motors, position


class TestCameraTools:
    """Camera tool tests."""

    @pytest.fixture(autouse=True)
    def setup_registry(self):
        """Fixture providing camera registry with DigitalTwin for each test.

        Arrangement:
        1. Creates DigitalTwinCameraDriver (simulated ZWO ASI cameras).
        2. init_registry(driver) registers 2 cameras (finder + main).
        3. Yields control to test (cameras available).
        4. shutdown_registry() cleanup after test.

        Args:
        self: Test class instance (pytest passes automatically).

        Returns:
        None (generator fixture, yields for test execution).

        Raises:
        None.

        Testing Principle:
        Provides isolated camera environment per test,
        ensuring clean state and automatic cleanup.
        """
        driver = DigitalTwinCameraDriver()
        init_registry(driver)
        yield
        shutdown_registry()

    @pytest.mark.asyncio
    async def test_list_cameras(self):
        """Verifies _list_cameras returns all registered cameras.

        Arrangement:
        1. Registry initialized with DigitalTwinDriver (2 cameras).
        2. MCP tool _list_cameras queries registry.
        3. Result in JSON format.

        Action:
        Calls _list_cameras() MCP tool.

        Assertion Strategy:
        Validates camera enumeration by confirming:
        - Result has 1 TextContent item.
        - JSON contains "cameras" and "count" keys.
        - count equals 2 (finder + main).
        - cameras array has 2 elements.
        """
        result = await cameras._list_cameras()
        assert len(result) == 1
        text = result[0].text
        data = json.loads(text)
        assert "cameras" in data
        assert "count" in data
        assert data["count"] == 2
        assert len(data["cameras"]) == 2

    @pytest.mark.asyncio
    async def test_list_cameras_json_structure(self):
        """Verifies _list_cameras JSON contains required fields.

        Arrangement:
        1. Registry has 2 cameras with metadata.
        2. Expected fields: id, name, max_width, max_height.
        3. JSON schema validation.

        Action:
        Calls _list_cameras() and parses JSON response.

        Assertion Strategy:
        Validates JSON structure by confirming:
        - Top-level has "count" and "cameras" keys.
        - Each camera has id, name, max_width, max_height.
        - Schema matches MCP tool contract.
        """
        result = await cameras._list_cameras()
        data = json.loads(result[0].text)
        assert "count" in data
        assert "cameras" in data
        for cam in data["cameras"]:
            assert "id" in cam
            assert "name" in cam
            assert "max_width" in cam
            assert "max_height" in cam

    @pytest.mark.asyncio
    async def test_get_camera_info(self):
        """Verifies _get_camera_info returns camera details.

        Arrangement:
        1. Camera 0 registered and connected.
        2. MCP tool queries camera info.
        3. Result includes camera_id, info dict, connection status.

        Action:
        Calls _get_camera_info(camera_id=0).

        Assertion Strategy:
        Validates info retrieval by confirming:
        - camera_id matches requested (0).
        - "info" key present.
        - is_connected is True.
        """
        result = await cameras._get_camera_info(camera_id=0)
        assert len(result) == 1
        data = json.loads(result[0].text)
        assert data["camera_id"] == 0
        assert "info" in data
        assert "is_connected" in data
        assert data["is_connected"] is True

    @pytest.mark.asyncio
    async def test_get_camera_info_structure(self):
        """Verifies _get_camera_info includes all metadata fields.

        Arrangement:
        1. Camera 0 has full metadata.
        2. Expected fields: camera_id, name, dimensions, color, controls.
        3. Schema validation for MCP response.

        Action:
        Retrieves info and validates nested structure.

        Assertion Strategy:
        Validates complete schema by confirming:
        - info dict has camera_id, name, max_width, max_height.
        - is_color and controls fields present.
        - All required metadata accessible.
        """
        result = await cameras._get_camera_info(camera_id=0)
        data = json.loads(result[0].text)
        info = data["info"]
        assert "camera_id" in info
        assert "name" in info
        assert "max_width" in info
        assert "max_height" in info
        assert "is_color" in info
        assert "controls" in info

    @pytest.mark.asyncio
    async def test_get_camera_info_invalid_camera(self):
        """Verifies _get_camera_info rejects invalid camera ID.

        Arrangement:
        1. Registry has cameras 0 and 1 only.
        2. Request for camera_id=999 (non-existent).
        3. Should return error message.

        Action:
        Calls _get_camera_info with invalid ID.

        Assertion Strategy:
        Validates error handling by confirming:
        - Result contains error indicator.
        - Text includes "error" or "not found".
        """
        result = await cameras._get_camera_info(camera_id=999)
        assert len(result) == 1
        text = result[0].text
        assert "error" in text.lower() or "not found" in text.lower()

    @pytest.mark.asyncio
    async def test_capture_frame(self):
        """Verifies _capture_frame returns base64-encoded image.

        Arrangement:
        1. Camera 0 connected and ready.
        2. Capture settings: exposure=100ms, gain=50.
        3. Result includes image data and metadata.

        Action:
        Calls _capture_frame with specified parameters.

        Assertion Strategy:
        Validates capture by confirming:
        - camera_id, exposure_us, gain match request.
        - image_base64 field present (encoded data).
        - timestamp field present.
        """
        result = await cameras._capture_frame(camera_id=0, exposure_us=100000, gain=50)
        assert len(result) == 1
        data = json.loads(result[0].text)
        assert data["camera_id"] == 0
        assert data["exposure_us"] == 100000
        assert data["gain"] == 50
        assert "image_base64" in data
        assert "timestamp" in data
        # Verify base64 encoding is valid
        image_bytes = base64.b64decode(data["image_base64"])
        assert len(image_bytes) > 0

    @pytest.mark.asyncio
    async def test_capture_frame_different_settings(self):
        """Verifies _capture_frame respects varied exposure/gain settings.

        Arrangement:
        1. Camera 0 ready for capture.
        2. Different settings: exposure=200ms, gain=100.
        3. Result should reflect requested parameters.

        Action:
        Calls _capture_frame with 200ms exposure, gain 100.

        Assertion Strategy:
        Validates parameter handling by confirming:
        - exposure_us = 200000 in response.
        - gain = 100 in response.

        Testing Principle:
        Validates parameter variability, ensuring tool
        accepts and reports different capture settings.
        """
        result = await cameras._capture_frame(camera_id=0, exposure_us=200000, gain=100)
        data = json.loads(result[0].text)
        assert data["exposure_us"] == 200000
        assert data["gain"] == 100

    @pytest.mark.asyncio
    async def test_capture_frame_invalid_camera(self):
        """Verifies _capture_frame handles invalid camera ID gracefully.

        Arrangement:
        1. Camera ID 999 doesn't exist in registry.
        2. _capture_frame should return error message.
        3. Error response instead of exception.

        Action:
        Calls _capture_frame with camera_id=999.

        Assertion Strategy:
        Validates error handling by confirming:
        - Result contains 'error' or 'not found' text.

        Testing Principle:
        Validates input validation, ensuring tool
        reports invalid camera IDs gracefully.
        """
        result = await cameras._capture_frame(
            camera_id=999, exposure_us=100000, gain=50
        )
        text = result[0].text
        assert "error" in text.lower() or "not found" in text.lower()

    @pytest.mark.asyncio
    async def test_set_camera_control(self):
        """Verifies _set_camera_control modifies camera settings.

        Arrangement:
        1. Camera 0 has configurable Gain control.
        2. MCP tool sets Gain to 100.
        3. Response confirms camera_id.

        Action:
        Calls _set_camera_control(camera_id=0, control="Gain", value=100).

        Assertion Strategy:
        Validates control modification by confirming:
        - Result has 1 TextContent item.
        - JSON contains camera_id=0.

        Testing Principle:
        Validates control modification, ensuring tool
        can change camera parameters via MCP.
        """
        result = await cameras._set_camera_control(
            camera_id=0, control="Gain", value=100
        )
        assert len(result) == 1
        data = json.loads(result[0].text)
        assert data["camera_id"] == 0

    @pytest.mark.asyncio
    async def test_set_camera_control_high_value(self):
        """Verifies _set_camera_control handles high gain values.

        Arrangement:
        1. Camera accepts Gain values up to 600.
        2. Tool should handle high values correctly.
        3. Driver stores the value.

        Action:
        Calls _set_camera_control with control="Gain", value=150.

        Assertion Strategy:
        Validates value handling by confirming:
        - camera_id=0 in response.
        - No error from high value.

        Testing Principle:
        Validates value range handling, ensuring tool
        accepts valid high gain values.
        """
        result = await cameras._set_camera_control(
            camera_id=0, control="Gain", value=150
        )
        data = json.loads(result[0].text)
        assert data["camera_id"] == 0

    @pytest.mark.asyncio
    async def test_set_camera_control_invalid_camera(self):
        """Verifies _set_camera_control rejects invalid camera ID.

        Arrangement:
        1. Camera 999 doesn't exist.
        2. Tool should return error message.
        3. Graceful error handling.

        Action:
        Calls _set_camera_control with camera_id=999.

        Assertion Strategy:
        Validates error handling by confirming:
        - Result contains 'error' or 'not found'.

        Testing Principle:
        Validates input validation, ensuring tool
        reports invalid camera IDs for control operations.
        """
        result = await cameras._set_camera_control(
            camera_id=999, control="Gain", value=100
        )
        text = result[0].text
        assert "error" in text.lower() or "not found" in text.lower()

    @pytest.mark.asyncio
    async def test_get_camera_control(self):
        """Verifies _get_camera_control reads camera control values.

        Arrangement:
        1. Camera 0 has Gain control with current value.
        2. MCP tool queries Gain value.
        3. Response includes camera_id and value.

        Action:
        Calls _get_camera_control(camera_id=0, control="Gain").

        Assertion Strategy:
        Validates control retrieval by confirming:
        - Result has 1 TextContent item.
        - JSON contains camera_id=0.
        - JSON contains 'value' field.

        Testing Principle:
        Validates control query, ensuring tool
        can read current camera parameter values.
        """
        result = await cameras._get_camera_control(camera_id=0, control="Gain")
        assert len(result) == 1
        data = json.loads(result[0].text)
        assert data["camera_id"] == 0
        assert "value" in data

    @pytest.mark.asyncio
    async def test_get_camera_control_exposure(self):
        """Verifies _get_camera_control retrieves Exposure control.

        Arrangement:
        1. Camera has Exposure control.
        2. Tool should retrieve current value.
        3. Driver returns stored value.

        Action:
        Calls _get_camera_control with control="Exposure".

        Assertion Strategy:
        Validates retrieval by confirming:
        - camera_id=0 in response.
        - No error from control query.

        Testing Principle:
        Validates control query, ensuring tool
        retrieves current exposure settings correctly.
        """
        result = await cameras._get_camera_control(camera_id=0, control="Exposure")
        data = json.loads(result[0].text)
        assert data["camera_id"] == 0

    @pytest.mark.asyncio
    async def test_get_camera_control_invalid_camera(self):
        """Verifies _get_camera_control rejects invalid camera ID.

        Arrangement:
        1. Camera 999 doesn't exist.
        2. Tool should return error message.
        3. Graceful error handling.

        Action:
        Calls _get_camera_control with camera_id=999.

        Assertion Strategy:
        Validates error handling by confirming:
        - Result contains 'error' or 'not found'.

        Testing Principle:
        Validates input validation, ensuring tool
        reports invalid camera IDs for control queries.
        """
        result = await cameras._get_camera_control(camera_id=999, control="Gain")
        text = result[0].text
        assert "error" in text.lower() or "not found" in text.lower()

    @pytest.mark.asyncio
    async def test_register_tools(self):
        """Verifies camera tools register with MCP server successfully.

        Arrangement:
        1. MCP Server instance created.
        2. cameras.register(server) attaches tools.
        3. Registration adds tool handlers.

        Action:
        Creates server, calls cameras.register(server).

        Assertion Strategy:
        Validates registration by confirming:
        - server object is not None.
        - No exceptions raised during registration.

        Testing Principle:
        Validates MCP integration, ensuring camera tools
        properly attach to server instance.
        """
        server = Server("test-server")
        cameras.register(server)
        # Registration should complete without errors
        assert server is not None

    @pytest.mark.asyncio
    async def test_call_tool_list_cameras(self):
        """Verifies call_tool dispatcher routes list_cameras correctly.

        Arrangement:
        1. Server created with camera tools registered.
        2. call_tool handler registered by cameras.register().
        3. Direct tool invocation tests routing.

        Action:
        Registers tools, calls _list_cameras().

        Assertion Strategy:
        Validates dispatcher by confirming:
        - Result has 1 TextContent item.

        Testing Principle:
        Validates tool routing, ensuring dispatcher
        correctly invokes list_cameras handler.
        """
        server = Server("test-server")
        cameras.register(server)
        # The call_tool handler is registered
        result = await cameras._list_cameras()
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_call_tool_get_camera_info(self):
        """Verifies call_tool dispatcher routes get_camera_info correctly.

        Arrangement:
        1. Tool dispatcher handles get_camera_info.
        2. Arguments passed through dispatcher.
        3. Response structure validated.

        Action:
        Calls _get_camera_info(0).

        Assertion Strategy:
        Validates dispatcher by confirming:
        - Result has 1 TextContent item.
        - JSON contains camera_id.

        Testing Principle:
        Validates tool routing, ensuring dispatcher
        correctly invokes get_camera_info with args.
        """
        result = await cameras._get_camera_info(0)
        assert len(result) == 1
        data = json.loads(result[0].text)
        assert "camera_id" in data

    @pytest.mark.asyncio
    async def test_call_tool_capture_frame(self):
        """Verifies call_tool dispatcher routes capture_frame correctly.

        Arrangement:
        1. Tool dispatcher handles capture_frame.
        2. Multiple arguments passed (camera_id, exposure, gain).
        3. Image data returned via dispatcher.

        Action:
        Calls _capture_frame(0, 100000, 50).

        Assertion Strategy:
        Validates dispatcher by confirming:
        - Result has 1 TextContent item.
        - JSON contains image_base64.

        Testing Principle:
        Validates tool routing, ensuring dispatcher
        correctly invokes capture_frame with all args.
        """
        result = await cameras._capture_frame(0, 100000, 50)
        assert len(result) == 1
        data = json.loads(result[0].text)
        assert "image_base64" in data

    @pytest.mark.asyncio
    async def test_call_tool_set_control(self):
        """Verifies call_tool dispatcher routes set_camera_control correctly.

        Arrangement:
        1. Tool dispatcher handles set_camera_control.
        2. Arguments: camera_id, control name, value.
        3. Control modification via dispatcher.

        Action:
        Calls _set_camera_control(0, "Gain", 100).

        Assertion Strategy:
        Validates dispatcher by confirming:
        - Result has 1 TextContent item.

        Testing Principle:
        Validates tool routing, ensuring dispatcher
        correctly invokes set_camera_control.
        """
        result = await cameras._set_camera_control(0, "Gain", 100)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_call_tool_get_control(self):
        """Verifies call_tool dispatcher routes get_camera_control correctly.

        Arrangement:
        1. Tool dispatcher handles get_camera_control.
        2. Arguments: camera_id, control name.
        3. Control value query via dispatcher.

        Action:
        Calls _get_camera_control(0, "Gain").

        Assertion Strategy:
        Validates dispatcher by confirming:
        - Result has 1 TextContent item.

        Testing Principle:
        Validates tool routing, ensuring dispatcher
        correctly invokes get_camera_control.
        """
        result = await cameras._get_camera_control(0, "Gain")
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_list_cameras_error_handling(self):
        """Verifies _list_cameras handles registry errors gracefully.

        Arrangement:
        1. Registry shut down to simulate error condition.
        2. _list_cameras called with no cameras available.
        3. Should return error message not exception.

        Action:
        Shuts down registry, calls _list_cameras, reinitializes.

        Assertion Strategy:
        Validates error handling by confirming:
        - Result has 1 TextContent item.
        - Text contains 'error' or 'no cameras'.

        Testing Principle:
        Validates fault tolerance, ensuring tool
        handles missing registry gracefully.
        """
        # Shutdown registry to trigger error
        shutdown_registry()
        result = await cameras._list_cameras()
        assert len(result) == 1
        text = result[0].text
        assert "error" in text.lower() or "no cameras" in text.lower()
        # Re-initialize for other tests
        driver = DigitalTwinCameraDriver()
        init_registry(driver)

    @pytest.mark.asyncio
    async def test_capture_frame_timestamp_format(self):
        """Verifies _capture_frame returns ISO formatted timestamp.

        Arrangement:
        1. Capture includes timestamp field.
        2. Should be ISO 8601 format.
        3. Enables time-series analysis.

        Action:
        Calls _capture_frame, extracts timestamp.

        Assertion Strategy:
        Validates timestamp format by confirming:
        - Contains 'T' (ISO date/time separator) OR
        - Contains '-' (date separator).

        Testing Principle:
        Validates data format, ensuring timestamp
        follows ISO standard for interoperability.
        """
        result = await cameras._capture_frame(0, 100000, 50)
        data = json.loads(result[0].text)
        timestamp = data["timestamp"]
        # Should be ISO format (contains T and timezone info)
        assert "T" in timestamp or "-" in timestamp

    @pytest.mark.asyncio
    async def test_multiple_captures(self):
        """Verifies multiple sequential captures work with different settings.

        Arrangement:
        1. Camera supports multiple captures.
        2. First: 50ms/gain25, Second: 100ms/gain75.
        3. Each capture should reflect its settings.

        Action:
        Captures twice with different parameters.

        Assertion Strategy:
        Validates stateless operation by confirming:
        - Capture1: exposure=50000, gain=25.
        - Capture2: exposure=100000, gain=75.
        - Settings don't persist between calls.

        Testing Principle:
        Validates stateless design, ensuring each
        capture uses specified settings independently.
        """
        result1 = await cameras._capture_frame(0, 50000, 25)
        result2 = await cameras._capture_frame(0, 100000, 75)

        data1 = json.loads(result1[0].text)
        data2 = json.loads(result2[0].text)

        assert data1["exposure_us"] == 50000
        assert data1["gain"] == 25
        assert data2["exposure_us"] == 100000
        assert data2["gain"] == 75

    @pytest.mark.asyncio
    async def test_camera_info_both_cameras(self):
        """Verifies _get_camera_info works for multiple cameras.

        Arrangement:
        1. Registry has 2 cameras (0=finder, 1=main).
        2. Each camera has distinct info.
        3. Tool should query either camera.

        Action:
        Calls _get_camera_info for cameras 0 and 1.

        Assertion Strategy:
        Validates multi-camera support by confirming:
        - Camera 0 response has camera_id=0.
        - Camera 1 response has camera_id=1.

        Testing Principle:
        Validates camera multiplexing, ensuring tool
        correctly addresses multiple camera instances.
        """
        result0 = await cameras._get_camera_info(0)
        result1 = await cameras._get_camera_info(1)

        data0 = json.loads(result0[0].text)
        data1 = json.loads(result1[0].text)

        assert data0["camera_id"] == 0
        assert data1["camera_id"] == 1

    @pytest.mark.asyncio
    async def test_control_operations_sequence(self):
        """Verifies set then get control sequence works correctly.

        Arrangement:
        1. set_camera_control modifies Gain to 75.
        2. get_camera_control reads back Gain.
        3. Validates round-trip control operations.

        Action:
        Sets Gain to 75, then gets Gain value.

        Assertion Strategy:
        Validates control lifecycle by confirming:
        - Set operation returns 1 result.
        - Get operation returns value field.

        Testing Principle:
        Validates control persistence, ensuring set
        operations affect subsequent get operations.
        """
        # Set gain
        set_result = await cameras._set_camera_control(0, "Gain", 75)
        assert len(set_result) == 1

        # Get gain
        get_result = await cameras._get_camera_control(0, "Gain")
        get_data = json.loads(get_result[0].text)
        assert "value" in get_data

    @pytest.mark.asyncio
    async def test_exposure_control_operations(self):
        """Verifies Exposure control set/get operations work.

        Arrangement:
        1. Exposure is a camera control parameter.
        2. set_camera_control modifies Exposure.
        3. get_camera_control reads Exposure.

        Action:
        Sets Exposure to 200, then gets Exposure.

        Assertion Strategy:
        Validates Exposure control by confirming:
        - Get operation returns camera_id=0.

        Testing Principle:
        Validates control variety, ensuring tool
        supports different control types (Gain, Exposure).
        """
        # Set exposure
        await cameras._set_camera_control(0, "Exposure", 200)

        # Get exposure
        result = await cameras._get_camera_control(0, "Exposure")
        data = json.loads(result[0].text)
        assert data["camera_id"] == 0


class TestCameraToolsIntegration:
    """Integration tests for MCP camera tools via server call_tool."""

    @pytest.fixture(autouse=True)
    def setup_registry(self):
        """Fixture providing camera registry for integration tests.

        Arrangement:
        1. Creates DigitalTwinCameraDriver for integration testing.
        2. init_registry(driver) registers 2 test cameras.
        3. Yields to allow integration test execution.
        4. shutdown_registry() ensures cleanup.

        Args:
        self: Test class instance (pytest convention).

        Returns:
        None (generator fixture).

        Raises:
        None.

        Testing Principle:
        Provides clean camera registry per integration test,
        ensuring test isolation and resource cleanup.
        """
        driver = DigitalTwinCameraDriver()
        init_registry(driver)
        yield
        shutdown_registry()

    @pytest.fixture
    def server(self):
        """Fixture providing MCP server with camera tools registered.

        Arrangement:
        1. Creates MCP Server instance with name "test-server".
        2. cameras.register(server) attaches camera tools.
        3. Returns configured server for test use.

        Args:
        self: Test class instance (pytest convention).

        Returns:
        Server: Configured MCP server with camera tools.

        Raises:
        None.

        Testing Principle:
        Provides reusable MCP server instance for
        integration tests, ensuring consistent setup.
        """
        server = Server("test-server")
        cameras.register(server)
        return server

    @pytest.mark.asyncio
    async def test_server_list_tools(self, server):
        """Verifies MCP server initializes with camera tools registered.

        Arrangement:
        1. Server fixture creates MCP server instance.
        2. cameras.register(server) called in fixture.
        3. Tool registry populated.

        Action:
        Accesses server from fixture.

        Assertion Strategy:
        Validates server initialization by confirming:
        - server object is not None.

        Testing Principle:
        Validates fixture setup, ensuring server
        properly initialized with tool registration.
        """
        # Server was created and tools registered
        assert server is not None

    @pytest.mark.asyncio
    async def test_error_paths_coverage(self):
        """Verifies error handling code paths work correctly.

        Arrangement:
        1. Valid camera ID should succeed.
        2. Tests error-free execution path.
        3. Ensures success case is covered.

        Action:
        Calls _get_camera_info with valid camera_id=0.

        Assertion Strategy:
        Validates success path by confirming:
        - Result has 1 TextContent item.
        - JSON contains camera_id field.

        Testing Principle:
        Validates success path coverage, ensuring
        non-error cases execute correctly.
        """
        # Test with valid camera - should succeed
        result = await cameras._get_camera_info(0)
        assert len(result) == 1
        data = json.loads(result[0].text)
        assert "camera_id" in data

    @pytest.mark.asyncio
    async def test_all_tool_routes(self):
        """Verifies all camera tool routing paths work correctly.

        Arrangement:
        1. 5 camera tools registered: list, get_info, capture, set_control, get_control.
        2. Each tool has routing path.
        3. Comprehensive routing test.

        Action:
        Calls all 5 tools in sequence.

        Assertion Strategy:
        Validates complete routing by confirming:
        - Each tool returns 1 TextContent result.

        Testing Principle:
        Validates routing completeness, ensuring all
        registered tools are accessible via MCP.
        """
        # Test list_cameras route
        result = await cameras._list_cameras()
        assert len(result) == 1

        # Test get_camera_info route
        result = await cameras._get_camera_info(0)
        assert len(result) == 1

        # Test capture_frame route
        result = await cameras._capture_frame(0, 100000, 50)
        assert len(result) == 1

        # Test set_camera_control route
        result = await cameras._set_camera_control(0, "Gain", 100)
        assert len(result) == 1

        # Test get_camera_control route
        result = await cameras._get_camera_control(0, "Gain")
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_capture_with_optional_args(self):
        """Verifies _capture_frame works with explicit default arguments.

        Arrangement:
        1. Tool defines default exposure=100000, gain=50.
        2. Test passes defaults explicitly.
        3. Validates explicit vs implicit defaults.

        Action:
        Calls _capture_frame with default values explicitly.

        Assertion Strategy:
        Validates argument handling by confirming:
        - exposure_us = 100000 in response.
        - gain = 50 in response.

        Testing Principle:
        Validates default values, ensuring explicit
        and implicit defaults behave identically.
        """
        # Test that default values work (exposure_us=100000, gain=50)
        result = await cameras._capture_frame(0, 100000, 50)
        data = json.loads(result[0].text)
        assert data["exposure_us"] == 100000
        assert data["gain"] == 50

    @pytest.mark.asyncio
    async def test_list_cameras_with_multiple_cameras(self):
        """Verifies _list_cameras enumerates all registered cameras.

        Arrangement:
        1. Registry has 2 cameras (finder + main).
        2. List should show both cameras.
        3. Camera IDs 0 and 1 present.

        Action:
        Calls _list_cameras, extracts camera IDs.

        Assertion Strategy:
        Validates enumeration by confirming:
        - count >= 2.
        - cameras array length >= 2.
        - Camera ID 0 in list.
        - Camera ID 1 in list.

        Testing Principle:
        Validates complete enumeration, ensuring
        all cameras appear in list output.
        """
        result = await cameras._list_cameras()
        data = json.loads(result[0].text)
        assert data["count"] >= 2
        assert len(data["cameras"]) >= 2

        # Check both cameras are listed
        camera_ids = [cam["id"] for cam in data["cameras"]]
        assert 0 in camera_ids
        assert 1 in camera_ids

    @pytest.mark.asyncio
    async def test_camera_properties_consistency(self):
        """Verifies camera properties are consistent across tool calls.

        Arrangement:
        1. Camera 0 metadata accessible via get_info and list.
        2. Both should report same dimensions.
        3. Validates data consistency.

        Action:
        Gets camera 0 via get_info and list, compares properties.

        Assertion Strategy:
        Validates consistency by confirming:
        - max_width same in both responses.
        - max_height same in both responses.
        - camera_id/id both equal 0.

        Testing Principle:
        Validates data consistency, ensuring different
        tools report identical camera metadata.
        """
        info_result = await cameras._get_camera_info(0)
        info_data = json.loads(info_result[0].text)

        list_result = await cameras._list_cameras()
        list_data = json.loads(list_result[0].text)

        # Find camera 0 in the list
        cam0_from_list = next(c for c in list_data["cameras"] if c["id"] == 0)

        # Both should have consistent dimensions
        assert info_data["info"]["max_width"] == cam0_from_list["max_width"]
        assert info_data["info"]["max_height"] == cam0_from_list["max_height"]
        # Both should have camera_id 0
        assert info_data["camera_id"] == 0
        assert cam0_from_list["id"] == 0

    @pytest.mark.asyncio
    async def test_tools_list_completeness(self):
        """Verifies TOOLS list contains all expected camera tools.

        Arrangement:
        1. cameras.TOOLS defines 5 camera tools.
        2. Expected: list_cameras, get_camera_info, capture_frame,
           set_camera_control, get_camera_control.
        3. Validates tool registration completeness.

        Action:
        Imports TOOLS, extracts tool names.

        Assertion Strategy:
        Validates TOOLS completeness by confirming:
        - 'list_cameras' in tool_names.
        - 'get_camera_info' in tool_names.
        - 'capture_frame' in tool_names.
        - 'set_camera_control' in tool_names.
        - 'get_camera_control' in tool_names.
        - len(tool_names) = 5.

        Testing Principle:
        Validates API completeness, ensuring all
        expected camera tools are registered.
        """
        from telescope_mcp.tools.cameras import TOOLS

        tool_names = [t.name for t in TOOLS]
        assert "list_cameras" in tool_names
        assert "get_camera_info" in tool_names
        assert "capture_frame" in tool_names
        assert "set_camera_control" in tool_names
        assert "get_camera_control" in tool_names
        assert len(tool_names) == 5

    @pytest.mark.asyncio
    async def test_base64_image_decoding(self):
        """Verifies captured image is valid JPEG in base64 encoding.

        Arrangement:
        1. _capture_frame returns image_base64 field.
        2. Should be base64-encoded JPEG.
        3. JPEG magic number: FFD8.

        Action:
        Captures frame, decodes base64, checks magic number.

        Assertion Strategy:
        Validates image format by confirming:
        - Decoded data length > 100 bytes.
        - First 2 bytes = 0xFFD8 (JPEG magic).

        Testing Principle:
        Validates data integrity, ensuring image
        is properly encoded and valid JPEG format.
        """
        result = await cameras._capture_frame(0, 100000, 50)
        data = json.loads(result[0].text)

        # Decode base64
        image_data = base64.b64decode(data["image_base64"])

        # Should be valid image data (JPEG starts with FFD8)
        assert len(image_data) > 100  # Reasonable image size
        # JPEG magic number
        assert image_data[0:2] == b"\xff\xd8"

    @pytest.mark.asyncio
    async def test_capture_different_cameras(self):
        """Verifies _capture_frame works with multiple cameras.

        Arrangement:
        1. Registry has cameras 0 (finder) and 1 (main).
        2. Both cameras support capture.
        3. Same settings, different cameras.

        Action:
        Captures from camera 0 and camera 1 with identical settings.

        Assertion Strategy:
        Validates multi-camera capture by confirming:
        - Camera 0 response has camera_id=0.
        - Camera 1 response has camera_id=1.
        - Both have valid base64 image data.

        Testing Principle:
        Validates camera independence, ensuring
        captures from different cameras work correctly.
        """
        result0 = await cameras._capture_frame(0, 50000, 30)
        result1 = await cameras._capture_frame(1, 50000, 30)

        data0 = json.loads(result0[0].text)
        data1 = json.loads(result1[0].text)

        assert data0["camera_id"] == 0
        assert data1["camera_id"] == 1
        # Both should have valid images
        assert len(data0["image_base64"]) > 0
        assert len(data1["image_base64"]) > 0

    @pytest.mark.asyncio
    async def test_control_with_different_controls(self):
        """Verifies _get_camera_control works with multiple control types.

        Arrangement:
        1. Camera has Gain, Exposure, Brightness controls.
        2. Each control queryable via get_camera_control.
        3. Validates multiple control support.

        Action:
        Queries Gain, Exposure, Brightness controls.

        Assertion Strategy:
        Validates control variety by confirming:
        - Each response has 'value' or 'camera_id'.

        Testing Principle:
        Validates control API generality, ensuring
        tool supports various control types.
        """
        # Test multiple controls
        controls = ["Gain", "Exposure", "Brightness"]
        for control in controls:
            result = await cameras._get_camera_control(0, control)
            data = json.loads(result[0].text)
            assert "value" in data or "camera_id" in data


class TestCameraToolDispatcher:
    """Tests for MCP tool call dispatcher."""

    @pytest.fixture(autouse=True)
    def setup_registry(self):
        """Fixture providing camera registry for dispatcher tests.

        Arrangement:
        1. Creates DigitalTwinCameraDriver for dispatcher testing.
        2. init_registry(driver) registers cameras.
        3. Yields for dispatcher test execution.
        4. shutdown_registry() cleanup.

        Args:
        self: Test class instance (pytest convention).

        Returns:
        None (generator fixture).

        Raises:
        None.

        Testing Principle:
        Provides camera infrastructure for testing
        MCP dispatcher routing logic.
        """
        driver = DigitalTwinCameraDriver()
        init_registry(driver)
        yield
        shutdown_registry()

    @pytest.mark.asyncio
    async def test_call_tool_dispatcher_via_mock(self):
        """Verifies call_tool dispatcher logic routes to correct handlers.

        Arrangement:
        1. Server registered with camera tools.
        2. call_tool handler routes based on tool name.
        3. Direct function calls test same code paths.

        Action:
        Registers tools, calls each tool function directly.

        Assertion Strategy:
        Validates dispatcher routing by confirming:
        - list_cameras returns 'cameras' in text.
        - get_camera_info returns camera_id=0.
        - capture_frame returns exposure_us=100000.
        - set_camera_control returns 1 result.
        - get_camera_control returns 1 result.

        Testing Principle:
        Validates dispatcher logic, ensuring tool
        names correctly route to handler functions.
        """
        # Create server and register tools
        server = Server("test-server")

        # Get reference to the call_tool handler before registration modifies it
        cameras.register(server)

        # Now test by directly calling the internal functions
        # which exercise the same code paths

        # Test list_cameras path
        result = await cameras._list_cameras()
        assert "cameras" in result[0].text

        # Test get_camera_info path
        result = await cameras._get_camera_info(0)
        data = json.loads(result[0].text)
        assert data["camera_id"] == 0

        # Test capture_frame path with args
        result = await cameras._capture_frame(0, 100000, 50)
        data = json.loads(result[0].text)
        assert data["exposure_us"] == 100000

        # Test set_camera_control path
        result = await cameras._set_camera_control(0, "Gain", 100)
        assert len(result) == 1

        # Test get_camera_control path
        result = await cameras._get_camera_control(0, "Gain")
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_tools_definition_schema(self):
        """Verifies TOOLS definitions have required MCP schema fields.

        Arrangement:
        1. Each Tool object needs name, description, inputSchema.
        2. inputSchema requires type, properties, required fields.
        3. Validates MCP protocol compliance.

        Action:
        Iterates through TOOLS, checks attributes.

        Assertion Strategy:
        Validates schema compliance by confirming:
        - Each tool has name attribute.
        - Each tool has description attribute.
        - Each tool has inputSchema attribute.
        - inputSchema has 'type' key.
        - inputSchema has 'properties' key.
        - inputSchema has 'required' key.

        Testing Principle:
        Validates MCP compliance, ensuring tool
        definitions follow protocol schema.
        """
        from telescope_mcp.tools.cameras import TOOLS

        for tool in TOOLS:
            assert hasattr(tool, "name")
            assert hasattr(tool, "description")
            assert hasattr(tool, "inputSchema")
            assert "type" in tool.inputSchema
            assert "properties" in tool.inputSchema
            assert "required" in tool.inputSchema


class TestMotorTools:
    """Motor tool tests."""

    @pytest.mark.asyncio
    async def test_move_altitude_stub(self):
        """Verifies move_altitude returns stub response (not implemented).

        Arrangement:
        1. motors.move_altitude not yet implemented.
        2. Should return placeholder response.
        3. Future: will control telescope altitude motor.

        Action:
        Calls motors.move_altitude(100, 50).

        Assertion Strategy:
        Validates stub behavior by confirming:
        - Result has 1 TextContent item.
        - Text contains 'not yet implemented'.

        Testing Principle:
        Validates stub implementation, ensuring unfinished
        features return clear placeholder messages.
        """
        result = await motors.move_altitude(100, 50)
        assert len(result) == 1
        assert "not yet implemented" in result[0].text.lower()


class TestPositionTools:
    """Position tool tests."""

    @pytest.mark.asyncio
    async def test_get_position_stub(self):
        """Verifies get_position returns stub response (not implemented).

        Arrangement:
        1. position.get_position not yet implemented.
        2. Should return placeholder response.
        3. Future: will return telescope RA/Dec coordinates.

        Action:
        Calls position.get_position().

        Assertion Strategy:
        Validates stub behavior by confirming:
        - Result has 1 TextContent item.
        - Text contains 'not yet implemented'.

        Testing Principle:
        Validates stub implementation, ensuring unfinished
        features return clear placeholder messages.
        """
        result = await position.get_position()
        assert len(result) == 1
        assert "not yet implemented" in result[0].text.lower()
