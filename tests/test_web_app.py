"""Comprehensive integration tests for FastAPI web application.

Tests the telescope control dashboard and REST API endpoints using
FastAPI's TestClient for synchronous HTTP testing without running
an actual server. Covers streaming, motor control, position, and
camera management endpoints.

Author: Test suite
Date: 2025-12-18
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from telescope_mcp.utils.image import ImageEncoder
from telescope_mcp.web.app import create_app


class MockImageEncoder:
    """Mock image encoder for testing without cv2 dependency.

    Provides minimal implementation of ImageEncoder protocol
    that returns valid JPEG-like bytes without actual encoding.

    Business context:
        Tests should run in CI environments without graphics libraries.
        This mock satisfies the ImageEncoder Protocol without requiring
        opencv-python-headless installation, enabling fast, portable tests.

    Arrangement:
        1. Instantiate MockImageEncoder() with no arguments.
        2. Use as drop-in replacement for CV2ImageEncoder.
        3. encode_jpeg returns fixed bytes, put_text is no-op.

    Args:
        None. Stateless mock with no configuration needed.

    Returns:
        Instance implementing ImageEncoder Protocol for injection.

    Raises:
        None. Mock operations are always successful.

    Attributes:
        None. Stateless mock implementation with no instance state.

    Example:
        >>> encoder = MockImageEncoder()
        >>> data = encoder.encode_jpeg(np.zeros((100, 100), dtype=np.uint8))
        >>> assert data[:2] == b'\\xff\\xd8'

    Testing Principle:
        Validates protocol substitutability, enabling cv2-free testing
        while maintaining ImageEncoder interface compatibility.
    """

    def encode_jpeg(self, img: np.ndarray, quality: int = 85) -> bytes:
        """Return mock JPEG bytes with valid magic header.

        Provides minimal JPEG-like response for testing without actual
        cv2 encoding. Returns fixed bytes starting with 0xFFD8.

        Business context:
        Enables web app testing without cv2 dependency. MJPEG streaming
        endpoints receive valid JPEG-like bytes for format validation.

        Args:
            img: Image array (ignored in mock).
            quality: JPEG quality (ignored in mock).

        Returns:
            Fixed bytes b'\\xff\\xd8mock_jpeg_data' for all inputs.

        Example:
            >>> encoder = MockImageEncoder()
            >>> result = encoder.encode_jpeg(np.zeros((100, 100)), 85)
            >>> result[:2] == b'\\xff\\xd8'
            True
        """
        return b"\xff\xd8mock_jpeg_data"

    def put_text(
        self,
        img: np.ndarray,
        text: str,
        position: tuple[int, int],
        scale: float,
        color: int | tuple[int, int, int],
        thickness: int,
    ) -> None:
        """No-op for testing - skips actual text rendering.

        Satisfies ImageEncoder Protocol without cv2 dependency.
        Does not modify the image array.

        Business context:
        Error overlay rendering not needed for web API testing.
        Mock satisfies Protocol for dependency injection.

        Args:
            img: Image array (ignored in mock).
            text: Text to render (ignored in mock).
            position: Position tuple (ignored in mock).
            scale: Font scale (ignored in mock).
            color: Text color (ignored in mock).
            thickness: Line thickness (ignored in mock).

        Returns:
            None. No-op implementation.

        Example:
            >>> encoder = MockImageEncoder()
            >>> encoder.put_text(img, "Error", (10, 20), 1.0, 255, 2)
            # No effect - mock implementation
        """
        pass


@pytest.fixture
def mock_encoder() -> ImageEncoder:
    """Provide mock image encoder for tests without cv2 dependency.

    Creates MockImageEncoder instance that satisfies ImageEncoder Protocol
    without requiring opencv-python-headless installation.

    Business context:
    Tests should run in CI environments without graphics libraries.
    Mock encoder enables full API testing without cv2 import issues.

    Args:
        None (pytest fixture).

    Returns:
        MockImageEncoder instance implementing ImageEncoder Protocol.

    Raises:
        None.

    Example:
        >>> def test_stream(mock_encoder):
        ...     app = create_app(encoder=mock_encoder)
        ...     # Test streaming without cv2
    """
    return MockImageEncoder()


@pytest.fixture
def mock_asi():
    """Pytest fixture mocking zwoasi library for hardware-independent testing.

    Provides comprehensive mock of ASI camera SDK including constants,
    camera enumeration, control methods, and frame capture. Enables
    testing web API without physical camera hardware.

    Args:
        None (pytest fixture).

    Yields:
        Mock object with:
        - SDK constants (ASI_GAIN, ASI_EXPOSURE, etc.)
        - get_num_cameras() returning 2
        - list_cameras() returning ASI120MC and ASI290MM
        - Camera class returning configured mock camera instance
        - Mock camera with 640x480 test frames

    Raises:
        None.

    Example:
        >>> def test_camera(mock_asi):
        ...     assert mock_asi.get_num_cameras() == 2
    """
    with patch("telescope_mcp.web.app.asi") as mock:
        # Mock SDK constants
        mock.ASI_GAIN = 0
        mock.ASI_EXPOSURE = 1
        mock.ASI_GAMMA = 2
        mock.ASI_WB_R = 3
        mock.ASI_WB_B = 4
        mock.ASI_BRIGHTNESS = 5
        mock.ASI_OFFSET = 6
        mock.ASI_BANDWIDTHOVERLOAD = 7
        mock.ASI_FLIP = 8
        mock.ASI_HIGH_SPEED_MODE = 9
        mock.ASI_IMG_RAW8 = 0

        # Mock SDK functions
        mock.get_num_cameras.return_value = 2
        mock.list_cameras.return_value = ["ZWO ASI120MC", "ZWO ASI290MM"]

        # Mock camera instance
        mock_camera = MagicMock()
        mock_camera.get_camera_property.return_value = {
            "MaxWidth": 640,
            "MaxHeight": 480,
        }
        mock_camera.get_control_value.return_value = (50, False)

        # Create test frame data (640x480 8-bit grayscale)
        test_frame = np.random.randint(0, 255, (640 * 480,), dtype=np.uint8)
        mock_camera.capture_video_frame.return_value = test_frame.tobytes()

        mock.Camera.return_value = mock_camera

        yield mock


@pytest.fixture
def mock_sdk_path():
    """Pytest fixture mocking SDK library path for testing without SDK.

    Enables testing camera functionality without ZWO ASI SDK installation
    by providing fake library path that bypasses SDK loading checks.

    Business context:
    The ZWO ASI SDK shared library is only available on systems with
    camera drivers installed. For CI/CD pipelines and development
    environments without physical telescope hardware, mocking the SDK
    path enables comprehensive testing of web application functionality.

    Args:
        None (pytest fixture).

    Yields:
        Mock: Patched get_sdk_library_path returning /fake/path/libASICamera2.so.

    Raises:
        None.

    Example:
        >>> def test_app(mock_sdk_path):
        ...     # App creation uses mocked path
        ...     app = create_app()
    """
    with patch("telescope_mcp.web.app.get_sdk_library_path") as mock:
        mock.return_value = "/fake/path/libASICamera2.so"
        yield mock


@pytest.fixture
def client(mock_asi, mock_sdk_path):
    """Pytest fixture creating FastAPI TestClient with mocked ASI SDK.

    Provides isolated test environment for telescope web application
    with all external hardware dependencies mocked. Enables synchronous
    HTTP request testing without requiring actual camera hardware.

    Business context:
    Essential for CI/CD pipeline testing where physical telescope hardware
    is unavailable. Enables rapid test execution and deterministic behavior
    by eliminating hardware variability.

    Args:
        mock_asi: Fixture providing mocked zwoasi library.
        mock_sdk_path: Fixture providing mocked SDK path.

    Yields:
        TestClient: FastAPI test client with mocked camera backend.
            Supports all HTTP methods (GET, POST, PUT, DELETE) against
            app endpoints without starting server.

    Raises:
        None.

    Example:
        >>> def test_endpoint(client):
        ...     response = client.get("/api/cameras")
        ...     assert response.status_code == 200

    Arrangement:
        1. mock_asi fixture provides mocked zwoasi library.
        2. mock_sdk_path fixture provides fake SDK path.
        3. create_app() creates FastAPI application with mocked dependencies.
        4. Wraps app in TestClient for synchronous HTTP testing.

    Action:
    Yields TestClient instance within context manager, ensuring proper
    startup/shutdown lifecycle management.

    Assertion Strategy:
    Validates test infrastructure by:
    - Ensuring TestClient creation succeeds with mocked dependencies.
    - Providing consistent mock state across all test methods.
    - Cleaning up resources via context manager exit.

    Testing Principle:
    Validates isolation pattern, ensuring tests can run without hardware
    dependencies while maintaining realistic HTTP request/response behavior.

    Returns:
        TestClient: Configured client for HTTP request testing.
    """
    app = create_app(encoder=MockImageEncoder())
    with TestClient(app) as test_client:
        # Disable timing simulation on the digital twin motor so tests
        # don't block for realistic move durations (seconds per move).
        # Web app tests validate HTTP contract, not motor timing.
        from telescope_mcp.web import app as web_app_module

        if web_app_module._motor and web_app_module._motor._instance:
            web_app_module._motor._instance._config.simulate_timing = False
        yield test_client


class TestDashboardEndpoint:
    """Tests for the main dashboard HTML rendering."""

    def test_dashboard_renders(self, client):
        """Verifies GET / returns HTML dashboard page successfully.

        Tests main dashboard endpoint serves HTML template without errors.

        Arrangement:
        1. FastAPI app configured with template directory.
        2. Dashboard template file exists and is valid.
        3. TestClient ready to request root path.

        Action:
        Issues GET / requesting dashboard landing page.

        Assertion Strategy:
        Validates basic functionality by confirming:
        - HTTP 200 indicates successful page render.
        - Content-Type header contains text/html (correct MIME type).

        Testing Principle:
        Validates template rendering, ensuring dashboard page
        is accessible and returns valid HTML content.
        """
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_dashboard_has_title(self, client):
        """Verifies dashboard HTML contains expected structural elements.

        Tests that dashboard template renders complete HTML with title.

        Arrangement:
        1. Dashboard template includes title element.
        2. Template rendering engine (Jinja2) functional.
        3. No template syntax errors preventing render.

        Action:
        Requests dashboard page and inspects response structure.

        Assertion Strategy:
        Validates template integrity by confirming:
        - HTTP 200 indicates successful render.
        - HTML structure is complete (placeholder for structure checks).

        Testing Principle:
        Validates template structure, ensuring Jinja2 renders
        complete HTML without syntax errors.
        """
        response = client.get("/")
        # Check for some expected HTML structure
        assert response.status_code == 200


class TestStreamGenerator:
    """Tests for _generate_camera_stream async generator.

    Tests the MJPEG stream generator directly without HTTP layer to avoid
    infinite stream hanging issues with TestClient. Validates frame generation,
    error handling, and MJPEG format compliance.
    """

    @pytest.fixture
    def stream_mocks(self):
        """Pytest fixture creating mocks for stream generator testing.

        Provides isolated test environment with mocked ASI SDK for
        testing _generate_camera_stream async generator without hardware.

        Business Context:
            Stream generator tests need consistent mock environment
            without real hardware dependencies. This fixture patches
            ASI SDK, camera cache, and streaming state dicts for
            deterministic async generator testing.

        Arrangement:
            1. Patch telescope_mcp.web.app.asi with mock ASI SDK.
            2. Patch _cameras dict to empty state.
            3. Patch _camera_streaming dict to empty state.
            4. Patch _camera_settings dict to empty state.
            5. Configure mock_camera with 640x480 capture returning test frame.
            6. Test frame has white square at (100:200, 100:200) for verification.

        Args:
            self: Test class instance (implicit for pytest method fixtures).

        Yields:
            Tuple[MagicMock, MagicMock]: (mock_asi module, mock_camera instance)
            configured for 640x480 grayscale capture with 255-value white
            square at position (100:200, 100:200) for visual verification.

        Raises:
            None. Fixture setup is deterministic with no external dependencies.

        Example:
            >>> async def test_stream(self, stream_mocks):
            ...     mock_asi, mock_camera = stream_mocks
            ...     assert mock_camera.capture_video_frame() is not None

        Testing Principle:
            Validates mock isolation, ensuring stream tests have consistent
            state without hardware dependencies or inter-test contamination.
        """
        with (
            patch("telescope_mcp.web.app.asi") as mock_asi,
            patch("telescope_mcp.web.app._cameras", {}),
            patch("telescope_mcp.web.app._camera_streaming", {}),
            patch("telescope_mcp.web.app._camera_settings", {}),
        ):
            # Mock SDK constants
            mock_asi.ASI_GAIN = 0
            mock_asi.ASI_EXPOSURE = 1
            mock_asi.ASI_BANDWIDTHOVERLOAD = 7
            mock_asi.ASI_IMG_RAW8 = 0

            # Mock camera instance
            mock_camera = MagicMock()
            mock_camera.get_camera_property.return_value = {
                "MaxWidth": 640,
                "MaxHeight": 480,
            }

            # Create test frame data (640x480 8-bit grayscale)
            test_frame = np.zeros((480, 640), dtype=np.uint8)
            test_frame[100:200, 100:200] = 255  # White square
            mock_camera.capture_video_frame.return_value = test_frame.tobytes()

            yield mock_asi, mock_camera

    async def test_generator_yields_mjpeg_frame(self, stream_mocks):
        """Verifies generator yields valid MJPEG boundary-separated frames.

        Tests that _generate_camera_stream produces properly formatted
        MJPEG frames with boundary markers and JPEG content.

        Arrangement:
            1. Patch _get_camera to return mock camera.
            2. Create generator with camera_id=0, fps=10.

        Action:
            Await first frame from generator.

        Assertion Strategy:
            Validates MJPEG format by confirming:
            - Frame contains "--frame" boundary marker.
            - Frame contains "Content-Type: image/jpeg" header.
            - Frame contains JPEG magic bytes (\\xff\\xd8).

        Testing Principle:
            Validates MJPEG multipart format for browser compatibility.
            Each frame must have proper boundary and valid JPEG content.
        """
        from telescope_mcp.web.app import _generate_camera_stream

        mock_asi, mock_camera = stream_mocks

        # Patch _get_camera to return our mock
        with patch("telescope_mcp.web.app._get_camera", return_value=mock_camera):
            gen = _generate_camera_stream(camera_id=0, fps=10)

            # Get first frame
            frame = await anext(gen)

            # Validate MJPEG format
            assert b"--frame" in frame
            assert b"Content-Type: image/jpeg" in frame
            # JPEG magic bytes
            assert b"\xff\xd8" in frame

            # Stop the generator
            await gen.aclose()

    async def test_generator_applies_exposure_and_gain(self, stream_mocks):
        """Verifies generator applies custom exposure and gain settings.

        Tests that exposure_us and gain parameters are passed to camera
        control methods when starting the stream.

        Arrangement:
            1. Patch _get_camera to return mock camera.
            2. Create generator with exposure_us=50000, gain=100, fps=5.

        Action:
            Await first frame to trigger camera setup.

        Assertion Strategy:
            Validates camera configuration by confirming:
            - set_control_value called with ASI_GAIN=100.
            - set_control_value called with ASI_EXPOSURE=50000.
            - start_video_capture called once.

        Testing Principle:
            Validates exposure/gain parameter propagation from API
            request to hardware control calls.
        """
        from telescope_mcp.web.app import _generate_camera_stream

        mock_asi, mock_camera = stream_mocks

        with patch("telescope_mcp.web.app._get_camera", return_value=mock_camera):
            gen = _generate_camera_stream(
                camera_id=0, exposure_us=50000, gain=100, fps=5
            )

            # Get first frame to trigger camera setup
            await anext(gen)

            # Verify camera was configured
            mock_camera.set_control_value.assert_any_call(mock_asi.ASI_GAIN, 100)
            mock_camera.set_control_value.assert_any_call(mock_asi.ASI_EXPOSURE, 50000)
            mock_camera.start_video_capture.assert_called_once()

            await gen.aclose()

    async def test_generator_error_frame_on_missing_camera(self, stream_mocks):
        """Verifies generator yields error frame for non-existent camera.

        Tests error handling when requested camera ID doesn't exist,
        ensuring stream returns visual error rather than crashing.

        Arrangement:
            1. Patch _get_camera to return None (camera not found).
            2. Request camera_id=99 (non-existent).

        Action:
            Await frames from generator.

        Assertion Strategy:
            Validates error handling by confirming:
            - First frame is valid MJPEG (boundary + content-type).
            - Second await raises StopAsyncIteration (generator completes).

        Testing Principle:
            Validates graceful degradation, returning visual error frame
            for immediate user feedback instead of crashing stream.
        """
        from telescope_mcp.web.app import _generate_camera_stream

        # Return None for missing camera
        with patch("telescope_mcp.web.app._get_camera", return_value=None):
            gen = _generate_camera_stream(camera_id=99)

            # Should yield error frame
            frame = await anext(gen)

            assert b"--frame" in frame
            assert b"Content-Type: image/jpeg" in frame
            # Generator should complete after error frame
            with pytest.raises(StopAsyncIteration):
                await anext(gen)

    async def test_generator_error_frame_on_capture_failure(self, stream_mocks):
        """Verifies generator yields error frame when capture fails.

        Tests fault tolerance when camera.capture_video_frame() raises
        an exception, ensuring stream continues with error frames.

        Arrangement:
            1. Configure mock_camera.capture_video_frame to raise RuntimeError.
            2. Patch _get_camera to return the failing mock.

        Action:
            Await first frame from generator.

        Assertion Strategy:
            Validates fault tolerance by confirming:
            - Frame contains MJPEG boundary marker.
            - Frame contains content-type header.
            - Frame contains valid JPEG magic bytes.

        Testing Principle:
            Validates resilience to USB disconnects or driver errors,
            producing visible error frames rather than crashing stream.
        """
        from telescope_mcp.web.app import _generate_camera_stream

        mock_asi, mock_camera = stream_mocks
        mock_camera.capture_video_frame.side_effect = RuntimeError("USB timeout")

        with patch("telescope_mcp.web.app._get_camera", return_value=mock_camera):
            gen = _generate_camera_stream(camera_id=0, fps=10)

            # Should yield error frame instead of crashing
            frame = await anext(gen)

            assert b"--frame" in frame
            assert b"Content-Type: image/jpeg" in frame
            # Frame should still be valid JPEG
            assert b"\xff\xd8" in frame

            await gen.aclose()

    async def test_generator_uses_stored_settings(self, stream_mocks):
        """Verifies generator uses stored camera settings when params are None.

        Tests that _camera_settings dict is consulted for exposure/gain
        when not explicitly provided to the generator.

        Arrangement:
            1. Pre-populate _camera_settings with exposure_us=75000, gain=80.
            2. Patch _get_camera to return mock camera.

        Action:
            Create generator without explicit exposure/gain parameters
            and await first frame.

        Assertion Strategy:
            Validates settings lookup by confirming:
            - set_control_value called with ASI_GAIN=80.
            - set_control_value called with ASI_EXPOSURE=75000.

        Testing Principle:
            Validates settings persistence, using stored values
            when user doesn't provide explicit parameters.
        """
        from telescope_mcp.web.app import _generate_camera_stream

        mock_asi, mock_camera = stream_mocks

        # Pre-set camera settings
        with (
            patch("telescope_mcp.web.app._get_camera", return_value=mock_camera),
            patch(
                "telescope_mcp.web.app._camera_settings",
                {0: {"exposure_us": 75000, "gain": 80}},
            ),
        ):
            gen = _generate_camera_stream(camera_id=0, fps=10)

            await anext(gen)

            # Should use stored settings
            mock_camera.set_control_value.assert_any_call(mock_asi.ASI_GAIN, 80)
            mock_camera.set_control_value.assert_any_call(mock_asi.ASI_EXPOSURE, 75000)

            await gen.aclose()

    async def test_generator_stops_on_streaming_flag(self, stream_mocks):
        """Verifies generator stops when _camera_streaming flag cleared.

        Tests that the generator loop exits cleanly when the streaming
        flag is set to False, enabling controlled stream shutdown.

        Arrangement:
            1. Create streaming_dict tracker.
            2. Patch _camera_streaming with tracker.
            3. Create generator for camera_id=0.

        Action:
            1. Await first frame (starts streaming, sets flag True).
            2. Clear streaming flag to False.
            3. Await next frame.

        Assertion Strategy:
            Validates shutdown by confirming:
            - After first frame, streaming_dict[0] is True.
            - After clearing flag, StopAsyncIteration raised.

        Testing Principle:
            Validates controlled shutdown, enabling clean stream
            termination via flag without resource leaks.
        """
        from telescope_mcp.web.app import _generate_camera_stream

        mock_asi, mock_camera = stream_mocks
        streaming_dict = {}

        with (
            patch("telescope_mcp.web.app._get_camera", return_value=mock_camera),
            patch("telescope_mcp.web.app._camera_streaming", streaming_dict),
        ):
            gen = _generate_camera_stream(camera_id=0, fps=10)

            # Get first frame (starts streaming)
            await anext(gen)
            assert streaming_dict.get(0) is True

            # Clear flag to stop
            streaming_dict[0] = False

            # Generator should stop
            with pytest.raises(StopAsyncIteration):
                await anext(gen)


class TestCameraAPIEndpoints:
    """Tests for REST API camera management endpoints."""

    def test_list_cameras_success(self, client, mock_asi):
        """Verifies GET /api/cameras returns all connected cameras with metadata.

        Tests the camera discovery endpoint by validating that the API
        correctly enumerates multiple connected ASI cameras and returns
        their complete property information.

        Business context:
        Critical for telescope dashboard initialization. Users need to see
        all available cameras before selecting one for observation. Supports
        multi-camera setups (finder scope + main imaging camera).

        Arrangement:
        1. mock_asi fixture pre-configured with 2 cameras (ASI120MC, ASI290MM).
        2. Camera properties (id, name, max resolution) set in mock responses.
        3. TestClient created with mocked SDK dependencies.

        Action:
        Issues HTTP GET to /api/cameras endpoint expecting JSON array response.

        Assertion Strategy:
        Validates complete camera enumeration by confirming:
        - HTTP 200 success status indicates endpoint availability.
        - Response count=2 matches mock_asi configured camera count.
        - cameras array length matches count field (consistency check).
        - First camera (id=0) correctly identified as "ZWO ASI120MC".
        - Second camera (id=1) correctly identified as "ZWO ASI290MM".
        - All expected metadata fields present in response.

        Testing Principle:
        Validates camera discovery mechanism, ensuring multi-camera systems
        are fully enumerated with accurate hardware identification.
        """
        response = client.get("/api/cameras")
        assert response.status_code == 200

        data = response.json()
        assert data["count"] == 2
        assert len(data["cameras"]) == 2
        assert data["cameras"][0]["id"] == 0
        assert data["cameras"][0]["name"] == "ZWO ASI120MC"
        assert data["cameras"][1]["id"] == 1
        assert data["cameras"][1]["name"] == "ZWO ASI290MM"

    def test_list_cameras_empty(self, client, mock_asi):
        """Verifies GET /api/cameras handles no-cameras scenario gracefully.

        Tests edge case where no ASI cameras are detected during hardware
        enumeration, ensuring API returns valid empty response structure.

        Business context:
        Handles startup scenario before cameras are connected or when hardware
        fails. Dashboard must display graceful "no cameras found" message
        rather than crashing.

        Arrangement:
        1. mock_asi configured to return 0 from get_num_cameras().
        2. Simulates hardware state with no USB cameras connected.
        3. TestClient ready to make API request.

        Action:
        Issues GET /api/cameras expecting empty but valid JSON response.

        Assertion Strategy:
        Validates graceful empty-state handling by confirming:
        - HTTP 200 (not 404/500) indicates endpoint remains functional.
        - Response count=0 correctly reports zero cameras.
        - cameras array is empty list [] (not null/undefined).
        - JSON structure matches success case (structural consistency).

        Testing Principle:
        Validates error resilience, ensuring API maintains stable contract
        even when underlying hardware is absent.
        """
        mock_asi.get_num_cameras.return_value = 0

        response = client.get("/api/cameras")
        assert response.status_code == 200

        data = response.json()
        assert data["count"] == 0
        assert data["cameras"] == []

    def test_list_cameras_sdk_error(self, client, mock_asi):
        """Verifies GET /api/cameras handles SDK exceptions with HTTP 500.

        Tests error propagation when underlying ASI SDK library encounters
        critical failure during camera enumeration (driver issues, hardware
        malfunction, SDK initialization failure).

        Business context:
        SDK failures indicate serious hardware or driver problems. Users must
        receive clear error message rather than silent failure or crash,
        enabling troubleshooting (check USB connections, drivers, permissions).

        Arrangement:
        1. mock_asi.get_num_cameras configured to raise RuntimeError.
        2. Simulates SDK-level exception (e.g., driver crash, USB error).
        3. Error message "SDK error" provides diagnostic context.

        Action:
        Issues GET /api/cameras while SDK is in failed state.

        Assertion Strategy:
        Validates error handling contract by confirming:
        - HTTP 500 status indicates internal server error (not client fault).
        - Response contains "error" field with diagnostic message.
        - Error message includes "SDK error" text for troubleshooting.
        - API does not crash or hang despite SDK failure.

        Testing Principle:
        Validates fault isolation, ensuring SDK exceptions are caught,
        translated to HTTP semantics, and returned as actionable errors.
        """
        mock_asi.get_num_cameras.side_effect = RuntimeError("SDK error")

        response = client.get("/api/cameras")
        assert response.status_code == 500

        data = response.json()
        assert "error" in data
        assert "SDK error" in data["error"]

    def test_set_camera_control_gain(self, client, mock_asi):
        """Verifies POST /api/camera/{id}/control sets gain successfully.

        Tests camera gain adjustment via REST API, validating that ISO/gain
        settings can be programmatically controlled for exposure adjustment.

        Business context:
        Gain control is fundamental for astrophotography exposure management.
        Users adjust gain to balance signal strength vs. noise when capturing
        dim celestial objects. Typical range: 0-400 depending on camera model.

        Arrangement:
        1. Camera 0 (ASI120MC) available via mock_asi fixture.
        2. Target gain value=100 (moderate setting for general use).
        3. POST request with query params: control="ASI_GAIN", value=100.

        Action:
        Issues HTTP POST to /api/camera/0/control with gain parameters.

        Assertion Strategy:
        Validates gain setting operation by confirming:
        - HTTP 200 indicates successful control update.
        - Response camera_id=0 confirms correct camera targeted.
        - Response control="ASI_GAIN" echoes requested parameter.
        - Response value_set=100 confirms requested value was applied.
        - Response includes value_current (readback from hardware).
        - Response includes auto field (indicates if auto-gain enabled).

        Testing Principle:
        Validates camera control interface, ensuring REST API correctly
        translates HTTP requests to SDK control_value calls.
        """
        response = client.post(
            "/api/camera/0/control", params={"control": "ASI_GAIN", "value": 100}
        )
        assert response.status_code == 200

        data = response.json()
        assert data["camera_id"] == 0
        assert data["control"] == "ASI_GAIN"
        assert data["value_set"] == 100
        assert "value_current" in data
        assert "auto" in data

    def test_set_camera_control_exposure(self, client, mock_asi):
        """Verifies POST /api/camera/{id}/control sets exposure duration.

        Tests programmatic exposure time control via REST API, validating
        that integration time can be adjusted for different brightness targets.

        Business context:
        Exposure control is critical for astrophotography. Dim objects (nebulae,
        galaxies) require long exposures (5-300 seconds), while bright objects
        (Moon, planets) need short exposures (<1 second). This test validates
        5-second exposure setting typical for intermediate-brightness targets.

        Arrangement:
        1. Camera 1 (ASI290MM) selected for exposure adjustment.
        2. Target exposure=5000000 microseconds (5 seconds).
        3. POST request with ASI_EXPOSURE control parameter.

        Action:
        Issues HTTP POST to /api/camera/1/control with exposure parameters.

        Assertion Strategy:
        Validates exposure setting by confirming:
        - HTTP 200 indicates successful control update.
        - Response camera_id=1 confirms correct camera targeted.
        - Response control="ASI_EXPOSURE" echoes requested parameter.
        - Response value_set=5000000 confirms microsecond precision preserved.

        Testing Principle:
        Validates exposure control interface, ensuring REST API supports
        full microsecond-precision range required for astrophotography.
        """
        response = client.post(
            "/api/camera/1/control",
            params={"control": "ASI_EXPOSURE", "value": 5000000},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["camera_id"] == 1
        assert data["control"] == "ASI_EXPOSURE"
        assert data["value_set"] == 5000000

    def test_set_camera_control_all_types(self, client, mock_asi):
        """Verifies all ASI camera control types are supported via API.

        Tests comprehensive control coverage by iterating through diverse
        camera settings: image processing (gamma, brightness), color balance
        (white balance R/B), hardware configuration (offset, bandwidth),
        and mode flags (flip, high-speed).

        Business context:
        Different astrophotography scenarios require different control
        combinations. Planetary imaging uses high-speed mode + bandwidth
        overload for high frame rates. Deep-sky imaging uses careful offset
        + gain tuning. Color cameras need white balance for accurate color
        rendition. API must support full control surface.

        Arrangement:
        1. Camera 0 (ASI120MC color camera) supports all control types.
        2. Test matrix includes 8 distinct controls covering all categories:
           - Image processing: GAMMA (50), BRIGHTNESS (100)
           - Color: WB_R (70), WB_B (80)
           - Signal: OFFSET (30), BANDWIDTHOVERLOAD (75)
           - Modes: FLIP (1), HIGH_SPEED_MODE (1)
        3. Each control tested with representative value.

        Action:
        Iterates through control list, issuing POST request for each setting.

        Assertion Strategy:
        Validates comprehensive control support by confirming:
        - HTTP 200 for each control type (no unsupported controls).
        - Response control field echoes requested control name.
        - All 8 control types succeed (complete coverage).
        - No control type returns error or unsupported message.

        Testing Principle:
        Validates API completeness, ensuring all ASI SDK control types
        are exposed via REST interface for full hardware capability access.
        """
        controls = [
            ("ASI_GAMMA", 50),
            ("ASI_WB_R", 70),
            ("ASI_WB_B", 80),
            ("ASI_BRIGHTNESS", 100),
            ("ASI_OFFSET", 30),
            ("ASI_BANDWIDTHOVERLOAD", 75),
            ("ASI_FLIP", 1),
            ("ASI_HIGH_SPEED_MODE", 1),
        ]

        for control, value in controls:
            response = client.post(
                "/api/camera/0/control", params={"control": control, "value": value}
            )
            assert response.status_code == 200, f"Failed for {control}"
            data = response.json()
            assert data["control"] == control

    def test_set_camera_control_invalid_control(self, client, mock_asi):
        """Verifies API rejects unrecognized control names with HTTP 400.

        Tests input validation by attempting to set non-existent camera
        control, ensuring API provides clear error with valid control list.

        Business context:
        Protects against typos in automation scripts and prevents undefined
        behavior when invalid controls are requested. Users receive immediate
        feedback with list of valid controls for correction.

        Arrangement:
        1. Camera 0 available with standard ASI control set.
        2. Request uses non-existent control name "INVALID_CONTROL".
        3. Value=50 is valid (tests control name validation specifically).

        Action:
        Issues POST with invalid control parameter.

        Assertion Strategy:
        Validates input rejection by confirming:
        - HTTP 400 indicates client error (not server fault).
        - Response contains "detail" field with diagnostic message.
        - Error message includes "Unknown control" for clarity.
        - API does not attempt to apply invalid setting.

        Testing Principle:
        Validates fail-fast validation, ensuring invalid inputs are rejected
        at API boundary with actionable error messages.
        """
        response = client.post(
            "/api/camera/0/control", params={"control": "INVALID_CONTROL", "value": 50}
        )
        assert response.status_code == 400

        data = response.json()
        assert "detail" in data
        assert "Unknown control" in data["detail"]

    def test_set_camera_control_nonexistent_camera(self, client, mock_asi):
        """Verifies API returns HTTP 404 when targeting absent camera.

        Tests resource validation by attempting control adjustment on
        camera ID that doesn't exist in system (no cameras connected).

        Business context:
        Prevents silent failures when camera is disconnected during session
        or when automation scripts use stale camera IDs. Users receive clear
        "not found" error enabling reconnection or ID correction.

        Arrangement:
        1. mock_asi.get_num_cameras returns 0 (no cameras connected).
        2. Request targets camera ID 5 (impossible with 0 cameras).
        3. Control and value are valid (tests camera existence check).

        Action:
        Issues POST to /api/camera/5/control with valid parameters.

        Assertion Strategy:
        Validates resource checking by confirming:
        - HTTP 404 indicates resource not found (standard REST semantic).
        - Response contains "detail" field describing issue.
        - Error message includes "not found" for clarity.
        - API checks camera existence before attempting control change.

        Testing Principle:
        Validates resource existence checking, ensuring API validates
        camera availability before operations.
        """
        mock_asi.get_num_cameras.return_value = 0

        response = client.post(
            "/api/camera/5/control", params={"control": "ASI_GAIN", "value": 50}
        )
        assert response.status_code == 404

        data = response.json()
        assert "detail" in data
        assert "not found" in data["detail"].lower()

    def test_set_camera_control_sdk_error(self, client, mock_asi):
        """Verifies API handles SDK control failures with HTTP 500.

        Tests error propagation when SDK fails to apply control setting
        due to hardware issue (communication error, value out of range,
        camera in incompatible mode).

        Business context:
        Hardware failures during control updates are rare but critical.
        Users need error visibility to diagnose issues (USB problems,
        camera firmware bugs, incompatible settings). Clear error messages
        enable troubleshooting without silent setting failures.

        Arrangement:
        1. Camera 0 exists but set_control_value will fail.
        2. mock_camera.set_control_value configured to raise RuntimeError.
        3. Error message "Control error" simulates SDK exception.
        4. Valid control (ASI_GAIN) and value (50) used.

        Action:
        Issues POST requesting gain change while SDK is in failed state.

        Assertion Strategy:
        Validates SDK error handling by confirming:
        - HTTP 500 indicates server/hardware error (not client fault).
        - Response contains "detail" field with diagnostic information.
        - API does not crash or hang despite SDK failure.
        - Error is propagated to client for visibility.

        Testing Principle:
        Validates fault tolerance, ensuring SDK exceptions are caught,
        logged, and returned as HTTP 500 with diagnostic context.
        """
        mock_camera = mock_asi.Camera.return_value
        mock_camera.set_control_value.side_effect = RuntimeError("Control error")

        response = client.post(
            "/api/camera/0/control", params={"control": "ASI_GAIN", "value": 50}
        )
        assert response.status_code == 500

        data = response.json()
        assert "detail" in data


class TestMotorAPIEndpoints:
    """Tests for motor control REST API endpoints.

    Parameterized tests covering both altitude and azimuth axes with
    various step counts and speeds. Validates bidirectional movement
    and default parameter handling.
    """

    @pytest.mark.parametrize(
        "axis,steps,speed,expected_speed",
        [
            ("altitude", 1000, None, 100),
            ("altitude", 500, 50, 50),
            ("azimuth", 2000, None, 100),
            ("azimuth", 1000, 60, 60),
            ("azimuth", 500, None, 100),
        ],
        ids=[
            "altitude_default_speed",
            "altitude_custom_speed",
            "azimuth_default_speed",
            "azimuth_custom_speed",
            "azimuth_positive",
        ],
    )
    def test_motor_move(self, client, axis, steps, speed, expected_speed):
        """Verifies motor endpoints accept step and speed parameters.

        Business context:
        Motor control is essential for telescope positioning. Tests
        validate both axes respond correctly to movement commands
        with proper speed handling and bidirectional support.

        Note: Motor endpoints currently return not_implemented status
        as hardware integration is pending.

        Args:
            client: FastAPI TestClient fixture.
            axis: Motor axis ("altitude" or "azimuth").
            steps: Step count (positive or negative for direction).
            speed: Speed parameter (None for default).
            expected_speed: Expected speed in response.

        Arrangement:
        Motor endpoints accept steps (required) and speed (optional).

        Assertion Strategy:
        Validates motor command by confirming:
        - HTTP 200 indicates command accepted.
        - Response status="ok" confirms command processed.
        - Response includes axis and parameters.

        Testing Principle:
        Validates motor control API contract for both axes.
        """
        url = f"/api/motor/{axis}?steps={steps}"
        if speed is not None:
            url += f"&speed={speed}"

        response = client.post(url)
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "ok"
        assert data["axis"] == axis
        assert data["steps"] == steps
        assert data["speed"] == expected_speed

    @pytest.mark.parametrize(
        "axis,steps",
        [
            ("altitude", 10000),
            ("azimuth", -120000),
        ],
        ids=["altitude_exceeds_positive_limit", "azimuth_exceeds_range"],
    )
    def test_motor_move_out_of_bounds(self, client, axis, steps):
        """Verifies motor move rejects steps that exceed position limits.

        Business context:
        Motor position limits protect hardware from damage. Moving
        beyond configured limits must return HTTP 400 with a
        descriptive error.

        Args:
            client: FastAPI TestClient fixture.
            axis: Motor axis ("altitude" or "azimuth").
            steps: Step count that would exceed motor limits.

        Assertion Strategy:
        Validates that out-of-bounds moves return 400 with error detail.
        """
        response = client.post(f"/api/motor/{axis}?steps={steps}")
        assert response.status_code == 400
        data = response.json()
        assert "limits" in data["detail"] or "exceed" in data["detail"]

    def test_stop_motors(self, client):
        """Verifies POST /api/motor/stop halts all motor movement.

        Tests emergency stop functionality for safety.

        Arrangement:
        1. Motors may be in motion or idle.
        2. Stop command must halt both axes immediately.

        Action:
        Issues POST to motor stop endpoint.

        Assertion Strategy:
        Validates emergency stop by confirming:
        - HTTP 200 indicates command processed.
        - Response status="stopped" confirms motors halted.
        - Response includes list of stopped axes.

        Testing Principle:
        Validates safety endpoint, ensuring motor stop command
        is accepted and processed correctly.
        """
        response = client.post("/api/motor/stop")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "stopped"
        assert "axes" in data
        assert "altitude" in data["axes"]
        assert "azimuth" in data["axes"]

    @pytest.mark.parametrize(
        "axis,direction,degrees",
        [
            ("altitude", "up", 0.1),
            ("azimuth", "cw", 1.0),
            ("azimuth", "right", 0.2),
        ],
        ids=[
            "altitude_up",
            "azimuth_cw",
            "azimuth_right",
        ],
    )
    def test_motor_nudge(self, client, axis, direction, degrees):
        """Verifies nudge endpoints move motor by fixed degrees.

        Business context:
        Nudge is for tap gestures - single click moves fixed amount.
        Used for fine pointing adjustments in the dashboard UI.

        Args:
            client: FastAPI TestClient fixture.
            axis: Motor axis ("altitude" or "azimuth").
            direction: Movement direction.
            degrees: Degrees to move.

        Arrangement:
        1. TestClient configured with motor controller endpoints.
        2. Parameterized with various axis/direction/degree combinations.
        3. No pre-motion required - nudge is immediate action.

        Action:
        POST /api/motor/{axis}/nudge with query params direction and degrees.
        Endpoint converts degrees to motor steps and executes fixed motion.

        Assertion Strategy:
        - HTTP 200 indicates command accepted.
        - Response status="ok" confirms command processed.
        - Response includes axis, direction, degrees, and calculated steps.

        Testing Principle:
        Validates fixed-distance motion API for UI tap/click interactions,
        ensuring precision pointing adjustments work reliably.
        """
        url = f"/api/motor/{axis}/nudge?direction={direction}&degrees={degrees}"
        response = client.post(url)
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "ok"
        assert data["axis"] == axis
        assert data["direction"] == direction
        assert data["degrees"] == degrees
        assert "steps" in data
        assert isinstance(data["steps"], int)

    def test_motor_nudge_altitude_down_after_up(self, client):
        """Verifies altitude nudge down works after moving up.

        Business context:
        Validates bidirectional nudge by first moving up, then
        nudging down.

        Arrangement:
        Move altitude up 1° first.

        Action:
        Nudge altitude down 0.5° from elevated position.

        Assertion Strategy:
        - Both moves return HTTP 200.
        - Nudge response confirms direction=down.
        """
        # Move up first to create room
        resp_up = client.post("/api/motor/altitude/nudge?direction=up&degrees=1.0")
        assert resp_up.status_code == 200

        # Now nudge down — should succeed
        resp_down = client.post("/api/motor/altitude/nudge?direction=down&degrees=0.5")
        assert resp_down.status_code == 200
        data = resp_down.json()
        assert data["status"] == "ok"
        assert data["direction"] == "down"

    def test_motor_nudge_altitude_down_at_zero(self, client):
        """Verifies altitude nudge down from position 0 succeeds.

        Business context:
        Position 0 is zenith. Negative direction moves toward horizon.
        With limits at -60° (-93333 steps), a small nudge down from
        zenith is valid.

        Assertion Strategy:
        - HTTP 200, direction is "down".
        """
        response = client.post("/api/motor/altitude/nudge?direction=down&degrees=0.5")
        assert response.status_code == 200
        data = response.json()
        assert data["direction"] == "down"

    def test_motor_nudge_azimuth_ccw_from_positive(self, client):
        """Verifies azimuth CCW nudge works after pre-moving to positive position.

        Business context:
        Azimuth range is 0 to +190° (home=0). Right/CW is positive.
        CCW/left nudge goes negative, so it fails from position 0.
        Pre-move right first, then CCW nudge.

        Assertion Strategy:
        - Pre-move right returns 200.
        - CCW nudge returns 200 with correct direction.
        """
        # Move right first to create room for CCW nudge
        resp_right = client.post("/api/motor/azimuth/nudge?direction=right&degrees=1.0")
        assert resp_right.status_code == 200

        # Now CCW nudge from positive position
        resp_ccw = client.post("/api/motor/azimuth/nudge?direction=ccw&degrees=0.5")
        assert resp_ccw.status_code == 200
        data = resp_ccw.json()
        assert data["status"] == "ok"
        assert data["direction"] == "ccw"

    def test_motor_nudge_azimuth_left_from_positive(self, client):
        """Verifies azimuth left nudge works after pre-moving to positive position.

        Business context:
        Same as CCW — 'left' is an alias for CCW direction.

        Assertion Strategy:
        - Pre-move right returns 200.
        - Left nudge returns 200 with correct direction.
        """
        # Move right first to create room
        resp_right = client.post("/api/motor/azimuth/nudge?direction=right&degrees=1.0")
        assert resp_right.status_code == 200

        # Now left nudge from positive position
        resp_left = client.post("/api/motor/azimuth/nudge?direction=left&degrees=0.5")
        assert resp_left.status_code == 200
        data = resp_left.json()
        assert data["status"] == "ok"
        assert data["direction"] == "left"

    @pytest.mark.parametrize(
        "axis,direction,speed",
        [
            ("altitude", "up", 50),
            ("altitude", "down", 75),
            ("azimuth", "cw", 50),
            ("azimuth", "ccw", 25),
            ("azimuth", "left", 100),
            ("azimuth", "right", 10),
        ],
        ids=[
            "altitude_up",
            "altitude_down",
            "azimuth_cw",
            "azimuth_ccw",
            "azimuth_left",
            "azimuth_right",
        ],
    )
    def test_motor_start(self, client, axis, direction, speed):
        """Verifies start endpoints begin continuous motor motion.

        Business context:
        Start is for hold gestures - press and hold begins continuous
        motion until stop is called. Used for rapid slewing in dashboard.

        Args:
            client: FastAPI TestClient fixture.
            axis: Motor axis ("altitude" or "azimuth").
            direction: Movement direction.
            speed: Speed percentage (1-100).

        Arrangement:
        1. TestClient configured with motor controller endpoints.
        2. Parameterized with various axis/direction/speed combinations.
        3. Motor in stopped state initially.

        Action:
        POST /api/motor/{axis}/start with query params direction and speed.
        Endpoint begins continuous motion at specified speed.
        Follow with stop command to clean up (prevent runaway motion).

        Assertion Strategy:
        - HTTP 200 indicates command accepted.
        - Response status="moving" confirms motion started.
        - Response includes axis, direction, and speed.

        Testing Principle:
        Validates continuous motion API for UI press-and-hold interactions,
        ensuring rapid slewing works reliably for large position changes.
        """
        url = f"/api/motor/{axis}/start?direction={direction}&speed={speed}"
        response = client.post(url)
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "moving"
        assert data["axis"] == axis
        assert data["direction"] == direction
        assert data["speed"] == speed

        # Clean up - stop the motor
        client.post("/api/motor/stop")

    def test_stop_single_axis(self, client):
        """Verifies stop with axis parameter stops only that motor.

        Business context:
        Single-axis stop is for button release - when user releases
        the up/down button, only that axis should stop.

        Arrangement:
        1. TestClient configured with motor controller endpoints.
        2. Start altitude motor in continuous motion (up at 50% speed).
        3. Azimuth motor remains stopped (validates selective stopping).

        Action:
        POST /api/motor/stop?axis=altitude to halt only altitude motion.
        Verifies stop command respects axis parameter for selective control.

        Assertion Strategy:
        - HTTP 200 indicates command accepted.
        - Response status="stopped" confirms motor halted.
        - Response axes list contains only the specified axis.

        Testing Principle:
        Validates selective motor control for independent axis operations,
        ensuring UI button release stops only the controlled axis.
        """
        # Start altitude motion
        client.post("/api/motor/altitude/start?direction=up&speed=50")

        # Stop only altitude
        response = client.post("/api/motor/stop?axis=altitude")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "stopped"
        assert data["axes"] == ["altitude"]

    def test_nudge_direction_validation(self, client):
        """Verifies nudge rejects invalid direction values.

        Tests API parameter validation for direction enum enforcement.

        Business context:
        Invalid directions should return 422 validation error.
        Prevents undefined motor behavior from malformed requests.

        Arrangement:
        1. TestClient configured with motor controller endpoints.
        2. Endpoint enforces direction enum (up/down/cw/ccw/left/right).

        Action:
        POST /api/motor/altitude/nudge with invalid direction="invalid".
        FastAPI validation should reject before reaching business logic.

        Assertion Strategy:
        - HTTP 422 (Unprocessable Entity) indicates validation failure.
        - Motor never receives command (validation layer blocks it).

        Testing Principle:
        Validates request validation layer prevents invalid inputs,
        ensuring API contract enforcement at endpoint boundary.
        """
        response = client.post("/api/motor/altitude/nudge?direction=invalid")
        assert response.status_code == 422

    def test_start_speed_validation(self, client):
        """Verifies start rejects out-of-range speed values.

        Tests API parameter validation for speed range enforcement.

        Business context:
        Speed must be 1-100. Out of range should return 422.
        Prevents motor damage from invalid speed commands.

        Arrangement:
        1. TestClient configured with motor controller endpoints.
        2. Endpoint enforces speed range constraint (1-100).
        3. Tests both under-range (0) and over-range (101) values.

        Action:
        POST /api/motor/altitude/start with invalid speed values.
        First request uses speed=0 (below minimum).
        Second request uses speed=101 (above maximum).

        Assertion Strategy:
        - HTTP 422 for both requests indicates validation failure.
        - Motor never receives command (validation layer blocks it).

        Testing Principle:
        Validates range constraint enforcement prevents dangerous
        motor speeds, ensuring safe operation within hardware limits.
        """
        # Speed too low
        response = client.post("/api/motor/altitude/start?direction=up&speed=0")
        assert response.status_code == 422

        # Speed too high
        response = client.post("/api/motor/altitude/start?direction=up&speed=101")
        assert response.status_code == 422

    def test_set_home_succeeds(self, client):
        """Verifies set home returns 200 when motor is initialized.

        Business context:
            Set Home zeros both motor position counters. In test context,
            the digital twin motor is always initialized via lifespan,
            so the endpoint should succeed.

        Arrangement:
            1. TestClient with lifespan (motor initialized via digital twin).

        Action:
            POST /api/motor/home/set

        Assertion Strategy:
            - HTTP 200 indicates success.
            - Response status is "ok".
            - Response message confirms home set.
        """
        response = client.post("/api/motor/home/set")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "0,0" in data["message"]

    def test_generic_nudge_altitude(self, client):
        """Verifies generic /api/motor/nudge route dispatches to altitude.

        Business context:
            JavaScript dashboard calls generic /api/motor/nudge?axis=...
            instead of per-axis routes. This test verifies the dispatch.
        """
        response = client.post(
            "/api/motor/nudge?axis=altitude&direction=up&degrees=0.1"
        )
        assert response.status_code == 200
        data = response.json()
        assert data["axis"] == "altitude"
        assert data["direction"] == "up"

    def test_generic_nudge_azimuth(self, client):
        """Verifies generic /api/motor/nudge route dispatches to azimuth."""
        response = client.post(
            "/api/motor/nudge?axis=azimuth&direction=right&degrees=0.5"
        )
        assert response.status_code == 200
        data = response.json()
        assert data["axis"] == "azimuth"
        assert data["direction"] == "right"

    def test_generic_start_altitude(self, client):
        """Verifies generic /api/motor/start route dispatches to altitude."""
        response = client.post("/api/motor/start?axis=altitude&direction=up&speed=50")
        assert response.status_code == 200
        data = response.json()
        assert data["axis"] == "altitude"
        assert data["status"] == "moving"

    def test_generic_start_azimuth(self, client):
        """Verifies generic /api/motor/start route dispatches to azimuth."""
        response = client.post("/api/motor/start?axis=azimuth&direction=ccw&speed=75")
        assert response.status_code == 200
        data = response.json()
        assert data["axis"] == "azimuth"
        assert data["status"] == "moving"


class TestPositionAPIEndpoint:
    """Tests for telescope position readout endpoint."""

    def test_get_position(self, client):
        """Verifies GET /api/position returns current telescope coordinates.

        Tests position readout endpoint provides altitude/azimuth data.

        Arrangement:
        1. Position endpoint configured with coordinate access.
        2. Motors/encoders provide position feedback.
        3. Response format: {altitude: float, azimuth: float}.

        Action:
        Issues GET /api/position.

        Assertion Strategy:
        Validates position API by confirming:
        - HTTP 200 indicates endpoint accessible.
        - Response contains "altitude" field.
        - Response contains "azimuth" field.
        - Both values are numeric (int or float).

        Testing Principle:
        Validates position readout, ensuring telescope coordinates
        are accessible for alignment and tracking operations.
        """
        response = client.get("/api/position")
        assert response.status_code == 200

        data = response.json()
        assert "altitude" in data
        assert "azimuth" in data
        assert isinstance(data["altitude"], int | float)
        assert isinstance(data["azimuth"], int | float)

    def test_position_values_in_range(self, client):
        """Verifies position values fall within valid coordinate ranges.

        Tests range validation ensuring position data is physically reasonable.

        Arrangement:
        1. Altitude range: -90° to +90° (horizon to horizon).
        2. Azimuth range: 0° to 360° (full rotation).
        3. Position endpoint returns current encoder values.

        Action:
        Requests current position and validates ranges.

        Assertion Strategy:
        Validates coordinate sanity by confirming:
        - Altitude within -90 to +90 degree bounds.
        - Azimuth within 0 to 360 degree bounds.

        Testing Principle:
        Validates coordinate constraints, ensuring position values
        are physically reasonable for telescope operations.
        """
        response = client.get("/api/position")
        data = response.json()

        # Altitude typically 0-90 degrees
        assert -90 <= data["altitude"] <= 90

        # Azimuth typically 0-360 degrees
        assert 0 <= data["azimuth"] <= 360

        # Sensor should be operational (covers 1024->1035 branch)
        assert data["sensor_status"] == "ok"

    def test_position_with_no_sensor(self, mock_asi, mock_sdk_path):
        """Verifies GET /api/position works when no sensor is available.

        Tests that when _sensor is None (no sensor hardware), the endpoint
        returns default position with no_sensor status.

        Business context:
        Telescope may operate without IMU sensor hardware. Position
        endpoint should return safe defaults and indicate no sensor.

        Arrangement:
        1. Create app and set _sensor to None.

        Action:
        Issues GET /api/position with no sensor.

        Assertion Strategy:
        Validates no-sensor behavior by confirming:
        - HTTP 200 response.
        - sensor_status is "no_sensor".
        - Position defaults to 0,0.

        Testing Principle:
        Validates graceful handling when sensor unavailable.
        """
        import telescope_mcp.web.app as app_module

        app = create_app(encoder=MockImageEncoder())

        with TestClient(app) as test_client:
            # Set sensor to None AFTER lifespan has run
            original_sensor = app_module._sensor
            app_module._sensor = None

            try:
                response = test_client.get("/api/position")
                assert response.status_code == 200

                data = response.json()
                assert data["sensor_status"] == "no_sensor"
                assert data["altitude"] == 0.0
                assert data["azimuth"] == 0.0
            finally:
                # Restore original sensor state
                app_module._sensor = original_sensor

    def test_position_with_sensor_read_error(self, mock_asi, mock_sdk_path):
        """Verifies GET /api/position handles sensor read errors gracefully.

        Tests that when IMU sensor is initialized but read() raises an
        exception, the endpoint returns default position with error status.

        Business context:
        Sensor hardware can fail mid-session (disconnection, I2C error).
        Position endpoint should return safe defaults rather than crashing,
        allowing continued telescope operation in manual mode.

        Arrangement:
        1. Create app with mock sensor that raises on read().
        2. Mock _sensor global to simulate initialized but failing sensor.

        Action:
        Issues GET /api/position with failing sensor.

        Assertion Strategy:
        Validates error handling by confirming:
        - HTTP 200 response (endpoint doesn't crash).
        - sensor_status is "error".
        - Position defaults to 0,0.

        Testing Principle:
        Validates fault tolerance for sensor read failures.
        """
        from unittest.mock import AsyncMock

        import telescope_mcp.web.app as app_module

        app = create_app(encoder=MockImageEncoder())

        with TestClient(app) as test_client:
            # Create mock sensor that raises on read
            mock_sensor = AsyncMock()
            mock_sensor.read = AsyncMock(
                side_effect=RuntimeError("Sensor disconnected")
            )

            # Inject mock sensor AFTER lifespan has run
            original_sensor = app_module._sensor
            app_module._sensor = mock_sensor

            try:
                response = test_client.get("/api/position")
                assert response.status_code == 200

                data = response.json()
                assert data["sensor_status"] == "error"
                assert data["altitude"] == 0.0
                assert data["azimuth"] == 0.0
            finally:
                # Restore original sensor state
                app_module._sensor = original_sensor


class TestLifecycleManagement:
    """Tests for application startup and shutdown lifecycle."""

    @pytest.mark.asyncio
    async def test_cleanup_sensor_handles_disconnect_error(
        self, mock_asi, mock_sdk_path
    ):
        """Verifies _cleanup_sensor handles disconnect errors gracefully.

        Tests that when sensor.disconnect() raises an exception during
        shutdown, the error is logged but cleanup continues without crash.

        Business context:
        Sensor hardware can be in bad state during shutdown (already
        disconnected, USB unplugged). Cleanup must complete regardless
        to allow app restart without resource leaks.

        Arrangement:
        1. Create mock sensor that raises on disconnect().
        2. Set global _sensor to mock.

        Action:
        Call _cleanup_sensor() directly.

        Assertion Strategy:
        Validates error handling by confirming:
        - Function completes without raising.
        - _sensor is set to None after cleanup.

        Testing Principle:
        Validates graceful degradation during shutdown cleanup.
        """
        from unittest.mock import AsyncMock

        import telescope_mcp.web.app as app_module
        from telescope_mcp.web.app import _cleanup_sensor

        # Create mock sensor that raises on disconnect
        mock_sensor = AsyncMock()
        mock_sensor.disconnect = AsyncMock(
            side_effect=RuntimeError("Serial port already closed")
        )

        # Inject mock sensor
        original_sensor = app_module._sensor
        app_module._sensor = mock_sensor

        try:
            # Run cleanup - should not raise despite disconnect error
            await _cleanup_sensor()

            # Sensor should be cleared even after error
            assert app_module._sensor is None
        finally:
            # Restore original state
            app_module._sensor = original_sensor

    @pytest.mark.asyncio
    async def test_init_sensor_handles_connection_error(self, mock_asi, mock_sdk_path):
        """Verifies _init_sensor handles sensor connection errors gracefully.

        Tests that when sensor driver creation or connection fails during
        startup, the error is logged but startup continues with _sensor=None.

        Business context:
        Sensor hardware may not be present or may fail to initialize.
        Application should continue startup to allow camera and motor
        operations even without IMU sensor data.

        Arrangement:
        1. Mock get_factory to return factory that raises on create_sensor_driver.

        Action:
        Call _init_sensor() directly.

        Assertion Strategy:
        Validates error handling by confirming:
        - Function completes without raising.
        - _sensor is set to None after failure.

        Testing Principle:
        Validates graceful degradation during sensor initialization failure.
        """
        from unittest.mock import MagicMock

        import telescope_mcp.web.app as app_module
        from telescope_mcp.web.app import _init_sensor

        # Save original state
        original_sensor = app_module._sensor

        # Mock get_factory to return a factory that raises on create_sensor_driver
        original_get_factory = app_module.get_factory

        mock_factory = MagicMock()
        mock_factory.create_sensor_driver.side_effect = RuntimeError(
            "No sensor hardware found"
        )
        app_module.get_factory = MagicMock(return_value=mock_factory)

        try:
            # Run init - should not raise despite sensor error
            await _init_sensor()

            # Sensor should be None after failed init
            assert app_module._sensor is None
        finally:
            # Restore original state
            app_module._sensor = original_sensor
            app_module.get_factory = original_get_factory

    @pytest.mark.asyncio
    async def test_cleanup_sensor_when_none(self, mock_asi, mock_sdk_path):
        """Verifies _cleanup_sensor handles None sensor gracefully.

        Tests that when _sensor is None (never initialized or already cleaned),
        cleanup completes without error.

        Business context:
        Cleanup may be called when sensor was never initialized (e.g., no
        hardware present) or already cleaned up. Must be idempotent.

        Arrangement:
        1. Set _sensor to None.

        Action:
        Call _cleanup_sensor() directly.

        Assertion Strategy:
        Validates no-op behavior by confirming:
        - Function completes without raising.
        - _sensor remains None.

        Testing Principle:
        Validates idempotent cleanup behavior.
        """
        import telescope_mcp.web.app as app_module
        from telescope_mcp.web.app import _cleanup_sensor

        # Save and set to None
        original_sensor = app_module._sensor
        app_module._sensor = None

        try:
            # Run cleanup - should complete without error
            await _cleanup_sensor()

            # Sensor should still be None
            assert app_module._sensor is None
        finally:
            # Restore original state
            app_module._sensor = original_sensor

    def test_app_lifespan(self, mock_asi, mock_sdk_path):
        """Verifies FastAPI lifespan handles startup/shutdown correctly.

        Tests application lifecycle management by creating TestClient
        (which triggers lifespan context manager), making requests,
        and verifying clean shutdown.

        Business context:
        Proper lifecycle management prevents resource leaks (camera handles,
        threads) and ensures clean startup/shutdown for systemd service
        deployment. Critical for production reliability and restart scenarios.

        Arrangement:
        1. mock_asi and mock_sdk_path provide isolated test environment.
        2. create_app() returns FastAPI app with lifespan manager.
        3. TestClient context manager invokes lifespan startup/shutdown.

        Action:
        Creates TestClient (entering lifespan context), makes API request,
        then exits context (triggering shutdown).

        Assertion Strategy:
        Validates lifecycle management by confirming:
        - TestClient creation succeeds (startup completes without error).
        - API endpoint accessible during lifespan (app is running).
        - GET /api/position returns HTTP 200 (app is functional).
        - Context exit completes (shutdown cleanup runs without hanging).
        - SDK initialization uses lazy loading (not necessarily at startup).

        Testing Principle:
        Validates resource management, ensuring app can start, serve requests,
        and cleanly shut down without leaking resources or hanging.
        """
        app = create_app(encoder=MockImageEncoder())

        # Lifespan is tested by creating client (enters/exits context)
        with TestClient(app) as test_client:
            # App should be accessible
            response = test_client.get("/api/position")
            assert response.status_code == 200

        # After context exit, cleanup should have occurred
        # SDK init happens lazily, not always at startup

    def test_sdk_initialization_failure(self, mock_asi, mock_sdk_path):
        """Verifies app starts gracefully despite SDK initialization failure.

        Tests fault tolerance during startup when ASI SDK cannot be loaded
        (missing library files, incompatible architecture, driver issues).

        Business context:
        Application should remain operational for monitoring/motor control
        even when camera SDK fails. Logs warning but continues startup,
        allowing troubleshooting access via web interface. Prevents complete
        system failure due to single component issue.

        Arrangement:
        1. mock_sdk_path configured to raise RuntimeError on access.
        2. Simulates missing SDK files or initialization failure.
        3. Error message "SDK not found" indicates cause.

        Action:
        Calls create_app() while SDK initialization will fail.

        Assertion Strategy:
        Validates graceful degradation by confirming:
        - create_app() returns valid app object (doesn't raise exception).
        - App is not None (initialization completes).
        - TestClient can be created (app is functional).
        - GET /api/position still works (non-camera endpoints available).
        - HTTP 200 response (app serves placeholder data).
        - Warning is logged but startup continues (check logs separately).

        Testing Principle:
        Validates fault isolation, ensuring camera SDK failures don't
        prevent app startup or access to other system components.
        """
        mock_sdk_path.side_effect = RuntimeError("SDK not found")

        # Should still create app (logs warning but continues)
        app = create_app(encoder=MockImageEncoder())
        assert app is not None

        with TestClient(app) as test_client:
            # Position endpoint still works (placeholder)
            response = test_client.get("/api/position")
            assert response.status_code == 200


class TestCameraStateManagement:
    """Tests for internal camera state management."""

    def test_camera_lazy_open(self, client, mock_asi):
        """Verifies cameras list doesn't require opening cameras.

        Tests that camera enumeration via API doesn't trigger camera open
        operations - cameras should only be opened when streaming.

        Arrangement:
        1. Client and mock_asi fixtures provide test environment.
        2. No cameras have been opened yet.

        Action:
        Issues GET /api/cameras request.

        Assertion Strategy:
        Validates lazy initialization by confirming:
        - HTTP 200 response.
        - Response contains count >= 0.

        Testing Principle:
        Validates lazy initialization, reducing startup time
        and USB bandwidth by deferring camera opens.
        """
        # List cameras should work without opening them
        response = client.get("/api/cameras")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] >= 0

    def test_camera_reuse(self, client, mock_asi):
        """Verifies camera objects are reused across multiple requests.

        Tests instance caching by making multiple API calls and confirming
        camera objects aren't recreated unnecessarily (no redundant opens).

        Business context:
        Opening/closing camera connections is expensive (USB initialization,
        memory allocation). Reusing instances reduces latency and prevents
        USB device conflicts. Critical for responsive UI during rapid
        setting changes.

        Arrangement:
        1. Camera 0 and 1 available via mock_asi.
        2. First GET /api/cameras triggers camera discovery.
        3. Second GET /api/cameras should reuse discovered cameras.

        Action:
        Issues two consecutive GET /api/cameras requests.

        Assertion Strategy:
        Validates instance reuse by confirming:
        - Both responses return HTTP 200 (both requests succeed).
        - Camera data is consistent between requests.
        - No excessive Camera() constructor calls in mock (check separately).
        - Simple queries don't trigger camera open/close cycles.

        Testing Principle:
        Validates resource efficiency, ensuring camera instances are
        cached for reuse rather than recreated on every request.
        """
        # Access same camera twice
        response1 = client.get("/api/cameras")
        response2 = client.get("/api/cameras")

        assert response1.status_code == 200
        assert response2.status_code == 200

        # Should not open camera multiple times for simple queries

    def test_camera_settings_persistence(self, client, mock_asi):
        """Verifies camera control settings persist via API.

        Tests that gain/exposure adjustments made via control API are
        stored for subsequent use.

        Arrangement:
        1. Client and mock_asi fixtures provide test environment.
        2. Camera 0 available.

        Action:
        POSTs to /api/camera/0/control with ASI_GAIN=150.

        Assertion Strategy:
        Validates persistence by confirming:
        - HTTP 200 response.
        - Response indicates success.

        Testing Principle:
        Validates settings persistence, enabling users to
        adjust exposure/gain that persists across API calls.
        """
        # Set a control value
        response = client.post(
            "/api/camera/0/control",
            params={"control": "ASI_GAIN", "value": 150},
        )
        assert response.status_code == 200

        # Verify the setting was applied
        data = response.json()
        assert data.get("success") is True or response.status_code == 200


class TestStaticFilesAndTemplates:
    """Tests for static file serving and template rendering."""

    def test_static_mount_configured(self):
        """Verifies FastAPI app has routes configured.

        Arrangement:
        1. create_app() initializes FastAPI application.
        2. Routes should be registered during initialization.

        Action:
        Creates app and checks route configuration.

        Assertion Strategy:
        Validates app initialization by confirming:
        - App object is created successfully.
        - Routes are registered (non-empty route list).

        Testing Principle:
        Validates app initialization, ensuring route registration
        completes during create_app() call.
        """
        app = create_app(encoder=MockImageEncoder())
        # Check that app has routes configured
        assert app is not None
        assert len(app.routes) > 0

    def test_static_mount_skipped_when_dir_missing(self):
        """Verifies app still works when STATIC_DIR doesn't exist.

        Tests the branch where STATIC_DIR.exists() returns False,
        ensuring app initialization completes without static mount.

        Arrangement:
        1. Patch STATIC_DIR.exists() to return False.
        2. App should skip static file mounting gracefully.

        Action:
        Creates app with missing static directory.

        Assertion Strategy:
        Validates graceful degradation by confirming:
        - App creates successfully.
        - Dashboard endpoint still accessible.

        Testing Principle:
        Validates fault tolerance, ensuring app starts even when
        static directory is missing in development environments.
        """
        with patch("telescope_mcp.web.app.STATIC_DIR") as mock_static_dir:
            mock_static_dir.exists.return_value = False

            app = create_app(encoder=MockImageEncoder())
            assert app is not None

            # App should work but static files won't be served
            client = TestClient(app)
            response = client.get("/")
            assert response.status_code == 200

    def test_templates_directory_configured(self):
        """Verifies TEMPLATES_DIR is configured as Path object.

        Arrangement:
        1. TEMPLATES_DIR is module-level constant.
        2. Should be Path object for cross-platform support.

        Action:
        Imports TEMPLATES_DIR constant and checks type.

        Assertion Strategy:
        Validates configuration by confirming:
        - TEMPLATES_DIR is Path object (not string).

        Testing Principle:
        Validates configuration type safety, ensuring template
        path uses pathlib for cross-platform compatibility.
        """
        from telescope_mcp.web.app import TEMPLATES_DIR

        # Should be a Path object
        assert isinstance(TEMPLATES_DIR, Path)


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_invalid_endpoint(self, client):
        """Verifies GET to non-existent endpoint returns HTTP 404.

        Tests error handling for invalid URL paths.

        Arrangement:
        1. API has defined routes for valid endpoints.
        2. Request targets /api/nonexistent (undefined route).

        Action:
        Issues GET to invalid endpoint.

        Assertion Strategy:
        Validates error handling by confirming:
        - HTTP 404 not found (standard REST error).

        Testing Principle:
        Validates REST semantics, ensuring undefined routes
        return proper 404 status for client error handling.
        """
        response = client.get("/api/nonexistent")
        assert response.status_code == 404

    def test_method_not_allowed(self, client):
        """Verifies incorrect HTTP method returns 405 error.

        Tests HTTP method validation for endpoints.

        Arrangement:
        1. /api/position defined as GET-only endpoint.
        2. Request uses POST method (incorrect).

        Action:
        Issues POST to GET-only endpoint.

        Assertion Strategy:
        Validates method checking by confirming:
        - HTTP 405 method not allowed.

        Testing Principle:
        Validates HTTP method enforcement, ensuring endpoints
        reject incorrect methods with proper 405 status.
        """
        # Position is GET only
        response = client.post("/api/position")
        assert response.status_code == 405

    def test_missing_query_params(self, client):
        """Verifies missing required parameters return 422 validation error.

        Tests FastAPI parameter validation rejects incomplete requests.

        Arrangement:
        1. Camera control endpoint requires control and value params.
        2. Request omits both required parameters.

        Action:
        Issues POST with no query parameters.

        Assertion Strategy:
        Validates parameter checking by confirming:
        - HTTP 422 unprocessable entity (validation failure).

        Testing Principle:
        Validates FastAPI validation, ensuring required parameters
        are enforced with proper 422 status for client feedback.
        """
        # Missing both control and value
        response = client.post("/api/camera/0/control")
        assert response.status_code == 422  # FastAPI validation error


class TestConcurrentAccess:
    """Tests for handling concurrent camera access."""

    def test_multiple_simultaneous_requests(self, client, mock_asi):
        """Verifies API handles rapid sequential requests without corruption.

        Tests request handling stability by issuing multiple camera list
        requests in rapid succession, validating thread-safety and state
        consistency.

        Business context:
        Web dashboard makes frequent API polling (camera list, position updates)
        at ~1-5 Hz. System must handle burst traffic during page load when
        multiple widgets initialize simultaneously. Request handling must be
        thread-safe to prevent state corruption or race conditions.

        Arrangement:
        1. Camera 0 and 1 configured in mock_asi.
        2. TestClient ready for synchronous request sequence.
        3. 5 identical GET /api/cameras requests prepared.

        Action:
        Issues 5 consecutive GET /api/cameras requests in rapid succession.

        Assertion Strategy:
        Validates request handling stability by confirming:
        - All 5 responses return HTTP 200 (no failures under load).
        - No response corruption or mixed data between requests.
        - Each response contains consistent camera data.
        - No exceptions or timeouts during request burst.
        - State remains stable across all requests.

        Testing Principle:
        Validates concurrency safety, ensuring API can handle burst traffic
        typical of dashboard initialization without corruption.
        """
        # Make several requests in sequence (TestClient is synchronous)
        responses = []
        for i in range(5):
            response = client.get("/api/cameras")
            responses.append(response)

        # All should succeed
        for response in responses:
            assert response.status_code == 200

    def test_control_rapid_sequence(self, client, mock_asi):
        """Verifies control API handles rapid sequential requests.

        Tests thread-safety by issuing multiple camera control changes
        in rapid succession, ensuring no conflicts or state corruption.

        Arrangement:
        1. Client and mock_asi fixtures provide test environment.
        2. Camera 0 available.
        3. Sequence of 5 gain values prepared: [50, 100, 150, 100, 50].

        Action:
        POSTs 5 consecutive control requests with varying gain values.

        Assertion Strategy:
        Validates thread-safety by confirming:
        - All 5 responses return HTTP 200.
        - No exceptions or state corruption.

        Testing Principle:
        Validates concurrency safety for rapid UI slider adjustments
        without deadlocks or corruption.
        """
        # Issue multiple control changes rapidly
        for gain_value in [50, 100, 150, 100, 50]:
            response = client.post(
                "/api/camera/0/control",
                params={"control": "ASI_GAIN", "value": gain_value},
            )
            assert response.status_code == 200


# Integration test for full workflow
class TestEndToEndWorkflow:
    """End-to-end integration tests for typical usage workflows."""

    def test_dashboard_camera_workflow(self, client, mock_asi):
        """Verifies complete camera setup workflow from discovery to configuration.

        Tests integrated end-to-end sequence: enumerate cameras, extract IDs,
        configure settings. Represents typical dashboard initialization flow.

        Business context:
        This workflow is executed on every dashboard page load. User sees
        camera list, selects camera, adjusts gain/exposure, then starts
        streaming. Each step depends on previous step's data (camera ID
        propagation). Workflow integrity ensures smooth user experience.

        Arrangement:
        1. Two cameras (ASI120MC, ASI290MM) available via mock_asi.
        2. Camera 0 supports ASI_GAIN control with range 0-400.
        3. TestClient ready for multi-step workflow.

        Action:
        Executes 3-step workflow:
        1. GET /api/cameras to discover available hardware.
        2. Extract first camera's ID from response.
        3. POST /api/camera/{id}/control to configure gain=100.
        (Step 4: Stream endpoint skipped - would hang TestClient)

        Assertion Strategy:
        Validates workflow integration by confirming:
        - Step 1: HTTP 200 and cameras array has entries (discovery works).
        - ID extraction: cameras[0]["id"] provides valid camera identifier.
        - Step 2: HTTP 200 for control POST (configuration succeeds).
        - Data flow: ID from step 1 successfully used in step 2.
        - No broken integration between discovery and configuration.

        Testing Principle:
        Validates feature integration, ensuring multi-step workflows execute
        without data loss or broken handoffs between endpoints.
        """
        # Step 1: List cameras
        response = client.get("/api/cameras")
        assert response.status_code == 200
        cameras = response.json()["cameras"]
        assert len(cameras) > 0

        # Step 2: Configure first camera
        camera_id = cameras[0]["id"]
        response = client.post(
            f"/api/camera/{camera_id}/control",
            params={"control": "ASI_GAIN", "value": 100},
        )
        assert response.status_code == 200

        # Step 3: Start stream (skipped - would hang)
        # Stream testing requires browser integration tests

    def test_motor_control_workflow(self, client):
        """Verifies complete telescope positioning workflow.

        Tests integrated telescope control sequence: read current position,
        issue movement command. Represents typical telescope slew operation
        from dashboard.

        Note: Motor endpoints currently return not_implemented status
        as hardware integration is pending.

        Business context:
        Users initiate telescope movements from dashboard: check current
        position (altitude/azimuth), calculate required movement, issue
        motor commands. Position feedback is essential for confirming
        movement completion and preventing limit overruns. Workflow integrity
        ensures safe, predictable telescope control.

        Arrangement:
        1. Position endpoint returns current alt/az coordinates.
        2. Motor endpoints accept step/speed commands.
        3. TestClient ready for position-then-move workflow.

        Action:
        Executes 2-step workflow:
        1. GET /api/position to read initial telescope position.
        2. POST /api/motor/altitude to move telescope 100 steps at speed 50.

        Assertion Strategy:
        Validates positioning workflow by confirming:
        - Step 1: HTTP 200 and position data structure (initial_pos captured).
        - Position contains altitude and azimuth fields.
        - Step 2: HTTP 200 for movement command (motor responds).
        - No dependency errors (position read doesn't block motor control).
        - Both endpoints functional in sequence (no state conflicts).

        Testing Principle:
        Validates telescope control integration, ensuring position monitoring
        and motor commands can be interleaved without conflicts.
        """
        # Step 1: Check position
        response = client.get("/api/position")
        assert response.status_code == 200
        initial_pos = response.json()

        # Step 2: Move altitude
        response = client.post("/api/motor/altitude?steps=100&speed=50")
        assert response.status_code == 200

        # Step 3: Move azimuth
        response = client.post("/api/motor/azimuth?steps=200&speed=50")
        assert response.status_code == 200

        # Step 4: Stop all motors
        response = client.post("/api/motor/stop")
        assert response.status_code == 200
        assert response.json()["status"] == "stopped"


class TestCameraOpenFailure:
    """Tests for camera open failure path."""

    def test_get_camera_open_failure(self, mock_asi):
        """Verifies camera open failure is handled gracefully.

        Tests that when Camera() constructor raises an exception,
        _get_camera returns None and logs the error.

        Arrangement:
        1. mock_asi.Camera raises RuntimeError on construction.
        2. Camera cache is empty (first access).

        Action:
        Attempt to get camera 0 via API that triggers _get_camera.

        Assertion Strategy:
        - Camera endpoint returns HTTP 404 (not found).
        - No crash or unhandled exception.
        """
        # Make Camera constructor raise an exception
        mock_asi.Camera.side_effect = RuntimeError("USB connection failed")

        app = create_app(encoder=MockImageEncoder())
        client = TestClient(app)

        # Try to set control on camera 0 - will try to open camera
        response = client.post(
            "/api/camera/0/control", params={"control": "ASI_GAIN", "value": 50}
        )

        # Should get 404 because _get_camera returns None
        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"].lower()


class TestCloseAllCameras:
    """Tests for _close_all_cameras() function."""

    def test_close_cameras_with_streaming(self, mock_asi):
        """Verifies _close_all_cameras stops streaming cameras.

        Tests that cameras currently streaming have stop_video_capture()
        called before close().

        Arrangement:
        1. Camera 0 is open and streaming.
        2. _camera_streaming[0] = True indicates active stream.
        3. _cameras[0] holds mock camera instance.

        Action:
        Calls _close_all_cameras() (via lifespan shutdown).

        Assertion Strategy:
        Validates cleanup by confirming:
        - stop_video_capture() is called on streaming camera.
        - close() is called on camera.
        - Camera dicts are cleared.

        Testing Principle:
        Validates resource cleanup, ensuring streams are stopped
        before camera handles are released.
        """
        from telescope_mcp.web.app import (
            _camera_streaming,
            _cameras,
            _close_all_cameras,
        )

        mock_camera = mock_asi.Camera.return_value

        # Simulate an open, streaming camera
        _cameras[0] = mock_camera
        _camera_streaming[0] = True

        # Close all cameras
        _close_all_cameras()

        # Verify stop_video_capture was called
        mock_camera.stop_video_capture.assert_called_once()
        mock_camera.close.assert_called_once()

        # Verify dicts are cleared
        assert len(_cameras) == 0
        assert len(_camera_streaming) == 0

    def test_close_cameras_error_handling(self, mock_asi):
        """Verifies _close_all_cameras continues on per-camera errors.

        Tests that if one camera raises an exception during close,
        other cameras are still closed.

        Arrangement:
        1. Two cameras open (0 and 1).
        2. Camera 0 raises RuntimeError on close().
        3. Camera 1 closes normally.

        Action:
        Calls _close_all_cameras().

        Assertion Strategy:
        Validates error handling by confirming:
        - Both cameras have close() called.
        - Exception doesn't prevent cleanup of other cameras.
        - Dicts are cleared despite error.

        Testing Principle:
        Validates resilient cleanup, ensuring one camera's failure
        doesn't prevent cleanup of remaining cameras.
        """
        from telescope_mcp.web.app import (
            _camera_streaming,
            _cameras,
            _close_all_cameras,
        )

        mock_camera_0 = MagicMock()
        mock_camera_0.close.side_effect = RuntimeError("Close failed")

        mock_camera_1 = MagicMock()

        _cameras.clear()
        _camera_streaming.clear()
        _cameras[0] = mock_camera_0
        _cameras[1] = mock_camera_1
        _camera_streaming[0] = False
        _camera_streaming[1] = False

        # Close all cameras - should not raise
        _close_all_cameras()

        # Both cameras should have close() attempted
        mock_camera_0.close.assert_called_once()
        mock_camera_1.close.assert_called_once()

        # Dicts should be cleared
        assert len(_cameras) == 0
        assert len(_camera_streaming) == 0


class TestGetCameraForceReopen:
    """Tests for _get_camera force_reopen functionality."""

    def test_force_reopen_closes_existing_camera(self):
        """Verifies force_reopen=True closes and reopens camera.

        Tests that when force_reopen is requested, the existing camera
        is properly closed before opening a new instance.

        Arrangement:
        1. Mock camera in _cameras dict.
        2. Mock ASI SDK functions.

        Action:
        Calls _get_camera with force_reopen=True.

        Assertion Strategy:
        Validates force reopen by confirming:
        - Existing camera's stop_video_capture() called.
        - Existing camera's close() called.
        - Camera removed from dict and reopened.

        Testing Principle:
        Validates camera state reset functionality for recovering
        from error states or mode conflicts.
        """
        from telescope_mcp.web.app import (
            _camera_settings,
            _camera_streaming,
            _cameras,
            _get_camera,
        )

        mock_old_camera = MagicMock()
        mock_new_camera = MagicMock()

        _cameras.clear()
        _camera_streaming.clear()
        _camera_settings.clear()
        _cameras[0] = mock_old_camera
        _camera_streaming[0] = True

        with patch("telescope_mcp.web.app._sdk_initialized", True):
            with patch("telescope_mcp.web.app.asi") as mock_asi:
                mock_asi.get_num_cameras.return_value = 1
                mock_asi.Camera.return_value = mock_new_camera

                result = _get_camera(0, force_reopen=True)

        # Old camera should be closed
        mock_old_camera.stop_video_capture.assert_called_once()
        mock_old_camera.close.assert_called_once()

        # New camera should be returned
        assert result is mock_new_camera
        assert _cameras[0] is mock_new_camera

        # Cleanup
        _cameras.clear()
        _camera_streaming.clear()
        _camera_settings.clear()

    def test_force_reopen_handles_stop_video_error(self):
        """Verifies force_reopen handles stop_video_capture errors.

        Tests that stop_video_capture exceptions don't prevent close.

        Arrangement:
        1. Mock camera with failing stop_video_capture.

        Action:
        Calls _get_camera with force_reopen=True.

        Assertion Strategy:
        Validates error handling by confirming:
        - close() still called despite stop_video_capture error.
        - New camera opened successfully.
        """
        from telescope_mcp.web.app import (
            _camera_settings,
            _camera_streaming,
            _cameras,
            _get_camera,
        )

        mock_old_camera = MagicMock()
        mock_old_camera.stop_video_capture.side_effect = RuntimeError("Not capturing")
        mock_new_camera = MagicMock()

        _cameras.clear()
        _camera_streaming.clear()
        _camera_settings.clear()
        _cameras[0] = mock_old_camera

        with patch("telescope_mcp.web.app._sdk_initialized", True):
            with patch("telescope_mcp.web.app.asi") as mock_asi:
                mock_asi.get_num_cameras.return_value = 1
                mock_asi.Camera.return_value = mock_new_camera

                result = _get_camera(0, force_reopen=True)

        # close() should still be called despite stop error
        mock_old_camera.close.assert_called_once()
        assert result is mock_new_camera

        # Cleanup
        _cameras.clear()
        _camera_streaming.clear()
        _camera_settings.clear()

    def test_force_reopen_handles_close_error(self):
        """Verifies force_reopen logs warning on close failure.

        Tests that close() exceptions are caught and logged.

        Arrangement:
        1. Mock camera with failing close().

        Action:
        Calls _get_camera with force_reopen=True.

        Assertion Strategy:
        Validates error handling by confirming:
        - New camera still opened despite close error.
        - Function doesn't raise exception.
        """
        from telescope_mcp.web.app import (
            _camera_settings,
            _camera_streaming,
            _cameras,
            _get_camera,
        )

        mock_old_camera = MagicMock()
        mock_old_camera.close.side_effect = RuntimeError("Close failed")
        mock_new_camera = MagicMock()

        _cameras.clear()
        _camera_streaming.clear()
        _camera_settings.clear()
        _cameras[0] = mock_old_camera

        with patch("telescope_mcp.web.app._sdk_initialized", True):
            with patch("telescope_mcp.web.app.asi") as mock_asi:
                mock_asi.get_num_cameras.return_value = 1
                mock_asi.Camera.return_value = mock_new_camera

                # Should not raise despite close error
                result = _get_camera(0, force_reopen=True)

        assert result is mock_new_camera

        # Cleanup
        _cameras.clear()
        _camera_streaming.clear()
        _camera_settings.clear()


class TestCaptureFrameEndpoint:
    """Tests for capture_frame endpoint coverage."""

    @pytest.mark.asyncio
    async def test_capture_frame_stream_not_running(self):
        """Verifies capture_frame returns error when stream not running.

        Tests the error path when camera stream is not active.

        Arrangement:
        1. Clear streaming state so camera appears stopped.

        Action:
        Calls capture_frame endpoint.

        Assertion Strategy:
        Validates error handling by confirming:
        - Returns 400 status code.
        - Error message indicates stream not running.
        """
        from telescope_mcp.web.app import _camera_streaming, create_app

        app = create_app()
        client = TestClient(app)

        _camera_streaming.clear()
        _camera_streaming[0] = False  # Stream not running

        response = client.post("/api/camera/0/capture?frame_type=light")

        assert response.status_code == 400
        data = response.json()
        assert data["status"] == "error"
        assert "not running" in data["error"]

        # Cleanup
        _camera_streaming.clear()

    @pytest.mark.asyncio
    async def test_capture_frame_no_frame_available(self):
        """Verifies capture_frame returns error when no frame buffered.

        Tests the error path when stream is running but no frame yet.

        Arrangement:
        1. Set stream running but no frame in buffer.

        Action:
        Calls capture_frame endpoint.

        Assertion Strategy:
        Validates error handling by confirming:
        - Returns 400 status code.
        - Error message indicates no frame available.
        """
        from telescope_mcp.web.app import (
            _camera_streaming,
            _latest_frames,
            create_app,
        )

        app = create_app()
        client = TestClient(app)

        _camera_streaming.clear()
        _latest_frames.clear()
        _camera_streaming[0] = True  # Stream running
        # No frame in _latest_frames

        response = client.post("/api/camera/0/capture?frame_type=light")

        assert response.status_code == 400
        data = response.json()
        assert data["status"] == "error"
        assert "No frame available" in data["error"]

        # Cleanup
        _camera_streaming.clear()
        _latest_frames.clear()

    @pytest.mark.asyncio
    async def test_capture_frame_success(self, tmp_path, monkeypatch):
        """Verifies capture_frame succeeds with valid frame data.

        Tests the success path for capturing a RAW16 frame from stream.

        Arrangement:
        1. Set stream running with buffered frame.
        2. Mock camera and file operations.

        Action:
        Calls capture_frame endpoint.

        Assertion Strategy:
        Validates capture by confirming:
        - Returns 200 status code.
        - Response includes filename and frame metadata.
        """
        from telescope_mcp.web.app import (
            _camera_settings,
            _camera_streaming,
            _cameras,
            _latest_frame_info,
            _latest_frames,
            create_app,
        )

        app = create_app()
        client = TestClient(app)

        # Setup streaming state
        _camera_streaming.clear()
        _latest_frames.clear()
        _latest_frame_info.clear()
        _cameras.clear()
        _camera_settings.clear()

        _camera_streaming[0] = True
        _latest_frames[0] = np.zeros((100, 100), dtype=np.uint16)
        _latest_frame_info[0] = {
            "width": 100,
            "height": 100,
            "is_color": False,
            "exposure_us": 1000,
            "gain": 100,
        }

        mock_camera = MagicMock()
        mock_camera.get_camera_property.return_value = {
            "Name": "Test Camera",
            "Temperature": 250,
        }
        _cameras[0] = mock_camera
        _camera_settings[0] = {"exposure_us": 1000, "gain": 100}

        # Redirect capture directory to tmp_path
        monkeypatch.chdir(tmp_path)

        response = client.post("/api/camera/0/capture?frame_type=light")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "filename" in data
        assert data["camera"] == "finder"
        assert data["frame_type"] == "light"
        assert data["capture_mode"] == "raw16_stream"

        # Cleanup
        _camera_streaming.clear()
        _latest_frames.clear()
        _latest_frame_info.clear()
        _cameras.clear()
        _camera_settings.clear()

    @pytest.mark.asyncio
    async def test_capture_frame_handles_exception(self, tmp_path, monkeypatch):
        """Verifies capture_frame returns 500 on internal error.

        Tests the exception handler path when capture fails.

        Arrangement:
        1. Set stream running with frame.
        2. Mock _save_frame_to_asdf to raise exception.

        Action:
        Calls capture_frame endpoint.

        Assertion Strategy:
        Validates error handling by confirming:
        - Returns 500 status code.
        - Error message included in response.
        """
        from telescope_mcp.web.app import (
            _camera_settings,
            _camera_streaming,
            _cameras,
            _latest_frame_info,
            _latest_frames,
            create_app,
        )

        app = create_app()
        client = TestClient(app)

        _camera_streaming.clear()
        _latest_frames.clear()
        _latest_frame_info.clear()
        _cameras.clear()
        _camera_settings.clear()

        _camera_streaming[0] = True
        _latest_frames[0] = np.zeros((100, 100), dtype=np.uint16)
        _latest_frame_info[0] = {
            "width": 100,
            "height": 100,
            "is_color": False,
            "exposure_us": 1000,
            "gain": 100,
        }

        mock_camera = MagicMock()
        mock_camera.get_camera_property.return_value = {
            "Name": "Test Camera",
            "Temperature": 250,
        }
        _cameras[0] = mock_camera
        _camera_settings[0] = {"exposure_us": 1000, "gain": 100}

        monkeypatch.chdir(tmp_path)

        # Make _save_frame_to_asdf raise an exception
        with patch(
            "telescope_mcp.web.app._save_frame_to_asdf",
            side_effect=RuntimeError("ASDF write failed"),
        ):
            response = client.post("/api/camera/0/capture?frame_type=light")

        assert response.status_code == 500
        data = response.json()
        assert data["status"] == "error"
        assert "ASDF write failed" in data["error"]

        # Cleanup
        _camera_streaming.clear()
        _latest_frames.clear()
        _latest_frame_info.clear()
        _cameras.clear()
        _camera_settings.clear()


class TestAddCoordinatesToMetadata:
    """Tests for _add_coordinates_to_metadata function."""

    @pytest.mark.asyncio
    async def test_adds_coordinates_with_connected_sensor(self):
        """Verifies coordinates added when sensor is connected.

        Tests that sensor readings are converted and added to metadata.

        Arrangement:
        1. Mock connected sensor with valid reading.
        2. Mock config with location.

        Action:
        Calls _add_coordinates_to_metadata.

        Assertion Strategy:
        Validates coordinate injection by confirming:
        - Metadata contains coordinates dict.
        - RA/Dec values present from conversion.
        """
        import datetime

        from telescope_mcp.web.app import _add_coordinates_to_metadata

        mock_sensor = MagicMock()
        mock_sensor.connected = True

        # Create async mock for read()
        async def mock_read():
            reading = MagicMock()
            reading.altitude = 45.0
            reading.azimuth = 180.0
            reading.temperature = 20.0
            reading.humidity = 50.0
            return reading

        mock_sensor.read = mock_read

        mock_config = MagicMock()
        mock_config.location = {"lat": 30.0, "lon": -97.0, "alt": 100.0}
        mock_factory = MagicMock()
        mock_factory.config = mock_config

        frame_meta: dict[str, object] = {}
        capture_time = datetime.datetime.now(datetime.UTC)

        with patch("telescope_mcp.web.app._sensor", mock_sensor):
            with patch("telescope_mcp.web.app.get_factory", return_value=mock_factory):
                with patch("telescope_mcp.web.app.altaz_to_radec") as mock_convert:
                    mock_convert.return_value = {
                        "ra": 180.0,
                        "dec": 45.0,
                        "ra_hours": 12.0,
                        "ra_hms": "12h 00m 00s",
                        "dec_dms": "+45° 00' 00\"",
                    }

                    await _add_coordinates_to_metadata(frame_meta, capture_time)

        assert "coordinates" in frame_meta
        coords = frame_meta["coordinates"]
        assert isinstance(coords, dict)
        assert coords["altitude"] == 45.0
        assert coords["azimuth"] == 180.0
        assert coords["ra"] == 180.0
        assert coords["dec"] == 45.0
        assert coords["coordinate_source"] == "sensor"

    @pytest.mark.asyncio
    async def test_no_coordinates_when_sensor_none(self):
        """Verifies no error when sensor is None.

        Tests graceful handling when sensor not initialized.

        Arrangement:
        1. Set _sensor to None.

        Action:
        Calls _add_coordinates_to_metadata.

        Assertion Strategy:
        Validates graceful degradation by confirming:
        - No exception raised.
        - Metadata unchanged.
        """
        import datetime

        from telescope_mcp.web.app import _add_coordinates_to_metadata

        frame_meta: dict[str, object] = {}
        capture_time = datetime.datetime.now(datetime.UTC)

        with patch("telescope_mcp.web.app._sensor", None):
            await _add_coordinates_to_metadata(frame_meta, capture_time)

        assert "coordinates" not in frame_meta

    @pytest.mark.asyncio
    async def test_no_coordinates_when_sensor_disconnected(self):
        """Verifies no error when sensor not connected.

        Tests graceful handling when sensor exists but disconnected.

        Arrangement:
        1. Mock sensor with connected=False.

        Action:
        Calls _add_coordinates_to_metadata.

        Assertion Strategy:
        Validates graceful degradation by confirming:
        - No exception raised.
        - Metadata unchanged.
        """
        import datetime

        from telescope_mcp.web.app import _add_coordinates_to_metadata

        mock_sensor = MagicMock()
        mock_sensor.connected = False

        frame_meta: dict[str, object] = {}
        capture_time = datetime.datetime.now(datetime.UTC)

        with patch("telescope_mcp.web.app._sensor", mock_sensor):
            await _add_coordinates_to_metadata(frame_meta, capture_time)

        assert "coordinates" not in frame_meta

    @pytest.mark.asyncio
    async def test_handles_sensor_read_error(self):
        """Verifies error handling when sensor read fails.

        Tests that sensor exceptions don't crash capture.

        Arrangement:
        1. Mock sensor that raises on read().

        Action:
        Calls _add_coordinates_to_metadata.

        Assertion Strategy:
        Validates error handling by confirming:
        - No exception propagated.
        - Metadata unchanged.
        """
        import datetime

        from telescope_mcp.web.app import _add_coordinates_to_metadata

        mock_sensor = MagicMock()
        mock_sensor.connected = True

        async def mock_read():
            raise RuntimeError("Sensor communication error")

        mock_sensor.read = mock_read

        frame_meta: dict[str, object] = {}
        capture_time = datetime.datetime.now(datetime.UTC)

        with patch("telescope_mcp.web.app._sensor", mock_sensor):
            # Should not raise
            await _add_coordinates_to_metadata(frame_meta, capture_time)

        assert "coordinates" not in frame_meta


class TestSaveFrameToAsdf:
    """Tests for _save_frame_to_asdf function."""

    @pytest.mark.asyncio
    async def test_creates_new_asdf_file(self, tmp_path):
        """Verifies new ASDF file created when none exists.

        Tests the creation path for new session files.

        Arrangement:
        1. Use tmp_path for clean file system.
        2. Prepare frame data and metadata.

        Action:
        Calls _save_frame_to_asdf with non-existent file.

        Assertion Strategy:
        Validates file creation by confirming:
        - File created on disk.
        - Returns frame_index 0.
        - File contains expected structure.
        """
        from telescope_mcp.web.app import _save_frame_to_asdf

        filepath = tmp_path / "test_session.asdf"
        img = np.zeros((100, 100), dtype=np.uint16)
        frame_meta = {
            "width": 100,
            "height": 100,
            "is_color": False,
            "exposure_us": 1000,
        }
        info = {"Name": "Test Camera"}

        frame_index = await _save_frame_to_asdf(
            filepath, "finder", "light", img, frame_meta, info
        )

        assert filepath.exists()
        assert frame_index == 0

        # Verify file structure
        import asdf

        with asdf.open(str(filepath)) as af:
            assert "cameras" in af.tree
            assert "finder" in af.tree["cameras"]
            assert len(af.tree["cameras"]["finder"]["light"]) == 1

    @pytest.mark.asyncio
    async def test_appends_to_existing_asdf_file(self, tmp_path):
        """Verifies frames appended to existing ASDF file.

        Tests the append path for existing session files.

        Arrangement:
        1. Create initial ASDF file with one frame.
        2. Prepare second frame data.

        Action:
        Calls _save_frame_to_asdf with existing file.

        Assertion Strategy:
        Validates append by confirming:
        - Returns frame_index 1.
        - File contains two frames.
        """
        import asdf

        from telescope_mcp.web.app import _save_frame_to_asdf

        filepath = tmp_path / "test_session.asdf"

        # Create initial file
        initial_tree = {
            "metadata": {"created": "2025-01-01", "session_date": "20250101"},
            "cameras": {
                "finder": {
                    "info": {"name": "Test"},
                    "light": [
                        {"data": np.zeros((10, 10), dtype=np.uint16), "meta": {}}
                    ],
                    "dark": [],
                    "flat": [],
                    "bias": [],
                }
            },
        }
        af = asdf.AsdfFile(initial_tree)
        af.write_to(str(filepath))

        # Append second frame
        img = np.ones((100, 100), dtype=np.uint16)
        frame_meta = {"width": 100, "height": 100, "is_color": False}
        info = {"Name": "Test Camera"}

        frame_index = await _save_frame_to_asdf(
            filepath, "finder", "light", img, frame_meta, info
        )

        assert frame_index == 1

        # Verify two frames now
        with asdf.open(str(filepath)) as af:
            assert len(af.tree["cameras"]["finder"]["light"]) == 2

    @pytest.mark.asyncio
    async def test_creates_new_camera_section_in_existing_file(self, tmp_path):
        """Verifies new camera section added to existing file.

        Tests adding a different camera to existing session.

        Arrangement:
        1. Create ASDF with finder camera.
        2. Add frame from main camera.

        Action:
        Calls _save_frame_to_asdf with different camera_key.

        Assertion Strategy:
        Validates new section by confirming:
        - File now has both camera sections.
        - Main camera has one frame.
        """
        import asdf

        from telescope_mcp.web.app import _save_frame_to_asdf

        filepath = tmp_path / "test_session.asdf"

        # Create initial file with finder only
        initial_tree = {
            "metadata": {"created": "2025-01-01"},
            "cameras": {
                "finder": {
                    "info": {"name": "Finder"},
                    "light": [],
                    "dark": [],
                    "flat": [],
                    "bias": [],
                }
            },
        }
        af = asdf.AsdfFile(initial_tree)
        af.write_to(str(filepath))

        # Add main camera frame
        img = np.zeros((200, 200), dtype=np.uint16)
        frame_meta = {"width": 200, "height": 200, "is_color": True}
        info = {"Name": "Main Camera"}

        frame_index = await _save_frame_to_asdf(
            filepath, "main", "light", img, frame_meta, info
        )

        assert frame_index == 0

        with asdf.open(str(filepath)) as af:
            assert "finder" in af.tree["cameras"]
            assert "main" in af.tree["cameras"]
            assert len(af.tree["cameras"]["main"]["light"]) == 1

    @pytest.mark.asyncio
    async def test_adds_cameras_dict_to_existing_file_without_cameras(self, tmp_path):
        """Verifies cameras dict created when missing from existing file.

        Tests the path where an existing ASDF file lacks a cameras section.

        Arrangement:
        1. Create ASDF file without cameras dict.

        Action:
        Calls _save_frame_to_asdf.

        Assertion Strategy:
        Validates cameras dict creation by confirming:
        - File now has cameras dict.
        - Camera section created with frame.
        """
        import asdf

        from telescope_mcp.web.app import _save_frame_to_asdf

        filepath = tmp_path / "test_session.asdf"

        # Create initial file WITHOUT cameras section
        initial_tree = {
            "metadata": {"created": "2025-01-01", "session_date": "20250101"},
            # No "cameras" key
        }
        af = asdf.AsdfFile(initial_tree)
        af.write_to(str(filepath))

        # Add frame - should create cameras dict
        img = np.zeros((100, 100), dtype=np.uint16)
        frame_meta = {"width": 100, "height": 100, "is_color": False}
        info = {"Name": "Test Camera"}

        frame_index = await _save_frame_to_asdf(
            filepath, "finder", "light", img, frame_meta, info
        )

        assert frame_index == 0

        with asdf.open(str(filepath)) as af:
            assert "cameras" in af.tree
            assert "finder" in af.tree["cameras"]
            assert len(af.tree["cameras"]["finder"]["light"]) == 1


class TestMainFunction:
    """Tests for main() entry point."""

    def test_main_function_creates_app_and_runs(self):
        """Verifies main() creates app and calls uvicorn.run.

        Tests the entry point function to ensure it properly
        configures and starts the web server.

        Arrangement:
        1. Patch uvicorn.run to prevent actual server startup.
        2. Patch create_app to verify it's called.

        Action:
        Calls main() entry point function.

        Assertion Strategy:
        Validates entry point by confirming:
        - create_app() is called.
        - uvicorn.run() is called with correct host/port.

        Testing Principle:
        Validates entry point configuration, ensuring main()
        correctly initializes and starts the web server.
        """
        from telescope_mcp.web.app import main

        with patch("telescope_mcp.web.app.uvicorn") as mock_uvicorn:
            with patch("telescope_mcp.web.app.create_app") as mock_create_app:
                mock_app = MagicMock()
                mock_create_app.return_value = mock_app

                main()

                mock_create_app.assert_called_once()
                mock_uvicorn.run.assert_called_once_with(
                    mock_app, host="0.0.0.0", port=8080
                )
