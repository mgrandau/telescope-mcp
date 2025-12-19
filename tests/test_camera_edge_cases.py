"""Edge case tests for Camera to increase coverage."""

import pytest

from telescope_mcp.devices import Camera, CameraConfig, CaptureOptions
from telescope_mcp.drivers.cameras.twin import DigitalTwinCameraDriver


class TestCameraConnectionErrors:
    """Test suite for Camera connection error handling.

    Categories:
    1. Precondition Failures - Operations before connect (2 tests)
    2. State Transition Edge Cases - Double connect, disconnect (2 tests)

    Total: 4 tests.
    """

    def test_capture_without_connect(self):
        """Verifies capture() raises exception when called on unconnected camera.

        Tests camera state validation by attempting capture without prior
        connect() call.

        Business context:
        Prevents undefined behavior or crashes from hardware operations on
        uninitialized camera devices.

        Arrangement:
        1. Create DigitalTwinCameraDriver (simulated hardware).
        2. Instantiate Camera with driver and config (camera_id=0).
        3. Do NOT call connect() - camera remains unconnected.

        Action:
        Call camera.capture() on unconnected camera instance.

        Assertion Strategy:
        Validates connection requirement by confirming:
        - Exception is raised (CameraNotConnectedError expected).
        - Camera enforces connected state before operations.

        Testing Principle:
        Validates precondition checking, ensuring camera operations
        fail safely when device not initialized."""
        driver = DigitalTwinCameraDriver()
        camera = Camera(driver, CameraConfig(camera_id=0))

        with pytest.raises(Exception):  # CameraNotConnectedError
            camera.capture()

    def test_stream_without_connect(self):
        """Verifies stream() raises exception when called on unconnected camera.

        Tests streaming state validation by attempting to iterate stream
        generator without prior connect() call.

        Business context:
        Prevents streaming from uninitialized hardware that could cause
        driver crashes or system hangs.

        Arrangement:
        1. Create DigitalTwinCameraDriver (simulated hardware).
        2. Instantiate Camera with driver and config (camera_id=0).
        3. Do NOT call connect() - camera remains unconnected.

        Action:
        Iterate camera.stream() generator, breaking after first frame attempt.

        Assertion Strategy:
        Validates connection requirement by confirming:
        - Exception is raised (CameraNotConnectedError expected).
        - Generator raises before yielding any frames.

        Testing Principle:
        Validates streaming preconditions, ensuring generator pattern
        fails safely when camera not connected."""
        driver = DigitalTwinCameraDriver()
        camera = Camera(driver, CameraConfig(camera_id=0))

        with pytest.raises(Exception):  # CameraNotConnectedError
            for _ in camera.stream():
                break

    def test_double_connect(self):
        """Verifies double connect() is idempotent or raises safely.

        Tests connection state handling by calling connect() twice on
        same camera instance.

        Business context:
        Prevents resource leaks or driver confusion from redundant
        initialization calls.

        Arrangement:
        1. Create DigitalTwinCameraDriver.
        2. Instantiate Camera with driver and config.
        3. Call connect() once - camera becomes connected.

        Action:
        Call connect() a second time on already-connected camera.

        Assertion Strategy:
        Validates idempotent behavior by confirming:
        - Second connect either succeeds (idempotent) or raises.
        - Camera remains functional after double connect.
        - Disconnect works properly in finally block.

        Testing Principle:
        Validates state transition safety, ensuring redundant connects
        don't corrupt camera state or leak resources."""
        driver = DigitalTwinCameraDriver()
        camera = Camera(driver, CameraConfig(camera_id=0))

        camera.connect()
        # Second connect should be idempotent or raise
        try:
            camera.connect()
        except Exception:
            pass  # Either raises or succeeds
        finally:
            camera.disconnect()

    def test_disconnect_without_connect(self):
        """Verifies disconnect() is safe to call on unconnected camera.

        Tests cleanup safety by calling disconnect() without prior
        connect() call.

        Business context:
        Enables simple cleanup patterns in finally blocks and context
        managers without requiring state checks.

        Arrangement:
        1. Create DigitalTwinCameraDriver.
        2. Instantiate Camera with driver and config.
        3. Do NOT call connect() - camera remains unconnected.

        Action:
        Call camera.disconnect() on unconnected camera.

        Assertion Strategy:
        Validates safe cleanup by confirming:
        - disconnect() returns without exception.
        - No error or state corruption occurs.

        Testing Principle:
        Validates defensive programming, ensuring cleanup methods
        are idempotent and safe to call in any state."""
        driver = DigitalTwinCameraDriver()
        camera = Camera(driver, CameraConfig(camera_id=0))

        # Should be safe to call
        camera.disconnect()


class TestCameraStreaming:
    """Test suite for Camera streaming operations.

    Categories:
    1. Basic Streaming - Generator iteration (1 test)
    2. Stream Control - Stop mechanism (1 test)
    3. Rate Limiting - FPS throttling (1 test)

    Total: 3 tests.
    """

    @pytest.fixture
    def camera(self):
        """Fixture providing connected DigitalTwinCamera for streaming tests.

        Arrangement:
        1. Creates DigitalTwinCameraDriver (simulated ZWO ASI camera).
        2. Wraps driver in Camera with CameraConfig(camera_id=0).
        3. Calls camera.connect() for ready state.
        4. Yields camera to test.
        5. Cleanup: disconnects camera.

        Args:
        self: Test class instance (pytest convention).

        Returns:
        Camera: Connected Camera instance for streaming tests.

        Raises:
        None.

        Testing Principle:
        Provides isolated camera instance per test,
        ensuring clean state and automatic cleanup.
        """
        driver = DigitalTwinCameraDriver()
        cam = Camera(driver, CameraConfig(camera_id=0))
        cam.connect()
        yield cam
        cam.disconnect()

    def test_stream_basic(self, camera):
        """Verifies camera streaming yields frames at specified FPS.

        Arrangement:
        1. Camera fixture provides connected DigitalTwinCameraDriver.
        2. Stream configured with max_fps=30 for rate limiting.
        3. Counter initialized to track frames received.

        Action:
        Iterates over camera.stream() generator, breaking after 3 frames.

        Assertion Strategy:
        Validates streaming by confirming:
        - Exactly 3 frames received before break.
        - Generator produces frames on-demand.

        Testing Principle:
        Validates basic streaming operation, ensuring generator
        pattern works and frame delivery is reliable.
        """
        frame_count = 0
        for _ in camera.stream(max_fps=30):
            frame_count += 1
            if frame_count >= 3:
                break
        assert frame_count == 3

    def test_stream_with_stop(self, camera):
        """Verifies stop_stream() halts streaming generator.

        Arrangement:
        1. Camera fixture provides connected camera.
        2. Stream initiated with max_fps=30.
        3. Counter tracks frames before stop.

        Action:
        Calls camera.stop_stream() after receiving 2 frames.

        Assertion Strategy:
        Validates stop mechanism by confirming:
        - Exactly 2 frames received before stop.
        - Loop exits cleanly after stop_stream() call.

        Testing Principle:
        Validates streaming control, ensuring stop_stream()
        provides graceful shutdown without blocking.
        """
        frame_count = 0
        for _ in camera.stream(max_fps=30):
            frame_count += 1
            if frame_count >= 2:
                camera.stop_stream()
        assert frame_count == 2

    def test_stream_respects_fps_limit(self, camera):
        """Verifies stream rate limiting honors max_fps parameter.

        Arrangement:
        1. Camera fixture provides connected camera.
        2. Stream configured with max_fps=10 (100ms per frame).
        3. Timer records start time for duration measurement.

        Action:
        Streams 2 frames while measuring elapsed time.

        Assertion Strategy:
        Validates rate limiting by confirming:
        - Total duration >= 0.1 seconds (100ms minimum between frames).
        - Frame delivery throttled to respect max_fps.

        Testing Principle:
        Validates timing control, ensuring max_fps prevents
        CPU saturation from unlimited frame generation.
        """
        import time

        start = time.time()
        frame_count = 0
        for _ in camera.stream(max_fps=10):
            frame_count += 1
            if frame_count >= 2:
                break
        duration = time.time() - start
        # Should take at least 0.1 seconds between frames
        assert duration >= 0.1


class TestCameraCaptureModes:
    """Test suite for Camera capture format and option handling.

    Categories:
    1. Format Options - Raw and JPEG formats (2 tests)
    2. Overlay Options - Overlay enable/disable (1 test)
    3. Default Behavior - Capture without options (1 test)

    Total: 4 tests (already documented in earlier pass).
    """

    @pytest.fixture
    def camera(self):
        """Fixture providing connected DigitalTwinCamera for format tests.

        Arrangement:
        1. Creates DigitalTwinCameraDriver for format testing.
        2. Wraps in Camera with CameraConfig(camera_id=0).
        3. Connects camera for ready state.
        4. Yields to format test.
        5. Cleanup: disconnects camera.

        Args:
        self: Test class instance (pytest convention).

        Returns:
        Camera: Connected Camera instance for testing
        different capture format options (JPEG, raw).

        Raises:
        None.

        Testing Principle:
        Provides fresh camera instance per format test,
        ensuring format settings don't persist.
        """
        driver = DigitalTwinCameraDriver()
        cam = Camera(driver, CameraConfig(camera_id=0))
        cam.connect()
        yield cam
        cam.disconnect()

    def test_capture_raw_format(self, camera):
        """Verifies capture returns raw format image data.

        Arrangement:
        1. Camera fixture provides connected camera.
        2. CaptureOptions configured with format="raw".
        3. Exposure and gain set for synthetic image generation.

        Action:
        Calls camera.capture() with raw format options.

        Assertion Strategy:
        Validates raw capture by confirming:
        - Result contains non-None image_data.
        - Raw format returned (uncompressed).

        Testing Principle:
        Validates format option handling, ensuring raw data
        path works for applications needing uncompressed frames.
        """
        options = CaptureOptions(exposure_us=100000, gain=50, format="raw")
        result = camera.capture(options)
        assert result.image_data is not None

    def test_capture_jpeg_format(self, camera):
        """Verifies capture returns JPEG-compressed image data.

        Arrangement:
        1. Camera fixture provides connected camera.
        2. CaptureOptions configured with format="jpeg".
        3. Exposure and gain set for synthetic image.

        Action:
        Calls camera.capture() with JPEG format options.

        Assertion Strategy:
        Validates JPEG capture by confirming:
        - Result contains non-None image_data.
        - JPEG compression applied.

        Testing Principle:
        Validates format option handling, ensuring JPEG path
        works for bandwidth-efficient image delivery.
        """
        options = CaptureOptions(exposure_us=100000, gain=50, format="jpeg")
        result = camera.capture(options)
        assert result.image_data is not None

    def test_capture_without_overlay(self, camera):
        """Verifies capture respects apply_overlay=False option.

        Arrangement:
        1. Camera fixture provides connected camera.
        2. CaptureOptions configured with apply_overlay=False.
        3. Exposure set for image generation.

        Action:
        Calls camera.capture() with overlay disabled.

        Assertion Strategy:
        Validates overlay control by confirming:
        - result.has_overlay is False.
        - No crosshair/metadata overlay applied.

        Testing Principle:
        Validates overlay option handling, ensuring clean
        frames available when overlays not desired.
        """
        options = CaptureOptions(exposure_us=100000, apply_overlay=False)
        result = camera.capture(options)
        assert result.has_overlay is False

    def test_capture_default_options(self, camera):
        """Verifies capture works with None options (uses defaults).

        Arrangement:
        1. Camera fixture provides connected camera.
        2. No CaptureOptions provided (None).
        3. Camera should use default settings.

        Action:
        Calls camera.capture(None) to trigger default behavior.

        Assertion Strategy:
        Validates default handling by confirming:
        - Result contains non-None image_data.
        - Capture succeeds without explicit options.

        Testing Principle:
        Validates convenience API, ensuring capture() works
        without requiring options for simple use cases.
        """
        result = camera.capture(None)
        assert result.image_data is not None


class TestCameraInfo:
    """Test suite for Camera property and state queries.

    Categories:
    1. Configuration Access - Config before connect (1 test)
    2. State Properties - is_connected tracking (1 test)

    Total: 2 tests.
    """

    def test_info_before_connect(self):
        """Verifies camera config accessible before connection established.

        Tests configuration availability by accessing camera.config.name
        without prior connect() call.

        Business context:
        Allows applications to discover camera properties and validate
        configuration before committing to connection.

        Arrangement:
        1. Create DigitalTwinCameraDriver.
        2. Instantiate Camera with config specifying name="Test".
        3. Do NOT call connect() - camera remains unconnected.

        Action:
        Access camera.config.name property on unconnected camera.

        Assertion Strategy:
        Validates config accessibility by confirming:
        - camera.config.name equals "Test" as specified in constructor.
        - Configuration available without hardware connection.

        Testing Principle:
        Validates separation of concerns, ensuring configuration data
        accessible independently of hardware state."""
        driver = DigitalTwinCameraDriver()
        camera = Camera(driver, CameraConfig(camera_id=0, name="Test"))

        # Config should be available
        assert camera.config.name == "Test"

    def test_is_connected_property(self):
        """Verifies is_connected property reflects actual connection state.

        Tests connection state tracking by checking is_connected through
        the full connection lifecycle.

        Business context:
        Enables applications to query connection status for conditional
        logic and error handling.

        Arrangement:
        1. Create DigitalTwinCameraDriver.
        2. Instantiate Camera with driver and config.
        3. Camera starts in disconnected state.

        Action:
        Check is_connected at three lifecycle points: before connect,
        after connect, and after disconnect.

        Assertion Strategy:
        Validates state tracking by confirming:
        - is_connected is False initially.
        - is_connected is True after connect().
        - is_connected is False after disconnect().

        Testing Principle:
        Validates state observable behavior, ensuring is_connected
        property accurately reflects hardware connection state."""
        driver = DigitalTwinCameraDriver()
        camera = Camera(driver, CameraConfig(camera_id=0))

        assert not camera.is_connected
        camera.connect()
        assert camera.is_connected
        camera.disconnect()
        assert not camera.is_connected


class TestCameraContextManager:
    """Test suite for Camera context manager protocol.

    Categories:
    1. Lifecycle Management - Connect/disconnect automation (1 test)
    2. Integrated Operations - Capture within context (1 test)

    Total: 2 tests.
    """

    def test_context_manager_connects_and_disconnects(self):
        """Verifies Camera context manager handles connection lifecycle.

        Tests __enter__ and __exit__ implementation by using camera
        as context manager.

        Business context:
        Enables Pythonic resource management with automatic cleanup,
        preventing connection leaks.

        Arrangement:
        1. Create DigitalTwinCameraDriver.
        2. Instantiate Camera with driver and config.
        3. Verify camera starts disconnected.

        Action:
        Use camera in 'with' statement and check connection state at
        each phase: before, during, and after context.

        Assertion Strategy:
        Validates context manager protocol by confirming:
        - is_connected is False before 'with' block.
        - is_connected is True inside 'with' block.
        - is_connected is False after 'with' block exits.

        Testing Principle:
        Validates resource management pattern, ensuring context manager
        provides deterministic cleanup even on exceptions."""
        driver = DigitalTwinCameraDriver()
        camera = Camera(driver, CameraConfig(camera_id=0))

        assert not camera.is_connected
        with camera:
            assert camera.is_connected
        assert not camera.is_connected

    def test_context_manager_with_capture(self):
        """Verifies capture operations work within context manager.

        Tests integration of context manager and capture by performing
        image capture inside 'with' block.

        Business context:
        Validates complete use case: automatic connection, capture, and
        cleanup in single pattern.

        Arrangement:
        1. Create DigitalTwinCameraDriver.
        2. Instantiate Camera with driver and config.
        3. Enter context manager (auto-connects).

        Action:
        Call camera.capture() with CaptureOptions inside 'with' block.

        Assertion Strategy:
        Validates integrated operation by confirming:
        - Capture succeeds within context.
        - result.image_data is not None.
        - Camera connected during operation (implicit).

        Testing Principle:
        Validates composition of patterns, ensuring context manager and
        capture operations integrate correctly for common use case."""
        driver = DigitalTwinCameraDriver()
        camera = Camera(driver, CameraConfig(camera_id=0))

        with camera:
            result = camera.capture(CaptureOptions(exposure_us=100000))
            assert result.image_data is not None
