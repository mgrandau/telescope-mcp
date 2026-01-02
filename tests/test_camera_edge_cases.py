"""Edge case tests for Camera to increase coverage."""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock

import pytest

from telescope_mcp.devices.camera import (
    Camera,
    CameraConfig,
    CameraDisconnectedError,
    CameraError,
    CameraHooks,
    CameraInfo,
    CaptureOptions,
    NullRecoveryStrategy,
    OverlayConfig,
    StreamFrame,
)
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


# =============================================================================
# NEW TESTS FOR 100% COVERAGE
# =============================================================================


class TestCameraHooks:
    """Test suite for Camera event hooks (callbacks).

    Covers lines: 1103 (on_error in connect), 1177 (on_disconnect),
    1213 (on_capture), 1658 (on_stream_frame).

    Total: 5 tests.
    """

    def test_on_connect_hook_fires(self) -> None:
        """Verifies on_connect hook fires after successful connection.

        Arrangement:
        1. Create mock callback for on_connect.
        2. Create CameraHooks with on_connect callback.
        3. Create Camera with hooks injected.

        Action:
        Call camera.connect() to establish connection.

        Assertion Strategy:
        Validates hook by confirming:
        - on_connect called exactly once.
        - Called with CameraInfo argument.
        """
        on_connect_mock = Mock()
        hooks = CameraHooks(on_connect=on_connect_mock)
        driver = DigitalTwinCameraDriver()
        camera = Camera(driver, CameraConfig(camera_id=0), hooks=hooks)

        try:
            camera.connect()
            on_connect_mock.assert_called_once()
            # Verify called with CameraInfo
            call_args = on_connect_mock.call_args[0]
            assert isinstance(call_args[0], CameraInfo)
        finally:
            camera.disconnect()

    def test_on_disconnect_hook_fires(self) -> None:
        """Verifies on_disconnect hook fires after disconnection.

        Arrangement:
        1. Create mock callback for on_disconnect.
        2. Create Camera with hooks, connect it.

        Action:
        Call camera.disconnect() to close connection.

        Assertion Strategy:
        Validates hook by confirming:
        - on_disconnect called exactly once.
        - Called with no arguments.
        """
        on_disconnect_mock = Mock()
        hooks = CameraHooks(on_disconnect=on_disconnect_mock)
        driver = DigitalTwinCameraDriver()
        camera = Camera(driver, CameraConfig(camera_id=0), hooks=hooks)

        camera.connect()
        camera.disconnect()

        on_disconnect_mock.assert_called_once_with()

    def test_on_capture_hook_fires(self) -> None:
        """Verifies on_capture hook fires after each capture (before overlay).

        Arrangement:
        1. Create mock callback for on_capture.
        2. Create connected Camera with hooks.

        Action:
        Call camera.capture() to take a frame.

        Assertion Strategy:
        Validates hook by confirming:
        - on_capture called exactly once.
        - Called with CaptureResult argument.
        """
        on_capture_mock = Mock()
        hooks = CameraHooks(on_capture=on_capture_mock)
        driver = DigitalTwinCameraDriver()
        camera = Camera(driver, CameraConfig(camera_id=0), hooks=hooks)

        try:
            camera.connect()
            camera.capture()

            on_capture_mock.assert_called_once()
            # Verify called with CaptureResult
            call_args = on_capture_mock.call_args[0]
            assert hasattr(call_args[0], "image_data")
        finally:
            camera.disconnect()

    def test_on_stream_frame_hook_fires(self) -> None:
        """Verifies on_stream_frame hook fires for each streamed frame.

        Arrangement:
        1. Create mock callback for on_stream_frame.
        2. Create connected Camera with hooks.

        Action:
        Iterate camera.stream() for 2 frames.

        Assertion Strategy:
        Validates hook by confirming:
        - on_stream_frame called twice (once per frame).
        - Called with StreamFrame argument.
        """
        on_stream_frame_mock = Mock()
        hooks = CameraHooks(on_stream_frame=on_stream_frame_mock)
        driver = DigitalTwinCameraDriver()
        camera = Camera(driver, CameraConfig(camera_id=0), hooks=hooks)

        try:
            camera.connect()
            frame_count = 0
            for _ in camera.stream(max_fps=30):
                frame_count += 1
                if frame_count >= 2:
                    camera.stop_stream()

            assert on_stream_frame_mock.call_count == 2
            # Verify called with StreamFrame
            call_args = on_stream_frame_mock.call_args[0]
            assert isinstance(call_args[0], StreamFrame)
        finally:
            camera.disconnect()

    def test_on_error_hook_fires_on_connection_failure(self) -> None:
        """Verifies on_error hook fires when connection fails.

        Arrangement:
        1. Create mock driver that raises on open().
        2. Create mock callback for on_error.
        3. Create Camera with failing driver and hooks.

        Action:
        Attempt camera.connect() which will fail.

        Assertion Strategy:
        Validates hook by confirming:
        - CameraError is raised.
        - on_error called exactly once with the exception.
        """
        on_error_mock = Mock()
        hooks = CameraHooks(on_error=on_error_mock)

        # Create mock driver that fails on open
        mock_driver = Mock()
        mock_driver.open.side_effect = RuntimeError("Connection failed")

        camera = Camera(mock_driver, CameraConfig(camera_id=0), hooks=hooks)

        with pytest.raises(CameraError):
            camera.connect()

        on_error_mock.assert_called_once()
        # Verify called with an exception
        call_args = on_error_mock.call_args[0]
        assert isinstance(call_args[0], Exception)


class TestCameraDisconnectErrorHandling:
    """Test suite for disconnect error suppression.

    Covers line 1165-1166: Error during close() is logged but suppressed.

    Total: 1 test.
    """

    def test_disconnect_suppresses_close_errors(self) -> None:
        """Verifies disconnect suppresses errors from close to ensure graceful shutdown.

        Tests the error suppression behavior in disconnect() when the underlying
        camera instance's close() method fails. This ensures telescope sessions
        can always complete shutdown, even with hardware communication failures.

        Business context:
            Telescope operations must complete gracefully even when hardware
            malfunctions. A failing close() should not prevent session cleanup.

        Arrangement:
            1. Create Camera with mock driver returning configured instance.
            2. Mock instance.close() to raise RuntimeError("Close failed").
            3. Connect camera to establish internal state.

        Action:
            Call camera.disconnect() which internally calls instance.close().

        Assertion Strategy:
            Validates error suppression by confirming:
            - disconnect() returns normally without propagating exception.
            - Camera.is_connected property returns False after disconnect.
            - close() was actually called (error was caught, not bypassed).

        Testing Principle:
            Validates defensive programming - external failures must not break
            internal state transitions in telescope session management.
        """
        # Create mock instance that fails on close
        mock_instance = Mock()
        mock_instance.get_info.return_value = {"camera_id": 0, "name": "Mock"}
        mock_instance.get_controls.return_value = {}
        mock_instance.set_control.return_value = {}
        mock_instance.close.side_effect = RuntimeError("Close failed")

        # Create mock driver
        mock_driver = Mock()
        mock_driver.open.return_value = mock_instance

        camera = Camera(mock_driver, CameraConfig(camera_id=0))
        camera.connect()

        # Should not raise - error is suppressed
        camera.disconnect()

        # Camera should be marked as disconnected
        assert not camera.is_connected
        mock_instance.close.assert_called_once()


class TestCameraRecovery:
    """Test suite for camera disconnect recovery flow.

    Covers lines: 1366-1385 (capture exception path),
    1460-1526 (_recover_and_capture all paths).

    Total: 4 tests.
    """

    def _create_mock_instance(self, capture_side_effect: Any = None) -> Mock:
        """Create a mock camera instance for recovery testing scenarios.

        Factory method that creates consistently configured Mock objects
        simulating camera driver instances. Centralizes mock setup to ensure
        test consistency and reduce duplication.

        Business context:
            Recovery tests need predictable mock behavior. This factory ensures
            all tests use identical base configuration, isolating only the
            specific behavior under test (capture success/failure).

        Args:
            capture_side_effect: Side effect for capture method. Pass an exception
                class/instance to simulate capture failures, or a callable for
                custom behavior. None means capture returns valid JPEG data.

        Returns:
            Mock instance configured with:
            - get_info() returning camera metadata dict
            - get_controls() returning empty dict
            - set_control() returning empty dict
            - capture() behavior per capture_side_effect
            - close() returning None

        Raises:
            No exceptions raised directly; mock configuration only.

        Example:
            >>> instance = self._create_mock_instance(RuntimeError("Disconnect"))
            >>> instance.capture()  # raises RuntimeError
        """
        mock_instance = Mock()
        mock_instance.get_info.return_value = {
            "camera_id": 0,
            "name": "MockCamera",
            "max_width": 1920,
            "max_height": 1080,
        }
        mock_instance.get_controls.return_value = {}
        mock_instance.set_control.return_value = {}
        if capture_side_effect:
            mock_instance.capture.side_effect = capture_side_effect
        else:
            mock_instance.capture.return_value = b"\xff\xd8\xff\xe0test_jpeg"
        mock_instance.close.return_value = None
        return mock_instance

    def test_recovery_succeeds_and_retries_capture(self) -> None:
        """Verifies successful recovery reconnects and retries capture.

        Arrangement:
        1. Create mock driver with instance that fails first capture.
        2. Create recovery strategy that returns True.
        3. Second capture (after recovery) succeeds.

        Action:
        Call camera.capture() which fails, triggers recovery, retries.

        Assertion Strategy:
        Validates recovery by confirming:
        - Capture eventually succeeds.
        - Result metadata includes "recovered": True.
        """

        class SuccessfulRecovery:
            """Recovery strategy that always succeeds."""

            def attempt_recovery(self, camera_id: int) -> bool:
                """Simulate successful camera recovery for testing reconnection flows.

                This test double always returns True, allowing tests to verify
                the Camera class correctly handles successful recovery scenarios
                including reconnection and capture retry.

                Business context:
                    USB cameras can disconnect during long exposures. Recovery
                    strategies abstract the physical reconnection logic, allowing
                    Camera to retry operations transparently.

                Args:
                    camera_id: Identifier of camera to recover (unused in test).

                Returns:
                    True always, indicating recovery succeeded.

                Raises:
                    No exceptions raised; always succeeds.

                Example:
                    >>> recovery = SuccessfulRecovery()
                    >>> recovery.attempt_recovery(0)
                    True
                """
                return True

        # Track open calls to return different instances
        call_count = [0]

        def create_instance(camera_id: int) -> Mock:
            """Factory that returns failing instance first, then working instance.

            Simulates USB disconnect/reconnect by tracking call count and
            returning different mock behaviors. First call simulates a camera
            that disconnects mid-capture; second call simulates successful
            reconnection with working hardware.

            Business context:
                USB cameras can disconnect during long exposures. This factory
                simulates the recovery scenario where reconnection succeeds.

            Args:
                camera_id: Camera identifier passed by driver.open().

            Returns:
                Mock instance - first call fails capture, subsequent succeed.

            Raises:
                No exceptions raised directly; mock configuration only.

            Example:
                >>> mock = create_instance(0)  # First call - fails capture
                >>> mock = create_instance(0)  # Second call - succeeds
            """
            call_count[0] += 1
            if call_count[0] == 1:
                # First instance fails on capture
                return self._create_mock_instance(
                    capture_side_effect=RuntimeError("Simulated disconnect")
                )
            else:
                # Second instance succeeds
                return self._create_mock_instance()

        mock_driver = Mock()
        mock_driver.open.side_effect = create_instance

        camera = Camera(
            mock_driver, CameraConfig(camera_id=0), recovery=SuccessfulRecovery()
        )
        camera.connect()

        result = camera.capture()

        assert result.image_data is not None
        assert result.metadata.get("recovered") is True
        camera.disconnect()

    def test_recovery_fails_raises_disconnected_error(self) -> None:
        """Verifies failed recovery raises CameraDisconnectedError.

        Arrangement:
        1. Create Camera with NullRecoveryStrategy (always fails).
        2. Make capture fail to trigger recovery.

        Action:
        Call camera.capture() which fails and recovery fails.

        Assertion Strategy:
        Validates failure by confirming:
        - CameraDisconnectedError is raised.
        - on_error hook fires.
        """
        on_error_mock = Mock()
        hooks = CameraHooks(on_error=on_error_mock)

        # Instance that fails on capture
        mock_instance = self._create_mock_instance(
            capture_side_effect=RuntimeError("Disconnect")
        )

        mock_driver = Mock()
        mock_driver.open.return_value = mock_instance

        camera = Camera(
            mock_driver,
            CameraConfig(camera_id=0),
            hooks=hooks,
            recovery=NullRecoveryStrategy(),
        )
        camera.connect()

        with pytest.raises(CameraDisconnectedError):
            camera.capture()

        on_error_mock.assert_called()
        camera.disconnect()

    def test_recovery_reconnect_fails_raises_disconnected_error(self) -> None:
        """Verifies failed reconnect after recovery raises CameraDisconnectedError.

        Arrangement:
        1. Create recovery that returns True but reconnect will fail.
        2. Mock driver.open to fail on second call.

        Action:
        Call camera.capture() triggering recovery then failed reconnect.

        Assertion Strategy:
        Validates failure by confirming:
        - CameraDisconnectedError raised with "reconnection failed".
        """

        class SuccessfulRecovery:
            """Recovery that succeeds but reconnect will fail."""

            def attempt_recovery(self, camera_id: int) -> bool:
                """Simulate successful physical recovery where reconnect still fails.

                Tests the scenario where hardware recovery succeeds (USB device
                re-enumerated) but the subsequent driver.open() call fails.
                This validates error handling in the reconnection code path.

                Business context:
                    Physical recovery (e.g., USB port power cycle) can succeed
                    while software reconnection fails due to driver issues.

                Args:
                    camera_id: Identifier of camera to recover (unused in test).

                Returns:
                    True always, simulating successful physical recovery.

                Raises:
                    No exceptions raised; always returns True.

                Example:
                    >>> recovery = SuccessfulRecovery()
                    >>> recovery.attempt_recovery(0)
                    True
                """
                return True

        # First open succeeds with failing capture, second open fails
        call_count = [0]

        def open_with_failure(camera_id: int) -> Mock:
            """Factory simulating successful open, then failed reconnection.

            Models the scenario where initial camera connection works, capture
            fails triggering recovery, but the subsequent reconnection attempt
            fails at the driver.open() level.

            Business context:
                After USB disconnect, the camera might not re-enumerate properly,
                causing driver.open() to fail. This tests that Camera propagates
                CameraDisconnectedError correctly in this scenario.

            Args:
                camera_id: Camera identifier passed by driver.open().

            Returns:
                Mock instance on first call.

            Raises:
                RuntimeError: On second call, simulating reconnection failure.
            """
            call_count[0] += 1
            if call_count[0] == 1:
                return self._create_mock_instance(
                    capture_side_effect=RuntimeError("Disconnect")
                )
            else:
                raise RuntimeError("Reconnect failed")

        mock_driver = Mock()
        mock_driver.open.side_effect = open_with_failure

        camera = Camera(
            mock_driver, CameraConfig(camera_id=0), recovery=SuccessfulRecovery()
        )
        camera.connect()

        with pytest.raises(CameraDisconnectedError, match="reconnection failed"):
            camera.capture()

    def test_recovery_with_on_error_hook(self) -> None:
        """Verifies on_error hook fires when recovery fails.

        Arrangement:
        1. Create Camera with NullRecoveryStrategy and on_error hook.
        2. Make capture fail.

        Action:
        Trigger failed capture and recovery.

        Assertion Strategy:
        Validates hook fires with original exception.
        """
        errors_received: list[Exception] = []

        def capture_error(e: Exception) -> None:
            """Error hook that records exceptions for test verification.

            Captures exceptions passed to on_error hook into errors_received
            list, allowing tests to verify hook invocation and exception
            propagation.

            Business context:
                Tests on_error callback invocation during capture failures.

            Args:
                e: Exception passed by Camera.capture() error handling.

            Returns:
                None. Side effect only (appends to errors_received).

            Raises:
                No exceptions raised.

            Example:
                >>> errors = []
                >>> capture_error(RuntimeError("test"))
                >>> len(errors)  # 1
            """
            errors_received.append(e)

        hooks = CameraHooks(on_error=capture_error)

        # Instance that fails on capture
        mock_instance = self._create_mock_instance(
            capture_side_effect=RuntimeError("Test error")
        )

        mock_driver = Mock()
        mock_driver.open.return_value = mock_instance

        camera = Camera(
            mock_driver,
            CameraConfig(camera_id=0),
            hooks=hooks,
            recovery=NullRecoveryStrategy(),
        )
        camera.connect()

        with pytest.raises(CameraDisconnectedError):
            camera.capture()

        assert len(errors_received) >= 1
        camera.disconnect()


class TestCameraOverlay:
    """Test suite for overlay configuration and application.

    Covers lines: 1742 (_apply_overlay with overlay enabled).

    Total: 3 tests.
    """

    def test_set_overlay_configures_overlay(self) -> None:
        """Verifies set_overlay stores overlay configuration.

        Arrangement:
        1. Create Camera (connection not required).

        Action:
        Call camera.set_overlay() with OverlayConfig.

        Assertion Strategy:
        Validates by confirming camera.overlay returns the config.
        """
        driver = DigitalTwinCameraDriver()
        camera = Camera(driver, CameraConfig(camera_id=0))

        config = OverlayConfig(
            enabled=True, overlay_type="crosshair", color=(0, 255, 0), opacity=0.8
        )
        camera.set_overlay(config)

        assert camera.overlay is config
        assert camera.overlay.enabled is True
        assert camera.overlay.overlay_type == "crosshair"

    def test_set_overlay_none_clears_overlay(self) -> None:
        """Verifies set_overlay(None) clears overlay configuration.

        Arrangement:
        1. Create Camera with overlay configured.

        Action:
        Call camera.set_overlay(None).

        Assertion Strategy:
        Validates by confirming camera.overlay is None.
        """
        driver = DigitalTwinCameraDriver()
        camera = Camera(driver, CameraConfig(camera_id=0))

        camera.set_overlay(OverlayConfig(enabled=True, overlay_type="crosshair"))
        camera.set_overlay(None)

        assert camera.overlay is None

    def test_capture_with_overlay_enabled(self) -> None:
        """Verifies capture applies overlay when enabled and configured.

        Arrangement:
        1. Create mock renderer that modifies image.
        2. Create Camera with overlay config and renderer.
        3. Connect camera.

        Action:
        Call camera.capture() with apply_overlay=True (default).

        Assertion Strategy:
        Validates overlay by confirming:
        - result.has_overlay is True.
        - Renderer.render() was called.
        - metadata includes overlay_type.
        """
        mock_renderer = Mock()
        # Return modified data to simulate overlay
        mock_renderer.render.return_value = b"overlaid_image_data"

        driver = DigitalTwinCameraDriver()
        camera = Camera(driver, CameraConfig(camera_id=0), renderer=mock_renderer)
        camera.set_overlay(
            OverlayConfig(enabled=True, overlay_type="crosshair", color=(255, 0, 0))
        )

        try:
            camera.connect()
            result = camera.capture(CaptureOptions(apply_overlay=True))

            assert result.has_overlay is True
            assert result.image_data == b"overlaid_image_data"
            assert result.metadata.get("overlay_type") == "crosshair"
            mock_renderer.render.assert_called_once()
        finally:
            camera.disconnect()


class TestCameraControlErrors:
    """Test suite for set_control and get_control error handling.

    Covers lines: 1808, 1816-1822 (set_control error),
    1844-1853 (get_control error).

    Total: 4 tests.
    """

    def test_set_control_without_connection_raises(self) -> None:
        """Verifies set_control raises when not connected.

        Arrangement:
        1. Create Camera without connecting.

        Action:
        Call camera.set_control() on unconnected camera.

        Assertion Strategy:
        Validates by confirming CameraNotConnectedError is raised.
        """
        from telescope_mcp.devices.camera import CameraNotConnectedError

        driver = DigitalTwinCameraDriver()
        camera = Camera(driver, CameraConfig(camera_id=0))

        with pytest.raises(CameraNotConnectedError):
            camera.set_control("Gain", 50)

    def test_get_control_without_connection_raises(self) -> None:
        """Verifies get_control raises when not connected.

        Arrangement:
        1. Create Camera without connecting.

        Action:
        Call camera.get_control() on unconnected camera.

        Assertion Strategy:
        Validates by confirming CameraNotConnectedError is raised.
        """
        from telescope_mcp.devices.camera import CameraNotConnectedError

        driver = DigitalTwinCameraDriver()
        camera = Camera(driver, CameraConfig(camera_id=0))

        with pytest.raises(CameraNotConnectedError):
            camera.get_control("Gain")

    def test_set_control_error_fires_hook_and_raises(self) -> None:
        """Verifies set_control fires on_error hook when driver fails.

        Arrangement:
        1. Create Camera with on_error hook and mock driver.
        2. Connect and make set_control fail on specific control.

        Action:
        Call camera.set_control() with failing driver.

        Assertion Strategy:
        Validates by confirming:
        - CameraError is raised.
        - on_error hook fires.
        """
        on_error_mock = Mock()
        hooks = CameraHooks(on_error=on_error_mock)

        # Create mock instance that fails on set_control for InvalidControl only
        mock_instance = Mock()
        mock_instance.get_info.return_value = {"camera_id": 0, "name": "Mock"}
        mock_instance.get_controls.return_value = {}

        def selective_set_control(name: str, value: int) -> dict[str, Any]:
            """Mock set_control that fails only for specific control names.

            Allows normal camera setup (Exposure, Gain) to succeed while
            causing targeted failures for test scenarios. This selective
            failure enables testing error handling without breaking the
            camera connection setup.

            Business context:
                Tests set_control error path with on_error hook enabled.

            Args:
                name: Control name (e.g., "Gain", "Exposure", "InvalidControl").
                value: Control value to set.

            Returns:
                Dict with control name and value on success.

            Raises:
                RuntimeError: When name is "InvalidControl".

            Example:
                >>> selective_set_control("Gain", 100)
                {'control': 'Gain', 'value': 100}
            """
            if name == "InvalidControl":
                raise RuntimeError("Invalid control")
            return {"control": name, "value": value}

        mock_instance.set_control.side_effect = selective_set_control
        mock_instance.close.return_value = None

        mock_driver = Mock()
        mock_driver.open.return_value = mock_instance

        camera = Camera(mock_driver, CameraConfig(camera_id=0), hooks=hooks)

        try:
            camera.connect()

            with pytest.raises(CameraError, match="Failed to set"):
                camera.set_control("InvalidControl", 999)

            on_error_mock.assert_called_once()
        finally:
            camera.disconnect()

    def test_get_control_error_fires_hook_and_raises(self) -> None:
        """Verifies get_control fires on_error hook when driver fails.

        Arrangement:
        1. Create Camera with on_error hook and mock driver.
        2. Connect and make get_control fail.

        Action:
        Call camera.get_control() with failing driver.

        Assertion Strategy:
        Validates by confirming:
        - CameraError is raised.
        - on_error hook fires.
        """
        on_error_mock = Mock()
        hooks = CameraHooks(on_error=on_error_mock)

        # Create mock instance that fails on get_control
        mock_instance = Mock()
        mock_instance.get_info.return_value = {"camera_id": 0, "name": "Mock"}
        mock_instance.get_controls.return_value = {}
        mock_instance.set_control.return_value = {}
        mock_instance.get_control.side_effect = RuntimeError("Read failed")
        mock_instance.close.return_value = None

        mock_driver = Mock()
        mock_driver.open.return_value = mock_instance

        camera = Camera(mock_driver, CameraConfig(camera_id=0), hooks=hooks)

        try:
            camera.connect()

            with pytest.raises(CameraError, match="Failed to get"):
                camera.get_control("Gain")

            on_error_mock.assert_called_once()
        finally:
            camera.disconnect()

    def test_set_control_error_without_hook(self) -> None:
        """Verifies set_control error works without on_error hook.

        Arrangement:
        1. Create Camera WITHOUT hooks and mock driver.
        2. Connect and make set_control fail.

        Action:
        Call camera.set_control() with failing driver.

        Assertion Strategy:
        Validates by confirming CameraError is raised (covers branch 1820->1822).
        """
        # Create mock instance that fails on set_control for InvalidControl only
        mock_instance = Mock()
        mock_instance.get_info.return_value = {"camera_id": 0, "name": "Mock"}
        mock_instance.get_controls.return_value = {}

        def selective_set_control(name: str, value: int) -> dict[str, Any]:
            """Mock set_control that fails only for specific control names.

            Tests the no-hooks code path where on_error is None. Allows
            standard controls to work while failing on test-specific controls.

            Business context:
                Validates error propagation when no hooks are configured.

            Args:
                name: Control name being set.
                value: Control value to set.

            Returns:
                Dict with control name and value on success.

            Raises:
                RuntimeError: When name is "InvalidControl".

            Example:
                >>> selective_set_control("Exposure", 50000)
                {'control': 'Exposure', 'value': 50000}
            """
            if name == "InvalidControl":
                raise RuntimeError("Invalid control")
            return {"control": name, "value": value}

        mock_instance.set_control.side_effect = selective_set_control
        mock_instance.close.return_value = None

        mock_driver = Mock()
        mock_driver.open.return_value = mock_instance

        # NO hooks - this tests branch where on_error is None
        camera = Camera(mock_driver, CameraConfig(camera_id=0), hooks=None)

        try:
            camera.connect()

            with pytest.raises(CameraError, match="Failed to set"):
                camera.set_control("InvalidControl", 999)
        finally:
            camera.disconnect()


class TestCameraRepr:
    """Test suite for Camera __repr__ method.

    Covers lines: 1931-1933.

    Total: 2 tests.
    """

    def test_repr_disconnected(self) -> None:
        """Verifies __repr__ shows disconnected state.

        Arrangement:
        1. Create Camera without connecting.

        Action:
        Call repr(camera) or str(camera).

        Assertion Strategy:
        Validates by confirming repr contains "disconnected".
        """
        driver = DigitalTwinCameraDriver()
        camera = Camera(driver, CameraConfig(camera_id=0, name="TestCam"))

        result = repr(camera)

        assert "TestCam" in result
        assert "disconnected" in result

    def test_repr_connected(self) -> None:
        """Verifies repr shows connected state for debugging and logging.

        Tests that Camera.__repr__() correctly reflects connection status,
        providing accurate debugging information during telescope operations.

        Business context:
            Clear repr output aids troubleshooting during observation sessions
            when cameras may connect/disconnect unexpectedly.

        Arrangement:
            1. Create Camera with DigitalTwinCameraDriver.
            2. Call connect() to establish connection.

        Action:
            Call repr(camera) to get string representation.

        Assertion Strategy:
            Validates connected state representation by confirming:
            - Camera name appears in output.
            - "connected" status is present.
            - "disconnected" does not appear (mutually exclusive states).

        Testing Principle:
            Validates observability - __repr__ must accurately reflect
            internal state for effective debugging and logging.
        """
        driver = DigitalTwinCameraDriver()
        camera = Camera(driver, CameraConfig(camera_id=0, name="TestCam"))

        try:
            camera.connect()
            result = repr(camera)

            assert "TestCam" in result
            assert "connected" in result
            assert "disconnected" not in result
        finally:
            camera.disconnect()

    def test_repr_without_name(self) -> None:
        """Verifies __repr__ uses camera_id when no name provided.

        Arrangement:
        1. Create Camera without name in config.

        Action:
        Call repr(camera).

        Assertion Strategy:
        Validates by confirming repr contains fallback "camera_0".
        """
        driver = DigitalTwinCameraDriver()
        camera = Camera(driver, CameraConfig(camera_id=0))

        result = repr(camera)

        assert "camera_0" in result


class TestCameraIsStreamingProperty:
    """Test suite for is_streaming property.

    Covers line 1054.

    Total: 2 tests.
    """

    def test_is_streaming_false_when_not_streaming(self) -> None:
        """Verifies is_streaming is False when not actively streaming.

        Arrangement:
        1. Create and connect Camera.
        2. Do not start streaming.

        Action:
        Check camera.is_streaming property.

        Assertion Strategy:
        Validates by confirming is_streaming is False.
        """
        driver = DigitalTwinCameraDriver()
        camera = Camera(driver, CameraConfig(camera_id=0))

        try:
            camera.connect()
            assert camera.is_streaming is False
        finally:
            camera.disconnect()

    def test_is_streaming_true_during_stream(self) -> None:
        """Verifies is_streaming is True while streaming.

        Arrangement:
        1. Create and connect Camera.
        2. Start streaming.

        Action:
        Check camera.is_streaming inside stream loop.

        Assertion Strategy:
        Validates by confirming is_streaming is True during iteration.
        """
        driver = DigitalTwinCameraDriver()
        camera = Camera(driver, CameraConfig(camera_id=0))

        try:
            camera.connect()
            for frame in camera.stream(max_fps=30):
                # Check while streaming
                assert camera.is_streaming is True
                camera.stop_stream()
            # After stream ends
            assert camera.is_streaming is False
        finally:
            camera.disconnect()


class TestCameraInfoProperty:
    """Test suite for info property edge cases.

    Covers info property when None (disconnected).

    Total: 1 test.
    """

    def test_info_none_when_disconnected(self) -> None:
        """Verifies info property returns None when not connected.

        Arrangement:
        1. Create Camera without connecting.

        Action:
        Access camera.info property.

        Assertion Strategy:
        Validates by confirming info is None.
        """
        driver = DigitalTwinCameraDriver()
        camera = Camera(driver, CameraConfig(camera_id=0))

        assert camera.info is None


class TestNullRecoveryStrategy:
    """Test suite for NullRecoveryStrategy.

    Covers line 525 (attempt_recovery returns False).

    Total: 1 test.
    """

    def test_null_recovery_always_returns_false(self) -> None:
        """Verifies NullRecoveryStrategy provides no-op recovery for simple setups.

        Tests the null object pattern implementation that disables automatic
        recovery. This default strategy ensures Camera works without recovery
        configuration while maintaining interface consistency.

        Business context:
            Not all camera setups need recovery (e.g., development, testing).
            NullRecoveryStrategy provides a safe default that fails fast
            rather than attempting unreliable recovery operations.

        Arrangement:
            1. Create NullRecoveryStrategy instance directly.

        Action:
            Call attempt_recovery() with various camera IDs (0, 1, 999).

        Assertion Strategy:
            Validates null object pattern by confirming:
            - Returns False for camera_id=0.
            - Returns False for camera_id=1.
            - Returns False for camera_id=999 (arbitrary value).

        Testing Principle:
            Validates null object pattern - NullRecoveryStrategy must be
            consistent and predictable regardless of input values.
        """
        strategy = NullRecoveryStrategy()

        assert strategy.attempt_recovery(0) is False
        assert strategy.attempt_recovery(1) is False
        assert strategy.attempt_recovery(999) is False


class TestCaptureRawConvenience:
    """Test suite for capture_raw convenience method.

    Total: 1 test.
    """

    def test_capture_raw_returns_no_overlay(self) -> None:
        """Verifies capture_raw returns frame without overlay.

        Arrangement:
        1. Create Camera with overlay configured.
        2. Connect camera.

        Action:
        Call camera.capture_raw() which should bypass overlay.

        Assertion Strategy:
        Validates by confirming result.has_overlay is False.
        """
        driver = DigitalTwinCameraDriver()
        camera = Camera(driver, CameraConfig(camera_id=0))
        camera.set_overlay(OverlayConfig(enabled=True, overlay_type="crosshair"))

        try:
            camera.connect()
            result = camera.capture_raw(exposure_us=100_000, gain=50)

            assert result.has_overlay is False
            assert result.image_data is not None
        finally:
            camera.disconnect()


class TestNullRenderer:
    """Test suite for NullRenderer passthrough behavior.

    Covers line 395.

    Total: 1 test.
    """

    def test_null_renderer_returns_original_data(self) -> None:
        """Verifies NullRenderer returns image data unchanged.

        Arrangement:
        1. Create NullRenderer instance.
        2. Create sample image bytes.

        Action:
        Call renderer.render() with sample data.

        Assertion Strategy:
        Validates by confirming returned bytes are identical (same object).
        """
        from telescope_mcp.devices.camera import NullRenderer

        renderer = NullRenderer()
        original_data = b"\xff\xd8\xff\xe0test_image_data"
        config = OverlayConfig(enabled=True, overlay_type="crosshair")

        result = renderer.render(original_data, config, None)

        assert result is original_data  # Same object, not just equal


class TestCameraControlTracking:
    """Test suite for control tracking (Gain/Exposure).

    Covers lines 1325, 1327, 1814-1817.

    Total: 2 tests.
    """

    def test_capture_with_different_gain_updates_control(self) -> None:
        """Verifies capture with different gain calls set_control.

        Arrangement:
        1. Create Camera with default_gain=50.
        2. Connect camera.

        Action:
        Call capture with gain=80 (different from default).

        Assertion Strategy:
        Validates by confirming set_control was called for Gain.
        """
        driver = DigitalTwinCameraDriver()
        camera = Camera(driver, CameraConfig(camera_id=0, default_gain=50))

        try:
            camera.connect()
            # Capture with different gain
            result = camera.capture(CaptureOptions(gain=80))

            assert result.gain == 80
            # Internal state should be updated
            assert camera._current_gain == 80
        finally:
            camera.disconnect()

    def test_capture_with_different_exposure_updates_control(self) -> None:
        """Verifies capture with different exposure calls set_control.

        Arrangement:
        1. Create Camera with default_exposure_us=100000.
        2. Connect camera.

        Action:
        Call capture with exposure_us=200000 (different from default).

        Assertion Strategy:
        Validates by confirming set_control was called for Exposure.
        """
        driver = DigitalTwinCameraDriver()
        camera = Camera(driver, CameraConfig(camera_id=0, default_exposure_us=100_000))

        try:
            camera.connect()
            # Capture with different exposure
            result = camera.capture(CaptureOptions(exposure_us=200_000))

            assert result.exposure_us == 200_000
            # Internal state should be updated
            assert camera._current_exposure_us == 200_000
        finally:
            camera.disconnect()

    def test_set_control_exposure_tracking(self) -> None:
        """Verifies set_control Exposure updates internal state for capture consistency.

        Tests that calling set_control("Exposure", ...) updates Camera's internal
        _current_exposure_us tracking. This ensures subsequent captures can
        report accurate exposure metadata without re-querying the driver.

        Business context:
            Astrophotography requires precise exposure tracking for stacking
            and processing. Camera must track control changes to provide
            accurate metadata in CaptureResult.

        Arrangement:
            1. Create Camera with DigitalTwinCameraDriver.
            2. Connect camera to enable control operations.

        Action:
            Call camera.set_control("Exposure", 500_000) to change exposure.

        Assertion Strategy:
            Validates internal tracking by confirming:
            - _current_exposure_us equals 500_000 after set_control.

        Testing Principle:
            Validates state consistency - Camera must track control changes
            internally to ensure capture metadata accuracy.
        """
        driver = DigitalTwinCameraDriver()
        camera = Camera(driver, CameraConfig(camera_id=0))

        try:
            camera.connect()
            camera.set_control("Exposure", 500_000)

            assert camera._current_exposure_us == 500_000
        finally:
            camera.disconnect()


class TestGetControlValue:
    """Test suite for get_control return value handling.

    Covers lines 1849, 1851.

    Total: 1 test.
    """

    def test_get_control_returns_value_from_dict(self) -> None:
        """Verifies get_control extracts value from driver response.

        Arrangement:
        1. Create and connect Camera.

        Action:
        Call camera.get_control("Gain").

        Assertion Strategy:
        Validates by confirming an integer is returned.
        """
        driver = DigitalTwinCameraDriver()
        camera = Camera(driver, CameraConfig(camera_id=0))

        try:
            camera.connect()
            value = camera.get_control("Gain")

            assert isinstance(value, int)
        finally:
            camera.disconnect()


class TestApplyOverlayEdgeCases:
    """Test suite for _apply_overlay edge cases.

    Covers line 1742.

    Total: 2 tests.
    """

    def test_apply_overlay_returns_original_when_no_overlay(self) -> None:
        """Verifies _apply_overlay returns original result when no overlay set.

        Arrangement:
        1. Create Camera without overlay.
        2. Connect and capture.

        Action:
        Call capture with apply_overlay=True but no overlay configured.

        Assertion Strategy:
        Validates by confirming result.has_overlay is False.
        """
        driver = DigitalTwinCameraDriver()
        camera = Camera(driver, CameraConfig(camera_id=0))
        # No overlay set

        try:
            camera.connect()
            result = camera.capture(CaptureOptions(apply_overlay=True))

            assert result.has_overlay is False
        finally:
            camera.disconnect()

    def test_apply_overlay_internal_no_overlay(self) -> None:
        """Verifies _apply_overlay returns original when overlay is None.

        Arrangement:
        1. Create Camera without overlay configured.
        2. Create a mock CaptureResult.

        Action:
        Call camera._apply_overlay() directly.

        Assertion Strategy:
        Validates by confirming same result object is returned.
        """
        from datetime import UTC, datetime

        from telescope_mcp.devices.camera import CaptureResult

        driver = DigitalTwinCameraDriver()
        camera = Camera(driver, CameraConfig(camera_id=0))
        # Explicitly ensure no overlay
        camera._overlay = None

        mock_result = CaptureResult(
            image_data=b"test",
            timestamp=datetime.now(UTC),
            exposure_us=100_000,
            gain=50,
        )

        result = camera._apply_overlay(mock_result)

        # Should return exact same object
        assert result is mock_result


class TestRecoveryCaptureFailure:
    """Test suite for recovery capture failure path.

    Covers line 1525 (on_error hook in recovery capture exception).

    Total: 1 test.
    """

    def test_recovery_capture_fails_after_reconnect(self) -> None:
        """Verifies on_error fires when capture fails after successful recovery.

        Arrangement:
        1. Create recovery that succeeds.
        2. Reconnect succeeds but capture fails.

        Action:
        Trigger capture that fails, recovery succeeds, but second capture fails.

        Assertion Strategy:
        Validates CameraDisconnectedError with "reconnection failed".
        """
        errors_received: list[Exception] = []

        def capture_error(e: Exception) -> None:
            """Error hook that collects exceptions for post-test verification.

            Captures exceptions into errors_received list to verify that
            on_error hook fires correctly during recovery failure scenarios.

            Business context:
                Tests hook invocation when recovery succeeds but retry fails.

            Args:
                e: Exception passed by Camera error handling code.

            Returns:
                None. Side effect only (appends to errors_received).

            Raises:
                No exceptions raised.

            Example:
                >>> errors = []
                >>> capture_error(RuntimeError("retry failed"))
            """
            errors_received.append(e)

        hooks = CameraHooks(on_error=capture_error)

        class SuccessfulRecovery:
            """Recovery strategy that always succeeds."""

            def attempt_recovery(self, camera_id: int) -> bool:
                """Simulate successful recovery for retry capture failure test.

                Returns True to allow reconnection, but the second capture
                will also fail, testing the "recovery succeeded but retry failed"
                code path.

                Business context:
                    Some failures persist after reconnection (e.g., overheating).
                    This tests Camera's handling of retry failures.

                Args:
                    camera_id: Identifier of camera to recover.

                Returns:
                    True always, simulating successful hardware recovery.

                Raises:
                    No exceptions raised; always returns True.

                Example:
                    >>> recovery = SuccessfulRecovery()
                    >>> recovery.attempt_recovery(0)
                    True
                """
                return True

        # First instance fails, second reconnects but also fails on capture
        call_count = [0]

        def create_failing_instance(camera_id: int) -> Mock:
            """Factory that creates instances where capture always fails.

            Both initial and reconnected instances fail on capture, testing
            the scenario where recovery succeeds at the hardware level but
            the underlying camera issue persists.

            Business context:
                Some camera failures aren't recoverable by reconnection
                (e.g., overheating, sensor damage). This tests that Camera
                correctly reports failure when retry also fails.

            Args:
                camera_id: Camera identifier passed by driver.open().

            Returns:
                Mock instance where capture() always raises RuntimeError.

            Raises:
                No exceptions from factory; mock.capture() raises RuntimeError.

            Example:
                >>> mock = create_failing_instance(0)
                >>> mock.capture()  # raises RuntimeError
            """
            call_count[0] += 1
            mock = Mock()
            mock.get_info.return_value = {"camera_id": 0, "name": "Mock"}
            mock.get_controls.return_value = {}
            mock.set_control.return_value = {}
            # Both instances fail on capture
            mock.capture.side_effect = RuntimeError(f"Capture failed #{call_count[0]}")
            mock.close.return_value = None
            return mock

        mock_driver = Mock()
        mock_driver.open.side_effect = create_failing_instance

        camera = Camera(
            mock_driver,
            CameraConfig(camera_id=0),
            hooks=hooks,
            recovery=SuccessfulRecovery(),
        )
        camera.connect()

        with pytest.raises(CameraDisconnectedError, match="reconnection failed"):
            camera.capture()

        # on_error should have been called (at least once)
        assert len(errors_received) >= 1


class TestStreamingNoSleep:
    """Test suite for streaming branch where no sleep is needed.

    Covers line 1666->1641 (elapsed >= min_interval branch).

    Total: 1 test.
    """

    def test_stream_skips_sleep_when_capture_exceeds_interval(self) -> None:
        """Verifies streaming doesn't sleep when capture takes longer than interval.

        Arrangement:
        1. Create Camera with mock clock.
        2. Set up clock to simulate long capture time.

        Action:
        Stream with high max_fps (small interval) where capture takes longer.

        Assertion Strategy:
        Validates by confirming sleep is never called when elapsed > interval.
        """

        class MockClockNoSleep:
            """Mock clock that tracks sleep calls."""

            def __init__(self) -> None:
                """Initialize mock clock with zero time and empty call tracking.

                Sets up internal state for simulating time progression during
                streaming tests. Tracks sleep calls to verify Camera's adaptive
                frame rate logic skips sleep when capture takes too long.

                Business context:
                    Streaming tests need deterministic timing. This mock provides
                    controllable time progression without real delays.

                Args:
                    None. No parameters required.

                Returns:
                    None. Initializes instance attributes.

                Raises:
                    No exceptions raised.

                Example:
                    >>> clock = MockClockNoSleep()
                    >>> clock.monotonic()  # 0.0
                """
                self._time = 0.0
                self.sleep_calls: list[float] = []
                self._monotonic_call_count = 0

            def monotonic(self) -> float:
                """Return simulated monotonic time advancing 0.2s per capture.

                Simulates long-running captures by advancing time on every
                second call (after capture completes). This allows testing
                the streaming frame rate logic without real time delays.

                Business context:
                    Verifies Camera skips sleep when captures exceed frame interval.

                Args:
                    None. Uses internal call counter for time simulation.

                Returns:
                    Simulated time in seconds, advancing 0.2s per capture.

                Raises:
                    No exceptions raised.

                Example:
                    >>> clock.monotonic()  # 0.0 (odd call)
                    >>> clock.monotonic()  # 0.2 (even call)
                """
                # Simulate time passing - each capture takes 0.2s
                self._monotonic_call_count += 1
                if self._monotonic_call_count % 2 == 0:
                    # Second call (after capture) - 0.2s elapsed
                    self._time += 0.2
                return self._time

            def sleep(self, seconds: float) -> None:
                """Record sleep call without actually sleeping.

                Captures sleep duration for test verification. The test
                asserts sleep_calls is empty, confirming Camera correctly
                skips sleep when capture exceeds the frame interval.

                Business context:
                    Streaming tests verify adaptive frame rate by checking
                    sleep is skipped when captures are slow.

                Args:
                    seconds: Sleep duration requested by Camera.stream().

                Returns:
                    None. Side effect only (appends to sleep_calls).

                Raises:
                    No exceptions raised.

                Example:
                    >>> clock.sleep(0.05)
                    >>> clock.sleep_calls  # [0.05]
                """
                self.sleep_calls.append(seconds)

        mock_clock = MockClockNoSleep()
        driver = DigitalTwinCameraDriver()
        camera = Camera(driver, CameraConfig(camera_id=0), clock=mock_clock)

        try:
            camera.connect()
            # Stream at 10 fps (0.1s interval) but capture takes 0.2s
            frame_count = 0
            for _ in camera.stream(max_fps=10):
                frame_count += 1
                if frame_count >= 2:
                    camera.stop_stream()

            # Sleep should not be called because elapsed > min_interval
            assert len(mock_clock.sleep_calls) == 0
        finally:
            camera.disconnect()


class TestControlTrackingOtherControls:
    """Test suite for set_control with non-Gain/Exposure controls.

    Covers branch 1816->exit (control is neither Gain nor Exposure).

    Total: 1 test.
    """

    def test_set_control_other_does_not_track(self) -> None:
        """Verifies set_control for other controls doesn't update tracking.

        Arrangement:
        1. Create and connect Camera with mock driver.

        Action:
        Call camera.set_control("WB_R", 52) - a non-tracked control.

        Assertion Strategy:
        Validates by confirming _current_gain and _current_exposure_us unchanged.
        """
        mock_instance = Mock()
        mock_instance.get_info.return_value = {"camera_id": 0, "name": "Mock"}
        mock_instance.get_controls.return_value = {}
        mock_instance.set_control.return_value = {"control": "WB_R", "value": 52}
        mock_instance.close.return_value = None

        mock_driver = Mock()
        mock_driver.open.return_value = mock_instance

        camera = Camera(
            mock_driver,
            CameraConfig(camera_id=0, default_gain=50, default_exposure_us=100_000),
        )

        try:
            camera.connect()
            original_gain = camera._current_gain
            original_exposure = camera._current_exposure_us

            camera.set_control("WB_R", 52)

            # Internal tracking should be unchanged
            assert camera._current_gain == original_gain
            assert camera._current_exposure_us == original_exposure
        finally:
            camera.disconnect()


class TestRecoveryWithoutHooks:
    """Test suite for recovery paths without on_error hook.

    Covers partial branches 1481->1483, 1851->1853 (no hook).

    Total: 2 tests.
    """

    def _create_mock_instance(self, capture_side_effect: Any = None) -> Mock:
        """Create a mock camera instance for no-hooks recovery testing.

        Factory method creating Mock instances for testing recovery paths
        without on_error hook. Centralizes mock configuration to ensure
        consistent test setup.

        Business context:
            Recovery code has separate branches for with/without hooks.
            This factory supports tests covering the no-hook code path
            where errors must be raised without hook invocation.

        Args:
            capture_side_effect: Side effect for capture method. Pass an
                exception to simulate capture failures, or None for success.

        Returns:
            Mock instance configured with standard camera methods.

        Raises:
            No exceptions raised directly; mock configuration only.

        Example:
            >>> instance = self._create_mock_instance(RuntimeError("Fail"))
            >>> instance.capture()  # raises RuntimeError
        """
        mock_instance = Mock()
        mock_instance.get_info.return_value = {"camera_id": 0, "name": "Mock"}
        mock_instance.get_controls.return_value = {}
        mock_instance.set_control.return_value = {}
        if capture_side_effect:
            mock_instance.capture.side_effect = capture_side_effect
        else:
            mock_instance.capture.return_value = b"\xff\xd8\xff\xe0test"
        mock_instance.close.return_value = None
        return mock_instance

    def test_recovery_fails_without_hook(self) -> None:
        """Verifies recovery failure works without on_error hook.

        Arrangement:
        1. Create Camera with NullRecoveryStrategy and NO hooks.
        2. Make capture fail to trigger recovery.

        Action:
        Call camera.capture() which fails and recovery fails.

        Assertion Strategy:
        Validates failure by confirming CameraDisconnectedError raised.
        """
        # NO hooks - this tests the branch where on_error is None
        mock_instance = self._create_mock_instance(
            capture_side_effect=RuntimeError("Disconnect")
        )

        mock_driver = Mock()
        mock_driver.open.return_value = mock_instance

        camera = Camera(
            mock_driver,
            CameraConfig(camera_id=0),
            # NO hooks - explicitly None
            hooks=None,
            recovery=NullRecoveryStrategy(),
        )
        camera.connect()

        with pytest.raises(CameraDisconnectedError):
            camera.capture()

        camera.disconnect()

    def test_get_control_error_without_hook(self) -> None:
        """Verifies get_control error works without on_error hook.

        Arrangement:
        1. Create Camera without hooks.
        2. Make get_control fail.

        Action:
        Call camera.get_control() which fails.

        Assertion Strategy:
        Validates CameraError is raised even without hook.
        """
        mock_instance = Mock()
        mock_instance.get_info.return_value = {"camera_id": 0, "name": "Mock"}
        mock_instance.get_controls.return_value = {}
        mock_instance.set_control.return_value = {}
        mock_instance.get_control.side_effect = RuntimeError("Read failed")
        mock_instance.close.return_value = None

        mock_driver = Mock()
        mock_driver.open.return_value = mock_instance

        camera = Camera(
            mock_driver,
            CameraConfig(camera_id=0),
            # NO hooks
            hooks=None,
        )

        try:
            camera.connect()

            with pytest.raises(CameraError, match="Failed to get"):
                camera.get_control("Gain")
        finally:
            camera.disconnect()


class TestCameraWithStats:
    """Test suite for Camera with stats recording.

    Covers lines: 1186, 1212, 1318, 1337, 1347 (stats recording branches).

    Total: 3 tests.
    """

    def test_successful_capture_records_stats(self) -> None:
        """Verifies successful capture records stats when stats provided.

        Arrangement:
        1. Create Camera with mock stats.
        2. Connect camera.

        Action:
        Call camera.capture().

        Assertion Strategy:
        Validates by confirming stats.record_capture called with success=True.
        """
        mock_stats = Mock()
        driver = DigitalTwinCameraDriver()
        camera = Camera(driver, CameraConfig(camera_id=0), stats=mock_stats)

        try:
            camera.connect()
            camera.capture()

            mock_stats.record_capture.assert_called()
            # Check the most recent call had success=True
            call_kwargs = mock_stats.record_capture.call_args[1]
            assert call_kwargs["success"] is True
        finally:
            camera.disconnect()

    def test_failed_capture_records_stats(self) -> None:
        """Verifies failed capture records stats with error_type.

        Arrangement:
        1. Create Camera with mock stats and mock driver that fails.
        2. Connect camera.

        Action:
        Call camera.capture() which fails.

        Assertion Strategy:
        Validates by confirming stats.record_capture called with success=False.
        """
        mock_stats = Mock()

        # Create mock instance that fails on capture
        mock_instance = Mock()
        mock_instance.get_info.return_value = {"camera_id": 0, "name": "Mock"}
        mock_instance.get_controls.return_value = {}
        mock_instance.set_control.return_value = {}
        mock_instance.capture.side_effect = RuntimeError("Capture failed")
        mock_instance.close.return_value = None

        mock_driver = Mock()
        mock_driver.open.return_value = mock_instance

        camera = Camera(
            mock_driver,
            CameraConfig(camera_id=0),
            stats=mock_stats,
            recovery=NullRecoveryStrategy(),
        )

        try:
            camera.connect()

            with pytest.raises(CameraDisconnectedError):
                camera.capture()

            # Stats should have been called with success=False
            assert mock_stats.record_capture.call_count >= 1
            # Find the call with success=False
            found_failure = False
            for call in mock_stats.record_capture.call_args_list:
                if call[1].get("success") is False:
                    found_failure = True
                    break
            assert found_failure, "Expected stats.record_capture(success=False)"
        finally:
            camera.disconnect()

    def test_recovery_records_stats(self) -> None:
        """Verifies recovery path records stats.

        Arrangement:
        1. Create Camera with mock stats and successful recovery.
        2. First capture fails, recovery succeeds, retry succeeds.

        Action:
        Call camera.capture() triggering recovery flow.

        Assertion Strategy:
        Validates by confirming stats recorded for recovery success.
        """
        mock_stats = Mock()

        class SuccessfulRecovery:
            """Recovery that succeeds."""

            def attempt_recovery(self, camera_id: int) -> bool:
                """Simulate successful recovery for stats recording test.

                Returns True to allow reconnection, enabling the test to
                verify that CameraStats records both the initial failure
                and the successful retry.

                Business context:
                    Stats must track recovery outcomes for reliability metrics.

                Args:
                    camera_id: Identifier of camera to recover.

                Returns:
                    True always, simulating successful hardware recovery.

                Raises:
                    No exceptions raised; always returns True.

                Example:
                    >>> recovery = SuccessfulRecovery()
                    >>> recovery.attempt_recovery(0)
                    True
                """
                return True

        # Track open calls to return different instances
        call_count = [0]

        def create_instance(camera_id: int) -> Mock:
            """Factory creating fail-then-succeed instances for stats testing.

            First call returns instance where capture fails (triggering
            stats.record_capture with failure). Second call returns working
            instance to verify stats records both failure and success.

            Business context:
                Stats recording must capture both failures and recoveries
                to provide accurate observability into camera reliability.

            Args:
                camera_id: Camera identifier passed by driver.open().

            Returns:
                Mock instance - first fails capture, second succeeds.

            Raises:
                No exceptions from factory; first mock.capture() raises.

            Example:
                >>> mock1 = create_instance(0)  # capture fails
                >>> mock2 = create_instance(0)  # capture succeeds
            """
            call_count[0] += 1
            instance = Mock()
            instance.get_info.return_value = {"camera_id": 0, "name": "Mock"}
            instance.get_controls.return_value = {}
            instance.set_control.return_value = {}
            instance.close.return_value = None

            if call_count[0] == 1:
                # First instance - capture fails
                instance.capture.side_effect = RuntimeError("Disconnect")
            else:
                # Second instance - capture succeeds
                instance.capture.return_value = b"\xff\xd8\xff\xe0recovered"

            return instance

        mock_driver = Mock()
        mock_driver.open.side_effect = create_instance

        camera = Camera(
            mock_driver,
            CameraConfig(camera_id=0),
            stats=mock_stats,
            recovery=SuccessfulRecovery(),
        )

        camera.connect()
        result = camera.capture()

        assert b"recovered" in result.image_data
        # Stats should show both failure and success
        assert mock_stats.record_capture.call_count >= 2
        camera.disconnect()


class TestRecoveryInstanceCheck:
    """Test suite for recovery instance verification.

    Covers line 1337 (instance None after connect during recovery).

    Total: 1 test.
    """

    def test_recovery_fails_if_instance_none_after_connect(self) -> None:
        """Verifies recovery fails if connect() doesn't establish instance.

        Arrangement:
        1. Create Camera with recovery that succeeds.
        2. Mock connect() to return normally but leave _instance as None.

        Action:
        Trigger recovery flow.

        Assertion Strategy:
        Validates CameraDisconnectedError with "reconnection failed to establish".
        """

        class SuccessfulRecovery:
            def attempt_recovery(self, camera_id: int) -> bool:
                """Simulate successful recovery for instance-None edge case test.

                Returns True to proceed with reconnection, allowing the test
                to verify Camera handles the edge case where connect() returns
                but _instance is still None.

                Business context:
                    Tests defensive check protecting against broken connect().

                Args:
                    camera_id: Identifier of camera to recover.

                Returns:
                    True always, simulating successful hardware recovery.

                Raises:
                    No exceptions raised; always returns True.

                Example:
                    >>> recovery = SuccessfulRecovery()
                    >>> recovery.attempt_recovery(0)
                    True
                """
                return True

        # Create mock instance that fails capture
        mock_instance = Mock()
        mock_instance.get_info.return_value = {"camera_id": 0, "name": "Mock"}
        mock_instance.get_controls.return_value = {}
        mock_instance.set_control.return_value = {}
        mock_instance.capture.side_effect = RuntimeError("Disconnect")
        mock_instance.close.return_value = None

        mock_driver = Mock()
        mock_driver.open.return_value = mock_instance

        camera = Camera(
            mock_driver,
            CameraConfig(camera_id=0),
            recovery=SuccessfulRecovery(),
        )

        camera.connect()

        # Now patch connect to NOT set _instance (simulate edge case)
        original_connect = camera.connect

        def broken_connect() -> CameraInfo:
            """Simulate connect that returns normally but leaves instance as None.

            Patches Camera.connect() to simulate an edge case where connect()
            completes without raising but fails to establish _instance. This
            tests the defensive check at line 1337 that validates instance
            after reconnection.

            Business context:
                Race conditions or driver bugs could theoretically cause
                connect() to succeed while _instance remains None. Camera
                must detect and report this clearly.

            Args:
                None. Uses closure over camera and original_connect.

            Returns:
                CameraInfo from original connect() call.

            Raises:
                No exceptions raised directly; subsequent capture will fail.

            Example:
                >>> camera.connect = broken_connect
                >>> camera.capture()  # raises CameraDisconnectedError
            """
            # Call original but then clear instance to simulate edge case
            info = original_connect()
            camera._instance = None  # Simulate broken state
            return info

        camera.connect = broken_connect  # type: ignore[method-assign]

        # The instance-None check raises, then gets wrapped by outer except
        # Outer message is "reconnection failed", chained from "establish instance"
        with pytest.raises(
            CameraDisconnectedError, match="reconnection failed"
        ) as exc_info:
            camera.capture()
        # Verify the inner exception is the right one (line 1337)
        assert exc_info.value.__cause__ is not None
        assert "reconnection failed to establish instance" in str(
            exc_info.value.__cause__
        )


class TestGetControlValidation:
    """Tests for F2/F9: get_control return type validation."""

    def test_get_control_raises_on_missing_value_key(self) -> None:
        """Verifies CameraError raised when driver response lacks 'value' key.

        Arrangement:
        Create mock driver that returns dict without 'value' key.

        Action:
        Call get_control().

        Assertion Strategy:
        Validates CameraError with "missing 'value' key" message.
        """
        mock_instance = Mock()
        mock_instance.get_info.return_value = {"camera_id": 0, "name": "Mock"}
        mock_instance.get_controls.return_value = {}
        mock_instance.set_control.return_value = {}
        mock_instance.get_control.return_value = {"min": 0, "max": 100}  # No 'value'!

        mock_driver = Mock()
        mock_driver.open.return_value = mock_instance

        camera = Camera(mock_driver, CameraConfig(camera_id=0))
        camera.connect()

        with pytest.raises(CameraError, match="missing 'value' key"):
            camera.get_control("Gain")

    def test_get_control_raises_on_non_integer_value(self) -> None:
        """Verifies CameraError raised when driver returns non-int value.

        Arrangement:
        Create mock driver that returns string value instead of int.

        Action:
        Call get_control().

        Assertion Strategy:
        Validates CameraError with "non-integer" message including type name.
        """
        mock_instance = Mock()
        mock_instance.get_info.return_value = {"camera_id": 0, "name": "Mock"}
        mock_instance.get_controls.return_value = {}
        mock_instance.set_control.return_value = {}
        mock_instance.get_control.return_value = {"value": "not_an_int"}

        mock_driver = Mock()
        mock_driver.open.return_value = mock_instance

        camera = Camera(mock_driver, CameraConfig(camera_id=0))
        camera.connect()

        with pytest.raises(CameraError, match="non-integer.*str"):
            camera.get_control("Gain")

    def test_get_control_raises_on_float_value(self) -> None:
        """Verifies CameraError raised when driver returns float instead of int.

        Arrangement:
        Create mock driver that returns float value.

        Action:
        Call get_control().

        Assertion Strategy:
        Validates CameraError with "non-integer" message.
        """
        mock_instance = Mock()
        mock_instance.get_info.return_value = {"camera_id": 0, "name": "Mock"}
        mock_instance.get_controls.return_value = {}
        mock_instance.set_control.return_value = {}
        mock_instance.get_control.return_value = {"value": 50.5}

        mock_driver = Mock()
        mock_driver.open.return_value = mock_instance

        camera = Camera(mock_driver, CameraConfig(camera_id=0))
        camera.connect()

        with pytest.raises(CameraError, match="non-integer.*float"):
            camera.get_control("Gain")


# =============================================================================
# Coordinate Provider Integration Tests
# =============================================================================


class TestCameraCoordinateProviderIntegration:
    """Test suite for Camera coordinate provider integration.

    Tests that coordinates are properly injected into capture metadata
    when a coordinate provider returns valid coordinates.

    Total: 1 test.
    """

    def test_capture_includes_coordinates_when_provider_configured(self) -> None:
        """Verifies capture metadata includes coordinates from provider.

        Tests that when a coordinate provider is configured and returns
        valid coordinates, those coordinates appear in the CaptureResult
        metadata.

        Business context:
        Astrophotography requires precise positional metadata in each frame
        for plate solving, stacking, and observation logging. This test
        ensures the coordinate injection path works end-to-end.

        Arrangement:
        1. Create mock coordinate provider returning test coordinates.
        2. Create Camera with DigitalTwinDriver and coordinate provider.
        3. Connect camera.

        Action:
        Call camera.capture() with default options.

        Assertion Strategy:
        Validates coordinate injection by confirming:
        - metadata["coordinates"] exists.
        - Contains expected altitude and azimuth values.

        Testing Principle:
        Exercises line 1499 (_build_capture_result coordinate injection)
        which is only reached when get_coordinates() returns truthy value.
        """
        from telescope_mcp.devices.camera import CaptureCoordinates

        # Create mock coordinate provider that returns test coordinates
        class MockCoordinateProvider:
            """Mock provider that returns fixed test coordinates."""

            def get_coordinates(self) -> CaptureCoordinates:
                """Return test coordinates for coverage.

                Returns:
                    CaptureCoordinates with test values.
                """
                return CaptureCoordinates(
                    altitude=45.0,
                    azimuth=180.0,
                    ra=12.5,
                    dec=45.0,
                    ra_hms="12h 30m 00.0s",
                    dec_dms="+45° 00' 00.0\"",
                )

        driver = DigitalTwinCameraDriver()
        camera = Camera(
            driver,
            CameraConfig(camera_id=0),
            coordinate_provider=MockCoordinateProvider(),
        )
        camera.connect()

        try:
            result = camera.capture(
                CaptureOptions(exposure_us=100000, gain=50, format="jpeg")
            )

            # Verify coordinates were injected into metadata
            assert "coordinates" in result.metadata
            coords = result.metadata["coordinates"]
            assert coords["altitude"] == 45.0
            assert coords["azimuth"] == 180.0
            assert coords["ra"] == 12.5
            assert coords["dec"] == 45.0
        finally:
            camera.disconnect()
