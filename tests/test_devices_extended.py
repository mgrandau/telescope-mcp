"""Extended tests for device layer - Camera, Controller, Registry."""

import time

import pytest

from telescope_mcp.devices.camera import (
    Camera,
    CameraConfig,
    CaptureOptions,
    OverlayConfig,
    SystemClock,
)
from telescope_mcp.devices.controller import (
    CameraController,
    CameraNotFoundError,
    SyncCaptureConfig,
)
from telescope_mcp.drivers.cameras.twin import DigitalTwinCameraDriver


class TestCameraCapture:
    """Tests for Camera capture functionality."""

    @pytest.fixture
    def camera(self):
        """Pytest fixture providing connected Camera for capture tests.

        Creates a DigitalTwinCameraDriver-backed Camera, connects it,
        yields for test usage, and automatically disconnects for cleanup.
        This ensures each test starts with a fresh connected camera
        instance, preventing state leakage between tests.

        Business Context:
        Camera capture is core to telescope operations. Testing requires
        isolated camera instances to validate exposure control, gain
        settings, and image acquisition without hardware dependencies.

        Arrangement:
        1. Instantiate DigitalTwinCameraDriver (simulated hardware).
        2. Create Camera with camera_id=0, name="Test".
        3. Connect camera to initialize hardware state.

        Args:
            None (pytest fixture with implicit request parameter).

        Returns:
            Camera: Connected camera instance ready for capture operations.
                Camera configured with camera_id=0 and name="Test".

        Raises:
            None. DigitalTwin driver doesn't raise connection errors.

        Example:
            >>> def test_example(camera):
            ...     result = camera.capture(CaptureOptions(exposure_us=100000))
            ...     assert result.image_data is not None

        Testing Principle:
            Validates fixture lifecycle management, ensuring proper
            resource initialization and cleanup for isolated tests.
        """
        driver = DigitalTwinCameraDriver()
        cam = Camera(driver, CameraConfig(camera_id=0, name="Test"))
        cam.connect()
        yield cam
        cam.disconnect()

    def test_capture_with_options(self, camera):
        """Verifies capture respects CaptureOptions settings.

        Arrangement:
        1. Camera fixture provides connected camera.
        2. CaptureOptions specifies exposure=150ms, gain=75.
        3. Result should reflect provided settings.

        Action:
        Calls camera.capture() with custom options.

        Assertion Strategy:
        Validates option handling by confirming:
        - result.exposure_us equals 150000.
        - result.gain equals 75.

        Testing Principle:
        Validates options pattern, ensuring capture settings
        properly passed through to result metadata.
        """
        options = CaptureOptions(exposure_us=150000, gain=75)
        result = camera.capture(options)
        assert result.exposure_us == 150000
        assert result.gain == 75

    def test_capture_multiple_sequential(self, camera):
        """Verifies camera handles sequential captures correctly.

        Arrangement:
        1. Camera fixture provides connected camera.
        2. Three captures with different exposure settings.
        3. Each capture independent with own settings.

        Action:
        Performs 3 consecutive captures with varying exposure.

        Assertion Strategy:
        Validates sequential operation by confirming:
        - First result has exposure_us=50000.
        - Second result has exposure_us=100000.
        - Third result has exposure_us=150000.

        Testing Principle:
        Validates stateless capture, ensuring each operation
        independent and settings not mixed between calls.
        """
        result1 = camera.capture(CaptureOptions(exposure_us=50000, gain=25))
        result2 = camera.capture(CaptureOptions(exposure_us=100000, gain=50))
        result3 = camera.capture(CaptureOptions(exposure_us=150000, gain=75))

        assert result1.exposure_us == 50000
        assert result2.exposure_us == 100000
        assert result3.exposure_us == 150000

    def test_capture_updates_timestamp(self, camera):
        """Verifies each capture has unique increasing timestamp.

        Arrangement:
        1. Camera fixture provides connected camera.
        2. Two captures with identical settings.
        3. Timestamps should differ despite same settings.

        Action:
        Performs 2 captures and compares timestamps.

        Assertion Strategy:
        Validates timestamp generation by confirming:
        - result2.timestamp > result1.timestamp.
        - Each capture gets current time.

        Testing Principle:
        Validates temporal ordering, ensuring frames
        distinguishable by capture time.
        """
        result1 = camera.capture(CaptureOptions(exposure_us=100000, gain=50))
        result2 = camera.capture(CaptureOptions(exposure_us=100000, gain=50))

        assert result2.timestamp > result1.timestamp

    def test_config_property(self, camera):
        """Verifies camera.config provides access to configuration.

        Arrangement:
        1. Camera fixture with camera_id=0, name="Test".
        2. config property should return CameraConfig instance.
        3. Config immutable, reflects construction values.

        Action:
        Accesses camera.config property.

        Assertion Strategy:
        Validates config access by confirming:
        - config.camera_id equals 0.
        - config.name equals "Test".

        Testing Principle:
        Validates configuration access, ensuring camera
        settings queryable post-initialization.
        """
        config = camera.config
        assert config.camera_id == 0
        assert config.name == "Test"


class TestCameraOverlay:
    """Tests for Camera overlay functionality."""

    @pytest.fixture
    def camera(self):
        """Pytest fixture providing connected Camera for overlay tests.

        Creates a DigitalTwinCameraDriver-backed Camera specifically for
        testing overlay configuration (crosshairs, grids, etc.). Manages
        full lifecycle: creation, connection, yield, and disconnection.

        Business Context:
        Overlays (crosshairs, grid lines) assist telescope alignment and
        target positioning. Testing requires isolated cameras to validate
        overlay rendering without affecting other test state.

        Arrangement:
        1. Instantiate DigitalTwinCameraDriver (simulated hardware).
        2. Create Camera with camera_id=0 (default name).
        3. Connect camera to ready for overlay operations.

        Args:
            None (pytest fixture with implicit request parameter).

        Returns:
            Camera: Connected camera instance (camera_id=0) ready for
                overlay configuration via set_overlay() method.

        Raises:
            None. DigitalTwin driver doesn't raise connection errors.

        Example:
            >>> def test_example(camera):
            ...     overlay = OverlayConfig(enabled=True, overlay_type="crosshair")
            ...     camera.set_overlay(overlay)
            ...     assert camera.overlay is not None

        Testing Principle:
            Validates fixture lifecycle, ensuring clean camera state
            for each overlay test without configuration leakage.
        """
        driver = DigitalTwinCameraDriver()
        cam = Camera(driver, CameraConfig(camera_id=0))
        cam.connect()
        yield cam
        cam.disconnect()

    def test_set_overlay(self, camera):
        """Verifies set_overlay() configures visual overlay.

        Arrangement:
        1. Camera fixture provides connected camera.
        2. OverlayConfig specifies crosshair with green color.
        3. Camera should store overlay configuration.

        Action:
        Calls camera.set_overlay() with OverlayConfig.

        Assertion Strategy:
        Validates overlay configuration by confirming:
        - camera.overlay is not None.
        - overlay.enabled is True.
        - overlay.overlay_type equals "crosshair".

        Testing Principle:
        Validates overlay configuration API, ensuring
        visual augmentation settings properly stored.
        """
        overlay = OverlayConfig(
            enabled=True,
            overlay_type="crosshair",
            color=(0, 255, 0),
            opacity=0.8,
        )
        camera.set_overlay(overlay)
        assert camera.overlay is not None
        assert camera.overlay.enabled is True
        assert camera.overlay.overlay_type == "crosshair"

    def test_clear_overlay(self, camera):
        """Verifies set_overlay(None) clears overlay configuration.

        Arrangement:
        1. Camera starts without overlay.
        2. Overlay set with crosshair config.
        3. set_overlay(None) should clear configuration.

        Action:
        Sets overlay, then clears with set_overlay(None).

        Assertion Strategy:
        Validates overlay clearing by confirming:
        - Overlay present after set_overlay().
        - camera.overlay is None after set_overlay(None).

        Testing Principle:
        Validates configuration reset, ensuring overlay
        removable when not desired.
        """
        overlay = OverlayConfig(enabled=True, overlay_type="crosshair")
        camera.set_overlay(overlay)
        assert camera.overlay is not None

        camera.set_overlay(None)
        assert camera.overlay is None

    def test_overlay_persists_across_captures(self, camera):
        """Verifies overlay config persists through capture operations.

        Arrangement:
        1. Camera overlay set to grid type.
        2. Capture operation performed.
        3. Overlay should remain configured post-capture.

        Action:
        Sets overlay, performs capture, checks overlay still present.

        Assertion Strategy:
        Validates persistence by confirming:
        - camera.overlay not None after capture.
        - overlay_type still equals "grid".

        Testing Principle:
        Validates configuration stability, ensuring overlay
        settings not cleared by capture operations.
        """
        overlay = OverlayConfig(enabled=True, overlay_type="grid")
        camera.set_overlay(overlay)

        camera.capture(CaptureOptions(exposure_us=100000, gain=50))
        assert camera.overlay is not None
        assert camera.overlay.overlay_type == "grid"


class TestCameraStreaming:
    """Tests for Camera streaming functionality."""

    @pytest.fixture
    def camera(self):
        """Pytest fixture providing connected Camera for streaming tests.

        Business Context:
            Streaming operations (sequential frame capture) are essential for
            telescope operations like alignment refinement, focus adjustment,
            and real-time object tracking. Each frame must be independent to
            avoid cumulative timing or state errors during long observation sessions.

        Creates a DigitalTwinCameraDriver-backed Camera for testing
        sequential frame capture (streaming scenarios). Ensures proper
        lifecycle management with automatic cleanup.

        Arrangement:
        1. Instantiate DigitalTwinCameraDriver (simulated hardware).
        2. Create Camera with camera_id=0 (default name).
        3. Connect camera to ready for sequential captures.

        Args:
            None (pytest fixture with implicit request parameter).

        Returns:
            Camera: Connected camera instance (camera_id=0) ready for
                sequential capture operations testing frame independence.

        Raises:
            None. DigitalTwin driver doesn't raise connection errors.

        Example:
            >>> def test_example(camera):
            ...     frame1 = camera.capture(CaptureOptions(exposure_us=100000))
            ...     frame2 = camera.capture(CaptureOptions(exposure_us=100000))
            ...     assert frame1.timestamp != frame2.timestamp

        Testing Principle:
            Validates fixture reusability, ensuring camera handles
            multiple sequential captures without state corruption.
        """
        driver = DigitalTwinCameraDriver()
        cam = Camera(driver, CameraConfig(camera_id=0))
        cam.connect()
        yield cam
        cam.disconnect()

    def test_capture_sequence(self, camera):
        """Verifies sequential frame capture produces distinct results.

        Arrangement:
        1. Camera fixture provides connected camera.
        2. Two captures with identical exposure settings.
        3. Each should produce independent result.

        Action:
        Performs 2 consecutive captures with same settings.

        Assertion Strategy:
        Validates independent captures by confirming:
        - Both results contain non-None image_data.
        - Timestamps differ (unique captures).

        Testing Principle:
        Validates capture independence, ensuring repeated
        operations produce distinct frames.
        """
        # Capture multiple frames
        result1 = camera.capture(CaptureOptions(exposure_us=100000))
        result2 = camera.capture(CaptureOptions(exposure_us=100000))

        # Both should succeed
        assert result1.image_data is not None
        assert result2.image_data is not None
        assert result1.timestamp != result2.timestamp


class TestCameraControls:
    """Tests for Camera control operations."""

    @pytest.fixture
    def camera(self):
        """Pytest fixture providing connected Camera for control tests.

        Business Context:
            Camera controls (gain, exposure, white balance) are critical for
            adapting telescope imaging to varying sky conditions, target brightness,
            and atmospheric transparency. Proper control validation ensures accurate
            hardware interaction and prevents invalid settings that could corrupt
            imaging sessions or damage equipment.

        Creates a DigitalTwinCameraDriver-backed Camera for testing
        hardware control operations (gain, exposure, etc. via set_control).
        Manages full lifecycle with automatic cleanup.

        Arrangement:
        1. Instantiate DigitalTwinCameraDriver (simulated hardware).
        2. Create Camera with camera_id=0 (default name).
        3. Connect camera to ready for control operations.

        Args:
            None (pytest fixture with implicit request parameter).

        Returns:
            Camera: Connected camera instance (camera_id=0) ready for
                set_control() operations testing hardware interaction.

        Raises:
            None. DigitalTwin driver doesn't raise connection errors.

        Example:
            >>> def test_example(camera):
            ...     camera.set_control("Gain", 100)
            ...     # DigitalTwin accepts control without validation

        Testing Principle:
            Validates fixture isolation, ensuring control operations
            don't affect subsequent tests via shared state.
        """
        driver = DigitalTwinCameraDriver()
        cam = Camera(driver, CameraConfig(camera_id=0))
        cam.connect()
        yield cam
        cam.disconnect()

    def test_set_control(self, camera):
        """Verifies set_control() accepts control parameters.

        Arrangement:
        1. Camera fixture provides connected camera.
        2. DigitalTwinDriver accepts arbitrary controls.
        3. No exception means control accepted.

        Action:
        Calls camera.set_control("Gain", 100).

        Assertion Strategy:
        Validates control API by confirming:
        - No exception raised (implicit success).
        - Control value accepted.

        Testing Principle:
        Validates control interface, ensuring parameter
        setting API works for hardware interaction.
        """
        # DigitalTwin accepts but doesn't validate control values
        camera.set_control("Gain", 100)
        # Control should be set (verified by no exception)


class TestCameraController:
    """Tests for CameraController sync capture."""

    @pytest.fixture
    def controller(self):
        """Pytest fixture providing CameraController with dual cameras.

        Business Context:
            Dual-camera telescope systems require precise synchronization between
            finder scope (alignment/target acquisition) and main imaging camera.
            Controller orchestration ensures both cameras work in harmony during
            GoTo operations, plate solving, and simultaneous widefield/narrowfield
            imaging without timing conflicts or resource contention.

        Creates a CameraController configured with finder and main cameras
        for testing synchronized capture operations. Both cameras share
        a DigitalTwinCameraDriver instance for consistent simulation.
        Manages full lifecycle with automatic cleanup of both cameras.

        Arrangement:
        1. Instantiate DigitalTwinCameraDriver (simulated hardware).
        2. Create finder camera (camera_id=0, name="Finder").
        3. Create main camera (camera_id=1, name="Main").
        4. Connect both cameras to ready for operations.
        5. Instantiate CameraController with cameras dict.

        Args:
            None (pytest fixture with implicit request parameter).

        Returns:
            CameraController: Controller managing "finder" and "main" cameras
                for synchronized capture testing via sync_capture() method.

        Raises:
            None. DigitalTwin driver doesn't raise connection errors.

        Example:
            >>> def test_example(controller):
            ...     config = SyncCaptureConfig(primary="finder", secondary="main",
            ...                                primary_exposure_us=1000000,
            ...                                secondary_exposure_us=100000)
            ...     result = controller.sync_capture(config)
            ...     assert result.primary_frame is not None

        Testing Principle:
            Validates multi-camera coordination, ensuring controller
            properly orchestrates dual-camera synchronized operations.
        """
        driver = DigitalTwinCameraDriver()
        cam0 = Camera(driver, CameraConfig(camera_id=0, name="Finder"))
        cam1 = Camera(driver, CameraConfig(camera_id=1, name="Main"))
        cam0.connect()
        cam1.connect()

        controller = CameraController(cameras={"finder": cam0, "main": cam1})
        yield controller

        cam0.disconnect()
        cam1.disconnect()

    def test_sync_capture_basic(self, controller):
        """Verifies synchronized dual-camera capture.

        Arrangement:
        1. Controller fixture with finder and main cameras.
        2. SyncCaptureConfig specifies primary/secondary exposures.
        3. Both cameras should capture simultaneously.

        Action:
        Calls controller.sync_capture() with config.

        Assertion Strategy:
        Validates sync capture by confirming:
        - result.primary_frame is not None.
        - result.secondary_frame is not None.

        Testing Principle:
        Validates multi-camera coordination, ensuring
        simultaneous capture for alignment/guiding.
        """
        config = SyncCaptureConfig(
            primary="finder",
            secondary="main",
            primary_exposure_us=1_000_000,
            secondary_exposure_us=100_000,
        )
        result = controller.sync_capture(config)

        assert result.primary_frame is not None
        assert result.secondary_frame is not None

    def test_sync_capture_with_custom_gain(self, controller):
        """Verifies sync capture applies independent gain settings.

        Arrangement:
        1. Controller with finder and main cameras.
        2. Config specifies different gain for each camera.
        3. Each camera should use own gain value.

        Action:
        Sync captures with primary_gain=100, secondary_gain=200.

        Assertion Strategy:
        Validates independent settings by confirming:
        - primary_frame.gain equals 100.
        - secondary_frame.gain equals 200.

        Testing Principle:
        Validates per-camera configuration, ensuring
        sync capture doesn't force identical settings.
        """
        config = SyncCaptureConfig(
            primary="finder",
            secondary="main",
            primary_exposure_us=500_000,
            secondary_exposure_us=50_000,
            primary_gain=100,
            secondary_gain=200,
        )
        result = controller.sync_capture(config)

        assert result.primary_frame.gain == 100
        assert result.secondary_frame.gain == 200

    def test_sync_capture_timing_error(self, controller):
        """Verifies sync result includes timing error metrics.

        Arrangement:
        1. Controller with finder and main cameras.
        2. Long primary exposure (2s), short secondary (100ms).
        3. Result should calculate timestamp difference.

        Action:
        Performs sync capture with mismatched exposures.

        Assertion Strategy:
        Validates timing metrics by confirming:
        - result has timing_error_us attribute.
        - result has timing_error_ms attribute.
        - timing_error_ms equals timing_error_us / 1000.

        Testing Principle:
        Validates observability metrics, ensuring sync
        quality measurable for alignment analysis.
        """
        config = SyncCaptureConfig(
            primary="finder",
            secondary="main",
            primary_exposure_us=2_000_000,
            secondary_exposure_us=100_000,
        )
        result = controller.sync_capture(config)

        # Timing error should be calculated
        assert hasattr(result, "timing_error_us")
        assert hasattr(result, "timing_error_ms")
        assert result.timing_error_ms == result.timing_error_us / 1000.0

    def test_get_camera(self, controller):
        """Verifies controller stores cameras by name.

        Arrangement:
        1. Controller fixture with finder and main cameras.
        2. Cameras stored in _cameras dict by key.
        3. Camera accessible via dict lookup.

        Action:
        Accesses controller._cameras["finder"].

        Assertion Strategy:
        Validates camera storage by confirming:
        - Camera is not None.
        - cam.config.name equals "Finder".

        Testing Principle:
        Validates internal organization, ensuring cameras
        properly registered and accessible.
        """
        cam = controller._cameras["finder"]
        assert cam is not None
        assert cam.config.name == "Finder"

    def test_sync_capture_invalid_camera(self, controller):
        """Verifies sync capture rejects unknown camera names.

        Arrangement:
        1. Controller with finder and main cameras only.
        2. Config references non-existent "invalid" camera.
        3. Should raise CameraNotFoundError.

        Action:
        Attempts sync_capture with primary="invalid".

        Assertion Strategy:
        Validates error handling by confirming:
        - CameraNotFoundError raised.
        - Invalid camera name detected.

        Testing Principle:
        Validates error handling, ensuring clear feedback
        for configuration mistakes.
        """
        config = SyncCaptureConfig(
            primary="invalid",
            secondary="main",
            primary_exposure_us=1_000_000,
            secondary_exposure_us=100_000,
        )
        with pytest.raises(CameraNotFoundError):
            controller.sync_capture(config)


# TestCameraRegistry removed - covered by test_devices.py


class TestSystemClock:
    """Test suite for SystemClock time measurement utilities.

    Categories:
    1. Monotonic Time - Increasing time values (1 test)
    2. Sleep Operations - Duration accuracy (1 test)

    Total: 2 tests.
    """

    def test_monotonic_increases(self):
        """Verifies SystemClock.monotonic() returns increasing values.

        Tests monotonic time source for timing measurements.

        Arrangement:
        1. Create SystemClock instance.
        2. Record time before and after 10ms sleep.

        Action:
        Calls clock.monotonic() twice with sleep between.

        Assertion Strategy:
        Validates monotonic property by confirming:
        - t2 > t1 (time increases).
        - Clock suitable for duration measurement.

        Testing Principle:
        Validates timing source, ensuring clock provides
        monotonic timestamps unaffected by system time changes."""
        clock = SystemClock()
        t1 = clock.monotonic()
        time.sleep(0.01)
        t2 = clock.monotonic()
        assert t2 > t1

    def test_sleep_duration(self):
        """Verifies SystemClock.sleep() pauses for specified duration.

        Tests sleep accuracy for timing control.

        Arrangement:
        1. Create SystemClock instance.
        2. Record start time before sleep.
        3. Target sleep duration: 50ms.

        Action:
        Calls clock.sleep(0.05) and measures elapsed time.

        Assertion Strategy:
        Validates sleep duration by confirming:
        - elapsed >= 0.04s (40ms minimum, small margin).
        - elapsed < 0.1s (not excessive delay).

        Testing Principle:
        Validates sleep accuracy, ensuring clock provides
        reasonable timing control for frame rate limiting."""
        clock = SystemClock()
        start = time.time()
        clock.sleep(0.05)
        elapsed = time.time() - start
        assert elapsed >= 0.04  # Allow small margin
        assert elapsed < 0.1


class TestCameraInfo:
    """Test suite for Camera.info property and metadata.

    Categories:
    1. Info Availability - Post-connection metadata (1 test)
    2. Info Properties - Attribute completeness (1 test)

    Total: 2 tests.
    """

    def test_camera_info_after_connect(self):
        """Verifies Camera.info populated after connect() call.

        Tests camera metadata availability lifecycle.

        Arrangement:
        1. Create DigitalTwinCameraDriver.
        2. Instantiate Camera without connecting.
        3. Info should be None before connect.

        Action:
        Checks info before connect, calls connect(), checks after.

        Assertion Strategy:
        Validates info lifecycle by confirming:
        - camera.info is None initially.
        - camera.info is not None after connect().
        - info.camera_id equals 0.

        Testing Principle:
        Validates lazy initialization, ensuring metadata
        loaded only after hardware connection established."""
        driver = DigitalTwinCameraDriver()
        camera = Camera(driver, CameraConfig(camera_id=0))

        assert camera.info is None
        camera.connect()
        assert camera.info is not None
        assert camera.info.camera_id == 0

        camera.disconnect()

    def test_camera_info_properties(self):
        """Verifies Camera.info contains all required metadata attributes.

        Tests CameraInfo completeness.

        Arrangement:
        1. Create DigitalTwinCameraDriver.
        2. Create and connect Camera to populate info.

        Action:
        Accesses camera.info and checks for attributes.

        Assertion Strategy:
        Validates info completeness by confirming:
        - info is not None.
        - Has camera_id attribute.
        - Has name, max_width, max_height attributes.
        - Has is_color attribute.

        Testing Principle:
        Validates metadata interface, ensuring CameraInfo
        provides complete camera specifications."""
        driver = DigitalTwinCameraDriver()
        camera = Camera(driver, CameraConfig(camera_id=0))
        camera.connect()

        info = camera.info
        assert info is not None
        assert hasattr(info, "camera_id")
        assert hasattr(info, "name")
        assert hasattr(info, "max_width")
        assert hasattr(info, "max_height")
        assert hasattr(info, "is_color")

        camera.disconnect()
