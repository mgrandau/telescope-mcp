"""Extended tests for device layer - Camera, Controller, Registry."""

import time

import pytest

from telescope_mcp.devices.camera import (
    Camera,
    CameraConfig,
    CameraInfo,
    CaptureOptions,
    OverlayConfig,
    SystemClock,
)
from telescope_mcp.devices.camera_controller import (
    CameraController,
    CameraNotFoundError,
    SyncCaptureConfig,
    SyncCaptureError,
)
from telescope_mcp.devices.camera_registry import (
    CameraNotInRegistryError,
    CameraRegistry,
    NullRecoveryStrategy,
    RecoveryStrategy,
    get_registry,
    init_registry,
    shutdown_registry,
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


class TestCameraControllerCoverage:
    """Additional coverage tests for CameraController."""

    def test_init_without_cameras(self):
        """Verifies CameraController initializes with empty cameras dict.

        Tests that controller can be created without cameras, allowing
        cameras to be added later via add_camera().

        Arrangement:
            1. No setup needed - testing default initialization.

        Action:
            Instantiate CameraController() with no arguments.

        Assertion Strategy:
            Validates empty initialization by confirming:
            - camera_names property returns empty list.

        Testing Principle:
            Validates flexible instantiation, ensuring controller
            supports incremental camera registration.
        """
        controller = CameraController()
        assert controller.camera_names == []

    def test_init_with_custom_clock(self):
        """Verifies CameraController accepts custom clock injection.

        Tests dependency injection of clock for deterministic testing.
        Production uses SystemClock; tests inject MockClock.

        Arrangement:
            1. Create Mock clock with monotonic() returning 0.0.

        Action:
            Instantiate CameraController with clock=mock_clock.

        Assertion Strategy:
            Validates clock injection by confirming:
            - controller._clock is the injected mock.

        Testing Principle:
            Validates dependency injection, ensuring timing can be
            mocked for deterministic sync capture tests.
        """
        from unittest.mock import Mock

        mock_clock = Mock()
        mock_clock.monotonic.return_value = 0.0
        controller = CameraController(cameras={}, clock=mock_clock)
        assert controller._clock is mock_clock

    def test_add_camera(self):
        """Verifies add_camera registers camera by name.

        Tests that cameras can be added after controller creation
        and are accessible by their registered name.

        Arrangement:
            1. Create DigitalTwinCameraDriver.
            2. Create and connect Camera instance.
            3. Instantiate empty CameraController.

        Action:
            Call controller.add_camera("test", camera).

        Assertion Strategy:
            Validates camera registration by confirming:
            - "test" appears in camera_names.
            - get_camera("test") returns same instance.

        Testing Principle:
            Validates registration API, ensuring cameras can be
            dynamically added to controller after instantiation.
        """
        driver = DigitalTwinCameraDriver()
        camera = Camera(driver, CameraConfig(camera_id=0, name="Test"))
        camera.connect()

        controller = CameraController()
        controller.add_camera("test", camera)

        assert "test" in controller.camera_names
        assert controller.get_camera("test") is camera

        camera.disconnect()

    def test_remove_camera_existing(self):
        """Verifies remove_camera returns and removes camera.

        Tests that removing an existing camera returns the instance
        and unregisters it from the controller.

        Arrangement:
            1. Create and connect camera via DigitalTwinCameraDriver.
            2. Create controller with camera registered as "test".

        Action:
            Call controller.remove_camera("test").

        Assertion Strategy:
            Validates removal by confirming:
            - Returned value is the original camera instance.
            - "test" no longer in camera_names.

        Testing Principle:
            Validates removal API, ensuring cameras can be
            dynamically unregistered from controller.
        """
        driver = DigitalTwinCameraDriver()
        camera = Camera(driver, CameraConfig(camera_id=0, name="Test"))
        camera.connect()

        controller = CameraController(cameras={"test": camera})
        removed = controller.remove_camera("test")

        assert removed is camera
        assert "test" not in controller.camera_names

        camera.disconnect()

    def test_remove_camera_nonexistent(self):
        """Verifies remove_camera returns None for unknown name.

        Tests graceful handling of removal request for non-existent camera.

        Arrangement:
            1. Create empty CameraController.

        Action:
            Call controller.remove_camera("nonexistent").

        Assertion Strategy:
            Validates graceful failure by confirming:
            - Returns None (not KeyError or exception).

        Testing Principle:
            Validates defensive coding, ensuring removal of
            non-existent camera doesn't raise exception.
        """
        controller = CameraController()
        removed = controller.remove_camera("nonexistent")
        assert removed is None

    def test_get_camera_raises_not_found(self):
        """Verifies get_camera raises CameraNotFoundError for unknown name.

        Tests that accessing non-existent camera raises structured error.

        Arrangement:
            1. Create empty CameraController.

        Action:
            Call controller.get_camera("unknown").

        Assertion Strategy:
            Validates error handling by confirming:
            - CameraNotFoundError is raised.
            - Error message contains requested name.

        Testing Principle:
            Validates explicit error reporting, ensuring clear
            exception for invalid camera lookup.
        """
        controller = CameraController()
        with pytest.raises(CameraNotFoundError) as exc_info:
            controller.get_camera("unknown")
        assert "unknown" in str(exc_info.value)

    def test_camera_names_property(self):
        """Verifies camera_names returns list of registered names.

        Tests that property exposes names of all registered cameras
        for enumeration and status reporting.

        Arrangement:
            1. Create two cameras via DigitalTwinCameraDriver.
            2. Connect both cameras.
            3. Register as "alpha" and "beta" in controller.

        Action:
            Access controller.camera_names property.

        Assertion Strategy:
            Validates name enumeration by confirming:
            - Returns list type.
            - Contains "alpha" and "beta".
            - Length equals 2 (no extra names).

        Testing Principle:
            Validates introspection API, ensuring callers can
            enumerate registered cameras for UI/logging.
        """
        driver = DigitalTwinCameraDriver()
        cam1 = Camera(driver, CameraConfig(camera_id=0, name="Cam1"))
        cam2 = Camera(driver, CameraConfig(camera_id=1, name="Cam2"))
        cam1.connect()
        cam2.connect()

        controller = CameraController(cameras={"alpha": cam1, "beta": cam2})
        names = controller.camera_names

        assert isinstance(names, list)
        assert "alpha" in names
        assert "beta" in names
        assert len(names) == 2

        cam1.disconnect()
        cam2.disconnect()

    def test_calculate_sync_timing(self):
        """Verifies calculate_sync_timing computes correct delay.

        Tests delay calculation for temporally centered secondary exposure.
        Primary: 176s finder exposure, Secondary: 312ms main exposure.

        Arrangement:
            1. Create empty CameraController.
            2. Primary exposure: 176,000,000us (176s).
            3. Secondary exposure: 312,000us (312ms).

        Action:
            Call calculate_sync_timing(176_000_000, 312_000).

        Assertion Strategy:
            Validates timing math by confirming:
            - Returns 87,844,000us (midpoint - half_secondary).
            - Formula: (176M/2) - (312K/2) = 88M - 156K = 87.844M.

        Testing Principle:
            Validates timing algorithm, ensuring secondary capture
            centers within primary exposure window.
        """
        controller = CameraController()

        # 176s primary, 312ms secondary
        delay = controller.calculate_sync_timing(176_000_000, 312_000)

        # Midpoint of primary: 88s = 88_000_000us
        # Half of secondary: 156ms = 156_000us
        # Delay: 88_000_000 - 156_000 = 87_844_000us
        assert delay == 87_844_000

    def test_calculate_sync_timing_equal_exposures(self):
        """Verifies calculate_sync_timing handles equal exposures.

        Tests edge case where both cameras have same exposure time.
        No delay needed - both start together.

        Arrangement:
            1. Create empty CameraController.
            2. Both exposures: 1,000,000us (1s).

        Action:
            Call calculate_sync_timing(1_000_000, 1_000_000).

        Assertion Strategy:
            Validates equal exposure handling by confirming:
            - Returns 0 (no delay needed).

        Testing Principle:
            Validates edge case handling, ensuring equal exposures
            result in simultaneous start (delay=0).
        """
        controller = CameraController()
        delay = controller.calculate_sync_timing(1_000_000, 1_000_000)
        assert delay == 0

    def test_calculate_sync_timing_secondary_longer(self):
        """Verifies calculate_sync_timing returns 0 when secondary >= primary.

        Tests edge case where secondary exposure is longer than primary.
        Secondary starts immediately (negative delay clamped to 0).

        Arrangement:
            1. Create empty CameraController.
            2. Primary: 100,000us, Secondary: 200,000us.

        Action:
            Call calculate_sync_timing(100_000, 200_000).

        Assertion Strategy:
            Validates clamping by confirming:
            - Returns 0 (not negative delay).

        Testing Principle:
            Validates boundary handling, ensuring longer secondary
            doesn't produce invalid negative delay.
        """
        controller = CameraController()
        # Secondary longer than primary - no delay needed
        delay = controller.calculate_sync_timing(100_000, 200_000)
        assert delay == 0

    def test_calculate_sync_timing_invalid_primary(self):
        """Verifies calculate_sync_timing raises ValueError for non-positive primary.

        Tests input validation rejects zero/negative primary exposure.

        Arrangement:
            1. Create empty CameraController.
            2. Primary: 0us (invalid), Secondary: 100,000us.

        Action:
            Call calculate_sync_timing(0, 100_000).

        Assertion Strategy:
            Validates input validation by confirming:
            - ValueError is raised.
            - Error message mentions "primary_exposure_us must be positive".

        Testing Principle:
            Validates input validation, ensuring invalid exposure
            values are rejected with clear error messages.
        """
        controller = CameraController()
        with pytest.raises(ValueError) as exc_info:
            controller.calculate_sync_timing(0, 100_000)
        assert "primary_exposure_us must be positive" in str(exc_info.value)

    def test_calculate_sync_timing_invalid_secondary(self):
        """Verifies calculate_sync_timing raises ValueError for non-positive secondary.

        Tests input validation rejects negative secondary exposure.

        Arrangement:
            1. Create empty CameraController.
            2. Primary: 100,000us, Secondary: -1us (invalid).

        Action:
            Call calculate_sync_timing(100_000, -1).

        Assertion Strategy:
            Validates input validation by confirming:
            - ValueError is raised.
            - Error message mentions "secondary_exposure_us must be positive".

        Testing Principle:
            Validates input validation, ensuring negative exposure
            values are rejected with clear error messages.
        """
        controller = CameraController()
        with pytest.raises(ValueError) as exc_info:
            controller.calculate_sync_timing(100_000, -1)
        assert "secondary_exposure_us must be positive" in str(exc_info.value)

    def test_add_camera_duplicate_raises(self):
        """Verifies add_camera raises ValueError on duplicate without overwrite.

        Tests that attempting to register a camera with an existing name
        raises an error unless overwrite=True is specified.

        Arrangement:
            1. Create empty CameraController.
            2. Create two mock cameras.
            3. Register first mock as "test".

        Action:
            Call add_camera("test", mock_cam2) without overwrite flag.

        Assertion Strategy:
            Validates duplicate protection by confirming:
            - ValueError is raised.
            - Error message contains "already registered".

        Testing Principle:
            Validates safety rails, ensuring accidental overwrites
            require explicit intent via overwrite=True.
        """
        from unittest.mock import Mock

        controller = CameraController()
        mock_cam1 = Mock()
        mock_cam2 = Mock()

        controller.add_camera("test", mock_cam1)
        with pytest.raises(ValueError) as exc_info:
            controller.add_camera("test", mock_cam2)
        assert "already registered" in str(exc_info.value)

    def test_add_camera_overwrite(self):
        """Verifies add_camera with overwrite=True replaces existing camera.

        Tests that overwrite flag allows replacing a registered camera.

        Arrangement:
            1. Create empty CameraController.
            2. Create two mock cameras.
            3. Register first mock as "test".

        Action:
            Call add_camera("test", mock_cam2, overwrite=True).

        Assertion Strategy:
            Validates replacement by confirming:
            - get_camera("test") returns mock_cam2 (not mock_cam1).

        Testing Principle:
            Validates explicit override API, ensuring cameras can be
            replaced when explicitly requested.
        """
        from unittest.mock import Mock

        controller = CameraController()
        mock_cam1 = Mock()
        mock_cam2 = Mock()

        controller.add_camera("test", mock_cam1)
        controller.add_camera("test", mock_cam2, overwrite=True)
        assert controller.get_camera("test") is mock_cam2

    def test_camera_not_found_error_has_name(self):
        """Verifies CameraNotFoundError stores camera_name attribute.

        Tests that exception provides programmatic access to requested name.

        Arrangement:
            1. Create empty CameraController.

        Action:
            Call get_camera("missing_cam") which raises.

        Assertion Strategy:
            Validates structured exception by confirming:
            - exc_info.value.camera_name equals "missing_cam".

        Testing Principle:
            Validates exception richness, ensuring error contains
            context for programmatic handling (e.g., suggest alternatives).
        """
        controller = CameraController()
        with pytest.raises(CameraNotFoundError) as exc_info:
            controller.get_camera("missing_cam")
        assert exc_info.value.camera_name == "missing_cam"


class TestCameraControllerErrorHandling:
    """Tests for CameraController error handling paths."""

    def test_sync_capture_primary_error(self):
        """Verifies sync_capture raises SyncCaptureError on primary failure.

        Tests that exceptions in primary camera capture are wrapped in
        SyncCaptureError with appropriate context.

        Arrangement:
            1. Create mock primary with capture_raw raising RuntimeError.
            2. Create mock secondary with successful capture.
            3. Build controller with both mocks.

        Action:
            Call sync_capture with config.

        Assertion Strategy:
            Validates error wrapping by confirming:
            - SyncCaptureError is raised.
            - Error message contains "Primary capture failed".

        Testing Principle:
            Validates error attribution, ensuring failures identify
            which camera caused the problem.
        """
        from unittest.mock import Mock

        mock_primary = Mock()
        mock_primary.capture_raw.side_effect = RuntimeError("Primary failed")
        mock_secondary = Mock()
        mock_secondary.capture_raw.return_value = Mock()

        controller = CameraController(
            cameras={"primary": mock_primary, "secondary": mock_secondary}
        )

        config = SyncCaptureConfig(
            primary="primary",
            secondary="secondary",
            primary_exposure_us=100_000,
            secondary_exposure_us=10_000,
        )

        with pytest.raises(SyncCaptureError) as exc_info:
            controller.sync_capture(config)

        assert "Primary capture failed" in str(exc_info.value)

    def test_sync_capture_secondary_error(self):
        """Verifies sync_capture raises SyncCaptureError on secondary failure.

        Tests that exceptions in secondary camera are wrapped with role context.

        Arrangement:
            1. Create real primary camera via DigitalTwinCameraDriver.
            2. Create mock secondary with capture_raw raising RuntimeError.
            3. Build controller with both.

        Action:
            Call sync_capture with config.

        Assertion Strategy:
            Validates error attribution by confirming:
            - SyncCaptureError is raised.
            - Error message contains "Secondary capture failed".
            - camera_role attribute equals "secondary".
            - original_error attribute is not None.

        Testing Principle:
            Validates structured exceptions, ensuring failures include
            role and original error for debugging.
        """
        from unittest.mock import Mock

        # Use real driver for primary to avoid timing issues
        driver = DigitalTwinCameraDriver()
        primary_cam = Camera(driver, CameraConfig(camera_id=0))
        primary_cam.connect()

        mock_secondary = Mock()
        mock_secondary.capture_raw.side_effect = RuntimeError("Secondary failed")

        controller = CameraController(
            cameras={"primary": primary_cam, "secondary": mock_secondary}
        )

        config = SyncCaptureConfig(
            primary="primary",
            secondary="secondary",
            primary_exposure_us=100_000,
            secondary_exposure_us=10_000,
        )

        with pytest.raises(SyncCaptureError) as exc_info:
            controller.sync_capture(config)

        assert "Secondary capture failed" in str(exc_info.value)
        # Verify structured exception attributes
        assert exc_info.value.camera_role == "secondary"
        assert exc_info.value.original_error is not None

        primary_cam.disconnect()

    def test_sync_capture_none_result(self):
        """Verifies sync_capture raises SyncCaptureError if capture returns None.

        Tests defensive check for None return from capture_raw.

        Arrangement:
            1. Create mock primary with capture_raw returning None.
            2. Create mock secondary with successful capture.
            3. Build controller with both mocks.

        Action:
            Call sync_capture with config.

        Assertion Strategy:
            Validates None handling by confirming:
            - SyncCaptureError is raised.
            - Error message contains "Capture returned None".
            - camera_role attribute equals "primary".

        Testing Principle:
            Validates defensive coding, ensuring None returns are
            caught and reported with context.
        """
        from unittest.mock import Mock

        mock_primary = Mock()
        mock_primary.capture_raw.return_value = None
        mock_secondary = Mock()
        mock_secondary.capture_raw.return_value = Mock()

        controller = CameraController(
            cameras={"primary": mock_primary, "secondary": mock_secondary}
        )

        config = SyncCaptureConfig(
            primary="primary",
            secondary="secondary",
            primary_exposure_us=100_000,
            secondary_exposure_us=10_000,
        )

        with pytest.raises(SyncCaptureError) as exc_info:
            controller.sync_capture(config)

        assert "Capture returned None" in str(exc_info.value)
        # Verify structured exception has camera_role
        assert exc_info.value.camera_role == "primary"

    def test_sync_capture_primary_error_structured(self):
        """Verifies SyncCaptureError has structured attributes on primary failure.

        Tests that exception stores role and original error for programmatic handling.

        Arrangement:
            1. Create mock primary with capture_raw raising RuntimeError.
            2. Create mock secondary with successful capture.
            3. Build controller with both mocks.

        Action:
            Call sync_capture with config.

        Assertion Strategy:
            Validates exception attributes by confirming:
            - camera_role equals "primary".
            - original_error is RuntimeError instance.

        Testing Principle:
            Validates exception richness, enabling callers to
            programmatically determine failure source.
        """
        from unittest.mock import Mock

        mock_primary = Mock()
        mock_primary.capture_raw.side_effect = RuntimeError("HW error")
        mock_secondary = Mock()
        mock_secondary.capture_raw.return_value = Mock()

        controller = CameraController(
            cameras={"primary": mock_primary, "secondary": mock_secondary}
        )

        config = SyncCaptureConfig(
            primary="primary",
            secondary="secondary",
            primary_exposure_us=100_000,
            secondary_exposure_us=10_000,
        )

        with pytest.raises(SyncCaptureError) as exc_info:
            controller.sync_capture(config)

        assert exc_info.value.camera_role == "primary"
        assert isinstance(exc_info.value.original_error, RuntimeError)


# TestCameraRegistry - comprehensive tests for camera_registry.py


class TestCameraRegistry:
    """Test suite for CameraRegistry discovery and singleton management.

    Covers:
    - Discovery with caching and refresh
    - Singleton Camera instances via get()
    - has(), remove(), clear() operations
    - Context manager support
    - driver property access
    - camera_ids and discovered_ids properties
    - __repr__ representation
    """

    def test_discover_returns_camera_info(self):
        """Verifies discover() returns dict of CameraInfo objects.

        Tests that discovery produces typed camera information dictionary.

        Arrangement:
            1. Create DigitalTwinCameraDriver.
            2. Create CameraRegistry with driver.

        Action:
            Call registry.discover().

        Assertion Strategy:
            Validates return structure by confirming:
            - Returns dict type.
            - Contains camera ID 0.
            - Value is CameraInfo instance.
            - CameraInfo.camera_id matches key.

        Testing Principle:
            Validates discovery contract, ensuring typed return
            for consumer enumeration.
        """
        driver = DigitalTwinCameraDriver()
        registry = CameraRegistry(driver)

        cameras = registry.discover()

        assert isinstance(cameras, dict)
        assert 0 in cameras
        assert isinstance(cameras[0], CameraInfo)
        assert cameras[0].camera_id == 0

    def test_discover_caches_results(self):
        """Verifies discover() caches results on subsequent calls.

        Tests that repeated discover() returns cached dict, not new scan.

        Arrangement:
            1. Create DigitalTwinCameraDriver.
            2. Create CameraRegistry with driver.

        Action:
            Call discover() twice consecutively.

        Assertion Strategy:
            Validates caching by confirming:
            - Both calls return same object (is check, not ==).

        Testing Principle:
            Validates performance optimization, ensuring discovery
            doesn't rescan hardware unnecessarily.
        """
        driver = DigitalTwinCameraDriver()
        registry = CameraRegistry(driver)

        cameras1 = registry.discover()
        cameras2 = registry.discover()

        assert cameras1 is cameras2  # Same object, cached

    def test_discover_refresh_clears_cache(self):
        """Verifies discover(refresh=True) forces re-discovery.

        Tests that refresh flag bypasses cache for fresh hardware scan.

        Arrangement:
            1. Create DigitalTwinCameraDriver.
            2. Create CameraRegistry with driver.
            3. Call discover() to populate cache.

        Action:
            Call discover(refresh=True).

        Assertion Strategy:
            Validates refresh behavior by confirming:
            - Returns different dict object (is not check).

        Testing Principle:
            Validates cache invalidation, enabling fresh scan
            when hardware state may have changed.
        """
        driver = DigitalTwinCameraDriver()
        registry = CameraRegistry(driver)

        cameras1 = registry.discover()
        cameras2 = registry.discover(refresh=True)

        # Different dict objects (refreshed)
        assert cameras1 is not cameras2

    def test_get_returns_singleton(self):
        """Verifies get() returns same Camera instance for same ID.

        Tests singleton pattern for camera instances.

        Arrangement:
            1. Create DigitalTwinCameraDriver.
            2. Create CameraRegistry and discover.

        Action:
            Call get(0) twice.

        Assertion Strategy:
            Validates singleton by confirming:
            - Both calls return same object (is check).

        Testing Principle:
            Validates singleton pattern, ensuring single Camera
            instance per hardware device for state consistency.
        """
        driver = DigitalTwinCameraDriver()
        registry = CameraRegistry(driver)
        registry.discover()

        cam1 = registry.get(0)
        cam2 = registry.get(0)

        assert cam1 is cam2

    def test_get_reuses_recovery_strategy(self):
        """Verifies get() reuses the same RecoveryStrategy for all cameras.

        Tests that recovery strategy is shared across camera instances.

        Arrangement:
            1. Create DigitalTwinCameraDriver (returns 2 cameras).
            2. Create CameraRegistry and discover.

        Action:
            Call get(0) then get(1).

        Assertion Strategy:
            Validates strategy reuse by confirming:
            - Both cameras exist and are different instances.
            - Registry._recovery_strategy is not None.

        Testing Principle:
            Validates resource sharing, ensuring single recovery
            coordinator for all cameras.
        """
        driver = DigitalTwinCameraDriver()  # Returns 2 cameras by default
        registry = CameraRegistry(driver)
        registry.discover()

        # First get() creates the recovery strategy
        cam0 = registry.get(0)
        # Second get() for different camera reuses the strategy
        cam1 = registry.get(1)

        # Both cameras should exist and be different instances
        assert cam0 is not cam1
        # Registry should have one shared recovery strategy (internal check)
        assert registry._recovery_strategy is not None

    def test_get_auto_connect(self):
        """Verifies get(auto_connect=True) connects the camera.

        Tests convenience auto-connection during retrieval.

        Arrangement:
            1. Create DigitalTwinCameraDriver.
            2. Create CameraRegistry and discover.

        Action:
            Call get(0, auto_connect=True).

        Assertion Strategy:
            Validates auto-connect by confirming:
            - camera.is_connected is True.

        Testing Principle:
            Validates convenience API, allowing single-call
            retrieval and connection.
        """
        driver = DigitalTwinCameraDriver()
        registry = CameraRegistry(driver)
        registry.discover()

        camera = registry.get(0, auto_connect=True)

        assert camera.is_connected
        camera.disconnect()

    def test_get_with_custom_name(self):
        """Verifies get() accepts custom name on first creation.

        Tests that cameras can be named for identification.

        Arrangement:
            1. Create DigitalTwinCameraDriver.
            2. Create CameraRegistry and discover.

        Action:
            Call get(0, name="CustomName").

        Assertion Strategy:
            Validates naming by confirming:
            - camera.config.name equals "CustomName".

        Testing Principle:
            Validates customization API, enabling meaningful
            camera names for multi-camera systems.
        """
        driver = DigitalTwinCameraDriver()
        registry = CameraRegistry(driver)
        registry.discover()

        camera = registry.get(0, name="CustomName")

        assert camera.config.name == "CustomName"

    def test_get_raises_for_unknown_id(self):
        """Verifies get() raises CameraNotInRegistryError for unknown ID.

        Tests error handling for invalid camera ID.

        Arrangement:
            1. Create DigitalTwinCameraDriver.
            2. Create CameraRegistry and discover.

        Action:
            Call get(99) (non-existent ID).

        Assertion Strategy:
            Validates error handling by confirming:
            - CameraNotInRegistryError is raised.
            - Error message contains the ID "99".
            - Error message contains "Available".

        Testing Principle:
            Validates helpful errors, ensuring users know valid
            IDs when requesting invalid ones.
        """
        driver = DigitalTwinCameraDriver()
        registry = CameraRegistry(driver)
        registry.discover()

        with pytest.raises(CameraNotInRegistryError) as exc_info:
            registry.get(99)

        assert "99" in str(exc_info.value)
        assert "Available" in str(exc_info.value)

    def test_get_triggers_discovery_if_needed(self):
        """Verifies get() calls discover() if not yet run.

        Tests lazy discovery on first get() call.

        Arrangement:
            1. Create DigitalTwinCameraDriver.
            2. Create CameraRegistry (no discover call).

        Action:
            Call get(0) directly.

        Assertion Strategy:
            Validates lazy discovery by confirming:
            - Camera is returned (discovery triggered).
            - is_connected is False (auto_connect default).

        Testing Principle:
            Validates convenience API, enabling get() without
            explicit discover() call.
        """
        driver = DigitalTwinCameraDriver()
        registry = CameraRegistry(driver)
        # No discover() call

        camera = registry.get(0)  # Should trigger discover internally

        assert camera is not None
        assert 0 in registry.discovered_ids  # At least camera 0 discovered

    def test_has_returns_true_for_existing(self):
        """Verifies has() returns True for cameras created via get().

        Tests that has() accurately reports active camera presence.

        Arrangement:
            1. Create DigitalTwinCameraDriver.
            2. Create CameraRegistry and discover.
            3. Call get(0) to create Camera instance.

        Action:
            Call has(0).

        Assertion Strategy:
            Validates presence check by confirming:
            - Returns True for ID with created Camera.

        Testing Principle:
            Validates introspection API, enabling callers to
            check if camera is already instantiated.
        """
        driver = DigitalTwinCameraDriver()
        registry = CameraRegistry(driver)
        registry.discover()
        registry.get(0)

        assert registry.has(0) is True

    def test_has_returns_false_for_missing(self):
        """Verifies has() returns False for cameras not created.

        Tests that has() distinguishes discovered vs instantiated.

        Arrangement:
            1. Create DigitalTwinCameraDriver.
            2. Create CameraRegistry and discover.
            3. Do NOT call get() - cameras only discovered.

        Action:
            Call has(0) and has(99).

        Assertion Strategy:
            Validates negative check by confirming:
            - Returns False for discovered but not created (0).
            - Returns False for non-existent ID (99).

        Testing Principle:
            Validates distinction between discovery and
            instantiation states.
        """
        driver = DigitalTwinCameraDriver()
        registry = CameraRegistry(driver)
        registry.discover()

        assert registry.has(0) is False  # Not yet created
        assert registry.has(99) is False

    def test_remove_returns_camera(self):
        """Verifies remove() returns the removed Camera instance.

        Tests that removal returns camera for caller cleanup.

        Arrangement:
            1. Create DigitalTwinCameraDriver.
            2. Create CameraRegistry and discover.
            3. Call get(0) to create Camera instance.

        Action:
            Call remove(0).

        Assertion Strategy:
            Validates removal by confirming:
            - Returns same Camera instance.
            - has(0) returns False after removal.

        Testing Principle:
            Validates removal API, returning camera for
            optional disconnect by caller.
        """
        driver = DigitalTwinCameraDriver()
        registry = CameraRegistry(driver)
        registry.discover()
        camera = registry.get(0)

        removed = registry.remove(0)

        assert removed is camera
        assert registry.has(0) is False

    def test_remove_returns_none_for_missing(self):
        """Verifies remove() returns None for non-existent ID.

        Tests graceful handling of removal for missing camera.

        Arrangement:
            1. Create DigitalTwinCameraDriver.
            2. Create CameraRegistry (no discover/get).

        Action:
            Call remove(99).

        Assertion Strategy:
            Validates graceful failure by confirming:
            - Returns None (not exception).

        Testing Principle:
            Validates defensive API, enabling idempotent
            removal without error checking.
        """
        driver = DigitalTwinCameraDriver()
        registry = CameraRegistry(driver)

        removed = registry.remove(99)

        assert removed is None

    def test_camera_ids_property(self):
        """Verifies camera_ids returns list of active camera IDs.

        Tests that property shows only instantiated cameras.

        Arrangement:
            1. Create DigitalTwinCameraDriver.
            2. Create CameraRegistry and discover.
            3. Call get(0) to create one Camera.

        Action:
            Access camera_ids property.

        Assertion Strategy:
            Validates active list by confirming:
            - Returns [0] (only created camera).

        Testing Principle:
            Validates introspection API, showing which cameras
            have been instantiated (not just discovered).
        """
        driver = DigitalTwinCameraDriver()
        registry = CameraRegistry(driver)
        registry.discover()
        registry.get(0)

        assert registry.camera_ids == [0]

    def test_discovered_ids_property(self):
        """Verifies discovered_ids returns list from discovery.

        Tests that property shows all discovered cameras.

        Arrangement:
            1. Create DigitalTwinCameraDriver.
            2. Create CameraRegistry and discover.

        Action:
            Access discovered_ids property.

        Assertion Strategy:
            Validates discovery list by confirming:
            - Contains 0 (DigitalTwin always has camera 0).

        Testing Principle:
            Validates introspection API, showing available
            cameras from hardware scan.
        """
        driver = DigitalTwinCameraDriver()
        registry = CameraRegistry(driver)
        registry.discover()

        assert 0 in registry.discovered_ids

    def test_discovered_ids_empty_before_discover(self):
        """Verifies discovered_ids returns empty list before discover().

        Tests initial state before discovery.

        Arrangement:
            1. Create DigitalTwinCameraDriver.
            2. Create CameraRegistry (no discover call).

        Action:
            Access discovered_ids property.

        Assertion Strategy:
            Validates initial state by confirming:
            - Returns empty list.

        Testing Principle:
            Validates safe access, ensuring property works
            before any operations.
        """
        driver = DigitalTwinCameraDriver()
        registry = CameraRegistry(driver)

        assert registry.discovered_ids == []

    def test_driver_property(self):
        """Verifies driver property returns injected driver.

        Tests that driver is accessible for advanced operations.

        Arrangement:
            1. Create DigitalTwinCameraDriver.
            2. Create CameraRegistry with driver.

        Action:
            Access driver property.

        Assertion Strategy:
            Validates property by confirming:
            - Returns same driver instance (is check).

        Testing Principle:
            Validates DI access, enabling callers to access
            underlying driver for low-level operations.
        """
        driver = DigitalTwinCameraDriver()
        registry = CameraRegistry(driver)

        assert registry.driver is driver

    def test_clear_disconnects_cameras(self):
        """Verifies clear() disconnects all connected cameras.

        Tests cleanup disconnects and clears all state.

        Arrangement:
            1. Create DigitalTwinCameraDriver.
            2. Create CameraRegistry and discover.
            3. Get and auto-connect camera 0.

        Action:
            Call clear().

        Assertion Strategy:
            Validates cleanup by confirming:
            - camera.is_connected is False.
            - camera_ids is empty.
            - discovered_ids is empty.

        Testing Principle:
            Validates resource cleanup, ensuring all cameras
            disconnected and state reset.
        """
        driver = DigitalTwinCameraDriver()
        registry = CameraRegistry(driver)
        registry.discover()
        camera = registry.get(0, auto_connect=True)

        assert camera.is_connected
        registry.clear()

        assert not camera.is_connected
        assert registry.camera_ids == []
        assert registry.discovered_ids == []

    def test_clear_skips_unconnected_cameras(self):
        """Verifies clear() skips cameras that are not connected.

        Tests safe cleanup of disconnected cameras.

        Arrangement:
            1. Create DigitalTwinCameraDriver.
            2. Create CameraRegistry and discover.
            3. Get camera 0 (not connected).

        Action:
            Call clear().

        Assertion Strategy:
            Validates safe cleanup by confirming:
            - No exception raised.
            - camera_ids is empty.
            - discovered_ids is empty.

        Testing Principle:
            Validates defensive cleanup, not attempting
            disconnect on already-disconnected cameras.
        """
        driver = DigitalTwinCameraDriver()
        registry = CameraRegistry(driver)
        registry.discover()
        camera = registry.get(0)  # Not connected

        assert not camera.is_connected
        registry.clear()  # Should not try to disconnect

        assert registry.camera_ids == []
        assert registry.discovered_ids == []

    def test_clear_handles_disconnect_errors(self):
        """Verifies clear() handles disconnect exceptions gracefully.

        Tests best-effort cleanup continues despite errors.

        Arrangement:
            1. Create registry with connected camera.
            2. Mock camera.disconnect to raise RuntimeError.

        Action:
            Call clear().

        Assertion Strategy:
            Validates error handling by confirming:
            - No exception propagated.
            - Best-effort cleanup completes.

        Testing Principle:
            Validates resilient cleanup, ensuring one camera's
            disconnect failure doesn't prevent others.
        """
        from unittest.mock import Mock

        driver = DigitalTwinCameraDriver()
        registry = CameraRegistry(driver)
        registry.discover()
        camera = registry.get(0, auto_connect=True)

        # Mock disconnect to raise exception
        original_disconnect = camera.disconnect
        camera.disconnect = Mock(side_effect=RuntimeError("Disconnect failed"))

        # Should not raise - best effort cleanup
        registry.clear()

        # Restore for cleanup
        camera.disconnect = original_disconnect

    def test_context_manager(self):
        """Verifies context manager enters and exits correctly.

        Tests with statement auto-cleanup on exit.

        Arrangement:
            1. Create DigitalTwinCameraDriver.
            2. Use CameraRegistry in with block.
            3. Discover and connect camera inside block.

        Action:
            Exit with block.

        Assertion Strategy:
            Validates context manager by confirming:
            - Camera connected inside block.
            - Camera disconnected after exit.

        Testing Principle:
            Validates RAII pattern, ensuring automatic
            cleanup on scope exit.
        """
        driver = DigitalTwinCameraDriver()

        with CameraRegistry(driver) as registry:
            registry.discover()
            camera = registry.get(0, auto_connect=True)
            assert camera.is_connected

        # After exit, camera should be disconnected
        assert not camera.is_connected

    def test_repr_before_discover(self):
        """Verifies __repr__ shows 0 discovered before discover().

        Tests repr output for initial state.

        Arrangement:
            1. Create DigitalTwinCameraDriver.
            2. Create CameraRegistry (no operations).

        Action:
            Call repr(registry).

        Assertion Strategy:
            Validates repr format by confirming:
            - Contains "discovered=0".
            - Contains "active=0".

        Testing Principle:
            Validates debugging output, showing meaningful
            state for logging and debugging.
        """
        driver = DigitalTwinCameraDriver()
        registry = CameraRegistry(driver)

        result = repr(registry)

        assert "discovered=0" in result
        assert "active=0" in result

    def test_repr_after_discover_and_get(self):
        """Verifies __repr__ shows correct counts after operations.

        Tests repr reflects discover and get state.

        Arrangement:
            1. Create DigitalTwinCameraDriver.
            2. Create CameraRegistry.
            3. Call discover() and get(0).

        Action:
            Call repr(registry).

        Assertion Strategy:
            Validates repr format by confirming:
            - Contains "discovered=2" (DigitalTwin default).
            - Contains "active=1" (one get() called).

        Testing Principle:
            Validates debugging output, showing accurate
            counts for diagnostics.
        """
        driver = DigitalTwinCameraDriver()
        registry = CameraRegistry(driver)
        registry.discover()
        registry.get(0)

        result = repr(registry)

        # DigitalTwinCameraDriver creates 2 cameras by default
        assert "discovered=2" in result
        assert "active=1" in result

    def test_discover_handles_dict_from_driver(self):
        """Verifies discover() converts dict to CameraInfo when driver returns dict.

        Tests compatibility with legacy driver format.

        Arrangement:
            1. Create mock driver returning dict instead of CameraInfo.
            2. Create CameraRegistry with mock driver.

        Action:
            Call discover().

        Assertion Strategy:
            Validates dict conversion by confirming:
            - Result contains camera ID 0.
            - Value is CameraInfo instance (converted).

        Testing Principle:
            Validates backward compatibility, supporting drivers
            that return raw dicts.
        """
        from unittest.mock import Mock

        mock_driver = Mock()
        # Return dict instead of CameraInfo (legacy driver format)
        mock_driver.get_connected_cameras.return_value = {
            0: {
                "name": "TestCam",
                "max_width": 1920,
                "max_height": 1080,
                "is_color": True,
                "bayer_pattern": "RGGB",
                "supported_bins": [1, 2],
                "controls": {"gain": {"min": 0, "max": 100}},
            }
        }

        registry = CameraRegistry(mock_driver)
        cameras = registry.discover()

        assert 0 in cameras
        assert isinstance(cameras[0], CameraInfo)
        assert cameras[0].name == "TestCam"
        assert cameras[0].max_width == 1920
        assert cameras[0].is_color is True


class TestRecoveryStrategy:
    """Test suite for RecoveryStrategy USB recovery mechanism."""

    def test_attempt_recovery_returns_true_on_success(self):
        """Verifies attempt_recovery returns True when camera found.

        Tests successful USB recovery scenario.

        Arrangement:
            1. Create DigitalTwinCameraDriver.
            2. Create CameraRegistry and discover.
            3. Create RecoveryStrategy with registry.

        Action:
            Call attempt_recovery(0) for existing camera.

        Assertion Strategy:
            Validates success by confirming:
            - Returns True (camera found in registry).

        Testing Principle:
            Validates recovery mechanism, returning True when
            camera can be re-discovered after USB reset.
        """
        driver = DigitalTwinCameraDriver()
        registry = CameraRegistry(driver)
        registry.discover()
        strategy = RecoveryStrategy(registry)

        result = strategy.attempt_recovery(0)

        assert result is True

    def test_attempt_recovery_returns_false_on_failure(self):
        """Verifies attempt_recovery returns False when camera not found.

        Tests failed USB recovery scenario.

        Arrangement:
            1. Create DigitalTwinCameraDriver.
            2. Create CameraRegistry and discover.
            3. Create RecoveryStrategy with registry.

        Action:
            Call attempt_recovery(99) for non-existent camera.

        Assertion Strategy:
            Validates failure by confirming:
            - Returns False (camera not in registry).

        Testing Principle:
            Validates recovery mechanism, returning False when
            camera cannot be found after USB reset.
        """
        driver = DigitalTwinCameraDriver()
        registry = CameraRegistry(driver)
        registry.discover()
        strategy = RecoveryStrategy(registry)

        result = strategy.attempt_recovery(99)  # Non-existent camera

        assert result is False


class TestNullRecoveryStrategy:
    """Test suite for NullRecoveryStrategy no-op implementation."""

    def test_attempt_recovery_always_returns_false(self):
        """Verifies NullRecoveryStrategy always returns False.

        Tests no-op recovery for cameras without recovery support.

        Arrangement:
            1. Create NullRecoveryStrategy.

        Action:
            Call attempt_recovery with various IDs.

        Assertion Strategy:
            Validates no-op by confirming:
            - Returns False for ID 0.
            - Returns False for ID 99.
            - No exceptions raised.

        Testing Principle:
            Validates null object pattern, providing safe no-op
            for cameras that don't support USB recovery.
        """
        strategy = NullRecoveryStrategy()

        assert strategy.attempt_recovery(0) is False
        assert strategy.attempt_recovery(99) is False


class TestModuleLevelRegistry:
    """Test suite for module-level registry functions."""

    def test_init_and_get_registry(self):
        """Verifies init_registry and get_registry work together.

        Tests module-level singleton initialization.

        Arrangement:
            1. Create DigitalTwinCameraDriver.

        Action:
            Call init_registry then get_registry.

        Assertion Strategy:
            Validates singleton by confirming:
            - get_registry returns same instance as init_registry.

        Testing Principle:
            Validates module-level singleton, ensuring single
            registry instance for global camera management.
        """
        driver = DigitalTwinCameraDriver()

        try:
            registry = init_registry(driver)
            retrieved = get_registry()

            assert retrieved is registry
        finally:
            shutdown_registry()

    def test_get_registry_raises_before_init(self):
        """Verifies get_registry raises RuntimeError before init.

        Tests guard against accessing uninitialized registry.

        Arrangement:
            1. Ensure clean state via shutdown_registry().

        Action:
            Call get_registry() before init_registry().

        Assertion Strategy:
            Validates guard by confirming:
            - RuntimeError is raised.
            - Error message contains "not initialized".

        Testing Principle:
            Validates fail-fast, ensuring clear error when
            accessing registry before initialization.
        """
        # Ensure clean state
        shutdown_registry()

        with pytest.raises(RuntimeError) as exc_info:
            get_registry()

        assert "not initialized" in str(exc_info.value)

    def test_shutdown_registry_clears_state(self):
        """Verifies shutdown_registry clears module-level registry.

        Tests cleanup of global singleton.

        Arrangement:
            1. Initialize registry with driver.

        Action:
            Call shutdown_registry().

        Assertion Strategy:
            Validates cleanup by confirming:
            - get_registry() raises RuntimeError after shutdown.

        Testing Principle:
            Validates resource cleanup, ensuring registry state
            is cleared for clean restart.
        """
        driver = DigitalTwinCameraDriver()
        init_registry(driver)

        shutdown_registry()

        with pytest.raises(RuntimeError):
            get_registry()

    def test_shutdown_registry_idempotent(self):
        """Verifies shutdown_registry can be called multiple times.

        Tests safe repeated shutdown calls.

        Arrangement:
            1. No initialization (clean state).

        Action:
            Call shutdown_registry() twice.

        Assertion Strategy:
            Validates idempotence by confirming:
            - No exception raised on either call.

        Testing Principle:
            Validates defensive API, enabling multiple shutdown
            calls without tracking initialization state.
        """
        # Should not raise even when not initialized
        shutdown_registry()
        shutdown_registry()


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
