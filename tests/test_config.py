"""Tests for driver configuration and digital twin."""

import pytest

from telescope_mcp.drivers import (
    DriverConfig,
    DriverFactory,
    DriverMode,
    configure,
    get_factory,
    use_digital_twin,
)
from telescope_mcp.drivers.cameras import DigitalTwinCameraDriver


class TestDriverConfig:
    """Test suite for DriverConfig configuration management.

    Categories:
    1. Default Behavior - Safe-by-default mode (1 test)
    2. Global Configuration - Runtime mode switching (1 test)

    Total: 2 tests.
    """

    def test_default_mode_is_digital_twin(self):
        """Verifies DriverConfig defaults to DIGITAL_TWIN mode for safety.

        Tests default initialization by creating DriverConfig without
        arguments.

        Business context:
        Ensures safe-by-default behavior preventing accidental hardware
        activation during development or testing.

        Arrangement:
        1. Create DriverConfig() with no arguments.
        2. No mode parameter provided, should use default.

        Action:
        Access config.mode property to check default value.

        Assertion Strategy:
        Validates safe defaults by confirming:
        - config.mode equals DriverMode.DIGITAL_TWIN.
        - Hardware mode requires explicit opt-in.

        Testing Principle:
        Validates fail-safe design, ensuring system defaults to
        simulation rather than potentially dangerous hardware control."""
        config = DriverConfig()
        assert config.mode == DriverMode.DIGITAL_TWIN

    def test_configure_changes_factory(self):
        """Verifies configure() updates global DriverFactory singleton.

        Tests global configuration mechanism by switching between modes
        and observing factory state changes.

        Business context:
        Enables runtime mode switching for testing workflows that start
        with simulation and transition to hardware.

        Arrangement:
        1. Call use_digital_twin() to establish known initial state.
        2. Retrieve factory via get_factory() to check initial mode.
        3. Create new DriverConfig with HARDWARE mode.

        Action:
        Call configure() with hardware config, then get_factory() again,
        then reset to digital twin.

        Assertion Strategy:
        Validates configuration propagation by confirming:
        - Initial factory mode is DIGITAL_TWIN.
        - After configure(HARDWARE), factory mode is HARDWARE.
        - Factory singleton updates globally.

        Testing Principle:
        Validates global state management, ensuring configure() provides
        consistent mode switching across application."""
        use_digital_twin()
        assert get_factory().config.mode == DriverMode.DIGITAL_TWIN

        configure(DriverConfig(mode=DriverMode.HARDWARE))
        assert get_factory().config.mode == DriverMode.HARDWARE

        # Reset to digital twin
        use_digital_twin()


class TestDigitalTwinCameraDriver:
    """Test suite for DigitalTwinCameraDriver simulation.

    Categories:
    1. Camera Discovery - List simulated devices (1 test)
    2. Camera Instantiation - Open/create instances (1 test)
    3. Error Handling - Invalid camera IDs (1 test)

    Total: 3 tests.
    """

    def test_list_cameras(self):
        """Verifies DigitalTwinCameraDriver returns simulated camera list.

        Tests camera discovery by calling get_connected_cameras() on
        digital twin driver.

        Business context:
        Provides predictable camera list for development and testing
        without requiring physical hardware.

        Arrangement:
        1. Create DigitalTwinCameraDriver instance.
        2. Driver initializes with predefined camera configuration.

        Action:
        Call driver.get_connected_cameras() to retrieve camera dict.

        Assertion Strategy:
        Validates simulated hardware by confirming:
        - Camera IDs 0 and 1 present in returned dict.
        - Camera 0 has Name containing b"ASI120MC-S".
        - Mimics real ASI camera SDK response format.

        Testing Principle:
        Validates simulation fidelity, ensuring digital twin provides
        realistic camera list matching hardware driver interface."""
        driver = DigitalTwinCameraDriver()
        cameras = driver.get_connected_cameras()
        assert 0 in cameras
        assert 1 in cameras
        assert b"ASI120MC-S" in cameras[0]["Name"]

    def test_open_camera(self):
        """Verifies DigitalTwinCameraDriver.open() returns valid camera instance.

        Tests camera instantiation by opening camera 0 and checking
        protocol compliance.

        Business context:
        Ensures digital twin provides functional camera instances for
        testing camera-dependent code paths.

        Arrangement:
        1. Create DigitalTwinCameraDriver instance.
        2. Camera 0 exists in simulated device list.

        Action:
        Call driver.open(0) to instantiate camera object.

        Assertion Strategy:
        Validates instance creation by confirming:
        - Returned camera is not None.
        - Camera has get_info method (CameraInstance protocol).
        - Camera has capture method (CameraInstance protocol).

        Testing Principle:
        Validates interface compliance, ensuring digital twin camera
        implements required CameraInstance protocol methods."""
        driver = DigitalTwinCameraDriver()
        camera = driver.open(0)
        assert camera is not None
        # Check that it implements CameraInstance protocol
        assert hasattr(camera, "get_info")
        assert hasattr(camera, "capture")

    def test_open_invalid_camera(self):
        """Verifies DigitalTwinCameraDriver.open() raises ValueError for invalid ID.

        Tests error handling by attempting to open non-existent camera.

        Business context:
        Ensures digital twin mimics real hardware error behavior for
        invalid camera IDs.

        Arrangement:
        1. Create DigitalTwinCameraDriver instance.
        2. Camera ID 99 does not exist (only 0 and 1 defined).

        Action:
        Call driver.open(99) with invalid camera ID.

        Assertion Strategy:
        Validates error handling by confirming:
        - ValueError is raised.
        - Matches real ASI driver behavior for missing cameras.

        Testing Principle:
        Validates error path simulation, ensuring digital twin raises
        same exceptions as hardware driver for consistent testing."""
        driver = DigitalTwinCameraDriver()
        with pytest.raises(ValueError):
            driver.open(99)


class TestDigitalTwinCameraInstance:
    """Test suite for DigitalTwinCameraInstance operations.

    Categories:
    1. Metadata Retrieval - Camera info and controls (2 tests)
    2. Control Management - Set/get control values (1 test)
    3. Image Capture - JPEG generation (1 test)

    Total: 4 tests.
    """

    def test_get_info(self):
        """Verifies camera instance returns complete info dictionary.

        Tests camera metadata retrieval by calling get_info() on
        opened digital twin camera.

        Business context:
        Provides camera specifications needed for UI display and
        capture parameter validation.

        Arrangement:
        1. Create DigitalTwinCameraDriver instance.
        2. Open camera 0 to get camera instance.

        Action:
        Call camera.get_info() to retrieve camera metadata.

        Assertion Strategy:
        Validates info completeness by confirming:
        - Info dict contains "MaxWidth" key.
        - Info dict contains "MaxHeight" key.
        - Mimics ASI SDK info structure.

        Testing Principle:
        Validates metadata provision, ensuring camera exposes
        necessary specifications for downstream use."""
        driver = DigitalTwinCameraDriver()
        camera = driver.open(0)
        info = camera.get_info()
        assert "MaxWidth" in info
        assert "MaxHeight" in info

    def test_get_controls(self):
        """Verifies camera instance returns available control parameters.

        Tests control discovery by calling get_controls() on digital
        twin camera instance.

        Business context:
        Enables dynamic UI generation and parameter validation by
        discovering supported camera controls.

        Arrangement:
        1. Create DigitalTwinCameraDriver instance.
        2. Open camera 0 to get camera instance.

        Action:
        Call camera.get_controls() to retrieve control dict.

        Assertion Strategy:
        Validates control availability by confirming:
        - Controls dict contains "ASI_GAIN" key.
        - Controls dict contains "ASI_EXPOSURE" key.
        - Matches ASI SDK control naming convention.

        Testing Principle:
        Validates control exposure, ensuring camera provides
        discoverable parameter interfaces for client code."""
        driver = DigitalTwinCameraDriver()
        camera = driver.open(0)
        controls = camera.get_controls()
        assert "ASI_GAIN" in controls
        assert "ASI_EXPOSURE" in controls

    def test_set_and_get_control(self):
        """Verifies camera control set/get round-trip preserves values.

        Tests control state management by setting gain value and
        reading it back.

        Business context:
        Validates control persistence for exposure/gain adjustment
        workflows used in astrophotography.

        Arrangement:
        1. Create DigitalTwinCameraDriver instance.
        2. Open camera 0 to get camera instance.
        3. Camera starts with default control values.

        Action:
        Set ASI_GAIN to 75, then immediately read back with get_control().

        Assertion Strategy:
        Validates state persistence by confirming:
        - get_control() result["value"] equals 75.
        - Set value successfully stored and retrieved.

        Testing Principle:
        Validates stateful behavior, ensuring camera maintains
        control settings across get/set operations."""
        driver = DigitalTwinCameraDriver()
        camera = driver.open(0)

        camera.set_control("ASI_GAIN", 75)
        result = camera.get_control("ASI_GAIN")
        assert result["value"] == 75

    def test_capture_returns_jpeg(self):
        """Verifies camera.capture() returns valid JPEG image data.

        Tests image capture by calling capture() and validating
        JPEG format.

        Business context:
        Ensures digital twin produces realistic image data for testing
        image processing and display pipelines.

        Arrangement:
        1. Create DigitalTwinCameraDriver instance.
        2. Open camera 0 to get camera instance.
        3. Configure exposure_us=100000 for capture.

        Action:
        Call camera.capture(exposure_us=100000) to get image bytes.

        Assertion Strategy:
        Validates JPEG output by confirming:
        - Result is bytes object (not None or other type).
        - Length > 0 (non-empty image data).
        - First two bytes are \xff\xd8 (JPEG magic number).

        Testing Principle:
        Validates output format compliance, ensuring capture produces
        valid JPEG data matching real camera output."""
        driver = DigitalTwinCameraDriver()
        camera = driver.open(0)

        jpeg = camera.capture(exposure_us=100000)
        assert isinstance(jpeg, bytes)
        assert len(jpeg) > 0
        # JPEG magic bytes
        assert jpeg[:2] == b"\xff\xd8"


class TestDriverFactory:
    """Test suite for DriverFactory driver instantiation.

    Categories:
    1. Camera Driver Creation - Digital twin and hardware modes (2 tests)
    2. Motor Controller Creation - Simulation mode (1 test)
    3. Position Sensor Creation - Simulation mode (1 test)

    Total: 4 tests.
    """

    def test_create_camera_driver_digital_twin(self):
        """Verifies DriverFactory creates DigitalTwinCameraDriver in simulation mode.

        Tests factory instantiation by requesting camera driver in
        DIGITAL_TWIN mode.

        Business context:
        Enables safe development and testing without hardware by
        providing simulated camera drivers.

        Arrangement:
        1. Create DriverFactory with DriverConfig(mode=DIGITAL_TWIN).
        2. Factory configured for simulation mode.

        Action:
        Call factory.create_camera_driver() to instantiate driver.

        Assertion Strategy:
        Validates factory routing by confirming:
        - Returned driver is instance of DigitalTwinCameraDriver.
        - Factory creates correct driver type for mode.

        Testing Principle:
        Validates factory pattern implementation, ensuring mode
        correctly determines driver instantiation."""
        factory = DriverFactory(DriverConfig(mode=DriverMode.DIGITAL_TWIN))
        driver = factory.create_camera_driver()
        assert isinstance(driver, DigitalTwinCameraDriver)

    def test_create_camera_driver_hardware_mode(self):
        """Verifies DriverFactory creates ASICameraDriver in hardware mode.

        Tests factory instantiation by requesting camera driver in
        HARDWARE mode.

        Business context:
        Enables production operation with real ZWO ASI cameras by
        instantiating hardware drivers.

        Arrangement:
        1. Import ASICameraDriver for type checking.
        2. Create DriverFactory with DriverConfig(mode=HARDWARE).
        3. Factory configured for real hardware operation.

        Action:
        Call factory.create_camera_driver() to instantiate driver.

        Assertion Strategy:
        Validates factory routing by confirming:
        - Returned driver is instance of ASICameraDriver.
        - Factory creates hardware driver for HARDWARE mode.

        Testing Principle:
        Validates mode-based dispatch, ensuring factory routes to
        appropriate driver implementation based on configuration."""
        from telescope_mcp.drivers.cameras import ASICameraDriver

        factory = DriverFactory(DriverConfig(mode=DriverMode.HARDWARE))
        driver = factory.create_camera_driver()
        assert isinstance(driver, ASICameraDriver)

    def test_create_motor_controller_digital_twin(self):
        """Verifies DriverFactory creates StubMotorController in simulation mode.

        Tests factory instantiation by requesting motor controller in
        DIGITAL_TWIN mode.

        Business context:
        Enables telescope motion testing without physical motors by
        providing simulated controller.

        Arrangement:
        1. Import StubMotorController for type checking.
        2. Create DriverFactory with DriverConfig(mode=DIGITAL_TWIN).
        3. Factory configured for simulation mode.

        Action:
        Call factory.create_motor_controller() to instantiate controller.

        Assertion Strategy:
        Validates factory routing by confirming:
        - Returned controller is instance of StubMotorController.
        - Factory creates stub for simulation mode.

        Testing Principle:
        Validates consistent factory behavior, ensuring motor
        controllers follow same mode-based instantiation pattern."""
        from telescope_mcp.drivers.motors import StubMotorController

        factory = DriverFactory(DriverConfig(mode=DriverMode.DIGITAL_TWIN))
        controller = factory.create_motor_controller()
        assert isinstance(controller, StubMotorController)

    def test_create_sensor_driver_digital_twin(self):
        """Verifies DriverFactory creates DigitalTwinSensorDriver in simulation mode.

        Tests factory instantiation by requesting sensor driver in
        DIGITAL_TWIN mode.

        Business context:
        Enables telescope positioning tests without hardware by
        providing simulated sensor feedback.

        Arrangement:
        1. Import DigitalTwinSensorDriver for type checking.
        2. Create DriverFactory with DriverConfig(mode=DIGITAL_TWIN).
        3. Factory configured for simulation mode.

        Action:
        Call factory.create_sensor_driver() to instantiate driver.

        Assertion Strategy:
        Validates factory routing by confirming:
        - Returned driver is instance of DigitalTwinSensorDriver.
        - Factory creates digital twin for simulation mode.

        Testing Principle:
        Validates factory completeness, ensuring all driver types
        support mode-based instantiation pattern."""
        from telescope_mcp.drivers.sensors import DigitalTwinSensorDriver

        factory = DriverFactory(DriverConfig(mode=DriverMode.DIGITAL_TWIN))
        driver = factory.create_sensor_driver()
        assert isinstance(driver, DigitalTwinSensorDriver)
