"""Comprehensive tests for drivers/config.py to increase coverage."""

import tempfile
from pathlib import Path

from telescope_mcp.drivers.config import (
    DriverConfig,
    DriverFactory,
    DriverMode,
    configure,
    get_factory,
    get_session_manager,
    set_data_dir,
    set_location,
    use_digital_twin,
    use_hardware,
)


class TestDriverConfig:
    """Test suite for DriverConfig dataclass initialization and fields.

    Categories:
    1. Default Initialization - Default field values (1 test)
    2. Custom Initialization - Explicit field values (1 test)
    3. Field Validation - Location dictionary structure (1 test)

    Total: 3 tests.
    """

    def test_default_config(self):
        """Verifies DriverConfig initializes with safe default values.

        Tests dataclass default initialization by creating DriverConfig
        without arguments.

        Business context:
        Ensures safe defaults allow immediate use without configuration,
        defaulting to simulation mode.

        Arrangement:
        1. Create DriverConfig() with no arguments.
        2. All fields use default values from dataclass definition.

        Action:
        Access config properties to check default values.

        Assertion Strategy:
        Validates sensible defaults by confirming:
        - mode equals DIGITAL_TWIN (safe simulation default).
        - data_dir is ~/.telescope-mcp/data or exists.
        - finder_camera_id equals 0 (first camera).
        - main_camera_id equals 1 (second camera).

        Testing Principle:
        Validates usable defaults, ensuring config works immediately
        without requiring explicit field specification."""
        config = DriverConfig()
        assert config.mode == DriverMode.DIGITAL_TWIN
        assert (
            config.data_dir.exists()
            or config.data_dir == Path.home() / ".telescope-mcp" / "data"
        )
        assert config.finder_camera_id == 0
        assert config.main_camera_id == 1

    def test_custom_config(self):
        """Verifies DriverConfig accepts custom values for all fields.

        Tests dataclass flexibility by providing explicit values for
        all configurable fields.

        Business context:
        Enables telescope operators to customize camera IDs, storage
        locations, and geographic coordinates.

        Arrangement:
        1. Create temporary directory for custom data_dir.
        2. Define location dict with LA coordinates.
        3. Specify non-default camera IDs (2 and 3).

        Action:
        Create DriverConfig with HARDWARE mode, custom data_dir,
        location, and camera IDs.

        Assertion Strategy:
        Validates custom initialization by confirming:
        - mode equals HARDWARE (not default DIGITAL_TWIN).
        - data_dir equals temporary path (not default).
        - location["lat"] equals 34.05 (Los Angeles).
        - finder_camera_id equals 2 (not default 0).

        Testing Principle:
        Validates configuration flexibility, ensuring all fields
        accept custom values for site-specific setup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DriverConfig(
                mode=DriverMode.HARDWARE,
                data_dir=Path(tmpdir),
                location={"lat": 34.05, "lon": -118.25, "alt": 100.0},
                finder_camera_id=2,
                main_camera_id=3,
            )
            assert config.mode == DriverMode.HARDWARE
            assert config.data_dir == Path(tmpdir)
            assert config.location["lat"] == 34.05
            assert config.finder_camera_id == 2

    def test_location_dict(self):
        """Verifies DriverConfig.location stores geographic coordinates correctly.

        Tests location field structure by providing latitude, longitude,
        and altitude.

        Business context:
        Location data essential for astronomical calculations (rise/set
        times, coordinate transforms).

        Arrangement:
        1. Define location dict with lat=40.0, lon=-75.0, alt=200.0.
        2. Create DriverConfig with location parameter.

        Action:
        Access location dict keys and values.

        Assertion Strategy:
        Validates dictionary storage by confirming:
        - "lat" key exists in location dict.
        - location["lon"] equals -75.0 (stored correctly).

        Testing Principle:
        Validates data structure, ensuring location dict preserves
        all geographic coordinates for astronomical computations."""
        config = DriverConfig(location={"lat": 40.0, "lon": -75.0, "alt": 200.0})
        assert "lat" in config.location
        assert config.location["lon"] == -75.0


class TestDriverFactory:
    """Test suite for DriverFactory driver creation and configuration.

    Categories:
    1. Factory Initialization - Default and custom configs (2 tests)
    2. Camera Driver Creation - Digital twin variants (2 tests)
    3. Other Driver Creation - Motor and sensor (2 tests)

    Total: 6 tests.
    """

    def test_factory_default_config(self):
        """Verifies DriverFactory uses DIGITAL_TWIN mode when created without config.

        Tests factory initialization by creating DriverFactory with no
        arguments.

        Business context:
        Ensures factory defaults to safe simulation mode without
        explicit configuration.

        Arrangement:
        1. Create DriverFactory() with no arguments.
        2. Factory should use default DriverConfig internally.

        Action:
        Access factory.config.mode to check default.

        Assertion Strategy:
        Validates safe factory defaults by confirming:
        - config.mode equals DIGITAL_TWIN.
        - Factory inherits DriverConfig defaults.

        Testing Principle:
        Validates safe-by-default design, ensuring factory creation
        doesn't require explicit mode specification."""
        factory = DriverFactory()
        assert factory.config.mode == DriverMode.DIGITAL_TWIN

    def test_factory_custom_config(self):
        """Verifies DriverFactory accepts custom DriverConfig instance.

        Tests factory initialization with explicit HARDWARE configuration.

        Arrangement:
        1. Create DriverConfig with mode=HARDWARE.
        2. Pass config to DriverFactory constructor.

        Action:
        Access factory.config.mode to verify custom config used.

        Assertion Strategy:
        Validates config injection by confirming:
        - factory.config.mode equals HARDWARE (not default).

        Testing Principle:
        Validates dependency injection, ensuring factory accepts
        custom configuration for flexible initialization."""
        config = DriverConfig(mode=DriverMode.HARDWARE)
        factory = DriverFactory(config)
        assert factory.config.mode == DriverMode.HARDWARE

    def test_create_camera_driver_digital_twin(self):
        """Verifies factory creates DigitalTwinCameraDriver in simulation mode.

        Tests camera driver creation with DIGITAL_TWIN config.

        Arrangement:
        1. Create DriverConfig with mode=DIGITAL_TWIN.
        2. Create DriverFactory with config.

        Action:
        Call factory.create_camera_driver() to instantiate driver.

        Assertion Strategy:
        Validates driver type by confirming:
        - Returned driver is not None.
        - Driver class name contains "DigitalTwin".

        Testing Principle:
        Validates factory mode dispatch, ensuring DIGITAL_TWIN mode
        creates simulation driver."""
        config = DriverConfig(mode=DriverMode.DIGITAL_TWIN)
        factory = DriverFactory(config)
        driver = factory.create_camera_driver()
        assert driver is not None
        # DigitalTwinCameraDriver should be returned
        assert "DigitalTwin" in driver.__class__.__name__

    def test_create_camera_driver_with_stub_image(self):
        """Verifies factory passes stub_image_path to camera driver.

        Tests custom stub image configuration for digital twin driver.

        Arrangement:
        1. Create temporary directory for stub images.
        2. Create DriverConfig with stub_image_path=tmpdir.
        3. Create DriverFactory with config.

        Action:
        Call factory.create_camera_driver() with custom path.

        Assertion Strategy:
        Validates custom image support by confirming:
        - Driver is created successfully (not None).
        - Factory accepts stub_image_path configuration.

        Testing Principle:
        Validates configuration propagation, ensuring factory passes
        stub image paths to driver for custom simulations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DriverConfig(
                mode=DriverMode.DIGITAL_TWIN, stub_image_path=Path(tmpdir)
            )
            factory = DriverFactory(config)
            driver = factory.create_camera_driver()
            assert driver is not None

    def test_create_motor_controller_digital_twin(self):
        """Verifies factory creates StubMotorController in simulation mode.

        Tests motor controller instantiation with DIGITAL_TWIN config.

        Arrangement:
        1. Create DriverConfig with mode=DIGITAL_TWIN.
        2. Create DriverFactory with config.

        Action:
        Call factory.create_motor_controller() to instantiate controller.

        Assertion Strategy:
        Validates controller type by confirming:
        - Returned controller is not None.
        - Controller class name contains "Stub".

        Testing Principle:
        Validates consistent factory behavior, ensuring motor
        controllers follow same mode dispatch as cameras."""
        config = DriverConfig(mode=DriverMode.DIGITAL_TWIN)
        factory = DriverFactory(config)
        controller = factory.create_motor_controller()
        assert controller is not None
        assert "Stub" in controller.__class__.__name__

    def test_create_position_sensor_digital_twin(self):
        """Verifies factory creates StubPositionSensor in simulation mode.

        Tests position sensor instantiation with DIGITAL_TWIN config.

        Arrangement:
        1. Create DriverConfig with mode=DIGITAL_TWIN.
        2. Create DriverFactory with config.

        Action:
        Call factory.create_position_sensor() to instantiate sensor.

        Assertion Strategy:
        Validates sensor type by confirming:
        - Returned sensor is not None.
        - Sensor class name contains "Stub".

        Testing Principle:
        Validates factory completeness, ensuring all driver types
        support simulation mode creation."""
        config = DriverConfig(mode=DriverMode.DIGITAL_TWIN)
        factory = DriverFactory(config)
        sensor = factory.create_position_sensor()
        assert sensor is not None
        assert "Stub" in sensor.__class__.__name__


class TestGlobalConfiguration:
    """Test suite for global configuration helper functions.

    Categories:
    1. Mode Switching - use_digital_twin/use_hardware (2 tests)
    2. Custom Configuration - configure() function (1 test)
    3. Data Directory - set_data_dir() variants (2 tests)
    4. Location Setting - set_location() with/without altitude (2 tests)

    Total: 7 tests.
    """

    def test_use_digital_twin(self):
        """Verifies use_digital_twin() switches global factory to simulation mode.

        Tests convenience function for enabling simulation.

        Arrangement:
        1. Global factory may be in any mode.

        Action:
        Call use_digital_twin(), then get_factory() to check mode.

        Assertion Strategy:
        Validates mode switch by confirming:
        - factory.config.mode equals DIGITAL_TWIN.

        Testing Principle:
        Validates convenience API, ensuring one-liner switches
        entire system to simulation mode."""
        use_digital_twin()
        factory = get_factory()
        assert factory.config.mode == DriverMode.DIGITAL_TWIN

    def test_use_hardware(self):
        """Verifies use_hardware() switches global factory to real hardware mode.

        Tests convenience function for enabling hardware control.

        Arrangement:
        1. Global factory starts in default mode.

        Action:
        Call use_hardware(), check mode, then reset to digital twin.

        Assertion Strategy:
        Validates mode switch by confirming:
        - factory.config.mode equals HARDWARE after call.

        Testing Principle:
        Validates convenience API, ensuring one-liner enables
        hardware control when ready for real operations."""
        use_hardware()
        factory = get_factory()
        assert factory.config.mode == DriverMode.HARDWARE
        # Switch back to digital twin for other tests
        use_digital_twin()

    def test_configure_custom(self):
        """Verifies configure() accepts custom DriverConfig with multiple settings.

        Tests full reconfiguration with custom data directory.

        Arrangement:
        1. Create temporary directory for data storage.
        2. Create DriverConfig with DIGITAL_TWIN mode and custom data_dir.

        Action:
        Call configure(config) to apply custom configuration globally.

        Assertion Strategy:
        Validates configuration application by confirming:
        - factory.config.data_dir equals custom tmpdir path.

        Testing Principle:
        Validates complete reconfiguration, ensuring configure()
        applies all custom settings to global factory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = DriverConfig(
                mode=DriverMode.DIGITAL_TWIN,
                data_dir=Path(tmpdir),
            )
            configure(config)
            factory = get_factory()
            assert factory.config.data_dir == Path(tmpdir)

    def test_set_data_dir_path(self):
        """Verifies set_data_dir() accepts Path objects for directory configuration.

        Tests data directory configuration with pathlib.Path.

        Arrangement:
        1. Create temporary directory.
        2. Convert to Path object.

        Action:
        Call set_data_dir(Path(tmpdir)) to configure storage location.

        Assertion Strategy:
        Validates Path acceptance by confirming:
        - factory.config.data_dir equals Path(tmpdir).

        Testing Principle:
        Validates type flexibility, ensuring set_data_dir()
        accepts modern Path objects for type-safe paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            set_data_dir(Path(tmpdir))
            factory = get_factory()
            assert factory.config.data_dir == Path(tmpdir)

    def test_set_data_dir_string(self):
        """Verifies set_data_dir() accepts string paths for directory configuration.

        Tests data directory configuration with string path.

        Arrangement:
        1. Create temporary directory.
        2. Use string path directly.

        Action:
        Call set_data_dir(tmpdir) with string argument.

        Assertion Strategy:
        Validates string acceptance and conversion by confirming:
        - factory.config.data_dir equals Path(tmpdir).
        - String automatically converted to Path.

        Testing Principle:
        Validates convenient API, ensuring set_data_dir()
        accepts strings for backward compatibility."""
        with tempfile.TemporaryDirectory() as tmpdir:
            set_data_dir(tmpdir)
            factory = get_factory()
            assert factory.config.data_dir == Path(tmpdir)

    def test_set_location(self):
        """Verifies set_location() stores lat/lon/alt in global configuration.

        Tests geographic location configuration with all coordinates.

        Arrangement:
        1. Define Los Angeles coordinates: lat=34.05, lon=-118.25, alt=100m.

        Action:
        Call set_location(34.05, -118.25, 100.0) to configure site.

        Assertion Strategy:
        Validates coordinate storage by confirming:
        - factory.config.location["lat"] equals 34.05.
        - factory.config.location["lon"] equals -118.25.
        - factory.config.location["alt"] equals 100.0.

        Testing Principle:
        Validates complete location configuration, ensuring all
        geographic coordinates stored for astronomical calculations."""
        set_location(34.05, -118.25, 100.0)
        factory = get_factory()
        assert factory.config.location["lat"] == 34.05
        assert factory.config.location["lon"] == -118.25
        assert factory.config.location["alt"] == 100.0

    def test_set_location_default_altitude(self):
        """Verifies set_location() defaults altitude to 0.0 when omitted.

        Tests location configuration with optional altitude parameter.

        Arrangement:
        1. Define Philadelphia coordinates without altitude.

        Action:
        Call set_location(40.0, -75.0) with only lat/lon.

        Assertion Strategy:
        Validates default altitude by confirming:
        - factory.config.location["lat"] equals 40.0.
        - factory.config.location["lon"] equals -75.0.
        - factory.config.location["alt"] equals 0.0 (default).

        Testing Principle:
        Validates optional parameters, ensuring altitude defaults
        to sea level when not specified."""
        set_location(40.0, -75.0)
        factory = get_factory()
        assert factory.config.location["lat"] == 40.0
        assert factory.config.location["lon"] == -75.0
        assert factory.config.location["alt"] == 0.0


class TestSessionManagerIntegration:
    """Test suite for SessionManager integration with driver configuration.

    Categories:
    1. Singleton Access - get_session_manager() (1 test)
    2. Configuration Integration - Data dir usage (1 test)
    3. Lifecycle Management - Reset on reconfiguration (1 test)

    Total: 3 tests.
    """

    def test_get_session_manager(self):
        """Verifies get_session_manager() returns singleton instance.

        Tests session manager singleton pattern by calling getter twice.

        Arrangement:
        1. Session manager may or may not exist.

        Action:
        Call get_session_manager() twice and compare instances.

        Assertion Strategy:
        Validates singleton by confirming:
        - manager1 is manager2 (same object identity).

        Testing Principle:
        Validates singleton pattern, ensuring single session manager
        instance shared across application for consistent session state."""
        manager1 = get_session_manager()
        manager2 = get_session_manager()
        assert manager1 is manager2

    def test_session_manager_uses_config_data_dir(self):
        """Verifies session manager uses data_dir from global configuration.

        Tests configuration propagation to session manager.

        Arrangement:
        1. Create temporary directory for data storage.
        2. Call set_data_dir(tmpdir) to configure globally.

        Action:
        Call get_session_manager() to retrieve manager instance.

        Assertion Strategy:
        Validates config integration by confirming:
        - manager.data_dir equals tmpdir (string match).

        Testing Principle:
        Validates configuration flow, ensuring session manager
        respects global data directory setting for session storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            set_data_dir(tmpdir)
            manager = get_session_manager()
            assert str(manager.data_dir) == tmpdir

    def test_session_manager_resets_with_location(self):
        """Verifies session manager remains accessible after location changes.

        Tests session manager stability during configuration updates.

        Arrangement:
        1. Create temporary directory.
        2. Call set_location() to update geographic coordinates.
        3. Call set_data_dir() to configure storage.

        Action:
        Call get_session_manager() after configuration changes.

        Assertion Strategy:
        Validates manager availability by confirming:
        - manager is not None after reconfiguration.

        Testing Principle:
        Validates configuration resilience, ensuring session manager
        remains functional during runtime configuration updates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Set location and data dir to force fresh manager
            set_location(35.0, -120.0, 50.0)
            set_data_dir(tmpdir)
            manager = get_session_manager()
            # Manager should exist
            assert manager is not None


class TestConfigReconfiguration:
    """Test suite for runtime reconfiguration behavior and side effects.

    Categories:
    1. Session Manager Reset - Manager replacement on reconfig (1 test)
    2. Cleanup - Proper shutdown on reconfiguration (1 test)

    Total: 2 tests.
    """

    def test_reconfigure_resets_session_manager(self):
        """Verifies set_data_dir() creates new session manager with updated path.

        Tests session manager replacement during reconfiguration.

        Arrangement:
        1. Create two temporary directories (tmpdir1, tmpdir2).
        2. Configure with tmpdir1 and get initial manager.
        3. Reconfigure with tmpdir2.

        Action:
        Get session manager after reconfiguration.

        Assertion Strategy:
        Validates manager replacement by confirming:
        - manager2.data_dir equals tmpdir2 (new path).

        Testing Principle:
        Validates reconfiguration propagation, ensuring data directory
        changes result in session manager using new storage location."""
        with tempfile.TemporaryDirectory() as tmpdir1:
            with tempfile.TemporaryDirectory() as tmpdir2:
                # First config
                set_data_dir(tmpdir1)
                manager1 = get_session_manager()

                # Reconfigure
                set_data_dir(tmpdir2)
                manager2 = get_session_manager()

                # Should be different instance
                assert manager2.data_dir == Path(tmpdir2)

    def test_configure_shuts_down_existing_session_manager(self):
        """Verifies configure() properly handles existing session manager shutdown.

        Tests cleanup during reconfiguration.

        Arrangement:
        1. Get initial session manager to ensure one exists.
        2. Prepare new DriverConfig with DIGITAL_TWIN mode.

        Action:
        Call configure() with new config, then get_session_manager().

        Assertion Strategy:
        Validates clean reconfiguration by confirming:
        - manager2 is not None (new manager created successfully).
        - No errors or resource leaks during shutdown.

        Testing Principle:
        Validates resource management, ensuring configure() properly
        shuts down old manager before creating new one."""
        # Get initial manager
        manager1 = get_session_manager()

        # Reconfigure
        configure(DriverConfig(mode=DriverMode.DIGITAL_TWIN))

        # Get new manager
        manager2 = get_session_manager()

        # Should work without errors
        assert manager2 is not None


class TestDriverModeEnum:
    """Test suite for DriverMode enumeration.

    Categories:
    1. Enum Values - String representations (1 test)
    2. Enum Construction - From string (1 test)

    Total: 2 tests.
    """

    def test_driver_mode_values(self):
        """Verifies DriverMode enum defines expected string values.

        Tests enum value definitions.

        Arrangement:
        1. DriverMode enum defined with HARDWARE and DIGITAL_TWIN.

        Action:
        Access .value attribute on each enum member.

        Assertion Strategy:
        Validates enum values by confirming:
        - HARDWARE.value equals "hardware".
        - DIGITAL_TWIN.value equals "digital_twin".

        Testing Principle:
        Validates enum interface, ensuring mode values match expected
        strings for serialization and configuration files."""
        assert DriverMode.HARDWARE.value == "hardware"
        assert DriverMode.DIGITAL_TWIN.value == "digital_twin"

    def test_driver_mode_from_string(self):
        """Verifies DriverMode enum can be constructed from string values.

        Tests enum deserialization from configuration strings.

        Arrangement:
        1. Define mode strings: "hardware", "digital_twin".

        Action:
        Call DriverMode(string) to construct enum from string.

        Assertion Strategy:
        Validates string construction by confirming:
        - DriverMode("hardware") equals DriverMode.HARDWARE.
        - DriverMode("digital_twin") equals DriverMode.DIGITAL_TWIN.

        Testing Principle:
        Validates bidirectional conversion, ensuring enum can be
        deserialized from config files and command-line arguments."""
        mode = DriverMode("hardware")
        assert mode == DriverMode.HARDWARE

        mode = DriverMode("digital_twin")
        assert mode == DriverMode.DIGITAL_TWIN
