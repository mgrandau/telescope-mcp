"""Driver configuration and factory.

Supports switching between real hardware drivers and digital twin (stub) drivers
for testing and development without physical hardware.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from telescope_mcp.drivers.cameras import (
    ASICameraDriver,
    CameraDriver,
    DigitalTwinCameraDriver,
)
from telescope_mcp.drivers.motors import MotorController, StubMotorController
from telescope_mcp.drivers.sensors import (
    DigitalTwinSensorDriver,
    SensorDriver,
)
from telescope_mcp.observability import get_logger

if TYPE_CHECKING:
    from telescope_mcp.data.session_manager import SessionManager

logger = get_logger(__name__)


class DriverMode(Enum):
    """Driver mode selection."""

    HARDWARE = "hardware"  # Real hardware drivers
    DIGITAL_TWIN = "digital_twin"  # Simulated drivers for testing


def _default_data_dir() -> Path:
    """Get default data directory for session storage.

    Returns ~/.telescope-mcp/data as the default location for
    storing ASDF session files.

    Business context: Provides consistent storage location across installations without
    configuration. Enables finding session data without environment variables. Used as
    default for DriverConfig.data_dir field.

    Args:
        None.

    Returns:
        Path to ~/.telescope-mcp/data directory.

    Raises:
        None. Path may not exist (created on first session write).

    Example:
        >>> data_dir = _default_data_dir()
        >>> print(data_dir)  # /home/user/.telescope-mcp/data
    """
    return Path.home() / ".telescope-mcp" / "data"


@dataclass
class DriverConfig:
    """Configuration for driver selection."""

    mode: DriverMode = DriverMode.DIGITAL_TWIN

    # Data storage settings
    data_dir: Path = field(default_factory=_default_data_dir)

    # Observer location (lat, lon, alt)
    location: dict[str, float] = field(default_factory=dict)

    # Camera settings
    finder_camera_id: int = 0
    main_camera_id: int = 1

    # Digital twin settings
    stub_image_path: Path | None = None  # For file/directory image source

    # Motor settings (for hardware mode)
    motor_serial_port: str | None = None
    motor_baud_rate: int = 115200

    # Sensor settings (for hardware mode)
    sensor_i2c_bus: int = 1
    sensor_i2c_address: int = 0x68


class DriverFactory:
    """Factory for creating drivers based on configuration.

    Provides a centralized way to create camera, motor, and sensor drivers
    based on the configured mode (hardware vs digital twin).
    """

    def __init__(self, config: DriverConfig | None = None):
        """Initialize driver factory with hardware/simulation configuration.

        Creates factory for producing camera, motor, sensor drivers based on mode.
        Default config uses digital twin simulation enabling immediate development
        without hardware setup.

        Business context: Central configuration point determining whether application
        uses real hardware or simulation throughout. Developers set once at startup,
        all code automatically uses appropriate drivers. Critical for CI/CD, offline
        development, demos without hardware.

        Implementation details: Stores config reference for create_* methods. Uses
        config.mode (HARDWARE/DIGITAL_TWIN) to determine driver types. Config includes
        data directory, camera IDs, motor ports, sensor addresses. Lazy evaluation -
        drivers created on demand by create_*.

        Args:
            config: DriverConfig with mode (default DIGITAL_TWIN), data_dir, camera
                IDs, hardware ports/addresses. None defaults to DriverConfig() with
                all defaults.

        Returns:
            None. Stores config for later driver creation.

        Raises:
            None. Construction always succeeds - errors occur during driver creation.

        Example:
            >>> factory = DriverFactory()  # Digital twin mode
            >>> camera_driver = factory.create_camera_driver()
            >>> # Gets DigitalTwinCameraDriver
            >>> # Or hardware mode:
            >>> config = DriverConfig(mode=DriverMode.HARDWARE)
            >>> factory = DriverFactory(config)
            >>> camera_driver = factory.create_camera_driver()  # Gets ASICameraDriver
        """
        self.config = config or DriverConfig()

    def create_camera_driver(self) -> CameraDriver:
        """Create camera driver for hardware or simulation based on config mode.

        Factory method returning ASICameraDriver (real ZWO ASI cameras) in HARDWARE
        mode or DigitalTwinCameraDriver (simulation) in DIGITAL_TWIN mode. The mode
        is set via self.config.mode during construction or via configure(). This
        abstraction enables identical code to work with both real and simulated
        cameras.

        Business context: Central abstraction enabling development without physical
        hardware. Developers write code once and run against simulated cameras (fast,
        reliable) or real cameras (actual data) by changing configuration. Critical
        for CI/CD running tests with digital twins at high speed. Enables demos
        without hardware. Used by Camera class to obtain drivers transparently.

        Returns:
            CameraDriver - ASICameraDriver in HARDWARE mode (ZWO SDK),
            DigitalTwinCameraDriver in DIGITAL_TWIN mode. Both implement
            get_connected_cameras() and open(camera_id).

        Raises:
            ImportError: If zwoasi module unavailable in HARDWARE mode
                (SDK not installed).

        Example:
            >>> factory = DriverFactory(DriverConfig(mode=DriverMode.DIGITAL_TWIN))
            >>> driver = factory.create_camera_driver()
            >>> cameras = driver.get_connected_cameras()  # Simulated cameras
        """
        if self.config.mode == DriverMode.HARDWARE:
            return ASICameraDriver()
        else:
            # Use DigitalTwinCameraDriver with configured image source
            from telescope_mcp.drivers.cameras import (
                DigitalTwinConfig,
                ImageSource,
            )

            twin_config = DigitalTwinConfig(
                image_source=ImageSource.DIRECTORY
                if self.config.stub_image_path
                else ImageSource.SYNTHETIC,
                image_path=self.config.stub_image_path,
            )
            return DigitalTwinCameraDriver(twin_config)

    def create_motor_controller(self) -> MotorController:
        """Create motor controller for telescope altitude/azimuth motors.

        Factory method returning motor controller. HARDWARE mode raises
        NotImplementedError (not yet implemented). DIGITAL_TWIN mode returns
        StubMotorController simulating movements with realistic timing but no
        hardware interaction.

        Business context: Enables development of telescope control logic (goto,
        tracking) without physical mount hardware. Digital twin allows testing
        pointing algorithms, slewing patterns in CI/CD. Future hardware mode will
        integrate stepper motors via serial/USB. Critical for teams with limited
        hardware access.

        Returns:
            MotorController - StubMotorController in DIGITAL_TWIN mode with
            move_altitude(), move_azimuth(), stop() methods.

        Raises:
            NotImplementedError: In HARDWARE mode (real motor driver not
                implemented yet).

        Example:
            >>> factory = DriverFactory(DriverConfig(mode=DriverMode.DIGITAL_TWIN))
            >>> motors = factory.create_motor_controller()
            >>> motors.move_altitude(steps=1000, speed=50)  # Simulated
        """
        if self.config.mode == DriverMode.HARDWARE:
            # TODO: Import and return real motor driver
            raise NotImplementedError("Hardware motor driver not yet implemented")
        else:
            return StubMotorController()

    def create_sensor_driver(self) -> SensorDriver:
        """Create sensor driver for telescope orientation sensing.

        Factory method returning sensor driver. HARDWARE mode returns
        ArduinoSensorDriver for real IMU hardware. DIGITAL_TWIN mode returns
        DigitalTwinSensorDriver for simulated sensor data.

        Business context: Enables closed-loop position control and pointing
        verification. Digital twin allows testing pointing accuracy, goto
        convergence in CI/CD. Hardware mode integrates Arduino Nano BLE33 Sense
        IMU for real orientation data.

        Returns:
            SensorDriver - DigitalTwinSensorDriver in DIGITAL_TWIN mode,
            ArduinoSensorDriver in HARDWARE mode.

        Example:
            >>> factory = DriverFactory(DriverConfig(mode=DriverMode.DIGITAL_TWIN))
            >>> driver = factory.create_sensor_driver()
            >>> instance = driver.open()
            >>> reading = instance.read()
            >>> print(f"Alt={reading.altitude:.2f}°, Az={reading.azimuth:.2f}°")
        """
        if self.config.mode == DriverMode.HARDWARE:
            from telescope_mcp.drivers.sensors import ArduinoSensorDriver

            return ArduinoSensorDriver()
        else:
            return DigitalTwinSensorDriver()


# Global driver factory instance - can be reconfigured
_factory: DriverFactory | None = None

# Global session manager instance
_session_manager: SessionManager | None = None


def get_factory() -> DriverFactory:
    """Get the global driver factory singleton for consistent driver access.

    Returns the global DriverFactory instance, creating with default config
    (DIGITAL_TWIN mode) on first access. Use configure(), use_hardware(),
    use_digital_twin() to change configuration. Singleton ensures all components
    use consistent driver configuration.

    Business context: Central access point for driver creation. By using singleton,
    configuration changes propagate to all components without passing factory
    references. Simplifies startup where initial config is set once then all
    components call get_factory(). Essential for tools layer needing drivers without
    managing config.

    Returns:
        DriverFactory singleton. First call creates with DriverConfig() defaults.
        After configure(), returns configured factory.

    Raises:
        None. Always returns valid DriverFactory.

    Example:
        >>> factory = get_factory()  # Default digital twin
        >>> use_hardware()
        >>> factory = get_factory()  # Same ref, hardware mode
    """
    global _factory
    if _factory is None:
        _factory = DriverFactory()
    return _factory


def get_session_manager() -> SessionManager:
    """Get the global session manager singleton for ASDF data storage.

    Returns the global SessionManager instance, creating on first access using
    current factory config (data_dir, location). Handles ASDF file storage for
    observations: frames, telemetry, calibration, events. Singleton ensures
    consistent session state.

    Business context: Central data storage interface for all observation data. By
    using singleton, all components write to same session files without explicit
    coordination. Enables atomic observation sessions where multiple data sources
    contribute to unified ASDF files. Essential for data provenance - all session
    data bundled with consistent metadata.

    Returns:
        SessionManager singleton with start_session(), log(), add_frame(), add_event(),
        end_session() methods. Data stored in config.data_dir.

    Raises:
        None. Creates on first access. Session operations may raise on disk/
        permission issues.

    Example:
        >>> set_location(lat=40.7, lon=-74.0, alt=10)
        >>> mgr = get_session_manager()
        >>> mgr.start_session(name="M42", session_type="science")
    """
    global _session_manager
    if _session_manager is None:
        from telescope_mcp.data import SessionManager

        config = get_factory().config
        _session_manager = SessionManager(
            data_dir=config.data_dir,
            location=config.location if config.location else None,
        )
    return _session_manager


def configure(config: DriverConfig) -> None:
    """Configure global factory and session manager with new settings.

    Replaces global factory singleton with new instance using provided configuration.
    Shuts down existing session manager gracefully first so new one uses updated
    config on next access. Enables runtime switching between hardware/digital twin
    or changing data storage locations.

    Business context: Primary mechanism for application-level configuration. Enables
    switching development (digital twin) to production (hardware) at startup based
    on flags. Allows changing data directories for campaigns/disks. Used by tests to
    ensure clean state. Essential for deployment flexibility.

    Args:
        config: DriverConfig with mode (HARDWARE/DIGITAL_TWIN), data_dir (Path),
            location (dict), camera_ids (int), stub_image_path (Path|None),
            motor/sensor settings.

    Returns:
        None. Global singletons updated.

    Raises:
        None. Handles session shutdown safely. Validation is caller's responsibility.

    Example:
        >>> hw = os.getenv("USE_HW")
        >>> mode = DriverMode.HARDWARE if hw else DriverMode.DIGITAL_TWIN
        >>> configure(DriverConfig(mode=mode, data_dir=Path("/data/obs")))
    """
    global _factory, _session_manager

    # Shutdown existing session manager if reconfiguring
    if _session_manager is not None:
        _session_manager.shutdown()
        _session_manager = None

    _factory = DriverFactory(config)


def use_digital_twin() -> None:
    """Switch to digital twin mode for simulated hardware operation.

    Convenience function calling configure(DriverConfig(mode=DIGITAL_TWIN)) to set
    all drivers to simulation. Useful for development, testing, demonstrations,
    CI/CD without physical devices. Keeps other config (data_dir, location) at
    defaults.

    Business context: Enables offline development where developers test telescope
    code without hardware access. Critical for CI/CD in cloud with no USB devices.
    Allows demonstrations without hauling equipment. Used by tests for repeatable
    fast execution. Essential for teams with limited hardware access.

    Returns:
        None. Global factory reconfigured for digital twin mode.

    Raises:
        None. Always succeeds as digital twin has no hardware dependencies.

    Example:
        >>> use_digital_twin()  # All drivers now simulated
        >>> driver = get_factory().create_camera_driver()  # DigitalTwinCameraDriver
    """
    configure(DriverConfig(mode=DriverMode.DIGITAL_TWIN))


def use_hardware() -> None:
    """Switch to hardware mode for real telescope equipment operation.

    Convenience function calling configure(DriverConfig(mode=DriverMode.HARDWARE))
    to set camera drivers to real ZWO ASI hardware. Motors/sensors raise
    NotImplementedError. Requires physical ASI cameras connected via USB. Keeps
    other config at defaults.

    Business context: Essential for production telescope operation where real data
    acquisition is the goal. Used when transitioning from development (digital twin)
    to observation sessions. Enables final hardware validation after simulation
    testing. Critical for automated workflows that dry-run in simulation then switch
    to hardware.

    Returns:
        None. Global factory reconfigured for hardware mode.

    Raises:
        None. Config succeeds even without hardware. Driver operations fail
        if unavailable.

    Example:
        >>> use_hardware()  # Switch to real cameras
        >>> driver = get_factory().create_camera_driver()  # ASICameraDriver
    """
    configure(DriverConfig(mode=DriverMode.HARDWARE))


def set_data_dir(data_dir: Path | str) -> None:
    """Set the data directory for ASDF session file storage.

    Updates factory config data_dir and resets session manager so new sessions use
    updated directory. Existing sessions closed first. Directory created by
    SessionManager on first session if doesn't exist.

    Business context: Enables organizing observation data by campaign/target/date
    without manual path management. Allows switching storage based on disk space
    (observatories have multiple drives). Essential for automated workflows creating
    dated directories. Used by CLI tools accepting custom directories via flags.
    Critical for multi-user systems with separate data directories.

    Args:
        data_dir: Path to directory for ASDF sessions. String or Path. Examples:
            "/mnt/observatory/data", "~/telescope-data", "./observations/m42"

    Returns:
        None. Factory config updated, session manager reset.

    Raises:
        None. Directory creation happens when SessionManager starts session. If not
        writable, session start raises PermissionError.

    Example:
        >>> set_data_dir("/mnt/obs/ngc6888")
        >>> mgr = get_session_manager()
        >>> mgr.start_session("ngc6888_001", "science")
    """
    factory = get_factory()
    factory.config.data_dir = Path(data_dir)

    # Reset session manager to pick up new config
    global _session_manager
    if _session_manager is not None:
        _session_manager.shutdown()
        _session_manager = None


def set_location(lat: float, lon: float, alt: float = 0.0) -> None:
    """Set the observer location for astronomical calculations and metadata.

    Configures geographic coordinates for the telescope's location which are
    stored in session metadata and used for astronomical calculations like
    local sidereal time, object altitude/azimuth, and atmospheric refraction
    corrections.

    Business context: Essential for accurate astronomical observations requiring
    coordinate transformations from celestial to horizontal coordinate systems.
    Location data enables plate solving, goto functionality based on alt-az or
    RA-Dec coordinates, and proper timestamps in observation logs. Required for
    scientific data provenance showing where observations were made.

    Args:
        lat: Latitude in decimal degrees. Range: -90 (South Pole) to 90 (North Pole).
            Positive=North, Negative=South. Example: 40.7128 for New York City.
        lon: Longitude in decimal degrees. Range: -180 to 180. Positive=East,
            Negative=West. Example: -74.0060 for New York City.
        alt: Altitude in meters above sea level. Default 0 (sea level). Used for
            atmospheric refraction corrections. Example: 2000 for mountain observatory.

    Returns:
        None. Location is stored in factory config for use by all components.

    Raises:
        None. Validation should be performed by caller if needed.

    Example:
        >>> # Set location for Mauna Kea Observatory
        >>> set_location(lat=19.8207, lon=-155.4681, alt=4205)
        >>> # Set location for home observatory in Denver
        >>> set_location(lat=39.7392, lon=-104.9903, alt=1609)
    """
    factory = get_factory()
    factory.config.location = {"lat": lat, "lon": lon, "alt": alt}
