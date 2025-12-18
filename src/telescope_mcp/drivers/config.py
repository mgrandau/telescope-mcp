"""Driver configuration and factory.

Supports switching between real hardware drivers and digital twin (stub) drivers
for testing and development without physical hardware.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from telescope_mcp.drivers.cameras import (
    ASICameraDriver,
    CameraDriver,
    DigitalTwinCameraDriver,
)
from telescope_mcp.drivers.motors import MotorController, StubMotorController
from telescope_mcp.drivers.sensors import PositionSensor, StubPositionSensor
from telescope_mcp.observability import get_logger

logger = get_logger(__name__)


class DriverMode(Enum):
    """Driver mode selection."""
    HARDWARE = "hardware"  # Real hardware drivers
    DIGITAL_TWIN = "digital_twin"  # Simulated drivers for testing


def _default_data_dir() -> Path:
    """Get default data directory for session storage.
    
    Returns ~/.telescope-mcp/data as the default location for
    storing ASDF session files.
    
    Returns:
        Path to the default data directory.
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
        """Initialize driver factory with configuration.
        
        Args:
            config: Driver configuration specifying mode and settings.
                Defaults to DriverConfig() (digital twin mode).
        """
        self.config = config or DriverConfig()
    
    def create_camera_driver(self) -> CameraDriver:
        """Create camera driver based on config mode.
        
        In HARDWARE mode, returns ASICameraDriver for real ZWO cameras.
        In DIGITAL_TWIN mode, returns DigitalTwinCameraDriver with
        synthetic or file-based images.
        
        Returns:
            CameraDriver instance appropriate for the configured mode.
        
        Raises:
            ImportError: If required driver modules are not available.
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
                image_source=ImageSource.DIRECTORY if self.config.stub_image_path else ImageSource.SYNTHETIC,
                image_path=self.config.stub_image_path,
            )
            return DigitalTwinCameraDriver(twin_config)
    
    def create_motor_controller(self) -> MotorController:
        """Create motor controller based on config mode.
        
        In HARDWARE mode, would return real motor driver (not yet implemented).
        In DIGITAL_TWIN mode, returns StubMotorController that simulates
        motor movements without actual hardware.
        
        Returns:
            MotorController instance appropriate for the configured mode.
        
        Raises:
            NotImplementedError: In HARDWARE mode (real motors not yet implemented).
        """
        if self.config.mode == DriverMode.HARDWARE:
            # TODO: Import and return real motor driver
            raise NotImplementedError("Hardware motor driver not yet implemented")
        else:
            return StubMotorController()
    
    def create_position_sensor(self) -> PositionSensor:
        """Create position sensor based on config mode.
        
        In HARDWARE mode, would return real sensor driver (not yet implemented).
        In DIGITAL_TWIN mode, returns StubPositionSensor that provides
        simulated position readings.
        
        Returns:
            PositionSensor instance appropriate for the configured mode.
        
        Raises:
            NotImplementedError: In HARDWARE mode (real sensors not yet implemented).
        """
        if self.config.mode == DriverMode.HARDWARE:
            # TODO: Import and return real sensor driver
            raise NotImplementedError("Hardware position sensor not yet implemented")
        else:
            return StubPositionSensor()


# Global driver factory instance - can be reconfigured
_factory: DriverFactory | None = None

# Global session manager instance
_session_manager: "SessionManager | None" = None


def get_factory() -> DriverFactory:
    """Get the global driver factory singleton.
    
    Creates a default DriverFactory on first access. Use configure()
    to change the configuration.
    
    Returns:
        The global DriverFactory instance.
    
    Raises:
        None. Always returns a valid factory.
    """
    global _factory
    if _factory is None:
        _factory = DriverFactory()
    return _factory


def get_session_manager() -> "SessionManager":
    """Get the global session manager singleton.
    
    Creates the session manager on first access using the current config.
    Session manager handles ASDF file storage for telescope data.
    
    Returns:
        The global SessionManager instance.
    
    Raises:
        None. Creates manager on first access if needed.
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
    """Configure the global driver factory and session manager.
    
    Replaces the global factory with one using the new config.
    If a session manager exists, shuts it down first so a new one
    will be created with the updated configuration.
    
    Args:
        config: New driver configuration to apply.
    
    Raises:
        None. Safely handles existing session manager shutdown.
    """
    global _factory, _session_manager
    
    # Shutdown existing session manager if reconfiguring
    if _session_manager is not None:
        _session_manager.shutdown()
        _session_manager = None
    
    _factory = DriverFactory(config)


def use_digital_twin() -> None:
    """Switch to digital twin mode.
    
    Convenience function to configure for simulated hardware.
    Useful for development and testing without physical devices.
    
    Raises:
        None. Always succeeds.
    """
    configure(DriverConfig(mode=DriverMode.DIGITAL_TWIN))


def use_hardware() -> None:
    """Switch to hardware mode.
    
    Convenience function to configure for real hardware drivers.
    Requires physical ASI cameras and other hardware to be connected.
    
    Raises:
        None. Always succeeds, but camera operations will fail if no hardware.
    """
    configure(DriverConfig(mode=DriverMode.HARDWARE))


def set_data_dir(data_dir: Path | str) -> None:
    """Set the data directory for session storage.
    
    Updates the configuration and resets the session manager so
    new sessions use the updated directory.
    
    Raises:
        None. Directory will be created if it doesn't exist.
    
    Args:
        data_dir: Path to directory for ASDF session files.
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
