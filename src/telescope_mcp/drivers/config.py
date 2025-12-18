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
    """Get default data directory."""
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
    """Factory for creating drivers based on configuration."""
    
    def __init__(self, config: DriverConfig | None = None):
        self.config = config or DriverConfig()
    
    def create_camera_driver(self) -> CameraDriver:
        """Create camera driver based on config."""
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
        """Create motor controller based on config."""
        if self.config.mode == DriverMode.HARDWARE:
            # TODO: Import and return real motor driver
            raise NotImplementedError("Hardware motor driver not yet implemented")
        else:
            return StubMotorController()
    
    def create_position_sensor(self) -> PositionSensor:
        """Create position sensor based on config."""
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
    """Get the global driver factory."""
    global _factory
    if _factory is None:
        _factory = DriverFactory()
    return _factory


def get_session_manager() -> "SessionManager":
    """Get the global session manager.
    
    Creates the session manager on first access using the current config.
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
    """Configure the global driver factory and session manager."""
    global _factory, _session_manager
    
    # Shutdown existing session manager if reconfiguring
    if _session_manager is not None:
        _session_manager.shutdown()
        _session_manager = None
    
    _factory = DriverFactory(config)


def use_digital_twin() -> None:
    """Switch to digital twin mode."""
    configure(DriverConfig(mode=DriverMode.DIGITAL_TWIN))


def use_hardware() -> None:
    """Switch to hardware mode."""
    configure(DriverConfig(mode=DriverMode.HARDWARE))


def set_data_dir(data_dir: Path | str) -> None:
    """Set the data directory for session storage."""
    factory = get_factory()
    factory.config.data_dir = Path(data_dir)
    
    # Reset session manager to pick up new config
    global _session_manager
    if _session_manager is not None:
        _session_manager.shutdown()
        _session_manager = None


def set_location(lat: float, lon: float, alt: float = 0.0) -> None:
    """Set the observer location."""
    factory = get_factory()
    factory.config.location = {"lat": lat, "lon": lon, "alt": alt}
