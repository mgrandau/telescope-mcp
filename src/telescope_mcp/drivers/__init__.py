"""Hardware drivers for telescope control.

Supports two modes:
- HARDWARE: Real physical devices (cameras, motors, sensors)
- DIGITAL_TWIN: Simulated devices for testing without hardware

Use drivers.config to switch modes:
    from telescope_mcp.drivers import config
    config.use_digital_twin()  # or config.use_hardware()

Serial Protocols:
    SerialPort and PortEnumerator protocols enable testing of
    serial-based drivers (sensors, motors) without hardware.

    from telescope_mcp.drivers import SerialPort, PortEnumerator
"""

from telescope_mcp.drivers import asi_sdk, config, motors, sensors
from telescope_mcp.drivers.config import (
    DriverConfig,
    DriverFactory,
    DriverMode,
    configure,
    get_factory,
    use_digital_twin,
    use_hardware,
)
from telescope_mcp.drivers.serial import PortEnumerator, SerialPort

__all__ = [
    # Submodules
    "asi_sdk",
    "motors",
    "sensors",
    "config",
    # Configuration
    "DriverMode",
    "DriverConfig",
    "DriverFactory",
    "get_factory",
    "configure",
    "use_digital_twin",
    "use_hardware",
    # Serial protocols (for testing)
    "SerialPort",
    "PortEnumerator",
]
