"""Hardware drivers for telescope control.

Supports two modes:
- HARDWARE: Real physical devices (cameras, motors, sensors)
- DIGITAL_TWIN: Simulated devices for testing without hardware

Use drivers.config to switch modes:
    from telescope_mcp.drivers import config
    config.use_digital_twin()  # or config.use_hardware()
"""

from telescope_mcp.drivers import pyasi, motors, sensors, config
from telescope_mcp.drivers.config import (
    DriverMode,
    DriverConfig,
    DriverFactory,
    get_factory,
    configure,
    use_digital_twin,
    use_hardware,
)

__all__ = [
    "pyasi",
    "motors",
    "sensors",
    "config",
    "DriverMode",
    "DriverConfig",
    "DriverFactory",
    "get_factory",
    "configure",
    "use_digital_twin",
    "use_hardware",
]
