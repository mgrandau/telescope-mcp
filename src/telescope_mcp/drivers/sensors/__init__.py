"""Sensor drivers for telescope orientation and position.

This module provides abstractions for orientation sensors (IMUs) that
report telescope pointing direction. Key components:

- SensorReading: Complete sensor reading with accelerometer, magnetometer, etc.
- SensorDriver: Protocol for sensor driver implementations
- SensorInstance: Protocol for connected sensor instances
- DigitalTwinSensorDriver: Simulated sensor for testing
- ArduinoSensorDriver: Real hardware driver for Arduino Nano BLE33 Sense

Example:
    from telescope_mcp.drivers.sensors import (
        DigitalTwinSensorDriver,
        ArduinoSensorDriver,
    )

    # For testing (no hardware required)
    driver = DigitalTwinSensorDriver()
    instance = driver.open()
    reading = instance.read()
    print(f"ALT: {reading.altitude:.2f}°, AZ: {reading.azimuth:.2f}°")

    # For real hardware
    driver = ArduinoSensorDriver()
    sensors = driver.get_available_sensors()
    if sensors:
        instance = driver.open(sensors[0]["port"])
        reading = instance.read()
"""

# Import order: types first (avoid circular imports), then implementations
# Driver implementations
from telescope_mcp.drivers.sensors.arduino import (
    ArduinoSensorDriver,
    ArduinoSensorInstance,
)
from telescope_mcp.drivers.sensors.twin import (
    DigitalTwinSensorConfig,
    DigitalTwinSensorDriver,
    DigitalTwinSensorInstance,
)
from telescope_mcp.drivers.sensors.types import (
    AvailableSensor,
    SensorDriver,
    SensorInstance,
    SensorReading,
    validate_position,
)

# Serial protocols (re-exported from parent for convenience)
from telescope_mcp.drivers.serial import PortEnumerator, SerialPort

__all__ = [
    # Data classes
    "SensorReading",
    # Protocols
    "SensorInstance",
    "SensorDriver",
    # Type definitions
    "AvailableSensor",
    # Validation helpers
    "validate_position",
    # Serial protocols (re-exported from drivers.serial)
    "SerialPort",
    "PortEnumerator",
    # Digital twin
    "DigitalTwinSensorConfig",
    "DigitalTwinSensorDriver",
    "DigitalTwinSensorInstance",
    # Arduino hardware
    "ArduinoSensorDriver",
    "ArduinoSensorInstance",
]
