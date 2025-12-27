"""Sensor type definitions and protocols.

This module contains base types, data classes, and protocols for sensor
drivers. By keeping these in a separate module, we avoid circular imports
when implementation modules need to reference these types.

Types defined here:
- SensorReading: Data class for sensor readings
- SensorInstance: Protocol for connected sensor instances
- SensorDriver: Protocol for sensor drivers

Example:
    from telescope_mcp.drivers.sensors.types import (
        SensorReading,
        SensorInstance,
        SensorDriver,
    )

    class MySensorDriver:
        def open(self, sensor_id: int | str = 0) -> SensorInstance:
            ...
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


@dataclass
class SensorReading:
    """A single sensor reading with all data.

    Attributes:
        accelerometer: Dict with aX, aY, aZ in g.
        magnetometer: Dict with mX, mY, mZ in µT.
        altitude: Calculated altitude in degrees.
        azimuth: Calculated azimuth in degrees.
        temperature: Temperature in Celsius.
        humidity: Relative humidity in %RH.
        timestamp: When reading was taken.
        raw_values: Raw tab-separated string (for compatibility).
    """

    accelerometer: dict[str, float]
    magnetometer: dict[str, float]
    altitude: float
    azimuth: float
    temperature: float
    humidity: float
    timestamp: datetime
    raw_values: str = ""


class SensorInstance(Protocol):  # pragma: no cover
    """Protocol for connected sensor instances.

    A SensorInstance represents an open connection to a sensor device.
    It provides methods to read data, calibrate, and control the sensor.
    """

    def get_info(self) -> dict:
        """Get sensor information.

        Returns:
            Dict with sensor type, capabilities, and configuration.
        """
        ...

    def read(self) -> SensorReading:
        """Read current sensor values.

        Returns:
            SensorReading with accelerometer, magnetometer, position,
            temperature, and humidity data.

        Raises:
            RuntimeError: If sensor is closed or unavailable.
        """
        ...

    def calibrate(self, true_altitude: float, true_azimuth: float) -> None:
        """Calibrate sensor to known true position.

        Args:
            true_altitude: Known true altitude in degrees.
            true_azimuth: Known true azimuth in degrees.
        """
        ...

    def get_status(self) -> dict:
        """Get sensor status information.

        Returns:
            Dict with connection status, calibration state, etc.
        """
        ...

    def reset(self) -> None:
        """Reset sensor to initial state."""
        ...

    def close(self) -> None:
        """Close the sensor connection."""
        ...


class SensorDriver(Protocol):  # pragma: no cover
    """Protocol for sensor drivers.

    A SensorDriver handles discovery and connection to sensor devices.
    It can enumerate available sensors and open connections.
    """

    def get_available_sensors(self) -> list[dict]:
        """List available sensors.

        Returns:
            List of sensor info dicts with id, type, name, port.
        """
        ...

    def open(self, sensor_id: int | str = 0) -> SensorInstance:
        """Open a sensor instance.

        Args:
            sensor_id: Sensor ID or port to open.

        Returns:
            SensorInstance for reading sensor data.

        Raises:
            RuntimeError: If sensor cannot be opened.
        """
        ...

    def close(self) -> None:
        """Close the current sensor instance."""
        ...
