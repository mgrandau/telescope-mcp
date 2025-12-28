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

    Business context: Defines the interface for all telescope orientation
    sensors. Implementations include ArduinoSensorInstance (real hardware)
    and DigitalTwinSensorInstance (simulation). Enables polymorphic use
    across different sensor types.
    """

    def get_info(self) -> dict:
        """Get sensor identification and capability information.

        Returns static information about the sensor hardware and its
        configuration. Used for logging, diagnostics, and UI display.

        Business context: Enables identification of sensor type for
        appropriate handling. UI can display sensor model, firmware
        version. Logs include sensor info for debugging.

        Returns:
            Dict containing:
            - type: Sensor type string (e.g., "arduino_ble33", "digital_twin")
            - name: Human-readable name
            - capabilities: List of supported features
            - port: Connection port/path (if applicable)
            - firmware: Firmware version (if available)

        Raises:
            RuntimeError: If sensor connection is closed.

        Example:
            >>> info = sensor.get_info()
            >>> print(f"Sensor: {info['type']} on {info['port']}")
        """
        ...

    def read(self) -> SensorReading:
        """Read current sensor values for telescope orientation.

        Returns the latest sensor data including accelerometer,
        magnetometer, calculated altitude/azimuth, and environmental
        readings. Core method for telescope position feedback.

        Business context: Primary interface for telescope orientation
        sensing. Called continuously during tracking to verify position
        and detect drift. Enables closed-loop pointing correction.

        Returns:
            SensorReading dataclass containing:
            - accelerometer: {aX, aY, aZ} in g units
            - magnetometer: {mX, mY, mZ} in µT
            - altitude: Calculated altitude in degrees (0-90)
            - azimuth: Calculated azimuth in degrees (0-360)
            - temperature: Ambient temperature in Celsius
            - humidity: Relative humidity in %RH
            - timestamp: Reading timestamp (UTC)

        Raises:
            RuntimeError: If sensor is closed or unavailable.
            ValueError: If sensor data is invalid or corrupted.

        Example:
            >>> reading = sensor.read()
            >>> print(f"ALT={reading.altitude:.2f}° AZ={reading.azimuth:.2f}°")
        """
        ...

    def calibrate(self, true_altitude: float, true_azimuth: float) -> None:
        """Calibrate sensor to known reference position.

        Sets calibration offsets so sensor readings match the true
        telescope position. Call when telescope is pointed at a known
        reference (star, landmark) with verified coordinates.

        Business context: Essential for accurate pointing. Raw sensor
        readings have installation offsets and magnetic declination
        errors. Calibration establishes mapping from sensor to sky
        coordinates. Run at session start or after disturbing mount.

        Args:
            true_altitude: Known true altitude in degrees (0-90).
                Zenith=90, horizon=0. From star catalog or reference.
            true_azimuth: Known true azimuth in degrees (0-360).
                North=0, East=90, South=180, West=270.

        Returns:
            None. Calibration applied to subsequent readings.

        Raises:
            ValueError: If altitude outside 0-90 or azimuth outside 0-360.
            RuntimeError: If sensor is closed.

        Example:
            >>> # After slewing to Polaris (alt=40°, az=0°)
            >>> sensor.calibrate(true_altitude=40.0, true_azimuth=0.0)
        """
        ...

    def get_status(self) -> dict:
        """Get current sensor operational status.

        Returns runtime status information including connection state,
        calibration status, error conditions, and data quality metrics.

        Business context: Enables monitoring of sensor health during
        observations. Detects calibration drift, communication issues,
        or environmental problems affecting accuracy.

        Returns:
            Dict containing:
            - connected: bool, whether sensor is responding
            - calibrated: bool, whether calibration has been set
            - is_open: bool, connection state
            - error: Optional error string if problem detected
            - last_reading_age_ms: Milliseconds since last successful read
            - reading_rate_hz: Current sample rate

        Raises:
            RuntimeError: If sensor connection is closed.

        Example:
            >>> status = sensor.get_status()
            >>> if not status['calibrated']:
            ...     print("Warning: Sensor not calibrated")
        """
        ...

    def reset(self) -> None:
        """Reset sensor to initial uncalibrated state.

        Clears calibration offsets and reinitializes sensor hardware.
        Use when starting fresh calibration or recovering from errors.

        Business context: Enables recovery from bad calibration or
        sensor errors. Part of troubleshooting workflow when readings
        seem incorrect.

        Returns:
            None.

        Raises:
            RuntimeError: If sensor connection is closed.

        Example:
            >>> sensor.reset()
            >>> # Now recalibrate from scratch
            >>> sensor.calibrate(40.0, 0.0)
        """
        ...

    def close(self) -> None:
        """Close sensor connection and release resources.

        Terminates communication with sensor hardware and releases
        any system resources (serial ports, threads). Should be called
        when done with sensor or during shutdown.

        Business context: Proper cleanup prevents resource leaks and
        allows reconnection. Essential for graceful shutdown and
        switching between sensors.

        Returns:
            None.

        Raises:
            None. Safe to call multiple times.

        Example:
            >>> try:
            ...     reading = sensor.read()
            ... finally:
            ...     sensor.close()
        """
        ...


class SensorDriver(Protocol):  # pragma: no cover
    """Protocol for sensor drivers.

    A SensorDriver handles discovery and connection to sensor devices.
    It can enumerate available sensors and open connections.

    Business context: Factory pattern for sensor creation. Enables
    automatic discovery of available sensors and uniform connection
    interface. Implementations include ArduinoSensorDriver and
    DigitalTwinSensorDriver.
    """

    def get_available_sensors(self) -> list[dict]:
        """Enumerate available sensor devices.

        Scans for compatible sensors (serial ports, simulated devices)
        and returns information about each. Used for device discovery
        and selection UI.

        Business context: Enables automatic sensor discovery without
        manual configuration. Users select from discovered devices
        rather than typing port names.

        Returns:
            List of sensor info dicts, each containing:
            - id: Integer index for selection
            - type: Sensor type (e.g., "arduino_ble33")
            - name: Human-readable name
            - port: Connection path (e.g., "/dev/ttyACM0")
            - description: Hardware description

        Raises:
            None. Returns empty list if no sensors found.

        Example:
            >>> sensors = driver.get_available_sensors()
            >>> for s in sensors:
            ...     print(f"{s['id']}: {s['name']} ({s['port']})")
        """
        ...

    def open(self, sensor_id: int | str = 0) -> SensorInstance:
        """Open connection to a sensor device.

        Creates and returns a connected SensorInstance for the specified
        sensor. Only one sensor can be open at a time per driver.

        Business context: Primary factory method for creating sensor
        connections. Accepts either index from get_available_sensors()
        or direct port path for advanced users.

        Args:
            sensor_id: Either integer index from get_available_sensors()
                or string port path (e.g., "/dev/ttyACM0"). Default 0
                opens first available sensor.

        Returns:
            SensorInstance connected and ready for reading.

        Raises:
            RuntimeError: If sensor already open, sensor not found,
                or connection fails.
            ValueError: If sensor_id is invalid index.

        Example:
            >>> driver = ArduinoSensorDriver()
            >>> sensor = driver.open(0)  # First sensor
            >>> # Or by port:
            >>> sensor = driver.open("/dev/ttyACM0")
        """
        ...

    def close(self) -> None:
        """Close the currently open sensor instance.

        Closes any sensor opened by this driver. Safe to call even
        if no sensor is open.

        Business context: Convenience method for cleanup. Equivalent
        to calling close() on the SensorInstance but accessible from
        the driver.

        Returns:
            None.

        Raises:
            None. Safe to call multiple times.

        Example:
            >>> driver.open(0)
            >>> # ... use sensor ...
            >>> driver.close()
        """
        ...
